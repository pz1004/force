"""Measure batch, replayable, and online FORCE in isolated processes.

Streaming sources generate bounded chunks; their oracle/data histories are never
materialized. RSS samples are post-warm-up observations, not exact workspace
accounting. Lifetime high-water RSS is retained separately for transparency.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import threading
import tracemalloc
from time import perf_counter_ns

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _lifetime_peak_kib() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak / 1024 if sys.platform == "darwin" else peak)


def _current_rss_kib() -> int | None:
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") // 1024
    except (OSError, ValueError, IndexError):
        return None


class _RssSampler:
    """Linux process RSS sampled without a new runtime package dependency."""

    def __init__(self, interval: float = 0.002):
        self.interval = interval
        self.baseline = _current_rss_kib()
        self.peak = self.baseline
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            value = _current_rss_kib()
            if value is not None:
                self.peak = value if self.peak is None else max(value, self.peak)
                self.samples += 1
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> None:
        self._stop.set()
        self._thread.join()
        value = _current_rss_kib()
        if value is not None:
            self.peak = value if self.peak is None else max(value, self.peak)


def generated_source(n_samples: int, n_features: int, seed: int, chunk_size: int):
    """Fresh, replayable deterministic source with O(chunk_size*p) storage."""
    import numpy as np

    rng = np.random.default_rng(seed)
    for start in range(0, n_samples, chunk_size):
        yield rng.standard_normal((min(chunk_size, n_samples - start), n_features))


def _run_case(
    n_samples: int,
    n_features: int,
    seed: int,
    protocol: str,
    runs: int,
    memory_limit_gib: float | None,
    mode: str = "batch",
    chunk_size: int = 128,
    window_size: int = 100,
) -> dict:
    if memory_limit_gib is not None:
        limit = int(memory_limit_gib * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    import numpy as np
    from force.protocols import build_estimator, get_protocol

    if protocol == "legacy" and mode != "batch":
        raise ValueError("Legacy FORCE has only the batch interface.")
    if mode == "online" and n_samples <= window_size:
        raise ValueError("Online case requires post-calibration observations.")

    def new_estimator():
        if mode != "online":
            return build_estimator(protocol, "FORCE")
        from force.streaming import StreamingForceEstimator

        # Online windows use their own pure-P² schedule. Batch quantile-mode
        # options are not constructor arguments for that estimator.
        parameters = {
            name: get_protocol(protocol).force_parameters[name]
            for name in ("lambda_scale", "use_ter", "ter_max", "epsilon")
        }
        return StreamingForceEstimator(window_size=window_size, **parameters)

    def evaluate(estimator, size, features, matrix=None):
        if mode == "batch":
            return estimator.fit(matrix)
        if mode == "replay":
            factory = lambda: generated_source(size, features, seed, chunk_size)
            return estimator.fit_replayable(factory, chunk_size=chunk_size)
        for chunk in generated_source(size, features, seed, chunk_size):
            estimator.update_batch(chunk)
        result = estimator.finalize()
        if result is None:
            raise RuntimeError("Online estimator did not produce an estimate.")
        return result

    warm_n = max(100, window_size + 10) if mode == "online" else 100
    warm_matrix = (
        np.random.default_rng(seed).normal(size=(warm_n, 2))
        if mode == "batch"
        else None
    )
    warm_estimator = new_estimator()
    evaluate(warm_estimator, warm_n, 2, warm_matrix)
    del warm_estimator, warm_matrix
    gc.collect()
    lifetime_warmup = _lifetime_peak_kib()
    sampler = _RssSampler()
    sampler.start()
    matrix = None
    estimator = None
    correlation = None
    times = []
    try:
        if mode == "batch":
            matrix = np.random.default_rng(seed).standard_normal(
                (n_samples, n_features)
            )
        for _ in range(runs):
            # Do not overlap a previous fitted estimator/output with the next fit.
            estimator, correlation = None, None
            gc.collect()
            estimator = new_estimator()
            start = perf_counter_ns()
            correlation = evaluate(estimator, n_samples, n_features, matrix)
            times.append((perf_counter_ns() - start) / 1e6)
        state_bytes = int(estimator.state_nbytes_)
        output_bytes = int(correlation.nbytes)
    finally:
        sampler.finish()
    # Trace one separate fit after JIT warm-up so tracing overhead does not
    # contaminate timings. The batch input is preallocated and excluded here.
    # Native runtimes can allocate outside tracemalloc, so this is evidence
    # complementary to RSS rather than a complete auxiliary-workspace bound.
    estimator, correlation = None, None
    gc.collect()
    tracemalloc.start()
    try:
        traced_estimator = new_estimator()
        traced_correlation = evaluate(traced_estimator, n_samples, n_features, matrix)
        traced_final, traced_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del traced_estimator, traced_correlation
    std = statistics.stdev(times) if runs > 1 else None
    delta = (
        max(0, sampler.peak - sampler.baseline)
        if sampler.peak is not None and sampler.baseline is not None
        else None
    )
    return {
        "status": "completed",
        "protocol": protocol,
        "mode": mode,
        "seed": seed,
        "n_samples": n_samples,
        "n_features": n_features,
        "runs": runs,
        "chunk_size": chunk_size if mode != "batch" else None,
        "window_size": window_size if mode == "online" else None,
        "calibration_rows": window_size if mode == "online" else 0,
        "times_ms": times,
        "mean_time_ms": statistics.mean(times),
        "std_time_ms": std,
        "ci95_time_ms": 1.96 * std / runs**0.5 if std is not None else None,
        "timing_scope": (
            "fit_only_preallocated_input"
            if mode == "batch"
            else "source_generation_and_fit"
        ),
        "logical_input_bytes": n_samples * n_features * 8,
        "input_bytes": int(matrix.nbytes) if matrix is not None else 0,
        "source_chunk_bytes": (
            min(chunk_size, n_samples) * n_features * 8 if mode != "batch" else 0
        ),
        "source_traversals": 2 if mode == "replay" else 1,
        "estimator_state_bytes": state_bytes,
        "output_bytes": output_bytes,
        "peak_rss_kib": _lifetime_peak_kib(),
        "post_warmup_lifetime_high_water_kib": lifetime_warmup,
        "post_warmup_rss_baseline_kib": sampler.baseline,
        "post_warmup_sampled_peak_rss_kib": sampler.peak,
        "post_warmup_sampled_peak_delta_kib": delta,
        "rss_sample_interval_ms": sampler.interval * 1000,
        "rss_sample_count": sampler.samples,
        "post_warmup_traced_peak_bytes": traced_peak,
        "post_warmup_traced_final_bytes": traced_final,
        "post_warmup_traced_peak_above_final_bytes": max(0, traced_peak - traced_final),
        "traced_memory_scope": (
            "separate_untimed_fit_preallocated_batch_input_excluded"
            if mode == "batch"
            else "separate_untimed_source_generation_and_fit"
        ),
        "memory_interpretation": "Retained numeric state excludes input/output and object overhead. Tracemalloc measures a separate untimed post-warm-up fit, including validation; it can omit native/Numba allocations. Peak minus final traced memory is observed transient allocation overhead, not exact workspace. Sampled RSS includes input, validation, runtime and sampler overhead; short transients and code holding the GIL may evade sampling. Lifetime peak includes JIT warm-up and tracing. Source may transiently overlap two chunks.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--protocol", choices=("paper", "legacy"), default="paper")
    parser.add_argument(
        "--mode",
        choices=("batch", "replay", "online"),
        default=None,
        help="Single mode for either parent or isolated child run.",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Parent run modes; default: batch,replay,online for paper, batch for legacy.",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--memory-limit-gib", type=float, default=16.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="./verification_results/scalability")
    parser.add_argument("--output", default=None, help="Deprecated direct JSON path.")
    args = parser.parse_args()
    if (
        args.runs < 1
        or args.n_samples < 5
        or args.n_features < 2
        or args.chunk_size < 1
        or args.window_size < 6
    ):
        parser.error(
            "Require positive runs/chunks, >=5 rows, >=6 window size and >=2 features."
        )
    if args.timeout <= 0 or args.memory_limit_gib <= 0:
        parser.error("Timeout and memory limit must be positive.")
    if args.mode and args.modes:
        parser.error("Choose either --mode or --modes.")
    if args.case:
        args.mode = args.mode or "batch"
        try:
            result = _run_case(
                args.n_samples,
                args.n_features,
                args.seed,
                args.protocol,
                args.runs,
                args.memory_limit_gib,
                args.mode,
                args.chunk_size,
                args.window_size,
            )
        except Exception as exc:
            result = {
                "status": "oom" if isinstance(exc, MemoryError) else "failed",
                "protocol": args.protocol,
                "mode": args.mode,
                "seed": args.seed,
                "n_samples": args.n_samples,
                "n_features": args.n_features,
                "message": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(result, allow_nan=False))
        return 0

    modes = (
        args.modes
        or args.mode
        or ("batch" if args.protocol == "legacy" else "batch,replay,online")
    ).split(",")
    if any(mode not in ("batch", "replay", "online") for mode in modes):
        parser.error("--modes must be a subset of batch,replay,online.")
    if args.protocol == "legacy" and modes != ["batch"]:
        parser.error("Legacy protocol supports batch only.")
    cases = (
        [(200, 5), (800, 5), (200, 10)]
        if args.smoke
        else [(1000, 10), (4000, 10), (1000, 20), (1000, 40)]
    )
    runs = 1 if args.smoke else args.runs
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
        **{key: "1" for key in THREAD_VARIABLES},
    }
    rows = []
    for mode in modes:
        for n, p in cases:
            metadata = {
                "protocol": args.protocol,
                "mode": mode,
                "seed": args.seed,
                "n_samples": n,
                "n_features": p,
            }
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case",
                "--protocol",
                args.protocol,
                "--mode",
                mode,
                "--runs",
                str(runs),
                "--n-samples",
                str(n),
                "--n-features",
                str(p),
                "--seed",
                str(args.seed),
                "--chunk-size",
                str(args.chunk_size),
                "--window-size",
                str(args.window_size),
                "--memory-limit-gib",
                str(args.memory_limit_gib),
            ]
            try:
                child = subprocess.run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                if child.returncode == 0:
                    try:
                        rows.append(json.loads(child.stdout.strip().splitlines()[-1]))
                    except (ValueError, IndexError):
                        rows.append(
                            {
                                **metadata,
                                "status": "failed",
                                "message": "Child returned no valid JSON: "
                                + child.stderr[-1000:],
                            }
                        )
                else:
                    # A signal alone does not establish an OOM; retain evidence.
                    rows.append(
                        {
                            **metadata,
                            "status": (
                                "oom" if "MemoryError" in child.stderr else "failed"
                            ),
                            "returncode": child.returncode,
                            "message": child.stderr[-1000:],
                        }
                    )
            except subprocess.TimeoutExpired:
                rows.append({**metadata, "status": "timeout"})
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from force.protocols import get_protocol
    from force.reporting import command_line, environment_metadata, write_json

    output = Path(args.output) if args.output else Path(args.output_dir) / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with (output.parent / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )
    statuses = {row["status"] for row in rows}
    status = (
        "completed"
        if statuses == {"completed"}
        else next(s for s in ("failed", "oom", "timeout") if s in statuses)
    )
    write_json(
        output,
        {
            "schema_version": 1,
            "status": status,
            "protocol": get_protocol(args.protocol).as_dict(),
            "smoke": args.smoke,
            "offline": args.offline,
            "data_dir": args.data_dir,
            "seed": args.seed,
            "runs": runs,
            "modes": modes,
            "threads": 1,
            "memory_limit_gib": args.memory_limit_gib,
            "rss_units": "KiB",
            "environment": environment_metadata(),
            "command_line": command_line(),
            "oracle_process_present": False,
            "rss_limitations": "Post-warm-up sampled RSS is Linux-only; short transients and code holding the GIL may evade sampling. Lifetime RSS includes JIT and the separate traced fit. Allocator retention can hide or exaggerate incremental workspace. Tracemalloc can omit native/Numba allocations; peak above final traced allocation is not exact auxiliary workspace. Source/state/output bytes, sampled RSS and traced allocations are complementary evidence, not a complete peak proof.",
            "cases": rows,
        },
    )
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
