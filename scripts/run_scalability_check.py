"""Measure FORCE timing, peak RSS, and retained/output state in subprocesses."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _set_memory_limit(limit_gib: float | None) -> None:
    if limit_gib is None:
        return
    limit_bytes = int(limit_gib * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def _run_case(
    n_samples: int,
    n_features: int,
    seed: int,
    protocol: str,
    runs: int,
    memory_limit_gib: float | None,
) -> int:
    _set_memory_limit(memory_limit_gib)
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    import numpy as np

    from force.protocols import build_estimator

    rng = np.random.default_rng(seed)
    build_estimator(protocol, "FORCE").fit(
        rng.normal(size=(100, 2))
    )  # throwaway JIT warm-up
    X = rng.normal(size=(n_samples, n_features))
    timings = []
    estimator = None
    correlation = None
    for _ in range(runs):
        estimator = build_estimator(protocol, "FORCE")
        start = perf_counter_ns()
        correlation = estimator.fit(X)
        timings.append((perf_counter_ns() - start) / 1_000_000.0)
    assert estimator is not None
    assert correlation is not None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # macOS reports bytes, Linux reports KiB.
        peak = peak / 1024
    std_ms = statistics.stdev(timings) if runs > 1 else None
    payload = {
        "status": "completed",
        "protocol": protocol,
        "seed": seed,
        "n_samples": n_samples,
        "n_features": n_features,
        "runs": runs,
        "times_ms": timings,
        "mean_time_ms": statistics.mean(timings),
        "std_time_ms": std_ms,
        "ci95_time_ms": (
            1.96 * std_ms / runs**0.5 if std_ms is not None else None
        ),
        "peak_rss_kib": int(peak),
        "input_bytes": int(X.nbytes),
        "estimator_state_bytes": estimator.state_nbytes_,
        "output_bytes": int(correlation.nbytes),
    }
    print(json.dumps(payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--protocol", choices=("paper", "legacy"), default="paper")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--memory-limit-gib", type=float, default=16.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="./verification_results/scalability",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Deprecated direct JSON path; prefer --output-dir.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.n_samples < 5 or args.n_features < 2:
        parser.error("cases require at least 5 samples and 2 features")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.memory_limit_gib is not None and args.memory_limit_gib <= 0:
        parser.error("--memory-limit-gib must be positive")
    if args.case:
        try:
            return _run_case(
                args.n_samples,
                args.n_features,
                args.seed,
                args.protocol,
                args.runs,
                args.memory_limit_gib,
            )
        except (MemoryError, OSError) as exc:
            print(
                json.dumps(
                    {
                        "status": "oom",
                        "protocol": args.protocol,
                        "seed": args.seed,
                        "n_samples": args.n_samples,
                        "n_features": args.n_features,
                        "message": str(exc),
                    }
                )
            )
            return 0

    cases = (
        [(200, 5), (800, 5), (200, 10)]
        if args.smoke
        else [(1000, 10), (4000, 10), (1000, 20), (1000, 40)]
    )
    runs = 1 if args.smoke else args.runs
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
        "NUMBA_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    rows = []
    for n_samples, n_features in cases:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case",
            "--protocol",
            args.protocol,
            "--runs",
            str(runs),
            "--n-samples",
            str(n_samples),
            "--n-features",
            str(n_features),
            "--seed",
            str(args.seed),
            "--memory-limit-gib",
            str(args.memory_limit_gib),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            if result.returncode == 0:
                rows.append(json.loads(result.stdout.strip().splitlines()[-1]))
            else:
                status = (
                    "oom"
                    if result.returncode < 0
                    or "MemoryError" in result.stderr
                    else "failed"
                )
                rows.append(
                    {
                        "status": status,
                        "protocol": args.protocol,
                        "seed": args.seed,
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "message": result.stderr[-1000:],
                    }
                )
        except subprocess.TimeoutExpired:
            rows.append(
                {
                    "status": "timeout",
                    "protocol": args.protocol,
                    "seed": args.seed,
                    "n_samples": n_samples,
                    "n_features": n_features,
                }
            )
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
    from force.protocols import get_protocol
    from force.reporting import (
        command_line,
        environment_metadata,
        write_json,
    )

    output = (
        Path(args.output)
        if args.output
        else Path(args.output_dir) / "report.json"
    )
    csv_path = output.parent / "results.csv"
    fieldnames = (
        "status",
        "protocol",
        "seed",
        "n_samples",
        "n_features",
        "runs",
        "mean_time_ms",
        "std_time_ms",
        "ci95_time_ms",
        "peak_rss_kib",
        "input_bytes",
        "estimator_state_bytes",
        "output_bytes",
        "times_ms",
        "message",
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key])
                        if key == "times_ms" and key in row
                        else row.get(key)
                    )
                    for key in fieldnames
                }
            )
    case_statuses = {row["status"] for row in rows}
    if case_statuses == {"completed"}:
        aggregate_status = "completed"
    elif "failed" in case_statuses:
        aggregate_status = "failed"
    elif "oom" in case_statuses:
        aggregate_status = "oom"
    else:
        aggregate_status = "timeout"
    write_json(
        output,
        {
            "schema_version": 1,
            "status": aggregate_status,
            "protocol": get_protocol(args.protocol).as_dict(),
            "smoke": bool(args.smoke),
            "offline": bool(args.offline),
            "data_dir": args.data_dir,
            "seed": args.seed,
            "runs": runs,
            "threads": 1,
            "memory_limit_gib": args.memory_limit_gib,
            "rss_units": "KiB",
            "environment": environment_metadata(),
            "command_line": command_line(),
            "cases": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
