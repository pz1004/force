"""Diagnose production P² quantile accuracy on skewed distributions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, List

for _thread_variable in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from force import P2Quantile
from force.diagnostics import quantile_rank_interval_error, reference_quantiles
from force.legacy import _legacy_p_square_kernel
from force.protocols import get_protocol
from force.reporting import command_line, environment_metadata, write_json


def _estimate(values: np.ndarray, probability: float, protocol: str) -> float:
    if protocol == "legacy":
        result = _legacy_p_square_kernel(
            np.asarray(values, dtype=np.float64),
            np.array([probability], dtype=np.float64),
        )
        return float(result[0])
    tracker = P2Quantile(probability)
    for value in values:
        tracker.update(float(value))
    return tracker.value


def run_diagnostic(
    *,
    distribution: str,
    sampler: Callable[[int], np.ndarray],
    probabilities: Iterable[float],
    sample_sizes: Iterable[int],
    repetitions: int,
    protocol: str,
) -> List[dict]:
    probabilities = tuple(float(probability) for probability in probabilities)
    if not probabilities or any(
        not np.isfinite(probability) or not 0 < probability < 1
        for probability in probabilities
    ):
        raise ValueError(
            "P² probabilities must be finite and strictly between 0 and 1."
        )
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    if protocol not in {"paper", "legacy"}:
        raise ValueError("protocol must be 'paper' or 'legacy'.")
    rows: List[dict] = []
    for n_samples in sample_sizes:
        if n_samples < 1:
            raise ValueError("Sample sizes must be positive.")
        errors = {probability: [] for probability in probabilities}
        rank_errors = {probability: [] for probability in probabilities}
        for _ in range(repetitions):
            values = np.asarray(sampler(n_samples), dtype=np.float64)
            if values.shape != (n_samples,) or not np.isfinite(values).all():
                raise ValueError(
                    "sampler must return the requested finite one-dimensional sample."
                )
            exact = reference_quantiles(values[:, None], probabilities)[:, 0]
            for probability, reference in zip(probabilities, exact):
                estimate = _estimate(values, probability, protocol)
                absolute_error = abs(estimate - float(reference))
                if not np.isfinite(estimate) or not np.isfinite(absolute_error):
                    raise FloatingPointError(
                        "Quantile estimate or absolute error exceeds the supported finite range."
                    )
                errors[probability].append(absolute_error)
                rank_errors[probability].append(
                    float(
                        quantile_rank_interval_error(
                            values[:, None], np.array([[estimate]]), [probability]
                        )[0, 0]
                    )
                )
        for probability in probabilities:
            rows.append(
                {
                    "distribution": distribution,
                    "protocol": protocol,
                    "status": "completed",
                    "n_samples": n_samples,
                    "probability": probability,
                    "repetitions": repetitions,
                    "mean_absolute_error": _nonnegative_mean(errors[probability]),
                    "mean_rank_error": float(np.mean(rank_errors[probability])),
                    "rank_error_definition": "distance_to_empirical_rank_interval",
                    "exact_quantile_method": "linear_decimal_reference",
                }
            )
    return rows


def _nonnegative_mean(values: List[float]) -> float:
    """Average finite nonnegative errors without overflowing their sum."""
    scale = max(values)
    return float(np.mean(np.asarray(values) / scale) * scale) if scale else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("paper", "legacy"), default="paper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Deprecated alias for --runs.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="./verification_results/p2_diagnostic",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Deprecated direct JSON path; prefer --output-dir.",
    )
    args = parser.parse_args()
    requested_runs = args.repetitions if args.repetitions is not None else args.runs
    if requested_runs < 1:
        parser.error("--runs must be at least 1")
    repetitions = 2 if args.smoke else requested_runs
    sample_sizes = (200, 1000) if args.smoke else (200, 500, 1000, 2000, 5000)
    probabilities = (0.01, 0.25, 0.50, 0.75, 0.99)
    rng = np.random.default_rng(args.seed)
    rows: List[dict] = []
    rows.extend(
        run_diagnostic(
            distribution="lognormal",
            sampler=lambda size: rng.lognormal(0.0, 1.0, size),
            probabilities=probabilities,
            sample_sizes=sample_sizes,
            repetitions=repetitions,
            protocol=args.protocol,
        )
    )
    rows.extend(
        run_diagnostic(
            distribution="negative_lognormal",
            sampler=lambda size: -rng.lognormal(0.0, 1.0, size),
            probabilities=probabilities,
            sample_sizes=sample_sizes,
            repetitions=repetitions,
            protocol=args.protocol,
        )
    )
    for row in rows:
        row["seed"] = args.seed
    output_dir = Path(args.output).parent if args.output else Path(args.output_dir)
    report_path = Path(args.output) if args.output else output_dir / "report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    write_json(
        report_path,
        {
            "schema_version": 2,
            "status": "completed",
            "protocol": get_protocol(args.protocol).as_dict(),
            "smoke": bool(args.smoke),
            "offline": bool(args.offline),
            "data_dir": args.data_dir,
            "seed": args.seed,
            "runs": repetitions,
            "sample_sizes": list(sample_sizes),
            "probabilities": list(probabilities),
            "environment": environment_metadata(),
            "command_line": command_line(),
            "results": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
