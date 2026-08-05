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
    rows: List[dict] = []
    for n_samples in sample_sizes:
        errors = {probability: [] for probability in probabilities}
        rank_errors = {probability: [] for probability in probabilities}
        for _ in range(repetitions):
            values = np.asarray(sampler(n_samples), dtype=np.float64)
            for probability in probabilities:
                exact = float(np.quantile(values, probability))
                estimate = _estimate(values, probability, protocol)
                errors[probability].append(abs(estimate - exact))
                empirical_rank = float(np.mean(values <= estimate))
                rank_errors[probability].append(
                    abs(empirical_rank - probability)
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
                    "mean_absolute_error": float(
                        np.mean(errors[probability])
                    ),
                    "mean_rank_error": float(
                        np.mean(rank_errors[probability])
                    ),
                }
            )
    return rows


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
    requested_runs = (
        args.repetitions if args.repetitions is not None else args.runs
    )
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
    output_dir = (
        Path(args.output).parent if args.output else Path(args.output_dir)
    )
    report_path = (
        Path(args.output) if args.output else output_dir / "report.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    write_json(
        report_path,
        {
            "schema_version": 1,
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
