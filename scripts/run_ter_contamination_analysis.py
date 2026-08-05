"""Reproduce Appendix B using the production FORCE estimator."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

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

from force import ForceEstimator, LegacyForceEstimator
from force.protocols import get_protocol
from force.reporting import (
    command_line,
    environment_metadata,
    estimator_parameters,
    write_json,
)


def _estimator(protocol: str, *, use_ter: bool):
    if protocol == "legacy":
        return LegacyForceEstimator(use_ter=use_ter)
    return ForceEstimator(
        exact_cutover=5,
        use_ter=use_ter,
        ter_max=None,
    )


def _mse(
    X: np.ndarray, rho: float, *, use_ter: bool, protocol: str
) -> float:
    estimate = _estimator(protocol, use_ter=use_ter).fit(X)
    return float((estimate[0, 1] - rho) ** 2)


def run_analysis(
    *, n_samples: int, repetitions: int, seed: int, protocol: str
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    rho = 0.6
    covariance = np.array([[1.0, rho], [rho, 1.0]])
    rows: List[Dict[str, object]] = []

    for contamination in (0.0, 0.05, 0.10, 0.15):
        with_ter: List[float] = []
        fixed: List[float] = []
        for _ in range(repetitions):
            n_outliers = int(contamination * n_samples)
            clean = rng.multivariate_normal(
                np.zeros(2), covariance, n_samples - n_outliers
            )
            outliers = rng.multivariate_normal(
                np.array([8.0, 0.0]),
                np.array([[0.1, 0.0], [0.0, 1.0]]),
                n_outliers,
            )
            X = np.vstack((clean, outliers))
            rng.shuffle(X)
            with_ter.append(
                _mse(X, rho, use_ter=True, protocol=protocol)
            )
            fixed.append(
                _mse(X, rho, use_ter=False, protocol=protocol)
            )
        rows.append(
            {
                "scenario": "asymmetric_contamination",
                "protocol": protocol,
                "status": "completed",
                "seed": seed,
                "parameter": contamination,
                "mse_ter": float(np.mean(with_ter)),
                "mse_fixed": float(np.mean(fixed)),
                "repetitions": repetitions,
            }
        )

    for degrees in (100, 10, 5, 3):
        with_ter = []
        fixed = []
        for _ in range(repetitions):
            normal = rng.multivariate_normal(
                np.zeros(2), covariance, n_samples
            )
            scale = rng.chisquare(degrees, n_samples) / degrees
            X = normal / np.sqrt(scale)[:, None]
            with_ter.append(
                _mse(X, rho, use_ter=True, protocol=protocol)
            )
            fixed.append(
                _mse(X, rho, use_ter=False, protocol=protocol)
            )
        rows.append(
            {
                "scenario": "coherent_student_t",
                "protocol": protocol,
                "status": "completed",
                "seed": seed,
                "parameter": f"df={degrees}",
                "mse_ter": float(np.mean(with_ter)),
                "mse_fixed": float(np.mean(fixed)),
                "repetitions": repetitions,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("paper", "legacy"), default="paper")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Deprecated alias for --runs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir", default="./verification_results/ter_analysis"
    )
    args = parser.parse_args()
    requested_runs = (
        args.repetitions if args.repetitions is not None else args.runs
    )
    if requested_runs < 1:
        parser.error("--runs must be at least 1")
    if args.n_samples < 5:
        parser.error("--n-samples must be at least 5")
    n_samples = 200 if args.smoke else args.n_samples
    repetitions = 2 if args.smoke else requested_runs
    _estimator(args.protocol, use_ter=True).fit(
        np.random.default_rng(args.seed).normal(size=(100, 2))
    )
    rows = run_analysis(
        n_samples=n_samples,
        repetitions=repetitions,
        seed=args.seed,
        protocol=args.protocol,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    write_json(
        output_dir / "report.json",
        {
            "schema_version": 1,
            "status": "completed",
            "protocol": get_protocol(args.protocol).as_dict(),
            "implementation": type(
                _estimator(args.protocol, use_ter=True)
            ).__name__,
            "smoke": bool(args.smoke),
            "offline": bool(args.offline),
            "data_dir": args.data_dir,
            "seed": args.seed,
            "n_samples": n_samples,
            "runs": repetitions,
            "estimator_parameters": {
                "ter": estimator_parameters(
                    _estimator(args.protocol, use_ter=True)
                ),
                "fixed": estimator_parameters(
                    _estimator(args.protocol, use_ter=False)
                ),
            },
            "environment": environment_metadata(),
            "command_line": command_line(),
            "results": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
