"""Smoke/time the shared exact trimmed-Pearson implementations."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter_ns

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

from force import TrimmedPearsonExact
from force.protocols import get_protocol
from force.reporting import (
    command_line,
    environment_metadata,
    estimator_parameters,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("paper", "legacy"), default="paper")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir",
        default="./verification_results/trimmed_pearson",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Deprecated direct JSON path; prefer --output-dir.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    runs = 1 if args.smoke else args.runs
    n_samples, n_features = ((200, 8) if args.smoke else (2000, 50))
    X = np.random.default_rng(args.seed).normal(size=(n_samples, n_features))
    ter_max = None if args.protocol == "paper" else 3.0
    configurations = (
        ("TP-Exact", {"use_ter": False}),
        ("TP-TER", {"use_ter": True, "ter_max": ter_max}),
    )
    for _, parameters in configurations:
        TrimmedPearsonExact(**parameters).fit(X)
    rows = []
    for name, parameters in configurations:
        for run_id in range(1, runs + 1):
            estimator = TrimmedPearsonExact(**parameters)
            start = perf_counter_ns()
            correlation = estimator.fit(X)
            elapsed = (perf_counter_ns() - start) / 1_000_000.0
            rows.append(
                {
                    "protocol": args.protocol,
                    "status": "completed",
                    "algorithm": name,
                    "run_id": run_id,
                    "seed": args.seed,
                    "time_ms": elapsed,
                    "shape": list(correlation.shape),
                    "minimum": float(correlation.min()),
                    "maximum": float(correlation.max()),
                }
            )
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
            "runs": runs,
            "n_samples": n_samples,
            "n_features": n_features,
            "estimator_parameters": {
                name: estimator_parameters(TrimmedPearsonExact(**parameters))
                for name, parameters in configurations
            },
            "environment": environment_metadata(),
            "command_line": command_line(),
            "results": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
