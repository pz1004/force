"""S&P volatility-reference sensitivity under explicit benchmark protocols."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence

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

from force.data import ExternalDataUnavailable
from force.protocols import (
    DatasetNotReproducible,
    build_estimator,
    build_estimators,
    get_protocol,
    prepare_dataset,
)
from force.reporting import (
    command_line,
    environment_metadata,
    estimator_parameters,
    write_json,
)


def reference_corr_low_vol(
    returns: np.ndarray, cutoff_fraction: float
) -> np.ndarray:
    volatility = np.mean(np.abs(returns), axis=1)
    threshold = np.quantile(volatility, 1.0 - cutoff_fraction)
    return np.corrcoef(returns[volatility <= threshold], rowvar=False)


def rmse_offdiag(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.triu(np.ones_like(left, dtype=bool), k=1)
    return float(np.sqrt(np.mean((left[mask] - right[mask]) ** 2)))


def _tickers(path: str | None) -> Sequence[str] | None:
    if path is None:
        return None
    ticker_path = Path(path)
    text = ticker_path.read_text(encoding="utf-8")
    if ticker_path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("tickers")
        if not isinstance(payload, list):
            raise ValueError("Ticker JSON must be a list or contain 'tickers'.")
        values = payload
    else:
        values = text.replace(",", "\n").splitlines()
    return tuple(
        str(value).strip().upper()
        for value in values
        if str(value).strip()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("paper", "legacy"), default="paper")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--sp500-tickers-file", default=None)
    parser.add_argument(
        "--output-dir", default="./verification_results/sp500_sensitivity"
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    runs = 1 if args.smoke else args.runs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "protocol": get_protocol(args.protocol).as_dict(),
        "status": "completed",
        "message": "",
        "smoke": bool(args.smoke),
        "offline": bool(args.offline),
        "data_dir": args.data_dir,
        "seed": args.seed,
        "runs": runs,
        "environment": environment_metadata(),
        "command_line": command_line(),
        "estimator_parameters": {
            name: estimator_parameters(estimator)
            for name, estimator in build_estimators(args.protocol).items()
        },
        "results": [],
    }
    try:
        prepared = prepare_dataset(
            "sp500",
            protocol=args.protocol,
            seed=args.seed,
            smoke=args.smoke,
            offline=args.offline,
            data_dir=Path(args.data_dir) if args.data_dir else None,
            sp500_tickers=_tickers(args.sp500_tickers_file),
        )
        rows: List[dict] = []
        for cutoff in (0.05, 0.10, 0.15):
            reference = reference_corr_low_vol(prepared.X, cutoff)
            for name in build_estimators(args.protocol):
                for run_id in range(1, runs + 1):
                    estimator = build_estimator(args.protocol, name)
                    estimate = estimator.fit(prepared.X)
                    rows.append(
                        {
                            "protocol": args.protocol,
                            "status": "completed",
                            "cutoff": cutoff,
                            "algorithm": name,
                            "run_id": run_id,
                            "seed": args.seed,
                            "rmse": rmse_offdiag(estimate, reference),
                        }
                    )
        report["results"] = rows
        report["provenance"] = prepared.provenance
        pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
    except DatasetNotReproducible as exc:
        report["status"] = "not_reproducible"
        report["message"] = str(exc)
    except ExternalDataUnavailable as exc:
        report["status"] = "skipped_external"
        report["message"] = str(exc)
    except Exception as exc:
        report["status"] = "failed"
        report["message"] = str(exc)
    results_path = output_dir / "results.csv"
    if not results_path.exists():
        pd.DataFrame(
            columns=(
                "protocol",
                "status",
                "cutoff",
                "algorithm",
                "run_id",
                "seed",
                "rmse",
            )
        ).to_csv(results_path, index=False)
    write_json(output_dir / "report.json", report)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
