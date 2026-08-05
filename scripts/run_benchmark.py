"""Run FORCE benchmarks under an equation-faithful or legacy protocol."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, Iterable, List, Sequence

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
    DATASET_NAMES,
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


LOGGER = logging.getLogger("force.benchmark")


def off_diagonal_rmse(
    estimate: np.ndarray, reference: np.ndarray
) -> float:
    if estimate.shape != reference.shape or estimate.ndim != 2:
        raise ValueError("Estimate and reference must be equal-sized matrices.")
    mask = np.triu(np.ones_like(estimate, dtype=bool), k=1)
    return float(np.sqrt(np.mean((estimate[mask] - reference[mask]) ** 2)))


def _parse_datasets(value: str) -> List[str]:
    datasets = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(datasets) - set(DATASET_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown datasets: {', '.join(unknown)}"
        )
    if not datasets:
        raise argparse.ArgumentTypeError("At least one dataset is required.")
    return datasets


def _read_tickers(path: Path) -> Sequence[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("tickers")
        if not isinstance(payload, list):
            raise ValueError("Ticker JSON must be a list or contain 'tickers'.")
        values = payload
    else:
        values = text.replace(",", "\n").splitlines()
    tickers = tuple(str(value).strip().upper() for value in values if str(value).strip())
    return tickers


def run_dataset(
    *,
    dataset_name: str,
    X: np.ndarray,
    reference: np.ndarray,
    protocol: str,
    runs: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    algorithms = tuple(build_estimators(protocol))
    for algorithm in algorithms:
        warmup_estimator = build_estimator(protocol, algorithm)
        parameters = estimator_parameters(warmup_estimator)
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                warmup_estimator.fit(X)  # Throwaway warm-up, never timed.
        except Exception as exc:
            rows.append(
                {
                    "protocol": protocol,
                    "dataset": dataset_name,
                    "algorithm": algorithm,
                    "run_id": None,
                    "seed": seed,
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "time_ms": None,
                    "rmse": None,
                    "status": "failed",
                    "message": f"warm-up failed: {exc}",
                    "warning_count": 0,
                    "estimator_parameters": json.dumps(
                        parameters, sort_keys=True
                    ),
                }
            )
            continue

        for run_id in range(1, runs + 1):
            estimator = build_estimator(protocol, algorithm)
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    start = perf_counter_ns()
                    estimate = estimator.fit(X)
                    elapsed_ms = (perf_counter_ns() - start) / 1_000_000.0
                rows.append(
                    {
                        "protocol": protocol,
                        "dataset": dataset_name,
                        "algorithm": algorithm,
                        "run_id": run_id,
                        "seed": seed,
                        "n_samples": int(X.shape[0]),
                        "n_features": int(X.shape[1]),
                        "time_ms": float(elapsed_ms),
                        "rmse": off_diagonal_rmse(estimate, reference),
                        "status": "completed",
                        "message": (
                            str(captured[0].message) if captured else ""
                        ),
                        "warning_count": len(captured),
                        "estimator_parameters": json.dumps(
                            estimator_parameters(estimator), sort_keys=True
                        ),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "protocol": protocol,
                        "dataset": dataset_name,
                        "algorithm": algorithm,
                        "run_id": run_id,
                        "seed": seed,
                        "n_samples": int(X.shape[0]),
                        "n_features": int(X.shape[1]),
                        "time_ms": None,
                        "rmse": None,
                        "status": "failed",
                        "message": str(exc),
                        "warning_count": 0,
                        "estimator_parameters": json.dumps(
                            estimator_parameters(estimator), sort_keys=True
                        ),
                    }
                )
    return rows


def _summary_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    completed = frame[frame["status"] == "completed"].copy()
    if completed.empty:
        return []

    records: List[Dict[str, Any]] = []
    groups = completed.groupby(
        ["protocol", "dataset", "algorithm"], observed=True, sort=True
    )
    for (protocol, dataset, algorithm), group in groups:
        runs = int(len(group))
        time_values = group["time_ms"].to_numpy(dtype=float)
        rmse_values = group["rmse"].to_numpy(dtype=float)
        std_time = float(np.std(time_values, ddof=1)) if runs > 1 else None
        std_rmse = float(np.std(rmse_values, ddof=1)) if runs > 1 else None
        records.append(
            {
                "protocol": str(protocol),
                "dataset": str(dataset),
                "algorithm": str(algorithm),
                "runs": runs,
                "mean_time_ms": float(np.mean(time_values)),
                "std_time_ms": std_time,
                "ci95_time_ms": (
                    1.96 * std_time / np.sqrt(runs)
                    if std_time is not None
                    else None
                ),
                "mean_rmse": float(np.mean(rmse_values)),
                "std_rmse": std_rmse,
                "ci95_rmse": (
                    1.96 * std_rmse / np.sqrt(runs)
                    if std_rmse is not None
                    else None
                ),
            }
        )
    return records


def _summary_markdown(summary_records: List[Dict[str, Any]]) -> str:
    if not summary_records:
        return "# FORCE benchmark summary\n\nNo completed measurements.\n"
    summary = pd.DataFrame.from_records(summary_records)
    return "# FORCE benchmark summary\n\n" + summary.to_markdown(index=False) + "\n"


def execute(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else None
    runs = 1 if args.smoke else args.runs
    tickers = (
        _read_tickers(Path(args.sp500_tickers_file))
        if args.sp500_tickers_file
        else None
    )
    protocol = get_protocol(args.protocol)
    result_rows: List[Dict[str, Any]] = []
    dataset_records: List[Dict[str, Any]] = []

    for dataset_name in args.datasets:
        LOGGER.info("Preparing %s under %s protocol", dataset_name, args.protocol)
        try:
            prepared = prepare_dataset(
                dataset_name,
                protocol=args.protocol,
                seed=args.seed,
                smoke=args.smoke,
                offline=args.offline,
                data_dir=data_dir,
                sp500_tickers=tickers,
                gemma_dataset=args.gemma_dataset,
            )
            dataset_records.append(
                {
                    "dataset": dataset_name,
                    "status": "completed",
                    "shape": list(prepared.X.shape),
                    "provenance": prepared.provenance,
                    "message": "",
                }
            )
            result_rows.extend(
                run_dataset(
                    dataset_name=dataset_name,
                    X=prepared.X,
                    reference=prepared.reference,
                    protocol=args.protocol,
                    runs=runs,
                    seed=args.seed,
                )
            )
        except DatasetNotReproducible as exc:
            dataset_records.append(
                {
                    "dataset": dataset_name,
                    "status": "not_reproducible",
                    "shape": None,
                    "provenance": {},
                    "message": str(exc),
                }
            )
        except ExternalDataUnavailable as exc:
            dataset_records.append(
                {
                    "dataset": dataset_name,
                    "status": "skipped_external",
                    "shape": None,
                    "provenance": {},
                    "message": str(exc),
                }
            )
        except Exception as exc:
            dataset_records.append(
                {
                    "dataset": dataset_name,
                    "status": "failed",
                    "shape": None,
                    "provenance": {},
                    "message": str(exc),
                }
            )

    columns = [
        "protocol",
        "dataset",
        "algorithm",
        "run_id",
        "seed",
        "n_samples",
        "n_features",
        "time_ms",
        "rmse",
        "status",
        "message",
        "warning_count",
        "estimator_parameters",
    ]
    frame = pd.DataFrame(result_rows, columns=columns)
    summaries = _summary_records(frame)
    frame.to_csv(output_dir / "results_raw.csv", index=False)
    (output_dir / "summary.md").write_text(
        _summary_markdown(summaries), encoding="utf-8"
    )
    dataset_statuses = {record["status"] for record in dataset_records}
    result_statuses = {row["status"] for row in result_rows}
    if "failed" in dataset_statuses or "failed" in result_statuses:
        aggregate_status = "failed"
    elif "completed" in dataset_statuses:
        aggregate_status = "completed"
    elif "not_reproducible" in dataset_statuses:
        aggregate_status = "not_reproducible"
    elif "skipped_external" in dataset_statuses:
        aggregate_status = "skipped_external"
    else:
        aggregate_status = "completed"
    report = {
        "schema_version": 1,
        "status": aggregate_status,
        "protocol": protocol.as_dict(),
        "smoke": bool(args.smoke),
        "offline": bool(args.offline),
        "seed": int(args.seed),
        "requested_runs": int(runs),
        "environment": environment_metadata(),
        "command_line": command_line(),
        "data_dir": str(data_dir) if data_dir is not None else None,
        "estimator_parameters": {
            name: estimator_parameters(estimator)
            for name, estimator in build_estimators(args.protocol).items()
        },
        "datasets": dataset_records,
        "summaries": summaries,
        "results": result_rows,
    }
    write_json(output_dir / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("paper", "legacy"), default="paper")
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=list(DATASET_NAMES),
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--sp500-tickers-file", default=None)
    parser.add_argument("--gemma-dataset", default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    report = execute(args)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
