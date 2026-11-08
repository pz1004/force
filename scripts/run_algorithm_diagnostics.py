"""Audit production FORCE against matched, independent offline references."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

for _variable in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_variable] = "1"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from force.diagnostics import POLICIES, diagnose_batch, diagnose_online, stress_sample
from force.reporting import command_line, environment_metadata, write_json


def run(args: argparse.Namespace) -> dict:
    n = 200 if args.smoke else args.n_samples
    p = min(args.n_features, 3) if args.smoke else args.n_features
    window = 50 if args.smoke else args.window_size
    fractions = [0.02] if args.smoke else [0.0, 0.02, 0.10, 0.26]
    magnitudes = [10.0, 1e6]
    scenarios = (
        ["one_sided_replacement", "cellwise"]
        if args.smoke
        else [
            "one_sided_replacement",
            "symmetric_heavy_tails",
            "clustered_rows",
            "cellwise",
        ]
    )
    orders = (
        ["random", "outliers_first"]
        if args.smoke
        else [
            "random",
            "sorted",
            "reversed",
            "outliers_first",
            "outliers_last",
        ]
    )
    rows = []
    for run_id in range(1 if args.smoke else args.runs):
        seed = args.seed + run_id
        for scenario in scenarios:
            for fraction in fractions:
                for magnitude in magnitudes:
                    for order in orders:
                        X, truth, contaminated = stress_sample(
                            n_samples=n,
                            n_features=p,
                            seed=seed,
                            fraction=fraction,
                            magnitude=magnitude,
                            scenario=scenario,
                            order=order,
                        )
                        metadata = {
                            "protocol": "paper",
                            "run_id": run_id,
                            "seed": seed,
                            "scenario": scenario,
                            "fraction_parameter": fraction,
                            "fraction_unit": (
                                "cells" if scenario == "cellwise" else "rows"
                            ),
                            "actual_contaminated_row_fraction": float(
                                contaminated.mean()
                            ),
                            "contamination_magnitude": magnitude,
                            "observation_order": order,
                        }
                        for policy in POLICIES.values():
                            for mode in args.modes.split(","):
                                try:
                                    keywords = dict(
                                        policy=policy,
                                        clean_reference=truth,
                                        contaminated_rows=contaminated,
                                    )
                                    result = (
                                        diagnose_batch(X, **keywords)
                                        if mode == "batch"
                                        else diagnose_online(
                                            X,
                                            window_size=window,
                                            chunk_size=args.chunk_size,
                                            **keywords,
                                        )
                                    )
                                    rows.append({**metadata, "mode": mode, **result})
                                except Exception as exc:
                                    rows.append(
                                        {
                                            **metadata,
                                            "mode": mode,
                                            "bound_policy": policy.name,
                                            "status": "failed",
                                            "message": f"{type(exc).__name__}: {exc}",
                                        }
                                    )
    return {
        "schema_version": 1,
        "status": (
            "failed" if any(row["status"] == "failed" for row in rows) else "completed"
        ),
        "protocol": "paper",
        "smoke": args.smoke,
        "seed": args.seed,
        "environment": environment_metadata(),
        "command_line": command_line(),
        "reference_storage": "Offline reference stores full samples and masks; excluded from production scalability measurements.",
        "interpretation": "Completed means executed. Approximation and clean-target errors are observations, not robustness or breakdown guarantees. The 2% one-sided cases intentionally probe uncapped-TER failure.",
        "replicates": "Independent data seeds; methods/policies/orders/magnitudes use paired samples. No timing variability is inferred here.",
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("paper",), default="paper")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=37)
    parser.add_argument("--modes", default="batch,online")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--output-dir", default="./verification_results/algorithm_diagnostics"
    )
    args = parser.parse_args()
    if (
        args.runs < 1
        or args.n_samples < 6
        or args.n_features < 2
        or args.window_size < 6
        or args.chunk_size < 1
    ):
        parser.error(
            "Require positive runs/chunks, >=6 samples, >=2 features and >=6 window size."
        )
    modes = args.modes.split(",")
    if not modes or any(mode not in ("batch", "online") for mode in modes):
        parser.error("--modes must be a comma-separated subset of batch,online")
    if "online" in modes and not args.smoke and args.n_samples <= args.window_size:
        parser.error("Online diagnostics require post-calibration observations.")
    report = run(args)
    output = Path(args.output_dir)
    write_json(output / "report.json", report)
    # CSV keeps scalar diagnostics; full pair matrices and window traces are in JSON.
    keys = sorted(
        {
            key
            for row in report["results"]
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
    )
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in report["results"]:
            writer.writerow({key: row.get(key) for key in keys})
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(report["results"]),
                "report": str(output / "report.json"),
            }
        )
    )
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
