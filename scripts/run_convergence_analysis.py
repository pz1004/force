"""Compare production P² FORCE with the shared exact-quantile baseline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter_ns
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

from force.data import generate_synthetic_data_fixed_corr
from force.diagnostics import diagnose_batch
from force._evaluation import correlation_score, estimator_pair_validity
from force.protocols import build_estimator, get_protocol
from force.reporting import (
    command_line,
    environment_metadata,
    estimator_parameters,
    write_json,
)


def run_convergence(
    *,
    sample_sizes: List[int],
    runs: int,
    n_features: int,
    contamination: float,
    seed: int,
    protocol: str,
) -> List[dict]:
    config = get_protocol(protocol)
    distribution = str(config.synthetic_parameters["contamination_distribution"])
    outlier_scale = float(config.synthetic_parameters["outlier_scale"])
    rows: List[dict] = []
    warmup = np.random.default_rng(seed).normal(size=(100, 2))
    build_estimator(protocol, "FORCE").fit(warmup)
    build_estimator(protocol, "TP-TER").fit(warmup)
    for n_samples in sample_sizes:
        for run_id in range(runs):
            run_seed = seed + run_id * 1000 + n_samples
            X, truth = generate_synthetic_data_fixed_corr(
                n_samples=n_samples,
                n_features=n_features,
                contamination=contamination,
                seed=run_seed,
                contamination_distribution=distribution,
                outlier_scale=outlier_scale,
            )
            force = build_estimator(protocol, "FORCE")
            exact = build_estimator(protocol, "TP-TER")

            start = perf_counter_ns()
            force_correlation = force.fit(X)
            force_ms = (perf_counter_ns() - start) / 1_000_000.0
            start = perf_counter_ns()
            exact_correlation = exact.fit(X)
            exact_ms = (perf_counter_ns() - start) / 1_000_000.0
            force_valid, force_validity_basis = estimator_pair_validity(
                force, force_correlation
            )
            exact_valid, _ = estimator_pair_validity(exact, exact_correlation)
            approximation_score = correlation_score(
                force_correlation,
                exact_correlation,
                estimate_valid=force_valid,
                reference_valid=exact_valid,
            )
            force_score = correlation_score(
                force_correlation,
                truth,
                estimate_valid=force_valid,
                require_complete=True,
            )
            exact_score = correlation_score(
                exact_correlation,
                truth,
                estimate_valid=exact_valid,
                require_complete=True,
            )
            # Run the independent oracle outside the timed region. Legacy
            # FORCE and TP-TER differ in more than their quantile backend.
            diagnostics = (
                diagnose_batch(X, clean_reference=truth)
                if protocol == "paper"
                else None
            )
            rows.append(
                {
                    "protocol": protocol,
                    "comparison_kind": (
                        "matched_quantile_approximation"
                        if protocol == "paper"
                        else "different_estimator_functionals"
                    ),
                    "comparison_target": (
                        "same_sample_exact_quantile_trimmed_Pearson"
                        if protocol == "paper"
                        else "paper_style_capped_TP_TER_vs_historical_FORCE"
                    ),
                    "clean_reference_target": "known_clean_population_Pearson",
                    "status": "completed",
                    "n_samples": n_samples,
                    "run_id": run_id + 1,
                    "seed": run_seed,
                    "rmse_force_vs_exact": approximation_score["rmse"],
                    "rmse_force_vs_true": force_score["rmse"],
                    "rmse_exact_vs_true": exact_score["rmse"],
                    "approximation_score": approximation_score,
                    "force_clean_target_score": force_score,
                    "exact_clean_target_score": exact_score,
                    "force_validity_basis": force_validity_basis,
                    "force_time_ms": force_ms,
                    "exact_time_ms": exact_ms,
                    "algorithm_diagnostics": diagnostics,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("paper", "legacy"), default="paper")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-features", type=int, default=10)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="./verification_results/convergence")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.n_features < 2:
        parser.error("--n-features must be at least 2")
    if not 0.0 <= args.contamination < 1.0:
        parser.error("--contamination must lie in [0, 1)")
    sizes = [50, 200] if args.smoke else [50, 100, 200, 500, 1000, 2000]
    runs = 2 if args.smoke else args.runs
    rows = run_convergence(
        sample_sizes=sizes,
        runs=runs,
        n_features=min(args.n_features, 5) if args.smoke else args.n_features,
        contamination=args.contamination,
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
            "smoke": bool(args.smoke),
            "offline": bool(args.offline),
            "data_dir": args.data_dir,
            "seed": args.seed,
            "sample_sizes": sizes,
            "runs": runs,
            "n_features": min(args.n_features, 5) if args.smoke else args.n_features,
            "contamination": args.contamination,
            "interpretation": (
                "Quantile approximation and clean-target error are separate quantities."
                if args.protocol == "paper"
                else "Historical FORCE and TP-TER have different tails, bound policies, caps, and centering; their discrepancy is not isolated P2 quantile error."
            ),
            "estimator_parameters": {
                name: estimator_parameters(build_estimator(args.protocol, name))
                for name in ("FORCE", "TP-TER")
            },
            "environment": environment_metadata(),
            "command_line": command_line(),
            "results": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
