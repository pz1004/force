"""
================================================================================
FORCE Convergence Analysis
================================================================================

This script generates a figure showing the convergence of the FORCE estimator's
P² quantile approximation to the exact trimmed correlation as the sample
size (N) increases.

It addresses a key question: what is the "cost" in accuracy of using a
streaming quantile approximation (P²) compared to an exact (but memory-intensive)
quantile calculation, especially in small-sample scenarios.

The script compares three correlation matrices:
1. FORCE (using P² quantiles)
2. Exact Trimmed (using numpy.percentile)
3. The ground truth correlation used to generate the data

Usage:
    python scripts/run_convergence_analysis.py --runs 20 --output_dir ./convergence_results
"""

import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from typing import List

import numpy as np
import pandas as pd

from force import ForceEstimator, ExactTrimmedEstimator
from force.data import generate_synthetic_data_fixed_corr
from force.utils import (
    compute_correlation_rmse,
    generate_convergence_figure,
    setup_logging,
)

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger("ConvergenceAnalysis")


@dataclass
class ConvergenceResult:
    """Data class for storing results from a single convergence run."""
    n_samples: int
    run_id: int
    rmse_force_vs_exact: float
    rmse_force_vs_true: float
    rmse_exact_vs_true: float
    force_time_ms: float
    exact_time_ms: float


def run_convergence_experiment(
    sample_sizes: List[int],
    n_runs: int,
    n_features: int,
    contamination: float,
) -> List[ConvergenceResult]:
    """
    Runs the convergence experiment across a range of sample sizes.

    For each sample size, it generates data and computes correlation using
    both the FORCE estimator and the ExactTrimmedEstimator, then compares
    the results against each other and against the ground truth.

    Args:
        sample_sizes: A list of sample sizes (N) to test.
        n_runs: The number of independent runs for each sample size.
        n_features: The number of features in the synthetic data.
        contamination: The outlier contamination rate for the synthetic data.

    Returns:
        A list of ConvergenceResult objects.
    """
    results = []
    force_estimator = ForceEstimator(exact_cutover=5)  # Use P² even for small N
    exact_estimator = ExactTrimmedEstimator()

    for n_samples in sample_sizes:
        logger.info("Testing N = %d...", n_samples)
        for run_id in range(n_runs):
            # Generate data with a unique seed for each run to ensure variety
            X, true_corr = generate_synthetic_data_fixed_corr(
                n_samples=n_samples,
                n_features=n_features,
                contamination=contamination,
                seed=run_id * 1000 + n_samples,
            )

            # Run FORCE (P² quantiles)
            start_time = time.time()
            force_corr = force_estimator.fit(X)
            force_time_ms = (time.time() - start_time) * 1000

            # Run Exact Trimmed (numpy.percentile)
            start_time = time.time()
            exact_corr = exact_estimator.fit(X)
            exact_time_ms = (time.time() - start_time) * 1000

            # Compute RMSE metrics
            results.append(
                ConvergenceResult(
                    n_samples=n_samples,
                    run_id=run_id + 1,
                    rmse_force_vs_exact=compute_correlation_rmse(force_corr, exact_corr),
                    rmse_force_vs_true=compute_correlation_rmse(force_corr, true_corr),
                    rmse_exact_vs_true=compute_correlation_rmse(exact_corr, true_corr),
                    force_time_ms=force_time_ms,
                    exact_time_ms=exact_time_ms,
                )
            )
    return results


def main(args: argparse.Namespace) -> None:
    """Main function to run the analysis and generate outputs."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "convergence_analysis.log")

    sample_sizes = [50, 100, 200, 500, 1000, 2000]

    logger.info("=" * 60)
    logger.info("Starting FORCE Convergence Analysis")
    logger.info("Sample sizes: %s", sample_sizes)
    logger.info("Runs per size: %d", args.runs)
    logger.info("=" * 60)

    results = run_convergence_experiment(
        sample_sizes=sample_sizes,
        n_runs=args.runs,
        n_features=args.n_features,
        contamination=args.contamination,
    )

    df = pd.DataFrame([asdict(r) for r in results])

    # Save raw results
    df.to_csv(out_dir / "convergence_raw_results.csv", index=False)

    # Generate convergence plot
    generate_convergence_figure(df, out_dir / "convergence_plot.pdf")

    logger.info("Convergence analysis complete. Results saved in %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FORCE Convergence Analysis")
    parser.add_argument(
        "--runs", type=int, default=50, help="Number of runs per sample size."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./convergence_results",
        help="Directory to save results and plots.",
    )
    parser.add_argument(
        "--n_features", type=int, default=10, help="Number of features for synthetic data."
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Contamination rate for synthetic data.",
    )
    main(parser.parse_args())
