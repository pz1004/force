"""
================================================================================
FORCE Benchmark Runner
================================================================================

This script runs a comprehensive benchmark comparing the FORCE estimator against
several other standard and robust correlation estimators.

Key Features:
- Runs experiments multiple times for statistical significance.
- Includes a warm-up phase to mitigate JIT compilation overhead in timings.
- Calculates mean, standard deviation, and 95% confidence intervals.
- Generates a summary markdown table and plots.

Usage:
    # Run a standard benchmark with 20 iterations per dataset
    python scripts/run_benchmark.py --runs 20 --output_dir ./benchmark_results

    # Run only the FastMCD algorithm (computationally intensive)
    python scripts/run_benchmark.py --only-fastmcd --runs 5
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path for imports when package is not installed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# --- Import from the new `force` library structure ---
from force import (
    CorrelationEstimator,
    FastMCDEstimator,
    ForceEstimator,
    PearsonEstimator,
    SpearmanEstimator,
    WinsorizedEstimator,
)
from force.trimmed_pearson import TrimmedPearsonExact
from force.data import (
    fetch_genomics_data,
    fetch_odds_dataset,
    fetch_sp500_data,
    generate_synthetic_data,
)
from force.utils import (
    BenchmarkResult,
    generate_benchmark_plots,
    generate_markdown_report,
    setup_logging,
)

# --- Basic logger setup ---
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("BenchmarkRunner")


def run_benchmark_loop(
    X: np.ndarray,
    true_corr: np.ndarray,
    dataset_label: str,
    algorithms: Dict[str, CorrelationEstimator],
    n_runs: int,
) -> List[BenchmarkResult]:
    """
    Executes the benchmark loop for a single dataset.

    This function performs a warm-up run, followed by a specified number of
    measurement runs for each algorithm on the given dataset.

    Args:
        X: The input data matrix.
        true_corr: The ground truth correlation matrix for RMSE calculation.
        dataset_label: A string label for the dataset being tested.
        algorithms: A dictionary of algorithm names to estimator instances.
        n_runs: The number of times to run the benchmark for each algorithm.

    Returns:
        A list of BenchmarkResult objects containing the detailed results.
    """
    results = []
    n_samples, n_features = X.shape
    logger.info(
        "--- Benchmarking %s (N=%d, D=%d, Runs=%d) ---",
        dataset_label, n_samples, n_features, n_runs,
    )

    for algo_name, estimator in algorithms.items():
        # 1. Warm-up Phase (to account for JIT compilation, etc.)
        try:
            estimator.fit(X)
        except Exception as e:
            logger.error("Warm-up failed for %s on %s: %s", algo_name, dataset_label, e)
            continue

        # 2. Measurement Phase
        timings = []
        for i in range(n_runs):
            try:
                start_time = time.time()
                est_corr = estimator.fit(X)
                duration_ms = (time.time() - start_time) * 1000

                mask = np.triu(np.ones_like(est_corr, dtype=bool), k=1)
                rmse = np.sqrt(np.mean((est_corr[mask] - true_corr[mask]) ** 2))

                results.append(
                    BenchmarkResult(
                        dataset=dataset_label,
                        n_samples=n_samples,
                        n_features=n_features,
                        algorithm=algo_name,
                        run_id=i + 1,
                        time_ms=duration_ms,
                        rmse=rmse,
                    )
                )
                timings.append(duration_ms)

            except Exception as e:
                logger.error("Run %d failed for %s on %s: %s", i + 1, algo_name, dataset_label, e)

        if timings:
            logger.info("%-15s | Avg Time: %6.2fms", algo_name, np.mean(timings))

    return results


def main(args: argparse.Namespace) -> None:
    """
    Main function to orchestrate the benchmark execution.
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Add a timestamp to all output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    setup_logging(out_dir / f"benchmark_{timestamp}.log")

    # --- Initialize Algorithms ---
    # The order in this dictionary determines the execution order.
    # FastMCD will now run on all datasets by default.
    algorithms_to_run: Dict[str, CorrelationEstimator] = {
        "Pearson": PearsonEstimator(),
        "TrimmedPearsonExact(no TER)": TrimmedPearsonExact(use_ter=False),
        "TrimmedPearsonExact(TER)": TrimmedPearsonExact(use_ter=True),       
        "Spearman": SpearmanEstimator(),
        "Winsorized": WinsorizedEstimator(),
        "FastMCD": FastMCDEstimator(),
        "FORCE": ForceEstimator(),
    }

    all_results: List[BenchmarkResult] = []

    # --- Dataset Execution Loop ---

    # 1. Synthetic Data
    logger.info("Preparing Synthetic dataset...")
    X_syn, true_corr_syn = generate_synthetic_data(
        n_samples=1000, n_features=50, contamination=0.1
    )
    all_results.extend(
        run_benchmark_loop(X_syn, true_corr_syn, "Synthetic", algorithms_to_run, args.runs)
    )

    # 2. S&P 500 Data
    try:
        logger.info("Preparing S&P 500 dataset...")
        X_sp, true_corr_sp = fetch_sp500_data()
        all_results.extend(
            run_benchmark_loop(X_sp, true_corr_sp, "SP500", algorithms_to_run, args.runs)
        )
    except Exception as e:
        logger.error("Skipping S&P 500 dataset: %s", e)

    # 3. ODDS Datasets
    for odds_name in ["mammography", "satellite"]:
        try:
            logger.info("Preparing ODDS-%s dataset...", odds_name)
            X_odds, true_corr_odds = fetch_odds_dataset(odds_name)
            all_results.extend(
                run_benchmark_loop(
                    X_odds, true_corr_odds, f"ODDS-{odds_name}", algorithms_to_run, args.runs
                )
            )
        except Exception as e:
            logger.error("Skipping ODDS-%s dataset: %s", odds_name, e)
    
    # 4. Genomics Data
    try:
        logger.info("Preparing Genomics dataset...")
        X_gen, ref_corr_gen = fetch_genomics_data()
        if X_gen is not None:
            all_results.extend(
                run_benchmark_loop(X_gen, ref_corr_gen, "Genomics", algorithms_to_run, args.runs)
            )
    except Exception as e:
        logger.error("Skipping Genomics dataset: %s", e)


    # --- Reporting ---
    if not all_results:
        logger.error("No results were collected. Aborting reporting.")
        return

    df = pd.DataFrame(all_results)
    
    # Reorder algorithms in DataFrame to match execution order for reporting
    algo_order = list(algorithms_to_run.keys())
    df['algorithm'] = pd.Categorical(df['algorithm'], categories=algo_order, ordered=True)
    df = df.sort_values(by=['dataset', 'algorithm'])
    
    # Save raw data
    raw_path = out_dir / f"results_raw_{timestamp}.csv"
    df.to_csv(raw_path, index=False)
    logger.info("Raw results saved to %s", raw_path)

    # Generate Markdown report and plots
    md_path = out_dir / f"summary_table_{timestamp}.md"
    generate_markdown_report(df, md_path)

    plot_path = out_dir / f"benchmark_plots_{timestamp}.png"
    generate_benchmark_plots(df, plot_path)

    logger.info("Benchmark Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FORCE Benchmarks")
    parser.add_argument(
        "--runs", type=int, default=20, help="Number of measured runs per experiment."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_results",
        help="Directory to save benchmark results, logs, and plots.",
    )
    main(parser.parse_args())
