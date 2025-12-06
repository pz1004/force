"""
Utilities for logging, analysis, and visualization.

This module contains helper functions used by the benchmark and analysis
scripts, including logging setup, statistical calculations, and plotting
routines for generating reports.
"""
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from numpy.typing import NDArray


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Path, level: int = logging.INFO) -> None:
    """
    Configures a logger to output to both console and a file.

    Args:
        log_file: Path to the log file.
        level: The logging level (e.g., logging.INFO).
    """
    logger = logging.getLogger("force")
    if logger.hasHandlers():
        return  # Avoid adding duplicate handlers

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# =============================================================================
# Data Structures for Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Data class for storing results from a single benchmark run."""
    dataset: str
    n_samples: int
    n_features: int
    algorithm: str
    run_id: int
    time_ms: float
    rmse: float


# =============================================================================
# Statistical and Analysis Functions
# =============================================================================

def compute_correlation_rmse(
    corr1: NDArray[np.float64], corr2: NDArray[np.float64]
) -> float:
    """
    Computes the Root Mean Squared Error (RMSE) between two correlation matrices.

    The comparison is performed only on the upper triangular part of the
    matrices to avoid redundant comparisons.

    Args:
        corr1: The first correlation matrix.
        corr2: The second correlation matrix.

    Returns:
        The RMSE value.
    """
    if corr1.shape != corr2.shape:
        raise ValueError("Input matrices must have the same shape.")
    mask = np.triu(np.ones_like(corr1, dtype=bool), k=1)
    return np.sqrt(np.mean((corr1[mask] - corr2[mask]) ** 2))


def calculate_ci(
    data: NDArray[np.float64], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculates the mean and confidence interval for a given dataset.

    Args:
        data: A 1D array of numerical data.
        confidence: The confidence level for the interval.

    Returns:
        A tuple containing (mean, lower_bound, upper_bound).
    """
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2:
        mean_val = np.mean(a)
        return mean_val, mean_val, mean_val
    
    mean_val, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean_val, mean_val - h, mean_val + h


# =============================================================================
# Reporting and Visualization
# =============================================================================

def generate_markdown_report(df: pd.DataFrame, output_path: Path) -> None:
    """
    Creates a Markdown summary table from benchmark results.

    The table includes mean, standard deviation, and 95% confidence intervals
    for both computation time and RMSE.

    Args:
        df: A pandas DataFrame containing `BenchmarkResult` data.
        output_path: The path to save the Markdown file.
    """
    grouped = df.groupby(['dataset', 'algorithm'])
    lines = [
        "# FORCE Algorithm Benchmark Results",
        f"\n**Total Runs per Experiment:** {df['run_id'].max()}\n",
        "| Dataset | Algorithm | Time (ms) [Mean ± Std] | Time 95% CI | RMSE [Mean ± Std] | RMSE 95% CI |",
        "|---|---|---|---|---|---|"
    ]

    for dataset in df['dataset'].unique():
        for i, algo in enumerate(df['algorithm'].unique()):
            subset = df[(df['dataset'] == dataset) & (df['algorithm'] == algo)]
            if subset.empty:
                continue

            time_mean, t_ci_low, t_ci_high = calculate_ci(subset['time_ms'].values)
            time_std = np.std(subset['time_ms'].values)
            
            rmse_mean, r_ci_low, r_ci_high = calculate_ci(subset['rmse'].values)
            rmse_std = np.std(subset['rmse'].values)

            ds_display = f"**{dataset}**" if i == 0 else ""
            algo_display = f"**{algo}**" if "FORCE" in algo else algo
            
            lines.append(
                f"| {ds_display} | {algo_display} | {time_mean:.2f} ± {time_std:.2f} | "
                f"[{t_ci_low:.2f}, {t_ci_high:.2f}] | {rmse_mean:.4f} ± {rmse_std:.4f} | "
                f"[{r_ci_low:.4f}, {r_ci_high:.4f}] |"
            )

    output_path.write_text("\n".join(lines))
    logging.info(f"Markdown report saved to {output_path}")


def generate_benchmark_plots(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generates box plots for time and RMSE from benchmark results.

    Args:
        df: A pandas DataFrame containing `BenchmarkResult` data.
        output_path: The path to save the plot image.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Time Performance Plot (Log Scale)
    sns.boxplot(data=df, x='dataset', y='time_ms', hue='algorithm', ax=axes[0])
    axes[0].set_yscale('log')
    axes[0].set_title('Computational Time (Log Scale)', fontsize=14)
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=45)

    # RMSE Accuracy Plot
    df_rmse = df[df['rmse'] > 0]
    if not df_rmse.empty:
        sns.barplot(data=df_rmse, x='dataset', y='rmse', hue='algorithm', ax=axes[1], errorbar='sd')
        axes[1].set_title('Estimation Error (RMSE)', fontsize=14)
        axes[1].set_ylabel('RMSE (Lower is Better)')
        axes[1].set_xlabel('')
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Benchmark plots saved to {output_path}")


def generate_convergence_figure(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generates the convergence analysis figure.

    This plot shows the RMSE between FORCE and the exact trimmed estimator
    as sample size increases, demonstrating the convergence of the P-Square
    approximation.

    Args:
        df: DataFrame with convergence experiment results.
        output_path: Path to save the figure (e.g., 'figure.pdf').
    """
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

    grouped = df.groupby('n_samples')['rmse_force_vs_exact'].agg(['mean', 'std']).reset_index()
    
    ax.errorbar(
        grouped['n_samples'], grouped['mean'], yerr=grouped['std'],
        fmt='o-', capsize=5, markersize=8, label='FORCE vs. Exact Trimmed'
    )
    ax.axhline(y=0.01, color='r', linestyle='--', label='Convergence Threshold (0.01)')
    ax.axhspan(0, 0.01, alpha=0.1, color='green', label='Converged Region')
    
    ax.set_xlabel('Sample Size (N)')
    ax.set_ylabel('RMSE (FORCE vs. Exact)')
    ax.set_title('Convergence of P² Quantile Approximation')
    ax.set_xscale('log')
    ax.set_xticks([50, 100, 200, 500, 1000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Convergence figure saved to {output_path}")
