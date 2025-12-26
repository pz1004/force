"""
S&P 500 Volatility Sensitivity Analysis.

This script evaluates correlation estimators' robustness to different
volatility regimes by comparing against "ground truth" correlations
estimated from low-volatility market days.

Reference correlations are computed from days with volatility below
specified percentile cutoffs (5%, 10%, 15%), following the manuscript
methodology of using quiet market periods as stable correlation benchmarks.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from force import (
    fetch_sp500_data,
    ForceEstimator,
    PearsonEstimator,
    SpearmanEstimator,
    WinsorizedEstimator,
    FastMCDEstimator,
)


def compute_daily_volatility(returns):
    """Compute daily volatility as mean absolute return across assets."""
    # returns: (T, p), consistent with manuscript
    return np.mean(np.abs(returns), axis=1)


def reference_corr_low_vol(returns, cutoff_percent):
    """
    Compute reference correlation from low-volatility days.

    Args:
        returns: (T, p) returns matrix
        cutoff_percent: fraction of high-volatility days to exclude
                       e.g., 0.10 keeps days below the 90th percentile
    """
    vol = compute_daily_volatility(returns)
    thr = np.quantile(vol, 1.0 - cutoff_percent)
    idx = vol <= thr
    X = returns[idx]
    # Pearson correlation on "stable" days
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    std = np.sqrt(np.diag(cov)) + 1e-12
    corr = cov / (std[:, None] * std[None, :])
    return corr


def rmse_offdiag(R_hat, R_ref):
    """Compute RMSE over off-diagonal elements of correlation matrices."""
    p = R_hat.shape[0]
    mask = ~np.eye(p, dtype=bool)
    return float(np.sqrt(np.mean((R_hat[mask] - R_ref[mask]) ** 2)))


def run_sensitivity(returns, estimators, cutoffs=(0.05, 0.10, 0.15)):
    """
    Run volatility sensitivity analysis across multiple cutoffs.

    Args:
        returns: (T, p) full unfiltered returns series
        estimators: dict mapping name -> callable(returns) -> corr_hat
        cutoffs: volatility percentile cutoffs for reference correlation

    Returns:
        List of dicts with cutoff, estimator name, and RMSE values
    """
    out = []
    for c in cutoffs:
        R_ref = reference_corr_low_vol(returns, cutoff_percent=c)

        for name, est in estimators.items():
            R_hat = est(returns)  # estimator runs on full data
            out.append({
                "cutoff": c,
                "estimator": name,
                "rmse": rmse_offdiag(R_hat, R_ref),
            })
    return out


if __name__ == "__main__":
    # Fetch S&P 500 data using the repo's existing loader
    print("=" * 60)
    print("S&P 500 Volatility Sensitivity Analysis")
    print("=" * 60)

    returns, _ = fetch_sp500_data()  # returns shape: (T, p)
    print(f"Loaded returns matrix: {returns.shape[0]} days × {returns.shape[1]} stocks")

    # Build estimator dictionary using repo's estimator wrappers
    # Each estimator callable maps returns -> correlation matrix
    force_est = ForceEstimator()
    pearson_est = PearsonEstimator()
    spearman_est = SpearmanEstimator()
    winsorized_est = WinsorizedEstimator()
    fastmcd_est = FastMCDEstimator()

    estimators = {
        "FORCE": lambda X: force_est.fit(X),
        "Pearson": lambda X: pearson_est.fit(X),
        "Spearman": lambda X: spearman_est.fit(X),
        "Winsorized": lambda X: winsorized_est.fit(X),
        "FastMCD": lambda X: fastmcd_est.fit(X),
    }

    # Run sensitivity analysis
    print("\nRunning sensitivity analysis across volatility cutoffs...")
    results = run_sensitivity(returns, estimators, cutoffs=(0.05, 0.10, 0.15))

    # Display results
    print("\n" + "-" * 60)
    print(f"{'Cutoff':<10} {'Estimator':<15} {'RMSE':<10}")
    print("-" * 60)

    for r in results:
        print(f"{r['cutoff']:<10.2f} {r['estimator']:<15} {r['rmse']:<10.4f}")

    # Summary by estimator (average across cutoffs)
    print("\n" + "=" * 60)
    print("Summary: Average RMSE across all cutoffs")
    print("=" * 60)

    from collections import defaultdict
    avg_rmse = defaultdict(list)
    for r in results:
        avg_rmse[r['estimator']].append(r['rmse'])

    sorted_estimators = sorted(avg_rmse.items(), key=lambda x: np.mean(x[1]))
    for name, rmses in sorted_estimators:
        print(f"{name:<15}: {np.mean(rmses):.4f}")

    # Save results to CSV
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sp500_volatility_sensitivity_{timestamp}.csv"

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Also save summary
    summary_data = [{"estimator": name, "avg_rmse": np.mean(rmses)}
                    for name, rmses in sorted_estimators]
    summary_file = output_dir / f"sp500_volatility_sensitivity_summary_{timestamp}.csv"
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

    print("\nAnalysis complete.")
