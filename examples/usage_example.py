"""
================================================================================
Usage Example for the FORCE library
================================================================================

This script provides a simple, self-contained example of how to use the
`ForceEstimator` to compute a robust correlation matrix on synthetic data.

To run this example:
    python examples/usage_example.py
"""

import numpy as np
from force import ForceEstimator, generate_synthetic_data

def run_simple_demonstration():
    """
    A simple demonstration of the ForceEstimator.
    """
    print("--- FORCE Library Usage Example ---")

    # 1. Generate synthetic data with known properties
    # Let's create a small dataset with 100 samples, 5 features, and 10% outliers.
    n_samples = 100
    n_features = 5
    contamination = 0.1
    
    print(f"\nGenerating synthetic data with:\n" 
          f"  - {n_samples} samples\n" 
          f"  - {n_features} features\n" 
          f"  - {contamination*100:.0f}% outlier contamination\n")
          
    X, true_corr = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        contamination=contamination,
        seed=42
    )

    print("Ground Truth Correlation Matrix:")
    print(np.round(true_corr, 2))

    # 2. Initialize and fit the ForceEstimator
    # The default estimator uses pure P² for every valid batch. Pass an
    # exact_cutover above n_samples to request the optional exact hybrid.
    print("\nInitializing and fitting ForceEstimator...")
    force_estimator = ForceEstimator()
    robust_corr = force_estimator.fit(X)
    
    # 3. Display the results
    print("\nRobust Correlation Matrix (estimated by FORCE):")
    print(np.round(robust_corr, 2))
    
    # For comparison, let's compute the classical Pearson correlation
    # which is sensitive to outliers.
    classical_corr = np.corrcoef(X, rowvar=False)
    print("\nClassical Pearson Correlation Matrix (sensitive to outliers):")
    print(np.round(classical_corr, 2))
    
    # Calculate the error of each estimate relative to the ground truth
    rmse_force = np.sqrt(np.mean((robust_corr - true_corr)**2))
    rmse_classical = np.sqrt(np.mean((classical_corr - true_corr)**2))
    
    print(f"\nEstimation Error (RMSE vs. Ground Truth):")
    print(f"  - FORCE Estimator:      {rmse_force:.4f}")
    print(f"  - Classical Estimator:  {rmse_classical:.4f}")

    if rmse_force < rmse_classical:
        print("\nAs expected, the FORCE estimate is closer to the ground truth.")
    else:
        print("\nIn this instance, the classical estimate was closer, which can happen with low contamination.")


if __name__ == "__main__":
    run_simple_demonstration()
