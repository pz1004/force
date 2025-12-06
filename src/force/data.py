"""
Data loading and generation utilities for benchmarking.

This module provides functions to generate synthetic datasets with known
properties and to fetch real-world datasets from various sources, which
are used in the benchmark and analysis scripts.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Tuple, Optional

def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    contamination: float = 0.1,
    seed: Optional[int] = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates synthetic data with a known correlation structure and contamination.

    A true correlation matrix is generated, and data is sampled from a
    multivariate normal distribution. A specified fraction of the samples
    is then replaced with outliers drawn from a uniform distribution.

    Args:
        n_samples: The total number of samples to generate.
        n_features: The number of features in the dataset.
        contamination: The fraction of samples to replace with outliers.
        seed: An optional random seed for reproducibility.

    Returns:
        A tuple containing:
        - X (NDArray): The generated data matrix of shape (n_samples, n_features).
        - true_corr (NDArray): The ground truth correlation matrix of shape
          (n_features, n_features).
    """
    rng = np.random.default_rng(seed)
    A = rng.random((n_features, n_features))
    true_cov = A @ A.T
    d = np.sqrt(np.diag(true_cov))
    true_corr = true_cov / np.outer(d, d)

    X = rng.multivariate_normal(np.zeros(n_features), true_corr, n_samples)

    n_outliers = int(n_samples * contamination)
    if n_outliers > 0:
        idx = rng.choice(n_samples, n_outliers, replace=False)
        X[idx] = rng.uniform(-10, 10, (n_outliers, n_features))
        
    return X, true_corr


def fetch_sp500_data() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fetches historical S&P 500 stock data using the yfinance library.

    Downloads daily closing prices for a predefined list of tickers, calculates
    log returns, and uses data from low-volatility days to estimate a
    "ground truth" correlation matrix.

    Returns:
        A tuple containing:
        - X (NDArray): The log returns data matrix.
        - true_corr (NDArray): The estimated ground truth correlation matrix.
        
    Raises:
        ImportError: If `yfinance` is not installed.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Please install yfinance: pip install yfinance")

    print("Fetching S&P 500 Historical Data via yfinance...")
    tickers = ['XOM', 'GE', 'MSFT', 'JPM', 'PG', 'JNJ', 'INTC', 'PFE', 'T', 'VZ', 'IBM', 'KO']
    data = yf.download(tickers, start="2000-01-01", end="2025-01-01", progress=False)['Close']
    
    if data.empty:
        raise ValueError("yfinance returned no data. Check network or ticker symbols.")
        
    returns = np.log(data / data.shift(1)).dropna()
    X = returns.values
    
    # Estimate ground truth from low-volatility days
    market_vol = np.mean(np.abs(X), axis=1)
    quiet_days_mask = market_vol < np.percentile(market_vol, 90)
    X_quiet = X[quiet_days_mask]
    
    return X, np.corrcoef(X_quiet, rowvar=False)


def fetch_odds_dataset(
    dataset_name: str, n_max_samples: int = 10000
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates a synthetic dataset modeled after the ODDS benchmark datasets.

    Creates data with properties (sample size, feature count, outlier fraction)
    matching either the 'mammography' or 'satellite' datasets from the ODDS
    (Outlier Detection DataSets) library.

    Args:
        dataset_name: The name of the dataset to model ('mammography' or 'satellite').
        n_max_samples: The maximum number of samples to generate.

    Returns:
        A tuple containing:
        - X (NDArray): The generated data matrix.
        - true_corr (NDArray): The ground truth correlation matrix.
    """
    dataset_specs = {
        'mammography': {'n_samples': 11183, 'n_features': 6, 'outlier_fraction': 0.023},
        'satellite': {'n_samples': 6435, 'n_features': 36, 'outlier_fraction': 0.317}
    }
    spec = dataset_specs.get(dataset_name, dataset_specs['satellite'])
    print(f"Generating ODDS-like dataset: {dataset_name} ({spec['outlier_fraction']:.1%} outliers)")
    
    n_samples = min(spec['n_samples'], n_max_samples)
    n_features = spec['n_features']
    n_inliers = int(n_samples * (1 - spec['outlier_fraction']))
    n_outliers = n_samples - n_inliers

    rng = np.random.default_rng(42)
    A = rng.standard_normal((n_features, n_features))
    cov = A @ A.T
    true_corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    
    X_clean = rng.multivariate_normal(np.zeros(n_features), true_corr, n_inliers)
    
    if dataset_name == 'mammography':
        X_out = rng.uniform(-5, 5, (n_outliers, n_features))
    else: # Satellite-like clustered outliers
        X_out = rng.multivariate_normal(np.ones(n_features) * 3, np.eye(n_features) * 0.5, n_outliers)
        
    X = np.vstack([X_clean, X_out])
    return rng.permutation(X), true_corr


def fetch_genomics_data(
    n_features_limit: int = 500
) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """
    Fetches a genomics dataset from Gemma using the gemmapy library.

    Searches for a 'cancer' dataset, downloads its processed expression data,
    and selects the most variable features.

    Args:
        n_features_limit: The maximum number of features to keep.

    Returns:
        A tuple containing:
        - X (NDArray | None): The data matrix, or None if fetching fails.
        - ref_corr (NDArray | None): A reference correlation matrix (computed
          with Spearman), or None.
          
    Raises:
        ImportError: If `gemmapy` is not installed.
    """
    try:
        import gemmapy
    except ImportError:
        raise ImportError("Please install gemmapy: pip install gemmapy")

    print("Fetching Genomics data via gemmapy...")
    try:
        api = gemmapy.GemmaPy()
        datasets = api.get_datasets(query='cancer', limit=20)
        if datasets.empty:
            raise ValueError("GemmaPy found no datasets for the query 'cancer'.")
            
        target = datasets.iloc[0]
        expr = api.get_dataset_processed_expression(str(target['experiment_ID']))
        df = expr.to_pandas().select_dtypes(include=[np.number]).dropna()
        
        X = df.values if df.shape[0] > df.shape[1] else df.values.T
        if X.shape[1] > n_features_limit:
            feature_variances = np.var(X, axis=0)
            top_feature_indices = np.argsort(feature_variances)[-n_features_limit:]
            X = X[:, top_feature_indices]
            
        print(f"Genomics Data Loaded: {X.shape}")
        
        # Since there's no ground truth, use a robust estimate (Spearman) as a reference
        ref_corr, _ = stats.spearmanr(X)
        return X, ref_corr
        
    except Exception as e:
        print(f"Genomics data fetching failed: {e}")
        return None, None
        
def generate_synthetic_data_fixed_corr(
    n_samples: int,
    n_features: int = 10,
    contamination: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generates synthetic data with a fixed true correlation structure.
    
    This version is used for the convergence analysis to ensure the underlying
    correlation is identical across runs with different sample sizes. It uses
    a fixed seed for the correlation structure but a variable seed for the
    data generation itself.

    Args:
        n_samples: The number of samples to generate.
        n_features: The number of features.
        contamination: The fraction of Cauchy-distributed outliers.
        seed: An optional random seed for the data generation phase.

    Returns:
        A tuple containing:
        - X (NDArray): The generated data matrix.
        - true_corr (NDArray): The fixed ground truth correlation matrix.
    """
    # Use a fixed seed to ensure the correlation structure is always the same
    rng_corr = np.random.default_rng(42)
    A = rng_corr.random((n_features, n_features))
    true_cov = A @ A.T
    d = np.sqrt(np.diag(true_cov))
    true_corr = true_cov / np.outer(d, d)
    
    # Use a different, optional seed for generating the actual data points
    rng_data = np.random.default_rng(seed)
    X = rng_data.multivariate_normal(np.zeros(n_features), true_corr, n_samples)
    
    n_outliers = int(n_samples * contamination)
    if n_outliers > 0:
        idx = rng_data.choice(n_samples, n_outliers, replace=False)
        X[idx] = rng_data.standard_cauchy((n_outliers, n_features)) * 10
        
    return X, true_corr
