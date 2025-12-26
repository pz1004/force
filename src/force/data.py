"""
Data loading and generation utilities for benchmarking.

This module provides functions to generate synthetic datasets with known
properties and to fetch real-world datasets from various sources, which
are used in the benchmark and analysis scripts.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
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
    # 12 Representative Tickers 
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

    print(f"S&P 500 Data Loaded: {X.shape}")
    return X, np.corrcoef(X_quiet, rowvar=False)


def fetch_odds_dataset(
    dataset_name: str, n_max_samples: int = 20000
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
    n_samples_target: int = 5000,
    n_features_target: int = 20
) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """
    Fetches a genomics dataset from Gemma using the gemmapy library.

    Searches for tissue-related datasets with sufficient samples, downloads
    processed expression data, and selects the most variable features.
    Default targets ~1200 samples × 20 features (GSE6306 brain tissue dataset).

    Args:
        n_samples_target: Target number of samples (will use up to this many).
        n_features_target: Target number of features (most variable genes).

    Returns:
        A tuple containing:
        - X (NDArray | None): The data matrix of shape (n_samples, n_features).
        - ref_corr (NDArray | None): A reference correlation matrix (Spearman).

    Raises:
        ImportError: If `gemmapy` is not installed.
    """
    try:
        import gemmapy
    except ImportError:
        raise ImportError("Please install gemmapy: pip install gemmapy")

    print(f"Fetching Genomics data (target: {n_samples_target} samples, {n_features_target} features)...")
    api = gemmapy.GemmaPy()

    # Search for datasets with sufficient samples - 'tissue' query has largest datasets
    datasets = api.get_datasets(query='tissue', limit=100)

    if datasets.empty:
        print("No datasets found for query 'tissue'")
        return None, None

    # Sort by sample count descending to find datasets with most samples
    datasets = datasets.sort_values('experiment_sample_count', ascending=False)

    # Find a dataset with enough samples (at least 500 for stable correlation)
    target_dataset = None
    for _, ds in datasets.iterrows():
        sample_count = ds.get('experiment_sample_count', 0)
        if sample_count >= 500:
            target_dataset = ds
            break

    if target_dataset is None:
        # Fallback to largest available
        print("No dataset with 500+ samples found, using largest available.")
        target_dataset = datasets.iloc[0]

    exp_name = target_dataset['experiment_short_name']
    exp_id = target_dataset['experiment_ID']
    exp_samples = target_dataset['experiment_sample_count']
    print(f"Selected: {exp_name} (ID: {exp_id}, samples: {exp_samples})")

    try:
        expr = api.get_dataset_processed_expression(str(exp_id))

        # Handle both DataFrame and objects with to_pandas() method
        if hasattr(expr, 'to_pandas'):
            df = expr.to_pandas()
        else:
            df = expr

        # Select only numeric columns (expression values)
        df = df.select_dtypes(include=[np.number])

        # Drop samples (columns) with any NaN before transposing
        valid_samples = df.columns[df.notna().all()]
        df = df[valid_samples]

        # Genomics standard: Rows=Genes (features), Cols=Samples (observations)
        # Shape is typically (genes, samples) -> we need (samples, genes)
        X = df.values.T  # Transpose: (samples, genes)

        print(f"Data shape after cleanup: {X.shape} (samples × genes)")

        # Subsample to target samples if needed
        if X.shape[0] > n_samples_target:
            rng = np.random.default_rng(42)
            idx = rng.choice(X.shape[0], n_samples_target, replace=False)
            X = X[idx]

        # Select most variable genes to reach target features
        if X.shape[1] > n_features_target:
            feature_variances = np.var(X, axis=0)
            top_indices = np.argsort(feature_variances)[-n_features_target:]
            X = X[:, top_indices]

        print(f"Genomics Data Loaded: {X.shape} (samples × genes)")

        if X.shape[0] < 50:
            print("Warning: Sample size is small. Correlation estimates may be unstable.")

        # Reference using Spearman (robust baseline)
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
