# -*- coding: utf-8 -*-
"""
FORCE: Fast Outlier-Robust Correlation Estimation

This package provides the implementation of the FORCE algorithm and related
tools for benchmarking and analysis.
"""
from .trimmed_pearson import TrimmedPearsonExact
from .core import ForceEstimator, P2Quantile, p2_desired_positions
from .legacy import LegacyForceEstimator
from .estimators import (
    CorrelationEstimator,
    PearsonEstimator,
    SpearmanEstimator,
    WinsorizedEstimator,
    FastMCDEstimator,
    ExactTrimmedEstimator,
)
from .data import (
    generate_synthetic_data,
    fetch_sp500_data,
    fetch_odds_dataset,
    fetch_genomics_data,
    generate_synthetic_data_fixed_corr,
    generate_odds_surrogate,
    load_odds_dataset,
    ExternalDataUnavailable,
)
from .utils import setup_logging

__version__ = "1.1.0"

__all__ = [
    # Core
    "ForceEstimator",
    "P2Quantile",
    "p2_desired_positions",
    "LegacyForceEstimator",
    # Comparative Estimators
    "CorrelationEstimator",
    "PearsonEstimator",
    "SpearmanEstimator",
    "WinsorizedEstimator",
    "FastMCDEstimator",
    "ExactTrimmedEstimator",
    "TrimmedPearsonExact",
    # Data Functions
    "generate_synthetic_data",
    "fetch_sp500_data",
    "fetch_odds_dataset",
    "fetch_genomics_data",
    "generate_synthetic_data_fixed_corr",
    "generate_odds_surrogate",
    "load_odds_dataset",
    "ExternalDataUnavailable",
    # Utilities
    "setup_logging",
]
