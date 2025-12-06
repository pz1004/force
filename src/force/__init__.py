# -*- coding: utf-8 -*-
"""
FORCE: Fast Outlier-Robust Correlation Estimation

This package provides the implementation of the FORCE algorithm and related
tools for benchmarking and analysis.
"""
from .core import ForceEstimator
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
)
from .utils import setup_logging

__version__ = "1.0.0"

__all__ = [
    # Core
    "ForceEstimator",
    # Comparative Estimators
    "CorrelationEstimator",
    "PearsonEstimator",
    "SpearmanEstimator",
    "WinsorizedEstimator",
    "FastMCDEstimator",
    "ExactTrimmedEstimator",
    # Data Functions
    "generate_synthetic_data",
    "fetch_sp500_data",
    "fetch_odds_dataset",
    "fetch_genomics_data",
    "generate_synthetic_data_fixed_corr",
    # Utilities
    "setup_logging",
]
