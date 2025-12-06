# FORCE: Fast Outlier-Robust Correlation Estimation

This repository provides the official Python implementation of the FORCE 
(Fast Outlier-Robust Correlation Estimation) algorithm, as presented in the 
accompanying paper submitted to MDPI Mathematics (2025).

## Overview

FORCE is a robust correlation estimator designed to be computationally efficient 
while maintaining high accuracy in the presence of outliers. It operates in 
three stages:

1.  **Streaming Quantile Estimation:** Utilizes the P² algorithm for a fast,
    single-pass calculation of statistical quantiles.
2.  **Adaptive Thresholding:** Employs a novel Tail-Extremity Ratio (TER) to
    dynamically set trimming bounds for outlier removal.
3.  **Robust Accumulation:** Computes the final correlation matrix on the 
    clean subset of the data.

This implementation is optimized with Numba for high-performance computation.

## Project Structure

The project is organized into a modern Python package structure:

-   `src/force/`: The core Python package.
    -   `core.py`: Main `ForceEstimator` implementation.
    -   `estimators.py`: Comparative estimators (Pearson, Spearman, etc.).
    -   `data.py`: Data loading and generation utilities.
    -   `utils.py`: Helper functions for analysis and plotting.
-   `scripts/`: Standalone scripts for running benchmarks and analyses.
    -   `run_benchmark.py`: Runs a full benchmark suite.
    -   `run_convergence_analysis.py`: Analyzes the P² algorithm's convergence.
-   `examples/`: Example usage of the `force` library.
-   `tests/`: Unit and integration tests.

## Installation

To install the required dependencies, run:

```bash
pip install -e .[dev]
```

## Quick Start

You can run a simple demonstration of the FORCE estimator using the example script:

```bash
python examples/usage_example.py
```

This will generate synthetic data, compute the robust correlation matrix using
FORCE, and compare it to the classical Pearson correlation.

## Running Benchmarks

To replicate the benchmarks from the paper, use the `run_benchmark.py` script:

```bash
# Run a full benchmark with 20 runs per dataset
python scripts/run_benchmark.py --runs 20 --output_dir ./benchmark_results
```

The results, including plots and a markdown summary, will be saved in the
`benchmark_results` directory.
