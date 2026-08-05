# FORCE: Fast Outlier-Robust Correlation Estimation

This repository contains the Python implementation accompanying the 2026
*Mathematics* paper “FORCE: Fast Outlier-Robust Correlation Estimation via
Streaming Quantile Approximation for High-Dimensional Data Streams.”

The default estimator follows the paper’s formal definitions:

1. Track marginal quantiles 0.01, 0.25, 0.50, 0.75 and 0.99 with standard P².
2. Estimate location with the median and scale with `IQR / 1.349`.
3. Compute the tail expansion ratio and inclusive adaptive bounds.
4. Compute Pearson correlation on each pair’s jointly accepted observations,
   centered on that pair’s accepted-sample means.

The historical result-generating behavior is available separately as
`LegacyForceEstimator`; it is not intended for new applications.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m pip install -e '.[dev]'
```

## Usage

```python
import numpy as np
from force import ForceEstimator

X = np.random.default_rng(0).normal(size=(1000, 20))
correlation = ForceEstimator().fit(X)
```

`ForceEstimator` accepts:

- `lambda_scale=3.0`
- `exact_cutover=5` (pure P² for all valid inputs)
- `use_ter=True`
- `ter_max=None`
- `epsilon=1e-10`

Set `exact_cutover` above five to use exact quantiles for smaller batches.

### Robustness caveat

The paper's 25% breakdown proof considers the median and IQR but not the
default uncapped tail expansion ratio. Because TER uses the 1st and 99th
percentiles, concentrated one-sided contamination can expand the acceptance
bounds before the IQR breaks down. Applications exposed to that failure mode
should set a finite `ter_max` or disable TER after validating the resulting
bias/efficiency trade-off. `VERIFICATION.md` records the mathematical conflict
and the Appendix B evidence; the implementation preserves the paper's formal
uncapped default.

## Reproducible benchmarks

The benchmark runner has two explicit protocols:

```bash
# Equation- and prose-faithful defaults
python3 scripts/run_benchmark.py --protocol paper --runs 20

# Historical committed result-generating settings
python3 scripts/run_benchmark.py --protocol legacy --runs 20

# Fast offline verification of every dataset/algorithm path
python3 scripts/run_benchmark.py --protocol paper --smoke --offline
```

All benchmark and analysis runners accept `--protocol`, `--runs`, `--seed`,
`--smoke`, `--offline`, `--data-dir`, and `--output-dir`. Outputs include raw
CSV and strict JSON containing parameters, versions, command line, provenance,
statuses, timings, and accuracy values. Numerical timing runs force Numba and
BLAS libraries to one thread, warm throwaway estimators, and time fresh
instances.

Additional production-code checks include:

```bash
python3 scripts/run_convergence_analysis.py --protocol paper
python3 scripts/run_p2_skewness_diagnostic.py --protocol paper
python3 scripts/run_ter_contamination_analysis.py --protocol paper
python3 scripts/run_sp500_volatility_sensitivity.py --protocol paper
python3 scripts/run_scalability_check.py --protocol paper
```

The paper does not identify its 50 S&P tickers or its 5000x100 Gemma dataset.
The paper protocol therefore reports these datasets as `not_reproducible`
unless the missing ticker manifest or Gemma accession is explicitly supplied.
It never substitutes synthetic data for a failed real-data download.

## Verification

Run:

```bash
pytest -q
```

See `VERIFICATION.md` for the equation-by-equation audit, paper errata,
commands, observed results, external-data provenance, and remaining
reproducibility limits.
