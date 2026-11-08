# FORCE

## Outline

[Installation](#installation) · [Batch usage](#batch) · [Streaming usage](#streaming)

FORCE estimates robust pairwise correlation for high-dimensional data streams.
It tracks marginal quantiles with P², uses the median and IQR with an adaptive
tail expansion ratio to set acceptance bounds, and computes Pearson correlation
on each pair's jointly accepted observations. The method is described in
[“FORCE: Fast Outlier-Robust Correlation Estimation via Streaming Quantile
Approximation for High-Dimensional Data Streams”](https://doi.org/10.3390/math14010191).

See the [algorithm and validation guide](VERIFICATION.md) for the estimator
contract, numerical conventions, benchmark interpretation, and verification
commands. [Reference materials](archive/README.md) and the
[research plan](force-improvement-plan.md) provide supporting records and
remaining limitations.

## Main Results

Section 3, Tables 4–6 of the [paper](mathematics-14-00191.pdf#page=19)
compare FORCE with six baselines on five datasets. The reported execution-time
speedup averages **470.53× over FastMCD** and **3.86× over Spearman**, with
dataset-dependent accuracy. The table below collects the reported mean FORCE
execution times and RMSE values; lower RMSE is better.

| Dataset | FORCE time (ms) | FORCE RMSE | TP-Exact RMSE | FastMCD RMSE |
| --- | ---: | ---: | ---: | ---: |
| Synthetic, 10% Cauchy contamination | 0.87 | 0.2001 | 0.0487 | 0.0180 |
| S&P 500 | 1.85 | 0.1186 | 0.0902 | 0.1606 |
| ODDS-mammography, 2.3% anomalies | 2.80 | 0.0157 | 0.0163 | 0.0074 |
| ODDS-satellite, 31.7% anomalies | 6.38 | 0.7274 | 0.7160 | 0.0160 |
| Genomics | 0.41 | 0.1267 | 0.1571 | 0.2583 |

The paper reports an accuracy–computation trade-off with exact trimming:
FORCE is **1.39× faster than TP-Exact on average**, but slower on mammography
(a speedup ratio of 0.71×). TP-Exact, which uses exact quantiles without TER,
has lower RMSE on the synthetic and S&P 500 benchmarks. On the synthetic
benchmark, FORCE's RMSE of 0.2001 is lower than Pearson's 0.6074.

RMSE is specific to each dataset's reference. The synthetic reference is known
by construction. The ODDS benchmarks use the FastMCD-based reference described
in Section 2.4.3. The S&P 500 reference uses days below the 90th percentile of
market volatility (Section 3.2.1); the genomics reference is Spearman correlation,
so Spearman's reported RMSE is zero by construction (Section 3.2.3). Errors
across these different targets should be interpreted separately: real-data
references are estimates, not known clean population correlations. Timings
refer to the paper's experimental environment described in Section 2.4.4.

The memory contribution is independent of stream length: quantile tracking
uses `O(p)` state and full pairwise correlation accumulators use `O(p²)`
(Section 2.3). The satellite benchmark also documents substantial estimation
error under high contamination. Its observed degradation does not establish a
universal breakdown guarantee; see the [robustness discussion](VERIFICATION.md#statistical-interpretation).

## Installation

Python 3.10 or newer is required. The current [CI matrix](.github/workflows/ci.yml)
targets Python 3.10–3.13 on Ubuntu; other versions and operating systems are
outside that matrix. From the repository root, create and activate a virtual
environment, then install the package using Bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, replace the activation line with `.\.venv\Scripts\Activate.ps1` in
PowerShell or `.venv\Scripts\activate.bat` in Command Prompt.

For comparisons across the FORCE family, keep MAD-FORCE and Dual-Bound FORCE
in separate virtual environments. MAD-FORCE requires `numba==0.63.1`, while
Dual-Bound requires `numba>=0.65`; their current requirements cannot be
satisfied in one environment.

## Usage

Provide dense, finite real numeric data: a batch has shape
`(n_samples, n_features)` and each streaming row has shape `(n_features,)`.
Keep the feature count and order consistent. There is no automatic
missing-value handling; resolve missing values before calling the estimator.
Batch `fit()` requires at least five rows and two features; streaming also
requires at least two features.

Each example is independently runnable and uses seeded synthetic data with
2,000 rows and 20 features. Features 0 and 1 have population Pearson
correlation 0.8 before trimming. Their estimated correlation provides a sanity
check beyond the output shape; finite samples and pairwise trimming can change
it. These settings illustrate the API, not a universal parameter
recommendation. `window_size` applies only to streaming; the other parameters
below apply to both estimators.

| Parameter | Constructor default | Example setting | Constraint | Practical effect |
| --- | --- | --- | --- | --- |
| `window_size` | `1000` | `100` | Integer ≥ 6; warns below 50 | Smaller windows refresh bounds sooner but repeatedly estimate quantiles from fewer rows. The first window is calibration only. |
| `lambda_scale` | `3.0` | Default | Finite and > 0 | Larger values widen acceptance bounds; smaller values reject more observations and can leave fewer usable pairs. |
| `use_ter` | `True` | Default | Boolean | Enables tail expansion of the IQR-based bounds; wider bounds can also retain more contamination. |
| `ter_max` | `None` | Default | `None` or finite ≥ 1 | `None` leaves TER uncapped. A finite cap limits expansion when `use_ter=True`, changing which observations are retained. |

Initial calls may incur Numba compilation or cache-loading overhead. To measure
steady-state performance, exercise the same execution path and configuration
on a disposable estimator in the same process, including readout when timing
it, then measure a fresh instance. This runtime warm-up is separate from the
initial streaming calibration window, which every fresh instance still consumes.

### Batch

```python
import numpy as np
from force import ForceEstimator

X = np.random.default_rng(0).normal(size=(2_000, 20))
X[:, 1] = 0.8 * X[:, 0] + 0.6 * X[:, 1]
estimator = ForceEstimator()
correlation = estimator.fit(X)
print(correlation.shape)  # (20, 20)
print(f"r[0,1] = {correlation[0, 1]:.3f}")
print("Accepted:", estimator.accepted_counts_[0, 1],
      "Status:", estimator.pair_status_codes_[0, 1])
```

`fit()` returns a NumPy `ndarray` with shape `(n_features, n_features)`.
It learns final bounds in the first pass, then accumulates pair-specific
moments in the second; each call replaces the previous batch fit. For
replayable input, `fit_replayable()` returns the same type and implements the
same two-pass estimator; see the [algorithm contract](VERIFICATION.md#algorithm-contract).

### Streaming

```python
import numpy as np
from force import StreamingForceEstimator

X = np.random.default_rng(0).normal(size=(2_000, 20))
X[:, 1] = 0.8 * X[:, 0] + 0.6 * X[:, 1]
estimator = StreamingForceEstimator(window_size=100)
for n_seen, row in enumerate(X, start=1):
    estimator.update(row)
    if n_seen % 100 == 0:
        correlation = estimator.finalize()
        if correlation is None:
            continue  # No estimation rows yet.
        print(f"rows={n_seen}, r[0,1]={correlation[0, 1]:.3f}")

correlation = estimator.finalize()  # Include any rows after the last checkpoint.
if correlation is None:
    print("No post-calibration observations yet.")
else:
    print("Final:", correlation.shape, f"r[0,1]={correlation[0, 1]:.3f}")
    print("Accepted:", estimator.accepted_counts_[0, 1],
          "Status:", estimator.pair_status_codes_[0, 1])
```

The first 100 rows calibrate the bounds and are excluded from correlation
moments. Each later window uses the preceding window's frozen bounds;
accepted-pair moments accumulate across estimation windows. `update()` returns
the estimator itself. `finalize()` returns a NumPy `ndarray` snapshot without
resetting state, or `None` until an estimation row has arrived. Here readouts
are attempted every 100 observations, with the first pair estimate printed at
row 200. The final readout includes the entire stream even when its length is
not a multiple of 100; it also handles streams ending during calibration.
Availability does not establish statistical adequacy: individual pairs may
still have too few accepted observations or zero variance. This delayed-window
estimate generally differs from the batch result.

For input arriving in chunks, `estimator.update_batch(chunk)` accepts shape
`(n_rows, n_features)` and returns the streaming estimator itself. It consumes
the rows in order under the same window schedule as repeated `update()` calls;
chunk boundaries do not define calibration or estimation windows. This
continues streaming state, whereas `ForceEstimator.fit()` performs a fresh
two-pass batch fit.

Both modes return a pairwise correlation matrix, which need not be positive
semidefinite. An indefinite result can cause Cholesky decomposition or
downstream methods requiring a positive semidefinite input to fail. Check
your downstream method's matrix requirements before using the output.

Undefined off-diagonal pairs are zero-filled; inspect
`pair_status_codes_` to distinguish them from estimable pairs:
`0=ok`, `1=insufficient_samples`, `2=zero_variance`. Check the corresponding
`accepted_counts_[i, j]` before using a pair: it counts jointly accepted
observations, not all rows seen. Status `0` means the correlation is defined,
not that its uncertainty is small. Choose any minimum-count policy for your
application and validate it on representative data. See the
[numerical conventions](VERIFICATION.md#notation-and-underspecified-cases) and
[statistical interpretation](VERIFICATION.md#statistical-interpretation),
including the limits of the default uncapped tail expansion ratio.

## License

The software is distributed under the [MIT License](LICENSE). The included
[paper PDF](mathematics-14-00191.pdf) retains its authors' copyright and the
Creative Commons Attribution (CC BY) license stated in the PDF.

## Citation

If you use this implementation in research, please cite:

S. Jang and C. Choi, “FORCE: Fast Outlier-Robust Correlation Estimation via
Streaming Quantile Approximation for High-Dimensional Data Streams,”
*Mathematics*, vol. 14, no. 1, article 191, 2026.
[doi:10.3390/math14010191](https://doi.org/10.3390/math14010191).

```bibtex
@article{jang2026force,
  author  = {Jang, Sooyoung and Choi, Changbeom},
  title   = {{FORCE}: Fast Outlier-Robust Correlation Estimation via Streaming Quantile Approximation for High-Dimensional Data Streams},
  journal = {Mathematics},
  year    = {2026},
  volume  = {14},
  number  = {1},
  pages   = {191},
  doi     = {10.3390/math14010191},
  url     = {https://doi.org/10.3390/math14010191}
}
```

Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).
