# FORCE algorithm and verification

This document specifies the implementation of [FORCE](https://doi.org/10.3390/math14010191),
*Mathematics* 2026, 14(1), 191,
[doi:10.3390/math14010191](https://doi.org/10.3390/math14010191), and describes
its validation. The paper PDF SHA-256 is:

```text
a44f2a3a7e085b40f979e1260c6a977f1c71e64e499295bd5b3c9908cc17afac
```

## Algorithm contract

`ForceEstimator.fit(X)` implements Algorithm 1's two-pass estimator.
`fit_replayable(source_factory, chunk_size)` implements the same estimator for
replayable sources. `StreamingForceEstimator` implements the delayed-window
policy described by Remark 9 and Equation 14.

The default parameters are:

```python
ForceEstimator(
    lambda_scale=3.0,
    exact_cutover=5,
    use_ter=True,
    ter_max=None,
    epsilon=1e-10,
    exact_prefix=0,
)
```

Inputs must be finite numeric matrices with at least five observations and two
features. The shared P² and pair-moment kernels make batch/replay decisions
independent of source chunk boundaries for identical ordered observations.
P² remains sensitive to observation order.

| Paper component | Implementation |
| --- | --- |
| P² quantiles, Equations 3–5 | Five standard five-marker P² trackers per coordinate estimate probabilities 0.01, 0.25, 0.50, 0.75 and 0.99. Five observations initialize each tracker. Interior marker adjustments require a desired-position discrepancy of at least one and sufficient neighboring spacing; the parabolic proposal has the standard linear fallback. |
| Location and scale, Equations 6–7 | Location is `q50`; scale is `(q75-q25)/1.349`. |
| Tail expansion, Equations 8–9 | `TER = max(1, abs(q99-q50)/(abs(q50-q01)+epsilon))`. A lower-tail distance below epsilon gives `TER=1` with a warning. The default has no TER cap. |
| Bounds and acceptance, Equations 10–12 | Inclusive bounds are `q50 ± lambda_scale*TER*scale`. A row contributes to a pair only if both coordinates pass their marginal bounds. |
| Pair correlation, Equation 13 | Counts, means, variances and co-moments belong to each pair's jointly accepted set. The output is that set's Pearson correlation. |
| Two-pass schedule, Algorithm 1 | The first pass learns quantiles and freezes final bounds. The second pass accumulates pair-specific moments. |
| Delayed-window schedule, Equation 14 | One complete window calibrates the first bounds and is excluded from correlation moments. Every subsequent window uses the preceding window's frozen bounds, while fresh P² trackers learn bounds for the next window. Pair moments are cumulative. |

### Notation and underspecified cases

Equation 3's desired-position expression is interpreted using the standard P²
positions
`[1, 1+(N-1)*phi/2, 1+(N-1)*phi, 1+(N-1)*(1+phi)/2, N]`.
Algorithm 1's sums carry pair indices, as required by Equation 13: a coordinate
can have different accepted means for different partners.

The following conventions make the API behavior explicit:

- Crossed P² quartiles produce a warning, a diagnostic flag and zero scale.
  The estimates are not reordered. This policy covers the negative-IQR case
  not specified by the paper; it does not establish zero population variance.
- Undefined off-diagonal correlations are represented by zero, with nonzero
  `pair_status_codes_`: `1=insufficient_samples`, `2=zero_variance`. Code `0`
  identifies an estimable pair. Diagonals are one as an output convention.
  Scientific scoring uses the validity codes, rather than treating every
  finite compatibility value as an estimate.
- Pairwise acceptance can produce a matrix that is not positive semidefinite.
  No PSD projection is applied.
- `exact_prefix>=5` provides exact linear quantile readouts through that prefix
  while ordinary P² consumes all observations in the background. The next
  observation switches readouts to P² and releases the prefix buffer. This
  specifies the optional handoff left open by the paper; it can be discontinuous
  and does not improve subsequent P² markers.
- `exact_cutover>5` instead provides exact quantiles for a complete batch shorter
  than the cutoff. It is mutually exclusive with `exact_prefix>0`. Both options
  are disabled by default.
- Winsorized comparisons use linear sample quantiles at 0.05 and 0.95. The paper
  does not specify the empirical quantile convention; the optional integer
  tail-count rule is separately identified in estimator metadata.

## Numerical behavior and memory

Pair moments use offset-centered Welford updates, adaptive power-of-two scaling
and exponent-aware normalization. There is no arbitrary variance floor for
positive finite moments. P² interpolation and exact linear readouts avoid
avoidable intermediate overflow. Unsupported arithmetic raises an error;
nonfinite quantities are not published as successful estimates.

Replay checksums validate observation identity, shape and order across the two
traversals. A one-shot iterator cannot replace a repeatable source factory.
Failed batch/replay fits leave the previous diagnostic snapshot unchanged.
A failed online processing operation invalidates that estimator instance.

| Execution mode | Memory accounting |
| --- | --- |
| Array `fit(X)` | Pair state and output require `O(p²)`; converting caller input can require `O(Np)`. |
| Replay, chunk size `B` | Auxiliary state and workspace are `O(p²+Bp)`, excluding source-owned buffering and storage. |
| Online | Retained numeric state is `O(p²+p)`; validation/conversion of an input chunk uses `O(Bp)`. |
| Exact prefix | Adds an `O(N_exact*p)` buffer until the transition. |

Quantile tracking alone uses `O(p)`. Full pair accumulators and a dense
correlation output require `O(p²)`. `state_nbytes_` measures retained NumPy
arrays, not peak workspace, Python overhead, external storage or returned
snapshots. Process RSS and tracked-allocation measurements have separate
sampling and native-allocation limitations.

## Statistical interpretation

The batch/replay target is correlation conditional on joint acceptance. It
need not equal untrimmed clean correlation, even if every contaminated
observation is rejected.

The paper's 25% breakdown argument considers the median and IQR, but the
uncapped TER also depends on the 1st and 99th percentiles. Concentrated
one-sided contamination can move an extreme quantile, expand the acceptance
bounds and admit arbitrarily large observations before the IQR breaks down.
The implementation follows the uncapped formula without asserting a universal
25% breakdown guarantee. A cap or disabled TER defines a different bound policy
and does not alone establish robustness of approximate P² under adversarial
observation order. Appendix B illustrates the TER trade-off.

Consistency for a deterministic trimmed population target is conditional on
quantile consistency, suitable marginal regularity, positive acceptance and
finite selected moments. Quantile consistency alone supplies no root-N rate.
The one-sided TER and absolute epsilon also limit sign and scale equivariance
of the complete acceptance rule, even when fixed-mask Pearson arithmetic has
those invariances.

Online moments retain previous acceptance decisions. With fixed window size
`B`, each calibration window produces random bounds whose error need not vanish
as the total stream grows. Under iid observations, define
`g_B,jk(x) = P_Z(x is accepted for pair (j,k) by calibration block Z)`.
When bounds are finite almost surely, acceptance is positive, selected moments
are finite and selected variances are positive, pooled moments target
correlation under the distribution proportional to `g_B,jk(x) dF_jk(x)`.
This is a consequence of the implemented schedule, not an additional theorem
from the paper. Adjacent block contributions share calibration/evaluation
observations, so ordinary iid correlation uncertainty formulas do not apply.
Drift and serial dependence require their own assumptions.

## Evaluation contract

Algorithmic validation compares quantiles, bounds, acceptance masks, accepted
counts and correlations with independent references. Approximate and exact
estimators share their bound policy and execution schedule. Clean-target error
is distinguished from approximation error. Undefined accuracy metrics remain
null with explicit coverage and validity reasons.

Random-order benchmark replicates independently permute one fixed dataset and
share each permutation across methods. Their intervals describe conditional
mean error over row orders, rather than sampling uncertainty across datasets.
Fixed-order repetitions measure timing variability and have no RMSE interval.
Timing uses fresh estimators after separate warm-up, one numerical thread and
matched data/order/reference identities. Failed, missing and unwarmed pairs
are counted; exploratory paired timing tests are unadjusted for multiplicity.

The paper protocol uses the specified Cauchy contamination and dataset-specific
reference definitions. For genomics, Section 3.2.3's Spearman reference is used.
The paper does not identify its 50 S&P tickers or its 5000-by-100 Gemma dataset;
those full-data paths require the missing identifiers. Data loaders validate
provenance and checksums and do not substitute synthetic data for unavailable
external datasets. Smoke fixtures are explicitly identified.

## Validation

The 8 November 2026 checks used Python 3.13.9:

- **454 tests passed in 20.05 seconds** against the extracted source distribution.
- The actual wheel installed in an isolated environment and passed dependency,
  license, batch, replay, online and example checks outside the checkout.
- Every publication file matched the built source archive, and wheel runtime
  modules matched their source files.
- Citation schema, workflow lint, archive hashes and whitespace checks passed.
- An independent P²/bounds/accepted-Pearson probe covered normal, Cauchy,
  ordered, discrete, constant, large-offset and concentrated-outlier streams.
  Accepted counts matched exactly; batch/replay results were identical. Maximum
  correlation differences were `2.22e-15` for batch and `1.11e-15` for online.

These checks establish the tested implementation behavior; they do not prove
statistical guarantees or numerical reproduction of every paper table.
GitHub CI is configured for Python 3.10–3.13. The hosted matrix must be checked
on the published commit.

Run the validation commands from the repository root:

```bash
python -m pip install -e '.[dev]'
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q
python examples/usage_example.py
python scripts/run_algorithm_diagnostics.py --smoke --offline
python scripts/run_scalability_check.py --mode replay --smoke --offline
python scripts/run_scalability_check.py --mode online --smoke --offline
```

Build into an empty distribution directory for packaging validation:

```bash
python -m pip install build
python -m build --outdir dist
python tools/verify_exclusion_build.py --dist-dir dist
python tools/verify_distribution.py --dist-dir dist
```

The checker installs dependencies from the configured package index. It checks
the installed wheel and executes the source archive's full suite. The software
uses the MIT license; the separately published paper retains its own CC BY notice.
Manuscripts, reference PDFs, historical archives, datasets, plans, and generated
outputs are local research material excluded from Git and built distributions.
The checker rejects these files in both wheels and source archives.

The [exclusion build regression](tools/verify_exclusion_build.py) extracts the
built source archive into a temporary tree and creates synthetic research files.
It checks the shipped `.gitignore` in a temporary Git repository, then prepends
fixture inclusion directives to `MANIFEST.in` so the original exclusions must
remove files already selected for inclusion. It builds both a source archive and
a wheel directly from that tree, checks their inventories independently, and
requires retained software and configuration files. Real local research material
is never copied or changed. Run this command with the build tooling used above;
ordinary unit tests do not require build tools or package downloads. CI runs the
regression after its Python 3.13 build.

Wheel inventory checks cover the archive root and the relocated installation
paths under `<distribution>.data/{data,purelib,platlib,scripts,headers}/`.
Repository disclosure checks continue to use repository-relative paths.

Runnable README examples use `<!-- executable-example: batch -->` and
`<!-- executable-example: streaming -->` immediately before their Python opening fences (`` ```python ``).
Both identifiers are required;
additional unique identifiers are allowed, using a lowercase letter followed by
lowercase letters, digits, underscores, or hyphens. The distribution checker
executes every marked example in a separate isolated interpreter against the
installed wheel. Unmarked snippets are ignored; duplicate identifiers, malformed
markers, and empty or unclosed marked fences fail verification.
