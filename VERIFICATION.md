# FORCE Implementation Verification

## Executive result

The primary estimator now implements the paper's formal FORCE definitions and
the two-pass batch form of Algorithm 1. The previous result-generating behavior
is retained only as `LegacyForceEstimator`.

The final audit was performed from commit
`36c2caebdda08775dd0a8da72aad19a6f2431956`. Existing uncommitted candidate
revisions were preserved, re-audited, and completed in place.

The paper was treated as an immutable audit input. Its SHA-256 before and after
the work is:

```text
a44f2a3a7e085b40f979e1260c6a977f1c71e64e499295bd5b3c9908cc17afac
```

The audit found four distinct classes of discrepancy:

1. The original implementation did not implement the formal estimator. It
   tracked the 5th and 95th percentiles, used a different TER expression,
   centered accepted observations on marginal medians, and adjusted P² markers
   even when their desired-position discrepancy had magnitude below one.
2. The paper contains internal mathematical conflicts: Equation (3) prints
   malformed P² desired positions; Algorithm 1 omits pair indices from
   pair-dependent moments; and the stated 25% breakdown proof ignores the
   uncapped TER's dependence on the 1st and 99th percentiles.
3. Several published experimental descriptions conflict with the paper's
   tables and committed pipeline. These conflicts are recorded rather than
   resolved by inventing missing data or modifying the PDF.
4. The candidate benchmark revisions still reused warmed estimator instances,
   capped the paper-mode Appendix B TER calculation, omitted common CLI/report
   metadata from several runners, refreshed cache retrieval timestamps, and
   could accept mutable S&P/Gemma caches without mandatory provenance
   sidecars. These execution and provenance gaps were revised during the final
   audit.

Status terms used below:

- **matched**: implementation agrees with the paper.
- **revised**: the implementation was changed to agree with the paper.
- **paper erratum**: the printed formula or description is internally
  inconsistent; the standard or otherwise unambiguous interpretation is used.
- **paper inconsistency**: two paper claims conflict and no implementation
  change can make both true.
- **partially applicable**: a theoretical statement is conditional and does
  not establish the stronger surrounding claim.
- **partially implemented**: the batch algorithm is present but an optional
  alternative described by the paper is not.
- **not reproducible**: essential identifiers or selection rules are absent.

## Equation and algorithm audit

| Paper item | Status | Verification and implementation |
| --- | --- | --- |
| Equations (1)-(2), contamination and breakdown definitions | matched | These define the evaluation model rather than executable estimator steps. Fixed-seed contamination tests exercise selected contamination regimes but do not establish a breakdown point. |
| Equation (3), P² desired marker positions | paper erratum | The printed cases are malformed and omit the standard five-marker indexing. FORCE uses `[1, 1+(N-1)φ/2, 1+(N-1)φ, 1+(N-1)(1+φ)/2, N]`. Tests check all positions. |
| Equations (4)-(5), P² marker adjustment | revised | The parabolic update with linear fallback is implemented. Interior markers move only when the desired-position discrepancy has magnitude at least one and the neighboring marker spacing permits movement. An independent incremental implementation is compared with the compiled kernel for all five probabilities. |
| Five P² instances at `φ={0.01,0.25,0.50,0.75,0.99}` | revised | The former 5th/95th percentile trackers were replaced. Five observations initialize each tracker; pure P² is the default after initialization. |
| Equation (6), location `q50` | revised | Marginal location is the streaming median. |
| Equation (7), scale `(q75-q25)/1.349` | revised | The paper's IQR calibration is used without an undocumented scale floor. |
| Equations (8)-(9), TER | revised | `max(1, abs(q99-q50)/(abs(q50-q01)+epsilon))` is used. A raw lower-tail distance below `epsilon` produces `TER=1` and a warning. |
| Equation (10), adaptive thresholds | revised | Inclusive lower and upper bounds are `location ± lambda_scale*TER*scale`, with `lambda_scale=3`. |
| Equation (11), optional TER cap | matched | `ter_max=None` leaves TER uncapped by default; a positive finite cap is available explicitly. |
| Equation (12), pairwise acceptance | revised | A row contributes to pair `(j,k)` only when both coordinates fall inside their own inclusive bounds. |
| Equation (13), trimmed correlation | revised | Each pair uses the means of its jointly accepted observations. Undefined off-diagonal correlations are zero, arithmetic roundoff is clipped to `[-1,1]`, and the paper's unit diagonal is retained. |
| Equation (14), delayed/windowed alternative | partially implemented | The default `fit(X)` follows Algorithm 1's two-pass batch interpretation. The delayed, windowed, and exponentially weighted online alternatives described in prose are not public estimator modes. |
| Equation (15) and Theorem 1, claimed 25% breakdown point | paper inconsistency | The proof considers only the median and IQR. The default uncapped TER also depends on `q01` and `q99`; slightly more than 1% one-sided contamination can move an extreme quantile arbitrarily, expand the bounds, and admit arbitrarily large observations. Appendix B's large 5%-contamination TER error is consistent with this omitted failure mode. The implementation retains Equations (8)-(11) and does not claim that the uncapped estimator has a proven 25% breakdown point. |
| Theorem 2, `O(Np²)` full-matrix time | matched | Quantile tracking is `O(Np)` and pairwise accepted-statistic accumulation is `O(Np²)`. Isolated-process measurements confirm approximately linear growth with `N` for fixed `p` and increasing work with `p`. |
| Theorem 3, `O(p²)` memory independent of `N` | matched | Retained estimator diagnostics are `O(p)` and the returned correlation matrix is `O(p²)`. Tests show retained state is independent of sample count. The caller-owned batch `X` is not counted as estimator state. |
| Equations (16)-(18), Pearson, Spearman, and Winsorized baselines | revised | Baselines validate finite 2-D inputs, handle zero variance consistently, and clip covariance-to-correlation roundoff. Winsorization retains the paper's 5% limits. The paper's `O(Np² log N)` full-matrix cost for rank/exact methods is an upper bound, not the production implementation's tight cost: ranks/quantiles are computed once per feature, followed by `O(Np²)` correlation. |
| FastMCD baseline | revised | Failure now propagates; there is no silent substitution of ordinary Pearson correlation. |
| Equation (19), RMSE | matched | Benchmarks compute matrix RMSE against the declared synthetic truth or real-data reference. |
| Equation (20), 95% confidence interval | matched | Benchmark output contains raw paired runs and summary mean, sample standard deviation, and normal 95% CI for 20 measured runs. |

## Algorithm 1 mapping

| Algorithm stage | Status | Batch implementation |
| --- | --- | --- |
| Initialize five P² estimators per coordinate | revised | The default probabilities and five-sample initialization are centralized in `force.core`. |
| First pass: update marginal quantiles | revised | Every finite input row updates every marginal tracker in input order. |
| Compute location, scale, TER, and bounds | revised | Equations (6)-(10) share one threshold routine used by FORCE and exact trimmed Pearson. |
| Second pass: pairwise acceptance and sufficient statistics | paper inconsistency; revised | Equation (13) requires pair-specific accepted means, but Algorithm 1 initializes `S_j` per coordinate and then updates it inside every pair loop. Production code follows Equation (13): each pair accumulates its own count, online means, second moments, and co-moment over jointly accepted rows. |
| Finalize the correlation matrix | revised | Pair-specific accepted means are used; degenerate pairs are deterministic and finite. |
| Incremental unbounded-stream API | partially implemented | The paper's executable pseudocode needs two passes or delayed bounds. This repository exposes the documented two-pass `fit(X)` batch API, not a threshold-changing one-pass stream API. |

## Theoretical statement audit

| Paper item | Status | Verification |
| --- | --- | --- |
| Proposition 1, consistency for the trimmed target | partially applicable | The implementation computes the stated empirical trimmed functional. The proposition is explicitly conditional on consistent P² quantiles, continuous marginals, finite second moments, and non-vanishing acceptance probability; the test suite verifies recurrence and rank accuracy but does not turn those assumptions into a proof. |
| Claimed `N^-1/2` asymptotic rate | partially applicable | No finite-sample or distribution-free rate is asserted by the package. Quantile-bound error can dominate in the small-sample and heavy-tail regimes documented by the paper. |
| Corollary 1, separated contamination | paper inconsistency | The corollary requires the estimated acceptance region to remain the clean region. With uncapped TER, contamination above the extreme-quantile tail probability can move `q99` or `q01` and expand that region even while the IQR remains stable. The corollary is valid only with an additional bound-stability assumption or a suitable TER cap. |
| Table 2 complexity claims | partially applicable | FORCE's `O(Np²)` time and `O(p²)` full-output memory are matched. For Spearman, Winsorized, and exact-quantile trimming, production code performs feature-wise ordering once and then `O(Np²)` correlation; the paper's multiplicative `O(Np² log N)` expression is not a tight bound for this implementation. |
| Repeated `O(p)` FORCE memory claims in Tables 8/A1 and prose | paper inconsistency | Quantile diagnostics alone are `O(p)`, but a full correlation matrix or all pair accumulators require `O(p²)`, as Theorem 3 correctly states. Reports therefore separate estimator diagnostics from output bytes. |

## API and package changes

The public estimator remains `ForceEstimator`; its constructor is now:

```python
ForceEstimator(
    lambda_scale=3.0,
    exact_cutover=5,
    use_ter=True,
    ter_max=None,
    epsilon=1e-10,
)
```

All parameters and inputs are validated. Inputs must be finite, numeric,
two-dimensional arrays with at least five rows and two columns.
`exact_cutover>5` is an explicit small-sample hybrid; the default uses P² for
every valid batch.

`TrimmedPearsonExact` shares the threshold and pairwise-correlation routines.
The broken historical `ExactTrimmedEstimator` name is a deprecated
compatibility wrapper. Historical FORCE mathematics are isolated in
`force.legacy.LegacyForceEstimator` and are selected only by the `legacy`
benchmark protocol.

The package now supports its `src` layout in pytest, has no `gemmapy`
dependency, requires Python 3.10 or newer, ignores generated caches/results,
and exports version `1.1.0`. A freshly built wheel and an isolated editable
target installation both imported `ForceEstimator`, `LegacyForceEstimator`,
and `TrimmedPearsonExact` successfully. The editable target was loaded with
`site.addsitedir`, which processes the `.pth` file generated by setuptools.
Python 3.11 was unavailable on this host, so execution used Python 3.12.3.

## Experimental protocol audit

### Paper protocol

The default protocol follows the paper's formal definitions and prose:

- `N=1000`, `p=10`;
- 10% whole-row standard-Cauchy contamination scaled by 10;
- corrected FORCE, pure P², `lambda_scale=3`, `epsilon=1e-10`, uncapped TER;
- verified ODDS mammography and satellite downloads;
- FastMCD as the real-data reference, except for the paper's separately
  described low-volatility Pearson S&P reference;
- 20 measured runs on fresh estimator instances after a separate throwaway
  warm-up, with Numba and numerical libraries restricted to one thread.

Actual ODDS files were validated:

| Dataset | Shape | Outliers | SHA-256 |
| --- | ---: | ---: | --- |
| Mammography | `11183 x 6` | 260 | `271ebb568314a856666d3504b4882e21b0ea6e1ba9e648ad256d572a36df597e` |
| Satellite | `6435 x 36` | 2036 | `6feac3112b9c14e1c3e60afc437f2f3d29dc1000119c9f950c628078778d6aa0` |

No synthetic data are substituted if an external download fails. Cache
metadata preserve source URL and retrieval time and add validation time,
checksum, shape, labels or identifiers, preprocessing, and reference details.
ODDS content is re-hashed against fixed expected values on every load. Mutable
S&P and Gemma caches now require a valid sidecar, matching source identifiers,
date/preprocessing parameters, a retrieval timestamp, and a recomputed
SHA-256; missing, malformed, stale, or corrupted caches are rejected instead
of being silently self-certified.

### Legacy protocol

The legacy protocol preserves the settings that generated the committed
results:

- 50-dimensional synthetic data with Uniform `[-10,10]` whole-row
  contamination;
- the historical 12-ticker, 2000-2024 S&P selection;
- generated mammography and satellite *surrogates*, explicitly labeled as
  synthetic;
- inferred Gemma source GSE6306, 1203 samples, and 20 top-variance genes;
- the historical reference choices and legacy FORCE equations.

### Paper inconsistencies and blockers

| Paper claim | Conflicting evidence | Status |
| --- | --- | --- |
| Synthetic data have 10 dimensions and scaled Cauchy corruption | Table 3 and the committed result path use 50 dimensions and Uniform corruption | paper erratum; both protocols are explicit |
| S&P experiment uses 50 constituents in 2015-2024 | Table 3 and the committed code use 12 tickers and 6288 observations from 2000-2024 | not reproducible under `paper` until the 50-ticker manifest is supplied |
| Genomics data are Gemma `5000 x 100` | Table 3 reports `1203 x 20`; the committed path is consistent with GSE6306 plus top-variance selection | not reproducible under `paper` until the accession and 5000-sample/100-feature rule are supplied |
| Timing was repeated 50 times | Methods, Equation (20), and result tables use `n=20` | paper erratum; 20 measured runs are used |
| Published synthetic result demonstrates Cauchy robustness | The exact Table 6 values are regenerated only by the Uniform/legacy path | published values are legacy-reproducible but do not verify the written Cauchy experiment |
| Published ODDS results use the named public datasets | Legacy values are regenerated by committed synthetic surrogates; actual ODDS files give materially different results | published ODDS values are not measurements of the downloaded ODDS files |

### Table and appendix traceability

| Paper item | Status | Observed result |
| --- | --- | --- |
| Table 1, FastMCD baseline | partially reproducible | Legacy data reproduce the displayed order of magnitude, subject to scikit-learn and hardware differences. The named public ODDS datasets do not reproduce the displayed ODDS rows. |
| Table 2, complexity comparison | partially applicable | FORCE time/state claims are verified; the sorting-baseline expressions are loose rather than tight production-code bounds, and the full FORCE output is `O(p²)`. |
| Table 3, dataset summary | paper inconsistency | Synthetic, S&P, and genomics dimensions conflict with the prose. Paper and legacy protocols expose both interpretations without inventing identifiers. |
| Tables 4-5, timings and speedups | not reproducible numerically | Timing is hardware/version dependent and the paper conflicts on 50 versus 20 repetitions. Fresh-estimator, one-thread measurements reproduce the expected structural ordering, not the absolute values. |
| Table 6, RMSE | partially reproducible | The complete legacy synthetic row is reproduced to four decimals. S&P and surrogate ODDS rows are mostly reproduced; corrected Cauchy and actual ODDS inputs yield materially different values. |
| Table 7, satellite breakdown demonstration | not reproducible as described | The displayed values are reproduced by the legacy synthetic surrogate, not the validated ODDS satellite file. The experiment also does not prove a 25% breakdown point for uncapped TER. |
| Tables 8-9, summary and contamination interpretation | paper inconsistency | These inherit the undocumented/legacy datasets and 25% TER claim. The repeated `O(p)` memory statement excludes the required full correlation output. |
| Appendix A / Table A1, S&P sensitivity | not reproducible under `paper` | The 50-ticker manifest is absent. The legacy 12-ticker sensitivity path remains available and is explicitly labeled. |
| Appendix B / Table A2, TER trade-off | matched numerically; contradicts Theorem 1 | The production estimator reproduces the displayed qualitative and near-numerical behavior. At only 5% one-sided contamination, TER MSE rises to about `0.1013` versus `0.0003` for fixed bounds, illustrating the extreme-quantile vulnerability omitted from the breakdown proof. |

## Observed verification results

All values below are means over 20 runs unless noted. Timing is machine
dependent; correctness comparisons use RMSE and raw paired records.

### Corrected paper protocol

| Dataset | Pearson | Spearman | Winsorized | FastMCD | TP-Exact | TP-TER | FORCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic Cauchy, `1000 x 10` | 0.8090 | 0.2112 | 0.3027 | 0.0120 | 0.0203 | 0.0473 | 0.2224 |
| ODDS mammography | 0.1498 | 0.2056 | 0.0466 | 0.0000 | 0.1797 | 0.2101 | 0.1306 |
| ODDS satellite | 0.4540 | 0.2849 | 0.4319 | 0.0000 | 0.3790 | 0.4540 | 0.4540 |

FastMCD has zero RMSE on the actual ODDS rows because the paper design defines
FastMCD itself as the real-data reference. Mammography produced scikit-learn
determinant warnings; the benchmark records their count and text instead of
suppressing or replacing that estimator.

S&P and genomics paper runs emitted structured `not_reproducible` records:

```text
sp500:   The paper specifies 50 S&P 500 constituents but provides no ticker manifest.
genomics: The paper specifies Gemma 5000x100 data but gives no dataset accession or sampling rule.
```

The aggregate report status is `not_reproducible` when no requested dataset
can be completed; individual dataset records retain their own blocker status
and explanation.

### Legacy reproduction

| Dataset | Pearson | Spearman | Winsorized | FastMCD | TP-Exact | TP-TER | FORCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic | 0.6074 | 0.2075 | 0.2665 | 0.0180 | 0.0487 | 0.0549 | 0.2001 |
| S&P, 12 tickers | 0.1335 | 0.1240 | 0.1321 | 0.1647 | 0.0901 | 0.0909 | 0.1186 |
| Mammography surrogate | 0.0723 | 0.0345 | 0.0308 | 0.0077 | 0.0163 | 0.0163 | 0.0157 |
| Satellite surrogate | 0.7160 | 0.6473 | 0.7361 | 0.0161 | 0.7160 | 0.7160 | 0.7274 |
| GSE6306, `1203 x 20` | 0.1326 | 0.0000 | 0.1333 | 0.2732 | 0.1571 | 0.1319 | 0.1401 |

The legacy synthetic row reproduces Table 6 exactly to four decimals. S&P
FORCE, Pearson, and Spearman also reproduce their displayed values; exact
trimmed Pearson differs by 0.0001 and FastMCD by 0.0041. The surrogate ODDS
rows reproduce all displayed values exactly except FastMCD by 0.0001-0.0003.
These small FastMCD differences are consistent with using scikit-learn 1.9.0
instead of the paper's 1.6.1.

For GSE6306, Pearson, Spearman, Winsorized, TP-Exact, and TP-TER reproduce the
displayed values. FORCE is 0.1401 versus 0.1267 and FastMCD is 0.2732 versus
0.2583. P² is observation-order dependent and the paper does not provide its
sample ordering; FastMCD also differs by library version. The genomics row is
therefore only partially reproducible.

### Convergence, TER, and complexity

For 20 repetitions at sample sizes 50, 100, 200, 500, 1000, and 2000, mean
RMSE between P² FORCE and exact-quantile trimmed Pearson was respectively
0.1461, 0.1804, 0.1447, 0.1182, 0.0604, and 0.0334. The non-monotone
small-sample region is expected for order-sensitive approximate extreme
quantiles; the overall error decreases as the trackers stabilize.

The production-code Appendix B run used 100 repetitions. Selected
`(TER MSE, fixed-threshold MSE)` values were:

- asymmetric contamination 5%, 10%, and 15%:
  `(0.101338, 0.000342)`, `(0.154670, 0.000207)`,
  `(0.186093, 0.000296)`;
- coherent Student-t tails with 10, 5, and 3 degrees of freedom:
  `(0.001255, 0.001399)`, `(0.002867, 0.003176)`,
  `(0.005660, 0.006164)`.

The isolated one-thread scalability check reported:

| `N x p` | Time (ms) | Explicit estimator state | Output |
| --- | ---: | ---: | ---: |
| `1000 x 10` | 0.515 | 800 B | 800 B |
| `4000 x 10` | 1.938 | 800 B | 800 B |
| `1000 x 20` | 1.213 | 1600 B | 3200 B |
| `1000 x 40` | 2.910 | 3200 B | 12,800 B |

Peak process RSS was approximately 303 MiB in each subprocess because it
includes Python, NumPy, and the JIT runtime. Explicit state and output sizes
separate the estimator's algorithmic storage from runtime overhead. Each case
has a timeout/OOM status path so a failed high-memory case cannot terminate the
main verification process.

## Tests and commands

The final fast suite contains 81 focused tests covering P² positions, marker
invariants, independent recurrence agreement, quantile accuracy, exact
cutover, Equations (6)-(13), inclusive boundaries, degenerate TER warnings,
accepted-pair centering, parameter/input validation, numerical output
contracts, Cauchy robustness, baseline failure semantics, protocol settings,
deterministic fixtures, mandatory S&P/Gemma sidecars, corrupted-cache
rejection, ODDS metadata, published legacy reproduction, aggregate blocker
statuses, common runner CLIs, uncapped TER expansion/capping, and strict
machine-readable schemas.

Final result:

```text
NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 pytest -q -W default
81 passed in 11.42s
```

Representative execution commands:

```bash
# Fast tests and all-path smoke verification
NUMBA_NUM_THREADS=1 pytest -q -W default
NUMBA_NUM_THREADS=1 python3 scripts/run_benchmark.py \
  --protocol paper --smoke --offline \
  --data-dir verification_results/data \
  --output-dir verification_results/final-audit-20260731/smoke/paper/run_benchmark

# Corrected synthetic and downloaded ODDS datasets
NUMBA_NUM_THREADS=1 python3 scripts/run_benchmark.py \
  --protocol paper \
  --datasets synthetic,odds-mammography,odds-satellite \
  --runs 20 --seed 42 \
  --offline --data-dir verification_results/data \
  --output-dir verification_results/final-audit-20260731/paper-full

# Explicit paper-data blockers
python3 scripts/run_benchmark.py \
  --protocol paper --datasets sp500,genomics --runs 20 --offline \
  --data-dir verification_results/data \
  --output-dir verification_results/final-audit-20260731/paper-blockers

# Historical synthetic reproduction
NUMBA_NUM_THREADS=1 python3 scripts/run_benchmark.py --protocol legacy \
  --datasets synthetic,sp500,odds-mammography,odds-satellite,genomics \
  --runs 20 --seed 42 --offline \
  --data-dir verification_results/data \
  --output-dir verification_results/final-audit-20260731/legacy-full

# Production-code diagnostics
NUMBA_NUM_THREADS=1 python3 scripts/run_convergence_analysis.py \
  --protocol paper --runs 20 --seed 42 --offline \
  --data-dir verification_results/data \
  --output-dir verification_results/final-audit-20260731/convergence-paper
NUMBA_NUM_THREADS=1 python3 scripts/run_p2_skewness_diagnostic.py \
  --protocol paper --runs 30 --seed 42 --offline \
  --data-dir verification_results/data \
  --output-dir verification_results/final-audit-20260731/p2-paper
NUMBA_NUM_THREADS=1 python3 scripts/run_ter_contamination_analysis.py \
  --protocol paper --runs 100 --seed 42 --offline \
  --data-dir verification_results/data \
  --output-dir verification_results/final-audit-20260731/ter-paper
python3 scripts/run_scalability_check.py \
  --protocol paper --runs 20 --offline --timeout 120 \
  --memory-limit-gib 16 --data-dir verification_results/data \
  --output-dir verification_results/final-audit-20260731/scalability-paper
```

All seven benchmark and analysis scripts completed their reduced smoke path
under both `paper` and `legacy` (14 runs total). Results are written as raw CSV plus strict JSON
(no NaN or Infinity tokens) with protocol, command line, versions, numerical
thread settings, seeds, estimator settings, per-run status/timing,
provenance, and preprocessing. Supported statuses are `completed`,
`skipped_external`, `not_reproducible`, `failed`, `timeout`, and `oom`.
Generated result and cache directories are intentionally ignored by Git.
The final audit bundle at
`verification_results/final-audit-20260731` contains 23 strict JSON reports
and 23 parseable CSV files: 21 reports completed and two paper-data reports
returned `not_reproducible`. Validation found no `NaN`/`Infinity`, missing
command/environment metadata, invalid status, or missing CSV sidecar.

## Verification environment and integrity

Full runs were made on:

```text
Python 3.12.3
NumPy 2.4.6
SciPy 1.17.1
pandas 3.0.3
scikit-learn 1.9.0
Numba 0.65.1
Linux 6.18.33.1-microsoft-standard-WSL2, x86_64
Intel Core i7-14700KF, 28 logical CPUs
62 GiB RAM
```

The paper-update script was neither run nor modified; its recorded SHA-256 is
`6c6ba79b7a05a51960e08fdc72551c46038f98b889514d4e87d873bec1255131`.
Rendered visual inspection of paper pages 8, 10-19, 22, and 37-39 confirmed
the printed P² formula, TER definitions, Algorithm 1, theoretical claims,
dataset descriptions, result tables, appendices, and 20/50-run conflict used
in this audit.
Final verification uses `git diff --check`, clean wheel/editable imports, the
complete test suite, strict parsing of all final reports, explicit inspection
of changed paths, and a second SHA-256 calculation of the PDF.
