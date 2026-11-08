# FORCE paper improvement plan

This plan improves the mathematical specification, claims and experimental
evidence in [the FORCE paper](mathematics-14-00191.pdf), *Mathematics* 2026,
14, 191, DOI [10.3390/math14010191](https://doi.org/10.3390/math14010191).
The implementation and paper define the baseline for this research plan. The
paper PDF is the reference source. Its SHA-256 is `a44f2a3a7e085b40f979e1260c6a977f1c71e64e499295bd5b3c9908cc17afac`.

The main algorithm follows the median/IQR/TER bounds and
pair-specific trimmed Pearson correlation. Independent verification
checks this baseline. The work below concerns manuscript clarifications,
conditional theory and separately evaluated enhancements. Matching published
RMSE or timing decimals is not an acceptance criterion.

The algorithm contract and validation are recorded in
[VERIFICATION.md](VERIFICATION.md).
Derivations below are proposed additions to the paper. They distinguish results
that follow directly from the specified functional from proof obligations and
experiments that remain to be completed.

## 1. Correct the specification before strengthening claims

| Paper location | Required correction or clarification | Acceptance criterion |
| --- | --- | --- |
| Equation 3 | Replace the malformed desired positions by `[1, 1+(N−1)φ/2, 1+(N−1)φ, 1+(N−1)(1+φ)/2, N]`. Five separate trackers per feature each contain five markers: 25p marker heights, still O(p). Their estimates use the same data and are not statistically independent. | The initialization and recurrence match [the shared P² implementation](src/force/_quantiles.py) and [the original specification](https://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf); distinguish actual integer ranks from desired ranks. |
| Algorithm 1, second pass | Give every accepted sum, mean, variance and co-moment pair indices. A coordinate can have different accepted means for different partners. | Pseudocode agrees with Equation 13 and [pair moments](src/force/_moments.py), including an asymmetric acceptance example. |
| Equation 7, Gaussian calibration | Dividing the population Gaussian IQR by `2Φ⁻¹(0.75)` gives its standard deviation. The implemented rounded constant 1.349 gives approximately `0.9999848 × sigma`. Neither statement implies finite-sample unbiasedness of the estimated scale. | Retain 1.349 for alignment; replace the paper's claim `E[scale_hat]=sigma` with the population calibration and its rounding qualification. |
| Equations 7–13 | Specify inclusive bounds, absolute epsilon and its branch, optional TER cap, nonfinite-input rejection, and unsupported-arithmetic errors. Separate genuine zero IQR, crossed estimated quartiles, insufficient accepted counts and zero accepted variance. | A policy table matches [bounds](src/force/core.py), validity codes and warnings. Warned negative-IQR collapse is identified as an implementation convention, not Equation 7's literal value. |
| Equation 13 output | State that undefined pairs have no correlation estimate. The API's zero fill is a compatibility representation; unit diagonals are an output convention. | Scientific tables use validity masks and coverage; an undefined zero never receives an ordinary accuracy score. |
| Remark 2 and Section 2.4.4 | Define exact-prefix readouts and their handoff: ordinary P² consumes every row in the background, exact readouts end after N_exact, and the buffer is released on the next row. | Disclose possible discontinuity and no improvement to later P² markers. Keep the separate whole-small-batch exact fallback distinct. |
| Remark 9 | Give executable pseudocode for the implemented delayed-window mode: initial calibration excluded, preceding window's bounds frozen, current trackers reset by logical window, moments cumulative. | Row-wise and arbitrarily chunked updates agree. Exponentially weighted and merged mini-batch modes are labeled proposed alternatives. |
| Equation 18 | Declare the empirical quantile convention. The paper protocol uses linear sample quantiles at 0.05 and 0.95 followed by clipping. | Record this convention in output metadata; the optional tail-count Winsorization convention must be identified separately. |
| Corollary 1, contamination notation | Write the law as `H=(1−eta)F+eta G`, or draw from F/G using a Bernoulli indicator. A weighted sum of two random variables is not a mixture draw. | The model, generated samples and contamination fraction have the same meaning. |

Also state that pairwise acceptance need not produce a positive-semidefinite
matrix. A downstream PSD projection changes the output and must be evaluated
as a separate postprocessing method. Distinguish coordinate-wise trimming from
multivariate outlier classification: marginal tail expansion alone cannot
establish whether an observation is a coherent event or contamination.

Also disclose the full estimator's equivariance limits. Sign reversal exchanges
the upper and lower spreads in the one-sided TER; its floor at one therefore
need not preserve the acceptance region. Fixed absolute epsilon can change
acceptance under positive rescaling, especially near its special branch.
Pearson's transformation properties under a fixed mask do not establish those
properties for FORCE's learned mask. A symmetric TER or relative tolerance
would define a new policy to evaluate, not an arithmetic repair.

## 2. Replace unsupported robustness and clean-target claims

### 2.1 Define the statistical target and its domain

Let P be the observed law on p coordinates. Fix the policy theta: lambda>0,
the numerical tolerance tau>0 (`epsilon` in code), whether TER is enabled, and
any TER cap>=1. Use the population quantile convention
`q_phi,j(P)=inf{x : P(X_j<=x)>=phi}`. For each coordinate define

```text
m_j = q_0.50,j(P),                   s_j = (q_0.75,j(P)−q_0.25,j(P))/1.349,
d_j = |m_j−q_0.01,j(P)|,
t_j = 1                              if TER is disabled or d_j<tau,
      max(1, |q_0.99,j(P)−m_j|/(d_j+tau)) otherwise,
```

then apply the declared cap to t_j, if any, and let

```text
b_theta,j(P) = [m_j−lambda*t_j*s_j, m_j+lambda*t_j*s_j],
A_theta,jk(P) = {X_j in b_theta,j(P), X_k in b_theta,k(P)},
T_theta,jk(P) = Corr_P(X_j,X_k | A_theta,jk(P)).
```

Require finite bounds, positive joint acceptance and positive finite conditional
variances. Undefined off-diagonal zeros and unit diagonals are API conventions,
not definitions of otherwise nonexistent population correlations. Ordered
population quantiles cannot cross; handling crossed P² estimates is a separate
finite-sample convention. A declared quantile convention resolves notation,
but does not prove convergence of P² or of empirical interpolated quantiles
at a nonunique quantile. State sufficient identification and regularity
conditions for the backend used in each theorem.

The order-dependent finite-sample P² estimator, this population functional,
and clean untrimmed Pearson `R_jk(F)` are different objects. The last exists
only when the clean second moments are finite and variances positive. For
`H=(1−eta)F+eta G`, where all terms exist, the decomposition

```text
r_hat_jk − R_jk(F)
  = [r_hat_jk − T_theta,jk(H)]
    + [T_theta,jk(H) − T_theta,jk(F)]
    + [T_theta,jk(F) − R_jk(F)]
```

separates estimation error, contamination-induced target change and clean
trimming bias. This is an algebraic identity, not proof the first term vanishes
or either bias is small. The delayed-window mode has the different target in
Section 4.

### 2.2 An explicit counterexample to the median/IQR-only argument

Theorem 1 omits TER from the robustness argument. The following population
example shows why that omission matters even with exact quantiles, before
considering P² approximation or stream ordering.

Let the bivariate clean distribution F have independent Uniform[−1,1]
coordinates and let `H_M = 0.98 F + 0.02 δ_(M,M)`, with M>1 tending to
infinity. This is a Huber
contamination model with continuous clean marginals and allowed atomic
contamination. For `phi ∈ {0.01,0.25,0.50,0.75}`,

```text
q_phi(H_M) = 2phi/0.98 − 1,   q_0.99(H_M) = M,
m = q_0.50 = 1/49,
IQR = 1/0.98,                m − q_0.01 = 1.
```

With the default lambda=3 and numerical epsilon=10^−10, the uncapped
half-width satisfies

```text
width/M → 3 / [1.349 × 0.98 × (1 + epsilon)] ≈ 2.269 > 1.
```

Eventually the bounds accept the entire clean support and the contaminating
point. Consequently, the trimmed correlation equals the mixture correlation,

```text
rho(H_M) = 0.02 M² / (1/3 + 0.02 M²) → 1,
```

although the clean correlation is zero and contamination is only 2%. This is
an algebraic counterexample to universal protection based solely on the median
and IQR. It is not a theorem establishing the exact finite-sample breakdown
point of P² FORCE. The contaminating law changes with M: this challenges
uniform protection, not consistency for the trimmed target of each fixed
observed law. Atomic contamination is permitted by the robustness model;
this example also does not meet Proposition 1's separate continuous
observed-marginal premise.

**Paper change:** replace the unconditional 25% end-to-end claim by a discussion
of component robustness and this counterexample. A bounded correlation cannot
diverge to infinity, so the covariance-norm breakdown definition cannot be
transferred verbatim. Define the intended failure criterion before asserting
a breakdown value: approach to ±1 from a nondegenerate reference, loss of
estimability, maximal bias, or a specified matrix failure criterion. Different
criteria need different arguments.

Specify the order of quantifiers: a population maximum-bias bound takes a
supremum over an explicit contaminating class G at fixed eta; finite-sample
replacement breakdown concerns replacements of a given dataset. For P²,
state whether observation order is random or adversarial. Pointwise quantile
consistency at each fixed H cannot supply a uniform result over contaminating
sequences G_N or orderings.

**Further proof work:** analyze exact-quantile uncapped, capped and TER-disabled
policies separately, under both explosion and implosion of scale and accepted
variances. Then analyze the additional effect of P² initialization and ordering.
A cap limits one expansion mechanism; it does not by itself prove 25%
breakdown, nonempty acceptance, or positive accepted variance. Rowwise and
cellwise contamination require distinct definitions and results. One satellite
dataset above 25% contamination cannot establish a universal breakdown point.

### 2.3 Correct the clean-correlation corollary

For `H=(1−eta)F+eta G`, with `0<=eta<1`, fix an acceptance region A with
`F(A)>0`. If `G(A)=0`, then

```text
[(1−eta)F + eta G]( · | A) = F( · | A).
```

The resulting correlation is `Corr_F(X_j,X_k | A)`, not generally
`Corr_F(X_j,X_k)`. Truncating a dependent clean distribution can change its
correlation. Moreover, bounds learned from a contaminated mixture need not
equal the bounds learned from F alone.

The paper's weaker phrase "negligible mass" requires a quantitative statement.
If `G(A)>0`, the accepted law is

```text
eta_A = eta*G(A) / [(1−eta)*F(A) + eta*G(A)],
H(. | A) = (1−eta_A)*F(. | A) + eta_A*G(. | A).
```

Even a small G(A) need not give a small retained fraction when F(A) is small.
A small eta_A gives no uniform bias bound tending to zero with eta_A without
control of retained moment magnitudes and variances. Derive such a bound under
stated conditions; do not replace nonzero contamination by exact clean equality.
Here A may be the fixed population region learned from H; conditioning on a
random sample-learned region needs its own argument.

**Paper change:** state consistency for the named trimmed functional.
Equality with clean untrimmed correlation needs an extra condition, such as
complete clean-support acceptance or a justified correlation-preservation
property. Replace the current general corollary with this conditional statement.
Report clean-target error as a practical performance measure, separate from
quantile approximation error and trimming bias.

## 3. State defensible batch consistency and rate results

Fix the observed law P, policy theta and pair (j,k). Write b for its four
rectangle endpoints, `b_0=b_theta,jk(P)` for the limiting endpoints, A_b for
joint acceptance, and take expectations under P:

```text
h(x,y) = (1, x, y, x², y², xy),
m(b)   = E[h(X_j,X_k) 1{(X_j,X_k) ∈ A_b}],
rho(b) = C(m(b)),
```

where C divides by the accepted mass, centers the moments and normalizes the
covariance. The implemented sample correlation is the same map applied to
empirical accepted moments, subject to the explicit undefined-pair convention.

A useful proposed result is

```text
rho_hat_N − rho(b_0) = O_p(N^−1/2 + a_N)
```

when `||b_hat_N−b_0|| = O_p(a_N)` for deterministic `a_N→0`, under sufficient
local conditions:

- iid observations, fixed policy parameters and a fixed pair;
- finite limiting endpoints, acceptance probability bounded away from zero,
  and both conditional variances bounded away from zero;
- locally bounded marginal densities at the endpoints, giving
  `P(A_b Δ A_b0) = O(||b−b_0||)`;
- a uniform empirical-process bound over nearby rectangles for the six
  weighted moment functions;
- a locally Lipschitz quantile-to-bound map, with a neighborhood that avoids
  a discontinuous switch at the special TER epsilon branch.

The empirical-process condition can be justified for a fixed compact
neighborhood of finite endpoints: the moment functions restricted to those
rectangles have bounded envelopes and rectangle classes have finite complexity.
For broader unbounded acceptance models, appropriate moment and empirical-
process assumptions must instead be supplied.

The derivation separates

```text
P_N[h 1_A(b_hat)] − P[h 1_A(b_0)]
  = (P_N−P)[h 1_A(b_hat)]
    + P[h (1_A(b_hat)−1_A(b_0))].
```

The first term is controlled uniformly by sampling error; the second by
boundary displacement and boundary mass. Local smoothness of C then transfers
the bound to correlation. This uniform argument permits quantiles and moments
to use the same observations; it does not incorrectly assume their independence.
When the bound map is locally Lipschitz, a quantile error rate transfers to a_N.
Positive limiting mass and variances imply the estimate is defined with
probability tending to one. The rate concerns that event, not guaranteed
estimability of every finite sample. The argument uses ideal arithmetic and
does not cover terminating numerical failures.

This is a proof outline to complete and review, not a universal P² guarantee
or a result uniform over contaminating laws. State backend-specific quantile
conditions, such as uniquely identified quantiles and suitable local densities
for a proposed exact-quantile rate. P² requires its own argument; empirical
accuracy plots are not a substitute.

**Paper change:** consistency of quantiles alone gives no root-N rate. A
root-N bound needs an appropriate rate for the estimated-bound contribution.
Even root-N quantile errors can affect the leading asymptotic distribution;
they cannot simply be ignored. Asymptotic normality additionally needs the
joint limit and appropriate differentiability; TER maxima/caps and branching
boundaries can require separate treatment. A theorem for fixed p also does
not establish simultaneous accuracy as p grows or accepted counts shrink.

**Acceptance:** provide a complete assumption-to-conclusion proof for the
chosen exact or approximate backend, or retain a clearly conditional statement
and empirical evidence. Do not replace missing P² rate theory with a universal
“full accuracy after 100 observations” or relative-error promise near zero
quantiles.

## 4. Add an online target theorem and an honest validation design

### 4.1 Derivation for the implemented fixed-window schedule

Let iid observations from P be partitioned into independent blocks Z_t of
fixed size B. Fix the pair and policy. Let A_B(Z) be its acceptance rectangle
produced by the actual calibration policy, assumed to return finite defined
bounds almost surely. Z includes the observation order used by P².
The contribution from estimation block t is

```text
U_t = sum_{i=1}^B h(X_ti) 1{X_ti ∈ A_B(Z_(t−1))}.
```

Define `g_B(x)=P_Z{x ∈ A_B(Z)}` for an independent calibration block Z. Then

```text
E[U_t]/B = integral h(x) g_B(x) dP_jk(x),
dP_B,jk(x) = g_B(x) dP_jk(x) / integral g_B dP_jk.
```

Each U_t depends on `(Z_(t−1), Z_t)`. Thus contributions are stationary and
1-dependent under iid sampling: nonadjacent contributions are independent,
while neighbors share a block. With finite absolute selected moments,
positive accepted mass and positive selected variances, a law of large
numbers follows by splitting contributions into even and odd subsequences.
Applying C gives the target `Corr_(P_B,jk)(X_j,X_k)`. Removing the initial
calibration block and including at most B rows of a partial final window do
not change this limit under the stated integrability conditions. This is an
ideal-arithmetic limit; finite-precision rank and count limits remain separate
implementation restrictions.

State integrability explicitly as `integral ||h(x)|| g_B(x) dP_jk(x)<infinity`.
Finite random bounds in each window do not by themselves establish this
integrability over calibration blocks, particularly under heavy tails.

This gives a direct proof route for the actual schedule. It does not require
the fixed-B quantile error to vanish, and it does not identify P_B,jk with the
deterministic population-bound target of Proposition 1. Exact quantiles in
finite calibration windows still give random acceptance regions. Crossed-
quantile handling is part of A_B and therefore part of the target.

**Further inference work:** define `V_t=U_t/B`, `mu=E[V_t]` and
`Gamma_l=Cov(V_t,V_(t+l))`. With finite second moments of V_t, the
1-dependent central limit argument and the delta method give the proposed
block-scale limit, for T complete estimation blocks,

```text
sqrt(T) * (r_hat_T − C(mu))  =>  Normal(0, v),
v = grad C(mu)^T * (Gamma_0 + Gamma_1 + Gamma_1^T) * grad C(mu).
```

Under `sqrt(N)` scaling, with `N≈BT`, the asymptotic variance is `B*v`.
Positive accepted mass and variances are still required; ordinary
nondegenerate intervals also require v>0 and a consistent variance estimate.
Treating accepted rows as iid omits adjacent-block dependence. Complete the
proof and variance-estimation method before claiming interval coverage.
This is a fixed-pair result at a predetermined evaluation horizon. Simultaneous
coverage across pairs, repeated monitoring and data-dependent stopping require
separate arguments; they do not follow from the pointwise central limit result.
Under serial dependence or drift, replace the iid argument with an explicit
process model. Cumulative historical acceptance does not automatically estimate
a current-regime target. Positive per-block probability of terminating
calibration failure also prevents unconditional indefinite operation under
the iid block model; do not condition this away.

### 4.2 Separate simulation precision from target bias

The current [target simulator](src/force/diagnostics.py) pools accepted moments
within each independent replicate, then averages the replicate correlations
only if every requested replicate succeeds. Define S as successful calibration
and estimability of every required pair in a replicate, and `s=P(S)`.
The code uses a matrix-wide event; conditioning on success for an individual
pair would define a different quantity. With W calibration windows, E evaluation
rows per window and R independent replicates, the familiar limit

```text
mean_r_R → E[r_hat_(W,E)] as R → infinity,
```

requires s=1. At fixed W,E, if s<1 the current aggregate is available with
probability `s^R`, which tends to zero. Its withheld output cannot be assigned
that unconditional limit. Whenever it is displayed under s<1, its distribution
is conditional on all R replicates succeeding.

This is relevant even for nonsingular Gaussian data. Conditional on successful
finite calibration rectangles with pair-acceptance probabilities alpha_l<1,

```text
P(no accepted evaluation rows for the pair | Z_1,...,Z_W)
  = product_(l=1)^W (1−alpha_l)^E > 0.
```

Thus positive population acceptance does not guarantee finite-replicate
estimability. Observing only successful replicates does not establish s=1.
For `0<s<1`, meaningful finite-study objects include s and
`E[r_hat_(W,E) | S]`, accompanied by conditional error measures and their
Monte Carlo uncertainty. Add clearly named conditional summaries to research
reports while retaining the current all-success aggregate's failure convention.
Never convert undefined output to zero scientific error. The uncertainty of
conditional summaries must reflect the number of usable replicates; zero
observed failures is not evidence of zero failure probability.

Even when s=1, the finite-replicate expectation need not equal the stationary
target. Finite W leaves a random mixture of calibration regions even if E
tends to infinity; finite E adds evaluation noise and nonlinear bias.
Increasing R alone addresses neither bias. Under almost-sure calibration
success and the selected-moment assumptions, increasing W can give pooled
moment consistency with any fixed E>=1. Increasing E alone at fixed W does
not average away calibration randomness. When reporting success-conditioned
means, selection is an additional source of discrepancy from the target.

**Required validation:** retain the independent finite-stream acceptance ledger
and same-schedule exact-quantile comparator. Add a two-dimensional convergence
study increasing W and E separately, with independent replicates and explicit
failure counts. Increasing both axes is a diagnostic design, not a claim that
both must diverge for consistency. For a stationary-target reference, pool raw
accepted moment contributions over many independent calibration/evaluation
cycles and apply C once; quantify uncertainty at the independent cycle level, with bias
checks as cycle count increases. This is a Monte Carlo reference, not exact
truth: propagate its uncertainty into comparisons, including covariance when
draws are shared, or demonstrate that it is negligible at the declared
precision. Averaging individual window correlations would target a different
quantity. Use the deployed dependent schedule separately to assess its sampling
variance. The existing small smoke study establishes execution and pooling only.

## 5. Improve experiments and computational claims

### 5.1 Studies and comparison targets

| Study | Design and required reporting |
| --- | --- |
| Quantile approximation | Compare P² with independently computed exact quantiles on the same ordered samples. Report value error, distance of phi to the empirical rank interval `[F_N(q−), F_N(q)]`, crossings, bound displacement and acceptance disagreement. Include ties and extreme finite units. |
| Functional approximation | Match tails, epsilon, TER cap, lambda, pair centering and schedule. Use the same declared pair universe and disclose incomplete coverage. TP-Exact without TER and uncapped FORCE differ in both policy and quantile backend; their difference is not pure P² approximation error. |
| Robustness | Vary fraction and magnitude separately; include one-sided replacement, symmetric heavy tails, clustered rows and cellwise corruption. Specify replacement versus addition, row versus cell units, fixed-count versus Bernoulli assignments, and realized counts. Pair random, sorted, reversed and early/late-contamination orders on identical samples. These are stress scenarios, not a proof of breakdown. |
| Transformation sensitivity | Reflect coordinates and change their units, transform correlations back, then compare quantiles, bounds, masks and outputs. Separate numeric fixed-mask checks from policy effects of one-sided TER and absolute epsilon. |
| Sampling uncertainty | Distinguish new-data replicates, random-order replicates conditional on one dataset, and fixed-order timing repetitions. Order intervals do not quantify new-dataset uncertainty. Record all requested outcomes; success-conditioned summaries must say so. |
| Reference choice | Separate known population truth from proxy disagreement. A FastMCD/Spearman reference can structurally favor that method, and a low-volatility reference targets a selected regime. Agreement with either is not evidence of clean Pearson recovery. Report sensitivity to defensible alternatives and preserve reference validity. |
| Runtime and memory | Warm numerical paths; use matched interleaved runs and report failed/unwarmed exclusions. Measure source storage, workspace, retained state and output separately. Archive software versions, thread settings, data/reference/order identities and raw results. |

For population claims using an estimated reference, account for its estimation
and its dependence on the evaluated sample. Independent or held-out reference
data can reduce reuse but do not make a proxy equal the desired population
quantity. Randomly permuting financial or other dependent streams measures
conditional order sensitivity; it destroys the temporal process. Preserve
chronology and use independent process replicates or a justified dependence-aware
design for online and drift claims.

### 5.2 Rules for confirmatory comparisons

Plan aims, generating mechanisms, targets, methods and performance measures
before a confirmatory run. Report Monte Carlo uncertainty and missing outcomes
as part of performance, following the principles in
[Morris, White and Crowther (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6492164/).
For FORCE, the following decisions must be recorded:

1. **Freeze the target and decision rule.** Name the primary pair/matrix target,
   metric, contrasts and scenario weights. Distinguish mean replicate RMSE
   from the square root of mean replicate squared error; nonlinear aggregation
   changes the quantity being estimated. Predeclare a minimum useful gain,
   acceptable loss on clean conditions, maximum failure/coverage degradation,
   and time/memory budgets, with numeric values justified for the intended use.
2. **Separate development from evaluation.** Choose lambda, TER caps, sketch
   settings, window schedules and failure policies on development datasets/seeds
   or chronological development periods. Freeze them before new evaluation
   seeds or held-out periods. Give competitors comparable tuning information
   and budgets. The best setting selected on an evaluation grid is an oracle
   sensitivity result, not prospective performance.
3. **Use the correct replication unit.** For sampling claims, independently
   generated streams/datasets are the units; orders within one dataset are
   nested conditional replicates. Coordinate pairs, timepoints and overlapping
   windows are not extra independent replicates. Use paired method differences
   on shared datasets and preserve their dependence in intervals/resampling.
   Independent cycles estimate the target-reference uncertainty in Section 4;
   they do not replace the deployed process's variance calculation.
4. **Make estimability part of the result.** Define the eligible pair universe
   from the scientific target/reference independently of method outputs.
   Full-universe accuracy requires complete coverage. Partial comparisons use
   a common identified subset and remain conditional on it. Report failure and
   undefined probabilities, counts and uncertainty alongside conditional error.
   Paired comparisons restricted to joint method success must name that event;
   methods' separate success-conditioned errors can concern different samples.
   Do not retry until success or invent a scientific error for a missing estimate.
5. **Justify inference and repetition counts.** The current paper protocol's
   `1.96 × SE` intervals and unadjusted paired timing t-tests are existing
   exploratory conventions. For new claims, justify interval coverage for the
   chosen unit and distribution, predeclare confirmatory comparison families
   and multiplicity handling, and choose replication counts by desired Monte
   Carlo precision or power for the stated effect. Use an interval that allows
   positive failure risk after zero observed failures. Non-significance does
   not establish equality; an equivalence claim needs a prespecified margin
   and suitable interval/test. Archive any prespecified stopping rule.
6. **Retain negative results.** Publish configurations, data/reference/seed
   identities, tuning/evaluation splits, all requested outcomes and the decision
   against the frozen criteria. Require ablations for a claimed mechanism.
   "No demonstrated improvement" is a valid conclusion. Keep the aligned
   baseline available regardless of an enhancement's result.

### 5.3 Computational accounting

Correct the abstract's broad O(p) memory claim: O(p) describes quantile state,
while dense pair state and output require O(p²). Table 2 and Theorem 3 already
give FORCE O(p²) space; preserve those correct entries. Replay uses O(p²+Cp)
auxiliary memory for chunk size C, plus any exact-prefix buffer; the array
convenience API may allocate O(Np). A replayable two-pass algorithm needs a
source it can revisit. The causal online estimator has a different statistical
target.

For fixed numbers of requested quantiles, the current exact backend uses
partition/selection, not a full sort. State its implementation-specific cost
and O(Np) observation storage, accounting separately for O(p²) dense output
and any pair state. Correct Table 2's full-sort/rank time accounting to
feature-wise ordering plus correlation work, O(pN log N + Np²), rather than
assigning a log N factor to every pair; selection-based methods need their
own accounting. Revise its breakdown entries only under an explicit failure
definition and justified assumptions, as in Section 2. Preserve O(Np²) FORCE
time for a full matrix, and avoid universal constant-factor speed or “only
viable method” claims.

## 6. Delivery order and boundaries

1. **P0 — Specification and errata.** Correct equations, pseudocode, degeneracy,
   quantile conventions, memory accounting and target names. Replace the
   unconditional breakdown and clean-target claims. Deliver a traceable
   paper-to-code table and the analytic counterexample above.
2. **P1 — Conditional theory.** Complete the batch perturbation proof and the
   fixed-window target argument with explicit assumptions. Separate rate and
   inference proposals from established consistency statements. Derive the
   block-scale variance and address both numerical failure and finite-sample
   estimability. If stronger results remain unproved, keep them conditional.
3. **P1 — Evidence.** First explore the matched approximation, contamination
   and W/E studies. Then freeze Section 5.2's design and run a separate
   confirmatory evaluation, retaining failed and undefined cases. Publish data
   manifests and commands; unavailable historical identifiers remain disclosed.
   New empirical claims require that evidence; its absence does not reopen
   completed implementation alignment.
4. **P2 — Optional extensions.** Study strict crossing rejection, online
   quarantine, alternative quantile sketches, growing calibration windows,
   symmetric TER, relative numerical tolerances, PSD postprocessing,
   exponential weighting or mini-batch moment merging as
   individually named methods. Each needs its own acceptance rule, target,
   initialization, failure policy and validation. A flag or alternate backend
   is not an improvement without the predeclared evidence. P² states are not
   assumed mergeable; accepted moments cannot be merged by averaging correlations.

Capped TER, TER-disabled trimming (`TER=1`) and exact-prefix readouts are already
available options. Disabling TER fixes the multiplier, not the learned scale or
absolute width. Their further calibration and comparison are research tasks, not
missing implementation work.

Implementation validation includes equation/oracle checks, replay and chunk
invariance, online schedule tests, validity-aware scores and bounded retained
state. Quantile diagnostics use independent references, Winsorization conventions
are explicit, and simulation reports identify finite-W/E estimands. Aggregation
preserves method parameters and rejects mixed conventions. See
[VERIFICATION.md](VERIFICATION.md#validation) for the measured checks and their
scope; engineering tests do not prove the proposed statistical guarantees.

[MAD-FORCE](https://doi.org/10.1109/ACCESS.2026.3704294) is a separate algorithm
and research line. Changing FORCE's scale estimator or adding drift handling
is outside baseline alignment. The paper should describe the estimator
actually implemented and present stronger results only after their assumptions
and evidence have been established.
