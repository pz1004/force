"""Offline algorithm diagnostics with independent mathematical references.

These routines deliberately retain samples and masks. They are validation tools,
not bounded-memory implementations or alternative production estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from numbers import Integral
from typing import Any, Callable, Iterable
import warnings

import numpy as np

from .core import ForceEstimator, _compute_bounds_from_quantiles
from ._evaluation import correlation_score


PROBABILITIES = np.array([0.01, 0.25, 0.50, 0.75, 0.99])


@dataclass(frozen=True)
class BoundPolicy:
    name: str
    use_ter: bool = True
    ter_max: float | None = None
    lambda_scale: float = 3.0
    epsilon: float = 1e-10

    def parameters(self) -> dict[str, Any]:
        return {
            "use_ter": self.use_ter,
            "ter_max": self.ter_max,
            "lambda_scale": self.lambda_scale,
            "epsilon": self.epsilon,
        }


POLICIES = {
    "uncapped": BoundPolicy("uncapped"),
    "capped": BoundPolicy("capped", ter_max=3.0),
    "fixed": BoundPolicy("fixed", use_ter=False),
}


def reference_bounds(
    quantiles: np.ndarray, policy: BoundPolicy
) -> tuple[np.ndarray, np.ndarray]:
    """Independent equations 6–11, including the documented degenerate rule."""
    values = np.asarray(quantiles, dtype=float)
    lower, upper = [], []
    # Decimal is independent of the production binary-exponent implementation.
    # It also represents tail spans and capped ratios beyond float64's range.
    with localcontext() as context:
        context.prec = 80
        eps = Decimal.from_float(float(policy.epsilon))
        lam = Decimal.from_float(float(policy.lambda_scale))
        calibration = Decimal.from_float(1.349)
        for column in values.T:
            q01, q25, q50, q75, q99 = map(Decimal.from_float, map(float, column))
            scale = max(Decimal(0), (q75 - q25) / calibration)
            multiplier = Decimal(1)
            if policy.use_ter:
                distance = abs(q50 - q01)
                if distance >= eps:
                    multiplier = max(Decimal(1), abs(q99 - q50) / (distance + eps))
                if policy.ter_max is not None:
                    multiplier = min(
                        multiplier, Decimal.from_float(float(policy.ter_max))
                    )
            width = lam * multiplier * scale
            lower.append(float(q50 - width))
            upper.append(float(q50 + width))
    lower, upper = np.asarray(lower), np.asarray(upper)
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise FloatingPointError("Independent reference bounds exceed float64 range.")
    return lower, upper


def _reference_probabilities(probabilities: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(probabilities), dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("Probabilities must be a nonempty finite sequence in [0, 1].")
    return values


def reference_quantiles(
    X: np.ndarray, probabilities: Iterable[float] = PROBABILITIES
) -> np.ndarray:
    """Independent sorted linear quantiles with Decimal interpolation.

    Return a (probabilities, features) matrix. Probability iterables are consumed
    once; the default is FORCE's five quantiles. This oracle does not call the
    production interpolation helper.
    """
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Reference quantiles require a nonempty finite matrix.")
    probabilities = _reference_probabilities(probabilities)
    ordered = np.sort(values, axis=0)
    result = np.empty((len(probabilities), values.shape[1]))
    with localcontext() as context:
        context.prec = 80
        for index, probability in enumerate(probabilities):
            rank = float(probability) * (len(values) - 1)
            lower, upper = int(np.floor(rank)), int(np.ceil(rank))
            fraction = Decimal.from_float(rank - lower)
            for column in range(values.shape[1]):
                a = Decimal.from_float(float(ordered[lower, column]))
                b = Decimal.from_float(float(ordered[upper, column]))
                result[index, column] = float((1 - fraction) * a + fraction * b)
    if not np.isfinite(result).all():
        raise FloatingPointError("Independent quantiles exceeded float64 range.")
    return result


def quantile_rank_interval_error(
    X: np.ndarray,
    estimates: np.ndarray,
    probabilities: Iterable[float] = PROBABILITIES,
) -> np.ndarray:
    """Distance from each probability to [F_n(q-), F_n(q)], allowing ties.

    ``X`` is (observations, features), and ``estimates`` and the result are
    (probabilities, features). A target inside a tied value's empirical rank
    interval has zero error. This measures empirical rank agreement, not
    population-quantile uncertainty or exact linear-quantile value error.
    """
    values = np.asarray(X, dtype=float)
    estimates = np.asarray(estimates, dtype=float)
    probabilities = _reference_probabilities(probabilities)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Rank diagnostics require a nonempty finite matrix.")
    if (
        estimates.shape != (len(probabilities), values.shape[1])
        or not np.isfinite(estimates).all()
    ):
        raise ValueError(
            "Quantile estimates must be finite with shape (probabilities, features)."
        )
    errors = np.empty_like(estimates)
    for index, (probability, quantiles) in enumerate(zip(probabilities, estimates)):
        left = np.mean(values < quantiles, axis=0)
        right = np.mean(values <= quantiles, axis=0)
        errors[index] = np.maximum(
            np.maximum(left - probability, probability - right), 0.0
        )
    return errors


def reference_pairwise(X: np.ndarray, accepted: np.ndarray, *, return_status=False):
    """Pearson over each jointly accepted subset; no production moment helpers."""
    p = X.shape[1]
    correlation = np.eye(p)
    counts = np.zeros((p, p), dtype=np.int64)
    codes = np.ones((p, p), dtype=np.int8)
    for j in range(p):
        for k in range(j, p):
            mask = accepted[:, j] & accepted[:, k]
            counts[j, k] = counts[k, j] = int(mask.sum())
            if counts[j, k] < 2:
                continue
            # Independent offline arithmetic: subtract an observed origin
            # before scaling, so large translations do not erase differences.
            # Keep extended precision through both centering and dot products.
            x = np.asarray(X[mask, j], dtype=np.longdouble)
            y = np.asarray(X[mask, k], dtype=np.longdouble)
            with np.errstate(over="ignore", invalid="ignore"):
                dx, dy = x - x[0], y - y[0]
            if not np.all(np.isfinite(dx)):
                dx = x / 2 - x[0] / 2
            if not np.all(np.isfinite(dy)):
                dy = y / 2 - y[0] / 2
            sx, sy = np.max(np.abs(dx)), np.max(np.abs(dy))
            x = dx / (sx if sx > 0 else 1)
            y = dy / (sy if sy > 0 else 1)
            x -= x.mean()
            y -= y.mean()
            nx, ny = np.sqrt(np.dot(x, x)), np.sqrt(np.dot(y, y))
            if nx and ny:
                codes[j, k] = codes[k, j] = 0
                value = float(np.dot(x / nx, y / ny))
                if not np.isfinite(value) or abs(value) > 1 + 1e-12:
                    raise FloatingPointError(
                        "Independent reference produced an invalid correlation."
                    )
                if j != k:
                    correlation[j, k] = correlation[k, j] = np.clip(value, -1.0, 1.0)
            else:
                codes[j, k] = codes[k, j] = 2
    return (correlation, counts, codes) if return_status else (correlation, counts)


def off_diagonal_rmse(left: np.ndarray, right: np.ndarray) -> float | None:
    """Score supplied defined matrices; estimator diagnostics require explicit masks."""
    return correlation_score(
        left, right, estimate_valid=np.isfinite(left), require_complete=True
    )["rmse"]


def _correlation_diagnostics(
    production,
    production_codes,
    oracle,
    oracle_codes,
    exact,
    exact_codes,
    clean_reference,
):
    valid = production_codes == 0
    exact_valid = exact_codes == 0
    oracle_valid = oracle_codes == 0
    upper = np.triu(np.ones(valid.shape, dtype=bool), k=1)
    compared = upper & valid & oracle_valid
    approximation = correlation_score(
        production, exact, estimate_valid=valid, reference_valid=exact_valid
    )
    clean = (
        correlation_score(
            production, clean_reference, estimate_valid=valid, require_complete=True
        )
        if clean_reference is not None
        else None
    )
    exact_clean = (
        correlation_score(
            exact, clean_reference, estimate_valid=exact_valid, require_complete=True
        )
        if clean_reference is not None
        else None
    )
    n_valid = int((upper & valid).sum())
    estimate_status = (
        "complete"
        if n_valid == int(upper.sum())
        else "partial" if n_valid else "undefined"
    )
    compatibility_eigenvalue = float(np.linalg.eigvalsh(production).min())
    return {
        "execution_status": "completed",
        "estimate_status": estimate_status,
        "production_pair_status_codes": production_codes.tolist(),
        "exact_pair_status_codes": exact_codes.tolist(),
        "oracle_pair_status_codes": oracle_codes.tolist(),
        "production_defined_pairs": valid.tolist(),
        "exact_defined_pairs": exact_valid.tolist(),
        "production_validity_matches_oracle": bool(np.array_equal(valid, oracle_valid)),
        "production_oracle_max_error": (
            float(np.max(np.abs(production[compared] - oracle[compared])))
            if np.any(compared)
            else None
        ),
        "compatibility_output_oracle_max_error": float(
            np.max(np.abs(production - oracle))
        ),
        "approximation_rmse": approximation["rmse"],
        "approximation_score": approximation,
        "approximation_interpretation": "conditional_on_jointly_defined_pairs",
        "clean_target_rmse": clean["rmse"] if clean is not None else None,
        "clean_target_score": clean,
        "exact_clean_target_rmse": (
            exact_clean["rmse"] if exact_clean is not None else None
        ),
        "exact_clean_target_score": exact_clean,
        "minimum_eigenvalue": compatibility_eigenvalue if valid.all() else None,
        "compatibility_minimum_eigenvalue": compatibility_eigenvalue,
        "eigenvalue_interpretation": (
            "all_coefficients_defined_no_PSD_guarantee"
            if valid.all()
            else "zero_filled_compatibility_matrix_only"
        ),
    }


def _quantile_metrics(
    X: np.ndarray, approximate: np.ndarray, exact: np.ndarray
) -> dict[str, Any]:
    return {
        "quantile_value_absolute_error": np.abs(approximate - exact).tolist(),
        "quantile_rank_interval_error": quantile_rank_interval_error(
            X, approximate
        ).tolist(),
        "quantile_crossing_count": int(np.sum(np.diff(approximate, axis=0) < 0.0)),
        "quantile_crossings": (np.diff(approximate, axis=0) < 0.0).tolist(),
        "crossed_iqr": (approximate[3] < approximate[1]).tolist(),
        "zero_iqr": (approximate[3] == approximate[1]).tolist(),
    }


def _mask_metrics(
    approximate: np.ndarray, exact: np.ndarray, contaminated: np.ndarray | None
) -> dict[str, Any]:
    p = approximate.shape[1]
    disagreement = np.zeros((p, p))
    contaminated_counts = np.zeros((p, p), dtype=np.int64)
    exact_contaminated_counts = np.zeros((p, p), dtype=np.int64)
    fractions = np.zeros((p, p))
    exact_fractions = np.zeros((p, p))
    for j in range(p):
        for k in range(j, p):
            a = approximate[:, j] & approximate[:, k]
            b = exact[:, j] & exact[:, k]
            disagreement[j, k] = disagreement[k, j] = float(np.mean(a != b))
            if contaminated is not None:
                bad = int(np.sum(a & contaminated))
                contaminated_counts[j, k] = contaminated_counts[k, j] = bad
                fractions[j, k] = fractions[k, j] = bad / max(int(a.sum()), 1)
                exact_bad = int(np.sum(b & contaminated))
                exact_contaminated_counts[j, k] = exact_contaminated_counts[k, j] = (
                    exact_bad
                )
                exact_fractions[j, k] = exact_fractions[k, j] = exact_bad / max(
                    int(b.sum()), 1
                )
    return {
        "coordinate_acceptance_disagreement": float(np.mean(approximate != exact)),
        "pair_acceptance_disagreement": disagreement.tolist(),
        "retained_contaminated_pair_counts": (
            contaminated_counts.tolist() if contaminated is not None else None
        ),
        "retained_contamination_fraction": (
            fractions.tolist() if contaminated is not None else None
        ),
        "exact_retained_contaminated_pair_counts": (
            exact_contaminated_counts.tolist() if contaminated is not None else None
        ),
        "exact_retained_contamination_fraction": (
            exact_fractions.tolist() if contaminated is not None else None
        ),
    }


def diagnose_batch(
    X: np.ndarray,
    *,
    policy: BoundPolicy = POLICIES["uncapped"],
    clean_reference: np.ndarray | None = None,
    contaminated_rows: np.ndarray | None = None,
) -> dict[str, Any]:
    """Separate quantile approximation from downstream arithmetic and target error."""
    X = np.asarray(X, dtype=float)
    estimator = ForceEstimator(exact_cutover=5, **policy.parameters())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        production = estimator.fit(X)
    approximate = np.asarray(estimator.quantiles)
    exact = reference_quantiles(X)
    lower, upper = reference_bounds(approximate, policy)
    exact_lower, exact_upper = reference_bounds(exact, policy)
    accepted = (X >= lower) & (X <= upper)
    exact_accepted = (X >= exact_lower) & (X <= exact_upper)
    oracle, counts, oracle_codes = reference_pairwise(X, accepted, return_status=True)
    exact_correlation, exact_counts, exact_codes = reference_pairwise(
        X, exact_accepted, return_status=True
    )
    production_lower, production_upper = estimator.thresholds
    production_counts = getattr(estimator, "accepted_counts_", None)
    return {
        "status": "completed",
        "estimator": type(estimator).__name__,
        "quantile_backend": "production_P2",
        "bound_policy": policy.name,
        "policy_parameters": policy.parameters(),
        "execution_schedule": "two_pass_final_bounds",
        "comparison_target": "same_sample_exact_quantile_trimmed_Pearson",
        "clean_reference_target": (
            "known_clean_population_Pearson" if clean_reference is not None else None
        ),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "quantiles": approximate.tolist(),
        "exact_quantiles": exact.tolist(),
        **_quantile_metrics(X, approximate, exact),
        "lower_bound_error": (lower - exact_lower).tolist(),
        "upper_bound_error": (upper - exact_upper).tolist(),
        "production_bound_oracle_max_error": float(
            max(
                np.max(np.abs(lower - production_lower)),
                np.max(np.abs(upper - production_upper)),
            )
        ),
        **_mask_metrics(accepted, exact_accepted, contaminated_rows),
        "accepted_pair_counts": counts.tolist(),
        "exact_accepted_pair_counts": exact_counts.tolist(),
        "production_count_oracle_max_error": (
            int(np.max(np.abs(production_counts - counts)))
            if production_counts is not None
            else None
        ),
        "production_pair_status": (
            np.asarray(estimator.pair_status_).tolist()
            if hasattr(estimator, "pair_status_")
            else None
        ),
        "production_correlation": production.tolist(),
        "exact_correlation": exact_correlation.tolist(),
        **_correlation_diagnostics(
            production,
            estimator.pair_status_codes_,
            oracle,
            oracle_codes,
            exact_correlation,
            exact_codes,
            clean_reference,
        ),
        "warnings": [str(w.message) for w in captured],
    }


def diagnose_online(
    X: np.ndarray,
    *,
    window_size: int,
    policy: BoundPolicy = POLICIES["uncapped"],
    clean_reference: np.ndarray | None = None,
    contaminated_rows: np.ndarray | None = None,
    chunk_size: int = 37,
) -> dict[str, Any]:
    """Offline reference with the production previous-window acceptance schedule.

    Exact reference quantiles use the same previous windows and calibration
    exclusion. Comparing this result to final-bound batch is a different-target
    comparison and is deliberately not reported as approximation error here.
    """
    from .streaming import StreamingForceEstimator

    X = np.asarray(X, dtype=float)
    if len(X) <= window_size:
        raise ValueError("Online diagnostics need at least one post-calibration row.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator = StreamingForceEstimator(
            window_size=window_size, **policy.parameters()
        )
        snapshots = {}
        for offset in range(0, len(X), chunk_size):
            stop = min(offset + chunk_size, len(X))
            while offset < stop:
                end = min(stop, (offset // window_size + 1) * window_size)
                estimator.update_batch(X[offset:end])
                if end % window_size == 0:
                    snapshots[end] = (estimator.quantiles_, estimator.thresholds_)
                offset = end
        production = estimator.finalize()
        accepted = np.zeros_like(X, dtype=bool)
        exact_accepted = np.zeros_like(X, dtype=bool)
        windows = []
        for start in range(window_size, len(X), window_size):
            previous = X[start - window_size : start]
            stop = min(start + window_size, len(X))
            q, (production_lower, production_upper) = snapshots[start]
            q_exact = reference_quantiles(previous)
            lower, upper = reference_bounds(q, policy)
            elo, ehi = reference_bounds(q_exact, policy)
            accepted[start:stop] = (X[start:stop] >= lower) & (X[start:stop] <= upper)
            exact_accepted[start:stop] = (X[start:stop] >= elo) & (X[start:stop] <= ehi)
            windows.append(
                {
                    "estimation_start": start,
                    "estimation_stop": stop,
                    "quantiles": q.tolist(),
                    "exact_quantiles": q_exact.tolist(),
                    "production_lower_bound": production_lower.tolist(),
                    "production_upper_bound": production_upper.tolist(),
                    "exact_lower_bound": elo.tolist(),
                    "exact_upper_bound": ehi.tolist(),
                    "lower_bound_error": (lower - elo).tolist(),
                    "upper_bound_error": (upper - ehi).tolist(),
                    "production_bound_oracle_max_error": float(
                        max(
                            np.max(np.abs(lower - production_lower)),
                            np.max(np.abs(upper - production_upper)),
                        )
                    ),
                    **_quantile_metrics(previous, q, q_exact),
                }
            )
    oracle, counts, oracle_codes = reference_pairwise(X, accepted, return_status=True)
    exact_correlation, exact_counts, exact_codes = reference_pairwise(
        X, exact_accepted, return_status=True
    )
    return {
        "status": "completed",
        "estimator": type(estimator).__name__,
        "quantile_backend": "production_P2_previous_window",
        "bound_policy": policy.name,
        "policy_parameters": policy.parameters(),
        "execution_schedule": "previous_window_cumulative_moments",
        "finite_sample_target": "cumulative_Pearson_over_historical_joint_acceptance",
        "stationary_target": "iid_fixed_window_selection_weighted_distribution",
        "stationary_target_assumptions": [
            "iid_observations_and_fixed_logical_window_size",
            "calibration_and_bounds_defined_finite_almost_surely",
            "positive_acceptance_probability",
            "finite_selected_second_moments_and_positive_selected_variances",
            "no_terminating_processing_failure_with_positive_window_probability",
        ],
        "comparison_target": "same_schedule_exact_previous_window_quantiles",
        "clean_reference_target": (
            "known_clean_population_Pearson" if clean_reference is not None else None
        ),
        "window_size": window_size,
        "chunk_size": chunk_size,
        "calibration_rows": window_size,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "windows": windows,
        "production_bound_oracle_max_error": max(
            window["production_bound_oracle_max_error"] for window in windows
        ),
        **_mask_metrics(
            accepted[window_size:],
            exact_accepted[window_size:],
            contaminated_rows[window_size:] if contaminated_rows is not None else None,
        ),
        "accepted_pair_counts": counts.tolist(),
        "exact_accepted_pair_counts": exact_counts.tolist(),
        "production_count_oracle_max_error": int(
            np.max(np.abs(estimator.accepted_counts_ - counts))
        ),
        "production_pair_status": np.asarray(estimator.pair_status_).tolist(),
        "production_correlation": production.tolist(),
        "exact_correlation": exact_correlation.tolist(),
        **_correlation_diagnostics(
            production,
            estimator.pair_status_codes_,
            oracle,
            oracle_codes,
            exact_correlation,
            exact_codes,
            clean_reference,
        ),
        "warnings": [str(w.message) for w in captured],
    }


def simulate_online_target(
    sample: Callable[[np.random.Generator, int], np.ndarray],
    *,
    window_size: int,
    evaluation_size: int,
    windows_per_replicate: int,
    n_replicates: int,
    seed: int = 0,
    policy: BoundPolicy = POLICIES["uncapped"],
    quantile_backend: str = "p2",
) -> dict[str, Any]:
    """Offline Monte Carlo check of the fixed-window selection-weighted target.

    ``sample(rng, n)`` must return n iid rows from one fixed distribution. Each
    cycle draws independent calibration and evaluation blocks. Accepted rows
    are pooled *within* each independent replicate before computing Pearson;
    window correlations are never averaged. The reported mean and standard
    error summarize those independent replicate estimates. With W calibration
    blocks and E evaluation rows per block fixed, increasing the replicate
    count estimates E[r_hat(W, E)], not necessarily the stationary target.
    Both finite W and finite E can cause nonlinear bias: even unlimited
    evaluation rows leave a random mixture of only W calibration regions.
    Increasing the replicate count reduces Monte Carlo error around that
    finite-replicate expectation; it does not remove either bias. The standard
    error is not a stationary-target error bound or an interval for drifting
    streams.

    An unconditional E[r_hat(W, E)] additionally requires calibration success
    and a defined finite-replicate correlation almost surely. Positive
    population acceptance alone does not guarantee enough accepted rows or
    positive accepted variances in a finite replicate. Observing success in all
    requested replicates does not establish these almost-sure assumptions.

    This deliberately retains offline data. Any failed calibration or undefined
    replicate leaves the overall estimate unavailable; successes are retained
    as diagnostics but never silently substituted for all requested replicates.
    The statistical target also assumes almost-sure finite defined bounds,
    positive acceptance, finite selected moments and positive selected variances.
    """
    from ._quantiles import P2State

    parameters = {
        "window_size": window_size,
        "evaluation_size": evaluation_size,
        "windows_per_replicate": windows_per_replicate,
        "n_replicates": n_replicates,
    }
    for name, value in parameters.items():
        if (
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if window_size <= 5:
        raise ValueError("window_size must exceed the five-row P² bootstrap.")
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if quantile_backend not in {"p2", "exact"}:
        raise ValueError("quantile_backend must be 'p2' or 'exact'.")
    features = None

    def draw(rng, size):
        nonlocal features
        values = np.asarray(sample(rng, size), dtype=float)
        if values.ndim != 2 or values.shape[0] != size or values.shape[1] < 2:
            raise ValueError(
                "sample must return a matrix with the requested rows and >=2 columns."
            )
        if not np.isfinite(values).all():
            raise ValueError("sample must return finite observations.")
        if features is None:
            features = values.shape[1]
        elif values.shape[1] != features:
            raise ValueError("sample must preserve its feature count.")
        return values

    records, estimates = [], []
    failed_windows = 0
    attempted_windows = 0
    for replicate, child_seed in enumerate(
        np.random.SeedSequence(seed).spawn(n_replicates)
    ):
        rng = np.random.default_rng(child_seed)
        evaluations, masks = [], []
        record = {
            "replicate_id": replicate,
            "requested_calibration_windows": int(windows_per_replicate),
            "attempted_calibration_windows": 0,
            "failed_calibration_windows": 0,
            "status": "completed",
            "warning_count": 0,
            "quantile_crossing_count": 0,
            "message": None,
        }
        for _ in range(windows_per_replicate):
            calibration = draw(rng, window_size)
            attempted_windows += 1
            record["attempted_calibration_windows"] += 1
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    quantiles = (
                        P2State(features).update_batch(calibration).quantiles
                        if quantile_backend == "p2"
                        else reference_quantiles(calibration)
                    )
                    lower, upper = _compute_bounds_from_quantiles(
                        quantiles, **policy.parameters()
                    )[3:]
                except FloatingPointError as exc:
                    record["status"] = "calibration_failed"
                    record["message"] = str(exc)
                    record["failed_calibration_windows"] += 1
                    failed_windows += 1
                record["warning_count"] += len(captured)
            if record["status"] != "completed":
                break
            record["quantile_crossing_count"] += int(
                np.sum(np.diff(quantiles, axis=0) < 0)
            )
            evaluation = draw(rng, evaluation_size)
            evaluations.append(evaluation)
            masks.append((evaluation >= lower) & (evaluation <= upper))
        if record["status"] == "completed":
            correlation, counts, codes = reference_pairwise(
                np.concatenate(evaluations), np.concatenate(masks), return_status=True
            )
            record.update(
                correlation=correlation.tolist(),
                accepted_pair_counts=counts.tolist(),
                pair_status_codes=codes.tolist(),
            )
            if np.all(codes == 0):
                estimates.append(correlation)
            else:
                record["status"] = "undefined_correlation"
        records.append(record)
    complete = len(estimates) == n_replicates
    mean, standard_error = None, None
    if complete:
        stack = np.asarray(estimates)
        mean = np.mean(stack, axis=0).tolist()
        if n_replicates > 1:
            standard_error = (
                np.std(stack, axis=0, ddof=1) / np.sqrt(n_replicates)
            ).tolist()
    return {
        "status": "completed" if complete else "incomplete",
        "execution_schedule": "independent_calibration_and_evaluation_blocks",
        "comparison_target": "iid_fixed_window_selection_weighted_distribution",
        "reported_estimand": "expectation_of_finite_pooled_replicate_correlation",
        "reported_estimand_parameters": {
            "calibration_blocks_per_replicate": int(windows_per_replicate),
            "evaluation_rows_per_block": int(evaluation_size),
        },
        "reported_estimand_requires": [
            "calibration_success_almost_surely",
            "finite_replicate_correlation_defined_almost_surely",
        ],
        "observed_success_establishes_estimand_assumptions": False,
        "quantile_backend": quantile_backend,
        "policy_parameters": policy.parameters(),
        "bound_policy": policy.name,
        **{name: int(value) for name, value in parameters.items()},
        "seed": int(seed),
        "n_features": features,
        "requested_calibration_windows": int(n_replicates * windows_per_replicate),
        "attempted_calibration_windows": attempted_windows,
        "failed_calibration_windows": failed_windows,
        "usable_replicates": len(estimates),
        "replicate_mean_correlation": mean,
        "monte_carlo_standard_error": standard_error,
        "uncertainty_basis": "independent_replicates_standard_error_of_finite_replicate_mean",
        "possible_bias_sources": [
            "finite_calibration_block_count",
            "finite_evaluation_rows_per_block",
        ],
        "stationary_target_error_bound": None,
        "replicates": records,
    }


def stress_sample(
    *,
    n_samples: int,
    n_features: int,
    seed: int,
    fraction: float,
    magnitude: float,
    scenario: str,
    order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paired clean samples, with independent contamination magnitude and order.

    For cellwise corruption the returned row label means any contaminated cell;
    it must not be interpreted as a cellwise breakdown guarantee.
    """
    if not 0 <= fraction < 1 or magnitude <= 0:
        raise ValueError("Require fraction in [0,1) and positive magnitude.")
    rng = np.random.default_rng(seed)
    correlation = 0.6 ** np.abs(
        np.subtract.outer(np.arange(n_features), np.arange(n_features))
    )
    X = rng.multivariate_normal(np.zeros(n_features), correlation, n_samples)
    priorities = rng.random(n_samples)
    contaminated = np.zeros(n_samples, dtype=bool)
    contaminated[np.argsort(priorities)[: int(fraction * n_samples)]] = True
    if scenario == "one_sided_replacement":
        X[contaminated, 0] = magnitude
    elif scenario == "symmetric_heavy_tails":
        noise = rng.standard_cauchy(X.shape)
        X[contaminated] = magnitude * noise[contaminated]
    elif scenario == "clustered_rows":
        noise = rng.normal(scale=0.1, size=X.shape)
        X[contaminated] = magnitude + noise[contaminated]
    elif scenario == "cellwise":
        cells = rng.random(X.shape) < fraction
        signs = rng.choice([-1.0, 1.0], size=X.shape)
        X[cells] = (magnitude * signs)[cells]
        contaminated = np.any(cells, axis=1)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    if order == "random":
        permutation = np.random.default_rng(seed + 104729).permutation(n_samples)
    elif order == "sorted":
        permutation = np.argsort(X[:, 0], kind="stable")
    elif order == "reversed":
        permutation = np.argsort(-X[:, 0], kind="stable")
    elif order == "outliers_first":
        permutation = np.argsort(~contaminated, kind="stable")
    elif order == "outliers_last":
        permutation = np.argsort(contaminated, kind="stable")
    else:
        raise ValueError(f"Unknown order: {order}")
    return X[permutation], correlation, contaminated[permutation]
