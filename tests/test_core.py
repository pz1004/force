"""Equation-level verification of FORCE."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from force import ForceEstimator, P2Quantile, PearsonEstimator, p2_desired_positions
from force.data import generate_synthetic_data
from force.core import (
    P2_PROBABILITIES,
    _compute_trimmed_corr_numba,
    _compute_bounds_from_quantiles,
    _covariance_to_correlation,
    _p_square_kernel,
)


def _independent_trimmed_correlation(
    X: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    result = np.eye(X.shape[1])
    for left in range(X.shape[1]):
        for right in range(left + 1, X.shape[1]):
            accepted = (
                (X[:, left] >= lower[left])
                & (X[:, left] <= upper[left])
                & (X[:, right] >= lower[right])
                & (X[:, right] <= upper[right])
            )
            x = X[accepted, left]
            y = X[accepted, right]
            value = 0.0
            if x.size > 1 and np.var(x) > 0.0 and np.var(y) > 0.0:
                value = float(np.corrcoef(x, y)[0, 1])
            result[left, right] = result[right, left] = value
    return result


def _independent_p2(values: np.ndarray, probability: float) -> float:
    """Canonical recurrence implemented independently from production code."""
    heights = np.sort(np.asarray(values[:5], dtype=float))
    positions = np.arange(1.0, 6.0)
    desired = np.array(
        [
            1.0,
            1.0 + 2.0 * probability,
            1.0 + 4.0 * probability,
            3.0 + 2.0 * probability,
            5.0,
        ]
    )
    increments = np.array(
        [0.0, probability / 2.0, probability, (1.0 + probability) / 2.0, 1.0]
    )
    for value in values[5:]:
        if value < heights[0]:
            heights[0] = value
            cell = 0
        elif value >= heights[4]:
            heights[4] = value
            cell = 3
        else:
            cell = int(np.searchsorted(heights, value, side="right") - 1)
            cell = min(3, max(0, cell))
        positions[cell + 1 :] += 1.0
        desired += increments
        for marker in (1, 2, 3):
            discrepancy = desired[marker] - positions[marker]
            if discrepancy >= 1.0 and positions[marker + 1] - positions[marker] > 1:
                direction = 1
            elif (
                discrepancy <= -1.0
                and positions[marker - 1] - positions[marker] < -1
            ):
                direction = -1
            else:
                continue
            candidate = heights[marker] + direction / (
                positions[marker + 1] - positions[marker - 1]
            ) * (
                (positions[marker] - positions[marker - 1] + direction)
                * (heights[marker + 1] - heights[marker])
                / (positions[marker + 1] - positions[marker])
                + (positions[marker + 1] - positions[marker] - direction)
                * (heights[marker] - heights[marker - 1])
                / (positions[marker] - positions[marker - 1])
            )
            if heights[marker - 1] < candidate < heights[marker + 1]:
                heights[marker] = candidate
            else:
                adjacent = marker + direction
                heights[marker] += direction * (
                    heights[adjacent] - heights[marker]
                ) / (positions[adjacent] - positions[marker])
            positions[marker] += direction
    return float(heights[2])


def test_standard_p2_desired_positions() -> None:
    observed = p2_desired_positions(101, 0.25)
    assert np.allclose(observed, [1.0, 13.5, 26.0, 63.5, 101.0])


@pytest.mark.parametrize("probability", P2_PROBABILITIES)
def test_p2_incremental_matches_numba_kernel(probability: float) -> None:
    values = np.random.default_rng(7).lognormal(size=1000)
    tracker = P2Quantile(float(probability))
    for value in values:
        tracker.update(float(value))
    kernel = _p_square_kernel(
        values.astype(np.float64), np.array([probability], dtype=np.float64)
    )
    assert tracker.value == pytest.approx(kernel[0], rel=1e-13, abs=1e-13)
    assert np.all(np.diff(tracker.marker_heights) >= 0.0)
    assert np.all(np.diff(tracker.marker_positions) > 0.0)
    assert np.allclose(
        tracker.desired_positions,
        p2_desired_positions(values.size, float(probability)),
    )
    assert tracker.value == pytest.approx(
        _independent_p2(values, float(probability)),
        rel=1e-13,
        abs=1e-13,
    )


def test_p2_marker_invariants_hold_after_every_initialized_update() -> None:
    tracker = P2Quantile(0.25)
    for value in np.random.default_rng(17).standard_cauchy(size=1000):
        tracker.update(float(value))
        if tracker.count >= 5:
            assert np.all(np.diff(tracker.marker_heights) >= 0.0)
            assert np.all(np.diff(tracker.marker_positions) > 0.0)
            assert tracker.marker_positions[0] == 1.0
            assert tracker.marker_positions[-1] == tracker.count
            assert np.allclose(
                tracker.desired_positions,
                p2_desired_positions(tracker.count, 0.25),
            )


@pytest.mark.parametrize("probability", P2_PROBABILITIES)
def test_p2_quantile_rank_accuracy(probability: float) -> None:
    values = np.random.default_rng(11).normal(size=20_000)
    tracker = P2Quantile(float(probability))
    for value in values:
        tracker.update(float(value))
    empirical_rank = np.mean(values <= tracker.value)
    tolerance = 0.006 if probability in (0.01, 0.99) else 0.012
    assert abs(empirical_rank - probability) < tolerance


def test_exact_cutover_uses_linear_numpy_quantiles() -> None:
    X = np.random.default_rng(2).normal(size=(30, 3))
    estimator = ForceEstimator(exact_cutover=31)
    estimator.fit(X)
    assert np.allclose(
        estimator.quantiles,
        np.quantile(X, P2_PROBABILITIES, axis=0, method="linear"),
    )


def test_equation_6_through_11_bounds() -> None:
    quantiles = np.array(
        [
            [-4.0, -2.0],
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 2.0],
            [8.0, 2.0],
        ]
    )
    location, scale, ter, lower, upper = _compute_bounds_from_quantiles(
        quantiles,
        lambda_scale=3.0,
        use_ter=True,
        ter_max=None,
        epsilon=1e-10,
    )
    assert np.allclose(location, [0.0, 0.0])
    assert np.allclose(scale, [2.0 / 1.349, 3.0 / 1.349])
    assert ter[0] == pytest.approx(2.0, rel=1e-10)
    assert ter[1] == pytest.approx(1.0)
    assert np.allclose(lower, location - 3.0 * ter * scale)
    assert np.allclose(upper, location + 3.0 * ter * scale)


def test_degenerate_ter_warns_and_collapses_constant_bounds() -> None:
    quantiles = np.full((5, 2), 4.0)
    with pytest.warns(RuntimeWarning, match="TER denominator"):
        location, scale, ter, lower, upper = _compute_bounds_from_quantiles(
            quantiles,
            lambda_scale=3.0,
            use_ter=True,
            ter_max=None,
            epsilon=1e-10,
        )
    assert np.array_equal(location, [4.0, 4.0])
    assert np.array_equal(scale, [0.0, 0.0])
    assert np.array_equal(ter, [1.0, 1.0])
    assert np.array_equal(lower, upper)


def test_uncapped_ter_can_expand_while_iqr_stays_fixed() -> None:
    quantiles = np.array([[-2.0], [-1.0], [0.0], [1.0], [1.0e9]])
    _, scale, uncapped, _, upper_uncapped = _compute_bounds_from_quantiles(
        quantiles,
        lambda_scale=3.0,
        use_ter=True,
        ter_max=None,
        epsilon=1e-10,
    )
    _, capped_scale, capped, _, upper_capped = _compute_bounds_from_quantiles(
        quantiles,
        lambda_scale=3.0,
        use_ter=True,
        ter_max=3.0,
        epsilon=1e-10,
    )
    assert scale[0] == pytest.approx(2.0 / 1.349)
    assert capped_scale[0] == scale[0]
    assert uncapped[0] > 1.0e8
    assert capped[0] == 3.0
    assert upper_uncapped[0] > 1.0e8
    assert upper_capped[0] < 20.0


def test_force_matches_independent_equation_13_reference() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(loc=[5.0, -3.0, 2.0], size=(300, 3))
    X[:20, 0] += 100.0
    X[20:40, 1] -= 100.0
    estimator = ForceEstimator(exact_cutover=X.shape[0] + 1)
    observed = estimator.fit(X)
    assert estimator.thresholds is not None
    expected = _independent_trimmed_correlation(X, *estimator.thresholds)
    assert np.allclose(observed, expected, atol=1e-12)


def test_pairwise_acceptance_bounds_are_inclusive() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    result = _compute_trimmed_corr_numba(
        X,
        np.array([0.0, 0.0]),
        np.array([2.0, 2.0]),
    )
    assert result[0, 1] == pytest.approx(1.0)


def test_covariance_normalization_zeroes_undefined_pairs() -> None:
    covariance = np.array([[4.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
    result = _covariance_to_correlation(covariance)
    assert np.array_equal(np.diag(result), np.ones(3))
    assert result[0, 1] == result[1, 0] == 0.0
    assert np.all(np.isfinite(result))


def test_force_correlation_is_finite_symmetric_and_bounded() -> None:
    X = np.random.default_rng(5).normal(size=(500, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = ForceEstimator().fit(X)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, result.T)
    assert np.array_equal(np.diag(result), np.ones(5))
    assert result.min() >= -1.0
    assert result.max() <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lambda_scale": 0.0},
        {"lambda_scale": True},
        {"exact_cutover": 4},
        {"exact_cutover": 5.0},
        {"use_ter": 1},
        {"ter_max": 0.9},
        {"ter_max": True},
        {"epsilon": 0.0},
        {"epsilon": True},
    ],
)
def test_constructor_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ForceEstimator(**kwargs)


def test_input_validation() -> None:
    estimator = ForceEstimator()
    with pytest.raises(ValueError, match="two-dimensional"):
        estimator.fit(np.ones(10))
    with pytest.raises(ValueError, match="at least 5"):
        estimator.fit(np.ones((4, 2)))
    with pytest.raises(ValueError, match="at least 2"):
        estimator.fit(np.ones((5, 1)))
    bad = np.ones((5, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimator.fit(bad)


def test_retained_state_is_independent_of_sample_count() -> None:
    rng = np.random.default_rng(9)
    small = ForceEstimator()
    large = ForceEstimator()
    small_result = small.fit(rng.normal(size=(100, 6)))
    large_result = large.fit(rng.normal(size=(1000, 6)))
    assert small.state_nbytes_ == large.state_nbytes_
    assert small_result.nbytes == large_result.nbytes == 6 * 6 * 8


def test_force_reduces_cauchy_contamination_error() -> None:
    X, truth = generate_synthetic_data(
        n_samples=1000, n_features=6, contamination=0.10, seed=42
    )
    mask = np.triu(np.ones_like(truth, dtype=bool), k=1)
    force_error = np.sqrt(
        np.mean((ForceEstimator().fit(X)[mask] - truth[mask]) ** 2)
    )
    pearson_error = np.sqrt(
        np.mean((PearsonEstimator().fit(X)[mask] - truth[mask]) ** 2)
    )
    assert force_error < pearson_error / 5.0
