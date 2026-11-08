"""Finite-result quantile arithmetic checked independently of production formulas."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from force import ForceEstimator, P2Quantile
from force._quantiles import (
    P2_PROBABILITIES,
    P2State,
    _p2_transition,
    linear_quantiles,
)


def _decimal_linear_quantile(values, probability):
    ordered = sorted(Decimal.from_float(float(value)) for value in values)
    with localcontext() as context:
        # Exact subnormal coefficients need hundreds of decimal digits; avoid
        # perturbing a half-subnormal tie before converting back to float64.
        context.prec = 1200
        rank = Decimal.from_float(float(probability)) * (len(ordered) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        weight = rank - lower
        return float((1 - weight) * ordered[lower] + weight * ordered[upper])


def _decimal_p2_step(heights, positions, desired, increments, value):
    """One canonical P² step with Decimal height arithmetic and float ranks."""
    with localcontext() as context:
        context.prec = 100
        h = [Decimal.from_float(float(v)) for v in heights]
        value = Decimal.from_float(float(value))
        n = np.array(positions, copy=True)
        goal = desired + increments
        if value < h[0]:
            h[0], cell = value, 0
        elif value >= h[4]:
            h[4], cell = value, 3
        else:
            cell = max(i for i in range(4) if h[i] <= value)
        n[cell + 1 :] += 1
        linear_candidates = {}
        for i in range(1, 4):
            gap = goal[i] - n[i]
            if gap >= 1 and n[i + 1] - n[i] > 1:
                d = Decimal(1)
            elif gap <= -1 and n[i] - n[i - 1] > 1:
                d = Decimal(-1)
            else:
                continue
            a, b = Decimal(int(n[i] - n[i - 1])), Decimal(int(n[i + 1] - n[i]))
            candidate = h[i] + d / (a + b) * (
                (a + d) * (h[i + 1] - h[i]) / b + (b - d) * (h[i] - h[i - 1]) / a
            )
            adjacent = i + int(d)
            linear = h[i] + d * (h[adjacent] - h[i]) / Decimal(int(n[adjacent] - n[i]))
            linear_candidates[i] = float(linear)
            h[i] = candidate if h[i - 1] < candidate < h[i + 1] else linear
            # Each stored marker is float64 in the implementation under test.
            h[i] = Decimal.from_float(float(h[i]))
            n[i] += int(d)
        return np.array([float(v) for v in h]), n, linear_candidates


@pytest.mark.parametrize("prefix", [0, 5])
@pytest.mark.parametrize("probability", P2_PROBABILITIES)
def test_provisional_readout_handles_opposite_finite_endpoints(prefix, probability):
    maximum = np.finfo(float).max
    tracker = P2Quantile(float(probability), exact_prefix=prefix)
    tracker.update(-maximum).update(maximum)
    with np.errstate(over="raise", invalid="raise"):
        actual = tracker.value
    expected = _decimal_linear_quantile([-maximum, maximum], probability)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, rel=2e-15, abs=0)


def test_exact_prefix_preserves_integer_rank_endpoints():
    maximum = np.finfo(float).max
    values = np.array([-maximum, -maximum, maximum, maximum, maximum])
    state = P2State(1, exact_prefix=5).update_batch(values[:, None])
    with np.errstate(over="raise", invalid="raise"):
        actual = state.quantiles[:, 0]
    expected = [_decimal_linear_quantile(values, p) for p in P2_PROBABILITIES]
    np.testing.assert_array_equal(actual, expected)
    assert actual[1] == -maximum


@pytest.mark.parametrize("axis", [0, 1, -1, None])
@pytest.mark.parametrize(
    "probabilities", [0.3, [0, 0.25, 0.5, 0.99, 1], [[0, 0.3], [0.8, 1]]]
)
def test_exact_linear_shape_and_ordinary_values_match_numpy(axis, probabilities):
    source = np.random.default_rng(47).normal(size=(17, 7))
    expected = np.quantile(source, probabilities, axis=axis, method="linear")
    source.setflags(write=False)
    observed = linear_quantiles(source, probabilities, axis=axis)
    assert np.shape(observed) == np.shape(expected)
    np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=1e-15)


def test_exact_linear_quantiles_preserve_subnormal_rounding_and_endpoints():
    tiny = np.nextafter(0.0, 1.0)
    values = np.array([-tiny, 2 * tiny])
    probabilities = [0, 0.25, 0.5, 0.75, 1]
    expected = [_decimal_linear_quantile(values, p) for p in probabilities]
    np.testing.assert_array_equal(linear_quantiles(values, probabilities), expected)
    assert linear_quantiles(values, 0.5) == 0.0


@pytest.mark.parametrize(
    "values,probabilities",
    [
        ([], [0.5]),
        ([0, np.inf], [0.5]),
        ([0, np.nan], [0.5]),
        ([0, 1], [-0.1]),
        ([0, 1], [np.nan]),
    ],
)
def test_exact_linear_readouts_reject_invalid_input(values, probabilities):
    with pytest.raises(ValueError):
        linear_quantiles(values, probabilities)


def test_large_stream_failing_step_matches_decimal_interpolation():
    values = np.linspace(-1.0, 1.0, 1000) * 1e306
    tracker = P2Quantile(0.01)
    for value in values[:606]:
        tracker.update(float(value))
    expected, positions, _ = _decimal_p2_step(
        tracker.marker_heights,
        tracker.marker_positions,
        tracker.desired_positions,
        np.array([0.0, 0.005, 0.01, 0.505, 1.0]),
        values[606],
    )
    tracker.update(float(values[606]))
    np.testing.assert_allclose(tracker.marker_heights, expected, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(tracker.marker_positions, positions)


def test_overflow_does_not_replace_a_valid_parabolic_update_with_linear():
    heights = np.array([-1e306, -0.9e306, -0.7e306, 0.0, 1e306])
    positions = np.array([1.0, 4.0, 7.0, 306.0, 606.0])
    desired = np.array([1.0, 4.025, 7.05, 306.525, 606.0])
    increments = np.array([0.0, 0.005, 0.01, 0.505, 1.0])
    expected, new_positions, linear = _decimal_p2_step(
        heights, positions, desired, increments, 0.2e306
    )
    assert not np.isclose(expected[3], linear[3], rtol=1e-8)
    _p2_transition(heights, positions, desired, increments, 0.2e306)
    np.testing.assert_allclose(heights, expected, rtol=2e-15, atol=0)
    np.testing.assert_array_equal(positions, new_positions)


@pytest.mark.parametrize("probability", P2_PROBABILITIES)
@pytest.mark.parametrize("order", ["ascending", "descending", "ties"])
def test_large_unit_p2_trajectories_follow_the_unit_scale(probability, order):
    values = np.linspace(-1.0, 1.0, 1000)
    if order == "descending":
        values = values[::-1]
    elif order == "ties":
        values = np.round(values, 1)
    unit = P2State(1, [probability]).update_batch(values[:, None])
    scaled = P2State(1, [probability]).update_batch(values[:, None] * 1e306)
    np.testing.assert_allclose(
        scaled.quantiles / 1e306, unit.quantiles, rtol=2e-12, atol=2e-14
    )


def test_force_and_replay_support_representable_large_unit_bounds():
    x = np.linspace(-1.0, 1.0, 1000) * 1e306
    data = np.column_stack([x, x])
    batch = ForceEstimator()
    replay = ForceEstimator()
    result = batch.fit(data)
    replayed = replay.fit_replayable(lambda: iter(data), chunk_size=19)
    np.testing.assert_allclose(result, np.ones((2, 2)), atol=1e-14)
    np.testing.assert_array_equal(replayed, result)
    np.testing.assert_array_equal(batch.quantiles, replay.quantiles)
    assert np.all(np.isfinite(batch.thresholds))


def test_opposite_maximum_endpoints_remain_finite_during_p2_updates():
    maximum = np.finfo(float).max
    state = P2State(1).update_batch(np.array([-maximum] * 3 + [maximum] * 10)[:, None])
    assert np.all(np.isfinite(state.quantiles))
    assert np.all(np.abs(state.quantiles) <= maximum)
