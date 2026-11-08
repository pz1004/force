"""Behavioral contracts for the shared quantile and accepted-pair states."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from force._moments import PairMoments, status_labels_from_codes
from force._numeric import normalize_correlation
from force._quantiles import P2_PROBABILITIES, P2Quantile, P2State, p2_desired_positions


def _accepted_oracle(X, lower, upper):
    """Independent long-double, rescaled two-pass accepted-subset reference."""
    p = X.shape[1]
    result = np.eye(p)
    counts = np.zeros((p, p), dtype=np.int64)
    statuses = np.full((p, p), "insufficient_samples", dtype="<U20")
    accepted = (X >= lower) & (X <= upper)
    for left in range(p):
        for right in range(left, p):
            subset = X[accepted[:, left] & accepted[:, right]][:, [left, right]]
            n = len(subset)
            counts[left, right] = counts[right, left] = n
            if n < 2:
                continue
            shifted = subset.astype(np.longdouble) - subset[0].astype(np.longdouble)
            magnitude = np.max(np.abs(shifted), axis=0)
            if np.any(magnitude == 0):
                statuses[left, right] = statuses[right, left] = "zero_variance"
                continue
            normalized = shifted / magnitude
            centered = normalized - np.mean(normalized, axis=0)
            value = np.sum(centered[:, 0] * centered[:, 1]) / (
                np.sqrt(np.sum(centered[:, 0] ** 2))
                * np.sqrt(np.sum(centered[:, 1] ** 2))
            )
            result[left, right] = result[right, left] = float(value)
            statuses[left, right] = statuses[right, left] = "ok"
    return result, counts, statuses


@pytest.mark.parametrize(
    "kind", ["skewed", "ties", "ascending", "descending", "constant"]
)
def test_quantile_chunks_and_scalar_share_state_transition(kind):
    rng = np.random.default_rng(63)
    X = rng.lognormal(size=(231, 3))
    if kind == "ties":
        X = np.round(X)
    elif kind == "ascending":
        X = np.sort(X, axis=0)
    elif kind == "descending":
        X = np.sort(X, axis=0)[::-1]
    elif kind == "constant":
        X[:] = [0.0, 1.0, -2.0]
    batch = P2State(3).update_batch(X)
    chunks = P2State(3)
    for start, stop in zip([0, 1, 4, 5, 6, 53, 231], [1, 4, 5, 6, 53, 231, 231]):
        chunks.update_batch(X[start:stop])
    assert chunks.count == len(X)
    np.testing.assert_array_equal(chunks.quantiles, batch.quantiles)
    assert chunks.state_nbytes_ == batch.state_nbytes_
    for q, probability in enumerate(P2_PROBABILITIES):
        tracker = P2Quantile(float(probability))
        for row in X:
            tracker.update(float(row[0]))
        assert tracker.value == batch.quantiles[q, 0]
        assert np.all(np.diff(tracker.marker_heights) >= 0)
        assert np.all(np.diff(tracker.marker_positions) > 0)
        np.testing.assert_allclose(
            tracker.desired_positions,
            p2_desired_positions(len(X), float(probability)),
            rtol=1e-13,
            atol=1e-11,
        )


def test_quantile_initialization_and_invalid_chunk_are_explicit():
    state = P2State(2)
    state.update_batch(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="at least five"):
        _ = state.quantiles
    with pytest.raises(ValueError, match="finite"):
        state.update_batch([[5.0, 6.0], [np.nan, 7.0]])
    assert state.count == 2
    state.update_batch([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    np.testing.assert_array_equal(state.quantiles, np.tile([5.0, 6.0], (5, 1)))
    value = state.quantiles
    value[:] = 0.0
    assert state.quantiles[0, 0] == 5.0  # snapshots cannot mutate state


def test_scalar_quantile_preinitialization_properties():
    tracker = P2Quantile(0.25)
    assert np.isnan(tracker.value)
    assert tracker.marker_heights.size == 0
    tracker.update(8.0).update(4.0)
    assert tracker.count == 2
    assert tracker.value == 5.0
    np.testing.assert_array_equal(tracker.marker_heights, [4.0, 8.0])
    np.testing.assert_array_equal(tracker.marker_positions, [1.0, 2.0])
    assert tracker.desired_positions.size == 0


def test_quantile_unsupported_arithmetic_poisoning():
    state = P2State(1, [0.5])
    state.update_batch(np.arange(5.0)[:, None])
    # Finite endpoint spans are supported; exhausted exact marker ranks are not.
    state.count = 2**53
    with pytest.raises(FloatingPointError, match="exactly representable"):
        state.update_batch([[5.0]])
    with pytest.raises(RuntimeError, match="failed"):
        _ = state.quantiles
    with pytest.raises(RuntimeError, match="failed"):
        state.update_batch([[0.0]])


@pytest.mark.parametrize(
    "scales",
    [
        [1.0, 1.0],
        [1e100, 1e100],
        [1e-100, 1e-100],
        [1e100, 1e-100],
        [1e200, 1e200],
        [1e-200, 1e-200],
        [1e-160, 1e-160],
        [1e200, 1e-200],
        [1e-200, 1e200],
    ],
)
def test_pair_normalization_is_stable_across_representable_moment_scales(scales):
    X = (
        np.array([[-2.0, -1.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
        * scales
    )
    lower, upper = X.min(axis=0), X.max(axis=0)
    observed = PairMoments(2).update_batch(X, lower, upper).finalize()
    expected = _accepted_oracle(X, lower, upper)
    np.testing.assert_allclose(observed[0], expected[0], rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(observed[1], expected[1])
    np.testing.assert_array_equal(observed[2], expected[2])
    assert observed[0][0, 1] == pytest.approx(0.8488746876271654, abs=1e-14)


def test_pair_specific_means_large_offsets_and_unequal_masks():
    X = np.array(
        [
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 4.0],
            [2.0, 2.0, 4.0],
            [-3.0, 1.0, 4.0],
            [5.0, -4.0, 4.0],
        ]
    )
    offsets = np.array([1e12, -1e12, 1e12])
    X += offsets
    lower, upper = offsets + [-1.0, -1.0, 4.0], offsets + [1.0, 1.0, 4.0]
    state = PairMoments(3)
    expected = _accepted_oracle(X, lower, upper)
    for row in X:
        state.update_batch(row[None, :], lower, upper)
    observed = state.finalize()
    np.testing.assert_allclose(observed[0], expected[0], atol=1e-14)
    np.testing.assert_array_equal(observed[1], expected[1])
    np.testing.assert_array_equal(observed[2], expected[2])
    assert observed[0][0, 1] == pytest.approx(-1.0 / 3.0)
    assert observed[2][2, 2] == "zero_variance"


def test_pair_state_chunk_boundaries_and_snapshot_are_stable():
    rng = np.random.default_rng(72)
    X = rng.normal(size=(317, 4))
    lower, upper = np.array([-1.0, -0.5, -2.0, -1.0]), np.array([1.0, 2.0, 0.5, 1.0])
    state = PairMoments(4)
    initial_bytes = state.state_nbytes_
    for chunk in np.array_split(X, 19):
        state.update_batch(chunk, lower, upper)
    whole = PairMoments(4).update_batch(X, lower, upper)
    for observed, expected in zip(state.finalize(), whole.finalize()):
        np.testing.assert_array_equal(observed, expected)
    oracle = _accepted_oracle(X, lower, upper)
    observed = state.finalize()
    np.testing.assert_allclose(observed[0], oracle[0], atol=1e-14)
    np.testing.assert_array_equal(observed[1], oracle[1])
    np.testing.assert_array_equal(observed[2], oracle[2])
    observed[0][:] = 0
    observed[1][:] = 0
    observed[2][:] = "modified"
    assert state.finalize()[0][0, 0] == 1.0
    assert state.finalize()[1][0, 0] > 0
    assert state.finalize()[2][0, 0] == "ok"
    assert state.state_nbytes_ == initial_bytes
    assert state.count == len(X)


def test_pair_degeneracy_distinguishes_zero_one_two_and_constant_rows():
    X = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 1.0]])
    lower, upper = np.array([10.0, 1.0, 1.0, 1.0]), np.array([11.0, 1.0, 2.0, 1.0])
    result, counts, statuses = PairMoments(4).update_batch(X, lower, upper).finalize()
    np.testing.assert_array_equal(np.diag(result), np.ones(4))
    np.testing.assert_array_equal(np.diag(counts), [0, 1, 2, 2])
    np.testing.assert_array_equal(
        np.diag(statuses),
        ["insufficient_samples", "insufficient_samples", "ok", "zero_variance"],
    )
    assert statuses[2, 3] == "zero_variance"
    assert result[2, 3] == 0.0


def test_pair_invalid_chunk_does_not_change_state():
    state = PairMoments(2).update_batch([[0.0, 1.0]], [-10.0, -10.0], [10.0, 10.0])
    before = state.finalize()
    with pytest.raises(ValueError, match="finite"):
        state.update_batch([[1.0, 1.0], [np.nan, 2.0]], [-10.0, -10.0], [10.0, 10.0])
    with pytest.raises(ValueError, match="finite"):
        state.update_batch([[1.0, 1.0]], [-np.inf, -10.0], [10.0, 10.0])
    with pytest.raises(ValueError, match="at most"):
        state.update_batch([[1.0, 1.0]], [11.0, -10.0], [10.0, 10.0])
    assert state.count == 1
    for actual, expected in zip(state.finalize(), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("object_array", [False, True])
def test_states_reject_complex_before_discarding_imaginary_parts(object_array):
    X = np.array([[1 + 2j, 3 + 0j]])
    if object_array:
        X = np.array([[np.complex128(1 + 2j), np.complex128(3)]], dtype=object)
    quantiles = P2State(2)
    moments = PairMoments(2)
    with pytest.raises(ValueError, match="Complex"):
        quantiles.update_batch(X)
    with pytest.raises(ValueError, match="Complex"):
        moments.update_batch(X, [-10.0, -10.0], [10.0, 10.0])
    assert quantiles.count == moments.count == 0


def test_pair_unrepresentable_count_fails_and_all_accessors_reject_failed_state():
    state = PairMoments(2)
    # No feasible allocation is needed to test the exactly represented count
    # boundary beyond which the next Welford weight would become ambiguous.
    state._counts[:] = 2**53
    with pytest.raises(FloatingPointError, match="Pair count"):
        state.update_batch([[0.0, 0.0]], [-1.0, -1.0], [1.0, 1.0])
    for accessor in (
        state.correlation,
        state.accepted_counts,
        state.status_codes,
        state.status_labels,
        state.finalize,
    ):
        with pytest.raises(RuntimeError, match="failed"):
            accessor()
    with pytest.raises(RuntimeError, match="failed"):
        state.update_batch([[0.0, 0.0]], [-1.0, -1.0], [1.0, 1.0])


def test_constant_coordinates_remain_valid_at_subnormal_magnitudes():
    tiny = np.nextafter(0.0, 1.0)
    X = np.tile([tiny, -tiny], (5, 1))
    result, counts, status = PairMoments(2).update_batch(X, X[0], X[0]).finalize()
    np.testing.assert_array_equal(result, np.eye(2))
    np.testing.assert_array_equal(counts, np.full((2, 2), 5))
    assert np.all(status == "zero_variance")


@pytest.mark.parametrize("shift", [0, -2])
def test_tiny_transient_prefix_does_not_abort_ordinary_final_moments(shift):
    X = np.array([[0, 0], [1e-160, 1e-160], [-2, -1], [-1, -1], [0, 0], [1, 2], [2, 1]])
    X = np.roll(X, shift, axis=0)
    expected = _accepted_oracle(X, [-10, -10], [10, 10])
    state = PairMoments(2)
    for row in X:
        state.update_batch(row[None, :], [-10, -10], [10, 10])
    np.testing.assert_allclose(state.correlation(), expected[0], rtol=1e-14)
    np.testing.assert_array_equal(state.accepted_counts(), np.full((2, 2), 7))
    np.testing.assert_array_equal(state.status_codes(), np.zeros((2, 2)))


def test_tiny_representable_correlation_survives_unequal_units_and_column_order():
    A, B, d = np.ldexp(1.0, 498), np.ldexp(1.0, -498), np.ldexp(1.0, -166)
    X = np.array([[0, 0], [A, 0], [-A, 0], [0, B], [0, -B], [d, B]])
    # Analytic moments avoid cancellation in even a long-double centered-dot
    # oracle: its opposite order-one terms would erase this tiny co-moment.
    with localcontext() as context:
        context.prec = 100
        a, b, c = Decimal(2) ** 498, Decimal(2) ** -498, Decimal(2) ** -166
        xx, yy, xy = 2 * a * a + 5 * c * c / 6, 17 * b * b / 6, 5 * c * b / 6
        expected = float(xy / xx.sqrt() / yy.sqrt())
    for values in (X, X[:, ::-1]):
        state = PairMoments(2).update_batch(values, [-1e155] * 2, [1e155] * 2)
        observed = state.correlation()[0, 1]
        assert observed != 0
        np.testing.assert_allclose(observed, expected, rtol=1e-14, atol=0)


def test_large_offsets_with_small_representable_spread_match_shifted_oracle():
    base = np.array([[-2, -1], [-1, -1], [0, 0], [1, 2], [2, 1]], dtype=float)
    X = np.array([np.ldexp(1.0, 800), -np.ldexp(1.0, 800)]) + np.ldexp(base, 748)
    lower, upper = X.min(axis=0), X.max(axis=0)
    observed = PairMoments(2).update_batch(X, lower, upper).correlation()
    expected = _accepted_oracle(X, lower, upper)[0]
    np.testing.assert_allclose(observed, expected, rtol=1e-14)


def test_scale_changes_across_chunks_preserve_pair_specific_accepted_moments():
    X = np.array(
        [
            [0, 0, 0],
            [1e-200, 2e-200, -1e-200],
            [-1e-200, 3e-200, 2e-200],
            [1e200, 1e-200, 3e100],
            [-2e200, -4e-200, 2e100],
            [3e200, -2e-200, -1e100],
            [-1e200, 2e-200, -2e100],
        ]
    )
    lower = np.array([-2e200, -4e-200, -2e100])
    upper = np.array([2e200, 3e-200, 2e100])
    whole = PairMoments(3).update_batch(X, lower, upper)
    chunked = PairMoments(3)
    for chunk in (X[:2], X[2:4], X[4:]):
        chunked.update_batch(chunk, lower, upper)
    expected = _accepted_oracle(X, lower, upper)
    for observed, reference in zip(chunked.finalize(), whole.finalize()):
        np.testing.assert_array_equal(observed, reference)
    np.testing.assert_allclose(chunked.correlation(), expected[0], rtol=1e-14)
    np.testing.assert_array_equal(chunked.accepted_counts(), expected[1])
    np.testing.assert_array_equal(chunked.status_labels(), expected[2])


@pytest.mark.parametrize("magnitude", [np.nextafter(0.0, 1.0), np.finfo(float).max])
def test_nonconstant_finite_extreme_endpoints_keep_positive_variance(magnitude):
    X = np.array([[-magnitude, magnitude], [0, 0], [magnitude, -magnitude]])
    state = PairMoments(2).update_batch(X, X.min(axis=0), X.max(axis=0))
    np.testing.assert_allclose(state.correlation(), [[1, -1], [-1, 1]], atol=1e-15)
    np.testing.assert_array_equal(state.status_codes(), np.zeros((2, 2)))


@pytest.mark.parametrize(
    "moments",
    [
        [1e300, 1e-300, 1e-200],
        [1e-300, 1e300, 1e-200],
        [1e300, 1e300, -1e300],
        [1e-300, 1e-300, 1e-300],
        [np.nextafter(0.0, 1.0)] * 3,
    ],
)
def test_exponent_normalization_matches_independent_decimal_arithmetic(moments):
    with localcontext() as context:
        context.prec = 100
        xx, yy, xy = map(Decimal.from_float, moments)
        expected = float(xy / (xx * yy).sqrt())
    np.testing.assert_allclose(
        normalize_correlation(*moments), expected, rtol=2e-15, atol=0
    )


def test_compact_diagnostics_are_independent_owned_snapshots(monkeypatch):
    state = PairMoments(3).update_batch(
        [[0, 0, 1], [1, -1, 1], [2, -2, 1]], [-10] * 3, [10] * 3
    )
    initial_bytes = state.state_nbytes_
    codes = state.status_codes()
    assert codes.dtype == np.int8
    np.testing.assert_array_equal(
        status_labels_from_codes(codes), state.status_labels()
    )
    codes[:] = 1
    counts = state.accepted_counts()
    counts[:] = 0
    correlation = state.correlation()
    correlation[:] = 0
    assert state.state_nbytes_ == initial_bytes
    assert state.accepted_counts()[0, 1] == 3
    assert state.status_codes()[0, 1] == 0
    assert state.status_codes()[2, 2] == 2
    assert state.correlation()[0, 1] == pytest.approx(-1)

    def unexpected_labels(_):
        raise AssertionError("Label allocation was not requested")

    monkeypatch.setattr("force._moments.status_labels_from_codes", unexpected_labels)
    state.correlation()
    state.accepted_counts()
    state.status_codes()
