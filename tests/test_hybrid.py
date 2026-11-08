"""Exact-prefix readouts, standard P² handoff, and replay contracts."""

import numpy as np
import pytest

from force import ForceEstimator, P2Quantile
from force._quantiles import P2_PROBABILITIES, P2State
from force.reporting import estimator_parameters


def linear_quantile(values, probability):
    """Small sorted-order-statistic oracle, independent of NumPy quantile."""
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    fraction = index - lower
    return (1 - fraction) * ordered[lower] + fraction * ordered[upper]


@pytest.mark.parametrize("probability", P2_PROBABILITIES)
def test_scalar_exact_prefix_and_all_history_handoff(probability):
    values = [8.0, -2.0, 3.0, 7.0, 1.0, 20.0, -4.0, 3.0, 100.0, -10.0, 0.0]
    hybrid = P2Quantile(float(probability), exact_prefix=8)
    standard = P2Quantile(float(probability))
    assert np.isnan(hybrid.value)
    for count, value in enumerate(values, 1):
        hybrid.update(value)
        standard.update(value)
        if count <= 8:
            assert hybrid.value == pytest.approx(
                linear_quantile(values[:count], probability)
            )
        else:
            assert hybrid.value == standard.value
        # Early exact readouts never alter the standard P² marker history.
        np.testing.assert_array_equal(hybrid.marker_heights, standard.marker_heights)
        np.testing.assert_array_equal(
            hybrid.marker_positions, standard.marker_positions
        )
        np.testing.assert_array_equal(
            hybrid.desired_positions, standard.desired_positions
        )
    assert hybrid.state_nbytes_ == standard.state_nbytes_


@pytest.mark.parametrize("chunk_size", [1, 4, 5, 8, 9, 30])
def test_prefix_chunking_snapshots_and_memory_release(chunk_size):
    X = np.random.default_rng(901).lognormal(size=(31, 3))
    state = P2State(3, exact_prefix=9)
    standard_bytes = P2State(3).state_nbytes_
    assert state.state_nbytes_ == standard_bytes + 9 * 3 * 8
    for start in range(0, len(X), chunk_size):
        stop = min(len(X), start + chunk_size)
        state.update_batch(X[start:stop])
        if stop < 5:
            continue
        if stop <= 9:
            expected = np.array(
                [
                    [linear_quantile(X[:stop, j], q) for j in range(3)]
                    for q in P2_PROBABILITIES
                ]
            )
            np.testing.assert_allclose(state.quantiles, expected)
            assert state.state_nbytes_ == standard_bytes + 9 * 3 * 8
        else:
            expected = P2State(3).update_batch(X[:stop]).quantiles
            np.testing.assert_array_equal(state.quantiles, expected)
            assert state.state_nbytes_ == standard_bytes
        snapshot = state.quantiles
        snapshot[:] = -999
        np.testing.assert_allclose(state.quantiles, expected)


def test_prefix_input_copy_and_failed_boundary_validation():
    X = np.arange(16.0).reshape(8, 2)
    original = X.copy()
    state = P2State(2, exact_prefix=8).update_batch(X)
    expected = state.quantiles
    initial_bytes = state.state_nbytes_
    X[:] = -100
    np.testing.assert_array_equal(state.quantiles, expected)
    state.update_batch(np.empty((0, 2)))
    assert state.state_nbytes_ == initial_bytes
    with pytest.raises(ValueError, match="finite"):
        state.update_batch([[1, 2], [3, np.nan]])
    assert state.count == 8
    assert state.state_nbytes_ == initial_bytes
    np.testing.assert_array_equal(state.quantiles, expected)
    state.update_batch([[16, 17]])
    pure = P2State(2).update_batch(np.vstack([original, [16, 17]]))
    np.testing.assert_array_equal(state.quantiles, pure.quantiles)
    assert state.state_nbytes_ == pure.state_nbytes_


@pytest.mark.parametrize("n_samples", [5, 9, 10, 11, 200])
@pytest.mark.parametrize("chunk_size", [1, 9, 11, 4096])
def test_exact_prefix_batch_replay_and_two_pass_parity(n_samples, chunk_size):
    X = np.random.default_rng(52).normal(size=(n_samples, 3))
    calls = []

    def source():
        calls.append(True)
        step = 7 if len(calls) == 1 else 13
        return (X[i : i + step] for i in range(0, len(X), step))

    batch = ForceEstimator(exact_prefix=10)
    replay = ForceEstimator(exact_prefix=10)
    expected = batch.fit(X)
    observed = replay.fit_replayable(source, chunk_size=chunk_size)
    assert len(calls) == 2
    np.testing.assert_array_equal(batch.quantiles, replay.quantiles)
    np.testing.assert_array_equal(batch.thresholds, replay.thresholds)
    np.testing.assert_array_equal(batch.accepted_counts_, replay.accepted_counts_)
    np.testing.assert_allclose(expected, observed, atol=1e-14, rtol=0)
    if n_samples <= 10:
        exact = np.array(
            [[linear_quantile(X[:, j], q) for j in range(3)] for q in P2_PROBABILITIES]
        )
        np.testing.assert_allclose(batch.quantiles, exact)
    else:
        pure = ForceEstimator()
        pure_result = pure.fit(X)
        np.testing.assert_array_equal(batch.quantiles, pure.quantiles)
        np.testing.assert_array_equal(expected, pure_result)


@pytest.mark.parametrize("invalid", [-1, 1, 4, 5.0, True, np.bool_(False), None])
def test_invalid_prefix_configuration_is_rejected(invalid):
    for construct in (
        lambda: ForceEstimator(exact_prefix=invalid),
        lambda: P2State(2, exact_prefix=invalid),
        lambda: P2Quantile(0.5, exact_prefix=invalid),
    ):
        with pytest.raises(ValueError, match="exact_prefix"):
            construct()


def test_exact_prefix_and_small_batch_fallback_are_separate_modes():
    with pytest.raises(ValueError, match="exact_cutover=5"):
        ForceEstimator(exact_prefix=100, exact_cutover=100)
    estimator = ForceEstimator(exact_prefix=np.int64(10))
    parameters = estimator_parameters(estimator)
    assert parameters["exact_prefix"] == 10
    assert parameters["exact_cutover"] == 5
    assert estimator_parameters(ForceEstimator())["exact_prefix"] == 0


def test_five_row_exact_prefix_changes_readout_without_reseeding_markers():
    X = np.arange(10.0).reshape(5, 2)
    exact = P2State(2, exact_prefix=5).update_batch(X)
    pure = P2State(2).update_batch(X)
    assert not np.array_equal(exact.quantiles, pure.quantiles)
    np.testing.assert_allclose(exact.quantiles[2], pure.quantiles[2])
    exact.update_batch([[10, 11]])
    pure.update_batch([[10, 11]])
    np.testing.assert_array_equal(exact.quantiles, pure.quantiles)
