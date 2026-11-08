"""Replay semantics, bounded conversion, and independent equation references."""

import numpy as np
import pytest

from force import ForceEstimator, TrimmedPearsonExact
from force.core import _compute_bounds_from_quantiles, _covariance_to_correlation
from force._input import validated_chunks


def reference(X, quantiles, *, lam=3.0, ter_max=None, use_ter=True, eps=1e-10):
    # Deliberately independent of production bounds and moment helpers.
    q01, q25, median, q75, q99 = quantiles
    scale = np.maximum(0.0, (q75 - q25) / 1.349)
    ratio = np.ones(X.shape[1])
    if use_ter:
        regular = np.abs(median - q01) >= eps
        ratio[regular] = np.maximum(
            1,
            np.abs(q99[regular] - median[regular])
            / (np.abs(median[regular] - q01[regular]) + eps),
        )
        if ter_max is not None:
            ratio = np.minimum(ratio, ter_max)
    lower, upper = median - lam * ratio * scale, median + lam * ratio * scale
    accepted = (X >= lower) & (X <= upper)
    counts = accepted.astype(np.int64).T @ accepted.astype(np.int64)
    corr = np.eye(X.shape[1])
    for j in range(X.shape[1]):
        for k in range(j + 1, X.shape[1]):
            subset = X[accepted[:, j] & accepted[:, k]][:, [j, k]]
            if len(subset) < 2 or np.any(np.ptp(subset, axis=0) == 0):
                corr[j, k] = corr[k, j] = 0
            else:
                corr[j, k] = corr[k, j] = np.corrcoef(subset, rowvar=False)[0, 1]
    return scale, ratio, lower, upper, counts, corr


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("policy", [{}, {"ter_max": 3.0}, {"use_ter": False}])
def test_all_equations_against_independent_reference(exact, policy):
    X = np.random.default_rng(7).normal(loc=[2, -4, 1], size=(301, 3))
    X[:19, 0] += 100
    X[30:43, 1] -= 50
    X[80:90] = [2, -4, 1]
    estimator = ForceEstimator(exact_cutover=302 if exact else 5, **policy)
    result = estimator.fit(X)
    q = (
        np.quantile(X, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0)
        if exact
        else estimator.quantiles
    )
    scale, ter, lower, upper, counts, corr = reference(X, q, **policy)
    np.testing.assert_allclose(estimator.quantiles, q, atol=1e-12)
    np.testing.assert_allclose(estimator.scale_, scale, atol=1e-12)
    np.testing.assert_allclose(estimator.ter_, ter, atol=1e-12)
    np.testing.assert_allclose(estimator.thresholds, (lower, upper), atol=1e-12)
    np.testing.assert_array_equal(estimator.accepted_counts_, counts)
    np.testing.assert_allclose(result, corr, atol=1e-12)
    if exact:
        expected = TrimmedPearsonExact(
            use_ter=policy.get("use_ter", True), ter_max=policy.get("ter_max")
        ).fit(X)
        np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("chunk_size", [1, 4, 5, 6, 19, 4096])
def test_replay_matches_batch_across_bootstrap_and_chunk_boundaries(chunk_size):
    X = np.random.default_rng(52).normal(size=(111, 4))
    calls = []

    def source():
        calls.append(True)
        # Different partitioning on each traversal is permitted.
        step = 7 if len(calls) == 1 else 13
        return (X[i : i + step] for i in range(0, len(X), step))

    expected, observed = ForceEstimator(), ForceEstimator()
    a = expected.fit(X)
    b = observed.fit_replayable(source, chunk_size=chunk_size)
    assert len(calls) == 2
    np.testing.assert_array_equal(expected.quantiles, observed.quantiles)
    np.testing.assert_array_equal(expected.accepted_counts_, observed.accepted_counts_)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-14)
    assert observed.n_samples_ == len(X)
    assert observed.execution_mode_ == "replay"


@pytest.mark.parametrize("change", ["content", "order", "length", "features"])
def test_changed_replay_rejected_without_overwriting_prior_fit(change):
    X = np.random.default_rng(25).normal(size=(20, 3))
    other = X.copy()
    if change == "content":
        other[0, 0] += 1
    elif change == "order":
        other = other[::-1]
    elif change == "length":
        other = other[:-1]
    else:
        other = other[:, :2]
    calls = []

    def source():
        calls.append(True)
        return iter(X if len(calls) == 1 else other)

    estimator = ForceEstimator()
    estimator.fit(X)
    old = estimator.quantiles.copy()
    with pytest.raises(ValueError, match="changed"):
        estimator.fit_replayable(source)
    np.testing.assert_array_equal(old, estimator.quantiles)
    assert estimator.execution_mode_ == "batch"


def test_replay_rejects_oneshot_empty_and_exact_fallback_sources():
    X = np.arange(40.0).reshape(20, 2)
    once = iter(X)
    with pytest.raises(ValueError, match="one-shot"):
        ForceEstimator().fit_replayable(lambda: once)
    with pytest.raises(ValueError, match="fresh"):
        ForceEstimator().fit_replayable(iter(X))
    with pytest.raises(ValueError, match="at least 5"):
        ForceEstimator().fit_replayable(lambda: iter(X[:4]))
    with pytest.raises(ValueError, match="exact_cutover=5"):
        ForceEstimator(exact_cutover=100).fit_replayable(lambda: iter(X))


@pytest.mark.parametrize("chunk_size", [0, -1, True, 2.5])
def test_invalid_chunk_sizes(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        ForceEstimator().fit_replayable(lambda: [], chunk_size)


def test_chunk_validation_and_reused_row_buffer():
    X = np.arange(30.0, dtype=np.float32).reshape(15, 2)

    def rows():
        row = np.zeros(2)
        for i in range(15):
            row[:] = X[i]
            yield row

    chunks = list(validated_chunks(rows(), 4))
    assert all(len(c) <= 4 and c.dtype == np.float64 for c in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), X)
    chunks = list(validated_chunks([X.tolist()], 4))
    assert max(map(len, chunks)) == 4
    np.testing.assert_array_equal(np.concatenate(chunks), X)
    with pytest.raises(ValueError, match="finite"):
        list(validated_chunks([[[1, 2], [3, np.inf]]], 1))
    with pytest.raises(ValueError, match="feature count"):
        list(validated_chunks([[1, 2], [1, 2, 3]], 4))


def test_shifted_exact_reference_regression():
    x = np.linspace(99, 101, 100)
    result = TrimmedPearsonExact(use_ter=True).fit(np.column_stack([x, 2 * x + 1]))
    assert result[0, 1] == pytest.approx(1)


def test_bound_diagnostics_and_explicit_overflow():
    estimator = ForceEstimator()
    estimator._compute_quantiles = lambda X: np.array(
        [[-3, 4], [1, 4], [0, 4], [-1, 4], [3, 4]], dtype=float
    )
    with pytest.warns(RuntimeWarning):
        estimator.fit(np.ones((8, 2)))
    np.testing.assert_array_equal(
        estimator.bound_status_["crossed_quartiles"], [True, False]
    )
    np.testing.assert_array_equal(estimator.bound_status_["zero_iqr"], [False, True])
    huge = np.array([[-1e308], [-1e308], [0], [1e308], [1e308]])
    with pytest.raises(FloatingPointError):
        _compute_bounds_from_quantiles(
            huge, lambda_scale=3, use_ter=True, ter_max=None, epsilon=1e-10
        )


@pytest.mark.parametrize("scale", [1e-200, 1, 1e200])
def test_covariance_normalization_extreme_units(scale):
    result = _covariance_to_correlation(np.array([[4, 1], [1, 9.0]]) * scale)
    assert result[0, 1] == pytest.approx(1 / 6)


def test_covariance_normalization_preserves_subnormal_variances():
    covariance = np.full((2, 2), np.nextafter(0.0, 1.0))
    np.testing.assert_allclose(_covariance_to_correlation(covariance), np.ones((2, 2)))


@pytest.mark.parametrize("object_array", [False, True])
def test_complex_input_is_rejected_consistently(object_array):
    X = np.ones((10, 2), dtype=complex) * (1 + 2j)
    if object_array:
        X = np.full((10, 2), None, dtype=object)
        for index in np.ndindex(X.shape):
            X[index] = np.complex128(1 + 2j)
    with pytest.raises(ValueError):
        ForceEstimator().fit(X)
    with pytest.raises(ValueError):
        ForceEstimator().fit_replayable(lambda: iter(X))


def test_invalid_covariance_is_not_clipped_to_a_valid_correlation():
    with pytest.raises(FloatingPointError, match="invalid correlation"):
        _covariance_to_correlation([[1, 2], [2, 1]])
