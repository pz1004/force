"""Numerical regressions preserve the paper's functional, not table values."""

import numpy as np
import pytest

from force import ForceEstimator, PearsonEstimator
from force.core import _compute_bounds_from_quantiles, _covariance_to_correlation
from force.diagnostics import POLICIES, BoundPolicy, diagnose_batch, reference_bounds


@pytest.mark.parametrize("offset", [0.0, 1e12, 1e15, -1e15])
def test_reference_and_baseline_preserve_translation(offset):
    base = np.tile(
        [[-2.0, -1.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 2.0], [2.0, 1.0]], (20, 1)
    )
    expected = np.corrcoef(base, rowvar=False)
    X = base + offset
    observed = diagnose_batch(X, policy=POLICIES["fixed"])
    np.testing.assert_allclose(PearsonEstimator().fit(X), expected, atol=1e-14)
    np.testing.assert_allclose(observed["production_correlation"], expected, atol=1e-14)
    np.testing.assert_allclose(observed["exact_correlation"], expected, atol=1e-14)
    assert observed["approximation_rmse"] < 1e-14
    np.testing.assert_array_equal(
        observed["accepted_pair_counts"], np.full((2, 2), len(X))
    )
    np.testing.assert_array_equal(
        observed["exact_accepted_pair_counts"], np.full((2, 2), len(X))
    )


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_pearson_preparation_handles_extreme_units(scale):
    base = np.array([[-2.0, -1.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
    np.testing.assert_allclose(
        PearsonEstimator().fit(base * scale),
        np.corrcoef(base, rowvar=False),
        atol=1e-14,
    )


@pytest.mark.parametrize("swap", [False, True])
def test_covariance_normalization_preserves_representable_tiny_correlation(swap):
    covariance = np.array([[1e300, 1e-200], [1e-200, 1e-300]])
    if swap:
        covariance = covariance[::-1, ::-1].copy()
    result = _covariance_to_correlation(covariance)
    assert result[0, 1] == pytest.approx(1e-200, rel=1e-14, abs=0)


@pytest.mark.parametrize(
    "quantiles,lam,cap",
    [
        ([-0.01, -0.005, 0.0, 0.005, 1e308], 3.0, 3.0),
        ([-1.7e308, -1.1e308, -1e308, -0.9e308, 1e308], 1.0, 3.0),
        ([-1e308, 0.9e308, 1e308, 1.1e308, 1.7e308], 1.0, 3.0),
        ([-1e-10, -0.5e-308, 0.0, 0.5e-308, 1.3e298], 3.0, None),
    ],
)
def test_bounds_avoid_unnecessary_intermediate_overflow(quantiles, lam, cap):
    q = np.array(quantiles)[:, None]
    policy = BoundPolicy("test", lambda_scale=lam, ter_max=cap)
    expected = reference_bounds(q, policy)
    result = _compute_bounds_from_quantiles(
        q, lambda_scale=lam, use_ter=True, ter_max=cap, epsilon=1e-10
    )
    np.testing.assert_allclose(result[3:], expected, rtol=1e-14, atol=0)
    assert np.all(np.isfinite(result))


def test_capped_exact_fit_trims_finite_extreme_tail_before_moment_accumulation():
    x = np.r_[np.linspace(-0.01, 0.01, 99), 1e308, 1e308]
    X = np.column_stack([x, x])
    estimator = ForceEstimator(exact_cutover=1000, ter_max=3)
    result = estimator.fit(X)
    np.testing.assert_allclose(result, np.ones((2, 2)), atol=1e-14)
    np.testing.assert_array_equal(estimator.accepted_counts_, np.full((2, 2), 99))
    np.testing.assert_allclose(estimator.ter_, [3, 3])


def test_batch_does_not_expand_status_labels_until_requested(monkeypatch):
    import force.core as core

    def forbidden(*args, **kwargs):
        raise AssertionError("String status labels were eagerly allocated")

    monkeypatch.setattr(core, "status_labels_from_codes", forbidden)
    estimator = ForceEstimator()
    estimator.fit(np.random.default_rng(1).normal(size=(50, 10)))
    assert estimator.pair_status_codes_.dtype == np.int8
    assert estimator.pair_status_codes_.nbytes == 100
    assert estimator.state_nbytes_ > 0
    with pytest.raises(AssertionError, match="eagerly allocated"):
        _ = estimator.pair_status_
