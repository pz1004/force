"""Validation of benchmark estimators and compatibility wrappers."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

from force import (
    ExactTrimmedEstimator,
    FastMCDEstimator,
    PearsonEstimator,
    SpearmanEstimator,
    TrimmedPearsonExact,
    WinsorizedEstimator,
)


@pytest.fixture
def data() -> np.ndarray:
    return np.random.default_rng(3).normal(size=(100, 4))


@pytest.mark.parametrize(
    "estimator",
    [
        PearsonEstimator(),
        SpearmanEstimator(),
        WinsorizedEstimator(),
        FastMCDEstimator(),
        TrimmedPearsonExact(),
        TrimmedPearsonExact(use_ter=True),
    ],
)
def test_estimator_output_contract(estimator, data: np.ndarray) -> None:
    result = estimator.fit(data)
    assert result.shape == (4, 4)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, result.T)
    assert np.array_equal(np.diag(result), np.ones(4))
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_zero_variance_baseline_output_is_finite() -> None:
    X = np.column_stack((np.arange(10.0), np.ones(10), np.arange(10.0)))
    result = PearsonEstimator().fit(X)
    assert np.all(np.isfinite(result))
    assert result[0, 1] == 0.0
    assert result[1, 2] == 0.0


@pytest.mark.parametrize(
    "estimator",
    [
        PearsonEstimator(),
        SpearmanEstimator(),
        WinsorizedEstimator(),
        TrimmedPearsonExact(),
        TrimmedPearsonExact(use_ter=True),
    ],
)
def test_validity_distinguishes_compatibility_zero_from_defined_correlation(estimator):
    X = np.column_stack((np.arange(20.0), np.ones(20), np.arange(20.0)[::-1]))
    warning = (
        pytest.warns(RuntimeWarning)
        if isinstance(estimator, TrimmedPearsonExact) and estimator.use_ter
        else nullcontext()
    )
    with warning:
        result = estimator.fit(X)
    assert result[0, 1] == 0.0
    assert result[0, 2] == pytest.approx(-1.0)
    expected = np.array([[0, 2, 0], [2, 2, 2], [0, 2, 0]], dtype=np.int8)
    np.testing.assert_array_equal(estimator.pair_status_codes_, expected)
    # A status snapshot must not mutate later statistical interpretation.
    estimator.pair_status_codes_[:] = 0
    np.testing.assert_array_equal(estimator.pair_status_codes_, expected)


@pytest.mark.parametrize("quantile_method", ["linear", "tail_count"])
def test_winsorized_validity_uses_transformed_values(quantile_method):
    # At n=21, the linear 95th percentile is exactly the last zero. At n=20
    # it interpolates between zero and one, leaving a nonconstant column.
    X = np.column_stack((np.arange(21.0), np.r_[np.zeros(20), 1.0]))
    estimator = WinsorizedEstimator(quantile_method=quantile_method)
    estimator.fit(X)
    assert np.var(X[:, 1]) > 0
    assert estimator.pair_status_codes_[1, 1] == 2
    assert estimator.pair_status_codes_[0, 1] == 2


def test_winsorized_linear_quantiles_clip_even_a_ten_row_sample():
    X = np.column_stack((np.arange(10.0), np.r_[np.arange(9.0), -100.0]))
    # Ranks .45 and 8.55 give x bounds [.45, 8.55], y bounds [-55, 7.55].
    # Write the transformed observations explicitly, independently of the
    # production quantile routine and its clipping implementation.
    transformed = np.column_stack(
        (np.r_[0.45, np.arange(1.0, 9.0), 8.55], np.r_[np.arange(8.0), 7.55, -55.0])
    )
    estimator = WinsorizedEstimator()
    result = estimator.fit(X)
    np.testing.assert_allclose(
        result, np.corrcoef(transformed, rowvar=False), atol=1e-14
    )
    assert result[0, 1] == pytest.approx(-0.37357915205977593, abs=1e-14)
    np.testing.assert_array_equal(estimator.pair_status_codes_, np.zeros((2, 2)))


def test_winsorized_tail_count_preserves_historical_small_sample_rule():
    X = np.column_stack((np.arange(10.0), np.r_[np.arange(9.0), -100.0]))
    # floor(10*.05) == 0 at both tails, so the historical rule clips no rows.
    estimator = WinsorizedEstimator(quantile_method="tail_count")
    np.testing.assert_allclose(
        estimator.fit(X), np.corrcoef(X, rowvar=False), atol=1e-14
    )


def test_winsorized_linear_asymmetric_limits():
    X = np.column_stack((np.arange(5.0), [4.0, 1.0, 3.0, 0.0, 2.0]))
    transformed = np.array([[1.0, 3.6], [1.0, 1.0], [2.0, 3.0], [3.0, 1.0], [3.6, 2.0]])
    actual = WinsorizedEstimator(limits=(0.25, 0.1)).fit(X)
    np.testing.assert_allclose(
        actual, np.corrcoef(transformed, rowvar=False), atol=1e-14
    )


@pytest.mark.parametrize("quantile_method", ["linear", "tail_count"])
def test_winsorized_zero_limits_preserve_the_pearson_sample(quantile_method):
    X = np.column_stack((np.arange(10.0), np.r_[np.arange(9.0), -100.0]))
    actual = WinsorizedEstimator(
        limits=(0.0, 0.0), quantile_method=quantile_method
    ).fit(X)
    np.testing.assert_allclose(actual, np.corrcoef(X, rowvar=False), atol=1e-14)


def test_winsorized_linear_extreme_endpoints_remain_finite():
    # Both sample quartiles are endpoints. Ordinary NumPy interpolation can
    # form an overflowing opposite-sign difference even at interpolation zero.
    base = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, 1.0]])
    X = base * np.finfo(float).max
    estimator = WinsorizedEstimator(limits=(0.25, 0.25))
    actual = estimator.fit(X)
    np.testing.assert_allclose(actual, np.corrcoef(base, rowvar=False), atol=1e-14)
    np.testing.assert_array_equal(estimator.pair_status_codes_, np.zeros((2, 2)))


@pytest.mark.parametrize(
    "quantile_method", [None, True, 1, "nearest", [], np.array(["linear"])]
)
def test_winsorized_quantile_method_validation(quantile_method):
    with pytest.raises(ValueError, match="quantile_method"):
        WinsorizedEstimator(quantile_method=quantile_method)


def test_exact_trimmed_diagnostics_are_pair_specific_and_parameters_stay_frozen():
    from dataclasses import FrozenInstanceError

    estimator = TrimmedPearsonExact(lam=1.0)
    x = np.arange(21.0)
    X = np.column_stack((x, x))
    estimator.fit(X)
    np.testing.assert_array_equal(estimator.accepted_counts_, np.full((2, 2), 15))
    np.testing.assert_array_equal(
        estimator.pair_status_codes_, np.zeros((2, 2), dtype=np.int8)
    )
    with pytest.raises(FrozenInstanceError):
        estimator.lam = 2.0


@pytest.mark.parametrize("scales", [[1e100, 1e-100], [1e-200, 1e200]])
def test_pearson_baseline_handles_extreme_coordinate_units(scales) -> None:
    X = np.array([[-2, -1], [-1, -1], [0, 0], [1, 2], [2, 1.0]])
    expected = np.corrcoef(X, rowvar=False)
    result = PearsonEstimator().fit(X * scales)
    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_fastmcd_does_not_silently_fallback() -> None:
    X = np.ones((10, 3))
    with pytest.warns(UserWarning, match="not full rank"):
        with pytest.raises(Exception):
            FastMCDEstimator().fit(X)


def test_fastmcd_rejects_negative_variance_without_publishing_validity(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "force.estimators.MinCovDet.fit",
        lambda self, X: SimpleNamespace(covariance_=np.diag([-1.0, 1.0])),
    )
    estimator = FastMCDEstimator()
    with pytest.raises(RuntimeError, match="invalid covariance"):
        estimator.fit(np.arange(20.0).reshape(10, 2))
    assert estimator.pair_status_codes_ is None


def test_exact_trimmed_compatibility_wrapper(data: np.ndarray) -> None:
    with pytest.warns(DeprecationWarning):
        compatibility = ExactTrimmedEstimator()
    expected = TrimmedPearsonExact(use_ter=True).fit(data)
    assert np.allclose(compatibility.fit(data), expected)


@pytest.mark.parametrize(
    "limits",
    [
        (-0.1, 0.1),
        (0.5, 0.0),
        (0.1,),
        (0.1, 0.1, 0.1),
        (True, 0.1),
        (np.nan, 0.1),
        None,
    ],
)
def test_winsorized_limits_validation(limits) -> None:
    with pytest.raises(ValueError):
        WinsorizedEstimator(limits)


@pytest.mark.parametrize("support_fraction", [True, 0.0, 1.1, np.nan, "bad"])
def test_fastmcd_support_fraction_validation(support_fraction) -> None:
    with pytest.raises(ValueError):
        FastMCDEstimator(support_fraction=support_fraction)
