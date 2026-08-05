"""Validation of benchmark estimators and compatibility wrappers."""

from __future__ import annotations

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


def test_fastmcd_does_not_silently_fallback() -> None:
    X = np.ones((10, 3))
    with pytest.warns(UserWarning, match="not full rank"):
        with pytest.raises(Exception):
            FastMCDEstimator().fit(X)


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
