"""Preserve baseline Pearson semantics while optimizing numerical preparation."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import stats

from force import PearsonEstimator, SpearmanEstimator, WinsorizedEstimator


@pytest.mark.parametrize(
    "scales",
    [
        [np.finfo(float).max / 2, np.finfo(float).max / 2],
        [np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)],
        [np.finfo(float).max / 2, np.nextafter(0.0, 1.0)],
    ],
    ids=["finite_endpoints", "subnormal", "unequal_extreme_units"],
)
def test_pearson_preserves_correlation_at_float64_range_limits(scales):
    # Integer multiples preserve the fixture even at minimum subnormal scale.
    # The first column spans -max to +max in the finite-endpoint cases.
    base = np.array([[-2.0, -1.0], [-1.0, -1.0], [0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
    result = PearsonEstimator().fit(base * scales)
    np.testing.assert_allclose(result, np.corrcoef(base, rowvar=False), atol=1e-14)


def test_pearson_extreme_constant_columns_remain_finite():
    X = np.column_stack(
        [
            np.arange(9.0),
            np.full(9, np.finfo(float).max),
            np.full(9, np.nextafter(0.0, 1.0)),
        ]
    )
    np.testing.assert_array_equal(PearsonEstimator().fit(X), np.eye(3))


def test_pearson_large_offset_with_first_row_outlier_preserves_small_correlation():
    X = np.column_stack(
        [1e15 + 4 * np.arange(9.0), 1e15 + np.array([0.0, -4, -3, -2, -1, 1, 2, 3, 4])]
    )
    X[0, 0] = -1e15
    # Decimal arithmetic is independent of the production origin/scaling path.
    with localcontext() as context:
        context.prec = 100
        x = [Decimal.from_float(float(value)) for value in X[:, 0]]
        y = [Decimal.from_float(float(value)) for value in X[:, 1]]
        mx, my = sum(x) / len(x), sum(y) / len(y)
        x, y = [value - mx for value in x], [value - my for value in y]
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = (sum(a * a for a in x) * sum(b * b for b in y)).sqrt()
        expected = float(numerator / denominator)
    for ordered in (X, X[::-1], np.roll(X, 3, axis=0)):
        actual = PearsonEstimator().fit(ordered)[0, 1]
        assert actual == pytest.approx(expected, rel=0, abs=2e-16)


@pytest.mark.parametrize(
    "estimator",
    [PearsonEstimator(), SpearmanEstimator(), WinsorizedEstimator(limits=(0.0, 0.0))],
)
def test_baselines_accept_readonly_strided_inputs_without_mutating_them(estimator):
    source = np.random.default_rng(83).normal(size=(40, 6))
    before = source.copy()
    source.setflags(write=False)
    X = source[::2, ::2]
    expected = (
        stats.spearmanr(X).statistic
        if isinstance(estimator, SpearmanEstimator)
        else np.corrcoef(X, rowvar=False)
    )
    np.testing.assert_allclose(estimator.fit(X), expected, atol=1e-14)
    np.testing.assert_array_equal(source, before)
