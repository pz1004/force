"""Comparative correlation estimators used by FORCE benchmarks."""

from __future__ import annotations

import abc
import warnings
from numbers import Real
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from sklearn.covariance import MinCovDet

from .core import _as_float_matrix, _covariance_to_correlation
from .trimmed_pearson import TrimmedPearsonExact


def _safe_correlation(X: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.asarray(np.corrcoef(X, rowvar=False), dtype=np.float64)
    correlation = np.atleast_2d(correlation)
    correlation[~np.isfinite(correlation)] = 0.0
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


class CorrelationEstimator(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        raise NotImplementedError


class PearsonEstimator(CorrelationEstimator):
    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        return _safe_correlation(_as_float_matrix(X))


class SpearmanEstimator(CorrelationEstimator):
    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_float_matrix(X)
        ranks = np.empty_like(matrix)
        for feature in range(matrix.shape[1]):
            ranks[:, feature] = stats.rankdata(matrix[:, feature], method="average")
        return _safe_correlation(ranks)


class WinsorizedEstimator(CorrelationEstimator):
    def __init__(self, limits: Sequence[float] = (0.05, 0.05)):
        if isinstance(limits, (str, bytes)) or not hasattr(limits, "__len__"):
            raise ValueError("limits must contain lower and upper fractions.")
        if len(limits) != 2:
            raise ValueError("limits must contain lower and upper fractions.")
        try:
            lower, upper = (float(value) for value in limits)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "winsorization limits must be finite numeric fractions."
            ) from exc
        if any(isinstance(value, (bool, np.bool_)) for value in limits):
            raise ValueError("winsorization limits must be numeric fractions.")
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("winsorization limits must be finite.")
        if not (0.0 <= lower < 0.5 and 0.0 <= upper < 0.5):
            raise ValueError("winsorization limits must lie in [0, 0.5).")
        self.limits = (lower, upper)

    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_float_matrix(X)
        winsorized = stats.mstats.winsorize(
            matrix, limits=self.limits, axis=0
        )
        return _safe_correlation(np.asarray(np.ma.getdata(winsorized)))


class FastMCDEstimator(CorrelationEstimator):
    def __init__(
        self,
        *,
        random_state: Optional[int] = 42,
        support_fraction: Optional[float] = None,
    ):
        if support_fraction is not None:
            if (
                not isinstance(support_fraction, Real)
                or isinstance(support_fraction, (bool, np.bool_))
            ):
                raise ValueError("support_fraction must be a numeric fraction.")
            if (
                not np.isfinite(support_fraction)
                or not 0.0 < support_fraction <= 1.0
            ):
                raise ValueError("support_fraction must lie in (0, 1].")
        self.random_state = random_state
        self.support_fraction = support_fraction

    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_float_matrix(X)
        model = MinCovDet(
            random_state=self.random_state,
            support_fraction=self.support_fraction,
        ).fit(matrix)
        covariance = np.asarray(model.covariance_, dtype=np.float64)
        try:
            return _covariance_to_correlation(covariance)
        except ValueError as exc:
            raise RuntimeError(
                "FastMCD produced an invalid covariance matrix."
            ) from exc


class ExactTrimmedEstimator(CorrelationEstimator):
    """Deprecated compatibility wrapper for exact TER-enabled trimming."""

    def __init__(self, lambda_scale: float = 3.0):
        warnings.warn(
            "ExactTrimmedEstimator is deprecated; use "
            "TrimmedPearsonExact(use_ter=True).",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(lambda_scale, (bool, np.bool_)):
            raise ValueError("lambda_scale must be a finite positive number.")
        self.lambda_scale = float(lambda_scale)
        self._delegate = TrimmedPearsonExact(
            lam=self.lambda_scale, use_ter=True
        )

    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        return self._delegate.fit(X)
