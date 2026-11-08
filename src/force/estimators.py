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
from ._moments import status_labels_from_codes
from ._quantiles import linear_quantiles
from .trimmed_pearson import TrimmedPearsonExact


def _safe_correlation(X: NDArray[np.float64], *, return_status: bool = False):
    """Pearson correlation with one float64 working matrix on ordinary inputs."""
    # Shift before scaling so a large common offset does not erase small spreads.
    # Column-contiguous workspace keeps the column reductions efficient.
    # Subsequent in-place operations never alter X.
    with np.errstate(over="ignore"):
        centered = np.subtract(X, X[0], order="F")
    scales = np.maximum(np.abs(centered.min(axis=0)), np.abs(centered.max(axis=0)))

    # Opposite finite endpoints can overflow the subtraction. Extended precision
    # is needed only for these columns, not for the whole input matrix.
    for feature in np.flatnonzero(~np.isfinite(scales)):
        column = X[:, feature].astype(np.longdouble)
        with np.errstate(over="ignore"):
            shifted = column - column[0]
        if not np.all(np.isfinite(shifted)):
            # Some platforms give longdouble the same range as float64.
            shifted = column / 2 - column[0] / 2
        scale = max(abs(shifted.min()), abs(shifted.max()))
        centered[:, feature] = shifted / scale
        scales[feature] = 1.0

    # Bounded coordinates prevent mean/squared-norm overflow and preserve
    # subnormal-unit inputs. Reductions avoid full-size absolute/square arrays.
    centered /= np.where(scales > 0, scales, 1.0)
    centered -= centered.mean(axis=0)
    norms = np.sqrt(np.einsum("ij,ij->j", centered, centered))
    centered /= np.where(norms > 0, norms, 1.0)
    correlation = centered.T @ centered
    if not np.all(np.isfinite(correlation)) or np.any(np.abs(correlation) > 1 + 1e-12):
        raise FloatingPointError(
            "Pearson normalization produced an invalid correlation."
        )
    np.clip(correlation, -1.0, 1.0, out=correlation)
    np.fill_diagonal(correlation, 1.0)
    if return_status:
        defined = norms > 0
        codes = np.where(defined[:, None] & defined[None, :], 0, 2).astype(np.int8)
        return correlation, codes
    return correlation


class CorrelationEstimator(abc.ABC):
    @property
    def pair_status_codes_(self):
        """Compact pair validity; a finite zero-filled output can be undefined."""
        codes = getattr(self, "_pair_status_codes", None)
        return None if codes is None else codes.copy()

    @property
    def pair_status_(self):
        codes = self.pair_status_codes_
        return None if codes is None else status_labels_from_codes(codes)

    def _fit_correlation(self, matrix):
        result, codes = _safe_correlation(matrix, return_status=True)
        self._pair_status_codes = codes
        return result

    @abc.abstractmethod
    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        raise NotImplementedError


class PearsonEstimator(CorrelationEstimator):
    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        return self._fit_correlation(_as_float_matrix(X))


class SpearmanEstimator(CorrelationEstimator):
    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_float_matrix(X)
        ranks = np.empty_like(matrix)
        for feature in range(matrix.shape[1]):
            ranks[:, feature] = stats.rankdata(matrix[:, feature], method="average")
        return self._fit_correlation(ranks)


class WinsorizedEstimator(CorrelationEstimator):
    """Equation 18 with an explicit empirical-quantile convention.

    ``linear`` clips at linearly interpolated sample quantiles, as used by the
    exact trimmed comparators. ``tail_count`` preserves the historical SciPy
    Winsorization rule based on integer tail counts. The two rules can differ
    for small samples and are recorded separately in benchmark provenance.
    """

    def __init__(
        self,
        limits: Sequence[float] = (0.05, 0.05),
        *,
        quantile_method: str = "linear",
    ):
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
        if not isinstance(quantile_method, str) or quantile_method not in (
            "linear",
            "tail_count",
        ):
            raise ValueError("quantile_method must be 'linear' or 'tail_count'.")
        self.limits = (lower, upper)
        self.quantile_method = quantile_method

    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_float_matrix(X)
        if self.quantile_method == "linear":
            lower, upper = linear_quantiles(
                matrix, [self.limits[0], 1.0 - self.limits[1]], axis=0
            )
            winsorized = np.clip(matrix, lower, upper)
        else:
            winsorized = np.asarray(
                np.ma.getdata(
                    stats.mstats.winsorize(matrix, limits=self.limits, axis=0)
                )
            )
        return self._fit_correlation(winsorized)


class FastMCDEstimator(CorrelationEstimator):
    def __init__(
        self,
        *,
        random_state: Optional[int] = 42,
        support_fraction: Optional[float] = None,
    ):
        if support_fraction is not None:
            if not isinstance(support_fraction, Real) or isinstance(
                support_fraction, (bool, np.bool_)
            ):
                raise ValueError("support_fraction must be a numeric fraction.")
            if not np.isfinite(support_fraction) or not 0.0 < support_fraction <= 1.0:
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
            if np.any(np.diag(covariance) < 0):
                raise ValueError("Covariance contains a negative variance.")
            correlation = _covariance_to_correlation(covariance)
        except ValueError as exc:
            raise RuntimeError(
                "FastMCD produced an invalid covariance matrix."
            ) from exc
        defined = np.diag(covariance) > 0
        self._pair_status_codes = np.where(
            defined[:, None] & defined[None, :], 0, 2
        ).astype(np.int8)
        return correlation


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
        self._delegate = TrimmedPearsonExact(lam=self.lambda_scale, use_ter=True)

    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        result = self._delegate.fit(X)
        self._pair_status_codes = self._delegate.pair_status_codes_
        return result
