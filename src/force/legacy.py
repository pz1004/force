"""Reproduction-only implementation of the originally committed FORCE code.

This module intentionally preserves the q05/q95 TER and median-centered
correlation used to generate the published benchmark values. New applications
must use :class:`force.ForceEstimator`.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Optional, Tuple

import numpy as np
from numba import njit, prange
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .core import _as_float_matrix


@njit(fastmath=True, cache=True)
def _legacy_p_square_kernel(
    data: NDArray[np.float64], probs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Original kernel, including its nonstandard sub-unit marker adjustment."""
    n_samples = len(data)
    if n_samples < 5:
        return np.quantile(data, probs)
    q = np.zeros((len(probs), 5), dtype=np.float64)
    positions = np.zeros((len(probs), 5), dtype=np.float64)
    desired = np.zeros((len(probs), 5), dtype=np.float64)
    increments = np.zeros((len(probs), 5), dtype=np.float64)
    initial = np.sort(data[:5])
    for p_idx in range(len(probs)):
        probability = probs[p_idx]
        q[p_idx, :] = initial
        positions[p_idx, :] = np.arange(1.0, 6.0)
        desired[p_idx, :] = np.array(
            [
                1.0,
                1.0 + 2.0 * probability,
                1.0 + 4.0 * probability,
                3.0 + 2.0 * probability,
                5.0,
            ]
        )
        increments[p_idx, :] = np.array(
            [
                0.0,
                probability / 2.0,
                probability,
                (1.0 + probability) / 2.0,
                1.0,
            ]
        )
    for sample in range(5, n_samples):
        value = data[sample]
        for p_idx in range(len(probs)):
            if value < q[p_idx, 0]:
                q[p_idx, 0] = value
                cell = 0
            elif value < q[p_idx, 1]:
                cell = 0
            elif value < q[p_idx, 2]:
                cell = 1
            elif value < q[p_idx, 3]:
                cell = 2
            elif value < q[p_idx, 4]:
                cell = 3
            else:
                q[p_idx, 4] = value
                cell = 3
            for marker in range(cell + 1, 5):
                positions[p_idx, marker] += 1.0
            for marker in range(5):
                desired[p_idx, marker] += increments[p_idx, marker]
            for marker in range(1, 4):
                delta = desired[p_idx, marker] - positions[p_idx, marker]
                direction = np.sign(delta)
                if (
                    direction > 0.0
                    and positions[p_idx, marker + 1] - positions[p_idx, marker] > 1.0
                ) or (
                    direction < 0.0
                    and positions[p_idx, marker - 1] - positions[p_idx, marker] < -1.0
                ):
                    proposed = q[p_idx, marker] + direction / (
                        positions[p_idx, marker + 1]
                        - positions[p_idx, marker - 1]
                    ) * (
                        (
                            positions[p_idx, marker]
                            - positions[p_idx, marker - 1]
                            + direction
                        )
                        * (q[p_idx, marker + 1] - q[p_idx, marker])
                        / (
                            positions[p_idx, marker + 1]
                            - positions[p_idx, marker]
                        )
                        + (
                            positions[p_idx, marker + 1]
                            - positions[p_idx, marker]
                            - direction
                        )
                        * (q[p_idx, marker] - q[p_idx, marker - 1])
                        / (
                            positions[p_idx, marker]
                            - positions[p_idx, marker - 1]
                        )
                    )
                    if q[p_idx, marker - 1] < proposed < q[p_idx, marker + 1]:
                        q[p_idx, marker] = proposed
                    else:
                        adjacent = marker + int(direction)
                        q[p_idx, marker] += direction * (
                            q[p_idx, adjacent] - q[p_idx, marker]
                        ) / (
                            positions[p_idx, adjacent]
                            - positions[p_idx, marker]
                        )
                    positions[p_idx, marker] += direction
    result = np.empty(len(probs), dtype=np.float64)
    for p_idx in range(len(probs)):
        result[p_idx] = q[p_idx, 2]
    return result


@njit(parallel=True, cache=True)
def _legacy_all_quantiles(
    X: NDArray[np.float64], probs: NDArray[np.float64]
) -> NDArray[np.float64]:
    output = np.empty((len(probs), X.shape[1]), dtype=np.float64)
    for feature in prange(X.shape[1]):
        output[:, feature] = _legacy_p_square_kernel(X[:, feature], probs)
    return output


@njit(parallel=True, fastmath=True, cache=True)
def _legacy_trimmed_corr(
    X: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    medians: NDArray[np.float64],
) -> NDArray[np.float64]:
    n_samples, n_features = X.shape
    correlation = np.eye(n_features, dtype=np.float64)
    for left in prange(n_features):
        for right in range(left + 1, n_features):
            sum_xx = 0.0
            sum_yy = 0.0
            sum_xy = 0.0
            count = 0
            for sample in range(n_samples):
                x = X[sample, left]
                y = X[sample, right]
                if (
                    lower[left] <= x <= upper[left]
                    and lower[right] <= y <= upper[right]
                ):
                    dx = x - medians[left]
                    dy = y - medians[right]
                    sum_xx += dx * dx
                    sum_yy += dy * dy
                    sum_xy += dx * dy
                    count += 1
            if count > 1 and sum_xx > 1e-12 and sum_yy > 1e-12:
                value = sum_xy / np.sqrt(sum_xx * sum_yy)
                correlation[left, right] = value
                correlation[right, left] = value
    return correlation


class LegacyForceEstimator:
    """Historical estimator retained solely for published-result reproduction."""

    def __init__(
        self,
        lambda_scale: float = 3.0,
        exact_cutover: int = 100,
        use_ter: bool = True,
    ):
        if (
            not isinstance(lambda_scale, Real)
            or isinstance(lambda_scale, (bool, np.bool_))
            or not np.isfinite(lambda_scale)
            or lambda_scale <= 0
        ):
            raise ValueError("lambda_scale must be finite and greater than zero.")
        if (
            not isinstance(exact_cutover, Integral)
            or isinstance(exact_cutover, bool)
            or exact_cutover < 5
        ):
            raise ValueError("exact_cutover must be an integer of at least 5.")
        if not isinstance(use_ter, (bool, np.bool_)):
            raise ValueError("use_ter must be a boolean.")
        self.lambda_scale = float(lambda_scale)
        self.exact_cutover = int(exact_cutover)
        self.use_ter = bool(use_ter)
        self.probs = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=np.float64)
        self._ter_norm = float(norm.ppf(0.95) - norm.ppf(0.05))
        self.quantiles: Optional[NDArray[np.float64]] = None
        self.thresholds: Optional[
            Tuple[NDArray[np.float64], NDArray[np.float64]]
        ] = None

    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_float_matrix(X)
        if matrix.shape[0] < self.exact_cutover:
            self.quantiles = np.quantile(matrix, self.probs, axis=0)
        else:
            self.quantiles = _legacy_all_quantiles(matrix, self.probs)
        q05, q25, q50, q75, q95 = self.quantiles
        scale = (q75 - q25) / 1.349
        scale[scale <= 1e-12] = 1.0
        denominator = scale * self._ter_norm
        if self.use_ter:
            ter = (q95 - q05) / np.where(
                denominator <= 1e-12, 1.0, denominator
            )
            ter[~np.isfinite(ter)] = 1.0
        else:
            ter = np.ones_like(q50)
        half_width = self.lambda_scale * ter * scale
        lower = q50 - half_width
        upper = q50 + half_width
        self.thresholds = (lower, upper)
        return _legacy_trimmed_corr(matrix, lower, upper, q50)

    @property
    def state_nbytes_(self) -> int:
        """Bytes retained by the reproduction estimator after fitting."""
        arrays = [self.quantiles]
        if self.thresholds is not None:
            arrays.extend(self.thresholds)
        return int(sum(array.nbytes for array in arrays if array is not None))
