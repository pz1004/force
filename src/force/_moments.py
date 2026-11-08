"""Persistent pair-specific moments for the accepted-observation functional.

Each pair has its own origin, means, square-sums, and co-moment. Coordinates are
translated by the first accepted observation and scaled by adaptive powers of
two. Scale changes rescale the existing state, preserving the accepted sample
and Pearson functional while avoiding squared units outside float64's range.
Arithmetic failures invalidate the state. Dense diagnostics are allocated only
when their individual accessors are called.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np
from numba import njit
from numpy.typing import ArrayLike

from ._numeric import normalize_correlation
from ._quantiles import _as_real_array


STATUS_LABELS = np.array(["ok", "insufficient_samples", "zero_variance"])
STATUS_LABELS.setflags(write=False)


def status_labels_from_codes(codes: ArrayLike):
    """Allocate readable labels only when explicitly requested."""
    values = np.asarray(codes)
    if not np.issubdtype(values.dtype, np.integer) or np.any(
        (values < 0) | (values >= len(STATUS_LABELS))
    ):
        raise ValueError("Pair status codes must be integers from zero to two.")
    return STATUS_LABELS[values]


@njit(cache=True)
def _centered_coordinate(value, origin, exponent):
    """Return a centered value with a scale at least as large as exponent."""
    difference = value - origin
    if np.isfinite(difference):
        if difference == 0.0:
            return 0.0, exponent
        _, required = math.frexp(difference)
        updated = max(exponent, required)
        return math.ldexp(difference, -updated), updated
    # Opposite finite endpoints can have a difference larger than float64.
    # Scale before subtraction; binary scaling retains the endpoints' bits.
    _, value_exponent = math.frexp(value)
    _, origin_exponent = math.frexp(origin)
    updated = max(exponent, max(value_exponent, origin_exponent) + 1)
    centered = math.ldexp(value, -updated) - math.ldexp(origin, -updated)
    return centered, updated


@njit(cache=True, nogil=True)
def _update_pair_moments(
    data, lower, upper, counts, origins, exponents, means, moments
):
    pair = 0
    for left in range(data.shape[1]):
        for right in range(left, data.shape[1]):
            for row in range(data.shape[0]):
                x, y = data[row, left], data[row, right]
                if not (
                    lower[left] <= x <= upper[left]
                    and lower[right] <= y <= upper[right]
                ):
                    continue
                count = counts[pair]
                if count >= 2**53:
                    raise FloatingPointError(
                        "Pair count exceeds the exactly representable update range."
                    )
                if count == 0:
                    origins[0, pair], origins[1, pair] = x, y
                    counts[pair] = 1
                    continue
                centered_x, exponent_x = _centered_coordinate(
                    x, origins[0, pair], exponents[0, pair]
                )
                centered_y, exponent_y = _centered_coordinate(
                    y, origins[1, pair], exponents[1, pair]
                )
                shift_x = exponents[0, pair] - exponent_x
                shift_y = exponents[1, pair] - exponent_y
                previous_mean_x = math.ldexp(means[0, pair], shift_x)
                previous_mean_y = math.ldexp(means[1, pair], shift_y)
                previous_xx = math.ldexp(moments[0, pair], 2 * shift_x)
                previous_yy = math.ldexp(moments[1, pair], 2 * shift_y)
                previous_xy = math.ldexp(moments[2, pair], shift_x + shift_y)
                delta_x = centered_x - previous_mean_x
                delta_y = centered_y - previous_mean_y
                new_count = count + 1
                mean_x = previous_mean_x + delta_x / new_count
                mean_y = previous_mean_y + delta_y / new_count
                sum_xx = previous_xx + delta_x * (centered_x - mean_x)
                sum_yy = previous_yy + delta_y * (centered_y - mean_y)
                sum_xy = previous_xy + delta_x * (centered_y - mean_y)
                if not (
                    np.isfinite(mean_x)
                    and np.isfinite(mean_y)
                    and np.isfinite(sum_xx)
                    and np.isfinite(sum_yy)
                    and np.isfinite(sum_xy)
                ):
                    raise FloatingPointError(
                        "Pair moments exceeded the supported arithmetic range."
                    )
                if sum_xx < 0.0 or sum_yy < 0.0:
                    raise FloatingPointError(
                        "Pair variance became negative through arithmetic error."
                    )
                if (sum_xx == 0.0 and (previous_xx > 0.0 or centered_x != 0.0)) or (
                    sum_yy == 0.0 and (previous_yy > 0.0 or centered_y != 0.0)
                ):
                    raise FloatingPointError("A positive pair variance was lost.")
                exponents[0, pair], exponents[1, pair] = exponent_x, exponent_y
                means[0, pair], means[1, pair] = mean_x, mean_y
                moments[0, pair], moments[1, pair], moments[2, pair] = (
                    sum_xx,
                    sum_yy,
                    sum_xy,
                )
                counts[pair] = new_count
            pair += 1


@njit(cache=True)
def _pair_status(count, xx, yy, xy):
    if not (np.isfinite(xx) and np.isfinite(yy) and np.isfinite(xy)):
        raise FloatingPointError("Nonfinite pair moments cannot be finalized.")
    if xx < 0.0 or yy < 0.0:
        raise FloatingPointError("Negative pair variance cannot be finalized.")
    if count < 2:
        return 1
    return 0 if xx > 0.0 and yy > 0.0 else 2


@njit(cache=True, nogil=True)
def _pair_correlation(n_features, counts, moments):
    correlation = np.eye(n_features, dtype=np.float64)
    pair = 0
    for left in range(n_features):
        for right in range(left, n_features):
            xx, yy, xy = moments[0, pair], moments[1, pair], moments[2, pair]
            code = _pair_status(counts[pair], xx, yy, xy)
            if left != right:
                value = 0.0
                if code == 0:
                    value = normalize_correlation(xx, yy, xy)
                    if abs(value) > 1.0 + 1e-12:
                        raise FloatingPointError(
                            "Pair normalization produced an invalid correlation."
                        )
                    value = min(1.0, max(-1.0, value))
                correlation[left, right] = correlation[right, left] = value
            pair += 1
    return correlation


@njit(cache=True, nogil=True)
def _pair_counts(n_features, counts):
    result = np.zeros((n_features, n_features), dtype=np.int64)
    pair = 0
    for left in range(n_features):
        for right in range(left, n_features):
            result[left, right] = result[right, left] = counts[pair]
            pair += 1
    return result


@njit(cache=True, nogil=True)
def _pair_status_codes(n_features, counts, moments):
    result = np.empty((n_features, n_features), dtype=np.int8)
    pair = 0
    for left in range(n_features):
        for right in range(left, n_features):
            code = _pair_status(
                counts[pair], moments[0, pair], moments[1, pair], moments[2, pair]
            )
            result[left, right] = result[right, left] = code
            pair += 1
    return result


class PairMoments:
    """Upper-triangular accepted moments with O(p²) persistent storage."""

    def __init__(self, n_features: int):
        if (
            not isinstance(n_features, Integral)
            or isinstance(n_features, (bool, np.bool_))
            or n_features < 1
        ):
            raise ValueError("n_features must be a positive integer.")
        self.n_features = int(n_features)
        self.count = 0
        self._failed = False
        n_pairs = self.n_features * (self.n_features + 1) // 2
        self._counts = np.zeros(n_pairs, dtype=np.int64)
        self._origins = np.zeros((2, n_pairs), dtype=np.float64)
        # The smallest nonzero float has frexp exponent -1073, so this sentinel
        # changes on the first nonzero centered observation.
        self._exponents = np.full((2, n_pairs), -1074, dtype=np.int64)
        self._means = np.zeros((2, n_pairs), dtype=np.float64)
        self._moments = np.zeros((3, n_pairs), dtype=np.float64)

    def _ensure_valid(self) -> None:
        if self._failed:
            raise RuntimeError(
                "Pair-moment state failed during arithmetic; discard and recreate it."
            )

    def update_batch(
        self, X: ArrayLike, lower: ArrayLike, upper: ArrayLike
    ) -> "PairMoments":
        """Accept this chunk using the supplied inclusive marginal bounds."""
        self._ensure_valid()
        data = _as_real_array(X)
        lo, hi = _as_real_array(lower), _as_real_array(upper)
        if data.ndim != 2 or data.shape[1] != self.n_features:
            raise ValueError("X must be a matrix with n_features columns.")
        if lo.shape != (self.n_features,) or hi.shape != (self.n_features,):
            raise ValueError("Bounds must have one entry per feature.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Pair observations must be finite.")
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError("Pair acceptance bounds must be finite.")
        if np.any(lo > hi):
            raise ValueError("Each lower bound must be at most its upper bound.")
        try:
            _update_pair_moments(
                data,
                lo,
                hi,
                self._counts,
                self._origins,
                self._exponents,
                self._means,
                self._moments,
            )
        except FloatingPointError:
            self._failed = True
            raise
        self.count += data.shape[0]
        return self

    def correlation(self):
        """Allocate the correlation alone, with a unit diagonal convention."""
        self._ensure_valid()
        try:
            return _pair_correlation(self.n_features, self._counts, self._moments)
        except FloatingPointError:
            self._failed = True
            raise

    def accepted_counts(self):
        """Allocate symmetric counts; diagonal entries are marginal counts."""
        self._ensure_valid()
        return _pair_counts(self.n_features, self._counts)

    def status_codes(self):
        """Allocate compact int8 statuses, with meaningful diagonal statuses."""
        self._ensure_valid()
        try:
            return _pair_status_codes(self.n_features, self._counts, self._moments)
        except FloatingPointError:
            self._failed = True
            raise

    def status_labels(self):
        """Allocate readable statuses on explicit request."""
        return status_labels_from_codes(self.status_codes())

    def finalize(self):
        """Compatibility tuple of freshly allocated correlation, counts, labels."""
        return self.correlation(), self.accepted_counts(), self.status_labels()

    @property
    def state_nbytes_(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self._counts,
                self._origins,
                self._exponents,
                self._means,
                self._moments,
            )
        )
