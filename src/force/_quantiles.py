"""Persistent P² states shared by scalar, batch, and replayable estimators.

Input errors are detected before a chunk changes the state. Arithmetic errors
can occur after part of a chunk has been consumed; such a state is marked failed
and must be discarded. No failed state supplies successful quantile estimates.
"""

from __future__ import annotations

from math import frexp, ldexp
from numbers import Integral, Real

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray


P2_PROBABILITIES = np.array([0.01, 0.25, 0.50, 0.75, 0.99], dtype=np.float64)


def validate_exact_prefix(exact_prefix: int) -> int:
    """Validate the opt-in exact-readout prefix length (zero disables it)."""
    if (
        not isinstance(exact_prefix, Integral)
        or isinstance(exact_prefix, (bool, np.bool_))
        or (exact_prefix != 0 and exact_prefix < 5)
    ):
        raise ValueError("exact_prefix must be zero or an integer of at least 5.")
    return int(exact_prefix)


def _as_real_array(values: ArrayLike) -> NDArray[np.float64]:
    """Reject complex values, including NumPy complex scalars in object arrays."""
    array = np.asarray(values)
    if np.iscomplexobj(array) or (
        array.dtype.kind == "O"
        and any(
            isinstance(value, (complex, np.complexfloating)) for value in array.flat
        )
    ):
        raise ValueError("Complex observations are not supported.")
    return np.asarray(array, dtype=np.float64)


@njit(cache=True)
def _linear_interpolate(left, right, fraction):
    """Interpolate finite endpoints without forming an overflowing span."""
    if fraction == 0.0 or left == right:
        return left
    if fraction == 1.0:
        return right
    magnitude = max(abs(left), abs(right))
    if (left < 0.0 < right) or magnitude < np.finfo(np.float64).tiny:
        # Opposite endpoints may have an unrepresentable difference. Scaling
        # also avoids rounding tiny products separately before their addition.
        _, exponent = frexp(magnitude)
        a, b = ldexp(left, -exponent), ldexp(right, -exponent)
        return ldexp((1.0 - fraction) * a + fraction * b, exponent)
    difference = right - left
    if fraction < 0.5:
        return left + fraction * difference
    return right - (1.0 - fraction) * difference


@njit(cache=True)
def _interpolate_order_statistics(ordered, ranks):
    result = np.empty((len(ranks), ordered.shape[1]), dtype=np.float64)
    for quantile in range(len(ranks)):
        lower = int(np.floor(ranks[quantile]))
        upper = int(np.ceil(ranks[quantile]))
        fraction = ranks[quantile] - lower
        for feature in range(ordered.shape[1]):
            result[quantile, feature] = _linear_interpolate(
                ordered[lower, feature], ordered[upper, feature], fraction
            )
    return result


def linear_quantiles(values: ArrayLike, probabilities: ArrayLike, axis=0):
    """Exact linear sample quantiles with finite-safe interpolation.

    Supports one- or two-dimensional finite real inputs and scalar or array
    probabilities in [0, 1]. The quantile dimensions precede the unreduced input
    dimensions, as in ``numpy.quantile``. Partitioning uses an independent
    workspace and retains the required exact order statistics.
    """
    data = _as_real_array(values)
    probabilities = _as_real_array(probabilities)
    if data.ndim not in (1, 2):
        raise ValueError("Quantile input must be one- or two-dimensional.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Quantile observations must be finite.")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("Quantile probabilities must be finite and in [0, 1].")
    if axis is None:
        data = data.ravel()
        axis = 0
    if (
        not isinstance(axis, Integral)
        or isinstance(axis, (bool, np.bool_))
        or not -data.ndim <= axis < data.ndim
    ):
        raise ValueError("axis must identify an input dimension or be None.")
    data = np.moveaxis(data, int(axis), 0)
    if len(data) == 0:
        raise ValueError("Quantile input must contain at least one observation.")
    shape = probabilities.shape + data.shape[1:]
    ranks = probabilities.ravel() * (len(data) - 1)
    indices = np.unique(np.concatenate((np.floor(ranks), np.ceil(ranks)))).astype(
        np.intp
    )
    ordered = np.partition(data, indices, axis=0).reshape(len(data), -1)
    result = _interpolate_order_statistics(ordered, ranks).reshape(shape)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(
            "Exact quantile interpolation produced a nonfinite value."
        )
    return result[()] if result.ndim == 0 else result


def p2_desired_positions(
    n_observations: int, probability: float
) -> NDArray[np.float64]:
    """Standard Jain–Chlamtac desired positions after ``n_observations``."""
    if (
        not isinstance(n_observations, Integral)
        or isinstance(n_observations, (bool, np.bool_))
        or n_observations < 5
    ):
        raise ValueError(
            "P² marker positions require an integer count of at least five."
        )
    if (
        not isinstance(probability, Real)
        or isinstance(probability, (bool, np.bool_))
        or not np.isfinite(probability)
        or not 0.0 < probability < 1.0
    ):
        raise ValueError("probability must be finite and strictly between 0 and 1.")
    span = float(n_observations - 1)
    probability = float(probability)
    return np.array(
        [
            1.0,
            1.0 + span * probability / 2.0,
            1.0 + span * probability,
            1.0 + span * (1.0 + probability) / 2.0,
            float(n_observations),
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def _p2_transition(heights, positions, desired, increments, value):
    """Consume one value into an initialized five-marker state, in place."""
    if value < heights[0]:
        heights[0] = value
        cell = 0
    elif value < heights[1]:
        cell = 0
    elif value < heights[2]:
        cell = 1
    elif value < heights[3]:
        cell = 2
    elif value < heights[4]:
        cell = 3
    else:
        heights[4] = value
        cell = 3
    for marker in range(cell + 1, 5):
        positions[marker] += 1.0
    for marker in range(5):
        desired[marker] += increments[marker]

    for marker in range(1, 4):
        delta = desired[marker] - positions[marker]
        direction = 1.0 if delta >= 1.0 else (-1.0 if delta <= -1.0 else 0.0)
        if direction == 0.0:
            continue
        if direction > 0.0 and positions[marker + 1] - positions[marker] <= 1.0:
            continue
        if direction < 0.0 and positions[marker] - positions[marker - 1] <= 1.0:
            continue

        proposed = heights[marker] + direction / (
            positions[marker + 1] - positions[marker - 1]
        ) * (
            (positions[marker] - positions[marker - 1] + direction)
            * (heights[marker + 1] - heights[marker])
            / (positions[marker + 1] - positions[marker])
            + (positions[marker + 1] - positions[marker] - direction)
            * (heights[marker] - heights[marker - 1])
            / (positions[marker] - positions[marker - 1])
        )
        accept_parabolic = heights[marker - 1] < proposed < heights[marker + 1]
        if not np.isfinite(proposed):
            # Re-evaluate the same candidate in binary-scaled coordinates.
            # Test its ordering before restoring the units: a candidate outside
            # finite neighboring markers selects Equation 5 even if that
            # candidate itself would exceed float64's range.
            _, exponent = frexp(
                max(
                    abs(heights[marker - 1]),
                    abs(heights[marker]),
                    abs(heights[marker + 1]),
                )
            )
            left = ldexp(heights[marker - 1], -exponent)
            middle = ldexp(heights[marker], -exponent)
            right = ldexp(heights[marker + 1], -exponent)
            lower_gap = positions[marker] - positions[marker - 1]
            upper_gap = positions[marker + 1] - positions[marker]
            scaled = middle + direction / (lower_gap + upper_gap) * (
                (lower_gap + direction) * (right - middle) / upper_gap
                + (upper_gap - direction) * (middle - left) / lower_gap
            )
            if not np.isfinite(scaled):
                raise FloatingPointError(
                    "P² interpolation exceeded the supported arithmetic range."
                )
            accept_parabolic = left < scaled < right
            if accept_parabolic:
                proposed = ldexp(scaled, exponent)
        if accept_parabolic:
            if not np.isfinite(proposed):
                raise FloatingPointError(
                    "P² interpolation exceeded the supported arithmetic range."
                )
            heights[marker] = proposed
        else:
            adjacent = marker + int(direction)
            linear = heights[marker] + direction * (
                heights[adjacent] - heights[marker]
            ) / (positions[adjacent] - positions[marker])
            if not np.isfinite(linear):
                # The linear fallback is a convex interpolation of neighbors.
                fraction = direction / (positions[adjacent] - positions[marker])
                if adjacent < marker:
                    linear = _linear_interpolate(
                        heights[adjacent], heights[marker], 1.0 - fraction
                    )
                else:
                    linear = _linear_interpolate(
                        heights[marker], heights[adjacent], fraction
                    )
            if not np.isfinite(linear):
                raise FloatingPointError(
                    "P² interpolation exceeded the supported arithmetic range."
                )
            heights[marker] = linear
        positions[marker] += direction


@njit(cache=True, nogil=True)
def _p2_update_batch(
    data, count, initial, heights, positions, desired, increments, probabilities
):
    for row in range(data.shape[0]):
        if count >= 2**53:
            raise FloatingPointError(
                "P² marker ranks are no longer exactly representable."
            )
        if count < 5:
            initial[count, :] = data[row, :]
            count += 1
            if count == 5:
                for feature in range(data.shape[1]):
                    ordered = np.sort(initial[:, feature])
                    for quantile in range(len(probabilities)):
                        probability = probabilities[quantile]
                        heights[quantile, feature, :] = ordered
                        for marker in range(5):
                            positions[quantile, feature, marker] = marker + 1.0
                        desired[quantile, feature, 0] = 1.0
                        desired[quantile, feature, 1] = 1.0 + 2.0 * probability
                        desired[quantile, feature, 2] = 1.0 + 4.0 * probability
                        desired[quantile, feature, 3] = 3.0 + 2.0 * probability
                        desired[quantile, feature, 4] = 5.0
            continue
        for feature in range(data.shape[1]):
            for quantile in range(len(probabilities)):
                _p2_transition(
                    heights[quantile, feature],
                    positions[quantile, feature],
                    desired[quantile, feature],
                    increments[quantile],
                    data[row, feature],
                )
        count += 1
    return count


class P2State:
    """Marginal P² state with an optional exact-readout prefix.

    ``exact_prefix > 0`` returns exact linear quantiles through that many rows.
    Standard P² processes every row in parallel, including the prefix. After
    the next row, the prefix buffer is released and estimates come from P².
    The paper does not specify a marker handoff rule; this policy preserves its
    standard P² recurrence rather than inventing a new initialization. It does
    not improve subsequent P² markers, and the readout switch may be discontinuous.

    State is O(p) by default and O(p * exact_prefix) during the optional prefix.
    """

    def __init__(
        self,
        n_features: int,
        probabilities: ArrayLike = P2_PROBABILITIES,
        *,
        exact_prefix: int = 0,
    ):
        if (
            not isinstance(n_features, Integral)
            or isinstance(n_features, (bool, np.bool_))
            or n_features < 1
        ):
            raise ValueError("n_features must be a positive integer.")
        probs = _as_real_array(probabilities)
        if probs.ndim != 1 or probs.size == 0:
            raise ValueError("probabilities must be a nonempty one-dimensional array.")
        if not np.all(np.isfinite(probs)) or np.any((probs <= 0.0) | (probs >= 1.0)):
            raise ValueError(
                "probabilities must be finite and strictly between 0 and 1."
            )
        self.n_features = int(n_features)
        self.probabilities = probs.copy()
        self._exact_prefix = validate_exact_prefix(exact_prefix)
        self._prefix = (
            np.empty((self._exact_prefix, self.n_features), dtype=np.float64)
            if self._exact_prefix
            else None
        )
        self.count = 0
        self._failed = False
        self._initial = np.empty((5, self.n_features), dtype=np.float64)
        shape = (len(probs), self.n_features, 5)
        self._heights = np.zeros(shape, dtype=np.float64)
        self._positions = np.zeros(shape, dtype=np.float64)
        self._desired = np.zeros(shape, dtype=np.float64)
        self._increments = np.column_stack(
            (
                np.zeros_like(probs),
                probs / 2.0,
                probs,
                (1.0 + probs) / 2.0,
                np.ones_like(probs),
            )
        )

    def _ensure_valid(self) -> None:
        if self._failed:
            raise RuntimeError(
                "P² state failed during arithmetic; discard and recreate it."
            )

    @property
    def exact_prefix(self) -> int:
        return self._exact_prefix

    def update_batch(self, X: ArrayLike) -> "P2State":
        """Consume a finite matrix; chunk boundaries never reset the trackers."""
        self._ensure_valid()
        data = _as_real_array(X)
        if data.ndim != 2 or data.shape[1] != self.n_features:
            raise ValueError("X must be a matrix with n_features columns.")
        if not np.all(np.isfinite(data)):
            raise ValueError("P² observations must be finite.")
        # No intermediate readout is observable within a batch call. If this
        # call crosses the handoff, no further exact-prefix copy is needed.
        if self._prefix is not None and self.count + len(data) <= self.exact_prefix:
            self._prefix[self.count : self.count + len(data)] = data
        try:
            self.count = int(
                _p2_update_batch(
                    data,
                    self.count,
                    self._initial,
                    self._heights,
                    self._positions,
                    self._desired,
                    self._increments,
                    self.probabilities,
                )
            )
        except FloatingPointError:
            self._failed = True
            raise
        if self.count > self.exact_prefix:
            self._prefix = None
        return self

    @property
    def quantiles(self) -> NDArray[np.float64]:
        """Copy of estimates in (probabilities, features) order after five rows."""
        self._ensure_valid()
        if self.count < 5:
            raise ValueError("P² quantiles require at least five observations.")
        if self._prefix is not None:
            return linear_quantiles(
                self._prefix[: self.count], self.probabilities, axis=0
            )
        return self._heights[:, :, 2].copy()

    @property
    def state_nbytes_(self) -> int:
        return (0 if self._prefix is None else self._prefix.nbytes) + sum(
            array.nbytes
            for array in (
                self.probabilities,
                self._initial,
                self._heights,
                self._positions,
                self._desired,
                self._increments,
            )
        )


class P2Quantile:
    """Scalar P² tracker, optionally reading exact quantiles through a prefix.

    ``exact_prefix`` uses the same explicit handoff policy as ``P2State``.
    Marker accessors always expose the background P² state, even while the
    public ``value`` uses the exact prefix.
    """

    def __init__(self, probability: float, *, exact_prefix: int = 0):
        p2_desired_positions(5, probability)
        self.probability = float(probability)
        self._state = P2State(1, [self.probability], exact_prefix=exact_prefix)

    @property
    def exact_prefix(self) -> int:
        return self._state.exact_prefix

    @property
    def count(self) -> int:
        return self._state.count

    def update(self, value: float) -> "P2Quantile":
        if not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("P² observations must be finite real numbers.")
        self._state.update_batch(np.array([[float(value)]], dtype=np.float64))
        return self

    @property
    def value(self) -> float:
        self._state._ensure_valid()
        if self.count == 0:
            return float("nan")
        if self.count < 5:
            return float(
                linear_quantiles(
                    self._state._initial[: self.count, 0], self.probability
                )
            )
        if self._state._prefix is None:
            return float(self._state._heights[0, 0, 2])
        return float(self._state.quantiles[0, 0])

    @property
    def marker_heights(self) -> NDArray[np.float64]:
        self._state._ensure_valid()
        if self.count < 5:
            return np.sort(self._state._initial[: self.count, 0])
        return self._state._heights[0, 0].copy()

    @property
    def marker_positions(self) -> NDArray[np.float64]:
        self._state._ensure_valid()
        if self.count < 5:
            return np.arange(1.0, self.count + 1.0)
        return self._state._positions[0, 0].copy()

    @property
    def desired_positions(self) -> NDArray[np.float64]:
        self._state._ensure_valid()
        if self.count < 5:
            return np.empty(0, dtype=np.float64)
        return self._state._desired[0, 0].copy()

    @property
    def state_nbytes_(self) -> int:
        return self._state.state_nbytes_
