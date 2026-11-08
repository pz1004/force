"""FORCE's final-bound, pairwise trimmed estimator (Algorithm 1)."""

from __future__ import annotations

import hashlib
import warnings
from math import frexp, ldexp
from numbers import Integral, Real
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from numba import njit

from ._input import DEFAULT_CHUNK_SIZE, validate_chunk_size, validated_chunks
from ._moments import PairMoments, status_labels_from_codes
from ._numeric import normalize_correlation
from ._quantiles import P2_PROBABILITIES, P2Quantile, P2State, p2_desired_positions
from ._quantiles import _as_real_array, linear_quantiles, validate_exact_prefix


def _p_square_kernel(data, probs):
    """Compatibility entry point backed by the shared compiled P² transition."""
    data = _as_real_array(data)
    if data.ndim != 1 or not np.all(np.isfinite(data)):
        raise ValueError("P² input must be a finite one-dimensional stream.")
    if data.size < 5:
        return linear_quantiles(data, probs)
    state = P2State(1, probs)
    state.update_batch(data.reshape(-1, 1))
    return state.quantiles[:, 0]


def _compute_all_quantiles_numba(X, probs, *, exact_prefix=0):
    state = P2State(X.shape[1], probs, exact_prefix=exact_prefix)
    for start in range(0, len(X), DEFAULT_CHUNK_SIZE):
        state.update_batch(X[start : start + DEFAULT_CHUNK_SIZE])
    return state.quantiles


def _compute_trimmed_corr_numba(X, lower_bounds, upper_bounds):
    """Compatibility entry point for shared compiled pair-moment updates."""
    state = PairMoments(X.shape[1])
    for start in range(0, len(X), DEFAULT_CHUNK_SIZE):
        state.update_batch(
            X[start : start + DEFAULT_CHUNK_SIZE], lower_bounds, upper_bounds
        )
    return state.correlation()


def _as_float_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """Validate estimator input and return a contiguous float64 matrix."""
    try:
        matrix = _as_real_array(X)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Input X must be a numeric two-dimensional array.") from exc
    if matrix.ndim != 2:
        raise ValueError("Input X must be a two-dimensional array.")
    if matrix.shape[0] < 5:
        raise ValueError("Input X must have at least 5 samples.")
    if matrix.shape[1] < 2:
        raise ValueError("Input X must have at least 2 features.")
    for start in range(0, matrix.shape[0], DEFAULT_CHUNK_SIZE):
        if not np.all(np.isfinite(matrix[start : start + DEFAULT_CHUNK_SIZE])):
            raise ValueError("Input X must contain only finite values.")
    return np.ascontiguousarray(matrix)


def _validate_force_parameters(
    lambda_scale: float,
    exact_cutover: int,
    use_ter: bool,
    ter_max: Optional[float],
    epsilon: float,
) -> None:
    if (
        not isinstance(lambda_scale, Real)
        or isinstance(lambda_scale, (bool, np.bool_))
        or not np.isfinite(lambda_scale)
    ):
        raise ValueError("lambda_scale must be finite.")
    if float(lambda_scale) <= 0.0:
        raise ValueError("lambda_scale must be greater than zero.")
    if (
        not isinstance(exact_cutover, Integral)
        or isinstance(exact_cutover, bool)
        or exact_cutover < 5
    ):
        raise ValueError("exact_cutover must be an integer of at least 5.")
    if not isinstance(use_ter, (bool, np.bool_)):
        raise ValueError("use_ter must be a boolean.")
    if ter_max is not None:
        if (
            not isinstance(ter_max, Real)
            or isinstance(ter_max, (bool, np.bool_))
            or not np.isfinite(ter_max)
        ):
            raise ValueError("ter_max must be finite or None.")
        if float(ter_max) < 1.0:
            raise ValueError("ter_max must be at least 1.")
    if (
        not isinstance(epsilon, Real)
        or isinstance(epsilon, (bool, np.bool_))
        or not np.isfinite(epsilon)
    ):
        raise ValueError("epsilon must be finite.")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be greater than zero.")


@njit(cache=True, nogil=True)
def _normalize_covariance(matrix):
    correlation = np.eye(matrix.shape[0], dtype=np.float64)
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            value = 0.0
            if matrix[left, left] > 0 and matrix[right, right] > 0:
                value = normalize_correlation(
                    matrix[left, left], matrix[right, right], matrix[left, right]
                )
                if not np.isfinite(value) or abs(value) > 1.0 + 1e-12:
                    raise FloatingPointError(
                        "Covariance normalization produced an invalid correlation."
                    )
                value = min(1.0, max(-1.0, value))
            correlation[left, right] = correlation[right, left] = value
    return correlation


def _covariance_to_correlation(covariance: ArrayLike) -> NDArray[np.float64]:
    """Normalize without multiplying variances or silently hiding overflow."""
    matrix = _as_real_array(covariance)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must contain only finite values.")
    # Preserve symmetric entries exactly, including subnormal variances that
    # would vanish if each operand were halved before summing.
    matrix = np.where(matrix == matrix.T, matrix, 0.5 * matrix + 0.5 * matrix.T)
    return _normalize_covariance(matrix)


def _distance_parts(left, right):
    """Return abs(left-right) as mantissa/exponent without range overflow."""
    largest = max(abs(left), abs(right))
    if largest == 0:
        return 0.0, 0
    _, exponent = frexp(largest)
    mantissa, adjustment = frexp(abs(ldexp(left, -exponent) - ldexp(right, -exponent)))
    return mantissa, exponent + adjustment


def _parts_less(left, right):
    if left[0] == 0 or right[0] == 0:
        return left[0] < right[0]
    return (left[1], left[0]) < (right[1], right[0])


def _positive_sum_parts(left, right):
    exponent = max(left[1], right[1])
    mantissa, adjustment = frexp(
        ldexp(left[0], left[1] - exponent) + ldexp(right[0], right[1] - exponent)
    )
    return mantissa, exponent + adjustment


def _finite_from_parts(mantissa, exponent):
    try:
        value = ldexp(mantissa, exponent)
    except OverflowError as exc:
        raise FloatingPointError("FORCE bounds exceeded floating-point range.") from exc
    if not np.isfinite(value):
        raise FloatingPointError("FORCE bounds exceeded floating-point range.")
    return value


def _finite_positive_product(*values):
    if any(value == 0 for value in values):
        return 0.0
    mantissa, exponent = 1.0, 0
    for value in values:
        part, power = frexp(value)
        mantissa *= part
        exponent += power
    return _finite_from_parts(mantissa, exponent)


def _compute_bounds_from_quantiles(
    quantiles, *, lambda_scale, use_ter, ter_max, epsilon
):
    """Equations 6–11; fail explicitly on nonrepresentable bound arithmetic."""
    quantiles = _as_real_array(quantiles)
    if quantiles.ndim != 2 or quantiles.shape[0] != 5:
        raise ValueError("quantiles must have shape (5, n_features).")
    if not np.all(np.isfinite(quantiles)):
        raise ValueError("quantiles must contain only finite values.")
    q01, q25, q50, q75, q99 = quantiles
    location = q50.copy()
    scale = np.empty_like(location)
    ter = np.ones_like(location)
    lower, upper = np.empty_like(location), np.empty_like(location)
    crossed = q75 < q25
    epsilon_parts = frexp(epsilon)
    degenerate = np.array(
        [
            _parts_less(_distance_parts(median, tail), epsilon_parts)
            for median, tail in zip(q50, q01)
        ],
        dtype=bool,
    )
    if np.any(crossed):
        warnings.warn(
            "Non-monotone P² quartiles produced a negative IQR; affected scales "
            "were collapsed to zero.",
            RuntimeWarning,
            stacklevel=2,
        )
    if use_ter and np.any(degenerate):
        warnings.warn(
            "TER denominator is below epsilon for one or more features; "
            "TER was set to 1 for those features.",
            RuntimeWarning,
            stacklevel=2,
        )
    for feature in range(len(location)):
        iqr, exponent = _distance_parts(q75[feature], q25[feature])
        scale[feature] = (
            0.0 if crossed[feature] else _finite_from_parts(iqr / 1.349, exponent)
        )
        if use_ter and not degenerate[feature]:
            numerator = _distance_parts(q99[feature], q50[feature])
            denominator = _positive_sum_parts(
                _distance_parts(q50[feature], q01[feature]), epsilon_parts
            )
            mantissa, adjustment = frexp(numerator[0] / denominator[0])
            ratio = mantissa, numerator[1] - denominator[1] + adjustment
            if _parts_less(ratio, frexp(1.0)):
                ter[feature] = 1.0
            elif ter_max is not None and _parts_less(frexp(float(ter_max)), ratio):
                ter[feature] = float(ter_max)
            else:
                ter[feature] = _finite_from_parts(*ratio)
        width = _finite_positive_product(
            float(lambda_scale), ter[feature], scale[feature]
        )
        with np.errstate(over="raise", invalid="raise"):
            lower[feature] = location[feature] - width
            upper[feature] = location[feature] + width
    return location, scale, ter, lower, upper


def _bound_status(quantiles, use_ter, epsilon):
    """Separate diagnostic flags: no reordering of independent quantiles."""
    return {
        "quantile_crossings": quantiles[1:] < quantiles[:-1],
        "any_quantile_crossing": np.any(quantiles[1:] < quantiles[:-1], axis=0),
        "crossed_quartiles": quantiles[3] < quantiles[1],
        "zero_iqr": quantiles[3] == quantiles[1],
        "degenerate_lower_tail": (
            np.array(
                [
                    _parts_less(_distance_parts(m, q), frexp(epsilon))
                    for m, q in zip(quantiles[2], quantiles[0])
                ],
                dtype=bool,
            )
            if use_ter
            else np.zeros(quantiles.shape[1], dtype=bool)
        ),
    }


class ForceEstimator:
    """Final-bound FORCE, using an array or two identical source traversals.

    ``fit`` returns a correlation matrix. ``fit_replayable`` has the same
    mathematical target and uses O(p² + (chunk_size + exact_prefix)*p)
    auxiliary memory, excluding source-owned storage.

    Array ``fit`` uses exact linear quantiles only when N < ``exact_cutover``;
    otherwise it uses P². This small-batch fallback is separate from the optional
    ``exact_prefix`` readout policy: exact quantiles through N <= exact_prefix,
    then standard P², which has processed every prefix row in the background.
    The paper does not define the handoff; no exact-marker reseeding is assumed.
    The switch may be discontinuous; later estimates equal pure P². Enable only
    one optional mode at a time. Defaults select the paper's experimental pure-P²
    mode. Repeated prefix readouts in the scalar API repeat exact computation.
    """

    def __init__(
        self,
        lambda_scale=3.0,
        exact_cutover=5,
        use_ter=True,
        ter_max=None,
        epsilon=1e-10,
        *,
        exact_prefix=0,
    ):
        _validate_force_parameters(
            lambda_scale, exact_cutover, use_ter, ter_max, epsilon
        )
        exact_prefix = validate_exact_prefix(exact_prefix)
        if exact_prefix and exact_cutover != 5:
            raise ValueError("exact_prefix requires exact_cutover=5.")
        self.lambda_scale = float(lambda_scale)
        self.exact_cutover = int(exact_cutover)
        self.exact_prefix = exact_prefix
        self.use_ter = bool(use_ter)
        self.ter_max = None if ter_max is None else float(ter_max)
        self.epsilon = float(epsilon)
        self.probs = P2_PROBABILITIES.copy()
        self.quantiles = self.thresholds = None
        self.location_ = self.scale_ = self.ter_ = None
        self.accepted_counts_ = self._pair_status_codes = None
        self.bound_status_ = None
        self.n_samples_ = self.n_features_in_ = None
        self.execution_mode_ = None

    def _bounds(self, quantiles):
        return _compute_bounds_from_quantiles(
            quantiles,
            lambda_scale=self.lambda_scale,
            use_ter=self.use_ter,
            ter_max=self.ter_max,
            epsilon=self.epsilon,
        )

    def _publish(self, quantiles, bounds, moments, n_samples, mode):
        correlation = moments.correlation()
        counts, codes = moments.accepted_counts(), moments.status_codes()
        self.quantiles = quantiles
        self.location_, self.scale_, self.ter_, lower, upper = bounds
        self.thresholds = (lower, upper)
        self.accepted_counts_, self._pair_status_codes = counts, codes
        self.bound_status_ = _bound_status(quantiles, self.use_ter, self.epsilon)
        self.n_samples_, self.n_features_in_ = n_samples, quantiles.shape[1]
        self.execution_mode_ = mode
        return correlation

    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        """Fit final-bound Eq. 13; array conversion may require O(Np) memory."""
        matrix = _as_float_matrix(X)
        quantiles = self._compute_quantiles(matrix)
        bounds = self._bounds(quantiles)
        moments = PairMoments(matrix.shape[1])
        for start in range(0, len(matrix), DEFAULT_CHUNK_SIZE):
            moments.update_batch(
                matrix[start : start + DEFAULT_CHUNK_SIZE], *bounds[3:]
            )
        return self._publish(quantiles, bounds, moments, len(matrix), "batch")

    def fit_replayable(self, source_factory, chunk_size=DEFAULT_CHUNK_SIZE):
        """Fit using two fresh traversals of the same finite ordered source.

        The factory returns an iterable yielding rows or 2-D blocks. Conversion,
        validation and hashing are bounded by ``chunk_size``. The small-batch
        fallback is unavailable (``exact_cutover=5`` is required); the optional
        ``exact_prefix`` readout needs an additional O(exact_prefix*p) buffer,
        released after the prefix. A checksum of float64 values plus shape
        detects changed rows, order, or length; chunk boundaries may differ.
        Source-owned storage and source buffering are outside this memory bound.
        Failed fits leave previously published diagnostics unchanged.
        """
        chunk_size = validate_chunk_size(chunk_size)
        if self.exact_cutover != 5:
            raise ValueError("fit_replayable requires exact_cutover=5.")
        if not callable(source_factory):
            raise ValueError(
                "source_factory must return a fresh iterable on each call."
            )
        try:
            first = iter(source_factory())
        except TypeError as exc:
            raise ValueError("source_factory must return an iterable.") from exc
        state = None
        first_digest = hashlib.sha256()
        for chunk in validated_chunks(first, chunk_size):
            if state is None:
                state = P2State(
                    chunk.shape[1], self.probs, exact_prefix=self.exact_prefix
                )
            state.update_batch(chunk)
            first_digest.update(memoryview(chunk).cast("B"))
        if state is None or state.count < 5:
            raise ValueError("Input source must have at least 5 samples.")
        quantiles = state.quantiles
        bounds = self._bounds(quantiles)
        try:
            second = iter(source_factory())
        except TypeError as exc:
            raise ValueError("source_factory must return an iterable.") from exc
        if second is first:
            raise ValueError("source_factory returned a reused one-shot iterator.")
        moments = PairMoments(quantiles.shape[1])
        second_digest = hashlib.sha256()
        n_samples = 0
        for chunk in validated_chunks(second, chunk_size):
            if chunk.shape[1] != quantiles.shape[1]:
                raise ValueError("Replay feature count changed between traversals.")
            moments.update_batch(chunk, *bounds[3:])
            n_samples += len(chunk)
            second_digest.update(memoryview(chunk).cast("B"))
        if n_samples != state.count or first_digest.digest() != second_digest.digest():
            raise ValueError("Replay content, row order, or sample count changed.")
        return self._publish(quantiles, bounds, moments, n_samples, "replay")

    def _compute_quantiles(self, X):
        if X.shape[0] < self.exact_cutover:
            return linear_quantiles(X, self.probs, axis=0)
        return _compute_all_quantiles_numba(
            X, self.probs, exact_prefix=self.exact_prefix
        )

    def _calculate_adaptive_thresholds(self, quantiles):
        """Compatibility helper returning only lower and upper bounds."""
        return self._bounds(quantiles)[3:]

    @property
    def pair_status_codes_(self):
        """Compact int8 status snapshot: 0 ok, 1 insufficient, 2 zero variance."""
        return (
            None if self._pair_status_codes is None else self._pair_status_codes.copy()
        )

    @property
    def pair_status_(self):
        """Expanded labels, allocated only when requested."""
        return (
            None
            if self._pair_status_codes is None
            else status_labels_from_codes(self._pair_status_codes)
        )

    @property
    def state_nbytes_(self):
        """Retained NumPy diagnostics, excluding input, output and fit workspace."""
        arrays = [
            self.probs,
            self.quantiles,
            self.location_,
            self.scale_,
            self.ter_,
            self.accepted_counts_,
            self._pair_status_codes,
        ]
        if self.thresholds is not None:
            arrays.extend(self.thresholds)
        if self.bound_status_ is not None:
            arrays.extend(self.bound_status_.values())
        return int(sum(a.nbytes for a in arrays if a is not None))
