"""Core implementation of the FORCE robust correlation estimator."""

from __future__ import annotations

import logging
import warnings
from numbers import Integral, Real
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only in minimal installations
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            def wrapper(*wrapper_args, **wrapper_kwargs):
                raise ImportError("Numba is required to run FORCE.")

            return wrapper

        return decorator

    prange = range


P2_PROBABILITIES = np.array([0.01, 0.25, 0.50, 0.75, 0.99], dtype=np.float64)


def p2_desired_positions(n_observations: int, probability: float) -> NDArray[np.float64]:
    """Return the standard Jain-Chlamtac P² desired marker positions."""
    if not isinstance(n_observations, Integral) or isinstance(n_observations, bool):
        raise ValueError("n_observations must be an integer.")
    if n_observations < 5:
        raise ValueError("P² marker positions require at least five observations.")
    if not isinstance(probability, Real) or not np.isfinite(probability):
        raise ValueError("probability must be finite.")
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1.")

    span = float(n_observations - 1)
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


class P2Quantile:
    """Incremental five-marker P² estimator used for diagnostics and testing."""

    def __init__(self, probability: float):
        # Reuse public validation and retain only the probability here.
        p2_desired_positions(5, probability)
        self.probability = float(probability)
        self.count = 0
        self._initial: list[float] = []
        self._heights: Optional[NDArray[np.float64]] = None
        self._positions: Optional[NDArray[np.float64]] = None
        self._desired: Optional[NDArray[np.float64]] = None
        self._increments = np.array(
            [
                0.0,
                self.probability / 2.0,
                self.probability,
                (1.0 + self.probability) / 2.0,
                1.0,
            ],
            dtype=np.float64,
        )

    def update(self, value: float) -> "P2Quantile":
        """Consume one finite observation and return ``self``."""
        if not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("P² observations must be finite real numbers.")
        value = float(value)

        if self.count < 5:
            self._initial.append(value)
            self.count += 1
            if self.count == 5:
                self._heights = np.sort(np.asarray(self._initial, dtype=np.float64))
                self._positions = np.arange(1.0, 6.0, dtype=np.float64)
                self._desired = p2_desired_positions(5, self.probability)
            return self

        assert self._heights is not None
        assert self._positions is not None
        assert self._desired is not None
        q = self._heights
        n = self._positions

        if value < q[0]:
            q[0] = value
            cell = 0
        elif value < q[1]:
            cell = 0
        elif value < q[2]:
            cell = 1
        elif value < q[3]:
            cell = 2
        elif value < q[4]:
            cell = 3
        else:
            q[4] = value
            cell = 3

        n[cell + 1 :] += 1.0
        self._desired += self._increments
        self.count += 1

        for marker in range(1, 4):
            delta = self._desired[marker] - n[marker]
            direction = 1.0 if delta >= 1.0 else (-1.0 if delta <= -1.0 else 0.0)
            if direction == 0.0:
                continue
            if direction > 0.0 and n[marker + 1] - n[marker] <= 1.0:
                continue
            if direction < 0.0 and n[marker] - n[marker - 1] <= 1.0:
                continue

            proposed = q[marker] + direction / (n[marker + 1] - n[marker - 1]) * (
                (n[marker] - n[marker - 1] + direction)
                * (q[marker + 1] - q[marker])
                / (n[marker + 1] - n[marker])
                + (n[marker + 1] - n[marker] - direction)
                * (q[marker] - q[marker - 1])
                / (n[marker] - n[marker - 1])
            )
            if q[marker - 1] < proposed < q[marker + 1]:
                q[marker] = proposed
            else:
                adjacent = marker + int(direction)
                q[marker] += direction * (q[adjacent] - q[marker]) / (
                    n[adjacent] - n[marker]
                )
            n[marker] += direction

        return self

    @property
    def value(self) -> float:
        """Current estimate, using exact interpolation before initialization."""
        if self.count == 0:
            return float("nan")
        if self.count < 5:
            return float(np.quantile(np.asarray(self._initial), self.probability))
        assert self._heights is not None
        return float(self._heights[2])

    @property
    def marker_heights(self) -> NDArray[np.float64]:
        if self._heights is None:
            return np.sort(np.asarray(self._initial, dtype=np.float64))
        return self._heights.copy()

    @property
    def marker_positions(self) -> NDArray[np.float64]:
        if self._positions is None:
            return np.arange(1.0, self.count + 1.0, dtype=np.float64)
        return self._positions.copy()

    @property
    def desired_positions(self) -> NDArray[np.float64]:
        if self.count < 5:
            return np.empty(0, dtype=np.float64)
        assert self._desired is not None
        return self._desired.copy()


@njit(fastmath=True, cache=True)
def _p_square_kernel(
    data: NDArray[np.float64], probs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Estimate multiple quantiles of one finite stream with independent P² states."""
    n_samples = len(data)
    n_probs = len(probs)
    if n_samples < 5:
        return np.quantile(data, probs)

    q = np.zeros((n_probs, 5), dtype=np.float64)
    positions = np.zeros((n_probs, 5), dtype=np.float64)
    desired = np.zeros((n_probs, 5), dtype=np.float64)
    increments = np.zeros((n_probs, 5), dtype=np.float64)
    initial = np.sort(data[:5])

    for p_idx in range(n_probs):
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

    for sample_idx in range(5, n_samples):
        value = data[sample_idx]
        for p_idx in range(n_probs):
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
                direction = 1.0 if delta >= 1.0 else (-1.0 if delta <= -1.0 else 0.0)
                if direction == 0.0:
                    continue
                if (
                    direction > 0.0
                    and positions[p_idx, marker + 1] - positions[p_idx, marker] <= 1.0
                ):
                    continue
                if (
                    direction < 0.0
                    and positions[p_idx, marker] - positions[p_idx, marker - 1] <= 1.0
                ):
                    continue

                proposed = q[p_idx, marker] + direction / (
                    positions[p_idx, marker + 1] - positions[p_idx, marker - 1]
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
                        positions[p_idx, adjacent] - positions[p_idx, marker]
                    )
                positions[p_idx, marker] += direction

    result = np.empty(n_probs, dtype=np.float64)
    for p_idx in range(n_probs):
        result[p_idx] = q[p_idx, 2]
    return result


@njit(parallel=True, cache=True)
def _compute_all_quantiles_numba(
    X: NDArray[np.float64], probs: NDArray[np.float64]
) -> NDArray[np.float64]:
    n_features = X.shape[1]
    quantiles = np.empty((len(probs), n_features), dtype=np.float64)
    for feature in prange(n_features):
        quantiles[:, feature] = _p_square_kernel(X[:, feature], probs)
    return quantiles


@njit(parallel=True, fastmath=True, cache=True)
def _compute_trimmed_corr_numba(
    X: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Equation 13 using pair-specific accepted-observation means."""
    n_samples, n_features = X.shape
    correlation = np.eye(n_features, dtype=np.float64)

    for left in prange(n_features):
        for right in range(left + 1, n_features):
            count = 0
            mean_left = 0.0
            mean_right = 0.0
            sum_xx = 0.0
            sum_yy = 0.0
            sum_xy = 0.0
            for sample in range(n_samples):
                x = X[sample, left]
                y = X[sample, right]
                if (
                    lower_bounds[left] <= x <= upper_bounds[left]
                    and lower_bounds[right] <= y <= upper_bounds[right]
                ):
                    count += 1
                    delta_left = x - mean_left
                    delta_right = y - mean_right
                    mean_left += delta_left / count
                    mean_right += delta_right / count
                    sum_xx += delta_left * (x - mean_left)
                    sum_yy += delta_right * (y - mean_right)
                    sum_xy += delta_left * (y - mean_right)

            value = 0.0
            if count > 1:
                if sum_xx > 0.0 and sum_yy > 0.0:
                    value = sum_xy / np.sqrt(sum_xx * sum_yy)
                    value = min(1.0, max(-1.0, value))

            correlation[left, right] = value
            correlation[right, left] = value

    return correlation


def _as_float_matrix(X: ArrayLike) -> NDArray[np.float64]:
    """Validate estimator input and return a contiguous float64 matrix."""
    try:
        matrix = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Input X must be a numeric two-dimensional array.") from exc
    if matrix.ndim != 2:
        raise ValueError("Input X must be a two-dimensional array.")
    if matrix.shape[0] < 5:
        raise ValueError("Input X must have at least 5 samples.")
    if matrix.shape[1] < 2:
        raise ValueError("Input X must have at least 2 features.")
    if not np.all(np.isfinite(matrix)):
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


def _covariance_to_correlation(
    covariance: ArrayLike,
) -> NDArray[np.float64]:
    """Normalize a finite covariance matrix with deterministic degeneracy rules."""
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must contain only finite values.")

    matrix = 0.5 * (matrix + matrix.T)
    variances = np.diag(matrix)
    positive = variances > 0.0
    scales = np.ones_like(variances)
    scales[positive] = np.sqrt(variances[positive])
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = matrix / np.outer(scales, scales)
    undefined = ~(positive[:, None] & positive[None, :])
    correlation[undefined] = 0.0
    correlation[~np.isfinite(correlation)] = 0.0
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _compute_bounds_from_quantiles(
    quantiles: NDArray[np.float64],
    *,
    lambda_scale: float,
    use_ter: bool,
    ter_max: Optional[float],
    epsilon: float,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Compute Equations 6-11 from q01, q25, q50, q75 and q99."""
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim != 2 or quantiles.shape[0] != 5:
        raise ValueError("quantiles must have shape (5, n_features).")
    if not np.all(np.isfinite(quantiles)):
        raise ValueError("quantiles must contain only finite values.")

    q01, q25, q50, q75, q99 = quantiles
    location = q50.copy()
    scale = (q75 - q25) / 1.349
    negative_scale = scale < 0.0
    if np.any(negative_scale):
        warnings.warn(
            "Non-monotone P² quartiles produced a negative IQR; affected scales "
            "were collapsed to zero.",
            RuntimeWarning,
            stacklevel=2,
        )
        scale = np.maximum(scale, 0.0)

    ter = np.ones_like(location)
    if use_ter:
        denominator = np.abs(q50 - q01)
        degenerate = denominator < epsilon
        if np.any(degenerate):
            warnings.warn(
                "TER denominator is below epsilon for one or more features; "
                "TER was set to 1 for those features.",
                RuntimeWarning,
                stacklevel=2,
            )
        ratio = np.abs(q99 - q50) / (denominator + epsilon)
        ter = np.maximum(1.0, ratio)
        ter[degenerate] = 1.0
        if ter_max is not None:
            ter = np.minimum(ter, float(ter_max))

    half_width = float(lambda_scale) * ter * scale
    lower = location - half_width
    upper = location + half_width
    return location, scale, ter, lower, upper


class ForceEstimator:
    """Fast Outlier-Robust Correlation Estimator from Algorithm 1."""

    def __init__(
        self,
        lambda_scale: float = 3.0,
        exact_cutover: int = 5,
        use_ter: bool = True,
        ter_max: Optional[float] = None,
        epsilon: float = 1e-10,
    ):
        if not HAS_NUMBA:
            raise ImportError("ForceEstimator requires Numba.")
        _validate_force_parameters(
            lambda_scale, exact_cutover, use_ter, ter_max, epsilon
        )
        self.lambda_scale = float(lambda_scale)
        self.exact_cutover = int(exact_cutover)
        self.use_ter = bool(use_ter)
        self.ter_max = None if ter_max is None else float(ter_max)
        self.epsilon = float(epsilon)
        self.probs = P2_PROBABILITIES.copy()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.quantiles: Optional[NDArray[np.float64]] = None
        self.thresholds: Optional[
            Tuple[NDArray[np.float64], NDArray[np.float64]]
        ] = None
        self.location_: Optional[NDArray[np.float64]] = None
        self.scale_: Optional[NDArray[np.float64]] = None
        self.ter_: Optional[NDArray[np.float64]] = None

    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        """Estimate the pairwise trimmed correlation matrix."""
        matrix = _as_float_matrix(X)
        self.quantiles = self._compute_quantiles(matrix)
        (
            self.location_,
            self.scale_,
            self.ter_,
            lower,
            upper,
        ) = _compute_bounds_from_quantiles(
            self.quantiles,
            lambda_scale=self.lambda_scale,
            use_ter=self.use_ter,
            ter_max=self.ter_max,
            epsilon=self.epsilon,
        )
        self.thresholds = (lower, upper)
        return _compute_trimmed_corr_numba(matrix, lower, upper)

    def _compute_quantiles(
        self, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if X.shape[0] < self.exact_cutover:
            return np.quantile(X, self.probs, axis=0, method="linear")
        return _compute_all_quantiles_numba(X, self.probs)

    def _calculate_adaptive_thresholds(
        self, quantiles: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compatibility helper returning only lower and upper bounds."""
        _, _, _, lower, upper = _compute_bounds_from_quantiles(
            quantiles,
            lambda_scale=self.lambda_scale,
            use_ter=self.use_ter,
            ter_max=self.ter_max,
            epsilon=self.epsilon,
        )
        return lower, upper

    @property
    def state_nbytes_(self) -> int:
        """Bytes retained by fitted estimator diagnostics, excluding input/output."""
        arrays = [
            self.quantiles,
            self.location_,
            self.scale_,
            self.ter_,
        ]
        if self.thresholds is not None:
            arrays.extend(self.thresholds)
        return int(sum(array.nbytes for array in arrays if array is not None))
