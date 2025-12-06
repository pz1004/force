"""
Core implementation of the FORCE algorithm.

This module contains the `ForceEstimator` class and the Numba-jitted kernels
that form the core of the Fast Outlier-Robust Correlation Estimation method.
"""
import logging
from typing import Tuple

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray

# --- Numba JIT Compilation Check ---
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logging.warning(
        "'numba' is not installed. FORCE will not be able to run its "
        "optimized routines. Please install it via: pip install numba"
    )

    # Define dummy decorators to allow the code to be imported without Numba.
    # The methods will raise an ImportError if called.
    def njit(fastmath: bool = False, parallel: bool = False):
        def decorator(func):
            def wrapper(*args, **kwargs):
                raise ImportError(
                    "Numba is required to run this function but it is not installed."
                )
            return wrapper
        return decorator

    prange = range

# =============================================================================
# JIT-Compiled Kernels for High-Performance Computation
# =============================================================================

@njit(fastmath=True)
def _p_square_kernel(
    data: NDArray[np.float64], probs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Computes quantiles for a single feature using the P-Square algorithm.

    This kernel implements the P-Square algorithm by Jain & Chlamtac (1985)
    for derivative-free estimation of quantiles. It processes a single feature
    (a 1D array) and computes all specified quantiles in a single pass, making
    it highly efficient for streaming or large datasets.

    Note:
        This function is Just-In-Time (JIT) compiled with Numba for performance.

    Args:
        data: A 1D NumPy array of float64 representing a single feature column.
        probs: A 1D NumPy array of probabilities for which to compute quantiles.

    Returns:
        A 1D NumPy array containing the computed quantile values corresponding
        to the input probabilities.
    """
    n_samples = len(data)
    n_probs = len(probs)
    results = np.zeros(n_probs, dtype=np.float64)

    # State variables for each probability, stored in arrays for Numba compatibility.
    # Dimensions: [n_probs, 5 markers]
    q = np.zeros((n_probs, 5), dtype=np.float64)  # Marker heights
    n = np.zeros((n_probs, 5), dtype=np.float64)  # Marker positions
    np_desired = np.zeros((n_probs, 5), dtype=np.float64)  # Desired positions
    dn = np.zeros((n_probs, 5), dtype=np.float64)  # Position increments

    # --- Initialization Phase ---
    # We must initialize with the first 5 sorted data points.
    if n_samples < 5:
        # Fallback to exact numpy quantile for very small samples
        return np.quantile(data, probs)

    buffer = np.sort(data[:5])
    for p_idx in range(n_probs):
        p = probs[p_idx]
        q[p_idx, :] = buffer
        n[p_idx, :] = np.arange(1.0, 6.0)
        np_desired[p_idx, :] = [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0]
        dn[p_idx, :] = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]

    # --- Update Phase for remaining data points ---
    for i in range(5, n_samples):
        val = data[i]
        for p_idx in range(n_probs):
            # Find which cell the new value falls into
            if val < q[p_idx, 0]:
                q[p_idx, 0] = val
                k = 0
            elif val < q[p_idx, 1]:
                k = 0
            elif val < q[p_idx, 2]:
                k = 1
            elif val < q[p_idx, 3]:
                k = 2
            elif val < q[p_idx, 4]:
                k = 3
            else:
                q[p_idx, 4] = val
                k = 3

            # Increment positions of markers beyond the insertion cell
            for j in range(k + 1, 5):
                n[p_idx, j] += 1.0
            for j in range(5):
                np_desired[p_idx, j] += dn[p_idx, j]

            # Adjust marker heights using parabolic or linear interpolation
            for j in range(1, 4):
                d = np_desired[p_idx, j] - n[p_idx, j]
                d_sign = np.sign(d)

                if (d_sign > 0 and (n[p_idx, j+1] - n[p_idx, j]) > 1) or \
                   (d_sign < 0 and (n[p_idx, j-1] - n[p_idx, j]) < -1):

                    # Try parabolic adjustment (Equation 5 in FORCE.pdf)
                    num = (d_sign / (n[p_idx, j+1] - n[p_idx, j-1])) * (
                        (n[p_idx, j] - n[p_idx, j-1] + d_sign) *
                        (q[p_idx, j+1] - q[p_idx, j]) / (n[p_idx, j+1] - n[p_idx, j]) +
                        (n[p_idx, j+1] - n[p_idx, j] - d_sign) *
                        (q[p_idx, j] - q[p_idx, j-1]) / (n[p_idx, j] - n[p_idx, j-1])
                    )
                    q_new = q[p_idx, j] + num

                    if q[p_idx, j-1] < q_new < q[p_idx, j+1]:
                        q[p_idx, j] = q_new
                    else:
                        # Fallback to linear adjustment (Equation 6 in FORCE.pdf)
                        idx_offset = int(d_sign)
                        q[p_idx, j] += d_sign * (q[p_idx, j + idx_offset] - q[p_idx, j]) / \
                                       (n[p_idx, j + idx_offset] - n[p_idx, j])

                    n[p_idx, j] += d_sign

    for p_idx in range(n_probs):
        results[p_idx] = q[p_idx, 2]  # The median marker q_3 is the estimate

    return results


@njit(parallel=True)
def _compute_all_quantiles_numba(
    X: NDArray[np.float64], probs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Drives the parallel computation of quantiles across all features.

    This function serves as a driver that iterates over the columns (features)
    of the input matrix `X` in parallel. For each column, it calls the
    `_p_square_kernel` to compute the required quantiles.

    Note:
        This function is JIT-compiled and parallelized with Numba.

    Args:
        X: A 2D NumPy array of shape (n_samples, n_features).
        probs: A 1D NumPy array of probabilities.

    Returns:
        A 2D NumPy array of shape (n_probs, n_features) containing the
        computed quantiles.
    """
    n_samples, n_features = X.shape
    n_probs = len(probs)
    quantiles = np.zeros((n_probs, n_features), dtype=np.float64)

    for j in prange(n_features):
        col_data = X[:, j]
        res = _p_square_kernel(col_data, probs)
        for i in range(n_probs):
            quantiles[i, j] = res[i]

    return quantiles


@njit(parallel=True, fastmath=True)
def _compute_trimmed_corr_numba(
    X: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    medians: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Computes the robust correlation matrix from adaptively trimmed data.

    This kernel implements the final step of the FORCE algorithm (Equation 8
    in FORCE.pdf). It computes the Pearson correlation on a subset of the
    data that falls within the adaptive `lower_bounds` and `upper_bounds`.
    The calculation is centered around the robust `medians` (q50).

    This process is performed in a single pass over the data for each pair
    of features, making it O(N). The outer loops are parallelized.

    Args:
        X: The input data matrix of shape (n_samples, n_features).
        lower_bounds: A 1D array of lower thresholds for each feature.
        upper_bounds: A 1D array of upper thresholds for each feature.
        medians: A 1D array of median values (q50) for each feature.

    Returns:
        The resulting (n_features, n_features) robust correlation matrix.
    """
    n_samples, n_features = X.shape
    corr = np.eye(n_features, dtype=np.float64)

    for i in prange(n_features):
        for j in range(i + 1, n_features):
            sxx, syy, sxy, count = 0.0, 0.0, 0.0, 0

            for k in range(n_samples):
                val_i, val_j = X[k, i], X[k, j]

                # Check if the observation is within the adaptive bounds for BOTH features
                if (val_i >= lower_bounds[i] and val_i <= upper_bounds[i] and
                    val_j >= lower_bounds[j] and val_j <= upper_bounds[j]):

                    dx = val_i - medians[i]
                    dy = val_j - medians[j]

                    sxx += dx * dx
                    syy += dy * dy
                    sxy += dx * dy
                    count += 1

            if count > 1 and sxx > 1e-12 and syy > 1e-12:
                r = sxy / np.sqrt(sxx * syy)
                corr[i, j] = r
                corr[j, i] = r
            # Off-diagonal elements are implicitly zero if condition is not met

    return corr


class ForceEstimator:
    """
    Fast Outlier-Robust Correlation (FORCE) Estimator.

    This class implements the FORCE algorithm, a three-stage process for
    robustly estimating correlation matrices in the presence of outliers.

    Stage 1: Streaming Quantile Estimation
        Uses the P-Square algorithm to efficiently compute quantiles (q05, q25,
        q50, q75, q95) for each feature in a single pass. For small datasets,
        it falls back to exact numpy quantiles.

    Stage 2: Adaptive Thresholding
        Calculates adaptive trimming thresholds based on the interquartile range
        (IQR) and a tail-extremity ratio (TER), as defined in Equations 3 and 4
        of the FORCE paper. This allows the trimming to be more aggressive in
        the presence of extreme outliers.

    Stage 3: Robust Accumulation
        Computes the correlation matrix on the data points that fall within the
        adaptive thresholds, centered around the robust median.

    Args:
        lambda_scale (float, optional): Scaling factor for the adaptive
            thresholds. Corresponds to λ in Equation 4 of the paper.
            Defaults to 3.0.
        exact_cutover (int, optional): Sample size below which exact quantiles
            (numpy.quantile) are used instead of the P-Square approximation.
            Defaults to 100.

    Attributes:
        logger: A logging instance for the estimator.
        quantiles (Optional[NDArray]): The computed quantiles from the last fit.
        thresholds (Optional[Tuple[NDArray, NDArray]]): The lower and upper
            trimming bounds from the last fit.
    """

    def __init__(
        self,
        lambda_scale: float = 3.0,
        exact_cutover: int = 100,
    ):
        if not HAS_NUMBA:
            raise ImportError(
                "ForceEstimator requires Numba to be installed. "
                "Please run `pip install numba`."
            )
        self.lambda_scale = lambda_scale
        self.exact_cutover = exact_cutover
        self.probs = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=np.float64)
        self._ter_norm = stats.norm.ppf(0.95) - stats.norm.ppf(0.05)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.quantiles: Optional[NDArray[np.float64]] = None
        self.thresholds: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None

    def fit(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Fit the FORCE model to the data.

        Args:
            X: The input data matrix of shape (n_samples, n_features).

        Returns:
            The (n_features, n_features) robust correlation matrix.

        Raises:
            ValueError: If `X` is not a 2D array or has too few samples
                or features.
        """
        self._validate_input(X)
        n_samples, n_features = X.shape
        self.logger.info(
            "Fitting FORCE model | samples=%d, features=%d", n_samples, n_features
        )

        # --- Stage 1: Quantile Estimation ---
        self.quantiles = self._compute_quantiles(X)
        q05, q25, q50, q75, q95 = self.quantiles

        # --- Stage 2: Adaptive Thresholding ---
        lower_bounds, upper_bounds = self._calculate_adaptive_thresholds(
            self.quantiles
        )
        self.thresholds = (lower_bounds, upper_bounds)

        # --- Stage 3: Robust Accumulation ---
        self.logger.info("Performing robust accumulation via Numba kernel.")
        corr_matrix = _compute_trimmed_corr_numba(
            X, lower_bounds, upper_bounds, q50
        )

        return corr_matrix

    def _validate_input(self, X: NDArray[np.float64]) -> None:
        """Basic input validation."""
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input `X` must be a 2D NumPy array.")
        if X.shape[0] < 5:
            raise ValueError("Input `X` must have at least 5 samples.")
        if X.shape[1] < 2:
            raise ValueError("Input `X` must have at least 2 features.")

    def _compute_quantiles(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Selects quantile computation strategy based on sample size."""
        n_samples = X.shape[0]
        if n_samples < self.exact_cutover:
            self.logger.info(
                "Using exact quantiles (n_samples < %d).", self.exact_cutover
            )
            return np.quantile(X, self.probs, axis=0)
        else:
            self.logger.info("Using P-Square streaming quantile estimation.")
            return _compute_all_quantiles_numba(X, self.probs)

    def _calculate_adaptive_thresholds(
        self,
        quantiles: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Calculates the adaptive upper and lower trimming bounds.

        This corresponds to Equations 3 and 4 in the FORCE paper.
        """
        q05, q25, q50, q75, q95 = quantiles

        # Robust standard deviation (Equation 3)
        sigma_robust = (q75 - q25) / 1.349  # 1.349 is 2*norm.ppf(0.75)
        sigma_robust[sigma_robust <= 1e-12] = 1.0  # Avoid division by zero

        # Tail-Extremity Ratio (TER) (Equation 4)
        ter_denom = sigma_robust * self._ter_norm
        ter = (q95 - q05) / np.where(ter_denom <= 1e-12, 1.0, ter_denom)
        ter[~np.isfinite(ter)] = 1.0  # Handle potential division by zero or NaN

        # Final adaptive thresholds (Equation 2)
        scaled_deviation = self.lambda_scale * ter * sigma_robust
        upper_bounds = q50 + scaled_deviation
        lower_bounds = q50 - scaled_deviation

        return lower_bounds, upper_bounds
