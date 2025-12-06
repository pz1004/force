"""
Comparative correlation estimators.

This module provides a collection of standard and robust correlation estimators
that are used for benchmarking against the FORCE algorithm. Each estimator
adheres to a common `CorrelationEstimator` interface.
"""
import abc
import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray
from sklearn.covariance import MinCovDet
from typing import List, Tuple


class CorrelationEstimator(abc.ABC):
    """
    Abstract Base Class for all correlation estimators.

    This class defines the interface that all estimator implementations must
    follow, ensuring they can be used interchangeably in benchmark scripts.
    """

    @abc.abstractmethod
    def fit(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Estimate the correlation matrix for the given data.

        Args:
            X: A 2D NumPy array of shape (n_samples, n_features).

        Returns:
            A 2D NumPy array of shape (n_features, n_features) representing
            the estimated correlation matrix.
        """
        raise NotImplementedError


class PearsonEstimator(CorrelationEstimator):
    """
    Standard Pearson product-moment correlation coefficient.

    This estimator calculates the pairwise correlation between features using
    the standard `numpy.corrcoef` function. It is not robust to outliers.
    """

    def fit(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the Pearson correlation matrix.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            The (n_features, n_features) Pearson correlation matrix.
        """
        if X.ndim != 2 or X.shape[1] < 2:
            return np.eye(X.shape[1])
        return np.corrcoef(X, rowvar=False)


class SpearmanEstimator(CorrelationEstimator):
    """
    Spearman rank-order correlation coefficient.

    This estimator first converts the data to ranks and then computes the
    Pearson correlation on the ranks. It is robust to monotonic transformations
    and less sensitive to outliers than Pearson correlation.
    """

    def fit(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the Spearman rank correlation matrix.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            The (n_features, n_features) Spearman correlation matrix.
        """
        if X.shape[1] < 2:
            return np.eye(X.shape[1])
        # spearmanr can return a single float for 2 features
        rho, _ = stats.spearmanr(X, axis=0)
        return rho if isinstance(rho, np.ndarray) else np.array([[1.0, rho], [rho, 1.0]])


class WinsorizedEstimator(CorrelationEstimator):
    """
    Winsorized correlation estimator.

    This estimator first applies Winsorization to the data, which involves
    clipping extreme values at specified quantiles. It then computes the
    Pearson correlation on the modified data.

    Args:
        limits (List[float], optional): The fractional limits for winsorization
            on each tail. Defaults to [0.05, 0.05], clipping the bottom 5%
            and top 5% of data.
    """

    def __init__(self, limits: List[float] = [0.05, 0.05]):
        self.limits = limits

    def fit(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the correlation matrix on Winsorized data.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            The (n_features, n_features) Winsorized correlation matrix.
        """
        if X.ndim != 2 or X.shape[1] < 2:
            return np.eye(X.shape[1])
            
        # stats.mstats.winsorize returns a masked array
        X_win = stats.mstats.winsorize(X, limits=self.limits, axis=0)
        
        # Convert back to a standard numpy array, filling masked values
        if np.ma.is_masked(X_win):
            X_win = np.ma.getdata(X_win)

        return np.corrcoef(X_win, rowvar=False)


class FastMCDEstimator(CorrelationEstimator):
    """
    Minimum Covariance Determinant (MCD) robust correlation estimator.

    This estimator uses the Minimum Covariance Determinant method to find a
    robust estimate of the covariance matrix, which is then converted to a
    correlation matrix. It is highly robust but computationally intensive.

    Note:
        This is a wrapper around `sklearn.covariance.MinCovDet`.
    """

    def fit(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes the MCD-based robust correlation matrix.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            The (n_features, n_features) MCD-based correlation matrix.
            Falls back to Pearson if MCD fails.
        """
        if X.ndim != 2 or X.shape[1] < 2:
            return np.eye(X.shape[1])
            
        try:
            mcd = MinCovDet(random_state=42).fit(X)
            cov = mcd.covariance_
            d = np.sqrt(np.diag(cov))
            # Guard against zero variance
            d[d == 0] = 1.0
            return cov / np.outer(d, d)
        except Exception:
            # Fallback to Pearson on failure (e.g., singular matrix)
            return np.corrcoef(X, rowvar=False)


class ExactTrimmedEstimator(CorrelationEstimator):
    """
    Trimmed correlation using exact quantiles from `numpy.percentile`.

    This estimator serves as a ground truth for evaluating the P-Square
    approximation used in the main FORCE algorithm. It follows the same
    adaptive trimming logic as FORCE but uses exact, non-streaming quantiles.

    Args:
        lambda_scale (float, optional): Scaling factor for the adaptive
            thresholds. Defaults to 3.0.
    """

    def __init__(self, lambda_scale: float = 3.0):
        self.lambda_scale = lambda_scale
        self.probs = [5, 25, 50, 75, 95]  # In percentiles for numpy
        self._ter_norm = stats.norm.ppf(0.95) - stats.norm.ppf(0.05)

    def fit(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Computes trimmed correlation using exact numpy quantiles.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            The (n_features, n_features) exact trimmed correlation matrix.
        """
        if X.ndim != 2 or X.shape[1] < 2:
            return np.eye(X.shape[1])

        # 1. Exact quantile computation
        quantiles = np.percentile(X, self.probs, axis=0) / 100.0

        # 2. Adaptive thresholding (same logic as FORCE)
        lower, upper = self._calculate_adaptive_thresholds(quantiles)
        
        # 3. Trimmed correlation computation
        return self._compute_trimmed_corr(X, lower, upper, quantiles[2])

    def _calculate_adaptive_thresholds(
        self, quantiles: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculates adaptive thresholds from quantiles."""
        q05, q25, q50, q75, q95 = quantiles
        
        sigma_robust = (q75 - q25) / 1.349
        sigma_robust[sigma_robust <= 1e-12] = 1.0
        
        ter_denom = sigma_robust * self._ter_norm
        ter = (q95 - q05) / np.where(ter_denom <= 1e-12, 1.0, ter_denom)
        ter[~np.isfinite(ter)] = 1.0
        
        scaled_dev = self.lambda_scale * ter * sigma_robust
        return q50 - scaled_dev, q50 + scaled_dev

    def _compute_trimmed_corr(
        self,
        X: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        medians: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Python implementation of trimmed correlation."""
        n_features = X.shape[1]
        corr = np.eye(n_features)

        for i in range(n_features):
            for j in range(i + 1, n_features):
                mask = (
                    (X[:, i] >= lower[i]) & (X[:, i] <= upper[i]) &
                    (X[:, j] >= lower[j]) & (X[:, j] <= upper[j])
                )
                
                if np.sum(mask) > 1:
                    x_i = X[mask, i] - medians[i]
                    x_j = X[mask, j] - medians[j]
                    
                    sxx, syy, sxy = np.sum(x_i**2), np.sum(x_j**2), np.sum(x_i * x_j)
                    
                    if sxx > 1e-12 and syy > 1e-12:
                        r = sxy / np.sqrt(sxx * syy)
                        corr[i, j] = corr[j, i] = r
        return corr
