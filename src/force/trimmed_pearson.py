# src/force/trimmed_pearson.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def _trimmed_pairwise_pearson_corr(
    X: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Pairwise-trimmed Pearson correlation:
    for each (j,k), keeps samples i such that:
        lower[j] <= X[i,j] <= upper[j] AND lower[k] <= X[i,k] <= upper[k]
    and computes Pearson on the retained subset.

    Returns:
        R: (p,p) correlation matrix
    """
    n, p = X.shape
    R = np.eye(p, dtype=np.float64)

    for j in prange(p):
        for k in range(j + 1, p):
            # First pass: compute trimmed means
            cnt = 0
            sx = 0.0
            sy = 0.0
            lj = lower[j]
            uj = upper[j]
            lk = lower[k]
            uk = upper[k]

            for i in range(n):
                x = X[i, j]
                y = X[i, k]
                if (x >= lj and x <= uj) and (y >= lk and y <= uk):
                    cnt += 1
                    sx += x
                    sy += y

            if cnt < 3:
                r = 0.0
            else:
                mx = sx / cnt
                my = sy / cnt

                # Second pass: covariance & variances on trimmed subset
                sxx = 0.0
                syy = 0.0
                sxy = 0.0
                for i in range(n):
                    x = X[i, j]
                    y = X[i, k]
                    if (x >= lj and x <= uj) and (y >= lk and y <= uk):
                        dx = x - mx
                        dy = y - my
                        sxx += dx * dx
                        syy += dy * dy
                        sxy += dx * dy

                if sxx <= 0.0 or syy <= 0.0:
                    r = 0.0
                else:
                    r = sxy / np.sqrt(sxx * syy)

            R[j, k] = r
            R[k, j] = r

    return R


def _compute_exact_quantiles(
    X: np.ndarray,
    qs=(0.01, 0.25, 0.50, 0.75, 0.99),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.quantile(X, qs, axis=0, method="linear")
    return q[0], q[1], q[2], q[3], q[4]


def _compute_bounds_from_quantiles(
    q01: np.ndarray,
    q25: np.ndarray,
    q50: np.ndarray,
    q75: np.ndarray,
    q99: np.ndarray,
    *,
    lam: float,
    use_ter: bool,
    ter_max: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches the paper’s definitions:
      mu_j = q50
      sigma_j = (q75 - q25) / 1.349
      TER_j = max(1, |q99-q50| / (|q50-q01| + eps)) and then capped to ter_max
      bounds: [mu - lam * TER * sigma, mu + lam * TER * sigma]
    """
    mu = q50
    sigma = (q75 - q25) / 1.349
    sigma = np.maximum(sigma, eps)

    if use_ter:
        ter = np.abs(q99 - q50) / (np.abs(q50 - q01) + eps)
        ter = np.maximum(1.0, ter)
        ter = np.minimum(ter_max, ter)
    else:
        ter = np.ones_like(mu)

    lower = mu - lam * ter * sigma
    upper = mu + lam * ter * sigma
    return lower, upper


@dataclass(frozen=True)
class TrimmedPearsonExact:
    """
    Exact-quantile (sorting-based) baseline:
      - no TER: symmetric bounds (TER=1)
      - with TER: adaptive bounds via TER

    Call signature matches a typical benchmark pipeline: estimator(X)->corr_matrix.
    """
    lam: float = 3.0
    use_ter: bool = False
    ter_max: float = 3.0
    eps: float = 1e-10

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        q01, q25, q50, q75, q99 = _compute_exact_quantiles(X)
        lower, upper = _compute_bounds_from_quantiles(
            q01, q25, q50, q75, q99,
            lam=self.lam,
            use_ter=self.use_ter,
            ter_max=self.ter_max,
            eps=self.eps,
        )
        return _trimmed_pairwise_pearson_corr(X, lower, upper)

    def fit(self, X: np.ndarray) -> np.ndarray:
        """Alias for __call__ to match CorrelationEstimator interface."""
        return self(X)
