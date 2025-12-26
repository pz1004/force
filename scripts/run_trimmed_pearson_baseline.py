import numpy as np
from time import perf_counter

try:
    import numba as nb
except ImportError:
    nb = None


def _compute_bounds_from_quantiles(q01, q25, q50, q75, q99, lam=3.0, use_ter=True, ter_max=None):
    """
    Mimics Algorithm 1 robust parameters:
      mu = q50
      sigma = (q75 - q25)/1.349
      TER = max(1, |q99-q50|/|q50-q01|)
      bounds = mu ± lam*TER*sigma
    If use_ter=False, TER is forced to 1 (fixed trimming).
    """
    mu = q50
    sigma = (q75 - q25) / 1.349
    sigma = np.where(sigma <= 1e-12, 1e-12, sigma)

    if use_ter:
        denom = np.abs(q50 - q01)
        denom = np.where(denom <= 1e-12, 1e-12, denom)
        ter = np.maximum(1.0, np.abs(q99 - q50) / denom)
        if ter_max is not None:
            ter = np.minimum(ter, ter_max)
    else:
        ter = np.ones_like(mu)

    low = mu - lam * ter * sigma
    high = mu + lam * ter * sigma
    return low, high


def trimmed_pearson_exact(X, lam=3.0, use_ter=True, ter_max=None):
    """
    Offline trimmed Pearson baseline using exact (batch) quantiles.
    X: (N, p)
    Returns: (p, p) correlation matrix.
    """
    qs = np.quantile(X, [0.01, 0.25, 0.50, 0.75, 0.99], axis=0)
    q01, q25, q50, q75, q99 = qs
    low, high = _compute_bounds_from_quantiles(q01, q25, q50, q75, q99, lam=lam, use_ter=use_ter, ter_max=ter_max)

    if nb is not None:
        return _trimmed_corr_numba(X, low, high)
    return _trimmed_corr_numpy(X, low, high)


def _trimmed_corr_numpy(X, low, high):
    N, p = X.shape
    R = np.eye(p, dtype=np.float64)

    for j in range(p):
        for k in range(j + 1, p):
            mask = (X[:, j] >= low[j]) & (X[:, j] <= high[j]) & (X[:, k] >= low[k]) & (X[:, k] <= high[k])
            xj = X[mask, j]
            xk = X[mask, k]
            if xj.size < 2:
                r = 0.0
            else:
                xj = xj - xj.mean()
                xk = xk - xk.mean()
                denom = np.sqrt((xj @ xj) * (xk @ xk)) + 1e-12
                r = float((xj @ xk) / denom)
            R[j, k] = r
            R[k, j] = r
    return R


if nb is not None:
    @nb.njit(parallel=True, fastmath=True)
    def _trimmed_corr_numba(X, low, high):
        N, p = X.shape
        R = np.eye(p, dtype=np.float64)

        for j in nb.prange(p):
            for k in range(j + 1, p):
                s_j = 0.0
                s_k = 0.0
                s_j2 = 0.0
                s_k2 = 0.0
                s_jk = 0.0
                n = 0

                for i in range(N):
                    xj = X[i, j]
                    xk = X[i, k]
                    if (low[j] <= xj <= high[j]) and (low[k] <= xk <= high[k]):
                        s_j += xj
                        s_k += xk
                        s_j2 += xj * xj
                        s_k2 += xk * xk
                        s_jk += xj * xk
                        n += 1

                if n > 1:
                    mj = s_j / n
                    mk = s_k / n
                    cov = s_jk / n - mj * mk
                    vj = s_j2 / n - mj * mj
                    vk = s_k2 / n - mk * mk
                    denom = np.sqrt(vj * vk) + 1e-12
                    r = cov / denom
                else:
                    r = 0.0

                R[j, k] = r
                R[k, j] = r

        return R


def main():
    # Example: integrate with your existing loaders/benchmark harness.
    # Replace this block with your dataset loading (the repo already has benchmark scripts).
    np.random.seed(0)
    N, p = 2000, 50
    X = np.random.randn(N, p)

    # dummy
    t0 = perf_counter()
    R_tp = trimmed_pearson_exact(X, lam=3.0, use_ter=False)  # fixed trimming baseline
    t1 = perf_counter()

    # noTER
    t0 = perf_counter()
    R_tp = trimmed_pearson_exact(X, lam=3.0, use_ter=False)  # fixed trimming baseline
    t1 = perf_counter()

    print("TrimmedPearsonExact(no TER) time (ms):", (t1 - t0) * 1000.0)
    print("R[0,1] =", R_tp[0, 1])

    # TER
    t0 = perf_counter()
    R_tp = trimmed_pearson_exact(X, lam=3.0, use_ter=True, ter_max=3.0) 
    t1 = perf_counter()

    print("TrimmedPearsonExact(TER) time (ms):", (t1 - t0) * 1000.0)
    print("R[0,1] =", R_tp[0, 1])


if __name__ == "__main__":
    main()
