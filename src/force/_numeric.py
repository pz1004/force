"""Scalar arithmetic shared by moment and covariance normalization."""

import math

import numpy as np
from numba import njit


@njit(cache=True)
def normalize_correlation(xx, yy, xy):
    """Compute ``xy / sqrt(xx * yy)`` without extreme intermediate products.

    Positive finite variances and a finite co-moment are required. Binary
    exponents preserve representable tiny correlations across unequal scales.
    A truly unrepresentable tiny result rounds to zero; overflow fails explicitly.
    Checking the correlation interval remains the caller's responsibility.
    """
    if not (
        np.isfinite(xx)
        and np.isfinite(yy)
        and np.isfinite(xy)
        and xx > 0.0
        and yy > 0.0
    ):
        raise FloatingPointError("Correlation requires positive finite variances.")
    if xy == 0.0:
        return 0.0
    mx, ex = math.frexp(xx)
    my, ey = math.frexp(yy)
    mc, ec = math.frexp(xy)
    exponent = ex + ey
    denominator = math.sqrt(mx * my * (2.0 if exponent % 2 else 1.0))
    result = math.ldexp(mc / denominator, ec - exponent // 2)
    if not np.isfinite(result):
        raise FloatingPointError("Correlation normalization overflowed.")
    return result
