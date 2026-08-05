"""Exact-quantile baselines sharing FORCE's trimming mathematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import (
    P2_PROBABILITIES,
    _as_float_matrix,
    _compute_bounds_from_quantiles,
    _compute_trimmed_corr_numba,
    _validate_force_parameters,
)


def _compute_exact_quantiles(
    X: NDArray[np.float64],
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    quantiles = np.quantile(X, P2_PROBABILITIES, axis=0, method="linear")
    return tuple(quantiles[index] for index in range(5))  # type: ignore[return-value]


@dataclass(frozen=True)
class TrimmedPearsonExact:
    """Sorting-based exact counterpart to FORCE."""

    lam: float = 3.0
    use_ter: bool = False
    ter_max: Optional[float] = None
    eps: float = 1e-10

    def __post_init__(self) -> None:
        _validate_force_parameters(
            self.lam, 5, self.use_ter, self.ter_max, self.eps
        )

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_float_matrix(X)
        quantiles = np.quantile(
            matrix, P2_PROBABILITIES, axis=0, method="linear"
        )
        _, _, _, lower, upper = _compute_bounds_from_quantiles(
            quantiles,
            lambda_scale=self.lam,
            use_ter=self.use_ter,
            ter_max=self.ter_max,
            epsilon=self.eps,
        )
        return _compute_trimmed_corr_numba(matrix, lower, upper)

    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        return self(X)
