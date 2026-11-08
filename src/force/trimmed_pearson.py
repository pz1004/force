"""Exact-quantile baselines sharing FORCE's trimming mathematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import (
    P2_PROBABILITIES,
    _as_float_matrix,
    _bound_status,
    _compute_bounds_from_quantiles,
    _validate_force_parameters,
)
from ._input import DEFAULT_CHUNK_SIZE
from ._moments import PairMoments, status_labels_from_codes
from ._quantiles import linear_quantiles


def _compute_exact_quantiles(
    X: NDArray[np.float64],
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    quantiles = linear_quantiles(X, P2_PROBABILITIES, axis=0)
    return tuple(quantiles[index] for index in range(5))  # type: ignore[return-value]


@dataclass(frozen=True)
class TrimmedPearsonExact:
    """Exact counterpart to FORCE using partition-selected linear quantiles."""

    lam: float = 3.0
    use_ter: bool = False
    ter_max: Optional[float] = None
    eps: float = 1e-10
    _pair_status_codes: Optional[NDArray[np.int8]] = field(
        default=None, init=False, repr=False, compare=False
    )
    _accepted_counts: Optional[NDArray[np.int64]] = field(
        default=None, init=False, repr=False, compare=False
    )
    _bound_flags: Optional[dict] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_force_parameters(self.lam, 5, self.use_ter, self.ter_max, self.eps)

    def __call__(self, X: ArrayLike) -> NDArray[np.float64]:
        matrix = _as_float_matrix(X)
        quantiles = linear_quantiles(matrix, P2_PROBABILITIES, axis=0)
        _, _, _, lower, upper = _compute_bounds_from_quantiles(
            quantiles,
            lambda_scale=self.lam,
            use_ter=self.use_ter,
            ter_max=self.ter_max,
            epsilon=self.eps,
        )
        moments = PairMoments(matrix.shape[1])
        for start in range(0, len(matrix), DEFAULT_CHUNK_SIZE):
            moments.update_batch(
                matrix[start : start + DEFAULT_CHUNK_SIZE], lower, upper
            )
        result = moments.correlation()
        codes, counts = moments.status_codes(), moments.accepted_counts()
        # Constructor parameters stay frozen; publish fit diagnostics only after
        # successful arithmetic, matching the array FORCE transaction contract.
        object.__setattr__(self, "_pair_status_codes", codes)
        object.__setattr__(self, "_accepted_counts", counts)
        object.__setattr__(
            self, "_bound_flags", _bound_status(quantiles, self.use_ter, self.eps)
        )
        return result

    def fit(self, X: ArrayLike) -> NDArray[np.float64]:
        return self(X)

    @property
    def pair_status_codes_(self):
        return (
            None if self._pair_status_codes is None else self._pair_status_codes.copy()
        )

    @property
    def pair_status_(self):
        codes = self.pair_status_codes_
        return None if codes is None else status_labels_from_codes(codes)

    @property
    def accepted_counts_(self):
        return None if self._accepted_counts is None else self._accepted_counts.copy()

    @property
    def bound_status_(self):
        if self._bound_flags is None:
            return None
        return {name: values.copy() for name, values in self._bound_flags.items()}
