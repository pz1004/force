"""Causal previous-window FORCE with cumulative accepted-pair moments.

This is the delayed-window alternative described in Remark 9 / Equation 14,
not the final-bound two-pass estimator. Calibration observations are excluded
from the estimate, and later acceptance decisions are never revised.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any, Optional
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._moments import PairMoments, STATUS_LABELS, status_labels_from_codes
from ._quantiles import P2State, _as_real_array
from .core import (
    _bound_status,
    _compute_bounds_from_quantiles,
    _validate_force_parameters,
)


class StreamingForceEstimator:
    """Estimate correlation using frozen bounds from the previous window.

    The first ``window_size`` observations calibrate five P² trackers per
    feature and do not update correlation moments. Each subsequent logical
    window is accepted against the preceding window's bounds, while fresh P²
    trackers learn bounds for the next window. Pair moments are cumulative.
    Calls to :meth:`update_batch` do not define the logical window boundaries.
    Schedule and bound parameters are read-only after construction; create a
    new estimator to change them.

    Each fixed-size window restarts P², so quantile error need not vanish as
    the total stream grows. The cumulative historical-acceptance target does
    not automatically equal Proposition 1's population-bound batch target.
    Under iid sampling, fixed window size B and an almost-surely finite bound
    rule, its stationary pair target weights the population by the probability
    that an independent calibration block's bounds accept each observation.
    Positive acceptance, finite selected moments and positive selected variances
    are required. Adjacent window contributions are dependent. This target does
    not imply estimation of current-regime correlation under drift.
    Windows of five observations only bootstrap P² and are rejected. Windows
    below 50 warn about the small-sample regime discussed in paper Remark 3;
    allowing a larger window is not an accuracy guarantee.

    ``finalize()`` is a non-destructive snapshot. It returns ``None`` until
    at least one estimation observation has arrived. Undefined pairs then use
    zero off-diagonal correlation and a unit diagonal, with accepted counts
    and reasons available separately. Pairwise rejection does not guarantee
    a positive-semidefinite output.

    Retained numerical state is O(p² + p), independent of stream length and
    window size. Input validation/conversion uses O(Bp) workspace for a chunk
    of B observations. A finite TER cap is an optional bound policy, not a
    proof of a breakdown guarantee for the complete streaming estimator.

    Invalid input is rejected before incorporating any row of its call. An
    arithmetic failure during processing invalidates the estimator; create
    a fresh instance rather than continuing from partially updated state.
    """

    def __init__(
        self,
        window_size: int = 1000,
        lambda_scale: float = 3.0,
        use_ter: bool = True,
        ter_max: Optional[float] = None,
        epsilon: float = 1e-10,
    ) -> None:
        if (
            not isinstance(window_size, Integral)
            or isinstance(window_size, (bool, np.bool_))
            or window_size <= 5
        ):
            raise ValueError(
                "window_size must be an integer of at least 6; five rows only "
                "bootstrap P² and yield zero IQR."
            )
        _validate_force_parameters(lambda_scale, 5, use_ter, ter_max, epsilon)
        if window_size < 50:
            warnings.warn(
                "window_size < 50 repeatedly restarts P² in the small-sample "
                "regime discouraged by paper Remark 3; larger windows do not "
                "guarantee accuracy.",
                UserWarning,
                stacklevel=2,
            )
        self._window_size = int(window_size)
        self._lambda_scale = float(lambda_scale)
        self._use_ter = bool(use_ter)
        self._ter_max = None if ter_max is None else float(ter_max)
        self._epsilon = float(epsilon)
        self.n_features_in_: Optional[int] = None
        self.n_samples_seen_ = 0
        self.bound_version_ = 0
        self._quantile_state: Optional[P2State] = None
        self._pair_moments: Optional[PairMoments] = None
        self._bounds: Optional[tuple[NDArray[np.float64], NDArray[np.float64]]] = None
        self._quantiles: Optional[NDArray[np.float64]] = None
        self._location: Optional[NDArray[np.float64]] = None
        self._scale: Optional[NDArray[np.float64]] = None
        self._ter: Optional[NDArray[np.float64]] = None
        self._failed = False

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def lambda_scale(self) -> float:
        return self._lambda_scale

    @property
    def use_ter(self) -> bool:
        return self._use_ter

    @property
    def ter_max(self) -> Optional[float]:
        return self._ter_max

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def status_(self) -> str:
        """Lifecycle status, including explicit invalidation after failure."""
        if self._failed:
            return "failed"
        if self.n_samples_seen_ < self.window_size:
            return "calibrating"
        if self.n_samples_seen_ == self.window_size:
            return "calibrated_no_estimates"
        return "estimating"

    @property
    def calibration_count_(self) -> int:
        return min(self.n_samples_seen_, self.window_size)

    @property
    def estimation_count_(self) -> int:
        return max(0, self.n_samples_seen_ - self.window_size)

    @property
    def window_count_(self) -> int:
        """Rows seen in the current logical window (zero at a boundary)."""
        return self.n_samples_seen_ % self.window_size

    def _ensure_usable(self) -> None:
        if self._failed:
            raise RuntimeError(
                "StreamingForceEstimator is invalid after a processing failure; "
                "create a fresh instance."
            )

    @staticmethod
    def _numeric_array(values: ArrayLike) -> NDArray[np.float64]:
        try:
            array = _as_real_array(values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Observations must be finite real numeric values."
            ) from exc
        if not np.all(np.isfinite(array)):
            raise ValueError("Observations must contain only finite values.")
        return array

    def update(self, row: ArrayLike) -> "StreamingForceEstimator":
        """Consume one row without revisiting any previous observation."""
        self._ensure_usable()
        array = self._numeric_array(row)
        if array.ndim != 1:
            raise ValueError("update requires a one-dimensional observation.")
        return self.update_batch(array[np.newaxis, :])

    def update_batch(self, chunk: ArrayLike) -> "StreamingForceEstimator":
        """Consume a finite chunk, preserving the row-wise window schedule.

        A two-dimensional empty chunk is a no-op. It does not initialize the
        estimator, but its feature count must be valid and, once initialized,
        match the established feature count.
        """
        self._ensure_usable()
        matrix = self._numeric_array(chunk)
        if matrix.ndim != 2:
            raise ValueError("update_batch requires a two-dimensional array.")
        n_rows, n_features = matrix.shape
        if n_features < 2:
            raise ValueError("Observations must have at least 2 features.")
        if self.n_features_in_ is not None and n_features != self.n_features_in_:
            raise ValueError("Feature count cannot change after initialization.")
        if n_rows == 0:
            return self
        matrix = np.ascontiguousarray(matrix)

        if self.n_features_in_ is None:
            self._quantile_state = P2State(n_features)
            self._pair_moments = PairMoments(n_features)
            self.n_features_in_ = n_features
        assert self._quantile_state is not None
        assert self._pair_moments is not None

        offset = 0
        try:
            while offset < n_rows:
                remaining = self.window_size - self._quantile_state.count
                end = min(offset + remaining, n_rows)
                segment = matrix[offset:end]
                if self._bounds is not None:
                    self._pair_moments.update_batch(segment, *self._bounds)
                self._quantile_state.update_batch(segment)
                self.n_samples_seen_ += end - offset
                offset = end

                if self._quantile_state.count == self.window_size:
                    quantiles = self._quantile_state.quantiles.copy()
                    location, scale, ter, lower, upper = _compute_bounds_from_quantiles(
                        quantiles,
                        lambda_scale=self.lambda_scale,
                        use_ter=self.use_ter,
                        ter_max=self.ter_max,
                        epsilon=self.epsilon,
                    )
                    self._quantiles = quantiles
                    self._location, self._scale, self._ter = location, scale, ter
                    self._bounds = (lower, upper)
                    self.bound_version_ += 1
                    self._quantile_state = P2State(n_features)
        except Exception:
            # Low-level arithmetic failures may leave partially updated state.
            # Never let a subsequent snapshot present that state as an estimate.
            self._failed = True
            raise
        return self

    def finalize(self) -> Optional[NDArray[np.float64]]:
        """Return the current cumulative estimate, or None before estimation."""
        self._ensure_usable()
        if self.estimation_count_ == 0:
            return None
        return self._pair_snapshot("correlation")

    def _pair_snapshot(self, field):
        self._ensure_usable()
        assert self._pair_moments is not None
        try:
            return getattr(self._pair_moments, field)()
        except Exception:
            self._failed = True
            raise

    @property
    def accepted_counts_(self) -> Optional[NDArray[np.int64]]:
        """Joint acceptance counts; diagonal entries are marginal counts."""
        self._ensure_usable()
        if self._pair_moments is None:
            return None
        return self._pair_snapshot("accepted_counts")

    @property
    def pair_status_codes_(self) -> Optional[NDArray[np.int8]]:
        """Compact codes: 0=ok, 1=insufficient_samples, 2=zero_variance."""
        self._ensure_usable()
        if self._pair_moments is None:
            return None
        return self._pair_snapshot("status_codes")

    @property
    def pair_status_(self) -> Optional[NDArray[np.str_]]:
        """Explicit string-label snapshot; use pair_status_codes_ to save memory."""
        self._ensure_usable()
        if self._pair_moments is None:
            return None
        return self._pair_snapshot("status_labels")

    @property
    def bound_status_(self) -> Optional[dict[str, NDArray[np.bool_]]]:
        """Independent flags for the last completed window's bound policy."""
        self._ensure_usable()
        if self._quantiles is None:
            return None
        return {
            name: value.copy()
            for name, value in _bound_status(
                self._quantiles, self.use_ter, self.epsilon
            ).items()
        }

    @property
    def thresholds_(self) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Frozen bounds that will apply to the next incoming observation."""
        self._ensure_usable()
        if self._bounds is None:
            return None
        return self._bounds[0].copy(), self._bounds[1].copy()

    @property
    def quantiles_(self) -> Optional[NDArray[np.float64]]:
        """Quantiles of the last completed window, used for active bounds."""
        self._ensure_usable()
        return None if self._quantiles is None else self._quantiles.copy()

    @property
    def state_nbytes_(self) -> int:
        """Retained numeric arrays, excluding input/output and object overhead."""
        total = 0
        if self._quantile_state is not None:
            total += self._quantile_state.state_nbytes_
        if self._pair_moments is not None:
            total += self._pair_moments.state_nbytes_
        arrays = [self._quantiles, self._location, self._scale, self._ter]
        if self._bounds is not None:
            arrays.extend(self._bounds)
        return int(total + sum(a.nbytes for a in arrays if a is not None))

    def get_diagnostics(self, *, include_status_labels: bool = False) -> dict[str, Any]:
        """Return independent snapshots, using compact pair-status codes.

        Set ``include_status_labels=True`` to additionally materialize the
        full Unicode ``pair_status`` matrix. Correlation is not computed here.
        """
        self._ensure_usable()
        if not isinstance(include_status_labels, (bool, np.bool_)):
            raise ValueError("include_status_labels must be boolean.")
        counts = self.accepted_counts_
        codes = self.pair_status_codes_
        diagnostics = {
            "mode": "previous_window",
            "target": "cumulative_historical_acceptance",
            "status": self.status_,
            "n_features": self.n_features_in_,
            "n_samples_seen": self.n_samples_seen_,
            "calibration_count": self.calibration_count_,
            "estimation_count": self.estimation_count_,
            "window_size": self.window_size,
            "window_count": self.window_count_,
            "bound_version": self.bound_version_,
            "accepted_counts": counts,
            "pair_status_codes": codes,
            "pair_status_code_labels": dict(enumerate(STATUS_LABELS)),
            "bound_status": self.bound_status_,
            "quantiles": self.quantiles_,
            "thresholds": self.thresholds_,
            "location": None if self._location is None else self._location.copy(),
            "scale": None if self._scale is None else self._scale.copy(),
            "ter": None if self._ter is None else self._ter.copy(),
            "state_nbytes": self.state_nbytes_,
        }
        if include_status_labels:
            diagnostics["pair_status"] = (
                None if codes is None else status_labels_from_codes(codes)
            )
        return diagnostics
