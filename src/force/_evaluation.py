"""Correlation accuracy with explicit estimability and comparison coverage.

Finite zero-filled compatibility outputs do not imply estimable correlations.
Callers supply the validity of each estimate and its independently defined
reference. Only jointly defined, off-diagonal pairs enter a comparison.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _valid_mask(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape or mask.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean matrix matching the correlations.")
    if not np.array_equal(mask, mask.T):
        raise ValueError(f"{name} must be symmetric.")
    return mask


def _mask_identity(mask: np.ndarray) -> str:
    digest = hashlib.sha256(np.asarray(mask.shape, dtype="<i8").tobytes())
    digest.update(np.packbits(mask.ravel(), bitorder="little").tobytes())
    return digest.hexdigest()


def estimator_pair_validity(estimator: Any, correlation: np.ndarray):
    """Return declared estimator validity, never inferred from zero-filled output."""
    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation must be a square matrix.")
    codes = getattr(estimator, "pair_status_codes_", None)
    if codes is None:
        return np.zeros(matrix.shape, dtype=bool), "validity_unavailable"
    codes = np.asarray(codes)
    if codes.shape != matrix.shape or codes.dtype.kind not in "iu":
        raise ValueError(
            "Pair status codes must be an integer matrix matching the output."
        )
    if np.any((codes < 0) | (codes > 2)) or not np.array_equal(codes, codes.T):
        raise ValueError("Pair status codes must be symmetric and in {0,1,2}.")
    # Preserve declarations so correlation_score can reject a contradictory
    # nonfinite value rather than silently treating arithmetic failure as missing.
    return codes == 0, "pair_status_codes"


def correlation_score(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    estimate_valid: np.ndarray,
    reference_valid: np.ndarray | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Score a declared pair universe, retaining its coverage and identity.

    The reference-valid finite upper triangle defines the eligible universe.
    ``reference_valid=None`` declares all finite reference coefficients valid;
    it is appropriate for a supplied known population matrix, not for an
    estimator whose degeneracy information was discarded. ``estimate_valid``
    is always required. A true validity entry with a nonfinite coefficient is
    rejected rather than silently reclassified.

    With ``require_complete=True``, any missing eligible coefficient makes the
    primary RMSE unavailable. Otherwise a partial score is explicitly conditional
    on the jointly defined set identified by ``comparison_mask_sha256``. Methods
    must share this set for a direct comparison of their partial scores.
    Defined-estimate counts and coverage use the reference-eligible universe,
    excluding coefficients for which the reference itself is undefined.
    """
    left = np.asarray(estimate, dtype=float)
    right = np.asarray(reference, dtype=float)
    if left.ndim != 2 or left.shape[0] != left.shape[1] or left.shape != right.shape:
        raise ValueError(
            "Estimate and reference must be equally sized square matrices."
        )
    upper = np.triu(np.ones(left.shape, dtype=bool), k=1)
    estimated = _valid_mask(estimate_valid, left.shape, "estimate_valid")
    referenced = (
        np.isfinite(right)
        if reference_valid is None
        else _valid_mask(reference_valid, right.shape, "reference_valid")
    )
    if not np.array_equal(referenced, referenced.T):
        raise ValueError("Reference validity must be symmetric.")
    if np.any(estimated & ~np.isfinite(left)):
        raise ValueError("A correlation declared estimable is nonfinite.")
    if np.any(referenced & ~np.isfinite(right)):
        raise ValueError("A reference correlation declared estimable is nonfinite.")
    eligible = upper & referenced
    compared = eligible & estimated
    total = int(upper.sum())
    eligible_count = int(eligible.sum())
    compared_count = int(compared.sum())
    if compared_count == 0:
        status = "no_comparable_pairs"
        reason = "no_reference_pairs" if eligible_count == 0 else "no_defined_pairs"
    elif compared_count < eligible_count:
        status = "incomplete" if require_complete else "partial"
        reason = "undefined_estimate_pairs"
    else:
        status, reason = "complete", None
    rmse = None
    if compared_count and (not require_complete or compared_count == eligible_count):
        difference = left[compared] - right[compared]
        rmse = float(np.sqrt(np.mean(difference * difference)))
        if not np.isfinite(rmse):
            raise FloatingPointError("Correlation error exceeded the supported range.")
    return {
        "rmse": rmse,
        "status": status,
        "reason": reason,
        "total_pair_count": total,
        "eligible_pair_count": eligible_count,
        "reference_defined_pair_count": eligible_count,
        "estimate_defined_pair_count": compared_count,
        "compared_pair_count": compared_count,
        "coverage": compared_count / eligible_count if eligible_count else None,
        "comparison_mask_sha256": _mask_identity(compared),
        "eligible_mask_sha256": _mask_identity(eligible),
        "require_complete": bool(require_complete),
    }
