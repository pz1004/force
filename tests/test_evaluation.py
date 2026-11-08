"""Scientific comparison coverage cannot be inferred from compatibility zeros."""

from types import SimpleNamespace

import numpy as np
import pytest

from force._evaluation import correlation_score, estimator_pair_validity


def test_primary_score_is_unavailable_for_partially_defined_estimate():
    reference = np.eye(3)
    estimate = reference.copy()
    estimate[0, 1] = estimate[1, 0] = 0.3
    valid = np.eye(3, dtype=bool)
    valid[0, 1] = valid[1, 0] = True
    primary = correlation_score(
        estimate, reference, estimate_valid=valid, require_complete=True
    )
    assert primary["status"] == "incomplete"
    assert primary["rmse"] is None
    assert primary["eligible_pair_count"] == 3
    assert primary["compared_pair_count"] == 1
    assert primary["coverage"] == pytest.approx(1 / 3)
    partial = correlation_score(estimate, reference, estimate_valid=valid)
    assert partial["status"] == "partial"
    assert partial["rmse"] == pytest.approx(0.3)
    assert partial["comparison_mask_sha256"] == primary["comparison_mask_sha256"]


def test_reference_validity_defines_the_eligible_universe():
    matrix = np.eye(3)
    reference_valid = np.eye(3, dtype=bool)
    reference_valid[0, 1] = reference_valid[1, 0] = True
    result = correlation_score(
        matrix,
        matrix,
        estimate_valid=reference_valid,
        reference_valid=reference_valid,
        require_complete=True,
    )
    assert result["status"] == "complete"
    assert result["total_pair_count"] == 3
    assert result["eligible_pair_count"] == result["compared_pair_count"] == 1
    assert result["rmse"] == 0


def test_no_estimated_pairs_cannot_be_perfect_accuracy():
    matrix = np.eye(3)
    result = correlation_score(
        matrix, matrix, estimate_valid=np.zeros((3, 3), dtype=bool)
    )
    assert result["status"] == "no_comparable_pairs"
    assert result["reason"] == "no_defined_pairs"
    assert result["rmse"] is None
    assert result["coverage"] == 0
    result = correlation_score(
        matrix,
        matrix,
        estimate_valid=np.ones((3, 3), dtype=bool),
        reference_valid=np.zeros((3, 3), dtype=bool),
    )
    assert result["reason"] == "no_reference_pairs"
    assert result["coverage"] is None


def test_comparison_identity_distinguishes_equal_size_pair_sets():
    reference = np.eye(3)
    first = np.eye(3, dtype=bool)
    first[0, 1] = first[1, 0] = True
    second = np.eye(3, dtype=bool)
    second[1, 2] = second[2, 1] = True
    a = correlation_score(reference, reference, estimate_valid=first)
    b = correlation_score(reference, reference, estimate_valid=second)
    assert a["compared_pair_count"] == b["compared_pair_count"]
    assert a["comparison_mask_sha256"] != b["comparison_mask_sha256"]
    assert a["eligible_mask_sha256"] == b["eligible_mask_sha256"]


def test_nonfinite_coefficient_cannot_claim_to_be_defined():
    estimate = np.array([[1.0, np.nan], [np.nan, 1.0]])
    estimator = SimpleNamespace(pair_status_codes_=np.zeros((2, 2), dtype=np.int8))
    valid, basis = estimator_pair_validity(estimator, estimate)
    assert basis == "pair_status_codes"
    with pytest.raises(ValueError, match="declared estimable is nonfinite"):
        correlation_score(estimate, np.eye(2), estimate_valid=valid)


def test_missing_validity_is_not_inferred_from_a_finite_matrix():
    valid, basis = estimator_pair_validity(SimpleNamespace(), np.eye(3))
    assert basis == "validity_unavailable"
    assert not valid.any()


@pytest.mark.parametrize(
    "invalid", [np.ones((2, 2)), np.array([[True, False], [True, True]])]
)
def test_invalid_comparison_masks_fail_explicitly(invalid):
    with pytest.raises(ValueError):
        correlation_score(np.eye(2), np.eye(2), estimate_valid=invalid)
