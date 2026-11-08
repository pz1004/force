"""Check diagnostic targets/oracles and bounded source behavior, not paper decimals."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from force.diagnostics import (
    POLICIES,
    diagnose_batch,
    diagnose_online,
    reference_pairwise,
    reference_quantiles,
    stress_sample,
)

ROOT = Path(__file__).resolve().parents[1]


def test_reference_uses_pair_subset_means_and_reports_counts():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    accepted = X <= 1
    result, counts = reference_pairwise(X, accepted)
    assert result[0, 1] == pytest.approx(-1 / 3)
    np.testing.assert_array_equal(counts, np.full((2, 2), 4))


@pytest.mark.parametrize("policy", POLICIES.values(), ids=POLICIES.keys())
def test_batch_arithmetic_matches_independent_subset_reference(policy):
    X, truth, labels = stress_sample(
        n_samples=300,
        n_features=3,
        seed=19,
        fraction=0.10,
        magnitude=20,
        scenario="one_sided_replacement",
        order="random",
    )
    result = diagnose_batch(
        X, policy=policy, clean_reference=truth, contaminated_rows=labels
    )
    assert result["production_bound_oracle_max_error"] < 1e-10
    assert result["production_oracle_max_error"] < 1e-10
    assert result["comparison_target"] == "same_sample_exact_quantile_trimmed_Pearson"
    assert result["clean_reference_target"] != result["comparison_target"]
    assert np.shape(result["quantile_rank_interval_error"]) == (5, 3)
    # Approximation error may be large: correctness does not promise P² accuracy.
    assert np.isfinite(result["approximation_rmse"])
    assert result["retained_contaminated_pair_counts"] is not None


def test_rank_interval_handles_ties_without_spurious_rank_failure():
    result = diagnose_batch(np.ones((100, 2)), policy=POLICIES["fixed"])
    np.testing.assert_array_equal(
        result["quantile_rank_interval_error"], np.zeros((5, 2))
    )


@pytest.mark.parametrize("mode", ["batch", "online"])
def test_undefined_pairs_have_no_accuracy_score_or_statistical_eigenvalue(mode):
    X = np.ones((60, 3))
    result = (
        diagnose_batch(X, clean_reference=np.eye(3))
        if mode == "batch"
        else diagnose_online(X, window_size=20, clean_reference=np.eye(3))
    )
    assert result["status"] == result["execution_status"] == "completed"
    assert result["estimate_status"] == "undefined"
    assert result["approximation_rmse"] is None
    assert result["clean_target_rmse"] is None
    assert result["exact_clean_target_rmse"] is None
    assert result["approximation_score"]["compared_pair_count"] == 0
    assert result["production_oracle_max_error"] is None
    assert result["compatibility_output_oracle_max_error"] == 0
    assert result["minimum_eigenvalue"] is None
    assert result["compatibility_minimum_eigenvalue"] == 1
    assert result["production_validity_matches_oracle"]


def test_reference_validity_distinguishes_counts_and_variance_independently():
    X = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 3.0, 2.0]])
    mask = np.ones(X.shape, dtype=bool)
    mask[1:, 2] = False
    _, counts, codes = reference_pairwise(X, mask, return_status=True)
    assert counts[0, 1] == 3 and codes[0, 1] == 2
    assert counts[1, 2] == 1 and codes[1, 2] == 1
    assert codes[1, 1] == 0
    _, _, pair_codes = reference_pairwise(
        X[:2, 1:], np.ones((2, 2), dtype=bool), return_status=True
    )
    assert np.all(pair_codes == 0)


def test_mixed_defined_pairs_have_explicit_partial_coverage():
    x = np.linspace(-1.0, 1.0, 100)
    result = diagnose_batch(np.column_stack([x, x, np.ones(100)]))
    assert result["estimate_status"] == "partial"
    assert result["approximation_score"]["compared_pair_count"] == 1
    assert result["approximation_score"]["eligible_pair_count"] == 1
    assert result["approximation_rmse"] == pytest.approx(0)
    assert result["minimum_eigenvalue"] is None


def test_real_crossed_iqr_is_disclosed_without_scoring_an_undefined_pair():
    raw = np.random.default_rng(0).standard_cauchy((50, 2000))
    fixture = np.column_stack([raw[:, 437], np.linspace(-1.0, 1.0, 50)])
    result = diagnose_batch(fixture)
    assert result["crossed_iqr"] == [True, False]
    assert not any(result["zero_iqr"])
    assert result["quantile_crossing_count"] > 0
    assert result["estimate_status"] == "undefined"
    assert result["approximation_rmse"] is None
    assert result["approximation_score"]["reason"] == "no_defined_pairs"


def test_independent_quantile_reference_handles_finite_opposite_endpoints():
    m = np.finfo(float).max
    result = reference_quantiles(np.array([[-m], [m]]))
    assert result[2, 0] == 0
    assert np.isfinite(result).all()


def test_stress_magnitude_and_order_are_paired_independent_axes():
    keywords = dict(
        n_samples=200,
        n_features=2,
        seed=42,
        fraction=0.02,
        scenario="one_sided_replacement",
        order="random",
    )
    small, truth, labels = stress_sample(magnitude=10, **keywords)
    large, same_truth, same_labels = stress_sample(magnitude=1e6, **keywords)
    np.testing.assert_array_equal(labels, same_labels)
    np.testing.assert_array_equal(truth, same_truth)
    np.testing.assert_array_equal(small[~labels], large[~labels])
    assert labels.sum() == 4
    np.testing.assert_array_equal(large[labels, 0], np.full(4, 1e6))
    ordered, _, ordered_labels = stress_sample(
        magnitude=1e6, **{**keywords, "order": "outliers_first"}
    )
    assert np.all(ordered_labels[:4]) and not np.any(ordered_labels[4:])
    # Reordering changes neither the multiset nor which values are corrupted.
    np.testing.assert_array_equal(np.sort(ordered[:, 0]), np.sort(large[:, 0]))


@pytest.mark.parametrize("policy", POLICIES.values(), ids=POLICIES.keys())
def test_online_reference_matches_historical_acceptance_and_chunking(policy):
    X = np.random.default_rng(7).normal(size=(137, 3))
    X[90:100, 0] += 15
    a = diagnose_online(X, window_size=40, policy=policy, chunk_size=7)
    b = diagnose_online(X, window_size=40, policy=policy, chunk_size=53)
    assert a["production_count_oracle_max_error"] == 0
    assert a["production_bound_oracle_max_error"] < 1e-10
    assert a["production_oracle_max_error"] < 1e-10
    assert a["calibration_rows"] == 40
    assert a["windows"][-1]["estimation_stop"] == 137
    assert a["comparison_target"] == "same_schedule_exact_previous_window_quantiles"
    np.testing.assert_allclose(
        a["production_correlation"], b["production_correlation"], atol=1e-12
    )
    np.testing.assert_array_equal(a["accepted_pair_counts"], b["accepted_pair_counts"])


def test_scalability_source_is_replayable_and_partition_invariant():
    spec = importlib.util.spec_from_file_location(
        "scalability_diagnostic", ROOT / "scripts/run_scalability_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    small_chunks = list(module.generated_source(113, 3, 11, 17))
    assert max(len(chunk) for chunk in small_chunks) <= 17
    assert sum(len(chunk) for chunk in small_chunks) == 113
    np.testing.assert_array_equal(
        np.concatenate(small_chunks),
        np.concatenate(list(module.generated_source(113, 3, 11, 31))),
    )


@pytest.mark.parametrize("mode", ["replay", "online"])
def test_scalability_case_reports_stream_storage_separately(mode):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_scalability_check.py"),
            "--case",
            "--mode",
            mode,
            "--runs",
            "1",
            "--n-samples",
            "120",
            "--n-features",
            "3",
            "--chunk-size",
            "17",
            "--window-size",
            "20",
        ],
        cwd=ROOT,
        env={
            **os.environ,
            "NUMBA_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "completed", payload
    assert payload["input_bytes"] == 0
    assert payload["logical_input_bytes"] == 120 * 3 * 8
    assert payload["source_chunk_bytes"] == 17 * 3 * 8
    assert payload["timing_scope"] == "source_generation_and_fit"
    assert payload["output_bytes"] == 3 * 3 * 8
    assert (
        payload["post_warmup_traced_peak_bytes"]
        >= payload["post_warmup_traced_final_bytes"]
        >= 0
    )
    assert (
        payload["traced_memory_scope"] == "separate_untimed_source_generation_and_fit"
    )
    assert "native/Numba" in payload["memory_interpretation"]
    if sys.platform.startswith("linux"):
        assert (
            payload["post_warmup_sampled_peak_rss_kib"]
            >= payload["post_warmup_rss_baseline_kib"]
        )
