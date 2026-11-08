"""The fixed-window target pools accepted moments across independent windows."""

import numpy as np
import pytest

from force.diagnostics import BoundPolicy, POLICIES, simulate_online_target


def test_online_target_pools_data_instead_of_averaging_window_correlations():
    # Each evaluation window has correlation +1; opposite shifts make the
    # pooled correlation negative. Broad calibration bounds accept every row.
    counter = 0
    evaluations = []

    def sample(rng, n):
        nonlocal counter
        counter += 1
        if counter % 2:
            x = np.linspace(-100.0, 100.0, n)
            return np.column_stack([x, x])
        x = np.linspace(-1.0, 1.0, n)
        shift = 10.0 if (counter // 2) % 2 else -10.0
        block = np.column_stack([x + shift, x - shift])
        evaluations.append(block)
        return block

    report = simulate_online_target(
        sample,
        window_size=20,
        evaluation_size=20,
        windows_per_replicate=2,
        n_replicates=2,
        policy=BoundPolicy("wide", lambda_scale=10.0, use_ter=False),
    )
    expected = np.corrcoef(np.concatenate(evaluations[:2]), rowvar=False)
    assert expected[0, 1] < -0.9
    assert report["status"] == "completed"
    np.testing.assert_allclose(
        report["replicate_mean_correlation"], expected, atol=1e-14
    )
    np.testing.assert_array_equal(
        report["monte_carlo_standard_error"], np.zeros((2, 2))
    )
    for replicate in report["replicates"]:
        np.testing.assert_array_equal(
            replicate["accepted_pair_counts"], np.full((2, 2), 40)
        )


@pytest.mark.parametrize("backend", ["p2", "exact"])
def test_online_target_independent_replication_is_reproducible(backend):
    def sample(rng, n):
        return rng.multivariate_normal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]], n)

    kwargs = dict(
        window_size=40,
        evaluation_size=60,
        windows_per_replicate=4,
        n_replicates=3,
        seed=913,
        quantile_backend=backend,
    )
    a = simulate_online_target(sample, **kwargs)
    b = simulate_online_target(sample, **kwargs)
    assert a == b
    assert a["status"] == "completed"
    assert a["usable_replicates"] == 3
    assert a["attempted_calibration_windows"] == 12
    assert a["failed_calibration_windows"] == 0
    matrices = np.array([rep["correlation"] for rep in a["replicates"]])
    np.testing.assert_allclose(a["replicate_mean_correlation"], matrices.mean(axis=0))
    np.testing.assert_allclose(
        a["monte_carlo_standard_error"], matrices.std(axis=0, ddof=1) / np.sqrt(3)
    )
    # More independent replicates reduce uncertainty around this finite-W,E
    # expectation; neither source of nonlinear target bias is thereby removed.
    assert (
        a["reported_estimand"] == "expectation_of_finite_pooled_replicate_correlation"
    )
    assert a["reported_estimand_parameters"] == {
        "calibration_blocks_per_replicate": 4,
        "evaluation_rows_per_block": 60,
    }
    assert a["reported_estimand_requires"] == [
        "calibration_success_almost_surely",
        "finite_replicate_correlation_defined_almost_surely",
    ]
    assert a["observed_success_establishes_estimand_assumptions"] is False
    assert a["possible_bias_sources"] == [
        "finite_calibration_block_count",
        "finite_evaluation_rows_per_block",
    ]
    assert a["uncertainty_basis"] == (
        "independent_replicates_standard_error_of_finite_replicate_mean"
    )
    assert a["stationary_target_error_bound"] is None


def test_target_never_drops_failed_calibration_to_claim_success():
    def sample(rng, n):
        x = np.linspace(-1.0, 1.0, n) * 1e308
        return np.column_stack([x, x])

    report = simulate_online_target(
        sample,
        window_size=100,
        evaluation_size=20,
        windows_per_replicate=2,
        n_replicates=3,
        policy=POLICIES["fixed"],
        quantile_backend="exact",
    )
    assert report["status"] == "incomplete"
    assert report["requested_calibration_windows"] == 6
    assert report["attempted_calibration_windows"] == 3
    assert report["failed_calibration_windows"] == 3
    assert report["usable_replicates"] == 0
    assert report["replicate_mean_correlation"] is None
    assert report["monte_carlo_standard_error"] is None


def test_undefined_replicates_do_not_become_identity_target_estimates():
    report = simulate_online_target(
        lambda rng, n: np.ones((n, 2)),
        window_size=20,
        evaluation_size=20,
        windows_per_replicate=2,
        n_replicates=2,
    )
    assert report["status"] == "incomplete"
    assert report["usable_replicates"] == 0
    assert report["replicate_mean_correlation"] is None
    assert all(rep["status"] == "undefined_correlation" for rep in report["replicates"])
