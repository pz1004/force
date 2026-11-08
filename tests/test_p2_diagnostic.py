"""Diagnostic references remain meaningful for tied and extreme finite samples."""

import numpy as np
import pytest

from force.diagnostics import quantile_rank_interval_error, reference_quantiles
from scripts import run_p2_skewness_diagnostic as diagnostic


def run_sample(values, probabilities, repetitions=1):
    values = np.asarray(values, dtype=float)
    return diagnostic.run_diagnostic(
        distribution="regression",
        sampler=lambda n: values.copy(),
        probabilities=probabilities,
        sample_sizes=[len(values)],
        repetitions=repetitions,
        protocol="paper",
    )


def test_constant_quantiles_do_not_receive_spurious_rank_error():
    rows = run_sample(np.ones(20), (0.01, 0.5, 0.99))
    assert len(rows) == 3
    for row in rows:
        assert row["status"] == "completed"
        assert row["mean_absolute_error"] == row["mean_rank_error"] == 0.0
        assert row["rank_error_definition"] == "distance_to_empirical_rank_interval"


def test_tied_empirical_rank_interval_uses_both_endpoints():
    # The ranks occupied by zero are [0, 3/6], and by one [3/6, 5/6].
    values = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0])[:, None]
    error = quantile_rank_interval_error(
        values,
        np.array([[0.0], [0.0], [1.0], [1.5]]),
        [0.2, 0.75, 0.75, 0.75],
    )
    np.testing.assert_allclose(error[:, 0], [0.0, 0.25, 0.0, 1 / 12])


def test_probability_generator_is_reused_across_repetitions_and_sizes():
    rows = diagnostic.run_diagnostic(
        distribution="constant",
        sampler=np.ones,
        probabilities=(p for p in [0.2, 0.8]),
        sample_sizes=(n for n in [10, 20]),
        repetitions=2,
        protocol="paper",
    )
    assert [(row["n_samples"], row["probability"]) for row in rows] == [
        (10, 0.2),
        (10, 0.8),
        (20, 0.2),
        (20, 0.8),
    ]
    assert all(row["mean_rank_error"] == 0 for row in rows)


def test_independent_exact_reference_handles_opposite_finite_endpoints():
    maximum = np.finfo(float).max
    values = np.array([-maximum, -maximum, maximum, maximum, maximum])[:, None]
    expected = np.array([-maximum, -maximum, maximum, maximum, maximum])
    np.testing.assert_array_equal(
        reference_quantiles(values, (p for p in [0, 0.25, 0.5, 0.75, 1]))[:, 0],
        expected,
    )
    row = run_sample([-maximum, maximum], [0.5])[0]
    assert row["mean_absolute_error"] == row["mean_rank_error"] == 0


def test_finite_absolute_errors_are_averaged_without_sum_overflow():
    maximum = np.finfo(float).max
    # At N=5 P² reads the bootstrap median M; the linear .375-quantile is zero.
    row = run_sample([-maximum, -maximum, maximum, maximum, maximum], [0.375], 2)[0]
    assert row["mean_absolute_error"] == maximum
    assert row["status"] == "completed"


def test_unrepresentable_absolute_error_cannot_be_reported_completed():
    maximum = np.finfo(float).max
    # The individual coefficient error is 2*M, which has no finite float64 value.
    with pytest.raises(FloatingPointError, match="absolute error"):
        run_sample([-maximum, -maximum, maximum, maximum, maximum], [0.25])


@pytest.mark.parametrize("estimate", [np.nan, np.inf, -np.inf])
def test_nonfinite_estimate_cannot_be_reported_completed(monkeypatch, estimate):
    monkeypatch.setattr(diagnostic, "_estimate", lambda *args: estimate)
    with pytest.raises(FloatingPointError, match="Quantile estimate"):
        run_sample(np.arange(20), [0.5])


@pytest.mark.parametrize("probabilities", [[], [-0.1], [1.1], [np.nan], [[0.5]]])
def test_independent_reference_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError, match="Probabilities"):
        reference_quantiles(np.arange(10)[:, None], probabilities)
