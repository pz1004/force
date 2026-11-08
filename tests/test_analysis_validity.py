"""Analysis runners do not score undefined compatibility coefficients."""

import warnings

import numpy as np

from scripts.run_convergence_analysis import run_convergence
from scripts.run_ter_contamination_analysis import _complete_mean, _mse


def test_convergence_keeps_bootstrap_degeneracy_out_of_clean_target_error():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        row = run_convergence(
            sample_sizes=[5],
            runs=1,
            n_features=3,
            contamination=0.0,
            seed=73,
            protocol="paper",
        )[0]
    assert row["status"] == "completed"
    assert row["rmse_force_vs_true"] is None
    assert row["force_clean_target_score"]["compared_pair_count"] == 0
    assert row["force_clean_target_score"]["status"] == "no_comparable_pairs"


def test_ter_analysis_does_not_impute_zero_or_drop_an_undefined_replicate():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert _mse(np.ones((20, 2)), 0.6, use_ter=True, protocol="paper") is None
    assert _complete_mean([0.1, None, 0.2]) is None
    assert _complete_mean([0.1, 0.3]) == 0.2
