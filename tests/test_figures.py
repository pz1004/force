"""Semantic safeguards for measured figures, using explicit unit-test fixtures."""

from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

import generate_breakdown_figure as diagnostics_plot
import generate_figures as benchmark_plot


def benchmark_rows():
    return pd.DataFrame([
        dict(dataset="fixture", algorithm="FORCE", run_id=i, time_ms=time, rmse=error,
             protocol="paper", mode="batch", comparison_target="fixture_reference",
             n_samples=100, n_features=2, data_sha256="fixture-data",
             reference_sha256="fixture-reference", replicate_kind="fixed_dataset_timing",
             estimator_parameters='{"ter_max": null, "use_ter": true}')
        for i, time, error in [(0, 2., .3), (1, 4., .5)]
    ])


@pytest.mark.parametrize("column,value", [
    ("protocol", "legacy"), ("mode", "online"),
    ("comparison_target", "different_reference"),
    ("estimator_parameters", '{"ter_max": 3, "use_ter": true}'),
])
def test_benchmark_rejects_mixed_identity(column, value):
    frame = benchmark_rows()
    frame.loc[1, column] = value
    with pytest.raises(ValueError, match="Mixed"):
        benchmark_plot.standardize_input_schema(frame)


def test_benchmark_summary_preserves_identity_and_measured_points(tmp_path, monkeypatch):
    frame = benchmark_plot.standardize_input_schema(benchmark_rows())
    summary = benchmark_plot.compute_summary(frame)
    assert summary.loc[0, "protocol"] == "paper"
    assert summary.loc[0, "comparison_target"] == "fixture_reference"
    figures = []
    monkeypatch.setattr(benchmark_plot, "_save_all_formats", lambda fig, *_: figures.append(fig))
    benchmark_plot.figure2_estimator_landscape(summary, ["FORCE"], tmp_path)
    ax = figures[0].axes[0]
    np.testing.assert_allclose(ax.collections[0].get_offsets(), [[3., .4]])
    assert "RMSE" in ax.get_ylabel()
    assert len(ax.lines) == 0  # No invented 25% horizontal threshold.


@pytest.mark.parametrize("column,value", [
    ("comparison_target", "different_reference"), ("n_samples", 1000),
    ("n_features", 3), ("data_sha256", "other-data"),
    ("reference_sha256", "other-reference"), ("data_identity", "other-data-id"),
])
def test_benchmark_rejects_cross_method_comparison_mismatch(column, value):
    frame = benchmark_rows()
    frame["data_identity"] = "data-id"
    other = frame.copy()
    other["algorithm"] = "Pearson"
    other[column] = value
    with pytest.raises(ValueError, match="Mixed"):
        benchmark_plot.standardize_input_schema(pd.concat([frame, other], ignore_index=True))


def test_benchmark_allows_estimator_specific_settings():
    frame = benchmark_rows()
    other = frame.copy()
    other["algorithm"] = "Pearson"
    other["estimator_parameters"] = '{"class": "PearsonEstimator"}'
    result = benchmark_plot.standardize_input_schema(pd.concat([frame, other], ignore_index=True))
    assert set(result["method"]) == {"FORCE", "Pearson"}


@pytest.mark.parametrize("declared", [True, False])
def test_fixed_or_unspecified_repetitions_have_no_rmse_interval(declared, tmp_path, monkeypatch):
    frame = benchmark_rows()
    if not declared:
        frame = frame.drop(columns="replicate_kind")
    summary = benchmark_plot.compute_summary(benchmark_plot.standardize_input_schema(frame))
    assert np.isnan(summary.loc[0, "rmse_ci"])
    assert summary.loc[0, "time_ci"] == pytest.approx(1.96)
    figures = []
    monkeypatch.setattr(benchmark_plot, "_save_all_formats", lambda fig, *_: figures.append(fig))
    benchmark_plot.figure3_rmse_comparison(summary, ["fixture"], ["FORCE"], tmp_path)
    ax = figures[0].axes[0]
    assert ax.containers[0].errorbar is None
    assert "no RMSE confidence interval" in ax.get_title()
    assert ax.patches[0].get_height() == pytest.approx(.4)


@pytest.mark.parametrize("kind", ["independent_data", "independent_order"])
def test_rmse_interval_requires_declared_unique_replicates(kind):
    frame = benchmark_rows()
    frame["replicate_kind"] = kind
    with pytest.raises(ValueError, match="replicate_id"):
        benchmark_plot.standardize_input_schema(frame)
    frame["replicate_id"] = [10, 11]
    frame["data_sha256"] = ["sample-10", "sample-11"]
    summary = benchmark_plot.compute_summary(benchmark_plot.standardize_input_schema(frame))
    assert summary.loc[0, "rmse_ci"] == pytest.approx(.196)
    assert summary.loc[0, "rmse_ci_basis"] == "normal95_" + kind
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark_plot.standardize_input_schema(pd.concat([frame, frame.iloc[:1]]))


def test_independent_methods_must_share_replicate_and_data_identities():
    frame = benchmark_rows()
    frame["replicate_kind"] = "independent_data"
    frame["replicate_id"] = [10, 11]
    frame["data_sha256"] = ["sample-10", "sample-11"]
    other = frame.copy()
    other["algorithm"] = "Pearson"
    benchmark_plot.standardize_input_schema(pd.concat([frame, other]))
    other.loc[0, "data_sha256"] = "different-sample"
    with pytest.raises(ValueError, match="data_sha256"):
        benchmark_plot.standardize_input_schema(pd.concat([frame, other]))
    other.loc[0, "replicate_id"] = 12
    with pytest.raises(ValueError, match="replicate identities"):
        benchmark_plot.standardize_input_schema(pd.concat([frame, other]))


@pytest.mark.parametrize("column", ["order_sha256", "order_seed"])
def test_order_comparisons_require_matched_permutations(column):
    frame = benchmark_rows()
    frame["replicate_kind"] = "independent_order"
    frame["replicate_id"] = [10, 11]
    frame["base_data_sha256"] = "base-sample"
    frame[column] = [100, 101]
    other = frame.copy()
    other["algorithm"] = "Pearson"
    benchmark_plot.standardize_input_schema(pd.concat([frame, other]))
    other.loc[0, column] = 999
    with pytest.raises(ValueError, match=column):
        benchmark_plot.standardize_input_schema(pd.concat([frame, other]))


@pytest.mark.parametrize("column", ["base_data_sha256", "reference_sha256"])
def test_order_replicates_keep_dataset_and_reference_fixed(column):
    frame = benchmark_rows()
    frame["replicate_kind"] = "independent_order"
    frame["replicate_id"] = [10, 11]
    frame["base_data_sha256"] = "base-sample"
    frame.loc[1, column] = "different"
    with pytest.raises(ValueError, match=column):
        benchmark_plot.standardize_input_schema(frame)


def test_archival_metadata_is_qualified_without_inventing_targets(tmp_path, monkeypatch):
    frame = benchmark_rows().drop(columns=["comparison_target", "data_sha256", "replicate_kind"])
    with pytest.warns(UserWarning, match="archival comparisons are not verified"):
        frame = benchmark_plot.standardize_input_schema(frame)
    assert "comparison_target" not in frame
    assert frame["identity_status"].eq("archival_partially_identified").all()
    summary = benchmark_plot.compute_summary(frame)
    assert summary["rmse_ci"].isna().all()
    fig = plt.figure()
    monkeypatch.setattr(fig, "savefig", lambda *_args, **_kwargs: None)
    benchmark_plot._save_all_formats(fig, tmp_path, "fixture", summary)
    assert any("comparability is unverified" in item.get_text() for item in fig.texts)
    plt.close("all")


@pytest.mark.parametrize("field,value", [
    ("estimate_status", "partial"),
    ("timing_status", "not_warmed"),
    ("requested_runs", 3),
])
def test_benchmark_figures_do_not_hide_coverage_or_unattempted_runs(field, value):
    frame = benchmark_rows()
    frame[field] = value
    with pytest.raises(ValueError, match="coverage|warm-up|requested runs"):
        benchmark_plot.standardize_input_schema(frame)


def test_benchmark_compares_the_same_pair_universe():
    frame = benchmark_rows()
    frame["eligible_mask_sha256"] = "all-pairs"
    other = frame.copy()
    other["algorithm"] = "Pearson"
    other["eligible_mask_sha256"] = "subset"
    with pytest.raises(ValueError, match="eligible_mask"):
        benchmark_plot.standardize_input_schema(pd.concat([frame, other]))


def diagnostic_report():
    results = []
    for seed, errors in [(10, (.1, .5)), (11, (.3, .9))]:
        for magnitude, error in zip((10., 1e6), errors):
            results.append(dict(
                protocol="paper", status="completed", scenario="one_sided_replacement",
                mode="batch", fraction_parameter=.02, fraction_unit="rows",
                observation_order="random", bound_policy="uncapped",
                contamination_magnitude=magnitude, seed=seed, run_id=seed - 10,
                estimator="ForceEstimator", quantile_backend="production_P2",
                execution_schedule="two_pass_final_bounds",
                comparison_target="same_sample_exact_quantile_trimmed_Pearson",
                clean_reference_target="known_clean_population_Pearson",
                n_samples=100, n_features=2, policy_parameters={"ter_max": None},
                clean_target_rmse=error,
            ))
    return dict(status="completed", protocol="paper", results=results)


def test_diagnostic_figure_rejects_partial_scores_even_when_they_are_finite():
    report = diagnostic_report()
    for row in report["results"]:
        row["clean_target_score"] = {"status": "partial", "rmse": .1}
    with pytest.raises(ValueError, match="coverage"):
        diagnostics_plot.diagnostic_frame(report)


@pytest.mark.parametrize("mutation", ["failed", "mixed_policy", "duplicate_seed", "wrong_target"])
def test_diagnostic_plot_rejects_misleading_inputs(mutation):
    report = diagnostic_report()
    if mutation == "failed":
        report["results"][0]["status"] = "failed"
    elif mutation == "mixed_policy":
        report["results"][0]["policy_parameters"] = {"ter_max": 3}
    elif mutation == "duplicate_seed":
        duplicate = deepcopy(report["results"][0])
        duplicate["run_id"] = 100
        report["results"].append(duplicate)
    else:
        report["results"][0]["clean_reference_target"] = "full_sample_proxy"
    with pytest.raises(ValueError):
        diagnostics_plot.diagnostic_frame(report)


def test_diagnostic_plot_uses_reported_values_without_robustness_threshold(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *_args, **_kwargs: figures.append(fig))
    paths = diagnostics_plot.create_diagnostic_figures(diagnostic_report(), tmp_path)
    assert [path.suffix for path in paths] == [".png", ".pdf"]
    assert len(figures) == 2
    ax = figures[0].axes[0]
    assert len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [10., 1e6])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [.2, .7])
    assert "2% of rows" in ax.get_title()
    assert "n=2 seeds" in ax.lines[0].get_label()
    assert "clean population Pearson" in ax.get_ylabel()
    plt.close("all")
