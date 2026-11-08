"""Paper run-order and current-result inference regressions."""

import json
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scripts import run_benchmark as benchmark


def _fake_estimators(monkeypatch, *, failing=None):
    calls = []
    names = ("FORCE", "Pearson", "Spearman")

    class Estimator:
        def __init__(self, name):
            self.name = name

        def fit(self, X):
            calls.append((self, X.copy()))
            if self.name == failing:
                raise RuntimeError("fixture fit failed")
            value = float(np.tanh(X[0, 0]))
            result = np.eye(X.shape[1])
            result[0, 1] = result[1, 0] = value
            self.pair_status_codes_ = np.zeros(result.shape, dtype=np.int8)
            return result

    monkeypatch.setattr(benchmark, "build_estimators", lambda _: dict.fromkeys(names))
    monkeypatch.setattr(benchmark, "build_estimator", lambda _, name: Estimator(name))
    return calls


def test_order_replicates_share_permutation_and_reference(monkeypatch):
    calls = _fake_estimators(monkeypatch)
    X = np.arange(60, dtype=float).reshape(30, 2) / 60
    kwargs = dict(
        dataset_name="fixture",
        X=X,
        reference=np.eye(2),
        protocol="paper",
        runs=3,
        seed=91,
        replicate_mode="order",
    )
    rows = benchmark.run_dataset(**kwargs)
    frame = pd.DataFrame(rows)
    assert frame.status.eq("completed").all()
    assert frame.replicate_kind.eq("independent_order").all()
    assert frame.base_data_sha256.nunique() == frame.reference_sha256.nunique() == 1
    assert frame.order_sha256.nunique() == frame.data_sha256.nunique() == 3
    assert frame.order_seed.nunique() == 3
    assert frame.replicate_id.nunique() == 3
    assert [row["algorithm"] for row in rows[:3]] == ["FORCE", "Pearson", "Spearman"]
    assert [row["algorithm"] for row in rows[3:6]] == ["Pearson", "Spearman", "FORCE"]
    assert len({id(estimator) for estimator, _ in calls}) == 12
    for run in range(3):
        inputs = [values for _, values in calls[3 + 3 * run : 6 + 3 * run]]
        for values in inputs:
            np.testing.assert_array_equal(values, inputs[0])
            np.testing.assert_array_equal(np.sort(values[:, 0]), X[:, 0])
        records = frame[frame.run_id == run + 1]
        for field in (
            "data_sha256",
            "order_sha256",
            "order_seed",
            "replicate_id",
            "rmse",
        ):
            assert records[field].nunique() == 1
        assert records.rmse.iloc[0] == pytest.approx(abs(np.tanh(inputs[0][0, 0])))
    repeated = pd.DataFrame(benchmark.run_dataset(**kwargs))
    for field in ("data_sha256", "order_sha256", "order_seed", "replicate_id", "rmse"):
        pd.testing.assert_series_equal(frame[field], repeated[field])
    np.testing.assert_array_equal(X, np.arange(60).reshape(30, 2) / 60)


def test_fixed_repetitions_preserve_order_and_have_no_rmse_interval(monkeypatch):
    calls = _fake_estimators(monkeypatch)
    X = np.arange(60, dtype=float).reshape(30, 2)
    rows = benchmark.run_dataset(
        dataset_name="fixture",
        X=X,
        reference=np.eye(2),
        protocol="paper",
        runs=3,
        seed=7,
    )
    frame = pd.DataFrame(rows)
    for _, values in calls:
        np.testing.assert_array_equal(values, X)
    assert frame.data_sha256.nunique() == frame.order_sha256.nunique() == 1
    assert frame.order_seed.isna().all()
    for summary in benchmark._summary_records(frame):
        assert summary["ci95_rmse"] is None
        assert summary["rmse_repeatability_std"] == summary["std_rmse"]
        assert summary["rmse_ci_basis"] == "not_estimated"


def _timing_rows():
    rows = []
    for algorithm, times in (("FORCE", [10, 11, 13, 15]), ("Pearson", [7, 8, 8, 9])):
        for run, elapsed in enumerate(times, start=1):
            rows.append(
                {
                    "protocol": "paper",
                    "dataset": "fixture",
                    "algorithm": algorithm,
                    "run_id": run,
                    "replicate_id": f"order:42:{run}",
                    "seed": 42,
                    "order_seed": run + 10,
                    "order_sha256": f"order-{run}",
                    "mode": "batch",
                    "n_samples": 30,
                    "n_features": 2,
                    "comparison_target": "fixed_reference",
                    "base_data_sha256": "base",
                    "data_sha256": f"data-{run}",
                    "reference_sha256": "ref",
                    "eligible_mask_sha256": "all-pairs",
                    "reference_validity_basis": "finite_reference_declaration",
                    "replicate_mode": "order",
                    "replicate_kind": "independent_order",
                    "rmse_interpretation": "conditional_mean_error_over_random_orders",
                    "status": "completed",
                    "estimate_status": "complete",
                    "timing_status": "warmed",
                    "time_ms": elapsed,
                    "rmse": run / 10,
                }
            )
    return pd.DataFrame(rows)


def _parameterized_timing_rows():
    frame = _timing_rows()
    frame.loc[frame.algorithm == "Pearson", "algorithm"] = "Winsorized"
    parameters = {
        "FORCE": {"class": "ForceEstimator", "lambda_scale": 3.0},
        "Winsorized": {
            "class": "WinsorizedEstimator",
            "limits": [0.05, 0.05],
            "quantile_method": "linear",
        },
    }
    frame["estimator_parameters"] = frame.algorithm.map(
        {name: json.dumps(value, sort_keys=True) for name, value in parameters.items()}
    )
    return frame, parameters


@pytest.mark.parametrize(
    "summarize", [benchmark._summary_records, benchmark.paired_timing_records]
)
def test_mixed_winsorization_conventions_cannot_share_one_method_summary(summarize):
    frame, parameters = _parameterized_timing_rows()
    modified = {**parameters["Winsorized"], "quantile_method": "tail_count"}
    row = frame.index[(frame.algorithm == "Winsorized") & (frame.run_id == 2)][0]
    frame.at[row, "estimator_parameters"] = json.dumps(modified, sort_keys=True)
    with pytest.raises(ValueError, match="Mixed estimator_parameters"):
        summarize(frame)


@pytest.mark.parametrize(
    "summarize", [benchmark._summary_records, benchmark.paired_timing_records]
)
@pytest.mark.parametrize("missing", [None, np.nan, ""])
def test_declared_parameter_column_cannot_omit_a_method_identity(summarize, missing):
    frame, _ = _parameterized_timing_rows()
    frame.at[frame.index[0], "estimator_parameters"] = missing
    with pytest.raises(ValueError, match="Missing estimator_parameters"):
        summarize(frame)


def test_distinct_method_parameters_survive_summary_and_csv_roundtrip():
    frame, parameters = _parameterized_timing_rows()
    # Harmless key ordering/whitespace differences have one canonical identity.
    frame.at[frame.index[0], "estimator_parameters"] = json.dumps(
        parameters["FORCE"], indent=2
    )
    summaries = benchmark._summary_records(frame)
    paired = benchmark.paired_timing_records(frame)
    for summary in summaries:
        assert summary["estimator_parameters"] == json.dumps(
            parameters[summary["algorithm"]], sort_keys=True
        )
    assert len(paired) == 1
    assert paired[0]["force_estimator_parameters"] == json.dumps(
        parameters["FORCE"], sort_keys=True
    )
    assert paired[0]["baseline_estimator_parameters"] == json.dumps(
        parameters["Winsorized"], sort_keys=True
    )
    restored = pd.read_csv(StringIO(frame.to_csv(index=False)))
    assert benchmark._summary_records(restored) == summaries
    assert benchmark.paired_timing_records(restored) == paired
    json.dumps({"summaries": summaries, "paired_timing": paired}, allow_nan=False)


def test_paired_inference_joins_identity_instead_of_row_position():
    frame = _timing_rows().sample(frac=1, random_state=6)
    observed = benchmark.paired_timing_records(frame, requested_runs=4)[0]
    reference = stats.ttest_rel([10, 11, 13, 15], [7, 8, 8, 9])
    assert observed["usable_pairs"] == 4
    assert observed["paired_runs"] == [1, 2, 3, 4]
    assert observed["excluded_pairs"] == 0
    assert observed["mean_difference_ms"] == pytest.approx(4.25)
    assert observed["t_statistic"] == pytest.approx(reference.statistic)
    assert observed["pvalue"] == pytest.approx(reference.pvalue)
    assert observed["status"] == "ok"
    json.dumps(observed, allow_nan=False)
    frame["run_id"] = frame["run_id"].astype(str)
    assert benchmark.paired_timing_records(frame, requested_runs=4)[0] == observed


def test_paired_inference_counts_missing_failed_and_invalid_times():
    frame = _timing_rows()
    frame = frame[~((frame.algorithm == "Pearson") & (frame.run_id == 2))].copy()
    frame.loc[(frame.algorithm == "FORCE") & (frame.run_id == 3), "status"] = "failed"
    frame["time_ms"] = frame.time_ms.astype(float)
    frame.loc[(frame.algorithm == "Pearson") & (frame.run_id == 4), "time_ms"] = np.inf
    observed = benchmark.paired_timing_records(frame, requested_runs=5)[0]
    assert observed["usable_pairs"] == 1
    assert observed["paired_runs"] == [1]
    assert observed["excluded_pairs"] == 4
    assert observed["missing_force_runs"] == 1
    assert observed["missing_baseline_runs"] == 2
    assert observed["failed_force_runs"] == 1
    assert observed["invalid_baseline_times"] == 1
    assert observed["status"] == "insufficient_pairs"
    assert observed["pvalue"] is observed["t_statistic"] is None
    json.dumps(observed, allow_nan=False)


@pytest.mark.parametrize("difference", [0, 3])
def test_constant_timing_differences_leave_inference_undefined(difference):
    frame = _timing_rows()
    frame.loc[frame.algorithm == "FORCE", "time_ms"] = (
        np.array([7, 8, 8, 9]) + difference
    )
    observed = benchmark.paired_timing_records(frame)[0]
    assert observed["usable_pairs"] == 4
    assert observed["mean_difference_ms"] == difference
    assert observed["status"] == "zero_variance_differences"
    assert observed["pvalue"] is observed["t_statistic"] is None
    json.dumps(observed, allow_nan=False)


@pytest.mark.parametrize(
    "field",
    [
        "order_sha256",
        "order_seed",
        "base_data_sha256",
        "data_sha256",
        "reference_sha256",
        "eligible_mask_sha256",
        "reference_validity_basis",
        "comparison_target",
        "seed",
        "mode",
        "replicate_id",
    ],
)
def test_mismatched_pair_identity_is_rejected(field):
    frame = _timing_rows()
    row = frame.index[(frame.algorithm == "Pearson") & (frame.run_id == 2)][0]
    frame.loc[row, field] = 999 if field in {"seed", "order_seed"} else "mismatch"
    with pytest.raises(ValueError, match="Mismatched"):
        benchmark.paired_timing_records(frame)


def test_duplicate_replicates_cannot_inflate_inference():
    frame = _timing_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.paired_timing_records(pd.concat([frame, frame.iloc[:1]]))
    frame.loc[1, "replicate_id"] = frame.loc[0, "replicate_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.paired_timing_records(frame)


def test_order_interval_describes_mean_over_orders():
    frame = _timing_rows()
    expected = 1.96 * np.std([0.1, 0.2, 0.3, 0.4], ddof=1) / np.sqrt(4)
    for summary in benchmark._summary_records(frame):
        assert summary["ci95_rmse"] == pytest.approx(expected)
        assert summary["rmse_ci_basis"] == "normal95_conditional_order_mean"
        assert summary["rmse_repeatability_std"] is None
        assert (
            summary["rmse_interpretation"]
            == "conditional_mean_error_over_random_orders"
        )
    markdown = benchmark._summary_markdown(benchmark._summary_records(frame))
    assert "not uncertainty across datasets" in markdown


def test_warmup_failure_keeps_requested_pairs_explicit(monkeypatch):
    _fake_estimators(monkeypatch, failing="FORCE")
    frame = pd.DataFrame(
        benchmark.run_dataset(
            dataset_name="fixture",
            X=np.arange(40).reshape(20, 2),
            reference=np.eye(2),
            protocol="paper",
            runs=3,
            seed=7,
            replicate_mode="order",
        )
    )
    assert len(frame) == 9
    for comparison in benchmark.paired_timing_records(frame, requested_runs=3):
        assert comparison["failed_force_runs"] == 3
        assert comparison["usable_pairs"] == 0
        assert comparison["excluded_pairs"] == 3
        assert comparison["status"] == "insufficient_pairs"


def test_failed_original_order_warmup_does_not_suppress_requested_fits(monkeypatch):
    calls = []

    class OrderSensitiveEstimator:
        def fit(self, X):
            calls.append(X.copy())
            if X[0, 0] == 0:
                raise FloatingPointError("fixture rejects this order")
            self.pair_status_codes_ = np.zeros((2, 2), dtype=np.int8)
            return np.eye(2)

    monkeypatch.setattr(benchmark, "build_estimators", lambda _: {"FORCE": None})
    monkeypatch.setattr(
        benchmark, "build_estimator", lambda *_: OrderSensitiveEstimator()
    )
    rows = benchmark.run_dataset(
        dataset_name="fixture",
        X=np.arange(40.0).reshape(20, 2),
        reference=np.eye(2),
        protocol="paper",
        runs=8,
        seed=7,
        replicate_mode="order",
    )
    assert len(calls) == 9  # One warm-up and all eight requested fits, no retries.
    successful = [row for row in rows if row["status"] == "completed"]
    assert len(successful) > 1
    assert all(row["warmup_status"] == "failed" for row in rows)
    assert successful[0]["timing_status"] == "not_warmed"
    assert successful[0]["time_ms"] is None
    assert successful[0]["rmse"] == 0
    for row in successful[1:]:
        assert row["timing_status"] == "warmed"
        assert row["time_ms"] > 0
    summary = benchmark._summary_records(pd.DataFrame(rows))[0]
    assert summary["attempted_runs"] == summary["requested_runs"] == 8
    assert summary["timing_eligible_runs"] == len(successful) - 1
    assert summary["mean_time_ms"] is None


def _selective_failure_rows(*, all_failed=False):
    prototype = _timing_rows().iloc[0].to_dict()
    rows = []
    for run in range(1, 11):
        succeeded = run <= 5 and not all_failed
        rows.append(
            {
                **prototype,
                "run_id": run,
                "replicate_id": f"order:42:{run}",
                "requested_runs": 10,
                "attempted": True,
                "status": "completed" if succeeded else "failed",
                "estimate_status": "complete" if succeeded else "fit_failed",
                "timing_status": "warmed" if succeeded else "fit_failed",
                "time_ms": float(run) if succeeded else None,
                "rmse": 0.0 if succeeded else None,
            }
        )
    return pd.DataFrame(rows)


def test_failed_orders_cannot_be_presented_as_zero_unconditional_error():
    summary = benchmark._summary_records(_selective_failure_rows())[0]
    assert summary["requested_runs"] == summary["attempted_runs"] == 10
    assert summary["successful_runs"] == summary["failed_runs"] == 5
    assert summary["rmse_summary_status"] == "incomplete"
    assert summary["mean_rmse"] is summary["ci95_rmse"] is None
    assert summary["mean_time_ms"] is None
    assert summary["success_conditioned_mean_rmse"] == 0
    assert summary["success_conditioned_ci95_rmse"] == 0
    assert summary["success_conditioned_rmse_interpretation"].endswith(
        "given_complete_estimate"
    )


def test_all_failed_methods_remain_visible_in_summary():
    summary = benchmark._summary_records(_selective_failure_rows(all_failed=True))[0]
    assert summary["requested_runs"] == summary["failed_runs"] == 10
    assert summary["successful_runs"] == summary["timing_eligible_runs"] == 0
    assert summary["rmse_summary_status"] == "unavailable"
    assert summary["mean_rmse"] is summary["success_conditioned_mean_rmse"] is None
    assert summary["timing_eligible_mean_time_ms"] is None


def test_summary_accounts_for_missing_replicates_and_rejects_duplicates():
    frame = _selective_failure_rows().iloc[:-1]
    summary = benchmark._summary_records(frame)[0]
    assert summary["missing_runs"] == 1
    assert summary["attempted_runs"] == 9
    assert summary["requested_runs"] == 10
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark._summary_records(pd.concat([frame, frame.iloc[:1]]))


@pytest.mark.parametrize("constant_columns", [1, 3])
def test_primary_benchmark_score_requires_complete_pair_validity(
    monkeypatch, constant_columns
):
    from force import ForceEstimator

    monkeypatch.setattr(benchmark, "build_estimators", lambda _: {"FORCE": None})
    monkeypatch.setattr(benchmark, "build_estimator", lambda *_: ForceEstimator())
    X = np.random.default_rng(31).normal(size=(100, 3))
    X[:, -constant_columns:] = 1
    rows = benchmark.run_dataset(
        dataset_name="fixture",
        X=X,
        reference=np.eye(3),
        protocol="paper",
        runs=2,
        seed=7,
    )
    for row in rows:
        assert row["status"] == "completed"
        assert row["rmse"] is None
        assert row["eligible_pair_count"] == 3
        assert row["compared_pair_count"] == (1 if constant_columns == 1 else 0)
        assert row["estimate_status"] == (
            "incomplete" if constant_columns == 1 else "no_comparable_pairs"
        )
    summary = benchmark._summary_records(pd.DataFrame(rows))[0]
    assert summary["incomplete_runs"] == 2
    assert summary["successful_runs"] == summary["failed_runs"] == 0
    assert summary["mean_rmse"] is None


def test_missing_validity_is_not_inferred_from_finite_output(monkeypatch):
    class UnspecifiedEstimator:
        def fit(self, X):
            return np.eye(2)

    monkeypatch.setattr(benchmark, "build_estimators", lambda _: {"legacy": None})
    monkeypatch.setattr(benchmark, "build_estimator", lambda *_: UnspecifiedEstimator())
    row = benchmark.run_dataset(
        dataset_name="fixture",
        X=np.ones((10, 2)),
        reference=np.eye(2),
        protocol="legacy",
        runs=1,
        seed=7,
    )[0]
    assert row["status"] == "completed"
    assert row["estimate_status"] == "validity_unavailable"
    assert row["rmse"] is None


def test_invalid_zero_filled_reference_pairs_are_excluded(tmp_path, monkeypatch):
    from force.protocols import PreparedDataset

    _fake_estimators(monkeypatch)
    X = np.arange(90, dtype=float).reshape(30, 3) + 1
    reference = np.eye(3)
    valid = np.ones((3, 3), dtype=bool)
    valid[0, 1] = valid[1, 0] = False
    prepared = PreparedDataset(
        "fixture",
        X,
        reference,
        {"reference": "FastMCD"},
        reference_valid=valid,
        reference_validity_basis="pair_status_codes",
    )
    monkeypatch.setattr(benchmark, "prepare_dataset", lambda *args, **kwargs: prepared)
    args = benchmark.build_parser().parse_args(
        [
            "--datasets",
            "synthetic",
            "--smoke",
            "--offline",
            "--output-dir",
            str(tmp_path),
        ]
    )
    report = benchmark.execute(args)
    assert report["datasets"][0]["reference_validity_basis"] == "pair_status_codes"
    for row in report["results"]:
        assert row["reference_validity_basis"] == "pair_status_codes"
        assert row["total_pair_count"] == 3
        assert row["eligible_pair_count"] == row["compared_pair_count"] == 2
        assert row["coverage"] == 1
        assert row["estimate_status"] == "complete"
        assert row["rmse"] == 0
    unmasked = benchmark.run_dataset(
        dataset_name="fixture",
        X=X,
        reference=reference,
        protocol="paper",
        runs=1,
        seed=42,
    )[0]
    assert unmasked["reference_validity_basis"] == "finite_reference_declaration"
    assert unmasked["eligible_pair_count"] == 3
    assert unmasked["rmse"] == pytest.approx(np.tanh(X[0, 0]) / np.sqrt(3))
    assert (
        unmasked["eligible_mask_sha256"] != report["results"][0]["eligible_mask_sha256"]
    )


def test_paired_timing_excludes_unwarmed_fits_even_with_a_numeric_duration():
    frame = _timing_rows()
    frame.loc[(frame.algorithm == "FORCE") & (frame.run_id == 1), "timing_status"] = (
        "not_warmed"
    )
    record = benchmark.paired_timing_records(frame, requested_runs=4)[0]
    assert record["unwarmed_force_runs"] == 1
    assert record["usable_pairs"] == 3
    assert record["paired_runs"] == [2, 3, 4]


@pytest.mark.parametrize(
    "protocol,option,expected",
    [
        ("paper", [], "order"),
        ("legacy", [], "fixed"),
        ("paper", ["--fixed"], "fixed"),
        ("legacy", ["--replicate-mode", "order"], "order"),
    ],
)
def test_cli_resolves_replicate_policy_and_writes_current_comparisons(
    tmp_path, monkeypatch, protocol, option, expected
):
    _fake_estimators(monkeypatch)
    args = benchmark.build_parser().parse_args(
        [
            "--protocol",
            protocol,
            "--datasets",
            "synthetic",
            "--smoke",
            "--offline",
            "--output-dir",
            str(tmp_path),
            *option,
        ]
    )
    report = benchmark.execute(args)
    assert report["replicate_mode"] == expected
    assert report["status"] == "completed"
    assert len(report["paired_timing"]) == 2
    assert all(
        record["status"] == "insufficient_pairs" for record in report["paired_timing"]
    )
    assert (tmp_path / "paired_timing.md").exists()
    json.loads(
        (tmp_path / "report.json").read_text(),
        parse_constant=lambda value: pytest.fail(value),
    )
