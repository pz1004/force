"""Reproducibility protocol and data-provenance tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from force.data import (
    ExternalDataUnavailable,
    ODDS_DATASETS,
    SP500_PREPROCESSING,
    _download_gemma_processed,
    fetch_sp500_data,
    generate_synthetic_data,
    load_odds_dataset,
)
from force.protocols import (
    DATASET_NAMES,
    DatasetNotReproducible,
    get_protocol,
    prepare_dataset,
)


def _rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    mask = np.triu(np.ones_like(estimate, dtype=bool), k=1)
    return float(np.sqrt(np.mean((estimate[mask] - reference[mask]) ** 2)))


def test_protocol_parameters() -> None:
    paper = get_protocol("paper")
    legacy = get_protocol("legacy")
    assert paper.force_parameters["exact_cutover"] == 5
    assert paper.force_parameters["ter_max"] is None
    assert paper.synthetic_parameters["n_features"] == 10
    assert paper.synthetic_parameters["contamination_distribution"] == "cauchy"
    assert legacy.synthetic_parameters["n_features"] == 50
    assert legacy.synthetic_parameters["contamination_distribution"] == "uniform"


def test_synthetic_generation_is_deterministic() -> None:
    first = generate_synthetic_data(seed=13)
    second = generate_synthetic_data(seed=13)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


@pytest.mark.parametrize("dataset", ["sp500", "genomics"])
def test_paper_omissions_are_explicit(dataset: str) -> None:
    with pytest.raises(DatasetNotReproducible):
        prepare_dataset(
            dataset,
            protocol="paper",
            seed=42,
            smoke=False,
            offline=True,
            data_dir=None,
        )


@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_every_dataset_has_a_deterministic_smoke_path(dataset: str) -> None:
    first = prepare_dataset(
        dataset,
        protocol="paper",
        seed=42,
        smoke=True,
        offline=True,
        data_dir=None,
    )
    second = prepare_dataset(
        dataset,
        protocol="paper",
        seed=42,
        smoke=True,
        offline=True,
        data_dir=None,
    )
    assert np.array_equal(first.X, second.X)
    assert first.X.shape[0] >= 5
    assert first.reference.shape == (first.X.shape[1], first.X.shape[1])


def test_genomics_smoke_reference_is_spearman():
    from force import SpearmanEstimator

    prepared = prepare_dataset(
        "genomics",
        protocol="paper",
        seed=42,
        smoke=True,
        offline=True,
        data_dir=None,
    )
    np.testing.assert_allclose(
        prepared.reference, SpearmanEstimator().fit(prepared.X), atol=1e-14
    )
    assert prepared.provenance["reference"] == "spearman"
    assert prepared.provenance["reference_basis"] == "paper_section_3.2.3"


def test_full_genomics_uses_dataset_specific_paper_reference(monkeypatch):
    def fetch(*, reference, **kwargs):
        assert reference == "spearman"
        return np.zeros((5000, 100)), np.eye(100), {"reference": reference}

    monkeypatch.setattr("force.protocols.fetch_genomics_data", fetch)
    prepared = prepare_dataset(
        "genomics",
        protocol="paper",
        seed=42,
        smoke=False,
        offline=True,
        data_dir=None,
        gemma_dataset="supplied-accession",
    )
    assert prepared.provenance["reference"] == "spearman"
    assert prepared.provenance["reference_basis"] == "paper_section_3.2.3"
    assert "Section 2.4.3" in prepared.provenance["reference_interpretation"]


def test_odds_reference_retains_invalid_pair_metadata(monkeypatch):
    X = np.arange(90, dtype=float).reshape(30, 3)
    codes = np.zeros((3, 3), dtype=np.int8)
    codes[0, 1] = codes[1, 0] = 2

    class ReferenceEstimator:
        def __init__(self, *, random_state):
            assert random_state == 42

        def fit(self, values):
            np.testing.assert_array_equal(values, X)
            self.pair_status_codes_ = codes.copy()
            return np.eye(3)

    monkeypatch.setattr("force.protocols.FastMCDEstimator", ReferenceEstimator)
    monkeypatch.setattr(
        "force.protocols.load_odds_dataset",
        lambda *args, **kwargs: (X, np.zeros(len(X)), {"source": "fixture"}),
    )
    prepared = prepare_dataset(
        "odds-mammography",
        protocol="paper",
        seed=42,
        smoke=False,
        offline=True,
        data_dir=None,
    )
    assert prepared.reference[0, 1] == 0
    np.testing.assert_array_equal(prepared.reference_valid, codes == 0)
    assert prepared.reference_validity_basis == "pair_status_codes"
    assert prepared.provenance["reference_validity_basis"] == "pair_status_codes"


def test_verified_odds_metadata() -> None:
    assert ODDS_DATASETS["mammography"]["shape"] == (11183, 6)
    assert ODDS_DATASETS["satellite"]["shape"] == (6435, 36)
    for spec in ODDS_DATASETS.values():
        assert len(spec["sha256"]) == 64
        int(spec["sha256"], 16)


def test_external_data_failure_is_not_replaced_with_synthetic_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_download(*args, **kwargs):
        raise OSError("mock network failure")

    monkeypatch.setattr("force.data.urllib.request.urlopen", fail_download)
    with pytest.raises(ExternalDataUnavailable, match="Could not download"):
        load_odds_dataset("mammography", cache_dir=tmp_path, offline=False)
    assert not (tmp_path / "odds" / "mammography.mat").exists()


def test_sp500_cached_schema_is_validated(tmp_path: Path) -> None:
    symbols = ("AAA", "BBB")
    key = hashlib.sha256(
        ("|".join(symbols) + "|2000-01-01|2025-01-01").encode("utf-8")
    ).hexdigest()[:16]
    path = tmp_path / "sp500" / f"returns-{key}.csv"
    path.parent.mkdir(parents=True)
    path.write_text("date,AAA,WRONG\n2020-01-01,0.1,0.2\n", encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": "yfinance",
                "tickers": list(symbols),
                "start": "2000-01-01",
                "end": "2025-01-01",
                "rows": 1,
                "columns": 2,
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "preprocessing": SP500_PREPROCESSING,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExternalDataUnavailable, match="ticker manifest"):
        fetch_sp500_data(
            tickers=symbols,
            cache_dir=tmp_path,
            offline=True,
        )


def test_sp500_cached_manifest_is_required(tmp_path: Path) -> None:
    symbols = ("AAA", "BBB")
    key = hashlib.sha256(
        ("|".join(symbols) + "|2000-01-01|2025-01-01").encode("utf-8")
    ).hexdigest()[:16]
    path = tmp_path / "sp500" / f"returns-{key}.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "date,AAA,BBB\n"
        "2020-01-01,0.1,0.2\n"
        "2020-01-02,0.2,0.3\n"
        "2020-01-03,0.3,0.4\n"
        "2020-01-04,0.4,0.5\n"
        "2020-01-05,0.5,0.6\n",
        encoding="utf-8",
    )
    with pytest.raises(ExternalDataUnavailable, match="provenance manifest"):
        fetch_sp500_data(
            tickers=symbols,
            cache_dir=tmp_path,
            offline=True,
        )


def test_sp500_cached_checksum_is_required_and_verified(tmp_path: Path) -> None:
    symbols = ("AAA", "BBB")
    key = hashlib.sha256(
        ("|".join(symbols) + "|2000-01-01|2025-01-01").encode("utf-8")
    ).hexdigest()[:16]
    path = tmp_path / "sp500" / f"returns-{key}.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "date,AAA,BBB\n"
        "2020-01-01,0.1,0.2\n"
        "2020-01-02,0.2,0.3\n"
        "2020-01-03,0.3,0.4\n"
        "2020-01-04,0.4,0.5\n"
        "2020-01-05,0.5,0.6\n",
        encoding="utf-8",
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": "yfinance",
                "tickers": list(symbols),
                "start": "2000-01-01",
                "end": "2025-01-01",
                "rows": 5,
                "columns": 2,
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "preprocessing": SP500_PREPROCESSING,
                "sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExternalDataUnavailable, match="checksum"):
        fetch_sp500_data(
            tickers=symbols,
            cache_dir=tmp_path,
            offline=True,
        )


@pytest.mark.parametrize("manifest_mode", ["missing", "schema", "checksum"])
def test_gemma_cached_manifest_and_checksum_are_required(
    tmp_path: Path, manifest_mode: str
) -> None:
    dataset_id = "TEST-GEMMA"
    path = tmp_path / "gemma" / f"{dataset_id}-processed-unfiltered.tsv"
    path.parent.mkdir(parents=True)
    path.write_text("Probe\tSample-1\np1\t1.0\n", encoding="utf-8")
    if manifest_mode != "missing":
        manifest = {
            "source": "Gemma REST",
            "url": (
                "https://gemma.msl.ubc.ca/rest/v2/datasets/"
                f"{dataset_id}/data/processed?filter=false"
            ),
            "dataset_id": dataset_id,
            "processed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "retrieved_at": "2026-01-01T00:00:00+00:00",
        }
        if manifest_mode == "schema":
            manifest.pop("retrieved_at")
        else:
            manifest["processed_sha256"] = "f" * 64
        path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    message = {
        "missing": "provenance manifest",
        "schema": "missing required fields",
        "checksum": "checksum mismatch",
    }[manifest_mode]
    with pytest.raises(ExternalDataUnavailable, match=message):
        _download_gemma_processed(dataset_id, cache_dir=tmp_path, offline=True)


def test_legacy_synthetic_reproduces_published_table() -> None:
    prepared = prepare_dataset(
        "synthetic",
        protocol="legacy",
        seed=42,
        smoke=False,
        offline=True,
        data_dir=None,
    )
    from force.protocols import build_estimators

    expected = {
        "Pearson": 0.6074,
        "Spearman": 0.2075,
        "Winsorized": 0.2665,
        "FastMCD": 0.0180,
        "TP-Exact": 0.0487,
        "TP-TER": 0.0549,
        "FORCE": 0.2001,
    }
    for name, estimator in build_estimators("legacy").items():
        assert _rmse(estimator.fit(prepared.X), prepared.reference) == pytest.approx(
            expected[name], abs=5e-5
        )


@pytest.mark.parametrize(
    "protocol,method", [("paper", "linear"), ("legacy", "tail_count")]
)
def test_winsorization_convention_is_explicit_in_protocol_and_provenance(
    protocol, method
):
    from force.protocols import build_estimators
    from force.reporting import estimator_parameters

    estimator = build_estimators(protocol)["Winsorized"]
    assert estimator.quantile_method == method
    parameters = estimator_parameters(estimator)
    assert parameters["quantile_method"] == method
    assert parameters["limits"] == [0.05, 0.05]


def test_benchmark_machine_readable_schema(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "benchmark"
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository / "src"),
        "NUMBA_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_benchmark.py"),
            "--protocol",
            "paper",
            "--datasets",
            "synthetic",
            "--smoke",
            "--offline",
            "--output-dir",
            str(output),
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text())
    report_text = (output / "report.json").read_text()
    assert "NaN" not in report_text
    assert "Infinity" not in report_text
    assert report["schema_version"] == 2
    assert report["estimate_status"] == "complete"
    assert all(row["estimate_status"] == "complete" for row in report["results"])
    assert all(row["coverage"] == 1.0 for row in report["results"])
    assert all(row["timing_status"] == "warmed" for row in report["results"])
    assert report["status"] == "completed"
    assert report["protocol"]["name"] == "paper"
    assert report["datasets"][0]["status"] == "completed"
    assert report["command_line"]
    assert report["environment"]["threads"]["NUMBA_NUM_THREADS"] == "1"
    assert report["estimator_parameters"]["FORCE"]["exact_cutover"] == 5
    assert all(row["estimator_parameters"] for row in report["results"])
    assert {row["status"] for row in report["results"]} == {"completed"}
    assert len(report["summaries"]) == 7
    summary = report["summaries"][0]
    assert {
        "runs",
        "mean_time_ms",
        "std_time_ms",
        "ci95_time_ms",
        "mean_rmse",
        "std_rmse",
        "ci95_rmse",
    } <= summary.keys()
    assert summary["runs"] == 1
    assert summary["std_time_ms"] is None
    assert summary["ci95_time_ms"] is None
    assert summary["ci95_rmse"] is None
    assert summary["replicate_kind"] == "independent_order"
    assert (output / "results_raw.csv").exists()
    raw = pd.read_csv(output / "results_raw.csv")
    assert {"seed", "status", "estimator_parameters"} <= set(raw.columns)
    assert set(raw["seed"]) == {42}
    assert raw["data_sha256"].nunique() == raw["reference_sha256"].nunique() == 1
    assert raw["comparison_target"].eq("known_clean_population_Pearson").all()
    assert (output / "summary.md").exists()


def test_actual_fixed_dataset_runner_reports_repeatability_not_rmse_uncertainty():
    from scripts.run_benchmark import run_dataset, _summary_records

    X = np.random.default_rng(24).normal(size=(80, 2))
    rows = run_dataset(
        dataset_name="fixture",
        X=X,
        reference=np.eye(2),
        protocol="paper",
        runs=3,
        seed=24,
    )
    frame = pd.DataFrame(rows)
    assert frame["status"].eq("completed").all()
    assert frame["replicate_kind"].eq("fixed_dataset_timing").all()
    assert frame["data_sha256"].nunique() == frame["reference_sha256"].nunique() == 1
    assert frame["comparison_target"].eq("supplied_reference_matrix").all()
    force = frame[frame["algorithm"] == "FORCE"]
    assert force["rmse"].nunique() == 1
    summaries = _summary_records(frame)
    assert len(summaries) == 7
    for summary in summaries:
        assert summary["runs"] == 3
        assert summary["ci95_rmse"] is None
        assert summary["rmse_interpretation"] == "conditional_fixed_dataset_error"
        assert summary["rmse_repeatability_std"] == summary["std_rmse"]
        assert np.isfinite(summary["ci95_time_ms"])


def test_paper_data_blockers_set_aggregate_status(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "blockers"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_benchmark.py"),
            "--protocol",
            "paper",
            "--datasets",
            "sp500,genomics",
            "--offline",
            "--output-dir",
            str(output),
        ],
        cwd=repository,
        env={
            **os.environ,
            "PYTHONPATH": str(repository / "src"),
            "NUMBA_NUM_THREADS": "1",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "not_reproducible"
    assert {record["status"] for record in report["datasets"]} == {"not_reproducible"}
    assert not report["results"]


@pytest.mark.parametrize(
    "script",
    [
        "run_benchmark.py",
        "run_convergence_analysis.py",
        "run_p2_skewness_diagnostic.py",
        "run_ter_contamination_analysis.py",
        "run_sp500_volatility_sensitivity.py",
        "run_trimmed_pearson_baseline.py",
        "run_scalability_check.py",
    ],
)
def test_runners_expose_common_protocol_interface(script: str) -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(repository / "scripts" / script), "--help"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    for option in (
        "--protocol",
        "--runs",
        "--seed",
        "--smoke",
        "--offline",
        "--data-dir",
        "--output-dir",
    ):
        assert option in completed.stdout


def test_paper_ter_analysis_is_uncapped(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "ter"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "run_ter_contamination_analysis.py"),
            "--protocol",
            "paper",
            "--smoke",
            "--offline",
            "--output-dir",
            str(output),
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "PYTHONPATH": str(repository / "src"),
            "NUMBA_NUM_THREADS": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["protocol"]["name"] == "paper"
    assert report["estimator_parameters"]["ter"]["ter_max"] is None
    assert report["estimator_parameters"]["fixed"]["use_ter"] is False
