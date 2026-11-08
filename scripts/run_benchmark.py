"""Run FORCE benchmarks under an equation-faithful or legacy protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict, Iterable, List, Sequence

for _thread_variable in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd
from scipy import stats

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from force.data import ExternalDataUnavailable
from force._evaluation import correlation_score, estimator_pair_validity
from force.protocols import (
    DATASET_NAMES,
    DatasetNotReproducible,
    build_estimator,
    build_estimators,
    get_protocol,
    prepare_dataset,
)
from force.reporting import (
    command_line,
    environment_metadata,
    estimator_parameters,
    json_safe,
    write_json,
)


LOGGER = logging.getLogger("force.benchmark")


def off_diagonal_rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Raw matrix discrepancy; benchmark scoring additionally requires validity."""
    if estimate.shape != reference.shape or estimate.ndim != 2:
        raise ValueError("Estimate and reference must be equal-sized matrices.")
    mask = np.triu(np.ones_like(estimate, dtype=bool), k=1)
    return float(np.sqrt(np.mean((estimate[mask] - reference[mask]) ** 2)))


def _parse_datasets(value: str) -> List[str]:
    datasets = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(datasets) - set(DATASET_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown datasets: {', '.join(unknown)}")
    if not datasets:
        raise argparse.ArgumentTypeError("At least one dataset is required.")
    return datasets


def _read_tickers(path: Path) -> Sequence[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("tickers")
        if not isinstance(payload, list):
            raise ValueError("Ticker JSON must be a list or contain 'tickers'.")
        values = payload
    else:
        values = text.replace(",", "\n").splitlines()
    tickers = tuple(
        str(value).strip().upper() for value in values if str(value).strip()
    )
    return tickers


def _array_identity(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256(str(canonical.shape).encode("ascii"))
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def _replicate_metadata(replicate_mode: str) -> Dict[str, str]:
    if replicate_mode == "fixed":
        return {
            "replicate_mode": "fixed",
            "replicate_kind": "fixed_dataset_timing",
            "rmse_interpretation": "conditional_fixed_dataset_error",
        }
    if replicate_mode == "order":
        return {
            "replicate_mode": "order",
            "replicate_kind": "independent_order",
            "rmse_interpretation": "conditional_mean_error_over_random_orders",
        }
    raise ValueError("replicate_mode must be 'fixed' or 'order'.")


def _estimate_score(estimator, estimate, reference, *, reference_valid=None):
    """Score only coefficients certified by the fitted estimator's pair codes."""
    valid, basis = estimator_pair_validity(estimator, estimate)
    score = correlation_score(
        estimate,
        reference,
        estimate_valid=valid,
        reference_valid=reference_valid,
        require_complete=True,
    )
    if basis == "validity_unavailable":
        score.update(
            status="validity_unavailable",
            reason="Estimator does not expose pair validity; finite output is insufficient.",
        )
    return {
        "estimate_status": score["status"],
        "metric_reason": score["reason"],
        "validity_basis": basis,
        **{
            key: value
            for key, value in score.items()
            if key not in {"status", "reason"}
        },
    }


def run_dataset(
    *,
    dataset_name: str,
    X: np.ndarray,
    reference: np.ndarray,
    protocol: str,
    runs: int,
    seed: int,
    comparison_target: str = "supplied_reference_matrix",
    replicate_mode: str = "fixed",
    reference_valid: np.ndarray | None = None,
    reference_validity_basis: str | None = None,
) -> List[Dict[str, Any]]:
    """Measure fresh fits with a common input and reference within each run.

    Order replicates independently permute one fixed dataset; they are not
    independent datasets or cross-validation folds. Hashing, permutation,
    estimator construction, and the throwaway warm-ups are outside timing.
    A missing reference mask explicitly declares finite supplied coefficients
    valid. Estimated references must retain their own pair-validity mask.
    """
    if runs < 1:
        raise ValueError("runs must be at least one.")
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    reference_score = correlation_score(
        reference,
        reference,
        estimate_valid=(
            np.isfinite(reference) if reference_valid is None else reference_valid
        ),
        reference_valid=reference_valid,
        require_complete=True,
    )
    if reference_validity_basis is None:
        reference_validity_basis = (
            "finite_reference_declaration"
            if reference_valid is None
            else "supplied_reference_mask"
        )
    rows: List[Dict[str, Any]] = []
    identity = {
        **_replicate_metadata(replicate_mode),
        "protocol": protocol,
        "dataset": dataset_name,
        "mode": "batch",
        "comparison_target": comparison_target,
        "base_data_sha256": _array_identity(X),
        "reference_sha256": _array_identity(reference),
        "reference_validity_basis": reference_validity_basis,
        "eligible_mask_sha256": reference_score["eligible_mask_sha256"],
        "seed": seed,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "requested_runs": runs,
    }
    algorithms = tuple(build_estimators(protocol))
    warmups = {}
    warmed = {}
    parameters = {}
    for algorithm in algorithms:
        warmup_estimator = build_estimator(protocol, algorithm)
        parameters[algorithm] = json.dumps(
            estimator_parameters(warmup_estimator), sort_keys=True
        )
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                warmup_estimator.fit(X)
            warmups[algorithm] = {"status": "completed", "message": ""}
            warmed[algorithm] = True
        except Exception as exc:
            warmups[algorithm] = {"status": "failed", "message": str(exc)}
            warmed[algorithm] = False

    for run_id in range(1, runs + 1):
        order = np.arange(len(X), dtype=np.int64)
        order_seed = None
        if replicate_mode == "order":
            # Distinct SeedSequence children retain reproducibility without
            # depending on which estimators succeeded in earlier runs.
            order_seed = int(
                np.random.SeedSequence([seed, run_id]).generate_state(1)[0]
            )
            order = np.random.default_rng(order_seed).permutation(order)
            run_X = X[order]
        else:
            run_X = X
        order_hash = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
        run_identity = {
            **identity,
            "run_id": run_id,
            "replicate_id": f"{replicate_mode}:{seed}:{run_id}",
            "order_seed": order_seed,
            "order_sha256": order_hash,
            "data_sha256": _array_identity(run_X),
        }
        # Run-major cyclic rotation spreads method-position effects over runs.
        # Pairing is on replicate identity, never on this execution position.
        shift = (run_id - 1) % len(algorithms)
        method_order = algorithms[shift:] + algorithms[:shift]
        for position, algorithm in enumerate(method_order):
            row = {
                **run_identity,
                "algorithm": algorithm,
                "execution_position": position,
                "estimator_parameters": parameters[algorithm],
                "time_ms": None,
                "rmse": None,
                "status": "failed",
                "attempted": True,
                "estimate_status": "fit_failed",
                "metric_reason": "Fit did not complete.",
                "validity_basis": "unavailable",
                "timing_status": "fit_failed",
                "warmup_status": warmups[algorithm]["status"],
                "warmup_message": warmups[algorithm]["message"],
                "message": "",
                "warning_count": 0,
            }
            estimator = build_estimator(protocol, algorithm)
            was_warmed = warmed[algorithm]
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    start = perf_counter_ns()
                    estimate = estimator.fit(run_X)
                    elapsed_ms = (perf_counter_ns() - start) / 1_000_000.0
                warmed[algorithm] = True
                if not np.isfinite(elapsed_ms) or elapsed_ms <= 0:
                    raise FloatingPointError("Benchmark measurement was not finite.")
                row.update(
                    time_ms=float(elapsed_ms) if was_warmed else None,
                    timing_status="warmed" if was_warmed else "not_warmed",
                    status="completed",
                    message=str(captured[0].message) if captured else "",
                    warning_count=len(captured),
                )
                # Validity/accuracy evaluation is outside the measured fit.
                # A successful un-warmed fit warms later runs without a retry.
                try:
                    row.update(
                        _estimate_score(
                            estimator,
                            estimate,
                            reference,
                            reference_valid=reference_valid,
                        )
                    )
                    row["bound_diagnostics"] = json.dumps(
                        json_safe(getattr(estimator, "bound_status_", None)),
                        sort_keys=True,
                    )
                except Exception as exc:
                    row.update(
                        estimate_status="invalid_output",
                        metric_reason=str(exc),
                    )
            except Exception as exc:
                row["message"] = str(exc)
            rows.append(row)
    return rows


def _method_parameter_identity(group: pd.DataFrame) -> str | None:
    """Require one declared parameter identity within a method, when supplied."""
    if "estimator_parameters" not in group:
        return None
    identities = set()
    for value in group["estimator_parameters"]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Missing estimator_parameters within an algorithm.")
        try:
            parameters = json.loads(value)
            if not isinstance(parameters, dict):
                raise ValueError("Parameters must be a JSON object.")
            canonical = json.dumps(parameters, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "estimator_parameters must be a finite serialized JSON object."
            ) from exc
        identities.add(canonical)
    if len(identities) != 1:
        raise ValueError("Mixed estimator_parameters within an algorithm.")
    return identities.pop()


def _summary_records(
    frame: pd.DataFrame, *, requested_runs: int | None = None
) -> List[Dict[str, Any]]:
    """Separate complete-order means from explicitly selected successful fits."""
    if frame.empty:
        return []
    records: List[Dict[str, Any]] = []
    groups = frame.groupby(
        ["protocol", "dataset", "algorithm"], observed=True, sort=True
    )
    for (protocol, dataset, algorithm), group in groups:
        parameter_identity = _method_parameter_identity(group)
        for name in ("replicate_mode", "replicate_kind", "rmse_interpretation"):
            if group[name].nunique(dropna=False) != 1:
                raise ValueError(f"Cannot summarize mixed {name} values.")
        if (
            group["replicate_id"].duplicated().any()
            or group["run_id"].duplicated().any()
        ):
            raise ValueError("Duplicate replicate IDs would inflate the sample size.")
        mode = str(group["replicate_mode"].iloc[0])
        ids = pd.to_numeric(group["run_id"], errors="raise")
        if (
            not np.isfinite(ids).all()
            or (ids < 1).any()
            or (ids != np.floor(ids)).any()
        ):
            raise ValueError("Summary run IDs must be positive integers.")
        requested = requested_runs
        if "requested_runs" in group:
            if (
                group["requested_runs"].isna().any()
                or group["requested_runs"].nunique() != 1
            ):
                raise ValueError("Inconsistent requested run counts.")
            recorded = float(group["requested_runs"].iloc[0])
            if not np.isfinite(recorded) or recorded < 1 or recorded != int(recorded):
                raise ValueError("Requested runs must be a positive integer.")
            if requested is not None and requested != int(recorded):
                raise ValueError("Requested run count disagrees with row metadata.")
            requested = int(recorded)
        if requested is None:
            requested = int(ids.max())
        if requested < 1 or requested != int(requested) or ids.max() > requested:
            raise ValueError("Observed run ID exceeds requested runs.")
        requested = int(requested)
        completed = group["status"].eq("completed")
        attempted = group.get("attempted", pd.Series(True, index=group.index)).eq(True)
        if (completed & ~attempted).any():
            raise ValueError("An unattempted fit cannot be completed.")
        if "estimate_status" not in group:
            raise ValueError("Summary requires explicit estimate validity status.")
        score_valid = completed & group["estimate_status"].eq("complete")
        errors = pd.to_numeric(group["rmse"], errors="coerce")
        if (score_valid & (~np.isfinite(errors) | (errors < 0))).any():
            raise ValueError("Complete estimates require finite nonnegative RMSE.")
        score_values = errors[score_valid].to_numpy(dtype=float)
        timing_status = group.get(
            "timing_status", pd.Series("warmed", index=group.index)
        )
        times = pd.to_numeric(group["time_ms"], errors="coerce")
        time_valid = (
            completed & timing_status.eq("warmed") & np.isfinite(times) & (times > 0)
        )
        time_values = times[time_valid].to_numpy(dtype=float)
        n_scores, n_times = len(score_values), len(time_values)
        all_scores = n_scores == requested
        all_times = n_times == requested
        mean_error = float(np.mean(score_values)) if n_scores else None
        std_error = float(np.std(score_values, ddof=1)) if n_scores > 1 else None
        error_ci = (
            float(1.96 * std_error / np.sqrt(n_scores))
            if mode == "order" and std_error is not None
            else None
        )
        mean_time = float(np.mean(time_values)) if n_times else None
        std_time = float(np.std(time_values, ddof=1)) if n_times > 1 else None
        time_ci = (
            float(1.96 * std_time / np.sqrt(n_times)) if std_time is not None else None
        )
        record = {
            "protocol": str(protocol),
            "dataset": str(dataset),
            "algorithm": str(algorithm),
            "runs": int(completed.sum()),
            "requested_runs": requested,
            "recorded_runs": len(group),
            "attempted_runs": int(attempted.sum()),
            "completed_runs": int(completed.sum()),
            "successful_runs": n_scores,
            "failed_runs": int((attempted & ~completed).sum()),
            "incomplete_runs": int((completed & ~score_valid).sum()),
            "missing_runs": requested - len(group),
            "not_attempted_runs": int((~attempted).sum()),
            "timing_eligible_runs": n_times,
            "mean_time_ms": mean_time if all_times else None,
            "std_time_ms": std_time if all_times else None,
            "ci95_time_ms": time_ci if all_times else None,
            "timing_eligible_mean_time_ms": mean_time,
            "timing_eligible_std_time_ms": std_time,
            "timing_eligible_ci95_time_ms": time_ci,
            "timing_summary_conditioning": "successful_warmed_fits_only",
            "mean_rmse": mean_error if all_scores else None,
            "std_rmse": std_error if all_scores else None,
            "rmse_repeatability_std": (
                std_error if mode == "fixed" and all_scores else None
            ),
            "ci95_rmse": error_ci if all_scores else None,
            "rmse_ci_basis": (
                "normal95_conditional_order_mean"
                if all_scores and mode == "order" and n_scores > 1
                else "not_estimated"
            ),
            "rmse_summary_status": (
                "complete"
                if all_scores
                else "incomplete" if n_scores else "unavailable"
            ),
            "success_conditioned_mean_rmse": mean_error,
            "success_conditioned_std_rmse": std_error,
            "success_conditioned_ci95_rmse": error_ci,
            "success_conditioned_rmse_interpretation": (
                "conditional_order_mean_given_complete_estimate"
                if mode == "order"
                else "conditional_fixed_dataset_error_given_complete_estimate"
            ),
            "success_conditioned_rmse_ci_basis": (
                "normal95_order_mean_given_complete_estimate"
                if mode == "order" and n_scores > 1
                else "not_estimated"
            ),
            **_replicate_metadata(mode),
        }
        if parameter_identity is not None:
            record["estimator_parameters"] = parameter_identity
        if "coverage" in group:
            coverage = pd.to_numeric(group["coverage"], errors="coerce")
            finite = coverage[np.isfinite(coverage)]
            record["minimum_score_coverage"] = (
                float(finite.min()) if len(finite) else None
            )
            record["mean_score_coverage"] = (
                float(finite.mean()) if len(finite) else None
            )
        records.append(record)
    return records


def paired_timing_records(
    frame: pd.DataFrame, *, requested_runs: int | None = None
) -> List[Dict[str, Any]]:
    """Two-sided paired t-tests on verified, successful matched run identities.

    Reported p-values are unadjusted, as in paper Section 3.5. Zero-variance
    differences have undefined inference even when their mean is nonzero.
    """
    if frame.empty:
        return []
    stable_identity = (
        "seed",
        "mode",
        "comparison_target",
        "n_samples",
        "n_features",
        "base_data_sha256",
        "reference_sha256",
        "replicate_mode",
        "replicate_kind",
        "rmse_interpretation",
    )
    stable_identity += tuple(
        name
        for name in ("eligible_mask_sha256", "reference_validity_basis")
        if name in frame
    )
    matched_identity = ("replicate_id", "order_seed", "order_sha256", "data_sha256")
    required = {
        "protocol",
        "dataset",
        "algorithm",
        "run_id",
        "status",
        "time_ms",
        *stable_identity,
        *matched_identity,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing paired timing identity fields: {sorted(missing)}")
    records = []
    for (protocol, dataset), group in frame.groupby(
        ["protocol", "dataset"], observed=True, sort=True
    ):
        parameter_identities = {
            algorithm: _method_parameter_identity(method)
            for algorithm, method in group.groupby("algorithm")
        }
        for name in stable_identity:
            if group[name].isna().any() or group[name].nunique(dropna=False) != 1:
                raise ValueError(f"Mismatched paired timing {name}.")
        if group[list(set(matched_identity) - {"order_seed"})].isna().any().any():
            raise ValueError("Paired timing identities must not be missing.")
        mode = str(group["replicate_mode"].iloc[0])
        for name, value in _replicate_metadata(mode).items():
            if not group[name].eq(value).all():
                raise ValueError(f"Inconsistent paired timing {name}.")
        if mode == "order" and group["order_seed"].isna().any():
            raise ValueError("Order replicates require order seeds.")
        run_ids = pd.to_numeric(group["run_id"], errors="raise")
        if (
            not np.isfinite(run_ids).all()
            or (run_ids < 1).any()
            or not (run_ids == np.floor(run_ids)).all()
        ):
            raise ValueError("Paired timing run IDs must be positive integers.")
        group = group.copy()
        group["run_id"] = run_ids.astype(np.int64)
        if group.duplicated(["algorithm", "run_id"]).any():
            raise ValueError("Duplicate algorithm/run IDs in paired timing data.")
        if group.duplicated(["algorithm", "replicate_id"]).any():
            raise ValueError("Duplicate replicate IDs in paired timing data.")
        expected = (
            set(range(1, requested_runs + 1))
            if requested_runs is not None
            else set(run_ids.astype(int))
        )
        if not set(run_ids.astype(int)) <= expected:
            raise ValueError("Observed run ID exceeds requested runs.")
        for _, replicate in group.groupby("run_id"):
            for name in matched_identity:
                if replicate[name].nunique(dropna=False) != 1:
                    raise ValueError(f"Mismatched paired timing {name}.")
        indexed = {
            name: method.set_index("run_id")
            for name, method in group.groupby("algorithm")
        }
        force = indexed.get("FORCE", group.iloc[:0].set_index("run_id"))
        for algorithm in sorted(set(indexed) - {"FORCE"}):
            baseline = indexed[algorithm]
            record = {
                "protocol": str(protocol),
                "dataset": str(dataset),
                "algorithm": str(algorithm),
                "comparison": "FORCE_minus_baseline",
                "test": "paired_t_two_sided",
                "pvalue_adjustment": "none",
                "conditioning": "successful_warmed_fits_on_matched_orders",
                "replicate_kind": str(group["replicate_kind"].iloc[0]),
                "requested_pairs": len(expected),
            }
            if "estimator_parameters" in group:
                record["force_estimator_parameters"] = parameter_identities.get("FORCE")
                record["baseline_estimator_parameters"] = parameter_identities[
                    algorithm
                ]
            valid_runs = []
            for label, method in (("force", force), ("baseline", baseline)):
                present = expected & set(method.index)
                valid = set()
                failed = nonfinite = unwarmed = 0
                for run in present:
                    row = method.loc[run]
                    if row["status"] != "completed":
                        failed += 1
                    elif row.get("timing_status", "warmed") != "warmed":
                        unwarmed += 1
                    elif (
                        pd.isna(row["time_ms"])
                        or not np.isfinite(float(row["time_ms"]))
                        or float(row["time_ms"]) <= 0
                    ):
                        nonfinite += 1
                    else:
                        valid.add(run)
                record.update(
                    {
                        f"missing_{label}_runs": len(expected - present),
                        f"failed_{label}_runs": failed,
                        f"invalid_{label}_times": nonfinite,
                        f"unwarmed_{label}_runs": unwarmed,
                    }
                )
                valid_runs.append(valid)
            paired_runs = sorted(valid_runs[0] & valid_runs[1])
            differences = np.array(
                [
                    float(force.loc[run, "time_ms"])
                    - float(baseline.loc[run, "time_ms"])
                    for run in paired_runs
                ]
            )
            count = len(paired_runs)
            scale = float(np.max(np.abs(differences))) if count else 0.0
            scaled = differences / scale if scale else differences
            record.update(
                paired_runs=paired_runs,
                usable_pairs=count,
                excluded_pairs=len(expected) - count,
                mean_difference_ms=(float(np.mean(scaled) * scale) if count else None),
                degrees_of_freedom=count - 1 if count > 1 else None,
                t_statistic=None,
                pvalue=None,
                status="insufficient_pairs",
            )
            if count > 1:
                if np.all(differences == differences[0]):
                    record["status"] = "zero_variance_differences"
                else:
                    statistic, pvalue = stats.ttest_1samp(scaled, 0.0)
                    if np.isfinite(statistic) and np.isfinite(pvalue):
                        record.update(
                            t_statistic=float(statistic),
                            pvalue=float(pvalue),
                            status="ok",
                        )
                    else:
                        record["status"] = "nonfinite_inference"
            records.append(record)
    return records


def _summary_markdown(summary_records: List[Dict[str, Any]]) -> str:
    if not summary_records:
        return "# FORCE benchmark summary\n\nNo requested-fit records.\n"
    summary = pd.DataFrame.from_records(summary_records)
    columns = [
        "protocol",
        "dataset",
        "algorithm",
        "requested_runs",
        "successful_runs",
        "failed_runs",
        "incomplete_runs",
        "mean_rmse",
        "ci95_rmse",
        "success_conditioned_mean_rmse",
        "mean_time_ms",
        "timing_eligible_runs",
    ]
    return (
        "# FORCE benchmark summary\n\n"
        "All errors are conditional on one fixed dataset and reference. Order-mode "
        "RMSE intervals describe the mean over independently randomized observation "
        "orders, not uncertainty across datasets or cross-validation folds. Fixed-mode "
        "repetitions measure repeatability and have no RMSE confidence interval. "
        "Primary means are unavailable when requested fits lack an eligible score "
        "or warmed timing. Success-conditioned accuracy and timing-eligible summaries "
        "are separate fields in the report, alongside requested, failed and incomplete "
        "counts. Intervals use the paper's normal 1.96 standard-error "
        "approximation.\n\n" + summary[columns].to_markdown(index=False) + "\n"
    )


def _paired_markdown(records: List[Dict[str, Any]]) -> str:
    intro = (
        "# Paired FORCE timing comparisons\n\n"
        "Two-sided paired t-tests condition on successful warmed measurements with matching "
        "run, order, data, reference and target identities. P-values are unadjusted. "
        "Missing, failed and unwarmed pairs are excluded and counted; insufficient or constant "
        "differences leave inference undefined. Timing inference assumes independent "
        "run differences; operating-system timing effects can violate this assumption.\n\n"
    )
    return intro + (
        pd.DataFrame(records).to_markdown(index=False) + "\n"
        if records
        else "No paired comparisons available.\n"
    )


def execute(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else None
    runs = 1 if args.smoke else args.runs
    tickers = (
        _read_tickers(Path(args.sp500_tickers_file))
        if args.sp500_tickers_file
        else None
    )
    protocol = get_protocol(args.protocol)
    replicate_mode = getattr(args, "replicate_mode", None) or (
        "order" if args.protocol == "paper" else "fixed"
    )
    result_rows: List[Dict[str, Any]] = []
    dataset_records: List[Dict[str, Any]] = []

    for dataset_name in args.datasets:
        LOGGER.info("Preparing %s under %s protocol", dataset_name, args.protocol)
        try:
            prepared = prepare_dataset(
                dataset_name,
                protocol=args.protocol,
                seed=args.seed,
                smoke=args.smoke,
                offline=args.offline,
                data_dir=data_dir,
                sp500_tickers=tickers,
                gemma_dataset=args.gemma_dataset,
            )
            dataset_records.append(
                {
                    "dataset": dataset_name,
                    "status": "completed",
                    "shape": list(prepared.X.shape),
                    "provenance": prepared.provenance,
                    "reference_validity_basis": prepared.reference_validity_basis,
                    "message": "",
                }
            )
            result_rows.extend(
                run_dataset(
                    dataset_name=dataset_name,
                    X=prepared.X,
                    reference=prepared.reference,
                    reference_valid=prepared.reference_valid,
                    reference_validity_basis=prepared.reference_validity_basis,
                    protocol=args.protocol,
                    runs=runs,
                    seed=args.seed,
                    replicate_mode=replicate_mode,
                    comparison_target=(
                        "known_clean_population_Pearson"
                        if dataset_name == "synthetic"
                        else str(
                            prepared.provenance.get(
                                "reference", "supplied_reference_matrix"
                            )
                        )
                    ),
                )
            )
        except DatasetNotReproducible as exc:
            dataset_records.append(
                {
                    "dataset": dataset_name,
                    "status": "not_reproducible",
                    "shape": None,
                    "provenance": {},
                    "message": str(exc),
                }
            )
        except ExternalDataUnavailable as exc:
            dataset_records.append(
                {
                    "dataset": dataset_name,
                    "status": "skipped_external",
                    "shape": None,
                    "provenance": {},
                    "message": str(exc),
                }
            )
        except Exception as exc:
            dataset_records.append(
                {
                    "dataset": dataset_name,
                    "status": "failed",
                    "shape": None,
                    "provenance": {},
                    "message": str(exc),
                }
            )

    columns = [
        "protocol",
        "dataset",
        "algorithm",
        "run_id",
        "replicate_id",
        "seed",
        "order_seed",
        "order_sha256",
        "execution_position",
        "n_samples",
        "n_features",
        "time_ms",
        "rmse",
        "status",
        "message",
        "warning_count",
        "estimator_parameters",
        "mode",
        "comparison_target",
        "data_sha256",
        "base_data_sha256",
        "reference_sha256",
        "replicate_mode",
        "replicate_kind",
        "rmse_interpretation",
    ]
    extra_columns = sorted({key for row in result_rows for key in row} - set(columns))
    frame = pd.DataFrame(result_rows, columns=columns + extra_columns)
    summaries = _summary_records(frame, requested_runs=runs)
    timing_comparisons = paired_timing_records(frame, requested_runs=runs)
    frame.to_csv(output_dir / "results_raw.csv", index=False)
    (output_dir / "summary.md").write_text(
        _summary_markdown(summaries), encoding="utf-8"
    )
    (output_dir / "paired_timing.md").write_text(
        _paired_markdown(timing_comparisons), encoding="utf-8"
    )
    dataset_statuses = {record["status"] for record in dataset_records}
    result_statuses = {row["status"] for row in result_rows}
    if "failed" in dataset_statuses or "failed" in result_statuses:
        aggregate_status = "failed"
    elif "completed" in dataset_statuses:
        aggregate_status = "completed"
    elif "not_reproducible" in dataset_statuses:
        aggregate_status = "not_reproducible"
    elif "skipped_external" in dataset_statuses:
        aggregate_status = "skipped_external"
    else:
        aggregate_status = "completed"
    report = {
        "schema_version": 2,
        "status": aggregate_status,
        "estimate_status": (
            "complete"
            if result_rows
            and all(row["estimate_status"] == "complete" for row in result_rows)
            else (
                "partial"
                if any(row["estimate_status"] == "complete" for row in result_rows)
                else "unavailable"
            )
        ),
        "protocol": protocol.as_dict(),
        "smoke": bool(args.smoke),
        "offline": bool(args.offline),
        "seed": int(args.seed),
        "requested_runs": int(runs),
        **_replicate_metadata(replicate_mode),
        "execution_schedule": "run_major_cyclic_method_rotation",
        "replicate_design": (
            "Independent seeded permutations of one fixed dataset; all methods "
            "share each permutation and the unchanged reference. RMSE inference "
            "concerns the conditional mean over orders, not population sampling. "
            "If fits fail or lack complete pair validity, primary order means are "
            "unavailable; success-conditioned summaries are identified separately."
            if replicate_mode == "order"
            else "Identical data/order/reference across runs; RMSE reports conditional "
            "error and repeatability, with no RMSE confidence interval."
        ),
        "environment": environment_metadata(),
        "command_line": command_line(),
        "data_dir": str(data_dir) if data_dir is not None else None,
        "estimator_parameters": {
            name: estimator_parameters(estimator)
            for name, estimator in build_estimators(args.protocol).items()
        },
        "datasets": dataset_records,
        "summaries": summaries,
        "paired_timing": timing_comparisons,
        "results": result_rows,
    }
    write_json(output_dir / "report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("paper", "legacy"), default="paper")
    parser.add_argument(
        "--datasets",
        type=_parse_datasets,
        default=list(DATASET_NAMES),
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    replicate_options = parser.add_mutually_exclusive_group()
    replicate_options.add_argument(
        "--replicate-mode",
        choices=("fixed", "order"),
        default=None,
        help="Observation order per run; default: order for paper, fixed for legacy.",
    )
    replicate_options.add_argument(
        "--fixed",
        dest="replicate_mode",
        action="store_const",
        const="fixed",
        help="Keep the input order fixed to measure timing repeatability.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="./benchmark_results")
    parser.add_argument("--sp500-tickers-file", default=None)
    parser.add_argument("--gemma-dataset", default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    report = execute(args)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
