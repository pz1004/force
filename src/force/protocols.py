"""Explicit paper-faithful and legacy benchmark protocols."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .core import ForceEstimator
from .data import (
    ExternalDataUnavailable,
    fetch_genomics_data,
    fetch_sp500_data,
    generate_odds_surrogate,
    generate_synthetic_data,
    load_odds_dataset,
)
from .estimators import (
    CorrelationEstimator,
    FastMCDEstimator,
    PearsonEstimator,
    SpearmanEstimator,
    WinsorizedEstimator,
)
from .legacy import LegacyForceEstimator
from .trimmed_pearson import TrimmedPearsonExact


DATASET_NAMES = (
    "synthetic",
    "sp500",
    "odds-mammography",
    "odds-satellite",
    "genomics",
)


class DatasetNotReproducible(RuntimeError):
    """Raised when the paper omits identifiers needed to construct a dataset."""


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    X: NDArray[np.float64]
    reference: NDArray[np.float64]
    provenance: Dict[str, Any]


@dataclass(frozen=True)
class Protocol:
    name: str
    description: str
    force_parameters: Mapping[str, Any]
    synthetic_parameters: Mapping[str, Any]
    runs: int = 20

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "force_parameters": dict(self.force_parameters),
            "synthetic_parameters": dict(self.synthetic_parameters),
            "runs": self.runs,
        }


PAPER_PROTOCOL = Protocol(
    name="paper",
    description=(
        "Equation-faithful FORCE and prose-specified datasets; omitted dataset "
        "identifiers are reported rather than invented."
    ),
    force_parameters={
        "lambda_scale": 3.0,
        "exact_cutover": 5,
        "use_ter": True,
        "ter_max": None,
        "epsilon": 1e-10,
    },
    synthetic_parameters={
        "n_samples": 1000,
        "n_features": 10,
        "contamination": 0.10,
        "contamination_distribution": "cauchy",
        "outlier_scale": 10.0,
    },
)

LEGACY_PROTOCOL = Protocol(
    name="legacy",
    description=(
        "Committed result-generating settings retained only for published-table "
        "reproduction."
    ),
    force_parameters={
        "lambda_scale": 3.0,
        "exact_cutover": 100,
        "quantiles": [0.05, 0.25, 0.50, 0.75, 0.95],
    },
    synthetic_parameters={
        "n_samples": 1000,
        "n_features": 50,
        "contamination": 0.10,
        "contamination_distribution": "uniform",
        "outlier_scale": 10.0,
    },
)


def get_protocol(name: str) -> Protocol:
    if name == "paper":
        return PAPER_PROTOCOL
    if name == "legacy":
        return LEGACY_PROTOCOL
    raise ValueError("protocol must be 'paper' or 'legacy'.")


def build_estimators(protocol: str) -> Dict[str, CorrelationEstimator]:
    """Build fresh estimator instances in the paper's displayed order."""
    if protocol == "paper":
        force: CorrelationEstimator = ForceEstimator(**PAPER_PROTOCOL.force_parameters)
        ter_max = None
    elif protocol == "legacy":
        force = LegacyForceEstimator()
        ter_max = 3.0
    else:
        raise ValueError("protocol must be 'paper' or 'legacy'.")
    return {
        "Pearson": PearsonEstimator(),
        "Spearman": SpearmanEstimator(),
        "Winsorized": WinsorizedEstimator(),
        "FastMCD": FastMCDEstimator(random_state=42),
        "TP-Exact": TrimmedPearsonExact(use_ter=False),
        "TP-TER": TrimmedPearsonExact(use_ter=True, ter_max=ter_max),
        "FORCE": force,
    }


def build_estimator(protocol: str, algorithm: str) -> CorrelationEstimator:
    """Build one fresh estimator by its stable benchmark label."""
    estimators = build_estimators(protocol)
    if algorithm not in estimators:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return estimators[algorithm]


def _spearman_reference(X: NDArray[np.float64]) -> NDArray[np.float64]:
    ranks = np.column_stack(
        [stats.rankdata(X[:, column]) for column in range(X.shape[1])]
    )
    return np.asarray(np.corrcoef(ranks, rowvar=False), dtype=np.float64)


def _smoke_fixture(
    dataset: str, seed: int
) -> PreparedDataset:
    if dataset.startswith("odds-"):
        odds_name = dataset.removeprefix("odds-")
        X, reference = generate_odds_surrogate(
            odds_name, n_max_samples=160, seed=seed
        )
        if X.shape[1] > 8:
            X = X[:, :8]
            reference = reference[:8, :8]
    elif dataset == "genomics":
        X, _ = generate_synthetic_data(
            n_samples=160,
            n_features=8,
            contamination=0.0,
            seed=seed,
        )
        reference = _spearman_reference(X)
    elif dataset == "sp500":
        rng = np.random.default_rng(seed)
        market = rng.normal(size=(160, 1))
        X = 0.6 * market + rng.normal(scale=0.8, size=(160, 8))
        volatility = np.mean(np.abs(X), axis=1)
        reference = np.corrcoef(
            X[volatility < np.quantile(volatility, 0.9)], rowvar=False
        )
    else:
        raise ValueError(f"No smoke fixture for {dataset}.")
    return PreparedDataset(
        name=dataset,
        X=np.asarray(X, dtype=np.float64),
        reference=np.asarray(reference, dtype=np.float64),
        provenance={
            "source": "deterministic_smoke_fixture",
            "seed": seed,
            "not_for_paper_metrics": True,
        },
    )


def prepare_dataset(
    dataset: str,
    *,
    protocol: str,
    seed: int,
    smoke: bool,
    offline: bool,
    data_dir: Optional[Path],
    sp500_tickers: Optional[Sequence[str]] = None,
    gemma_dataset: Optional[str] = None,
) -> PreparedDataset:
    """Construct one benchmark dataset or raise an explicit blocker."""
    if dataset not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {dataset}")
    config = get_protocol(protocol)

    if dataset == "synthetic":
        parameters = dict(config.synthetic_parameters)
        if smoke:
            parameters["n_samples"] = 200
            parameters["n_features"] = min(8, int(parameters["n_features"]))
        X, reference = generate_synthetic_data(seed=seed, **parameters)
        return PreparedDataset(
            name=dataset,
            X=X,
            reference=reference,
            provenance={
                "source": "generated",
                "seed": seed,
                **parameters,
            },
        )

    if smoke:
        return _smoke_fixture(dataset, seed)

    if protocol == "paper":
        if dataset == "sp500":
            if sp500_tickers is None:
                raise DatasetNotReproducible(
                    "The paper specifies 50 S&P 500 constituents but provides no "
                    "ticker manifest."
                )
            if len(sp500_tickers) != 50:
                raise DatasetNotReproducible(
                    "The paper protocol requires exactly 50 supplied tickers."
                )
            X, reference, metadata = fetch_sp500_data(
                tickers=sp500_tickers,
                start="2015-01-01",
                end="2025-01-01",
                cache_dir=data_dir,
                offline=offline,
                return_metadata=True,
            )
            return PreparedDataset(
                dataset,
                X,
                reference,
                metadata,
            )
        if dataset == "genomics":
            if gemma_dataset is None:
                raise DatasetNotReproducible(
                    "The paper specifies Gemma 5000x100 data but gives no "
                    "dataset accession or sampling rule."
                )
            X, reference, metadata = fetch_genomics_data(
                n_samples_target=5000,
                n_features_target=100,
                dataset_id=gemma_dataset,
                seed=seed,
                cache_dir=data_dir,
                offline=offline,
                reference="fastmcd",
                return_metadata=True,
            )
            if X.shape != (5000, 100):
                raise DatasetNotReproducible(
                    f"Supplied Gemma dataset produced {X.shape}, not (5000, 100)."
                )
            return PreparedDataset(
                dataset,
                X,
                reference,
                metadata,
            )
        odds_name = dataset.removeprefix("odds-")
        X, labels, metadata = load_odds_dataset(
            odds_name, cache_dir=data_dir, offline=offline
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            reference = FastMCDEstimator(random_state=42).fit(X)
        metadata = {
            **metadata,
            "reference": "FastMCD",
            "contamination": float(labels.mean()),
            "reference_warning_count": len(captured),
            "reference_warning": (
                str(captured[0].message) if captured else ""
            ),
        }
        return PreparedDataset(dataset, X, reference, metadata)

    if dataset == "sp500":
        X, reference, metadata = fetch_sp500_data(
            cache_dir=data_dir, offline=offline, return_metadata=True
        )
        return PreparedDataset(
            dataset,
            X,
            reference,
            metadata,
        )
    if dataset.startswith("odds-"):
        odds_name = dataset.removeprefix("odds-")
        X, reference = generate_odds_surrogate(odds_name, seed=seed)
        return PreparedDataset(
            dataset,
            X,
            reference,
            {
                "source": "legacy_synthetic_surrogate",
                "seed": seed,
                "warning": "This is not the ODDS dataset.",
            },
        )
    if dataset == "genomics":
        try:
            X, reference, metadata = fetch_genomics_data(
                n_samples_target=1203,
                n_features_target=20,
                dataset_id="GSE6306",
                seed=seed,
                cache_dir=data_dir,
                offline=offline,
                reference="spearman",
                return_metadata=True,
            )
        except ExternalDataUnavailable:
            raise
        return PreparedDataset(
            dataset,
            X,
            reference,
            metadata,
        )
    raise AssertionError("unreachable")
