"""Dataset generation, download, validation, and preprocessing utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.io import loadmat


LEGACY_SP500_TICKERS = (
    "XOM",
    "GE",
    "MSFT",
    "JPM",
    "PG",
    "JNJ",
    "INTC",
    "PFE",
    "T",
    "VZ",
    "IBM",
    "KO",
)

SP500_PREPROCESSING: Dict[str, str] = {
    "price": "auto-adjusted close",
    "transform": "daily log return",
    "missing_rows": "drop any",
    "reference": "Pearson below 90th mean-absolute-return percentile",
}

ODDS_DATASETS: Dict[str, Dict[str, Any]] = {
    "mammography": {
        "url": (
            "https://www.dropbox.com/scl/fi/x36bm1kj7atqrd2/"
            "mammography.mat?rlkey=poytkapk2no64s7yy2ux4nzg8&dl=1"
        ),
        "sha256": "271ebb568314a856666d3504b4882e21b0ea6e1ba9e648ad256d572a36df597e",
        "shape": (11183, 6),
        "outliers": 260,
    },
    "satellite": {
        "url": (
            "https://www.dropbox.com/scl/fi/vmty1xcfhk2bnaz/"
            "satellite.mat?rlkey=13tlpynr63wmcpk323pvb1o40&dl=1"
        ),
        "sha256": "6feac3112b9c14e1c3e60afc437f2f3d29dc1000119c9f950c628078778d6aa0",
        "shape": (6435, 36),
        "outliers": 2036,
    },
}


class ExternalDataUnavailable(RuntimeError):
    """Raised when an explicitly requested external dataset cannot be obtained."""


def _default_cache_dir() -> Path:
    configured = os.environ.get("FORCE_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "force-estimator"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_required_manifest(path: Path, cache_name: str) -> Dict[str, Any]:
    """Load a mandatory cache sidecar without silently self-certifying data."""
    if not path.exists():
        raise ExternalDataUnavailable(
            f"Cached {cache_name} is missing its provenance manifest: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExternalDataUnavailable(
            f"Could not read {cache_name} cache manifest: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ExternalDataUnavailable(
            f"{cache_name} cache manifest must contain a JSON object."
        )
    return payload


def _require_manifest_fields(
    payload: Dict[str, Any], fields: Sequence[str], cache_name: str
) -> None:
    missing = sorted(field for field in fields if field not in payload)
    if missing:
        raise ExternalDataUnavailable(
            f"{cache_name} cache manifest is missing required fields: "
            + ", ".join(missing)
        )


def _validated_manifest_sha256(value: Any, cache_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ExternalDataUnavailable(
            f"{cache_name} cache manifest has an invalid SHA-256."
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise ExternalDataUnavailable(
            f"{cache_name} cache manifest has an invalid SHA-256."
        ) from exc
    return value.lower()


def _download_verified(
    *,
    url: str,
    destination: Path,
    expected_sha256: Optional[str],
    offline: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = _sha256(destination)
        if expected_sha256 is None or actual == expected_sha256:
            return destination
        raise ExternalDataUnavailable(
            f"Cached file checksum mismatch for {destination}: {actual}"
        )
    if offline:
        raise ExternalDataUnavailable(
            f"Offline mode is enabled and {destination.name} is not cached."
        )

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "force-estimator/1.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        actual = _sha256(temporary)
        if expected_sha256 is not None and actual != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise ExternalDataUnavailable(
                f"Downloaded checksum mismatch for {destination.name}: {actual}"
            )
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, ExternalDataUnavailable):
            raise
        raise ExternalDataUnavailable(
            f"Could not download {destination.name}: {exc}"
        ) from exc
    return destination


def _random_correlation(
    n_features: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    matrix = rng.random((n_features, n_features))
    covariance = matrix @ matrix.T
    scale = np.sqrt(np.diag(covariance))
    return covariance / np.outer(scale, scale)


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    contamination: float = 0.1,
    seed: Optional[int] = 42,
    *,
    contamination_distribution: str = "cauchy",
    outlier_scale: float = 10.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate the paper's Gaussian core and whole-row contamination model."""
    if n_samples < 5 or n_features < 2:
        raise ValueError("Synthetic data require at least 5 samples and 2 features.")
    if not 0.0 <= contamination < 1.0:
        raise ValueError("contamination must lie in [0, 1).")
    if contamination_distribution not in {"cauchy", "uniform"}:
        raise ValueError(
            "contamination_distribution must be 'cauchy' or 'uniform'."
        )
    if not np.isfinite(outlier_scale) or outlier_scale <= 0.0:
        raise ValueError("outlier_scale must be finite and positive.")

    rng = np.random.default_rng(seed)
    true_correlation = _random_correlation(n_features, rng)
    X = rng.multivariate_normal(
        np.zeros(n_features), true_correlation, n_samples
    )
    n_outliers = int(n_samples * contamination)
    if n_outliers:
        rows = rng.choice(n_samples, n_outliers, replace=False)
        if contamination_distribution == "cauchy":
            X[rows] = (
                rng.standard_cauchy((n_outliers, n_features)) * outlier_scale
            )
        else:
            X[rows] = rng.uniform(
                -outlier_scale,
                outlier_scale,
                (n_outliers, n_features),
            )
    return X, true_correlation


def generate_synthetic_data_fixed_corr(
    n_samples: int,
    n_features: int = 10,
    contamination: float = 0.1,
    seed: Optional[int] = None,
    *,
    contamination_distribution: str = "cauchy",
    outlier_scale: float = 10.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate protocol-selectable contamination with a fixed correlation matrix."""
    if n_samples < 5 or n_features < 2:
        raise ValueError("Synthetic data require at least 5 samples and 2 features.")
    if not 0.0 <= contamination < 1.0:
        raise ValueError("contamination must lie in [0, 1).")
    if contamination_distribution not in {"cauchy", "uniform"}:
        raise ValueError(
            "contamination_distribution must be 'cauchy' or 'uniform'."
        )
    if not np.isfinite(outlier_scale) or outlier_scale <= 0.0:
        raise ValueError("outlier_scale must be finite and positive.")
    correlation_rng = np.random.default_rng(42)
    true_correlation = _random_correlation(n_features, correlation_rng)
    data_rng = np.random.default_rng(seed)
    X = data_rng.multivariate_normal(
        np.zeros(n_features), true_correlation, n_samples
    )
    n_outliers = int(n_samples * contamination)
    if n_outliers:
        rows = data_rng.choice(n_samples, n_outliers, replace=False)
        if contamination_distribution == "cauchy":
            X[rows] = (
                data_rng.standard_cauchy((n_outliers, n_features))
                * outlier_scale
            )
        else:
            X[rows] = data_rng.uniform(
                -outlier_scale,
                outlier_scale,
                (n_outliers, n_features),
            )
    return X, true_correlation


def generate_odds_surrogate(
    dataset_name: str,
    *,
    n_max_samples: int = 20000,
    seed: int = 42,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reproduce the historical synthetic ODDS-like benchmark."""
    specs = {
        "mammography": (11183, 6, 0.023),
        "satellite": (6435, 36, 0.317),
    }
    if dataset_name not in specs:
        raise ValueError("dataset_name must be 'mammography' or 'satellite'.")
    n_samples, n_features, fraction = specs[dataset_name]
    n_samples = min(n_samples, n_max_samples)
    n_inliers = int(n_samples * (1.0 - fraction))
    n_outliers = n_samples - n_inliers
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n_features, n_features))
    covariance = matrix @ matrix.T
    scale = np.sqrt(np.diag(covariance))
    true_correlation = covariance / np.outer(scale, scale)
    clean = rng.multivariate_normal(
        np.zeros(n_features), true_correlation, n_inliers
    )
    if dataset_name == "mammography":
        outliers = rng.uniform(-5.0, 5.0, (n_outliers, n_features))
    else:
        outliers = rng.multivariate_normal(
            np.full(n_features, 3.0),
            np.eye(n_features) * 0.5,
            n_outliers,
        )
    X = np.vstack((clean, outliers))
    return rng.permutation(X), true_correlation


def load_odds_dataset(
    dataset_name: str,
    *,
    cache_dir: Optional[Path] = None,
    offline: bool = False,
) -> Tuple[NDArray[np.float64], NDArray[np.int64], Dict[str, Any]]:
    """Download and validate an official ODDS matrix and labels."""
    if dataset_name not in ODDS_DATASETS:
        raise ValueError("dataset_name must be 'mammography' or 'satellite'.")
    spec = ODDS_DATASETS[dataset_name]
    root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    destination = root / "odds" / f"{dataset_name}.mat"
    manifest_path = destination.with_suffix(".json")
    was_cached = destination.exists()
    retrieved_at = None
    if was_cached and manifest_path.exists():
        try:
            prior_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            retrieved_at = prior_manifest.get("retrieved_at")
        except (OSError, ValueError):
            retrieved_at = None
    path = _download_verified(
        url=spec["url"],
        destination=destination,
        expected_sha256=spec["sha256"],
        offline=offline,
    )
    payload = loadmat(path)
    if "X" not in payload or "y" not in payload:
        raise ExternalDataUnavailable(f"{path} does not contain X and y arrays.")
    X = np.asarray(payload["X"], dtype=np.float64)
    labels = np.asarray(payload["y"], dtype=np.int64).reshape(-1)
    if X.shape != tuple(spec["shape"]):
        raise ExternalDataUnavailable(
            f"{dataset_name} shape {X.shape} does not match {spec['shape']}."
        )
    if labels.shape != (X.shape[0],):
        raise ExternalDataUnavailable("ODDS labels do not match the data rows.")
    if set(np.unique(labels)) != {0, 1}:
        raise ExternalDataUnavailable("ODDS labels must contain only 0 and 1.")
    if int(labels.sum()) != int(spec["outliers"]):
        raise ExternalDataUnavailable("ODDS outlier count does not match metadata.")
    if not np.all(np.isfinite(X)):
        raise ExternalDataUnavailable("ODDS data contain non-finite values.")
    metadata = {
        "source": "ODDS",
        "dataset": dataset_name,
        "url": spec["url"],
        "path": str(path),
        "sha256": spec["sha256"],
        "shape": list(X.shape),
        "outliers": int(labels.sum()),
        "retrieved_at": retrieved_at
        or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "cache": "hit" if was_cached else "downloaded",
    }
    _write_manifest(manifest_path, metadata)
    return X, labels, metadata


def fetch_odds_dataset(
    dataset_name: str,
    n_max_samples: int = 20000,
    *,
    cache_dir: Optional[Path] = None,
    offline: bool = False,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compatibility loader returning real ODDS data and a FastMCD reference."""
    X, labels, _ = load_odds_dataset(
        dataset_name, cache_dir=cache_dir, offline=offline
    )
    if n_max_samples < X.shape[0]:
        rng = np.random.default_rng(42)
        selected = rng.choice(X.shape[0], n_max_samples, replace=False)
        selected.sort()
        X = X[selected]
        labels = labels[selected]
    from .estimators import FastMCDEstimator

    reference = FastMCDEstimator(random_state=42).fit(X)
    return X, reference


def fetch_sp500_data(
    *,
    tickers: Optional[Sequence[str]] = None,
    start: str = "2000-01-01",
    end: str = "2025-01-01",
    cache_dir: Optional[Path] = None,
    offline: bool = False,
    return_metadata: bool = False,
) -> tuple:
    """Load daily log returns and the paper's low-volatility reference."""
    symbols = tuple(tickers or LEGACY_SP500_TICKERS)
    if len(symbols) < 2:
        raise ValueError("At least two tickers are required.")
    root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    key = hashlib.sha256(
        ("|".join(symbols) + f"|{start}|{end}").encode("utf-8")
    ).hexdigest()[:16]
    returns_path = root / "sp500" / f"returns-{key}.csv"
    manifest_path = returns_path.with_suffix(".json")
    retrieved_at = None
    prior_manifest: Optional[Dict[str, Any]] = None
    if returns_path.exists():
        prior_manifest = _load_required_manifest(
            manifest_path, "S&P returns"
        )
        _require_manifest_fields(
            prior_manifest,
            (
                "sha256",
                "source",
                "tickers",
                "start",
                "end",
                "rows",
                "columns",
                "retrieved_at",
                "preprocessing",
            ),
            "S&P returns",
        )
        expected_checksum = _validated_manifest_sha256(
            prior_manifest["sha256"], "S&P returns"
        )
        if _sha256(returns_path) != expected_checksum:
            raise ExternalDataUnavailable(
                "Cached S&P returns checksum does not match its manifest."
            )
        if prior_manifest["source"] != "yfinance":
            raise ExternalDataUnavailable(
                "S&P cache source does not match the yfinance protocol."
            )
        if tuple(prior_manifest["tickers"]) != symbols:
            raise ExternalDataUnavailable(
                "S&P cache manifest does not match the requested ticker manifest."
            )
        if prior_manifest["start"] != start or prior_manifest["end"] != end:
            raise ExternalDataUnavailable(
                "S&P cache manifest does not match the requested date range."
            )
        if prior_manifest["preprocessing"] != SP500_PREPROCESSING:
            raise ExternalDataUnavailable(
                "S&P cache preprocessing does not match the requested protocol."
            )
        retrieved_at = prior_manifest["retrieved_at"]
        if not isinstance(retrieved_at, str) or not retrieved_at:
            raise ExternalDataUnavailable(
                "S&P cache manifest has an invalid retrieval timestamp."
            )
        returns = pd.read_csv(returns_path, index_col=0)
    else:
        if offline:
            raise ExternalDataUnavailable(
                "Offline mode is enabled and S&P returns are not cached."
            )
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ExternalDataUnavailable("yfinance is not installed.") from exc
        try:
            downloaded = yf.download(
                list(symbols),
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            prices = downloaded["Close"]
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(symbols[0])
            prices = prices.loc[:, list(symbols)]
            returns = np.log(prices / prices.shift(1)).dropna(how="any")
        except Exception as exc:
            raise ExternalDataUnavailable(
                f"Could not download S&P data: {exc}"
            ) from exc
        if returns.empty:
            raise ExternalDataUnavailable("S&P download returned no complete rows.")
        returns_path.parent.mkdir(parents=True, exist_ok=True)
        returns.to_csv(returns_path)
        retrieved_at = datetime.now(timezone.utc).isoformat()
    if list(returns.columns) != list(symbols):
        raise ExternalDataUnavailable(
            "S&P cache/download columns do not match the requested ticker manifest."
        )
    try:
        returns = returns.apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise ExternalDataUnavailable(
            f"S&P returns contain non-numeric values: {exc}"
        ) from exc
    if returns.shape[0] < 5 or returns.shape[1] < 2:
        raise ExternalDataUnavailable(
            f"S&P preprocessing produced unusable shape {returns.shape}."
        )
    if prior_manifest is not None:
        if (
            prior_manifest["rows"] != int(returns.shape[0])
            or prior_manifest["columns"] != int(returns.shape[1])
        ):
            raise ExternalDataUnavailable(
                "S&P cache shape does not match its provenance manifest."
            )
    if not np.all(np.isfinite(returns.to_numpy(dtype=np.float64))):
        raise ExternalDataUnavailable("S&P returns contain non-finite values.")

    metadata_payload = {
        "source": "yfinance",
        "tickers": list(symbols),
        "start": start,
        "end": end,
        "rows": int(returns.shape[0]),
        "columns": int(returns.shape[1]),
        "retrieved_at": retrieved_at
        or datetime.fromtimestamp(
            returns_path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "sha256": _sha256(returns_path),
        "path": str(returns_path),
        "preprocessing": SP500_PREPROCESSING,
        "python": platform.python_version(),
    }
    _write_manifest(manifest_path, metadata_payload)

    X = np.asarray(returns, dtype=np.float64)
    volatility = np.mean(np.abs(X), axis=1)
    quiet = X[volatility < np.quantile(volatility, 0.90)]
    reference = np.corrcoef(quiet, rowvar=False)
    result = (X, np.asarray(reference, dtype=np.float64))
    if return_metadata:
        return (*result, metadata_payload)
    return result


def _download_gemma_processed(
    dataset_id: str,
    *,
    cache_dir: Path,
    offline: bool,
) -> Path:
    url = (
        "https://gemma.msl.ubc.ca/rest/v2/datasets/"
        f"{dataset_id}/data/processed?filter=false"
    )
    destination = (
        cache_dir / "gemma" / f"{dataset_id}-processed-unfiltered.tsv"
    )
    manifest_path = destination.with_suffix(".json")
    expected_sha256 = None
    was_cached = destination.exists()
    if was_cached:
        manifest = _load_required_manifest(manifest_path, "Gemma expression")
        _require_manifest_fields(
            manifest,
            (
                "source",
                "url",
                "dataset_id",
                "processed_sha256",
                "retrieved_at",
            ),
            "Gemma expression",
        )
        if manifest["source"] != "Gemma REST":
            raise ExternalDataUnavailable(
                "Gemma cache source does not match the Gemma REST protocol."
            )
        if manifest["url"] != url or manifest["dataset_id"] != dataset_id:
            raise ExternalDataUnavailable(
                "Gemma cache manifest does not match the requested dataset."
            )
        if (
            not isinstance(manifest["retrieved_at"], str)
            or not manifest["retrieved_at"]
        ):
            raise ExternalDataUnavailable(
                "Gemma cache manifest has an invalid retrieval timestamp."
            )
        expected_sha256 = _validated_manifest_sha256(
            manifest["processed_sha256"], "Gemma expression"
        )
    path = _download_verified(
        url=url,
        destination=destination,
        expected_sha256=expected_sha256,
        offline=offline,
    )
    if not was_cached:
        _write_manifest(
            manifest_path,
            {
                "source": "Gemma REST",
                "url": url,
                "dataset_id": dataset_id,
                "processed_sha256": _sha256(path),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return path


def fetch_genomics_data(
    n_samples_target: int = 1203,
    n_features_target: int = 20,
    *,
    dataset_id: str = "GSE6306",
    seed: int = 42,
    cache_dir: Optional[Path] = None,
    offline: bool = False,
    reference: str = "spearman",
    return_metadata: bool = False,
) -> tuple:
    """Fetch a pinned Gemma processed-expression dataset deterministically."""
    if n_samples_target < 5 or n_features_target < 2:
        raise ValueError("Genomics targets require at least 5 samples and 2 genes.")
    if reference not in {"spearman", "fastmcd"}:
        raise ValueError("reference must be 'spearman' or 'fastmcd'.")
    root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    manifest_path = (
        root / "gemma" / f"{dataset_id}-processed-unfiltered.json"
    )
    retrieved_at = None
    if manifest_path.exists():
        try:
            retrieved_at = json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("retrieved_at")
        except (OSError, ValueError):
            retrieved_at = None
    path = _download_gemma_processed(
        dataset_id, cache_dir=root, offline=offline
    )
    try:
        with path.open("rb") as handle:
            compressed = handle.read(2) == b"\x1f\x8b"
        frame = pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
            compression="gzip" if compressed else None,
            comment="#",
        )
    except Exception as exc:
        raise ExternalDataUnavailable(
            f"Could not parse Gemma expression data: {exc}"
        ) from exc

    metadata_columns = {
        "Probe",
        "Sequence",
        "GeneSymbol",
        "GeneName",
        "GemmaId",
        "NCBIid",
    }
    sample_columns = [
        column for column in frame.columns if column not in metadata_columns
    ]
    expression = frame.loc[:, sample_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    expression = expression.dropna(axis=1, how="any").dropna(axis=0, how="any")
    sample_ids = np.asarray(expression.columns.astype(str), dtype=object)
    if "GeneSymbol" in frame:
        feature_labels = frame.loc[expression.index, "GeneSymbol"]
        fallback_labels = frame.loc[expression.index].index.astype(str)
        feature_ids = np.asarray(
            [
                str(label)
                if pd.notna(label) and str(label).strip()
                else str(fallback)
                for label, fallback in zip(feature_labels, fallback_labels)
            ],
            dtype=object,
        )
    elif "Probe" in frame:
        feature_ids = np.asarray(
            frame.loc[expression.index, "Probe"].astype(str), dtype=object
        )
    else:
        feature_ids = np.asarray(expression.index.astype(str), dtype=object)
    X = np.asarray(expression.T, dtype=np.float64)
    if X.shape[0] < 5 or X.shape[1] < 2:
        raise ExternalDataUnavailable(
            f"Gemma preprocessing produced unusable shape {X.shape}."
        )

    rng = np.random.default_rng(seed)
    if X.shape[0] > n_samples_target:
        rows = rng.choice(X.shape[0], n_samples_target, replace=False)
        rows.sort()
        X = X[rows]
        sample_ids = sample_ids[rows]
    if X.shape[1] > n_features_target:
        variances = np.var(X, axis=0)
        columns = np.argsort(variances, kind="stable")[-n_features_target:]
        columns.sort()
        X = X[:, columns]
        feature_ids = feature_ids[columns]

    if reference == "spearman":
        ranks = np.column_stack(
            [stats.rankdata(X[:, index]) for index in range(X.shape[1])]
        )
        reference_correlation = np.corrcoef(ranks, rowvar=False)
    else:
        from .estimators import FastMCDEstimator

        reference_correlation = FastMCDEstimator(random_state=42).fit(X)
    metadata_payload = {
        "source": "Gemma REST",
        "url": (
            "https://gemma.msl.ubc.ca/rest/v2/datasets/"
            f"{dataset_id}/data/processed?filter=false"
        ),
        "dataset_id": dataset_id,
        "processed_sha256": _sha256(path),
        "retrieved_at": retrieved_at
        or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "path": str(path),
        "selected_shape": list(X.shape),
        "selected_sample_ids": sample_ids.tolist(),
        "selected_feature_ids": feature_ids.tolist(),
        "sample_target": n_samples_target,
        "feature_target": n_features_target,
        "seed": seed,
        "reference": reference,
        "preprocessing": {
            "numeric_conversion": "coerce and drop incomplete rows/columns",
            "sample_selection": "fixed-seed sorted subsample when needed",
            "feature_selection": "top variance with stable tie ordering",
        },
    }
    _write_manifest(path.with_suffix(".json"), metadata_payload)
    result = (X, np.asarray(reference_correlation, dtype=np.float64))
    if return_metadata:
        return (*result, metadata_payload)
    return result
