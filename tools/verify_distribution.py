#!/usr/bin/env python3
"""Verify built distributions in a fresh environment outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile


def _extract_source(archive_path: Path, destination: Path) -> Path:
    """Extract ordinary source files without links or paths outside destination."""
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.name
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsupported source archive member: {member.name!r}")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"source archive path escapes destination: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("source archive must contain one project directory")
    return roots[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    distributions = args.dist_dir.resolve()
    wheels = sorted(distributions.glob("*.whl"))
    archives = sorted(distributions.glob("*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise ValueError("expected exactly one wheel and one source archive")

    environment = os.environ.copy()
    for name in (
        "PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"
    ):
        environment.pop(name, None)
    for name in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    environment["MPLBACKEND"] = "Agg"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    with tempfile.TemporaryDirectory(prefix="force-distribution-check-") as temporary:
        work = Path(temporary)
        virtualenv = work / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(virtualenv)],
            check=True, env=environment, cwd=work,
        )
        binaries = virtualenv / ("Scripts" if os.name == "nt" else "bin")
        python = binaries / ("python.exe" if os.name == "nt" else "python")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check",
             str(wheels[0]), "pytest"],
            check=True, env=environment, cwd=work,
        )
        subprocess.run(
            [str(python), "-m", "pip", "check"],
            check=True, env=environment, cwd=work,
        )
        # Isolated mode excludes the checkout, ambient user site and PYTHONPATH.
        subprocess.run(
            [str(python), "-I", "-c", """
from importlib import metadata
from pathlib import Path
import sys
import numpy as np
import force
from force import ForceEstimator, StreamingForceEstimator

assert Path(force.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
distribution = metadata.distribution('force-estimator')
assert distribution.metadata['License-Expression'] == 'MIT'
licenses = [path for path in distribution.files if path.name == 'LICENSE']
assert licenses and all(distribution.locate_file(path).is_file() for path in licenses)
values = np.random.default_rng(19).normal(size=(500, 6))
batch = ForceEstimator()
matrix = batch.fit(values)
assert matrix.shape == (6, 6) and np.isfinite(matrix).all()

def source():
    for start in range(0, len(values), 37):
        yield values[start:start + 37]

replay = ForceEstimator()
np.testing.assert_allclose(replay.fit_replayable(source, chunk_size=23), matrix,
                           rtol=1e-12, atol=1e-12)
np.testing.assert_array_equal(replay.accepted_counts_, batch.accepted_counts_)
online = StreamingForceEstimator(window_size=100)
for block in source():
    online.update_batch(block)
snapshot = online.finalize()
assert snapshot.shape == (6, 6) and np.isfinite(snapshot).all()
assert online.status_ == 'estimating'
assert online.calibration_count_ == 100 and online.estimation_count_ == 400
print('Installed wheel: license, batch, replay and online APIs passed.')
"""],
            check=True, env=environment, cwd=work,
        )

        source = _extract_source(archives[0], work / "source")
        required = (
            "LICENSE", "CITATION.cff", "README.md", "VERIFICATION.md",
            "mathematics-14-00191.pdf", "MANIFEST.in", "pyproject.toml",
            ".github/workflows/ci.yml", "tests/__init__.py",
            "scripts/run_benchmark.py", "scripts/run_algorithm_diagnostics.py",
            "generate_figures.py", "generate_breakdown_figure.py",
            "examples/usage_example.py", "tools/verify_distribution.py",
            "archive/README.md", "archive/MANIFEST.json",
        )
        missing = [name for name in required if not (source / name).is_file()]
        if missing:
            raise ValueError(f"source archive is missing required files: {missing}")
        inventory = json.loads((source / "archive/MANIFEST.json").read_text(encoding="utf-8"))
        if inventory.get("schema_version") != 1 or not inventory.get("files"):
            raise ValueError("unsupported or empty historical archive inventory")
        seen = set()
        for record in inventory["files"]:
            path = source / record["archive_path"]
            if not path.resolve().is_relative_to((source / "archive").resolve()):
                raise ValueError("historical inventory path escapes archive directory")
            if path.resolve() in seen:
                raise ValueError("duplicate historical inventory path")
            seen.add(path.resolve())
            contents = path.read_bytes()
            if (
                len(contents) != record["bytes"]
                or hashlib.sha256(contents).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"historical artifact does not match inventory: {path}")
        # The archive's pytest configuration inserts its own src directory.
        # Check that import explicitly too; the installed wheel cannot mask it.
        subprocess.run(
            [str(python), "-c", """
from pathlib import Path
import sys
import pytest

root = Path.cwd().resolve()
sys.path.insert(0, str(root / 'src'))
import force
assert Path(force.__file__).resolve().is_relative_to(root / 'src')
raise SystemExit(pytest.main(['-q']))
"""],
            check=True, env=environment, cwd=source,
        )
        subprocess.run(
            [str(python), "-I", "examples/usage_example.py"],
            check=True, env=environment, cwd=source,
        )
        print("Verified installed wheel and the source archive's full test suite.")


if __name__ == "__main__":
    main()
