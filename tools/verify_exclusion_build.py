#!/usr/bin/env python3
"""Exercise shipped ignore and packaging exclusions using only synthetic files."""
from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from verify_distribution import _extract_source


# Independent expectations: do not import the inventory verifier's exclusions.
RESEARCH_DIRECTORIES = (
    "manuscript", "paper", "supplement", "literature", "archive", "data",
    "results", "figures", "output", "generated_figures", "generated-paper-assets",
    "verification_results",
)
RESEARCH_FILES = (
    "mathematics-14-00191.pdf", "force-improvement-plan.md", "local-improvement-plan.md",
)
FIXTURES = tuple(f"{name}/nested/exclusion-fixture.txt" for name in RESEARCH_DIRECTORIES) + RESEARCH_FILES
SOURCE_REQUIRED = (
    ".gitignore", "MANIFEST.in", "pyproject.toml", "README.md", "VERIFICATION.md",
    "LICENSE", "CITATION.cff", ".github/workflows/ci.yml",
    "tests/test_distribution_boundary.py", "tools/verify_distribution.py",
    "tools/verify_exclusion_build.py",
    'src/force/__init__.py',
    'src/force/estimators.py',
    'examples/usage_example.py',
    'scripts/run_benchmark.py',
)
WHEEL_REQUIRED = (
    'force/__init__.py',
    'force/estimators.py',
)


def verify_ignore_rules(source: Path) -> None:
    """Check only the shipped .gitignore, independent of the user's Git setup."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    subprocess.run(
        ["git", "init", "--quiet", "--template=", str(source)],
        check=True, env=environment,
    )
    result = subprocess.run(
        ["git", "-c", f"core.excludesFile={os.devnull}", "check-ignore", "--no-index", "-z", "--stdin"],
        input="\0".join((*FIXTURES, *SOURCE_REQUIRED)) + "\0",
        capture_output=True, text=True, cwd=source, env=environment,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git check-ignore failed: {result.stderr}")
    ignored = set(result.stdout.split("\0")) - {""}
    missing = set(FIXTURES) - ignored
    retained = set(SOURCE_REQUIRED) & ignored
    if missing or retained:
        raise ValueError(
            f"shipped .gitignore regression: not ignored={sorted(missing)}, "
            f"retained files ignored={sorted(retained)}"
        )


def verify_inventory(names, required, artifact: str, *, wheel: bool = False) -> None:
    """Assert exclusions and retained files without using verifier policy constants."""
    names = set(names)
    leaked = []
    for name in names:
        parts = PurePosixPath(name).parts
        if (
            wheel and len(parts) >= 3 and parts[0].endswith(".data")
            and parts[1] in ("data", "purelib", "platlib", "scripts", "headers")
        ):
            parts = parts[2:]
        if parts and (
            parts[0] in RESEARCH_DIRECTORIES
            or (len(parts) == 1 and (
                parts[0] in RESEARCH_FILES or fnmatch.fnmatchcase(parts[0], "*-improvement-plan.md")
            ))
        ):
            leaked.append(name)
    missing = set(required) - names
    if leaked or missing:
        raise ValueError(
            f"{artifact} fixture build regression: excluded files={sorted(leaked)}, "
            f"missing retained files={sorted(missing)}"
        )


def verify_fixture_build(source: Path, destination: Path) -> None:
    # Force discovery first; the shipped rules must then remove these candidates.
    manifest = source / "MANIFEST.in"
    manifest.write_text(
        "".join(f"include {name}\n" for name in FIXTURES)
        + manifest.read_text(encoding="utf-8"), encoding="utf-8",
    )
    # Both flags are intentional: build the wheel directly from the fixture tree,
    # so a clean sdist cannot conceal a leak in a direct wheel build.
    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(destination)],
        check=True, cwd=source,
    )
    archives = list(destination.glob("*.tar.gz"))
    wheels = list(destination.glob("*.whl"))
    if len(archives) != 1 or len(wheels) != 1:
        raise ValueError("fixture build must produce exactly one source archive and one wheel")
    with tarfile.open(archives[0]) as archive:
        names = [PurePosixPath(member.name).parts for member in archive.getmembers() if member.isfile()]
    if not names or len({parts[0] for parts in names}) != 1:
        raise ValueError("fixture source archive must contain one project directory")
    verify_inventory(
        [PurePosixPath(*parts[1:]).as_posix() for parts in names], SOURCE_REQUIRED, "source archive",
    )
    with zipfile.ZipFile(wheels[0]) as archive:
        verify_inventory(archive.namelist(), WHEEL_REQUIRED, "direct wheel", wheel=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()
    archives = list(args.dist_dir.resolve().glob("*.tar.gz"))
    if len(archives) != 1:
        raise ValueError("expected exactly one built source archive")
    with tempfile.TemporaryDirectory(prefix="force-exclusion-build-") as temporary:
        work = Path(temporary)
        source = _extract_source(archives[0], work / "source")
        missing = [name for name in SOURCE_REQUIRED if not (source / name).is_file()]
        if missing:
            raise ValueError(f"input source archive is missing retained files: {missing}")
        for name in FIXTURES:
            path = source / name
            if path.exists():
                raise ValueError(f"input source archive already contains excluded fixture path: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Synthetic exclusion regression fixture; no research content.\n", encoding="utf-8")
        verify_ignore_rules(source)
        verify_fixture_build(source, work / "built")
    print("Verified shipped Git ignores, source exclusions, and direct-wheel exclusions with synthetic fixtures.")


if __name__ == "__main__":
    main()
