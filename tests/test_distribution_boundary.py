"""Independent publication boundaries and executable README contracts."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "distribution_boundary", ROOT / "tools/verify_distribution.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
# Deliberately independent of the verifier: deleting a rule must not delete tests.
EXCLUDED = [
    "manuscript/nested/local.txt", "paper/nested/local.txt",
    "supplement/nested/local.txt", "literature/nested/local.txt",
    "archive/nested/local.txt", "data/nested/local.txt",
    "results/nested/local.txt", "figures/nested/local.txt",
    "output/nested/local.txt", "generated_figures/nested/local.txt",
    "generated-paper-assets/nested/local.txt", "verification_results/nested/local.txt",
    "mathematics-14-00191.pdf", "force-improvement-plan.md",
    "local-improvement-plan.md",
]
WHEEL_PREFIXES = [
    "", "package-1.0.data/data/", "package-1.0.data/purelib/",
    "package-1.0.data/platlib/", "package-1.0.data/scripts/",
    "package-1.0.data/headers/",
]


@pytest.mark.parametrize("relative", EXCLUDED)
@pytest.mark.parametrize("prefix", WHEEL_PREFIXES)
def test_wheel_rejects_local_research_material(tmp_path, relative, prefix):
    source = tmp_path / "source"
    source.mkdir()
    member = prefix + relative
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/__init__.py", "")
        archive.writestr(member, "synthetic research fixture")
    with pytest.raises(ValueError, match="excluded research material") as error:
        verifier.verify_distribution_contents(wheel, source)
    assert repr(member) in str(error.value)


@pytest.mark.parametrize("relative", EXCLUDED)
def test_source_rejects_local_research_material(tmp_path, relative):
    source = tmp_path / "source"
    target = source / relative
    target.parent.mkdir(parents=True)
    target.write_text("synthetic research fixture")
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("package/__init__.py", "")
    with pytest.raises(ValueError, match="excluded research material"):
        verifier.verify_distribution_contents(wheel, source)


@pytest.mark.parametrize("prefix", WHEEL_PREFIXES)
def test_distribution_retains_software_and_configuration(tmp_path, prefix):
    source = tmp_path / "source"
    source.mkdir()
    retained = [
        "README.md", "CITATION.cff", "LICENSE", "tests/test_estimator.py",
        "src/package/data.py", "configuration/paper_configuration.json",
        "preregistration/frozen_configuration.json",
        "provenance/CURRENT_IMPLEMENTATION_MANIFEST.json",
        "src/package/experiments/paper_configuration.json",
        "package/data/runtime.json", "package/archive/runtime.json",
    ]
    for relative in retained:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for relative in retained:
            archive.writestr(prefix + relative, "fixture")
    verifier.verify_distribution_contents(wheel, source)


@pytest.mark.parametrize("scheme", ["data", "purelib", "platlib", "scripts", "headers"])
def test_repository_paths_are_not_wheel_installation_paths(scheme):
    verifier.verify_no_research_files([f"package-1.0.data/{scheme}/archive/local.txt"])


@pytest.mark.parametrize("relative", [
    "package/data/archive/local.txt", "package-1.0.data/unknown/archive/local.txt",
    "nested/package-1.0.data/data/archive/local.txt",
])
def test_wheel_layout_is_only_normalized_at_root(relative):
    verifier.verify_no_research_wheel_members([relative])


def marked(name, code="pass"):
    return f"<!-- executable-example: {name} -->\n```python\n{code}\n```\n"


@pytest.mark.parametrize("extra", [False, True])
def test_all_marked_examples_execute_independently(tmp_path, extra):
    # Real standard-library-only subprocesses also prove namespace isolation.
    code = "assert 'batch_only' not in globals(); batch_only = 1"
    text = marked("batch", code) + marked("streaming", code)
    if extra:
        text += marked("extra_3-demo", code)
    # Unmarked snippets need not be valid Python or standalone programs.
    text += "```python\nthis is not valid Python!\n```\n"
    text += '~~~~text\n<!-- executable-example: not-a-directive -->\n~~~~\n'
    (tmp_path / "README.md").write_text(text)
    assert list(verifier.readme_examples(text)) == (
        ["batch", "streaming", "extra_3-demo"] if extra else ["batch", "streaming"]
    )
    verifier.run_readme_examples(tmp_path, Path(sys.executable), os.environ.copy(), tmp_path)


def test_readme_runner_executes_every_marked_block_with_isolation(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text(
        marked("batch", "a = 1") + marked("streaming", "b = 2") + marked("third", "c = 3")
    )
    calls = []
    monkeypatch.setattr(verifier.subprocess, "run", lambda args, **kw: calls.append((args, kw)))
    python = tmp_path / "installed-venv/python"
    environment = {"fixture": "environment"}
    verifier.run_readme_examples(tmp_path, python, environment, tmp_path)
    assert calls == [
        ([str(python), "-I", "-c", code + "\n"],
         {"check": True, "env": environment, "cwd": tmp_path})
        for code in ["a = 1", "b = 2", "c = 3"]
    ]


@pytest.mark.parametrize("names", [[], ["batch"], ["streaming"], ["additional"]])
def test_readme_requires_batch_and_streaming(names):
    with pytest.raises(ValueError, match="missing required"):
        verifier.readme_examples("".join(marked(name) for name in names))


@pytest.mark.parametrize("name", ["batch", "streaming", "extra"])
def test_readme_rejects_duplicate_identifiers(name):
    text = marked("batch") + marked("streaming")
    if name == "extra":
        text += marked("extra")
    with pytest.raises(ValueError, match="duplicate"):
        verifier.readme_examples(text + marked(name))


@pytest.mark.parametrize("invalid", [
    "<!-- executable-example -->\n```python\npass\n```",
    "<!-- executable-example: -->\n```python\npass\n```",
    "<!-- executable-example: two words -->\n```python\npass\n```",
    "<!-- executable-example: Extra -->\n```python\npass\n```",
    "<!-- executable-example: extra --\n```python\npass\n```",
    "prefix <!-- executable-example: extra -->\n```python\npass\n```",
    "<!-- executable-example: extra -->\n\n```python\npass\n```",
    "<!-- executable-example: extra -->\n```bash\ntrue\n```",
    "<!-- executable-example: extra -->",
    "<!-- executable-example: extra -->\n```python\npass",
    "<!-- executable-example: extra -->\n```python\n```",
])
def test_readme_rejects_malformed_marked_examples(invalid):
    with pytest.raises(ValueError, match="executable-example"):
        verifier.readme_examples(marked("batch") + marked("streaming") + invalid)


def test_checked_in_readme_has_required_examples():
    assert {"batch", "streaming"} <= verifier.readme_examples(
        (ROOT / "README.md").read_text(encoding="utf-8")
    ).keys()
