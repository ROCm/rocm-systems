# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_assemble_changelog.py."""

import os
import subprocess
from pathlib import Path

import pytest

import rccl_assemble_changelog as assemble

CHANGELOG = """\
# Changelog

## RCCL 2.32.3 for ROCm 10.3.0 (Unreleased)

### Added
* `RCCL_OLD`: old.

### Resolved issues
* Fixed an old bug.

## RCCL 2.32.3 for ROCm 10.2.0

### Added
* `RCCL_SHIPPED`: shipped.
"""


def _write(root: Path, rel: Path, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _commit(repo: Path, files: dict[Path, str], message: str) -> None:
    for rel, text in files.items():
        _write(repo, rel, text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)


def _repo(tmp_path: Path) -> Path:
    tmp_path.mkdir(exist_ok=True)
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.email", "rccl-ci@example.com")
    _git(tmp_path, "config", "user.name", "RCCL CI")
    _commit(tmp_path, {assemble.CHANGELOG: CHANGELOG}, "base")
    return tmp_path


def test_a_fragment_goes_to_the_bottom_of_its_section_in_the_newest_release() -> None:
    merged = assemble.merge(CHANGELOG, {"Added": ["* `RCCL_NEW`: new."]})
    assert "* `RCCL_OLD`: old.\n* `RCCL_NEW`: new.\n\n### Resolved issues" in merged
    assert merged.endswith("* `RCCL_SHIPPED`: shipped.\n")
    assert merged.count("RCCL_NEW") == 1


def test_a_newest_release_that_is_not_unreleased_is_an_error() -> None:
    shipped = CHANGELOG.replace(" (Unreleased)", "")
    with pytest.raises(ValueError, match="Unreleased"):
        assemble.merge(shipped, {"Added": ["* `RCCL_NEW`: new."]})


def test_a_missing_section_is_created_in_the_standard_order() -> None:
    merged = assemble.merge(CHANGELOG, {"Changed": ["* Changed."], "Known issues": ["* Known."]})
    unreleased = merged.split("## RCCL 2.32.3 for ROCm 10.2.0")[0]
    titles = [line for line in unreleased.splitlines() if line.startswith("### ")]
    assert titles == ["### Added", "### Changed", "### Resolved issues", "### Known issues"]


def test_a_multi_line_fragment_keeps_its_lines() -> None:
    merged = assemble.merge(CHANGELOG, {"Resolved issues": ["* Fixed a bug.\n  It was bad."]})
    assert "* Fixed an old bug.\n* Fixed a bug.\n  It was bad.\n\n## RCCL" in merged


def test_fragments_are_read_in_merge_order_and_the_readme_is_skipped(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {assemble.FRAGMENTS / "9999-b.added.md": "* B.\n"}, "merge #9999")
    _commit(repo, {assemble.FRAGMENTS / "14316-a.added.md": "* A.\n"}, "merge #14316")
    _commit(repo, {assemble.FRAGMENTS / "README.md": "Format notes.\n"}, "docs")
    fragments, paths = assemble.read_fragments(repo)
    assert fragments["Added"] == ["* B.", "* A."]
    assert [path.name for path in paths] == ["9999-b.added.md", "14316-a.added.md"]


def test_editing_a_fragment_keeps_its_merge_order(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {assemble.FRAGMENTS / "1-a.added.md": "* A.\n"}, "merge #1")
    _commit(repo, {assemble.FRAGMENTS / "2-b.added.md": "* B.\n"}, "merge #2")
    _commit(repo, {assemble.FRAGMENTS / "1-a.added.md": "* A, reworded.\n"}, "merge #3")
    fragments, _paths = assemble.read_fragments(repo)
    assert fragments["Added"] == ["* A, reworded.", "* B."]


def test_a_non_ascii_fragment_name_keeps_its_merge_order(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {assemble.FRAGMENTS / "2-café.added.md": "* First.\n"}, "merge #2")
    _commit(repo, {assemble.FRAGMENTS / "1-b.added.md": "* Second.\n"}, "merge #1")
    fragments, _paths = assemble.read_fragments(repo)
    assert fragments["Added"] == ["* First.", "* Second."]


def test_a_non_utf8_fragment_name_keeps_its_merge_order(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {assemble.FRAGMENTS / os.fsdecode(b"2-caf\xe9.added.md"): "* First.\n"}, "merge #2")
    _commit(repo, {assemble.FRAGMENTS / "1-b.added.md": "* Second.\n"}, "merge #1")
    fragments, _paths = assemble.read_fragments(repo)
    assert fragments["Added"] == ["* First.", "* Second."]


def test_a_shallow_clone_is_an_error_because_it_loses_the_merge_order(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "full")
    _commit(repo, {assemble.FRAGMENTS / "9999-b.added.md": "* B.\n"}, "merge #9999")
    _commit(repo, {assemble.FRAGMENTS / "14316-a.added.md": "* A.\n"}, "merge #14316")
    _git(tmp_path, "clone", "--depth", "1", f"file://{repo}", "shallow")
    with pytest.raises(ValueError, match="shallow"):
        assemble.read_fragments(tmp_path / "shallow")


def test_a_symlinked_fragment_is_an_error(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / assemble.FRAGMENTS).mkdir(parents=True)
    (repo / assemble.FRAGMENTS / "10-a.added.md").symlink_to("../../../.git/config")
    _commit(repo, {}, "merge #10")
    with pytest.raises(ValueError, match="10-a.added.md"):
        assemble.read_fragments(repo)


def test_a_fragment_with_an_unknown_section_is_an_error(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {assemble.FRAGMENTS / "10-a.fixed.md": "* A.\n"}, "merge #10")
    with pytest.raises(ValueError, match="10-a.fixed.md"):
        assemble.read_fragments(repo)


def test_main_writes_the_fragments_in_merge_order_and_deletes_them(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _commit(repo, {assemble.FRAGMENTS / "9999-b.added.md": "* `RCCL_B`: b.\n"}, "merge #9999")
    _commit(repo, {assemble.FRAGMENTS / "14316-a.added.md": "* `RCCL_A`: a.\n"}, "merge #14316")
    _commit(repo, {assemble.FRAGMENTS / "README.md": "Format notes.\n"}, "docs")

    assert assemble.main(["--root", str(repo)]) == 0
    changelog = (repo / assemble.CHANGELOG).read_text(encoding="utf-8")
    assert "* `RCCL_OLD`: old.\n* `RCCL_B`: b.\n* `RCCL_A`: a.\n\n### Resolved issues" in changelog
    assert sorted(path.name for path in (repo / assemble.FRAGMENTS).iterdir()) == ["README.md"]


def test_main_with_no_fragments_leaves_a_released_changelog_alone(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    shipped = CHANGELOG.replace(" (Unreleased)", "")
    _commit(repo, {assemble.CHANGELOG: shipped}, "release")
    (repo / assemble.FRAGMENTS).mkdir(parents=True)

    assert assemble.main(["--root", str(repo)]) == 0
    assert (repo / assemble.CHANGELOG).read_text(encoding="utf-8") == shipped
