"""Integration test for scripts/apply-opencode-patches.sh.

Runs the script in CHECK_ONLY mode against the real submodule to guarantee
every .patches/*.patch applies in sequence without leaving the submodule
dirty. Skipped when the submodule is not initialized.
"""

from __future__ import annotations

import os
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest


_PERFXPERT_ROOT = Path(__file__).resolve().parents[2]
_OPENCODE_DIR = _PERFXPERT_ROOT / "opencode"
_SCRIPT = _PERFXPERT_ROOT / "scripts" / "apply-opencode-patches.sh"
_PATCH_DIR = _PERFXPERT_ROOT / ".patches"
_BUILD_SCRIPT = _PERFXPERT_ROOT / "scripts" / "build-bundled-opencode.sh"


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)


def _init_fake_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "opencode"
    repo.mkdir()
    _run(["git", "init"], cwd=repo)
    tracked = repo / "tracked.txt"
    tracked.write_text("line one\n")
    _run(["git", "add", "tracked.txt"], cwd=repo)
    _run(
        [
            "git",
            "-c",
            "user.name=PerfXpert Tests",
            "-c",
            "user.email=perfxpert-tests@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=repo,
    )
    tracked.write_text("line one\nline two\n")
    patch_text = _run(["git", "diff", "--", "tracked.txt"], cwd=repo).stdout
    _run(["git", "checkout", "--", "tracked.txt"], cwd=repo)
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    patch_path = patch_dir / "0001-test.patch"
    patch_path.write_text(patch_text)
    return repo, patch_path


def _write_manifest(manifest: Path, patch_path: Path, checksum: str | None = None) -> None:
    checksum = checksum or sha256(patch_path.read_bytes()).hexdigest()
    manifest.write_text(f"{checksum}  {patch_path.name}\n")


def _submodule_initialized() -> bool:
    return (_OPENCODE_DIR / ".git").exists() or (_OPENCODE_DIR / "package.json").exists()


@pytest.mark.skipif(
    not _submodule_initialized(),
    reason="opencode submodule not initialized; run `git submodule update --init`",
)
def test_apply_script_check_only_is_clean() -> None:
    """Every patch in .patches/ must apply in sequence, then revert clean."""
    assert _SCRIPT.exists(), f"missing apply script: {_SCRIPT}"
    assert _PATCH_DIR.exists(), f"missing patch dir: {_PATCH_DIR}"

    env = dict(os.environ)
    env["PERFXPERT_PATCH_CHECK_ONLY"] = "1"
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"script exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "CHECK OK" in result.stdout, result.stdout

    # Script must leave the submodule clean (check-only reverts).
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=_OPENCODE_DIR,
        capture_output=True,
        text=True,
    )
    assert git_status.returncode == 0
    assert git_status.stdout.strip() == "", (
        f"submodule is dirty after check-only apply:\n{git_status.stdout}"
    )


@pytest.mark.skipif(
    not _submodule_initialized(),
    reason="opencode submodule not initialized",
)
def test_every_patch_has_a_target() -> None:
    """Every patch must reference an existing file in the submodule."""
    for patch in sorted(_PATCH_DIR.glob("*.patch")):
        text = patch.read_text()
        # Extract 'diff --git a/<path>' targets.
        targets = []
        for line in text.splitlines():
            if line.startswith("diff --git a/"):
                # format: diff --git a/<path> b/<path>
                parts = line.split()
                a_path = parts[2][2:]  # strip "a/"
                targets.append(a_path)
        assert targets, f"{patch.name} contains no 'diff --git' header"
        for t in targets:
            assert (_OPENCODE_DIR / t).exists(), (
                f"{patch.name}: target {t!r} not found in submodule"
            )


def test_apply_script_check_only_succeeds_with_matching_manifest(tmp_path: Path) -> None:
    repo, patch_path = _init_fake_repo(tmp_path)
    manifest = patch_path.parent / "SHA256SUMS"
    _write_manifest(manifest, patch_path)

    env = dict(os.environ)
    env["PERFXPERT_PATCH_DIR"] = str(patch_path.parent)
    env["PERFXPERT_PATCH_MANIFEST"] = str(manifest)
    env["PERFXPERT_OPENCODE_DIR"] = str(repo)
    env["PERFXPERT_PATCH_CHECK_ONLY"] = "1"
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"script exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "manifest OK" in result.stdout


def test_apply_script_rejects_missing_manifest_entry(tmp_path: Path) -> None:
    repo, patch_path = _init_fake_repo(tmp_path)
    manifest = patch_path.parent / "SHA256SUMS"
    manifest.write_text("")

    env = dict(os.environ)
    env["PERFXPERT_PATCH_DIR"] = str(patch_path.parent)
    env["PERFXPERT_PATCH_MANIFEST"] = str(manifest)
    env["PERFXPERT_OPENCODE_DIR"] = str(repo)
    env["PERFXPERT_PATCH_CHECK_ONLY"] = "1"
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "patch manifest is empty" in result.stderr


def test_apply_script_rejects_checksum_mismatch(tmp_path: Path) -> None:
    repo, patch_path = _init_fake_repo(tmp_path)
    manifest = patch_path.parent / "SHA256SUMS"
    _write_manifest(manifest, patch_path, checksum="0" * 64)

    env = dict(os.environ)
    env["PERFXPERT_PATCH_DIR"] = str(patch_path.parent)
    env["PERFXPERT_PATCH_MANIFEST"] = str(manifest)
    env["PERFXPERT_OPENCODE_DIR"] = str(repo)
    env["PERFXPERT_PATCH_CHECK_ONLY"] = "1"
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "checksum verification failed" in result.stderr


def test_build_script_uses_locked_install_and_explicit_postinstall() -> None:
    text = _BUILD_SCRIPT.read_text()
    assert "bun install --frozen-lockfile --ignore-scripts" in text
    assert "bun run --cwd packages/opencode fix-node-pty" in text
