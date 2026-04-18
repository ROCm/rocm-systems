"""Integration test for scripts/apply-opencode-patches.sh.

Runs the script in CHECK_ONLY mode against the real submodule to guarantee
every .patches/*.patch applies in sequence without leaving the submodule
dirty. Skipped when the submodule is not initialized.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


_PERFXPERT_ROOT = Path(__file__).resolve().parents[2]
_OPENCODE_DIR = _PERFXPERT_ROOT / "opencode"
_SCRIPT = _PERFXPERT_ROOT / "scripts" / "apply-opencode-patches.sh"
_PATCH_DIR = _PERFXPERT_ROOT / ".patches"


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
