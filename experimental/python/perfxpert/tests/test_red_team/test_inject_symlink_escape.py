"""Attack #11 — symlink in project tree pointing outside project root.

patch.apply must resolve symlinks BEFORE editing; target must be within
project_root after canonicalization. Otherwise reject.
"""

import os
from pathlib import Path

import pytest

from perfxpert.tools.patch_mgr import apply as patch_apply
from perfxpert.tools._safety import PathConfinementError
from tests.test_red_team.conftest import record_outcome


@pytest.mark.red_team
def test_patch_apply_rejects_symlink_escape(outcomes_dir, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("keep secret")

    # Create a symlink INSIDE project_root pointing OUTSIDE
    link = project_root / "looks_local.cpp"
    os.symlink(outside, link)

    with pytest.raises(PathConfinementError):
        patch_apply(
            project_root=str(project_root),
            rel_path="looks_local.cpp",
            new_content="OWNED",
        )

    # File outside must be untouched
    assert outside.read_text() == "keep secret"

    record_outcome(
        outcomes_dir,
        attack_id="symlink_escape",
        status="defeated",
        details={"symlink": str(link), "target_outside": str(outside)},
    )
