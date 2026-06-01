"""Attack #10 — trace metadata references `../etc/passwd`; patch.apply MUST
reject (project-root confinement).
"""

from pathlib import Path
import pytest

from perfxpert.tools.patch_mgr import apply as patch_apply
from perfxpert.tools._safety import PathConfinementError
from tests.test_red_team.conftest import record_outcome


@pytest.mark.red_team
def test_patch_apply_rejects_path_traversal(outcomes_dir, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("secret")

    with pytest.raises(PathConfinementError):
        patch_apply(
            project_root=str(project_root),
            rel_path="../../victim.txt",
            new_content="pwned",
        )
    # Ensure victim.txt is UNTOUCHED
    assert victim.read_text() == "secret"

    record_outcome(
        outcomes_dir,
        attack_id="path_traversal_metadata",
        status="defeated",
        details={"target_outside_project": str(victim)},
    )
