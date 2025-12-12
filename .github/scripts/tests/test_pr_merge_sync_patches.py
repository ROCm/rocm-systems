"""
Unit tests for pr_merge_sync_patches: patch range derivation and patch generation.

- PR 3277: rebase-and-merge example (1 parent → range = parent..merge).
- PR 2514: squash-merge example (1 parent → range = parent..merge).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Merge SHAs from GitHub API (ROCm/rocm-systems)
PR_3277_MERGE_SHA = "c6d7e08fd4b06070e37b3e9aca500224fcc8404c"  # rebase-and-merge
PR_2514_MERGE_SHA = "bf6c504f4f3db50457a04340d9f49f2dfc6c94c8"  # squash-merge


class TestGetPatchRangeWithMocks:
    """Test _get_patch_range with mocked git (no repo required)."""

    def test_squash_merge_one_parent_returns_parent_and_merge(self):
        # One parent: squash or rebase-and-merge → range = parent..merge
        from pr_merge_sync_patches import _get_patch_range

        merge_sha = "cccc000000000000000000000000000000000000"
        parent_sha = "aaaa000000000000000000000000000000000000"
        rev_list_out = f"{merge_sha} {parent_sha}"

        with patch("pr_merge_sync_patches._run_git", return_value=rev_list_out):
            base, range_end = _get_patch_range(merge_sha)
        assert base == parent_sha
        assert range_end == merge_sha

    def test_merge_commit_two_parents_returns_first_and_second_parent(self):
        from pr_merge_sync_patches import _get_patch_range

        merge_sha = "cccc000000000000000000000000000000000000"
        first_parent = "aaaa000000000000000000000000000000000000"
        second_parent = "bbbb000000000000000000000000000000000000"
        rev_list_out = f"{merge_sha} {first_parent} {second_parent}"

        with patch("pr_merge_sync_patches._run_git", return_value=rev_list_out):
            base, range_end = _get_patch_range(merge_sha)
        assert base == first_parent
        assert range_end == second_parent


def _is_rocm_systems_repo_and_has_commits():
    """True if cwd looks like rocm-systems and the test merge SHAs exist."""
    try:
        root = Path(__file__).resolve().parents[2]  # .github/scripts/tests -> repo root
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return False
        for sha in (PR_3277_MERGE_SHA, PR_2514_MERGE_SHA):
            r2 = subprocess.run(
                ["git", "cat-file", "-t", sha],
                cwd=root,
                capture_output=True,
                text=True,
            )
            if r2.returncode != 0 or "commit" not in (r2.stdout or ""):
                return False
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _is_rocm_systems_repo_and_has_commits(),
    reason="Not in rocm-systems repo or merge SHAs (3277/2514) not available",
)
class TestGetPatchRangeRealRepo:
    """Integration tests: real git in rocm-systems repo with known PR merge SHAs."""

    def test_pr_3277_rebase_merge_range(self):
        """PR 3277 (rebase-and-merge): range = parent..merge, 1 commit in range for rocr-debug-agent."""
        from pr_merge_sync_patches import _get_patch_range

        base, range_end = _get_patch_range(PR_3277_MERGE_SHA)
        # Merge commit c6d7e08 has one parent b3e0321
        assert base.startswith("b3e0321")
        assert range_end.startswith("c6d7e08")

    def test_pr_2514_squash_merge_range(self):
        """PR 2514 (squash-merge): range = parent..merge, 1 commit in range."""
        from pr_merge_sync_patches import _get_patch_range

        base, range_end = _get_patch_range(PR_2514_MERGE_SHA)
        # Merge bf6c504 has one parent a5b467d
        assert base.startswith("a5b467d")
        assert range_end.startswith("bf6c504")


def _patch_test_dir():
    """Temp dir under repo for patch-generation tests (avoids pytest tmp_path permissions)."""
    root = Path(__file__).resolve().parents[2]
    return Path(tempfile.mkdtemp(prefix="patch_test_", dir=str(root)))


@pytest.mark.skipif(
    not _is_rocm_systems_repo_and_has_commits(),
    reason="Not in rocm-systems repo or merge SHAs not available",
)
class TestPatchGenerationRealRepo:
    """Integration tests: generate_patch for known PRs (no apply)."""

    def test_pr_3277_generates_patches_for_rocr_debug_agent(self):
        """PR 3277 touches projects/rocr-debug-agent: at least one patch, all touch subtree."""
        from pr_merge_sync_patches import _get_patch_range, generate_patch

        tmp = _patch_test_dir()
        try:
            base, range_end = _get_patch_range(PR_3277_MERGE_SHA)
            prefix = "projects/rocr-debug-agent/"
            anchor = tmp / "rocr-debug-agent.patch"
            patch_files = generate_patch(prefix, anchor, base, range_end, debug=False)
            assert patch_files is not None
            assert len(patch_files) >= 1
            for p in patch_files:
                assert p.exists()
                content = p.read_text(encoding="utf-8")
                assert "rocr-debug-agent" in content or "debug_agent" in content
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_pr_2514_generates_patches_for_clr(self):
        """PR 2514 (squash) touches projects/clr: exactly one patch."""
        from pr_merge_sync_patches import _get_patch_range, generate_patch

        tmp = _patch_test_dir()
        try:
            base, range_end = _get_patch_range(PR_2514_MERGE_SHA)
            prefix = "projects/clr/"
            anchor = tmp / "clr.patch"
            patch_files = generate_patch(prefix, anchor, base, range_end, debug=False)
            assert patch_files is not None
            assert len(patch_files) == 1
            assert patch_files[0].exists()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
