"""Tests for WorkflowSession 7-phase interactive profiling workflow.

Run with system rocpd first in PYTHONPATH:

    ROCPD_SYS=$(python3 -c "import site; print(site.getsitepackages()[-1])")
    ROCPD_SRC=<repo>/projects/rocprofiler-sdk/source/lib/python
    PYTHONPATH="${ROCPD_SYS}:${ROCPD_SRC}" pytest --noconftest test_workflow.py -v
"""

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

# If ROCPD_SYS is set, ensure the system-installed rocpd wins over any path that
# pytest may have prepended during package-discovery (e.g. the build tree).
_ROCPD_SYS = os.environ.get("ROCPD_SYS", "")
if _ROCPD_SYS:
    if not os.path.isdir(_ROCPD_SYS):
        pytest.skip(
            f"ROCPD_SYS={_ROCPD_SYS!r} does not exist; skipping workflow tests",
            allow_module_level=True,
        )
    sys.path.insert(0, _ROCPD_SYS)
    # Purge any partially-initialised rocpd loaded from the wrong tree.
    for _key in list(sys.modules):
        if _key == "rocpd" or _key.startswith("rocpd."):
            del sys.modules[_key]

from rocpd.ai_analysis.interactive import WorkflowState  # noqa: E402


def test_workflow_state_defaults():
    s = WorkflowState(app_command="./my_app --batch 64")
    assert s.app_command == "./my_app --batch 64"
    assert s.source_paths == []
    assert s.profiling_command == ""
    assert s.trace_history == []
    assert s.analysis_history == []
    assert s.edit_history == []
    assert s.iteration_count == 0


def test_checkpoint_record_defaults():
    from rocpd.ai_analysis.interactive import CheckpointRecord
    cp = CheckpointRecord(
        cp_id=0,
        commit_hash="abc1234",
        ref_name="refs/rocpd/session-1/cp-0",
        worktree_path="/tmp/cp-0",
        timestamp="2026-03-13T00:00:00Z",
        files_modified=["kernel.hip"],
        edit_summary="increased thread block size",
        file_snapshots={"kernel.hip": "__global__ void k() {}"},
    )
    assert cp.cp_id == 0
    assert cp.run_index is None
    assert cp.performance_delta_pct is None
    assert cp.blacklisted is False
    assert cp.blacklist_category == ""
    assert cp.blacklist_description == ""


def test_checkpoint_error_is_exception():
    from rocpd.ai_analysis.interactive import CheckpointError
    with pytest.raises(CheckpointError):
        raise CheckpointError("git failed")


def test_workflow_state_has_checkpoint_fields():
    from rocpd.ai_analysis.interactive import WorkflowState
    s = WorkflowState(app_command="./app")
    assert s.repo_root == ""
    assert s.baseline_commit == ""
    assert s.checkpoints == []
    assert s.active_checkpoint is None


def test_edit_record_has_checkpoint_id():
    from rocpd.ai_analysis.interactive import _EditRecord
    r = _EditRecord(
        timestamp="2026-03-13T00:00:00Z",
        file_path="/src/kernel.hip",
        backup_path="/src/kernel.hip.bak",
    )
    assert r.checkpoint_id is None


def _make_gcm(repo_root="/repo", session_id="sess1"):
    from rocpd.ai_analysis.interactive import GitCheckpointManager
    return GitCheckpointManager(
        repo_root=repo_root,
        session_id=session_id,
        sessions_dir="/home/user/.rocpd/sessions",
    )


def test_gcm_detect_repo_success():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="/repo\n")
        result = gcm.detect_repo("/repo/src")
    assert result == "/repo"


def test_gcm_detect_repo_not_git():
    from rocpd.ai_analysis.interactive import CheckpointError
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        with pytest.raises(CheckpointError):
            gcm.detect_repo("/not/a/repo")


def test_gcm_is_dirty_true():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=" M kernel.hip\n")
        assert gcm.is_dirty() is True


def test_gcm_is_dirty_false():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        assert gcm.is_dirty() is False


def test_gcm_get_head():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="abc1234\n")
        assert gcm.get_head() == "abc1234"


def test_gcm_create_checkpoint_commit():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="def5678\n")
        result = gcm.create_checkpoint_commit(["kernel.hip"], "cp-0 — test edit")
    assert result == "def5678"
    calls = mock_run.call_args_list
    assert any("add" in str(c) for c in calls)
    assert any("commit" in str(c) for c in calls)


def test_gcm_create_checkpoint_commit_passes_no_verify():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="abc\n")
        gcm.create_checkpoint_commit(["f.hip"], "msg")
    commit_call = [c for c in mock_run.call_args_list if "commit" in str(c)][0]
    assert "--no-verify" in str(commit_call)


def test_gcm_create_checkpoint_commit_passes_identity():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="abc\n")
        gcm.create_checkpoint_commit(["f.hip"], "msg")
    for c in mock_run.call_args_list:
        assert "rocpd@local" in str(c)


def test_gcm_tag_checkpoint():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        ref = gcm.tag_checkpoint(0, "abc1234")
    assert ref == "refs/rocpd/sess1/cp-0"
    assert "update-ref" in str(mock_run.call_args_list)


def test_gcm_tag_checkpoint_not_a_branch():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        ref = gcm.tag_checkpoint(0, "abc")
    assert "refs/heads" not in ref
    assert ref.startswith("refs/rocpd/")


def test_gcm_add_worktree():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        path = gcm.add_worktree(0, "abc1234")
    assert path == "/home/user/.rocpd/sessions/sess1/cp-0"
    assert "--detach" in str(mock_run.call_args)


def test_gcm_add_worktree_cleans_stale_path():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run, \
         patch("os.path.exists", return_value=True), \
         patch("shutil.rmtree") as mock_rmtree:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        gcm.add_worktree(0, "abc1234")
    mock_rmtree.assert_called_once()


def test_gcm_commit_reachable_true():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        assert gcm.commit_reachable("abc1234") is True


def test_gcm_commit_reachable_false():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        assert gcm.commit_reachable("abc1234") is False


def test_gcm_remove_worktree_silently_skips_missing():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run, \
         patch("os.path.exists", return_value=False):
        gcm.remove_worktree("/tmp/nonexistent")
    mock_run.assert_not_called()


def test_gcm_delete_ref():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        gcm.delete_ref("refs/rocpd/sess1/cp-0")
    assert "update-ref" in str(mock_run.call_args)
    assert "-d" in str(mock_run.call_args)


def test_gcm_files_in_commit():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="src/kernel.hip\nsrc/main.cpp\n")
        files = gcm.files_in_commit("abc1234")
    assert files == ["src/kernel.hip", "src/main.cpp"]


def test_gcm_list_worktrees():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="worktree /repo\nHEAD abc\n\nworktree /home/user/.rocpd/sessions/s/cp-0\nHEAD def\n"
        )
        paths = gcm.list_worktrees()
    assert "/repo" in paths
    assert "/home/user/.rocpd/sessions/s/cp-0" in paths


def test_gcm_restore_files_from_commit():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        # ls-tree returns the file; checkout succeeds
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="kernel.hip\n"),  # ls-tree
            MagicMock(returncode=0, stdout=""),               # checkout
        ]
        gcm.restore_files_from_commit("abc1234", ["kernel.hip"])
    # Both ls-tree and checkout were called
    assert mock_run.call_count == 2


def test_session_start_sets_repo_root_when_git_available():
    from rocpd.ai_analysis.interactive import WorkflowSession
    with patch("subprocess.run") as mock_run:
        # detect_repo: success; is_dirty: clean; get_head: hash
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="/repo\n"),   # rev-parse --show-toplevel
            MagicMock(returncode=0, stdout=""),           # status --porcelain (clean)
            MagicMock(returncode=0, stdout="abc1234\n"), # rev-parse HEAD
        ]
        ws = WorkflowSession(
            app_command="./app", source_paths=["/repo/src"]
        )
        ws._init_checkpoints()
    assert ws._state.repo_root == "/repo"
    assert ws._state.baseline_commit == "abc1234"


def test_session_start_disables_checkpoints_when_not_git():
    from rocpd.ai_analysis.interactive import WorkflowSession
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        ws = WorkflowSession(app_command="./app", source_paths=["/not/git"])
        ws._init_checkpoints()
    assert ws._state.repo_root == ""   # disabled


def test_session_start_aborts_when_dirty():
    from rocpd.ai_analysis.interactive import WorkflowSession
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="/repo\n"),     # detect_repo
            MagicMock(returncode=0, stdout=" M file.hip"), # is_dirty -> True
        ]
        ws = WorkflowSession(app_command="./app", source_paths=["/repo/src"])
        with pytest.raises(SystemExit):
            ws._init_checkpoints()


def test_init_checkpoints_disables_on_get_head_failure():
    from rocpd.ai_analysis.interactive import WorkflowSession, CheckpointError
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="/repo\n"),  # detect_repo
            MagicMock(returncode=0, stdout=""),          # is_dirty (clean)
            MagicMock(returncode=1, stdout=""),          # get_head (fails)
        ]
        ws = WorkflowSession(app_command="./app", source_paths=["/repo/src"])
        ws._init_checkpoints()
    assert ws._state.repo_root == ""
    assert ws._gcm is None
