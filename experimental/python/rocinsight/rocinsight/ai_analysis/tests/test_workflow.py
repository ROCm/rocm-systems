"""Tests for WorkflowSession 7-phase interactive profiling workflow.

Run with system rocinsight first in PYTHONPATH:

    ROCINSIGHT_SYS=$(python3 -c "import site; print(site.getsitepackages()[-1])")
    ROCINSIGHT_SRC=<repo>/projects/rocprofiler-sdk/source/lib/python
    PYTHONPATH="${ROCINSIGHT_SYS}:${ROCINSIGHT_SRC}" pytest --noconftest test_workflow.py -v
"""

import os
import pathlib
import sys
from unittest.mock import patch, MagicMock

import pytest

# If ROCINSIGHT_SYS is set, ensure the system-installed rocpd wins over any path that
# pytest may have prepended during package-discovery (e.g. the build tree).
_ROCINSIGHT_SYS = os.environ.get("ROCINSIGHT_SYS", "")
if _ROCINSIGHT_SYS:
    if not os.path.isdir(_ROCINSIGHT_SYS):
        pytest.skip(
            f"ROCINSIGHT_SYS={_ROCINSIGHT_SYS!r} does not exist; skipping workflow tests",
            allow_module_level=True,
        )
    sys.path.insert(0, _ROCINSIGHT_SYS)
    # Purge any partially-initialised rocinsight loaded from the wrong tree.
    for _key in list(sys.modules):
        if _key == "rocinsight" or _key.startswith("rocinsight."):
            del sys.modules[_key]

from rocinsight.ai_analysis.interactive import WorkflowState  # noqa: E402


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
    from rocinsight.ai_analysis.interactive import CheckpointRecord

    cp = CheckpointRecord(
        cp_id=0,
        commit_hash="abc1234",
        ref_name="refs/rocinsight/session-1/cp-0",
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
    from rocinsight.ai_analysis.interactive import CheckpointError

    with pytest.raises(CheckpointError):
        raise CheckpointError("git failed")


def test_workflow_state_has_checkpoint_fields():
    from rocinsight.ai_analysis.interactive import WorkflowState

    s = WorkflowState(app_command="./app")
    assert s.repo_root == ""
    assert s.baseline_commit == ""
    assert s.checkpoints == []
    assert s.active_checkpoint is None


def test_edit_record_has_checkpoint_id():
    from rocinsight.ai_analysis.interactive import _EditRecord

    r = _EditRecord(
        timestamp="2026-03-13T00:00:00Z",
        file_path="/src/kernel.hip",
        backup_path="/src/kernel.hip.bak",
    )
    assert r.checkpoint_id is None


def _make_gcm(repo_root="/repo", session_id="sess1"):
    from rocinsight.ai_analysis.interactive import GitCheckpointManager

    return GitCheckpointManager(
        repo_root=repo_root,
        session_id=session_id,
        sessions_dir="/home/user/.rocinsight/sessions",
    )


def test_gcm_detect_repo_success():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="/repo\n")
        result = gcm.detect_repo("/repo/src")
    assert result == "/repo"


def test_gcm_detect_repo_not_git():
    from rocinsight.ai_analysis.interactive import CheckpointError

    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        with pytest.raises(CheckpointError):
            gcm.detect_repo("/not/a/repo")


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
        assert "rocinsight@local" in str(c)


def test_gcm_tag_checkpoint():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        ref = gcm.tag_checkpoint(0, "abc1234")
    assert ref == "refs/rocinsight/sess1/cp-0"
    assert "update-ref" in str(mock_run.call_args_list)


def test_gcm_tag_checkpoint_not_a_branch():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        ref = gcm.tag_checkpoint(0, "abc")
    assert "refs/heads" not in ref
    assert ref.startswith("refs/rocinsight/")


def test_gcm_add_worktree():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        path = gcm.add_worktree(0, "abc1234")
    assert path == "/home/user/.rocinsight/sessions/sess1/cp-0"
    assert "--detach" in str(mock_run.call_args)


def test_gcm_add_worktree_cleans_stale_path():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run, patch(
        "os.path.exists", return_value=True
    ), patch("shutil.rmtree") as mock_rmtree:
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
    with patch("subprocess.run") as mock_run, patch("os.path.exists", return_value=False):
        gcm.remove_worktree("/tmp/nonexistent")
    mock_run.assert_not_called()


def test_gcm_delete_ref():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        gcm.delete_ref("refs/rocinsight/sess1/cp-0")
    assert "update-ref" in str(mock_run.call_args)
    assert "-d" in str(mock_run.call_args)


def test_gcm_files_in_commit():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="src/kernel.hip\nsrc/main.cpp\n"
        )
        files = gcm.files_in_commit("abc1234")
    assert files == ["src/kernel.hip", "src/main.cpp"]


def test_gcm_list_worktrees():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="worktree /repo\nHEAD abc\n\nworktree /home/user/.rocinsight/sessions/s/cp-0\nHEAD def\n",
        )
        paths = gcm.list_worktrees()
    assert "/repo" in paths
    assert "/home/user/.rocinsight/sessions/s/cp-0" in paths


def test_gcm_restore_files_from_commit():
    gcm = _make_gcm()
    with patch("subprocess.run") as mock_run:
        # ls-tree returns the file; checkout succeeds
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="kernel.hip\n"),  # ls-tree
            MagicMock(returncode=0, stdout=""),  # checkout
        ]
        gcm.restore_files_from_commit("abc1234", ["kernel.hip"])
    # Both ls-tree and checkout were called
    assert mock_run.call_count == 2


def test_session_start_sets_repo_root_when_git_available():
    from rocinsight.ai_analysis.interactive import WorkflowSession

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="/repo\n"),  # rev-parse --show-toplevel
            MagicMock(returncode=0, stdout="abc1234\n"),  # rev-parse HEAD
        ]
        ws = WorkflowSession(app_command="./app", source_paths=["/repo/src"])
        ws._init_checkpoints()
    assert ws._state.repo_root == "/repo"
    assert ws._state.baseline_commit == "abc1234"


def test_session_start_disables_checkpoints_when_not_git():
    from rocinsight.ai_analysis.interactive import WorkflowSession

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=128, stdout="")
        ws = WorkflowSession(app_command="./app", source_paths=["/not/git"])
        ws._init_checkpoints()
    assert ws._state.repo_root == ""  # disabled


def test_checkpoints_work_with_dirty_tree():
    from rocinsight.ai_analysis.interactive import WorkflowSession

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="/repo\n"),  # detect_repo
            MagicMock(returncode=0, stdout="abc123\n"),  # get_head
        ]
        ws = WorkflowSession(app_command="./app", source_paths=["/repo/src"])
        ws._init_checkpoints()
        # Dirty working tree does not affect checkpoints — _gcm should be set
        assert ws._gcm is not None
        assert ws._state.baseline_commit == "abc123"


def test_init_checkpoints_disables_on_get_head_failure():
    from rocinsight.ai_analysis.interactive import WorkflowSession, CheckpointError

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="/repo\n"),  # detect_repo
            MagicMock(returncode=1, stdout=""),  # get_head (fails)
        ]
        ws = WorkflowSession(app_command="./app", source_paths=["/repo/src"])
        ws._init_checkpoints()
    assert ws._state.repo_root == ""
    assert ws._gcm is None


def _make_workflow_session_with_gcm(mock_gcm=None):
    """Helper: WorkflowSession with mocked git and source paths."""
    from rocinsight.ai_analysis.interactive import WorkflowSession, GitCheckpointManager

    ws = WorkflowSession(app_command="./app", source_paths=["/repo/src"])
    ws._state.repo_root = "/repo"
    ws._state.baseline_commit = "baseline123"
    ws._gcm = mock_gcm or MagicMock(spec=GitCheckpointManager)
    return ws


def test_create_checkpoint_appends_checkpoint_record():
    from rocinsight.ai_analysis.interactive import CheckpointRecord

    ws = _make_workflow_session_with_gcm()
    ws._gcm.create_checkpoint_commit.return_value = "abc1234"
    ws._gcm.tag_checkpoint.return_value = "refs/rocinsight/sess/cp-0"
    ws._gcm.add_worktree.return_value = "/tmp/cp-0"

    with patch.object(ws, "_save_session"):
        ws._create_checkpoint(
            files_modified=["kernel.hip"],
            edit_summary="increased block size",
            file_snapshots={"kernel.hip": "content"},
        )

    assert len(ws._state.checkpoints) == 1
    cp = ws._state.checkpoints[0]
    assert cp.cp_id == 0
    assert cp.commit_hash == "abc1234"
    assert cp.files_modified == ["kernel.hip"]
    assert cp.file_snapshots == {"kernel.hip": "content"}


def test_create_checkpoint_links_edit_record():
    ws = _make_workflow_session_with_gcm()
    ws._gcm.create_checkpoint_commit.return_value = "abc"
    ws._gcm.tag_checkpoint.return_value = "refs/rocinsight/s/cp-0"
    ws._gcm.add_worktree.return_value = "/tmp/cp-0"
    from rocinsight.ai_analysis.interactive import _EditRecord

    ws._state.edit_history.append(
        _EditRecord(timestamp="t", file_path="/f", backup_path="/f.bak")
    )

    with patch.object(ws, "_save_session"):
        ws._create_checkpoint(["f"], "edit", {"f": "c"})

    assert ws._state.edit_history[-1].checkpoint_id == 0


def test_create_checkpoint_skipped_when_no_gcm():
    ws = _make_workflow_session_with_gcm()
    ws._gcm = None  # checkpoints disabled

    with patch.object(ws, "_save_session"):
        ws._create_checkpoint(["f"], "edit", {"f": "c"})

    assert ws._state.checkpoints == []


def test_create_checkpoint_skipped_on_git_error():
    from rocinsight.ai_analysis.interactive import CheckpointError

    ws = _make_workflow_session_with_gcm()
    ws._gcm.create_checkpoint_commit.side_effect = CheckpointError("git fail")

    with patch.object(ws, "_save_session"):
        ws._create_checkpoint(["f"], "edit", {"f": "c"})  # should not raise

    assert ws._state.checkpoints == []


def test_update_checkpoint_records_run_index():
    from rocinsight.ai_analysis.interactive import (
        CheckpointRecord,
        _AnalysisSnapshot,
        _TraceRun,
    )

    ws = _make_workflow_session_with_gcm()

    cp = CheckpointRecord(
        cp_id=0,
        commit_hash="abc",
        ref_name="refs/r",
        worktree_path="/wt",
        timestamp="t",
        files_modified=[],
        edit_summary="e",
        file_snapshots={},
    )
    ws._state.checkpoints.append(cp)

    ws._state.trace_history.append(
        _TraceRun(timestamp="t", command="cmd", db_path="/db.db")
    )
    ws._state.analysis_history.append(
        _AnalysisSnapshot(
            timestamp="t",
            iteration=0,
            execution_breakdown={"total_runtime_ns": 1_000_000},
        )
    )

    with patch.object(ws, "_save_session"):
        ws._update_checkpoint_with_run()

    assert ws._state.checkpoints[0].run_index == 0
    assert ws._state.checkpoints[0].performance_delta_pct is None  # only 1 analysis


def test_update_checkpoint_computes_delta_from_total_runtime_ns():
    from rocinsight.ai_analysis.interactive import (
        CheckpointRecord,
        _AnalysisSnapshot,
        _TraceRun,
    )

    ws = _make_workflow_session_with_gcm()

    for i in range(2):
        cp = CheckpointRecord(
            cp_id=i,
            commit_hash="h",
            ref_name="r",
            worktree_path="w",
            timestamp="t",
            files_modified=[],
            edit_summary="e",
            file_snapshots={},
        )
        ws._state.checkpoints.append(cp)

    ws._state.checkpoints[0].run_index = 0
    ws._state.trace_history.append(
        _TraceRun(timestamp="t", command="c", db_path="/db0.db")
    )
    ws._state.trace_history.append(
        _TraceRun(timestamp="t", command="c", db_path="/db1.db")
    )
    ws._state.checkpoints[1].run_index = 1  # set by Phase 3 already
    ws._state.analysis_history.append(
        _AnalysisSnapshot(
            timestamp="t",
            iteration=0,
            execution_breakdown={"total_runtime_ns": 1_000_000},
        )
    )
    ws._state.analysis_history.append(
        _AnalysisSnapshot(
            timestamp="t",
            iteration=1,
            execution_breakdown={"total_runtime_ns": 900_000},  # 10% faster
        )
    )

    # Delta computed by Phase 4 method (after analysis_history updated)
    ws._update_checkpoint_delta()

    import pytest as _pytest

    delta = ws._state.checkpoints[1].performance_delta_pct
    assert delta == _pytest.approx(10.0, abs=0.1)  # (1M-900K)/1M * 100


def test_update_checkpoint_delta_noop_when_insufficient_history():
    from rocinsight.ai_analysis.interactive import CheckpointRecord

    ws = _make_workflow_session_with_gcm()
    cp = CheckpointRecord(
        cp_id=0,
        commit_hash="h",
        ref_name="r",
        worktree_path="w",
        timestamp="t",
        files_modified=[],
        edit_summary="e",
        file_snapshots={},
        run_index=0,
    )
    ws._state.checkpoints.append(cp)
    ws._update_checkpoint_delta()  # only 0 analyses, should not raise
    assert ws._state.checkpoints[0].performance_delta_pct is None


def test_update_checkpoint_noop_when_no_checkpoints():
    ws = _make_workflow_session_with_gcm()
    with patch.object(ws, "_save_session"):
        ws._update_checkpoint_with_run()  # should not raise


def _make_ws_with_checkpoints():
    """Helper: WorkflowSession with 3 checkpoints and 3 runs."""
    from rocinsight.ai_analysis.interactive import (
        WorkflowSession,
        CheckpointRecord,
        _TraceRun,
        _AnalysisSnapshot,
    )

    ws = _make_workflow_session_with_gcm()
    ws._state.baseline_commit = "base000"

    deltas = [10.0, -67.0, -15.0]
    for i, delta in enumerate(deltas):
        cp = CheckpointRecord(
            cp_id=i,
            commit_hash=f"hash{i}",
            ref_name=f"refs/rocinsight/s/cp-{i}",
            worktree_path=f"/wt/cp-{i}",
            timestamp="t",
            files_modified=["kernel.hip"],
            edit_summary=f"edit {i}",
            file_snapshots={"kernel.hip": f"content{i}"},
            run_index=i,
            performance_delta_pct=delta,
        )
        ws._state.checkpoints.append(cp)
        ws._state.trace_history.append(
            _TraceRun(timestamp="t", command="c", db_path=f"/db{i}.db")
        )
        ws._state.analysis_history.append(
            _AnalysisSnapshot(
                timestamp="t",
                iteration=i,
                execution_breakdown={"total_runtime_ns": 1_000_000 - i * 100_000},
                recommendations=[],
            )
        )
    ws._state.iteration_count = 3
    return ws


def test_rollback_restores_files_from_git():
    ws = _make_ws_with_checkpoints()
    with patch.object(ws, "_save_session"), patch.object(
        ws._gcm, "commit_reachable", return_value=True
    ), patch.object(ws._gcm, "restore_files_from_commit") as mock_restore, patch.object(
        ws._gcm, "remove_worktree"
    ):
        ws._rollback_to_checkpoint(target_cp_id=0)
    mock_restore.assert_called_once_with("hash0", ["kernel.hip"])


def test_rollback_uses_file_snapshots_when_commit_unreachable():
    ws = _make_ws_with_checkpoints()
    with patch.object(ws, "_save_session"), patch.object(
        ws._gcm, "commit_reachable", return_value=False
    ), patch.object(ws._gcm, "remove_worktree"), patch(
        "pathlib.Path.write_text"
    ) as mock_write, patch(
        "pathlib.Path.mkdir"
    ):
        ws._rollback_to_checkpoint(target_cp_id=0)
    mock_write.assert_called()


def test_rollback_truncates_checkpoints_after_target():
    ws = _make_ws_with_checkpoints()
    with patch.object(ws, "_save_session"), patch.object(
        ws._gcm, "commit_reachable", return_value=True
    ), patch.object(ws._gcm, "restore_files_from_commit"), patch.object(
        ws._gcm, "remove_worktree"
    ):
        ws._rollback_to_checkpoint(target_cp_id=0)
    assert len(ws._state.checkpoints) == 1
    assert ws._state.checkpoints[0].cp_id == 0


def test_rollback_truncates_trace_and_analysis_history():
    ws = _make_ws_with_checkpoints()
    with patch.object(ws, "_save_session"), patch.object(
        ws._gcm, "commit_reachable", return_value=True
    ), patch.object(ws._gcm, "restore_files_from_commit"), patch.object(
        ws._gcm, "remove_worktree"
    ):
        ws._rollback_to_checkpoint(target_cp_id=0)
    assert len(ws._state.trace_history) == 1
    assert len(ws._state.analysis_history) == 1
    assert ws._state.iteration_count == 1


def test_rollback_sets_active_checkpoint():
    ws = _make_ws_with_checkpoints()
    with patch.object(ws, "_save_session"), patch.object(
        ws._gcm, "commit_reachable", return_value=True
    ), patch.object(ws._gcm, "restore_files_from_commit"), patch.object(
        ws._gcm, "remove_worktree"
    ):
        ws._rollback_to_checkpoint(target_cp_id=0)
    assert ws._state.active_checkpoint == 0


def test_blacklist_sets_fields():
    ws = _make_ws_with_checkpoints()
    ws._blacklist_checkpoint(1)
    cp = ws._state.checkpoints[1]
    assert cp.blacklisted is True
    assert cp.blacklist_category == "edit 1"
    assert "-67" in cp.blacklist_description


def test_build_blacklist_block_empty_when_none():
    ws = _make_ws_with_checkpoints()
    assert ws._build_blacklist_block() == ""


def test_build_blacklist_block_contains_description():
    ws = _make_ws_with_checkpoints()
    ws._blacklist_checkpoint(1)
    block = ws._build_blacklist_block()
    assert "Blacklisted approaches" in block
    assert "edit 1" in block


def test_build_blacklist_block_deduplicates():
    ws = _make_ws_with_checkpoints()
    ws._blacklist_checkpoint(1)
    ws._blacklist_checkpoint(1)  # blacklist same cp twice
    block = ws._build_blacklist_block()
    assert block.count("edit 1") == 1


def test_rollback_baseline_no_git_still_clears_state():
    ws = _make_ws_with_checkpoints()
    ws._gcm = None  # no git available
    ws._state.repo_root = ""
    with patch.object(ws, "_save_session"):
        ws._rollback_to_checkpoint(target_cp_id=-1)
    # State should be cleared even without git restore
    assert ws._state.checkpoints == []
    assert ws._state.trace_history == []
    assert ws._state.analysis_history == []
    assert ws._state.iteration_count == 0


def test_phase5_shows_rollback_option_when_checkpoints_exist():
    ws = _make_ws_with_checkpoints()
    from rocinsight.ai_analysis.interactive import _AnalysisSnapshot

    snap = _AnalysisSnapshot(
        timestamp="t",
        iteration=2,
        recommendations=[
            {
                "priority": "HIGH",
                "category": "C",
                "issue": "i",
                "suggestion": "s",
                "actions": [],
                "id": "R1",
                "estimated_impact": "",
                "commands": [],
            }
        ],
    )
    # Simulate user typing "b" then "0" then "n" (no blacklist)
    with patch("builtins.input", side_effect=["b", "0", "n"]), patch.object(
        ws, "_rollback_to_checkpoint"
    ) as mock_rollback, patch.object(ws, "_save_session"):
        ws._phase5_rec_menu(snap)
    mock_rollback.assert_called_once_with(target_cp_id=0)


def test_phase5_does_not_crash_when_no_checkpoints():
    from rocinsight.ai_analysis.interactive import WorkflowSession, _AnalysisSnapshot

    ws = WorkflowSession(app_command="./app")  # no checkpoints
    snap = _AnalysisSnapshot(
        timestamp="t",
        iteration=0,
        recommendations=[
            {
                "priority": "INFO",
                "category": "C",
                "issue": "i",
                "suggestion": "s",
                "actions": [],
                "id": "R1",
                "estimated_impact": "",
                "commands": [],
            }
        ],
    )
    with patch("builtins.input", side_effect=["n"]):
        result = ws._phase5_rec_menu(snap)
    assert result is None


def test_phase5_rollback_with_blacklist():
    ws = _make_ws_with_checkpoints()
    from rocinsight.ai_analysis.interactive import _AnalysisSnapshot

    snap = _AnalysisSnapshot(
        timestamp="t",
        iteration=2,
        recommendations=[
            {
                "priority": "HIGH",
                "category": "C",
                "issue": "i",
                "suggestion": "s",
                "actions": [],
                "id": "R1",
                "estimated_impact": "",
                "commands": [],
            }
        ],
    )
    with patch("builtins.input", side_effect=["b", "0", "1"]), patch.object(
        ws, "_rollback_to_checkpoint"
    ), patch.object(ws, "_blacklist_checkpoint") as mock_blacklist, patch.object(
        ws, "_save_session"
    ):
        ws._phase5_rec_menu(snap)
    mock_blacklist.assert_called_once_with(1)


def test_blacklist_block_injected_into_phase6_prompt():
    import pathlib

    ws = _make_ws_with_checkpoints()
    ws._blacklist_checkpoint(1)  # blacklist cp-1 (edit 1, -67%)

    from rocinsight.ai_analysis.interactive import _AnalysisSnapshot

    snap = _AnalysisSnapshot(
        timestamp="t",
        iteration=2,
        recommendations=[
            {
                "priority": "HIGH",
                "category": "C",
                "issue": "i",
                "suggestion": "s",
                "actions": [],
                "id": "R1",
                "estimated_impact": "",
                "commands": [],
            }
        ],
    )

    captured_suggestions = []

    def fake_llm_rewrite(file, suggestions):
        captured_suggestions.append(suggestions)
        return "rewritten content"

    with patch.object(
        ws, "_llm_rewrite_file", side_effect=fake_llm_rewrite
    ), patch.object(
        ws, "_pick_file_from_source_paths", return_value=pathlib.Path("/repo/kernel.hip")
    ), patch(
        "pathlib.Path.read_text", return_value="original"
    ), patch(
        "pathlib.Path.write_text"
    ), patch(
        "pathlib.Path.with_suffix", return_value=pathlib.Path("/repo/kernel.hip.bak")
    ), patch(
        "pathlib.Path.exists", return_value=False
    ), patch(
        "builtins.input", side_effect=["y", "done"]
    ), patch.object(
        ws, "_save_session"
    ), patch.object(
        ws, "_create_checkpoint"
    ):
        ws._phase6_apply_direct(snap)

    assert any("Blacklisted" in s for s in captured_suggestions)


def test_worktrees_removed_on_session_exit():
    ws = _make_ws_with_checkpoints()

    with patch.object(ws._gcm, "remove_worktree") as mock_remove, patch.object(
        ws, "_save_session"
    ), patch.object(ws, "_init_checkpoints"):
        ws._teardown_checkpoints()

    assert mock_remove.call_count == 3  # 3 checkpoints


def test_stale_worktrees_pruned_on_start():
    ws = _make_ws_with_checkpoints()
    sessions_dir = str(ws._sessions_dir)

    with patch.object(
        ws._gcm,
        "list_worktrees",
        return_value=[
            f"{sessions_dir}/other_session/cp-0",  # no JSON → stale
            f"{sessions_dir}/{ws._session_id}/cp-0",  # current session → keep
            "/repo/.git",  # not under sessions_dir → ignore
        ],
    ), patch.object(ws._gcm, "remove_worktree") as mock_remove, patch(
        "pathlib.Path.exists", return_value=False
    ):  # no JSON files exist
        ws._prune_stale_worktrees()

    # Only the other_session worktree should be pruned
    pruned_paths = [str(c.args[0]) for c in mock_remove.call_args_list]
    assert any("other_session" in p for p in pruned_paths)
    assert not any(ws._session_id in p for p in pruned_paths)


# ── Rendering helpers ──────────────────────────────────────────────────────────


def test_priority_badge_renders_visible_brackets():
    """_priority_badge must escape inner [ so Rich doesn't strip the label."""
    from rocinsight.ai_analysis.interactive import _priority_badge

    for pri in ("HIGH", "MEDIUM", "LOW", "INFO"):
        badge = _priority_badge(pri)
        # The escaped bracket pattern \[PRI] must be present so Rich renders [PRI].
        assert f"\\[{pri}]" in badge, f"badge for {pri!r} missing escaped brackets: {badge!r}"


def test_priority_badge_contains_style():
    from rocinsight.ai_analysis.interactive import _priority_badge

    assert "bold red" in _priority_badge("HIGH")
    assert "bold yellow" in _priority_badge("MEDIUM")
    assert "bold green" in _priority_badge("LOW")
    assert "bold blue" in _priority_badge("INFO")


# ── Phase 5: save choice ───────────────────────────────────────────────────────


def _make_high_snap():
    from rocinsight.ai_analysis.interactive import _AnalysisSnapshot

    return _AnalysisSnapshot(
        timestamp="t",
        iteration=0,
        recommendations=[
            {
                "priority": "HIGH",
                "category": "C",
                "issue": "Kernel slow",
                "suggestion": "s",
                "actions": [],
                "id": "R1",
                "estimated_impact": "",
                "commands": [],
            }
        ],
    )


def test_phase5_save_calls_save_session_and_continues():
    """[s] in Phase 5 must call _save_session() and not exit the loop."""
    from rocinsight.ai_analysis.interactive import WorkflowSession

    ws = WorkflowSession(app_command="./app")
    snap = _make_high_snap()

    with patch.object(ws, "_save_session") as mock_save, patch(
        "builtins.input", side_effect=["s", "n"]
    ):
        result = ws._phase5_rec_menu(snap)

    mock_save.assert_called_once()
    assert result is None  # [n] after save exits normally


def test_phase5_save_does_not_use_conv_attribute():
    """WorkflowSession._phase5_rec_menu [s] must not reference self._conv."""
    from rocinsight.ai_analysis.interactive import WorkflowSession

    ws = WorkflowSession(app_command="./app")
    snap = _make_high_snap()

    assert not hasattr(ws, "_conv"), "WorkflowSession must not have _conv"

    with patch.object(ws, "_save_session"), patch("builtins.input", side_effect=["s", "n"]):
        # Must not raise AttributeError
        ws._phase5_rec_menu(snap)


# ── Phase 7: save choice ───────────────────────────────────────────────────────


def test_phase7_save_calls_save_session_and_reprompts():
    """[s] in Phase 7 must save and re-show the prompt (recursion)."""
    from rocinsight.ai_analysis.interactive import WorkflowSession

    ws = WorkflowSession(app_command="./app")
    ws._state.profiling_command = "rocprofv3 --sys-trace -- ./app"

    with patch.object(ws, "_save_session") as mock_save, patch(
        "builtins.input", side_effect=["s", "n"]
    ):
        result = ws._phase7_reprofiling_prompt()

    mock_save.assert_called_once()
    assert result is None  # [n] after save → stop


def test_phase7_save_does_not_use_conv_attribute():
    """WorkflowSession._phase7_reprofiling_prompt [s] must not reference self._conv."""
    from rocinsight.ai_analysis.interactive import WorkflowSession

    ws = WorkflowSession(app_command="./app")
    ws._state.profiling_command = "rocprofv3 --sys-trace -- ./app"

    assert not hasattr(ws, "_conv"), "WorkflowSession must not have _conv"

    with patch.object(ws, "_save_session"), patch("builtins.input", side_effect=["s", "n"]):
        ws._phase7_reprofiling_prompt()  # must not raise


# ── Resume session ─────────────────────────────────────────────────────────────

import json as _json
import dataclasses as _dc
from datetime import datetime, timezone


def _write_workflow_session(tmp_path, state_overrides: dict) -> pathlib.Path:
    """Write a minimal workflow_*.json session file and return its path."""
    from rocinsight.ai_analysis.interactive import (
        WorkflowState,
        _TraceRun,
        _AnalysisSnapshot,
    )

    trace_run = _TraceRun(
        timestamp="2026-03-26T03:25:31Z",
        command="rocprofv3 --sys-trace -d /tmp/run -- ./app",
        db_path="/tmp/run/results.db",
    )
    snap = _AnalysisSnapshot(
        timestamp="2026-03-26T03:25:40Z",
        iteration=1,
        recommendations=[
            {
                "priority": "HIGH",
                "category": "C",
                "issue": "slow kernel",
                "suggestion": "s",
                "actions": [],
                "id": "R1",
                "estimated_impact": "",
                "commands": [],
            }
        ],
        execution_breakdown={"kernel_time_pct": 80.0},
    )
    state = WorkflowState(
        app_command="./app",
        source_paths=[],
        profiling_command="rocprofv3 --sys-trace -d /tmp/run -- ./app",
        trace_history=[trace_run],
        analysis_history=[snap],
        iteration_count=1,
    )
    raw_state = _dc.asdict(state)
    raw_state.update(state_overrides)

    session_id = "workflow_2026-03-26_03-25-31___app"
    payload = {
        "session_id": session_id,
        "type": "workflow",
        "app_command": "./app",
        "state": raw_state,
    }
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    p = sessions_dir / f"{session_id}.json"
    p.write_text(_json.dumps(payload, indent=2))
    return p


def test_load_session_returns_state_and_id(tmp_path):
    from rocinsight.ai_analysis.interactive import WorkflowSession

    p = _write_workflow_session(tmp_path, {})
    ws = WorkflowSession(app_command="./app")
    ws._sessions_dir = tmp_path / "sessions"
    result = ws._load_session(str(p))
    assert result is not None
    state, sid = result
    assert sid == "workflow_2026-03-26_03-25-31___app"
    assert len(state.trace_history) == 1
    assert state.trace_history[0].db_path == "/tmp/run/results.db"
    assert state.iteration_count == 1


def test_load_session_nonexistent_returns_none(tmp_path):
    from rocinsight.ai_analysis.interactive import WorkflowSession

    ws = WorkflowSession(app_command="./app")
    ws._sessions_dir = tmp_path / "sessions"
    assert ws._load_session("/nonexistent/path.json") is None


def test_load_session_wrong_type_returns_none(tmp_path):
    """A non-workflow session file must be rejected."""
    from rocinsight.ai_analysis.interactive import WorkflowSession

    p = tmp_path / "sessions" / "interactive_test.json"
    p.parent.mkdir(parents=True)
    p.write_text(_json.dumps({"type": "interactive", "session_id": "x", "state": {}}))
    ws = WorkflowSession(app_command="./app")
    ws._sessions_dir = tmp_path / "sessions"
    assert ws._load_session(str(p)) is None


def test_resume_session_sets_resumed_flag(tmp_path):
    """When --resume-session is given with a valid file, _resumed must be True."""
    from rocinsight.ai_analysis.interactive import WorkflowSession

    p = _write_workflow_session(tmp_path, {})
    # Patch sessions_dir before WorkflowSession resolves paths
    with patch.object(
        pathlib.Path,
        "home",
        return_value=tmp_path,
    ):
        ws = WorkflowSession(app_command="./app", resume_session=str(p))
    assert ws._resumed is True
    assert len(ws._state.trace_history) == 1


def test_resume_session_skips_phase1_2(tmp_path):
    """run() on a resumed session must not call _phase1b or _phase2."""
    from rocinsight.ai_analysis.interactive import WorkflowSession

    p = _write_workflow_session(tmp_path, {})
    with patch.object(pathlib.Path, "home", return_value=tmp_path):
        ws = WorkflowSession(app_command="./app", resume_session=str(p))

    assert ws._resumed is True

    snap_stub = _make_high_snap()
    with (
        patch.object(ws, "_init_checkpoints"),
        patch.object(ws, "_prune_stale_worktrees"),
        patch.object(ws, "_phase1b_quick_workload_analysis") as mock_p1b,
        patch.object(ws, "_phase2_show_command") as mock_p2,
        patch.object(ws, "_phase4_analyze", return_value=snap_stub),
        patch.object(ws, "_phase5_rec_menu", return_value=None),
        patch.object(ws, "_phase7_reprofiling_prompt", return_value=None),
        patch.object(ws, "_teardown_checkpoints"),
        patch.object(ws, "_save_session"),
        patch.object(ws, "print_session_summary"),
    ):
        ws.run()

    mock_p1b.assert_not_called()
    mock_p2.assert_not_called()


def test_resume_session_calls_phase4_on_last_db(tmp_path):
    """run() resumed must call _phase4_analyze with the last trace DB path."""
    from rocinsight.ai_analysis.interactive import WorkflowSession

    p = _write_workflow_session(tmp_path, {})
    with patch.object(pathlib.Path, "home", return_value=tmp_path):
        ws = WorkflowSession(app_command="./app", resume_session=str(p))

    snap_stub = _make_high_snap()
    with (
        patch.object(ws, "_init_checkpoints"),
        patch.object(ws, "_prune_stale_worktrees"),
        patch.object(ws, "_phase4_analyze", return_value=snap_stub) as mock_p4,
        patch.object(ws, "_phase5_rec_menu", return_value=None),
        patch.object(ws, "_phase7_reprofiling_prompt", return_value=None),
        patch.object(ws, "_teardown_checkpoints"),
        patch.object(ws, "_save_session"),
        patch.object(ws, "print_session_summary"),
    ):
        ws.run()

    mock_p4.assert_called_once_with("/tmp/run/results.db")


def test_load_session_latest(tmp_path):
    """'latest' keyword picks the most-recently-modified workflow file."""
    import time
    from rocinsight.ai_analysis.interactive import WorkflowSession

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write two session files; the second one is newer
    p_old = _write_workflow_session(tmp_path, {})
    time.sleep(0.05)
    # Write a second session file
    payload2 = _json.loads(p_old.read_text())
    payload2["session_id"] = "workflow_2026-03-26_04-00-00___app"
    p_new = sessions_dir / "workflow_2026-03-26_04-00-00___app.json"
    p_new.write_text(_json.dumps(payload2, indent=2))

    ws = WorkflowSession(app_command="./app")
    ws._sessions_dir = sessions_dir
    result = ws._load_session("latest")
    assert result is not None
    _, sid = result
    assert sid == "workflow_2026-03-26_04-00-00___app"
