"""Tests for WorkflowSession 7-phase interactive profiling workflow.

Run with system rocpd first in PYTHONPATH:

    ROCPD_SYS=$(python3 -c "import site; print(site.getsitepackages()[-1])")
    ROCPD_SRC=<repo>/projects/rocprofiler-sdk/source/lib/python
    PYTHONPATH="${ROCPD_SYS}:${ROCPD_SRC}" pytest --noconftest test_workflow.py -v
"""

import os
import sys

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
