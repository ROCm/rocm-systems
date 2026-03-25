"""Tests for WorkflowSession 7-phase interactive profiling workflow.

Run with system rocpd first in PYTHONPATH:

    ROCPD_SYS=/opt/rocm-7.2.0/lib/python3.12/site-packages
    ROCPD_SRC=<repo>/source/lib/python
    PYTHONPATH="${ROCPD_SYS}:${ROCPD_SRC}" pytest --noconftest test_workflow.py -v
"""

import sys
import importlib

# Always insert the system-installed rocpd at sys.path[0] so it wins over any path
# that pytest may have prepended during package-discovery (e.g. the build tree).
_ROCPD_SYS = "/opt/rocm-7.2.0/lib/python3.12/site-packages"
sys.path.insert(0, _ROCPD_SYS)

# Purge any partially-initialised rocpd that pytest loaded from the wrong tree.
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
