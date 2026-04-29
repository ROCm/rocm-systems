"""anchors — anchors.check EXECUTION tool.

Invokes a user-provided test command against a new binary.
No flag allowlist here — the test command itself is the contract;
shell-metachar denial protects against injection via kernel metadata
that would compose a malicious test command.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools._safety import build_safe_env, reject_shell_metachars


_ANCHOR_TIMEOUT_SEC = 300


@tool_class(ToolClass.EXECUTION)
def check(
    project_root: Path,
    test_command: List[str],
    *,
    timeout: int = _ANCHOR_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Run `test_command` under `project_root`; report pass/fail.

    Returns:
        {"all_passed": bool, "returncode": int, "stdout": str, "stderr": str}
    """
    if not test_command:
        raise ValueError("test_command must not be empty")
    for tok in test_command:
        reject_shell_metachars(tok)

    proc = subprocess.run(
        test_command,
        shell=False,
        capture_output=True,
        cwd=str(project_root),
        env=build_safe_env(),
        timeout=timeout,
        check=False,
    )
    return {
        "all_passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.decode("utf-8", errors="replace"),
        "stderr": proc.stderr.decode("utf-8", errors="replace"),
    }
