"""anchors — anchors.check EXECUTION tool.

Invokes a tightly-scoped test runner against a new binary.
The runner executable and a small set of command-shape escape hatches are
allowlisted to prevent arbitrary-command execution through "test_command".
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools._safety import (
    build_safe_env,
    confine_to_project_root,
    reject_shell_metachars,
)


_ANCHOR_TIMEOUT_SEC = 300
_ALLOWED_EXECUTABLES = frozenset({"pytest", "pytest-3", "ctest"})
_DANGEROUS_FLAGS = {
    "pytest": frozenset({"-p", "--pyargs", "--pastebin"}),
    "pytest-3": frozenset({"-p", "--pyargs", "--pastebin"}),
    "ctest": frozenset({"-S", "--build-and-test"}),
}
_PATH_VALUE_FLAGS = {
    "pytest": frozenset(
        {"--rootdir", "--confcutdir", "--basetemp", "--ignore", "--ignore-glob", "--junitxml", "-c"}
    ),
    "pytest-3": frozenset(
        {"--rootdir", "--confcutdir", "--basetemp", "--ignore", "--ignore-glob", "--junitxml", "-c"}
    ),
    "ctest": frozenset({"--test-dir", "--output-junit"}),
}
_TRUSTED_EXECUTABLE_DIRS = (
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
    "/usr/local/sbin",
    "/usr/sbin",
    "/sbin",
    "/opt/rocm/bin",
)
_TRUSTED_EXECUTABLE_PATH = os.pathsep.join(_TRUSTED_EXECUTABLE_DIRS)


class AnchorCommandError(ValueError):
    """Raised when the anchor command is outside the supported allowlist."""


def _resolve_trusted_executable(exe: str) -> str:
    resolved = shutil.which(exe, path=_TRUSTED_EXECUTABLE_PATH)
    if resolved is None:
        raise AnchorCommandError(f"test executable not found on trusted PATH: {exe!r}")
    return resolved


def _validate_test_command(project_root: Path, test_command: List[str]) -> None:
    if not test_command:
        raise ValueError("test_command must not be empty")

    for tok in test_command:
        reject_shell_metachars(tok)

    exe = test_command[0]
    if Path(exe).name != exe:
        raise AnchorCommandError(
            f"test executable must be a bare allowlisted name, got {exe!r}"
        )
    if exe not in _ALLOWED_EXECUTABLES:
        raise AnchorCommandError(f"test executable not in allowlist: {exe!r}")

    for tok in test_command[1:]:
        if tok.startswith("-"):
            key = tok.split("=", 1)[0]
            dangerous = _DANGEROUS_FLAGS.get(exe, frozenset())
            if key in dangerous or any(
                len(flag) == 2 and key.startswith(flag) for flag in dangerous if flag.startswith("-")
            ):
                raise AnchorCommandError(
                    f"dangerous flag not allowed for {exe}: {tok!r}"
                )
            if "=" in tok and key in _PATH_VALUE_FLAGS.get(exe, frozenset()):
                _, value = tok.split("=", 1)
                confine_to_project_root(project_root, value)
            for path_flag in _PATH_VALUE_FLAGS.get(exe, frozenset()):
                if len(path_flag) == 2 and tok.startswith(path_flag) and tok != path_flag:
                    confine_to_project_root(project_root, tok[len(path_flag):])
                    break
            continue
        confine_to_project_root(project_root, tok)


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
    _validate_test_command(project_root, test_command)
    resolved_command = [_resolve_trusted_executable(test_command[0]), *test_command[1:]]

    proc = subprocess.run(
        resolved_command,
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
