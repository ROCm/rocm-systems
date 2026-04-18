"""compile_runner — compile.build EXECUTION tool.

Invokes amdclang++ (or ${PERFXPERT_CXX}) on sanitized inputs. Flag set is
filtered against knowledge/compiler_flags.yaml's allowlist entries; any
unknown flag raises CompileFlagError before subprocess launch (§5.8).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Set

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools._safety import (
    build_safe_env,
    confine_to_project_root,
    filter_by_allowlist,
    reject_shell_metachars,
)
from perfxpert.tools._tooldep import require_tool


_DEFAULT_CXX = os.environ.get("PERFXPERT_CXX", "amdclang++")
_BUILD_TIMEOUT_SEC = 120


class CompileFlagError(Exception):
    """Raised when a flag is not in the compiler_flags.yaml allowlist."""


def _load_allowlist() -> Set[str]:
    """Return set of allowed flag keys from knowledge/compiler_flags.yaml.

    An entry counts as allowed iff it sets `allowlist: true`.
    """
    catalog = load_yaml("compiler_flags")
    allowed = set()
    for entry in catalog:
        if entry.get("allowlist") is True:
            allowed.add(entry["flag"])
    return allowed


_COMPILE_ERROR_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<sev>error|warning):\s*(?P<msg>.+)$",
    re.MULTILINE,
)


def _parse_errors(stderr: str) -> List[Dict[str, Any]]:
    """Parse clang-style error lines into structured entries."""
    return [m.groupdict() for m in _COMPILE_ERROR_RE.finditer(stderr)]


@tool_class(ToolClass.EXECUTION)
def build(
    project_root: Path,
    source_rel: str,
    flags: List[str],
    *,
    output_rel: str | None = None,
    timeout: int = _BUILD_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Compile `source_rel` with `flags` under `project_root`.

    All flags are validated against knowledge/compiler_flags.yaml; unknown
    flags raise CompileFlagError. Source + output paths are confined to
    project_root. API keys are NOT forwarded.

    Returns:
        {"returncode": int, "stdout": str, "stderr": str, "errors": [...]}

    Raises:
        CompileFlagError — flag not on allowlist.
        PathConfinementError — source or output escapes project_root.
        ShellMetacharError — any input contains shell metachars.
        subprocess.TimeoutExpired — build exceeds `timeout`.
    """
    # Check external dependencies
    require_tool("amdclang++")

    # Sanitize + confine paths
    reject_shell_metachars(source_rel)
    source = confine_to_project_root(Path(project_root), source_rel)
    if output_rel is not None:
        reject_shell_metachars(output_rel)

    # Sanitize flags
    for f in flags:
        reject_shell_metachars(f)
    allowed = _load_allowlist()
    # For value-taking flags like --offload-arch=gfx942, match on the key
    accepted, rejected = filter_by_allowlist(flags, allowed)
    if rejected:
        raise CompileFlagError(
            f"flags not in compiler_flags.yaml allowlist: {rejected!r}"
        )

    # Build argv
    argv: List[str] = [_DEFAULT_CXX, str(source)]
    argv.extend(accepted)
    if output_rel is not None:
        output = confine_to_project_root(Path(project_root), output_rel)
        argv.extend(["-o", str(output)])

    proc = subprocess.run(
        argv,
        shell=False,
        capture_output=True,
        env=build_safe_env(),
        cwd=str(project_root),
        timeout=timeout,
        check=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "errors": _parse_errors(stderr),
    }
