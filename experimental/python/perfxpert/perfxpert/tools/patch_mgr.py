"""patch_mgr — patch.apply / revert / verify_output.

EXECUTION class. Never register with MCP server (§5.8).

All three tools route through `perfxpert.tools._safety` for path confinement
and shell-metachar denial.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools._safety import (
    confine_to_project_root,
    reject_shell_metachars,
)


_BACKUP_SUFFIX = ".bak"


def _prepare_target(project_root: Path, rel_path: str) -> Path:
    """Run all §5.8 checks on the target path, return resolved Path."""
    reject_shell_metachars(rel_path)
    return confine_to_project_root(Path(project_root), rel_path)


@tool_class(ToolClass.EXECUTION)
def apply(project_root: Path, rel_path: str, new_content: str) -> Dict[str, Any]:
    """Replace file contents; write `.bak` of prior content if no backup exists.

    Args:
        project_root: project root directory. Anchors path confinement.
        rel_path: file path relative to project_root. Must NOT contain `..`,
                  symlinks escaping root, or shell metachars.
        new_content: full new file content (text mode).

    Returns:
        {"applied": True, "backup_path": str, "target_path": str}

    Raises:
        PathConfinementError, ShellMetacharError — if the path violates §5.8.
        FileNotFoundError — if the target does not exist.
    """
    target = _prepare_target(project_root, rel_path)
    if not target.exists():
        raise FileNotFoundError(f"target {target} does not exist")

    backup = target.with_suffix(target.suffix + _BACKUP_SUFFIX)
    # Preserve the FIRST backup across repeated apply() calls (see test_apply_is_idempotent).
    if not backup.exists():
        backup.write_bytes(target.read_bytes())

    target.write_text(new_content)
    return {
        "applied": True,
        "backup_path": str(backup),
        "target_path": str(target),
    }


@tool_class(ToolClass.EXECUTION)
def revert(project_root: Path, rel_path: str) -> Dict[str, Any]:
    """Restore file from its `.bak` backup; remove backup after restore.

    Raises:
        FileNotFoundError if no `.bak` exists for this path.
        PathConfinementError, ShellMetacharError on §5.8 violation.
    """
    target = _prepare_target(project_root, rel_path)
    backup = target.with_suffix(target.suffix + _BACKUP_SUFFIX)
    if not backup.exists():
        raise FileNotFoundError(f"no backup at {backup}")

    target.write_bytes(backup.read_bytes())
    backup.unlink()
    return {
        "reverted": True,
        "target_path": str(target),
    }


@tool_class(ToolClass.EXECUTION)
def verify_output(
    project_root: Path,
    baseline_rel: str,
    new_rel: str,
    *,
    tolerance: float | None = None,
) -> Dict[str, Any]:
    """Compare two output files — bit-exact or np.allclose.

    Method selected by file extension:
    - `.npy` / `.npz` → `numpy.load` + `np.allclose(..., rtol=tolerance)`
    - everything else → `bytes ==`

    `tolerance=None` means bit-exact even for numpy arrays.

    Returns:
        {"match": bool, "method": "bit_exact"|"np_allclose", "reason": str}
    """
    baseline = _prepare_target(project_root, baseline_rel)
    new = _prepare_target(project_root, new_rel)

    # Numeric path for .npy/.npz
    if baseline.suffix in {".npy", ".npz"} and tolerance is not None:
        import numpy as np  # imported lazily; test guards with importorskip
        a = np.load(baseline)
        b = np.load(new)
        match = bool(np.allclose(a, b, rtol=tolerance, atol=tolerance))
        return {
            "match": match,
            "method": "np_allclose",
            "reason": f"np.allclose(rtol={tolerance}) = {match}",
        }

    # Bit-exact path
    match = baseline.read_bytes() == new.read_bytes()
    return {
        "match": match,
        "method": "bit_exact",
        "reason": "byte-equal" if match else "byte-unequal",
    }
