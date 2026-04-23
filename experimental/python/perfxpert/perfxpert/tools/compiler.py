"""compiler — flag lookup + explanation.

Reads knowledge/compiler_flags.yaml. Only allowlisted flags are returned by
lookup_flags. explain_flag works for any catalog entry.

Tool class: READ_ONLY.
"""

from typing import Any, Dict, List

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def lookup_flags(goal: str = None, arch: str = None, kernel_class: str = None) -> List[Dict[str, Any]]:
    """Return allowlisted compiler flags, optionally filtered by goal/arch/kernel_class.

    Filters only match flags that explicitly declare the relevant metadata.
    """
    flags = load_yaml("compiler_flags")
    matches = [f for f in flags if f.get("allowlist", False)]
    if goal is not None:
        matches = [
            f for f in matches
            if goal in f.get("goals", [])
        ]
    if arch is not None:
        matches = [
            f for f in matches
            if "all" in f.get("arches", []) or arch in f.get("arches", [])
        ]
    if kernel_class is not None:
        matches = [
            f for f in matches
            if "any" in f.get("kernel_classes", []) or kernel_class in f.get("kernel_classes", [])
        ]
    return matches


@tool_class(ToolClass.READ_ONLY)
def explain_flag(flag_name: str) -> Dict[str, Any]:
    """Return description + risk + note for a specific flag."""
    flags = load_yaml("compiler_flags")
    for entry in flags:
        if entry["flag"] == flag_name:
            return entry
    raise KeyError(f"Unknown flag {flag_name!r}")
