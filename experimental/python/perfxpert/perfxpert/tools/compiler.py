"""compiler — flag lookup + explanation.

Reads knowledge/compiler_flags.yaml. Only allowlisted flags are returned by
lookup_flags. explain_flag works for any catalog entry.

Tool class: READ_ONLY.
"""

from typing import Any, Dict, List

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def lookup_flags() -> List[Dict[str, Any]]:
    """Return allowlisted compiler flags."""
    flags = load_yaml("compiler_flags")
    return [f for f in flags if f.get("allowlist", False)]


@tool_class(ToolClass.READ_ONLY)
def explain_flag(flag_name: str) -> Dict[str, Any]:
    """Return description + risk + note for a specific flag."""
    flags = load_yaml("compiler_flags")
    for entry in flags:
        if entry["flag"] == flag_name:
            return entry
    raise KeyError(f"Unknown flag {flag_name!r}")
