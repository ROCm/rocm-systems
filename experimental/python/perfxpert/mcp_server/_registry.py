"""_registry — discover perfxpert READ_ONLY tools via @tool_class introspection.

Every public callable in `perfxpert.tools.*` annotated with
`@tool_class(ToolClass.READ_ONLY)` is registered. Anything else is skipped.

The returned dict keys use the `<module_path>.<function>` convention so
MCP clients can address tools unambiguously — dots in the module path
are preserved (e.g. `agents.root.agent_root`). The MCP server replaces
dots with underscores when wiring the tool name onto the wire.

Sub-packages under `perfxpert.tools.*` (e.g. `perfxpert.tools.agents`)
are walked recursively so agent-as-tool wrappers are discovered
alongside the flat tool modules.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Callable, Dict

import perfxpert.tools as _tools_pkg
from perfxpert.tools._class import ToolClass


_SKIP_MODULES = {
    "_class",
    "_safety",
    # Specialist-internal catalog shims. These exist so the layer-2 agents can
    # fetch deterministic recommendation catalogs, but they are not part of the
    # user-facing MCP surface.
    "compute_techniques",
    "memory_techniques",
    "latency_techniques",
}


def discover_read_only_tools() -> Dict[str, Callable]:
    """Walk perfxpert.tools.* (including sub-packages); collect READ_ONLY callables.

    Returns:
        {"<module_path_without_perfxpert_tools_prefix>.<fn_name>": fn, ...}
        where the module path uses dots and omits the ``perfxpert.tools.``
        prefix. Example keys:
          - ``arch.lookup_peaks``                (flat module)
          - ``agents.root.agent_root``           (sub-package module)
    """
    registry: Dict[str, Callable] = {}
    prefix = _tools_pkg.__name__ + "."
    for mod_info in pkgutil.walk_packages(
        _tools_pkg.__path__, prefix=prefix
    ):
        # Drop the "perfxpert.tools." prefix so keys stay short.
        rel_name = mod_info.name[len(prefix):]
        # Skip the private helpers and any dotted-path whose final
        # segment is on the skip list.
        if rel_name.split(".")[-1] in _SKIP_MODULES:
            continue
        # Packages themselves (their __init__.py) are skipped; we only
        # walk them to reach their children.
        if mod_info.ispkg:
            continue
        mod = importlib.import_module(mod_info.name)
        for fn_name, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn_name.startswith("_"):
                continue
            cls = getattr(fn, "__tool_class__", None)
            if cls is ToolClass.READ_ONLY:
                override = getattr(fn, "__tool_name__", None)
                key = override if override else f"{rel_name}.{fn_name}"
                registry[key] = fn
    return registry
