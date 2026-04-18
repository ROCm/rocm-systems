"""_registry — discover perfxpert READ_ONLY tools via @tool_class introspection.

Every public callable in `perfxpert.tools.*` annotated with
`@tool_class(ToolClass.READ_ONLY)` is registered. Anything else is skipped.

The returned dict keys use the `<module>.<function>` convention so MCP
clients can address tools unambiguously.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Callable, Dict

import perfxpert.tools as _tools_pkg
from perfxpert.tools._class import ToolClass


_SKIP_MODULES = {"_class", "_safety"}


def discover_read_only_tools() -> Dict[str, Callable]:
    """Walk perfxpert.tools.*; collect READ_ONLY callables.

    Returns:
        {"<module_stem>.<fn_name>": fn, ...}
    """
    registry: Dict[str, Callable] = {}
    for mod_info in pkgutil.iter_modules(_tools_pkg.__path__):
        if mod_info.ispkg or mod_info.name in _SKIP_MODULES:
            continue
        mod = importlib.import_module(f"perfxpert.tools.{mod_info.name}")
        for fn_name, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn_name.startswith("_"):
                continue
            cls = getattr(fn, "__tool_class__", None)
            if cls is ToolClass.READ_ONLY:
                registry[f"{mod_info.name}.{fn_name}"] = fn
    return registry
