"""Deterministic tool floor — pure-Python functions, no LLM calls.

Each tool is annotated with its MCP exposure class (READ_ONLY vs EXECUTION).
See perfxpert.tools._class for the class system. The MCP server wraps
READ_ONLY tools for external clients.
"""

from ._class import ToolClass, tool_class

__all__ = ["ToolClass", "tool_class"]
