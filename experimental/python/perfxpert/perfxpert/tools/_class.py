"""Tool MCP exposure class system.

READ_ONLY tools are safe to expose via MCP to external clients.
EXECUTION tools modify the user's environment and are in-process only.

See design spec §5.8 (Code-modification threat model) for rationale.
"""

from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T", bound=Callable)


class ToolClass(Enum):
    """MCP exposure class for tools. See spec §5.8."""

    READ_ONLY = "read_only"     # lookups, analysis, classification — safe for MCP
    EXECUTION = "execution"     # modifies filesystem/processes — in-process only


def tool_class(klass: ToolClass) -> Callable[[T], T]:
    """Decorator to annotate a tool with its MCP exposure class.

    Example:
        @tool_class(ToolClass.READ_ONLY)
        def lookup_peaks(gfx_id: str) -> dict: ...

        @tool_class(ToolClass.EXECUTION)
        def apply_patch(file: str, diff: str) -> bool: ...

    The MCP server only registers functions annotated with
    ToolClass.READ_ONLY. CI test test_mcp_exposure.py enforces that no
    EXECUTION tool is in the MCP registry.
    """
    def decorator(fn: T) -> T:
        fn.__tool_class__ = klass  # type: ignore[attr-defined]
        return fn
    return decorator
