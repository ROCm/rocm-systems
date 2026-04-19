"""Tool-name length CI lint (Task 9, I8).

Claude Code caps MCP tool names at 64 chars on the wire. Our claude
template is `mcp__perfxpert__<underscored_name>` (16 prefix chars);
anything > 48 chars in the internal tool stem would overflow.

The Gemini template is `mcp_perfxpert_<underscored_name>` (14
prefix chars); the cap is the same 64 so Gemini has 50 chars of
headroom.

This test iterates the perfxpert MCP READ_ONLY registry and:

* FAILS when the rendered name exceeds 64 chars.
* WARNS (via stderr) when it exceeds 56 chars (approaching limit).
"""

from __future__ import annotations

import sys

import pytest


_MAX_CHARS = 64
_WARN_CHARS = 56

_CLAUDE_PREFIX = "mcp__perfxpert__"
_GEMINI_PREFIX = "mcp_perfxpert_"


def _load_registry() -> dict:
    """Return the perfxpert READ_ONLY tool registry; skip on import error."""
    # mcp_server lives as a top-level package alongside `perfxpert` in the
    # editable tree; try both import paths so the test runs regardless of
    # how the repo is checked out.
    for path in (
        "perfxpert.mcp_server._registry",
        "mcp_server._registry",
    ):
        try:
            mod = __import__(path, fromlist=["discover_read_only_tools"])
            return mod.discover_read_only_tools()
        except ImportError:
            continue
    pytest.skip("perfxpert mcp_server registry unavailable")


def _wire_name(internal: str) -> str:
    """Convert `module.fn` → `module_fn` the way the MCP server does."""
    return internal.replace(".", "_")


@pytest.mark.parametrize("prefix", [_CLAUDE_PREFIX, _GEMINI_PREFIX])
def test_every_tool_name_fits_within_64_chars(prefix: str) -> None:
    """Every tool, in every backend template, stays ≤ 64 chars."""
    registry = _load_registry()
    if not registry:
        pytest.skip("empty perfxpert tool registry")
    offenders: list[tuple[str, int]] = []
    warnings: list[tuple[str, int]] = []
    for internal in registry:
        wire = _wire_name(internal)
        rendered = prefix + wire
        n = len(rendered)
        if n > _MAX_CHARS:
            offenders.append((rendered, n))
        elif n > _WARN_CHARS:
            warnings.append((rendered, n))

    for name, n in warnings:
        sys.stderr.write(
            f"[tool-name-length WARN] {name!r} is {n} chars "
            f"(>{_WARN_CHARS}; approaching 64-char cap)\n"
        )

    assert not offenders, (
        "These tool names exceed the 64-char MCP cap:\n  "
        + "\n  ".join(f"{name} ({n} chars)" for name, n in offenders)
    )


def test_registry_lint_discovers_tools() -> None:
    """Sanity: the registry must return at least one tool, else the
    length lint above is silently skipped."""
    registry = _load_registry()
    assert registry, "perfxpert MCP registry is empty — lint would be a no-op"
