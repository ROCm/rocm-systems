"""Integration: MCP server constructs cleanly."""

import pytest


def test_server_module_imports():
    """Module must import whether or not MCP SDK is installed."""
    from mcp_server import server  # noqa: F401


def test_server_constructs_when_sdk_present():
    """Server constructs successfully when MCP SDK is available."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        pytest.skip("MCP SDK not installed")
    from mcp_server.server import build_server
    s = build_server()
    assert s is not None


def test_build_server_raises_without_sdk():
    """build_server() raises clean error if MCP SDK missing.

    Uses unittest.mock to simulate absent SDK without uninstalling it.
    This test always runs (no skip).
    """
    import sys
    from unittest.mock import patch

    # Patch sys.modules to make MCP imports fail
    mcp_modules = {
        'mcp': None,
        'mcp.server': None,
        'mcp.server.stdio': None,
        'mcp.types': None,
    }
    with patch.dict(sys.modules, mcp_modules):
        # Force reimport of mcp_server.server to see the patched modules
        import importlib
        if 'mcp_server.server' in sys.modules:
            del sys.modules['mcp_server.server']
        if 'mcp_server' in sys.modules:
            del sys.modules['mcp_server']

        from mcp_server.server import build_server
        with pytest.raises(RuntimeError, match="MCP SDK not available"):
            build_server()
