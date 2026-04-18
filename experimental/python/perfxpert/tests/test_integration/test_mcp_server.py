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
    """build_server() raises clean error if MCP SDK missing."""
    # This test verifies the adapter pattern works
    # (normally skipped since MCP is installed)
    try:
        import mcp  # noqa: F401
        pytest.skip("MCP SDK is installed; test only relevant when SDK missing")
    except ImportError:
        from mcp_server.server import build_server
        with pytest.raises(RuntimeError, match="MCP SDK not installed"):
            build_server()
