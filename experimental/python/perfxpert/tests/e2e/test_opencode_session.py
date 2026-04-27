"""E2E: opencode session — real MCP call through perfxpert-mcp.

Requires:
- perfxpert-code installed (entry point)
- perfxpert-mcp installed (entry point)
- managed bundled opencode binary
- Any LLM provider configured, OR --no-llm / air-gap mode

Skips launcher smoke gracefully if the managed opencode binary isn't available.
"""

import os
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def opencode_available():
    # The default perfxpert-code path intentionally uses only the managed
    # bundled binary. User-owned upstream binaries are limited to the explicit
    # `perfxpert-code opencode ...` escape hatch.
    try:
        from perfxpert.cli.opencode_launcher import resolve_opencode_binary

        resolve_opencode_binary()
    except FileNotFoundError:
        return False
    return shutil.which("perfxpert-code") is not None


def test_perfxpert_code_launches_if_opencode_available(opencode_available):
    if not opencode_available:
        pytest.skip("managed opencode/perfxpert-code not available on this system")

    # Smoke: perfxpert-code must print our AMD banner to stderr BEFORE handing
    # off to opencode's interactive TUI. opencode is an alternate-screen
    # renderer that doesn't exit on stdin "exit\n", so we start it with Popen,
    # wait briefly for the banner, then kill it. We only assert the banner
    # emerged — proving the launcher ran.
    import signal
    proc = subprocess.Popen(
        ["perfxpert-code"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PERFXPERT_CODE_NO_BANNER": "0"},
        start_new_session=True,  # so we can kill the whole process group
    )
    try:
        import time as _t
        _t.sleep(1.5)
    finally:
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        else:
            proc.kill()
        try:
            _, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            _, stderr = proc.communicate(timeout=5)
    assert b"AMD ROCm PerfXpert" in stderr


def test_mcp_server_accepts_a_call_from_shell():
    """Verify perfxpert-mcp starts and responds through the MCP stdio transport."""
    import anyio
    from datetime import timedelta
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def _probe() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
        )
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=10),
            ) as session:
                init = await session.initialize()
                assert init.serverInfo.name == "perfxpert"
                tools = await session.list_tools()
                assert tools.tools

    anyio.run(_probe)


# NOTE: The test_end_to_end_interactive_session was a flaky pexpect-based TUI test.
# It has been replaced by test_perfxpert_code_live_mcp.py::test_perfxpert_code_calls_mcp_and_returns_expected_value
# which uses the non-interactive `perfxpert-code run` path and is more robust.
# Both tests verify the same contract:
#   launcher → opencode → MCP → Python brain → response
# but the live-MCP version avoids pexpect's alternate-screen fragility.
