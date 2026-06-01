"""E2E: opencode session — real MCP call through perfxpert-mcp.

Requires:
- perfxpert-code installed (entry point)
- perfxpert-mcp installed (entry point)
- bundled patched opencode binary available
- Any LLM provider configured, OR --no-llm / air-gap mode

Skips gracefully if the opencode binary isn't available.
"""

import os
import subprocess

import pytest


@pytest.fixture
def opencode_available():
    try:
        from perfxpert.cli.opencode_launcher import resolve_opencode_binary

        resolve_opencode_binary()
        return True
    except Exception:
        return False


def test_perfxpert_code_launches_if_opencode_available(opencode_available):
    if not opencode_available:
        pytest.skip("opencode binary not available on this system")

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
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            _, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            _, stderr = proc.communicate(timeout=5)
    assert b"AMD ROCm PerfXpert" in stderr


def test_mcp_server_accepts_a_call_from_shell(opencode_available):
    """Verify perfxpert-mcp at least starts and can receive an initialization message."""
    if not opencode_available:
        pytest.skip("opencode binary not available on this system")

    import json
    import select

    # Start perfxpert-mcp with stdio
    p = subprocess.Popen(
        ["perfxpert-mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Send an MCP initialize request (simplified; real MCP uses JSON-RPC framing)
        init = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-e2e", "version": "1.0"},
            },
        }) + "\n"
        p.stdin.write(init.encode("utf-8"))
        p.stdin.flush()

        ready, _, _ = select.select([p.stdout], [], [], 5)
        assert ready, "perfxpert-mcp did not respond to initialize within 5s"
        response_line = p.stdout.readline().decode("utf-8").strip()
        response = json.loads(response_line)
        assert "result" in response or "error" in response
    finally:
        if p.stdin and not p.stdin.closed:
            p.stdin.close()
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=2)


# NOTE: The test_end_to_end_interactive_session was a flaky pexpect-based TUI test.
# It has been replaced by test_perfxpert_code_live_mcp.py::test_perfxpert_code_calls_mcp_and_returns_expected_value
# which uses the non-interactive `perfxpert-code run` path and is more robust.
# Both tests verify the same contract:
#   launcher → opencode → MCP → Python brain → response
# but the live-MCP version avoids pexpect's alternate-screen fragility.
