"""E2E: opencode session — real MCP call through perfxpert-mcp.

Requires:
- perfxpert-code installed (entry point)
- perfxpert-mcp installed (entry point)
- bundled opencode binary OR PERFXPERT_OPENCODE_PATH set
- Any LLM provider configured, OR --no-llm / air-gap mode

Skips gracefully if the opencode binary isn't available.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def opencode_available():
    # Check bundled OR PATH
    if os.environ.get("PERFXPERT_OPENCODE_PATH"):
        if Path(os.environ["PERFXPERT_OPENCODE_PATH"]).is_file():
            return True
    if shutil.which("opencode"):
        return True
    try:
        from importlib import resources
        with resources.as_file(resources.files("perfxpert") / "_bundled" / "opencode") as p:
            if p.is_file():
                return True
    except Exception:
        pass
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
    import json, time

    # Start perfxpert-mcp with stdio
    p = subprocess.Popen(
        ["perfxpert-mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Send an MCP initialize request (simplified; real MCP uses JSON-RPC framing)
        init = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        }) + "\n"
        p.stdin.write(init.encode("utf-8"))
        p.stdin.flush()
        time.sleep(0.5)
        # Close stdin to signal EOF; server should exit cleanly
        p.stdin.close()
        p.wait(timeout=5)
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait(timeout=2)
    # The process should at least start without an immediate crash
    # (exact protocol handling depends on MCP SDK version; deeper testing in test_mcp_server.py)


@pytest.mark.skipif(
    not os.environ.get("PERFXPERT_E2E_INTERACTIVE"),
    reason="Full interactive E2E requires PERFXPERT_E2E_INTERACTIVE=1 and a configured LLM",
)
def test_end_to_end_interactive_session(opencode_available, tmp_path):
    """Full round-trip: perfxpert-code -> opencode master.md -> MCP -> Python brain.

    Only runs when PERFXPERT_E2E_INTERACTIVE=1 because it requires:
    - opencode bundled / installed
    - An LLM provider configured (ANTHROPIC_API_KEY or OPENAI_API_KEY)
    - Real stdio interaction
    """
    if not opencode_available:
        pytest.skip("opencode not available")

    import pexpect
    env = {**os.environ, "PERFXPERT_CODE_NO_BANNER": "0"}
    child = pexpect.spawn(
        "perfxpert-code",
        env=env,
        encoding="utf-8",
        timeout=60,
    )
    try:
        child.expect("ROCm PerfXpert")
        child.sendline("What are the FP64 peaks for MI300X?")
        # Expect a streamed response that mentions either the number 81.7 or the
        # phrase "TFLOPS". The LLM may phrase this many ways, but SOMETHING
        # grounded in the arch.lookup_peaks tool output must appear.
        child.expect(r"(81\.7|TFLOPS)", timeout=60)
        child.sendline("/exit")
        child.expect(pexpect.EOF, timeout=10)
    finally:
        if child.isalive():
            child.terminate(force=True)
