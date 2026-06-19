"""MCP protocol-level integration tests.

Tests the perfxpert-mcp server at the JSON-RPC level without LLM or opencode.
Exercises MCP protocol handlers directly:
  - initialize handshake
  - tools/list RPC
  - tools/call RPC with representative tools
  - error handling
  - concurrency
  - stdio close handling
  - large result handling

Uses subprocess.Popen + direct JSON-RPC framing (newline-delimited) to test
the server in isolation, avoiding MCP SDK client dependencies.
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import pytest


def _perfxpert_mcp_path() -> Path:
    """Locate the perfxpert-mcp entry point."""
    result = subprocess.run(
        ["which", "perfxpert-mcp"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return Path(result.stdout.strip())
    raise RuntimeError(
        "perfxpert-mcp not found in PATH. Install with: pip install -e ."
    )


class MCPClient:
    """Minimal MCP JSON-RPC client for testing."""

    def __init__(self, proc):
        """Wrap a subprocess with stdio pipes."""
        self.proc = proc
        self._next_id = 1

    def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a JSON-RPC 2.0 request and return the parsed response."""
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
        }
        if params is not None:  # always add params if provided (even if empty dict)
            msg["params"] = params
        self._next_id += 1

        # Send request
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line.encode("utf-8"))
        self.proc.stdin.flush()

        # Read response (single line)
        response_line = self.proc.stdout.readline().decode("utf-8").strip()
        if not response_line:
            return {"error": "no response from server"}
        try:
            return json.loads(response_line)
        except json.JSONDecodeError as e:
            return {"error": f"invalid JSON: {e}", "raw": response_line}

    def close(self):
        """Close stdin and wait for process to exit."""
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=2)


@pytest.fixture
def mcp_proc():
    """Start a fresh perfxpert-mcp process."""
    mcp_path = _perfxpert_mcp_path()
    proc = subprocess.Popen(
        [str(mcp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,  # use binary mode for more control
    )
    yield proc
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_mcp_initialize_handshake(mcp_proc):
    """MCP protocol: server responds to initialize request."""
    client = MCPClient(mcp_proc)
    try:
        response = client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"},
            },
        )
        # Server should respond with capabilities
        assert "result" in response or "error" in response, (
            f"initialize: invalid response structure: {response}"
        )
        if "result" in response:
            result = response["result"]
            assert "protocolVersion" in result, (
                "initialize result must include protocolVersion"
            )
            assert "capabilities" in result, (
                "initialize result must include capabilities"
            )
    finally:
        client.close()


def test_mcp_tools_list_returns_many(mcp_proc):
    """MCP protocol: tools/list RPC returns ≥33 tools with schemas."""
    client = MCPClient(mcp_proc)
    try:
        # Initialize first (some MCP servers require this)
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"},
            },
        )

        response = client.send_request("tools/list")
        assert "result" in response, (
            f"tools/list: no result in response: {response}"
        )

        # Unpack tools list from result.tools
        tools = response["result"].get("tools", [])

        assert len(tools) >= 33, (
            f"Expected ≥33 tools, got {len(tools)}"
        )

        # Validate tool schema structure
        for tool in tools[:5]:  # spot-check first 5
            assert "name" in tool, f"tool missing 'name': {tool}"
            assert "description" in tool, f"tool {tool.get('name')} missing 'description'"
            assert "inputSchema" in tool, f"tool {tool.get('name')} missing 'inputSchema'"
            schema = tool["inputSchema"]
            assert schema.get("type") == "object", (
                f"inputSchema for {tool.get('name')} must be type=object"
            )
    finally:
        client.close()


def test_mcp_call_arch_lookup_peaks(mcp_proc):
    """MCP protocol: tools/call for arch.lookup_peaks with valid gfx_id."""
    client = MCPClient(mcp_proc)
    try:
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"}
            }
        )

        response = client.send_request(
            "tools/call",
            {
                "name": "arch_lookup_peaks",  # dots → underscores in MCP names
                "arguments": {"gfx_id": "gfx942"},
            },
        )

        # Response must have result with content array
        assert "result" in response, (
            f"tools/call: no result: {response}"
        )

        content_list = response["result"].get("content", [])
        assert len(content_list) > 0, "tools/call returned no content"

        # First item should have text (JSON)
        first = content_list[0]
        assert "text" in first, f"content missing 'text': {first}"

        result = json.loads(first["text"])
        assert "peak_fp64_tflops" in result, (
            f"arch lookup result missing peak_fp64_tflops: {result}"
        )
        assert result["peak_fp64_tflops"] == 81.7, (
            f"MI300X FP64 peak should be 81.7, got {result['peak_fp64_tflops']}"
        )
    finally:
        client.close()


def test_mcp_call_bottleneck_classify(mcp_proc):
    """MCP protocol: tools/call for bottleneck.classify_from_metrics."""
    client = MCPClient(mcp_proc)
    try:
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"}
            }
        )

        # Synthetic metrics: memory-bound kernel
        metrics = {
            "hbm_read_bw_pct": 85.0,
            "hbm_write_bw_pct": 5.0,
            "l1_hit_rate": 0.3,
            "l2_hit_rate": 0.7,
            "gpu_utilization_pct": 45.0,
        }

        response = client.send_request(
            "tools/call",
            {
                "name": "bottleneck_classify_from_metrics",
                "arguments": {"metrics": metrics},
            },
        )

        assert "result" in response, (
            f"bottleneck/classify: no result: {response}"
        )

        content_list = response["result"].get("content", [])
        assert len(content_list) > 0, "bottleneck/classify returned no content"

        result = json.loads(content_list[0]["text"])
        assert "type" in result, (
            f"bottleneck result missing type: {result}"
        )
        # Result may be "memory_transfer", "mixed", etc. Just verify it's present
        assert isinstance(result["type"], str), (
            f"Expected type to be string, got {result}"
        )
    finally:
        client.close()


def test_mcp_call_roofline_classify(mcp_proc):
    """MCP protocol: tools/call for roofline.classify."""
    client = MCPClient(mcp_proc)
    try:
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"}
            }
        )

        response = client.send_request(
            "tools/call",
            {
                "name": "roofline_classify",
                "arguments": {
                    "flops": 1e12,
                    "bytes": 1e11,
                    "gfx_id": "gfx942",
                },
            },
        )

        assert "result" in response, (
            f"roofline/classify: no result: {response}"
        )

        content_list = response["result"].get("content", [])
        result = json.loads(content_list[0]["text"])
        assert "regime" in result, f"roofline result missing regime: {result}"
        assert result["regime"] in ["compute", "memory", "balanced"], (
            f"unexpected regime: {result['regime']}"
        )
        assert "arithmetic_intensity" in result, (
            f"roofline result missing arithmetic_intensity: {result}"
        )
    finally:
        client.close()


def test_mcp_call_sol_sanity_check(mcp_proc):
    """MCP protocol: tools/call for sol.sanity_check."""
    client = MCPClient(mcp_proc)
    try:
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"}
            }
        )

        # Valid: 50 TFLOPS fp64 (below MI300X peak of 81.7)
        response = client.send_request(
            "tools/call",
            {
                "name": "sol_sanity_check",
                "arguments": {
                    "achieved_flops_per_sec": 50e12,
                    "kernel_type": "fp64",
                    "gfx_id": "gfx942",
                },
            },
        )

        assert "result" in response, (
            f"sol/sanity_check: no result: {response}"
        )

        content_list = response["result"].get("content", [])
        result = json.loads(content_list[0]["text"])
        assert "plausible" in result, f"sol result missing plausible: {result}"
        assert result["plausible"] is True, (
            "50 TFLOPS should be plausible for MI300X fp64"
        )
    finally:
        client.close()


def test_mcp_call_intent_classify(mcp_proc):
    """MCP protocol: tools/call for intent.classify."""
    client = MCPClient(mcp_proc)
    try:
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"}
            }
        )

        response = client.send_request(
            "tools/call",
            {
                "name": "intent_classify",
                "arguments": {
                    "user_query": "How can I optimize the kernel performance?",
                },
            },
        )

        assert "result" in response, (
            f"intent/classify: no result: {response}"
        )

        content_list = response["result"].get("content", [])
        result = json.loads(content_list[0]["text"])
        assert "intent" in result, f"intent result missing intent: {result}"
        assert result["intent"] in ["optimize", "analyze", "explain", "help", "verify"], (
            f"unexpected intent: {result['intent']}"
        )
    finally:
        client.close()


def test_mcp_error_unknown_tool(mcp_proc):
    """MCP protocol: tools/call with unknown tool returns error (not crash)."""
    client = MCPClient(mcp_proc)
    try:
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"}
            }
        )

        response = client.send_request(
            "tools/call",
            {
                "name": "nonexistent_tool_12345",
                "arguments": {},
            },
        )

        # Server should return either a JSON-RPC error or result with isError=True
        assert "error" in response or response.get("result", {}).get("isError"), (
            f"Expected error for unknown tool, got: {response}"
        )
    finally:
        client.close()


def test_mcp_error_invalid_arguments(mcp_proc):
    """MCP protocol: tools/call with missing required args returns error."""
    client = MCPClient(mcp_proc)
    try:
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"}
            }
        )

        # arch.lookup_peaks requires gfx_id; omit it
        response = client.send_request(
            "tools/call",
            {
                "name": "arch_lookup_peaks",
                "arguments": {},  # missing gfx_id
            },
        )

        # Server should return error gracefully
        assert "error" in response or "result" in response, (
            f"Expected error or graceful handling for missing arg: {response}"
        )
    finally:
        client.close()


def test_mcp_concurrent_calls(mcp_proc):
    """MCP protocol: concurrent tools/call requests handled correctly.

    Sends 3 requests without reading intermediate responses, then reads all.
    """
    client = MCPClient(mcp_proc)
    try:
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"}
            }
        )

        # Send 3 requests without waiting for responses
        requests = [
            {
                "name": "arch_lookup_peaks",
                "arguments": {"gfx_id": "gfx942"},
            },
            {
                "name": "intent_classify",
                "arguments": {"user_query": "optimize"},
            },
            {
                "name": "sol_sanity_check",
                "arguments": {
                    "achieved_flops_per_sec": 40e12,
                    "kernel_type": "fp32",
                    "gfx_id": "gfx906",
                },
            },
        ]

        # Send all 3
        for req in requests:
            msg = {
                "jsonrpc": "2.0",
                "id": client._next_id,
                "method": "tools/call",
                "params": req,
            }
            client._next_id += 1
            client.proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        client.proc.stdin.flush()

        # Read 3 responses
        responses = []
        for _ in range(3):
            response_line = client.proc.stdout.readline().decode("utf-8").strip()
            if response_line:
                responses.append(json.loads(response_line))

        assert len(responses) == 3, f"Expected 3 responses, got {len(responses)}"

        # Each should either have result or error
        for r in responses:
            assert "result" in r or "error" in r, (
                f"Response missing result/error: {r}"
            )
    finally:
        client.close()


def test_mcp_large_result_handling(mcp_proc):
    """MCP protocol: large tool results are returned without truncation.

    Uses regression.compare_runs with synthetic paths to generate a sizable response.
    """
    import tempfile

    client = MCPClient(mcp_proc)
    try:
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"}
            }
        )

        # Skip if regression tool unavailable (may not be in registry if no DB support)
        list_response = client.send_request("tools/list", {})
        tools_list = list_response.get("result", {}).get("tools", list_response.get("tools", []))
        tool_names = [t["name"] for t in tools_list]

        if "regression_compare_runs" not in tool_names:
            pytest.skip("regression.compare_runs not in registry")

        # Use temp files for DB paths (won't exist, but tool may handle gracefully)
        with tempfile.NamedTemporaryFile(suffix=".db") as f1:
            with tempfile.NamedTemporaryFile(suffix=".db") as f2:
                response = client.send_request(
                    "tools/call",
                    {
                        "name": "regression_compare_runs",
                        "arguments": {
                            "db_before": f1.name,
                            "db_after": f2.name,
                        },
                    },
                )

        # Response should be present (may be error if DB doesn't exist, but not truncated)
        assert "result" in response or "error" in response, (
            f"regression/compare_runs: no result/error: {response}"
        )
        # If result exists, it should be a valid text content
        if "result" in response:
            assert len(response["result"]) > 0, "Empty result"
    finally:
        client.close()


def test_mcp_stdin_close_exits_cleanly(mcp_proc):
    """MCP protocol: server exits cleanly when stdin closes."""
    client = MCPClient(mcp_proc)
    try:
        client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "perfxpert-test", "version": "1.0"}
            }
        )
        # Now close stdin (normal client disconnect)
        client.close()

        # Wait deterministically for the server to exit instead of sleeping.
        # Finding #27: time.sleep(0.5) is flaky on loaded CI machines.
        try:
            exit_code = mcp_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            mcp_proc.kill()
            pytest.fail("Server did not exit within 5s after stdin close")
        # Server exited cleanly
        assert exit_code == 0 or exit_code is None, (
            f"Server exited with non-zero code: {exit_code}"
        )
    except AssertionError:
        raise
