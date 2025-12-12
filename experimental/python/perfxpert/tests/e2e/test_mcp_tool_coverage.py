"""Finding #25 — E2E MCP tool coverage beyond the 1-tool tested by test_mcp_protocol.

Tests 5 representative tools via the stdio JSON-RPC protocol, using the
MCPClient from test_mcp_protocol.py, without a real LLM.  Covers:
  - arch_lookup_peaks        (arch.lookup_peaks)
  - bottleneck_classify_from_metrics
  - sol_sanity_check         (anti-Sakana: 1e20 FLOPS → plausible=false)
  - intent_classify
  - counters_lookup_info

Each test asserts structural correctness of the JSON response and key values.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Helpers — copied from test_mcp_protocol.py pattern
# ---------------------------------------------------------------------------

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
    """Minimal MCP JSON-RPC client for testing — duplicated from test_mcp_protocol."""

    def __init__(self, proc):
        self.proc = proc
        self._next_id = 1

    def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        self._next_id += 1

        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line.encode("utf-8"))
        self.proc.stdin.flush()

        response_line = self.proc.stdout.readline().decode("utf-8").strip()
        if not response_line:
            return {"error": "no response from server"}
        try:
            return json.loads(response_line)
        except json.JSONDecodeError as e:
            return {"error": f"invalid JSON: {e}", "raw": response_line}

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=2)


def _initialize(client: MCPClient) -> None:
    """Send the MCP initialize handshake."""
    client.send_request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "perfxpert-e2e-coverage", "version": "1.0"},
        },
    )


def _call_tool(client: MCPClient, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Send tools/call and return the parsed result dict (or None on error)."""
    response = client.send_request(
        "tools/call",
        {"name": tool_name, "arguments": arguments},
    )
    assert "result" in response, (
        f"tools/call {tool_name!r}: expected 'result' key, got: {response}"
    )
    content_list = response["result"].get("content", [])
    assert len(content_list) > 0, f"tools/call {tool_name!r}: empty content"
    first = content_list[0]
    assert "text" in first, f"tools/call {tool_name!r}: content[0] missing 'text'"
    return json.loads(first["text"])


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp_proc():
    """Start a fresh perfxpert-mcp process for the test."""
    mcp_path = _perfxpert_mcp_path()
    proc = subprocess.Popen(
        [str(mcp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    yield proc
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ---------------------------------------------------------------------------
# Finding #25 coverage tests — 5 representative tools
# ---------------------------------------------------------------------------

def test_mcp_coverage_arch_lookup_peaks(mcp_proc):
    """arch_lookup_peaks: gfx942 → peak_fp64_tflops == 81.7 (MI300X spec).

    This is the canonical anti-Sakana calibration value.
    """
    client = MCPClient(mcp_proc)
    try:
        _initialize(client)
        result = _call_tool(client, "arch_lookup_peaks", {"gfx_id": "gfx942"})
        assert "peak_fp64_tflops" in result, (
            f"arch_lookup_peaks response missing 'peak_fp64_tflops': {result}"
        )
        assert result["peak_fp64_tflops"] == pytest.approx(81.7, rel=0.01), (
            f"MI300X FP64 peak should be 81.7 TFLOPS, got {result['peak_fp64_tflops']}"
        )
    finally:
        client.close()


def test_mcp_coverage_bottleneck_classify_compute(mcp_proc):
    """bottleneck_classify_from_metrics: high VALU util + AI above ridge → type 'compute'.

    Part D (Finding #26) introduced evidence-strength weighting: a single rule
    matching (1/3 = 0.333) now scores "mixed", not "compute".  To get "compute"
    we need at least 2/3 compute rules to pass (score ≥ 0.5).  Provide both
    valu_util_pct and arithmetic_intensity_above_ridge so the classifier has
    sufficient evidence (2/3 rules evaluated and passing → score = 0.667).
    """
    client = MCPClient(mcp_proc)
    try:
        _initialize(client)
        result = _call_tool(
            client,
            "bottleneck_classify_from_metrics",
            {"metrics": {"valu_util_pct": 0.85, "arithmetic_intensity_above_ridge": 1}},
        )
        assert "type" in result, (
            f"bottleneck_classify_from_metrics missing 'type': {result}"
        )
        assert result["type"] == "compute", (
            f"valu_util_pct=0.85 + AI above ridge should classify as 'compute', got {result['type']!r}"
        )
    finally:
        client.close()


def test_mcp_coverage_sol_sanity_check_implausible(mcp_proc):
    """sol_sanity_check: 1e20 FLOPS/s fp32 → plausible=false (anti-Sakana).

    No physical GPU achieves 1e20 FLOPS/s; any claimed speedup requiring
    that must be rejected.
    """
    client = MCPClient(mcp_proc)
    try:
        _initialize(client)
        result = _call_tool(
            client,
            "sol_sanity_check",
            {
                "achieved_flops_per_sec": 1e20,
                "kernel_type": "fp32",
                "gfx_id": "gfx942",
            },
        )
        assert "plausible" in result, (
            f"sol_sanity_check response missing 'plausible': {result}"
        )
        assert result["plausible"] is False, (
            f"1e20 FLOPS/s fp32 must be implausible for gfx942, got plausible={result['plausible']}"
        )
    finally:
        client.close()


def test_mcp_coverage_intent_classify_analyze(mcp_proc):
    """intent_classify: 'why is my kernel slow' → intent='analyze'.

    Diagnostic questions must route to the analysis agent.
    """
    client = MCPClient(mcp_proc)
    try:
        _initialize(client)
        result = _call_tool(
            client,
            "intent_classify",
            {"user_query": "why is my kernel slow"},
        )
        assert "intent" in result, (
            f"intent_classify response missing 'intent': {result}"
        )
        # Diagnostic phrasing → analyze (or explain — both route to analysis)
        assert result["intent"] in ("analyze", "explain"), (
            f"'why is my kernel slow' should map to 'analyze'/'explain', "
            f"got {result['intent']!r}"
        )
    finally:
        client.close()


def test_mcp_coverage_counters_lookup_info_sq_waves(mcp_proc):
    """counters_lookup_info: SQ_WAVES → block='SQ'.

    SQ_WAVES is in the SQ block and is a fundamental occupancy counter.
    """
    client = MCPClient(mcp_proc)
    try:
        _initialize(client)
        result = _call_tool(
            client,
            "counters_lookup_info",
            {"name": "SQ_WAVES"},
        )
        assert "block" in result, (
            f"counters_lookup_info(SQ_WAVES) response missing 'block': {result}"
        )
        assert result["block"] == "SQ", (
            f"SQ_WAVES should be in block 'SQ', got {result['block']!r}"
        )
        assert "name" in result and result["name"] == "SQ_WAVES", (
            f"Response name should be 'SQ_WAVES', got {result.get('name')!r}"
        )
    finally:
        client.close()
