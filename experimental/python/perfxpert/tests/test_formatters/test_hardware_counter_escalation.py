"""Formatter parity tests for hardware-counter escalation guidance."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from perfxpert.formatters import format_analysis_output
from perfxpert.formatters.json_fmt import _format_as_json
from perfxpert.formatters.markdown import _format_as_markdown
from perfxpert.formatters.webview import _format_as_webview
from perfxpert.tools import counters


@pytest.fixture
def hw_counter_payload() -> Dict[str, Any]:
    """Hardware-counter payload carrying multi-pass escalation guidance."""
    result = counters.validate_for_gpu(
        ["SQ_WAVES", "GRBM_COUNT", "FETCH_SIZE", "WRITE_SIZE"],
        gpu_arch="gfx942",
    )
    return {
        "has_counters": False,
        "metrics": {},
        "counters": {},
        "escalation": result["escalation"],
    }


def _base_args() -> Dict[str, Any]:
    return {
        "time_breakdown": {
            "total_runtime": 10_000_000,
            "kernel_percent": 60.0,
            "memcpy_percent": 10.0,
            "overhead_percent": 5.0,
            "total_kernel_time": 6_000_000,
            "total_memcpy_time": 1_000_000,
        },
        "hotspots": [],
        "memory_analysis": {},
        "recommendations": [],
        "database_path": "/tmp/fake.db",
    }


def test_webview_renders_hardware_counter_escalation(hw_counter_payload):
    args = _base_args()
    html = _format_as_webview(hardware_counters=hw_counter_payload, **args)
    assert "Counter Collection Escalation" in html
    assert "pmc: FETCH_SIZE" in html
    assert "rocprofv3 -i pmc_groups.txt -- ./app" in html
    assert "rocprof-compute profile -- ./app" in html


def test_markdown_renders_hardware_counter_escalation(hw_counter_payload):
    args = _base_args()
    md = _format_as_markdown(hardware_counters=hw_counter_payload, **args)
    assert "### Counter Collection Escalation" in md
    assert "pmc: WRITE_SIZE" in md
    assert "```bash" in md
    assert "rocprofv3 -i pmc_groups.txt -- ./app" in md


def test_text_renders_hardware_counter_escalation(hw_counter_payload):
    args = _base_args()
    out = format_analysis_output(
        output_format="text",
        hardware_counters=hw_counter_payload,
        **args,
    )
    assert "Counter Collection Escalation:" in out
    assert "PMC Groups File: pmc_groups.txt" in out
    assert "$ rocprof-compute profile -- ./app" in out
    assert "$ rocprofv3 -i pmc_groups.txt -- ./app" in out


def test_json_preserves_hardware_counter_escalation(hw_counter_payload):
    import json as _json

    args = _base_args()
    doc = _json.loads(_format_as_json(hardware_counters=hw_counter_payload, **args))
    escalation = doc["hardware_counters"]["escalation"]
    assert escalation["pass_count"] == 3
    assert escalation["pmc_groups"] == [
        "pmc: SQ_WAVES GRBM_COUNT",
        "pmc: FETCH_SIZE",
        "pmc: WRITE_SIZE",
    ]
    commands = {cmd["tool"]: cmd["full_command"] for cmd in escalation["commands"]}
    assert commands["rocprof-compute"] == "rocprof-compute profile -- ./app"
    assert commands["rocprofv3"] == "rocprofv3 -i pmc_groups.txt -- ./app"
