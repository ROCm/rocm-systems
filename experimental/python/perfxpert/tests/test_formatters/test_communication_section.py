"""Formatter parity tests for the Phase 10 Communication section."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from perfxpert.formatters import format_analysis_output
from perfxpert.formatters.json_fmt import _format_as_json
from perfxpert.formatters.markdown import _format_as_markdown
from perfxpert.formatters.webview import _format_as_webview


_NANOSECONDS_PER_MILLISECOND = 1_000_000


def _duration_ns(milliseconds: int) -> int:
    return milliseconds * _NANOSECONDS_PER_MILLISECOND


def _observed_duration_ns(collectives: list[Dict[str, Any]]) -> int:
    return sum(int(c.get("duration_ns", 0) or 0) for c in collectives)


@pytest.fixture
def comm_payload() -> Dict[str, Any]:
    """Small communication dict matching the rccl_analysis.py contract."""
    collectives = [
        {
            "op_type": "AllReduce",
            "msg_bytes": len("measured-message"),
            "duration_ns": _duration_ns(len("measured-kernel")),
        }
    ]
    return {
        "collectives": collectives,
        "summary": {
            "op_count": len(collectives),
            "dominant_op": "AllReduce",
            "capture_incomplete": False,
        },
    }


@pytest.fixture
def incomplete_comm_payload() -> Dict[str, Any]:
    """Fallback RCCL payload with kernel hits but no measured byte metrics."""
    collectives = [
        {
            "op_type": "Broadcast",
            "duration_ns": _duration_ns(len("first-fallback-kernel")),
        },
        {
            "op_type": "Broadcast",
            "duration_ns": _duration_ns(len("second-fallback-kernel")),
        },
    ]
    return {
        "collectives": collectives,
        "summary": {
            "op_count": len(collectives),
            "dominant_op": "Broadcast",
            "capture_incomplete": True,
            "measured_metrics_available": False,
            "fallback_kernel_count": len(collectives),
            "observed_total_duration_ns": _observed_duration_ns(collectives),
            "incomplete_reason": (
                "RCCL kernels were detected, but RCCL API region arguments "
                "were not captured; message size, bus bandwidth, and "
                "efficiency are unavailable."
            ),
        },
        "capture_incomplete": True,
    }


def _base_args() -> Dict[str, Any]:
    return {
        "time_breakdown": {},
        "hotspots": [],
        "memory_analysis": {},
        "recommendations": [],
        "hardware_counters": None,
        "database_path": "/tmp/fake.db",
    }


def _assert_observed_kernel_time_line(text: str) -> None:
    line = next(
        (line for line in text.splitlines() if "Observed RCCL kernel time:" in line),
        "",
    )
    assert line
    assert " ms" in line


# --------------------------------------------------------------------------- #
# Webview: section exists + class="scard" present                             #
# --------------------------------------------------------------------------- #

def test_webview_renders_communication_scard(comm_payload):
    args = _base_args()
    html = _format_as_webview(communication=comm_payload, **args)
    assert "<h2>Communication</h2>" in html
    assert 'class="scard"' in html
    assert "AllReduce" in html
    # busBW bar class reused from the existing bar CSS.
    assert 'class="btrack"' in html or "btrack" in html


# --------------------------------------------------------------------------- #
# Markdown: ## Communication + table header                                   #
# --------------------------------------------------------------------------- #

def test_markdown_renders_communication_section(comm_payload):
    args = _base_args()
    md = _format_as_markdown(communication=comm_payload, **args)
    assert "## Communication" in md
    assert "| Op |" in md
    assert "AllReduce" in md
    assert "Bus BW" in md


# --------------------------------------------------------------------------- #
# Text (via format_analysis_output): COMMUNICATION header box                 #
# --------------------------------------------------------------------------- #

def test_text_renders_communication_box(comm_payload):
    args = _base_args()
    out = format_analysis_output(
        output_format="text",
        communication=comm_payload,
        **args,
    )
    assert "COMMUNICATION" in out
    assert "AllReduce" in out


def test_text_incomplete_capture_hides_zero_metric_table(incomplete_comm_payload):
    args = _base_args()
    out = format_analysis_output(
        output_format="text",
        communication=incomplete_comm_payload,
        **args,
    )
    assert "COMMUNICATION" in out
    assert "RCCL kernel fallback hit(s)" in out
    assert "Capture incomplete" in out
    _assert_observed_kernel_time_line(out)
    assert "Fallback evidence" in out
    assert "metric table hidden" in out
    assert "Bus BW" not in out


def test_markdown_incomplete_capture_hides_zero_metric_table(incomplete_comm_payload):
    args = _base_args()
    md = _format_as_markdown(communication=incomplete_comm_payload, **args)
    assert "## Communication" in md
    assert "Capture incomplete" in md
    assert "zero-byte placeholder rows" in md
    assert "| Op | Observed duration |" in md
    assert "Bus BW" not in md


def test_webview_incomplete_capture_hides_zero_metric_table(incomplete_comm_payload):
    args = _base_args()
    html = _format_as_webview(communication=incomplete_comm_payload, **args)
    assert "<h2>Communication</h2>" in html
    assert "Capture incomplete" in html
    assert "Observed duration" in html
    assert "placeholder rows" in html
    assert "Bus BW (vs peak)" not in html


def test_webview_zero_metric_api_capture_is_not_labeled_fallback():
    args = _base_args()
    communication = {
        "collectives": [
            {
                "op_type": "AllReduce",
                "efficiency_label": "unknown",
                "algo_hint": "direct",
                "topology_hint": "intra-node",
                "regime": "latency-bound",
            }
        ],
        "summary": {
            "dominant_op": "AllReduce",
            "capture_incomplete": False,
            "measured_metrics_available": False,
        },
        "capture_incomplete": False,
    }

    html = _format_as_webview(communication=communication, **args)

    assert "Metrics unavailable" in html
    assert "RCCL API regions were captured" in html
    assert "Capture incomplete" not in html
    assert "Fallback hits" not in html


def test_summary_only_incomplete_capture_uses_fallback_kernel_count():
    args = _base_args()
    fallback_kernel_names = ("broadcast", "reduce", "barrier")
    communication = {
        "collectives": [],
        "summary": {
            "capture_incomplete": True,
            "measured_metrics_available": False,
            "fallback_kernel_count": len(fallback_kernel_names),
            "dominant_op": "Broadcast",
        },
        "capture_incomplete": True,
    }

    text = format_analysis_output(
        output_format="text",
        communication=communication,
        **args,
    )
    md = _format_as_markdown(communication=communication, **args)
    html = _format_as_webview(communication=communication, **args)

    assert "RCCL kernel fallback hit(s)" in text
    assert "RCCL kernel fallback hit(s)" in md
    assert "Fallback hits" in html


# --------------------------------------------------------------------------- #
# JSON: passthrough + schema bump                                             #
# --------------------------------------------------------------------------- #

def test_json_passthrough_bumps_schema(comm_payload):
    import json as _json
    args = _base_args()
    out = _format_as_json(communication=comm_payload, **args)
    doc = _json.loads(out)
    assert "communication" in doc
    assert doc["communication"] == comm_payload
    assert doc["schema_version"] == "0.3.2"


def test_json_att_trumps_communication(comm_payload):
    """Even with communication present, ATT data bumps schema to 0.4.0."""
    import json as _json
    args = _base_args()
    att = {"has_att_data": True, "kernels": [], "summary": {}}
    out = _format_as_json(communication=comm_payload, att_analysis=att, **args)
    doc = _json.loads(out)
    assert doc["schema_version"] == "0.4.0"


def test_json_includes_summary_only_incomplete_capture():
    import json as _json
    args = _base_args()
    communication = {
        "collectives": [],
        "summary": {
            "capture_incomplete": True,
            "measured_metrics_available": False,
        },
        "capture_incomplete": True,
    }
    out = _format_as_json(communication=communication, **args)
    doc = _json.loads(out)
    assert doc["schema_version"] == "0.3.2"
    assert doc["communication"] == communication


# --------------------------------------------------------------------------- #
# Absent key: no section rendered                                             #
# --------------------------------------------------------------------------- #

def test_formatters_skip_communication_when_absent():
    args = _base_args()
    html = _format_as_webview(**args)
    md = _format_as_markdown(**args)
    import json as _json
    doc = _json.loads(_format_as_json(**args))
    assert "<h2>Communication</h2>" not in html
    assert "## Communication" not in md
    assert "communication" not in doc
