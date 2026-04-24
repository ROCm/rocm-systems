###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

from __future__ import annotations

import json

from perfxpert import analyze as analyze_mod
from perfxpert.analysis import payload as payload_mod


class _Connection:
    _paths: list[str] = []


def test_build_analysis_payload_records_recoverable_section_warning(monkeypatch):
    def _boom(_connection):
        raise RuntimeError("synthetic SQL failure")

    monkeypatch.setattr(payload_mod, "compute_time_breakdown", _boom)
    monkeypatch.setattr(payload_mod, "identify_hotspots", lambda *_, **__: [])
    monkeypatch.setattr(payload_mod, "analyze_memory_copies", lambda *_: {})
    monkeypatch.setattr(
        payload_mod,
        "analyze_hardware_counters",
        lambda *_: {"has_counters": False, "metrics": {}, "counters": {}},
    )
    monkeypatch.setattr(payload_mod, "analyze_kernel_resources", lambda *_: {})
    monkeypatch.setattr(payload_mod, "analyze_api_overhead", lambda *_: {})
    monkeypatch.setattr(
        payload_mod,
        "detect_warmup_issues",
        lambda *_: {"has_warmup_issues": False, "outliers": []},
    )
    monkeypatch.setattr(payload_mod, "_detect_already_collected", lambda *_: frozenset())
    monkeypatch.setattr(payload_mod, "generate_recommendations", lambda **_: [])

    payload = payload_mod.build_analysis_payload(_Connection())

    assert payload["time_breakdown"] == {}
    assert payload["warnings"] == [
        "time breakdown unavailable (RuntimeError: synthetic SQL failure)"
    ]


def test_agentic_json_output_includes_deterministic_payload_warnings():
    output = analyze_mod._format_agentic_output(
        {
            "narrative": "Deterministic fallback was used.",
            "primary_bottleneck": "mixed",
            "recommendations": [],
            "warnings": ["agent warning"],
            "metadata": {},
        },
        "json",
        database_path="sample.db",
        analysis_payload={
            "time_breakdown": {},
            "hotspots": [],
            "memory_analysis": {},
            "hardware_counters": {"has_counters": False, "metrics": {}, "counters": {}},
            "recommendations_deterministic": [],
            "warnings": ["time breakdown unavailable (RuntimeError: synthetic SQL failure)"],
        },
    )

    document = json.loads(output)
    assert document["warnings"] == [
        "agent warning",
        "time breakdown unavailable (RuntimeError: synthetic SQL failure)",
    ]
