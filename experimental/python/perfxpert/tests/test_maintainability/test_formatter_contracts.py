###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Golden-style formatter contracts used before splitting large renderers."""

from __future__ import annotations

import json

from perfxpert.formatters import format_analysis_output


TIME_BREAKDOWN = {
    "total_runtime": 2_000_000,
    "total_kernel_time": 1_200_000,
    "total_memcpy_time": 400_000,
    "kernel_percent": 60.0,
    "memcpy_percent": 20.0,
    "overhead_percent": 20.0,
}
HOTSPOTS = [
    {
        "name": "vector_add",
        "calls": 4,
        "total_duration": 1_200_000,
        "avg_duration": 300_000,
        "percent_of_total": 100.0,
    }
]
MEMORY_ANALYSIS = {
    "Host-to-Device": {
        "count": 2,
        "total_bytes": 4096,
        "total_duration": 400_000,
        "avg_bytes": 2048,
        "bandwidth_bytes_per_sec": 10_240_000,
    }
}
RECOMMENDATIONS = [
    {
        "priority": "MEDIUM",
        "category": "Memory Transfer",
        "issue": "Host-to-device copies are visible in the profile.",
        "suggestion": "Batch transfers and use async copies where possible.",
        "actions": ["Group small transfers into larger batches."],
    }
]
HARDWARE_COUNTERS = {"has_counters": False, "metrics": {}, "counters": {}}


def _render(output_format: str) -> str:
    return format_analysis_output(
        time_breakdown=TIME_BREAKDOWN,
        hotspots=HOTSPOTS,
        memory_analysis=MEMORY_ANALYSIS,
        recommendations=RECOMMENDATIONS,
        hardware_counters=HARDWARE_COUNTERS,
        database_path="sample.db",
        output_format=output_format,
    )


def test_text_formatter_section_contract():
    output = _render("text")

    assert output.index("TIME BREAKDOWN") < output.index("HOTSPOTS")
    assert output.index("HOTSPOTS") < output.index("MEMORY COPY ANALYSIS")
    assert output.index("MEMORY COPY ANALYSIS") < output.index("RECOMMENDATIONS")
    assert "vector_add" in output
    assert "Host-to-Device" in output


def test_markdown_formatter_section_contract():
    output = _render("markdown")

    assert output.index("## Time Breakdown") < output.index("## Top Kernel Hotspots")
    assert output.index("## Top Kernel Hotspots") < output.index("## Memory Copy Analysis")
    assert output.index("## Memory Copy Analysis") < output.index("## Recommendations")
    assert "vector_add" in output
    assert "Host-to-Device" in output


def test_webview_formatter_section_contract():
    output = _render("webview")

    assert output.count("<!DOCTYPE html>") == 1
    assert output.index("<h2>Execution Breakdown</h2>") < output.index("<h2>Top Kernel Hotspots</h2>")
    assert output.index("<h2>Top Kernel Hotspots</h2>") < output.index("<h2>Memory Transfer Analysis</h2>")
    assert output.index("<h2>Memory Transfer Analysis</h2>") < output.index("<h2>Optimization Recommendations</h2>")
    assert "vector_add" in output
    assert "Host-to-Device" in output


def test_json_formatter_section_contract():
    document = json.loads(_render("json"))

    for key in (
        "execution_breakdown",
        "hotspots",
        "memory_analysis",
        "recommendations",
        "hardware_counters",
        "warnings",
        "metadata",
    ):
        assert key in document
    assert document["hotspots"][0]["name"] == "vector_add"
