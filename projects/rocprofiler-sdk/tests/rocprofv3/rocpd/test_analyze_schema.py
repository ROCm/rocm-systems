#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
Tests for the AI analysis JSON schema (analysis-output.schema.json).

Validates:
  - The schema file is present, parseable, and structurally correct.
  - rocpd analyze --format json output conforms to the schema.
  - Recommendations contain the structured commands array.
"""

import json
import sys
import importlib.resources as pkg_resources

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "0.1.0"

REQUIRED_TOP_LEVEL = [
    "schema_version",
    "metadata",
    "profiling_info",
    "summary",
    "execution_breakdown",
    "hotspots",
    "memory_analysis",
    "hardware_counters",
    "recommendations",
    "warnings",
    "errors",
]

COMMAND_TOOLS = {"rocprofv3", "rocprof-sys", "rocprof-compute"}


def _load_schema():
    """Load the schema JSON from the installed package."""
    schema_text = (
        pkg_resources.files("rocpd.ai_analysis")
        .joinpath("docs/analysis-output.schema.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(schema_text)


def _make_synthetic_json_output():
    """Generate a minimal JSON analysis document using the public API."""
    from rocpd.analyze import format_analysis_output, generate_recommendations

    # Keys must match what compute_time_breakdown() actually returns.
    time_breakdown = {
        "kernel_percent": 50.0,
        "memcpy_percent": 30.0,
        "overhead_percent": 15.0,
        "total_runtime": 100_000_000,
        "total_kernel_time": 50_000_000,
        "total_memcpy_time": 30_000_000,
    }
    hotspots = [
        {
            "name": "test_kernel",
            "total_duration": 45_000_000,
            "calls": 10,  # matches identify_hotspots() key (COUNT(*) as calls)
            "avg_duration": 4_500_000,
            "min_duration": 4_000_000,
            "max_duration": 5_000_000,
        }
    ]
    # Keys must match the hyphenated direction strings produced by analyze_memory_copies()
    memory_analysis = {
        "Host-to-Device": {
            "count": 5,
            "size": 1024,
            "total_duration": 30_000_000,
            "bandwidth_bytes_per_sec": 1e9,
        }
    }
    recommendations = generate_recommendations(
        time_breakdown, hotspots, memory_analysis
    )
    output = format_analysis_output(
        time_breakdown,
        hotspots,
        memory_analysis,
        recommendations,
        output_format="json",
    )
    return json.loads(output)


# ---------------------------------------------------------------------------
# Schema file tests
# ---------------------------------------------------------------------------


def test_schema_file_is_readable():
    """Schema file can be located and read through the package."""
    text = (
        pkg_resources.files("rocpd.ai_analysis")
        .joinpath("docs/analysis-output.schema.json")
        .read_text(encoding="utf-8")
    )
    assert len(text) > 0, "Schema file is empty"


def test_schema_file_is_valid_json():
    """Schema file is valid JSON."""
    schema = _load_schema()
    assert isinstance(schema, dict), "Schema root must be a JSON object"


def test_schema_file_has_json_schema_keyword():
    """Schema file declares a JSON Schema dialect."""
    from urllib.parse import urlparse

    schema = _load_schema()
    assert "$schema" in schema, "Schema must contain $schema keyword"
    parsed = urlparse(schema["$schema"])
    assert (
        parsed.netloc == "json-schema.org"
    ), f"$schema must point to json-schema.org, got netloc={parsed.netloc!r}"


def test_schema_file_version_const():
    """schema_version property enum includes SCHEMA_VERSION."""
    schema = _load_schema()
    props = schema.get("properties", {})
    assert "schema_version" in props, "schema_version must be in properties"
    enum_vals = props["schema_version"].get("enum", [])
    assert SCHEMA_VERSION in enum_vals, (
        f"schema_version enum must include {SCHEMA_VERSION!r}, " f"got {enum_vals!r}"
    )


def test_schema_file_required_fields():
    """Schema requires all expected top-level fields."""
    schema = _load_schema()
    required = schema.get("required", [])
    for field in REQUIRED_TOP_LEVEL:
        assert field in required, f"Required field missing from schema: {field!r}"


def test_schema_file_defines_recommendation_command():
    """Schema $defs contains a recommendation_command definition."""
    schema = _load_schema()
    defs = schema.get("$defs", {})
    assert "recommendation_command" in defs, "$defs must define recommendation_command"
    cmd_def = defs["recommendation_command"]
    required_cmd_fields = {"tool", "description", "flags", "args", "full_command"}
    defined = set(cmd_def.get("properties", {}).keys())
    missing = required_cmd_fields - defined
    assert not missing, f"recommendation_command missing properties: {missing}"


def test_schema_file_tool_enum():
    """recommendation_command.tool is an enum of the three ROCm tools."""
    schema = _load_schema()
    cmd_props = schema["$defs"]["recommendation_command"]["properties"]
    tool_enum = set(cmd_props["tool"].get("enum", []))
    assert (
        tool_enum == COMMAND_TOOLS
    ), f"tool enum must be {COMMAND_TOOLS}, got {tool_enum}"


# ---------------------------------------------------------------------------
# JSON output conformance tests (using synthetic data)
# ---------------------------------------------------------------------------


def test_json_output_schema_version():
    """format_analysis_output JSON output carries correct schema_version."""
    doc = _make_synthetic_json_output()
    assert (
        doc.get("schema_version") == SCHEMA_VERSION
    ), f"Expected schema_version {SCHEMA_VERSION!r}, got {doc.get('schema_version')!r}"


def test_json_output_required_fields_present():
    """All required top-level fields are present in JSON output."""
    doc = _make_synthetic_json_output()
    for field in REQUIRED_TOP_LEVEL:
        assert field in doc, f"Required field missing from JSON output: {field!r}"


def test_json_output_metadata_fields():
    """metadata object contains expected sub-fields."""
    doc = _make_synthetic_json_output()
    meta = doc["metadata"]
    for field in (
        "rocpd_version",
        "analysis_version",
        "database_file",
        "analysis_timestamp",
    ):
        assert field in meta, f"metadata missing field: {field!r}"
    assert meta["analysis_version"] == SCHEMA_VERSION


def test_json_output_hardware_counters_has_flag():
    """hardware_counters always contains has_counters boolean."""
    doc = _make_synthetic_json_output()
    hw = doc["hardware_counters"]
    assert "has_counters" in hw, "hardware_counters must have has_counters"
    assert isinstance(hw["has_counters"], bool)


def test_json_output_recommendations_are_list():
    """recommendations is a list."""
    doc = _make_synthetic_json_output()
    assert isinstance(doc["recommendations"], list)


def test_json_output_recommendation_required_fields():
    """Each recommendation has required fields: id, priority, category, issue, suggestion."""
    doc = _make_synthetic_json_output()
    for i, rec in enumerate(doc["recommendations"]):
        for field in ("id", "priority", "category", "issue", "suggestion"):
            assert field in rec, f"recommendations[{i}] missing field {field!r}"
        assert rec["priority"] in (
            "HIGH",
            "MEDIUM",
            "LOW",
            "INFO",
        ), f"recommendations[{i}] has invalid priority {rec['priority']!r}"


def test_json_output_recommendations_have_commands():
    """Recommendations include a commands array."""
    doc = _make_synthetic_json_output()
    recs_with_commands = [r for r in doc["recommendations"] if r.get("commands")]
    assert (
        len(recs_with_commands) > 0
    ), "At least one recommendation must have a non-empty commands array"


def test_json_output_command_structure():
    """Each command object has all required fields with correct types."""
    doc = _make_synthetic_json_output()
    for i, rec in enumerate(doc["recommendations"]):
        for j, cmd in enumerate(rec.get("commands", [])):
            loc = f"recommendations[{i}].commands[{j}]"
            assert "tool" in cmd, f"{loc} missing 'tool'"
            assert "description" in cmd, f"{loc} missing 'description'"
            assert "flags" in cmd, f"{loc} missing 'flags'"
            assert "args" in cmd, f"{loc} missing 'args'"
            assert "full_command" in cmd, f"{loc} missing 'full_command'"
            assert (
                cmd["tool"] in COMMAND_TOOLS
            ), f"{loc} tool {cmd['tool']!r} not in {COMMAND_TOOLS}"
            assert isinstance(cmd["flags"], list), f"{loc} flags must be a list"
            assert isinstance(cmd["args"], list), f"{loc} args must be a list"
            assert isinstance(
                cmd["full_command"], str
            ), f"{loc} full_command must be a string"
            assert (
                cmd["tool"] in cmd["full_command"]
            ), f"{loc} full_command must start with tool name"


def test_json_output_command_args_structure():
    """Each arg in commands.args has name and value fields."""
    doc = _make_synthetic_json_output()
    for i, rec in enumerate(doc["recommendations"]):
        for j, cmd in enumerate(rec.get("commands", [])):
            for k, arg in enumerate(cmd.get("args", [])):
                loc = f"recommendations[{i}].commands[{j}].args[{k}]"
                assert "name" in arg, f"{loc} missing 'name'"
                assert "value" in arg, f"{loc} missing 'value'"
                assert isinstance(arg["name"], str), f"{loc} name must be a string"
                # value may be str or None
                assert arg["value"] is None or isinstance(
                    arg["value"], str
                ), f"{loc} value must be str or null"


def test_json_output_validates_against_schema():
    """JSON output passes jsonschema validation against analysis-output.schema.json."""
    jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
    schema = _load_schema()
    doc = _make_synthetic_json_output()
    try:
        jsonschema.validate(instance=doc, schema=schema)
    except jsonschema.ValidationError as exc:
        pytest.fail(f"JSON output failed schema validation: {exc.message}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use --noconftest to avoid loading conftest.py which requires rocprofiler_sdk module
    exit_code = pytest.main(["--noconftest", "-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
