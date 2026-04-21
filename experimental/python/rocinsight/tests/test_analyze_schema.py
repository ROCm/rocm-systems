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
  - rocinsight analyze --format json output conforms to the schema.
  - Recommendations contain the structured commands array.
"""

import json
import os
import sys
import tempfile

try:
    import importlib.resources as pkg_resources
except ImportError:  # Python 3.6
    import pkgutil as _pkgutil

    class pkg_resources:  # type: ignore[no-redef]
        """Minimal shim so _load_schema() works on Python 3.6."""

        class _Traversable:
            def __init__(self, package, resource):
                self._package = package
                self._resource = resource

            def read_text(self, encoding="utf-8"):
                data = _pkgutil.get_data(self._package, self._resource)
                return data.decode(encoding) if data is not None else ""

        class _Package:
            def __init__(self, package):
                self._package = package

            def joinpath(self, resource):
                return pkg_resources._Traversable(self._package, resource)

        @staticmethod
        def files(package):
            return pkg_resources._Package(package)


import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# The version emitted by Tier 1/2 analysis (no TraceLens fields).
# This constant is only used to verify the schema enum includes the Tier 1/2 version;
# conformance tests derive allowed versions from the loaded schema enum directly.
TIER12_SCHEMA_VERSION = "0.1.0"

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
        pkg_resources.files("rocinsight.ai_analysis")
        .joinpath("docs/analysis-output.schema.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(schema_text)


def _make_synthetic_json_output():
    """Generate a minimal JSON analysis document using the public API."""
    from rocinsight.analyze import format_analysis_output, generate_recommendations

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
    # Keys must match the actual return shape of analyze_memory_copies():
    # count, total_bytes, total_duration, avg_bytes, avg_duration, bandwidth_bytes_per_sec
    memory_analysis = {
        "Host-to-Device": {
            "count": 5,
            "total_bytes": 5120,
            "total_duration": 30_000_000,
            "avg_bytes": 1024.0,
            "avg_duration": 6_000_000.0,
            "bandwidth_bytes_per_sec": 1e9,
        }
    }
    recommendations = generate_recommendations(time_breakdown, hotspots, memory_analysis)
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
        pkg_resources.files("rocinsight.ai_analysis")
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


def test_schema_file_version_enum():
    """schema_version property enum includes the Tier 1/2 version (0.1.0)."""
    schema = _load_schema()
    props = schema.get("properties", {})
    assert "schema_version" in props, "schema_version must be in properties"
    enum_vals = props["schema_version"].get("enum", [])
    assert TIER12_SCHEMA_VERSION in enum_vals, (
        f"schema_version enum must include {TIER12_SCHEMA_VERSION!r}, "
        f"got {enum_vals!r}"
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
    """format_analysis_output JSON output carries a schema_version in the allowed enum."""
    schema = _load_schema()
    allowed = schema["properties"]["schema_version"]["enum"]
    doc = _make_synthetic_json_output()
    assert (
        doc.get("schema_version") in allowed
    ), f"schema_version {doc.get('schema_version')!r} not in allowed enum {allowed}"


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
    schema = _load_schema()
    allowed = schema["properties"]["schema_version"]["enum"]
    assert (
        meta["analysis_version"] in allowed
    ), f"metadata.analysis_version {meta['analysis_version']!r} not in allowed enum {allowed}"


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
# Tier 0 (source-only) JSON output helpers
# ---------------------------------------------------------------------------

_MINIMAL_HIP_SOURCE = """\
__global__ void my_kernel(float* x) { *x = 1.0f; }
void run() {
    hipLaunchKernelGGL(my_kernel, dim3(1), dim3(64), 0, 0, nullptr);
    hipMemcpy(nullptr, nullptr, 0, hipMemcpyHostToDevice);
}
"""

TIER0_SCHEMA_VERSION = "0.2.0"


def _make_synthetic_tier0_json_output():
    """Generate a Tier 0 (source-only) JSON document via format_analysis_output."""
    from rocinsight.analyze import analyze_source_code, format_analysis_output

    with tempfile.TemporaryDirectory() as tmpdir:
        hip_file = os.path.join(tmpdir, "test.cpp")
        with open(hip_file, "w") as fh:
            fh.write(_MINIMAL_HIP_SOURCE)

        tier0_result = analyze_source_code(tmpdir)
        output = format_analysis_output(
            {},
            [],
            {},
            [],
            output_format="json",
            tier0_result=tier0_result,
            source_only=True,
        )
    return json.loads(output)


def _make_synthetic_combined_json_output():
    """Generate a combined (Tier 0 + Tier 1/2) JSON document."""
    from rocinsight.analyze import (
        analyze_source_code,
        format_analysis_output,
        generate_recommendations,
    )

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
            "calls": 10,
            "avg_duration": 4_500_000,
            "min_duration": 4_000_000,
            "max_duration": 5_000_000,
        }
    ]
    memory_analysis = {
        "Host-to-Device": {
            "count": 5,
            "total_bytes": 5120,
            "total_duration": 30_000_000,
            "avg_bytes": 1024.0,
            "avg_duration": 6_000_000.0,
            "bandwidth_bytes_per_sec": 1e9,
        }
    }
    recommendations = generate_recommendations(time_breakdown, hotspots, memory_analysis)

    with tempfile.TemporaryDirectory() as tmpdir:
        hip_file = os.path.join(tmpdir, "test.cpp")
        with open(hip_file, "w") as fh:
            fh.write(_MINIMAL_HIP_SOURCE)

        tier0_result = analyze_source_code(tmpdir)
        output = format_analysis_output(
            time_breakdown,
            hotspots,
            memory_analysis,
            recommendations,
            output_format="json",
            tier0_result=tier0_result,
            source_only=False,
        )
    return json.loads(output)


# ---------------------------------------------------------------------------
# Tier 0 (source-only) schema conformance tests
# ---------------------------------------------------------------------------


def test_tier0_json_output_schema_version():
    """Tier 0 JSON output has schema_version in the allowed enum."""
    schema = _load_schema()
    allowed = schema["properties"]["schema_version"]["enum"]
    doc = _make_synthetic_tier0_json_output()
    assert (
        doc.get("schema_version") in allowed
    ), f"tier0 schema_version {doc.get('schema_version')!r} not in allowed enum {allowed}"
    assert (
        doc.get("schema_version") == TIER0_SCHEMA_VERSION
    ), f"tier0 schema_version should be {TIER0_SCHEMA_VERSION!r}"


def test_tier0_json_output_required_fields_present():
    """All required top-level fields are present in Tier 0 JSON output."""
    doc = _make_synthetic_tier0_json_output()
    for field in REQUIRED_TOP_LEVEL:
        assert field in doc, f"Tier 0 JSON missing required field: {field!r}"


def test_tier0_json_output_execution_breakdown_is_null():
    """execution_breakdown is null in source-only (Tier 0) output."""
    doc = _make_synthetic_tier0_json_output()
    assert (
        doc["execution_breakdown"] is None
    ), "execution_breakdown must be null in Tier 0 source-only output"


def test_tier0_json_output_profiling_mode_is_source_only():
    """profiling_info.profiling_mode is 'source_only' in Tier 0 output."""
    doc = _make_synthetic_tier0_json_output()
    assert (
        doc["profiling_info"]["profiling_mode"] == "source_only"
    ), "Tier 0 profiling_mode must be 'source_only'"


def test_tier0_json_output_analysis_tier_is_zero():
    """profiling_info.analysis_tier is 0 in Tier 0 source-only output."""
    doc = _make_synthetic_tier0_json_output()
    assert doc["profiling_info"]["analysis_tier"] == 0, "Tier 0 analysis_tier must be 0"


def test_tier0_json_output_has_tier0_field():
    """Tier 0 JSON output includes a top-level 'tier0' object."""
    doc = _make_synthetic_tier0_json_output()
    assert "tier0" in doc, "Tier 0 JSON output must include a 'tier0' field"
    tier0 = doc["tier0"]
    assert isinstance(tier0, dict), "'tier0' must be a JSON object"
    for field in ("source_dir", "programming_model", "files_scanned", "kernel_count"):
        assert field in tier0, f"tier0 missing field {field!r}"


def test_tier0_json_output_validates_against_schema():
    """Tier 0 JSON output passes jsonschema validation."""
    jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
    schema = _load_schema()
    doc = _make_synthetic_tier0_json_output()
    try:
        jsonschema.validate(instance=doc, schema=schema)
    except jsonschema.ValidationError as exc:
        pytest.fail(f"Tier 0 JSON failed schema validation: {exc.message}")


# ---------------------------------------------------------------------------
# Combined (Tier 0 + Tier 1/2) schema conformance tests
# ---------------------------------------------------------------------------


def test_combined_json_output_has_tier0_field():
    """Combined (Tier 0 + Tier 1/2) JSON output includes a top-level 'tier0' object."""
    doc = _make_synthetic_combined_json_output()
    assert "tier0" in doc, "Combined JSON output must include a 'tier0' field"
    assert isinstance(doc["tier0"], dict), "'tier0' must be a JSON object"


def test_combined_json_output_tier12_required_fields_present():
    """Combined JSON output has all required Tier 1/2 top-level fields."""
    doc = _make_synthetic_combined_json_output()
    for field in REQUIRED_TOP_LEVEL:
        assert field in doc, f"Combined JSON missing required field: {field!r}"


def test_combined_json_output_execution_breakdown_not_null():
    """execution_breakdown is non-null in combined (Tier 0 + Tier 1/2) output."""
    doc = _make_synthetic_combined_json_output()
    assert (
        doc["execution_breakdown"] is not None
    ), "execution_breakdown must not be null in combined output"


def test_combined_json_output_validates_against_schema():
    """Combined (Tier 0 + Tier 1/2) JSON output passes jsonschema validation."""
    jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
    schema = _load_schema()
    doc = _make_synthetic_combined_json_output()
    try:
        jsonschema.validate(instance=doc, schema=schema)
    except jsonschema.ValidationError as exc:
        pytest.fail(f"Combined JSON failed schema validation: {exc.message}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use --noconftest to avoid loading conftest.py which requires rocprofiler_sdk module
    exit_code = pytest.main(["--noconftest", "-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
