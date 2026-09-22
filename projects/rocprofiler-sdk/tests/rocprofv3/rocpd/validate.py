#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sqlite3
import sys
from collections import defaultdict
import pytest


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data,
        json_data,
        ("hip", "marker", "kernel", "memory_copy"),
    )


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(
        otf2_data,
        json_data,
        ("hip", "marker", "kernel", "memory_copy", "memory_allocation"),
    )


def test_otf2_system_tree_node_data(otf2_system_tree_node_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_system_tree_node(
        otf2_system_tree_node_data,
    )


def test_csv_data(csv_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_csv_data(
        csv_data,
        json_data,
        (
            "agent",
            "counter_collection",
            "kernel",
            "memory_allocation",
            "memory_copy",
            "regions",
        ),
    )


def test_counter_collection_fields_match_across_rocpd_csv_json(
    csv_data, json_data, db_input
):
    def aggregate(records):
        result = defaultdict(float)
        for dispatch_id, counter_name, value in records:
            result[(int(dispatch_id), counter_name)] += float(value)
        return dict(result)

    data = json_data["rocprofiler-sdk-tool"]
    counter_names = {
        counter["id"]["handle"]: counter["name"] for counter in data["counters"]
    }
    json_records = []
    for entry in data["callback_records"]["counter_collection"]:
        dispatch_id = entry["dispatch_data"]["dispatch_info"]["dispatch_id"]
        for record in entry["records"]:
            json_records.append(
                (
                    int(dispatch_id),
                    counter_names[record["counter_id"]["handle"]],
                    float(record["value"]),
                )
            )

    counter_csv = next(
        rows for filename, rows in csv_data if "counter_collection" in filename
    )
    csv_records = [
        (
            int(row["Dispatch_Id"]),
            row["Counter_Name"],
            float(row["Counter_Value"]),
        )
        for row in counter_csv
    ]

    with sqlite3.connect(db_input) as connection:
        tables = {
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        event_tables = sorted(
            name for name in tables if name.startswith("rocpd_pmc_event_")
        )
        assert event_tables

        db_records = []
        for event_table in event_tables:
            suffix = event_table.split("rocpd_pmc_event_", 1)[1]
            info_table = "rocpd_info_pmc_{}".format(suffix)
            dispatch_table = "rocpd_kernel_dispatch_{}".format(suffix)
            assert info_table in tables and dispatch_table in tables
            db_records.extend(
                connection.execute(
                    """
                    SELECT dispatch.dispatch_id, info.name, event.value
                    FROM {event} AS event
                    JOIN {info} AS info ON event.pmc_id = info.id
                    JOIN {dispatch} AS dispatch ON event.event_id = dispatch.event_id
                    """.format(
                        event=event_table,
                        info=info_table,
                        dispatch=dispatch_table,
                    )
                )
            )

    json_aggregates = aggregate(json_records)
    assert aggregate(csv_records) == json_aggregates
    assert aggregate(db_records) == json_aggregates
    assert all(value >= 0.0 for value in json_aggregates.values())
    assert any(value > 0.0 for value in json_aggregates.values())


def test_counter_collection_fields_match_perfetto(pftrace_reader, json_data):
    samples = pftrace_reader.query_tp("""
        SELECT counter_track.name AS track_name, counter.ts, counter.value
        FROM counter
        JOIN counter_track ON counter.track_id = counter_track.id
        WHERE counter_track.name GLOB 'Agent * PMC *'
        """)
    assert not samples.empty
    assert (samples["ts"] > 0).all()
    assert (samples["value"] >= 0).all()

    data = json_data["rocprofiler-sdk-tool"]
    counter_names = {
        counter["id"]["handle"]: counter["name"] for counter in data["counters"]
    }
    agent_nodes = {
        agent["id"]["handle"]: agent["logical_node_id"] for agent in data["agents"]
    }
    expected_tracks = {
        "Agent [{}] PMC {}".format(
            agent_nodes[entry["dispatch_data"]["dispatch_info"]["agent_id"]["handle"]],
            counter_names[record["counter_id"]["handle"]],
        )
        for entry in data["callback_records"]["counter_collection"]
        for record in entry["records"]
    }
    actual_tracks = set(samples["track_name"])
    assert expected_tracks == actual_tracks
    maxima = samples.groupby("track_name")["value"].max()
    assert all(maxima[track] > 0 for track in expected_tracks)


def test_arg_annotations_exist(pftrace_reader):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_arg_annotations(pftrace_reader)


def test_event_id_annotations(pftrace_reader):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_event_id_annotations(pftrace_reader)


#########################################################################################
#
# ROCPD Summary Validation
#
#########################################################################################


def _validate_summary_region_category_filtering(
    csv_files, expected_categories=None, allow_none=False
):
    """
    Test that summary output contains ONLY the expected categories.

    Args:
        csv_files: List of CSV file paths to validate
        expected_categories: List of category names that should be present (e.g., ['kernel', 'hip'])
        allow_none: If True, allows no region summaries (for --region-categories NONE test)
    """
    import os

    if not csv_files:
        raise ValueError("No CSV files provided for validation")

    basenames = [os.path.basename(f) for f in csv_files]

    assert len(basenames) > 0, "No summary files provided for validation"

    print(f"\nValidating {len(basenames)} summary files:")
    for name in sorted(basenames):
        print(f"  - {name}")

    # For NONE test: ensure no region-based summaries are generated
    if allow_none:
        # Region summaries have filenames like "rocm_hip_*.csv", "rocm_hsa_*.csv"
        region_files = [f for f in basenames if f.lower().startswith("rocm_")]
        assert len(region_files) == 0, (
            f"--region-categories NONE should not generate region summaries, "
            f"but found: {region_files}"
        )

    # Check that expected categories are present and ONLY those categories exist
    if expected_categories:
        # 1. Check all expected categories are present
        for category in expected_categories:
            category_lower = category.lower()
            matching_files = [f for f in basenames if category_lower in f.lower()]
            assert len(matching_files) > 0, (
                f"Expected category '{category}' not found. "
                f"No files matching '{category_lower}' in {basenames}"
            )

        # 2. Check no unexpected categories exist
        for filename in basenames:
            filename_lower = filename.lower()
            matches_expected = any(
                cat.lower() in filename_lower for cat in expected_categories
            )
            assert matches_expected, (
                f"Unexpected file '{filename}' found. "
                f"Does not match any expected category: {expected_categories}"
            )


def test_summary_region_category_kernel(summary_kernel_csv_files):
    """Test that --region-categories KERNEL only produces kernel summaries."""
    _validate_summary_region_category_filtering(
        summary_kernel_csv_files,
        expected_categories=["kernel"],
    )


def test_summary_region_category_hip(summary_hip_csv_files):
    """Test that --region-categories HIP only produces HIP summaries."""
    _validate_summary_region_category_filtering(
        summary_hip_csv_files,
        expected_categories=["hip"],
    )


def test_summary_region_category_multiple(summary_multiple_csv_files):
    """Test that --region-categories HIP KERNEL produces those summaries."""
    _validate_summary_region_category_filtering(
        summary_multiple_csv_files,
        expected_categories=["hip", "kernel"],
    )


def test_summary_region_category_none(summary_none_csv_files):
    """Test that --region-categories NONE includes views but no regions."""
    _validate_summary_region_category_filtering(
        summary_none_csv_files,
        expected_categories=["kernel", "memory"],
        allow_none=True,
    )


def _kernel_names(df):
    """Extract kernel names from dataframe as list of strings."""
    return [str(name) for name in df["Name"].tolist()]


def _extract_kernel_names_from_json(json_data, name_type="full"):
    """
    Extract kernel names from JSON data (source of truth).

    Extracts names only for kernels that were actually dispatched.

    Args:
        json_data: JSON data from rocprofiler output
        name_type: Type of name to extract - "full", "truncated", or "mangled"

    Returns:
        Set of kernel names from dispatched kernels in JSON
    """
    tool_data = json_data.get("rocprofiler-sdk-tool")
    assert tool_data, "Missing rocprofiler-sdk-tool in JSON"

    kernel_symbols = tool_data.get("kernel_symbols", [])
    assert kernel_symbols, "No kernel_symbols found in JSON"
    kernel_dispatches = tool_data.get("buffer_records", {}).get("kernel_dispatch", [])
    assert kernel_dispatches, "No kernel_dispatch records found in JSON"

    # Map to JSON fields based on what summary.py actually uses:
    # - "full": uses display_name which equals formatted_kernel_name
    # - "truncated": uses truncated_kernel_name
    # - "mangled": uses kernel_name (raw mangled name)
    name_field_map = {
        "mangled": "kernel_name",
        "full": "formatted_kernel_name",
        "truncated": "truncated_kernel_name",
    }

    assert name_type in name_field_map, f"Unknown kernel name type: {name_type}"
    field = name_field_map[name_type]

    # Build lookup table of kernel symbols by ID
    symbols_by_id = {
        symbol["kernel_id"]: symbol
        for symbol in kernel_symbols
        if symbol.get("kernel_id", 0) > 0
    }

    # Extract names only from dispatched kernels
    names = set()
    for dispatch in kernel_dispatches:
        kernel_id = dispatch["dispatch_info"]["kernel_id"]

        assert kernel_id in symbols_by_id, (
            f"kernel_dispatch references kernel_id={kernel_id}, "
            "but no matching kernel symbol was found"
        )

        name = symbols_by_id[kernel_id].get(field)
        assert name, f"kernel_id={kernel_id} does not contain expected field '{field}'"

        names.add(str(name))

    return names


def _verify_names_match_json(csv_df, json_data, name_type):
    """
    Verify CSV kernel names match JSON source exactly.

    Args:
        csv_df: DataFrame from CSV summary
        json_data: Original JSON data (source of truth)
        name_type: "full", "truncated", or "mangled"
    """
    json_kernel_names = _extract_kernel_names_from_json(json_data, name_type)
    csv_kernel_names = set(_kernel_names(csv_df))

    assert json_kernel_names, "No kernel names found in JSON data"
    assert csv_kernel_names, "No kernel names found in CSV data"

    # Verify exact match: CSV and JSON should have the same dispatched kernel names
    missing_in_csv = json_kernel_names - csv_kernel_names
    unexpected_in_csv = csv_kernel_names - json_kernel_names

    assert not missing_in_csv and not unexpected_in_csv, (
        f"CSV kernel names do not match JSON ({name_type} format).\n"
        f"Missing in CSV: {sorted(list(missing_in_csv))[:5]}\n"
        f"Unexpected in CSV: {sorted(list(unexpected_in_csv))[:5]}"
    )


def _looks_demangled(name: str) -> bool:
    """Check if kernel name appears to be demangled (has C++ syntax)."""
    return "(" in name or "<" in name


def _looks_mangled_cpp(name: str) -> bool:
    """Check if kernel name appears to be C++ mangled (starts with _Z)."""
    return name.startswith("_Z")


def _assert_demangled_names(df):
    """Verify that at least one kernel name is demangled."""
    names = _kernel_names(df)

    assert any(
        _looks_demangled(name) for name in names
    ), "Expected at least one demangled kernel name, but none contained '(' or '<'."


def _assert_truncated_names(df):
    """Verify that all kernel names are truncated."""
    names = _kernel_names(df)

    invalid_names = [name for name in names if _looks_demangled(name)]
    assert not invalid_names, (
        "Expected all kernel names to be truncated, "
        f"but found non-truncated name: {invalid_names[0]}"
    )


def _assert_mangled_names(df):
    """Verify that kernel names remain mangled/raw."""
    names = _kernel_names(df)

    demangled_names = [name for name in names if _looks_demangled(name)]
    assert not demangled_names, (
        "Expected no demangled kernel names when mangled names are requested, "
        f"but found: {demangled_names[0]}"
    )

    mangled_names = [name for name in names if _looks_mangled_cpp(name)]
    assert mangled_names, (
        "Expected at least one C++ mangled kernel name starting with '_Z', "
        "but none were found."
    )


def _assert_statistics_match(df1, df2, name1, name2):
    """
    Helper function to verify that statistics match between two kernel summaries.

    Args:
        df1: First dataframe
        df2: Second dataframe (reference)
        name1: Label for first dataframe (e.g., "truncated")
        name2: Label for second dataframe (e.g., "full")
    """
    # Verify we have data
    assert len(df1) > 0, f"No kernels found in {name1} summary"
    assert len(df2) > 0, f"No kernels found in {name2} summary"

    # Verify same number of entries
    assert len(df1) == len(
        df2
    ), f"Mismatch in number of kernels: {name1}={len(df1)}, {name2}={len(df2)}"

    # Verify statistics are preserved
    total_calls_1 = df1["Calls"].sum()
    total_calls_2 = df2["Calls"].sum()
    assert (
        total_calls_1 == total_calls_2
    ), f"Total call count mismatch: {name1}={total_calls_1}, {name2}={total_calls_2}"

    total_duration_1 = df1["Duration (Nsec)"].sum()
    total_duration_2 = df2["Duration (Nsec)"].sum()
    assert (
        total_duration_1 == total_duration_2
    ), f"Total duration mismatch: {name1}={total_duration_1}, {name2}={total_duration_2}"


def test_summary_truncate_kernels(csv_kernels_truncated, csv_kernels_full, json_data):
    """
    Test that --truncate-kernels flag only affects kernel names, not statistics.

    This test verifies:
    1. Full kernel names and truncated kernel names are valid
    2. Statistics (calls, duration) are preserved between truncated and full summaries
    3. Names match the JSON source of truth exactly
    """

    _assert_demangled_names(csv_kernels_full)
    _assert_truncated_names(csv_kernels_truncated)
    _assert_statistics_match(csv_kernels_truncated, csv_kernels_full, "truncated", "full")

    # Verify against JSON source of truth
    _verify_names_match_json(csv_kernels_full, json_data, "full")
    _verify_names_match_json(csv_kernels_truncated, json_data, "truncated")


def test_summary_mangled_kernels(csv_kernels_mangled, csv_kernels_full, json_data):
    """
    Test that --mangled-kernels flag only affects kernel names, not statistics.

    This test verifies:
    1. Full kernel names and mangled kernel names are valid
    2. Statistics (calls, duration) are preserved between mangled and demangled summaries
    3. Names match the JSON source of truth exactly
    """

    _assert_demangled_names(csv_kernels_full)
    _assert_mangled_names(csv_kernels_mangled)
    _assert_statistics_match(csv_kernels_mangled, csv_kernels_full, "mangled", "full")

    # Verify against JSON source of truth
    _verify_names_match_json(csv_kernels_full, json_data, "full")
    _verify_names_match_json(csv_kernels_mangled, json_data, "mangled")


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
