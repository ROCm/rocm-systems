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

import sys
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


#########################################################################################
#
# ROCPD CSV comparison Validation
#
#########################################################################################


def test_compare_csv_files(csv_compare_1, csv_compare_2):
    """
    Test that two CSV files are identical using pandas.
    """
    # compare the headers
    assert csv_compare_1.columns.equals(
        csv_compare_2.columns
    ), "CSV files have different headers"
    # compare the data
    for col in csv_compare_1.columns:
        assert csv_compare_1[col].equals(
            csv_compare_2[col]
        ), f"CSV files have different data in column {col}"


def _decimal_places(value: str):
    """Return the number of decimal places in a numeric string, or None if not numeric.

    The exponent shifts the decimal point, so it is factored into the result.
    For example: "1.23e+1" -> 1, "1.23e-1" -> 3, "1.23e+10" -> 0.
    """
    value = value.strip()
    if not value:
        return None
    try:
        float(value)
    except ValueError:
        return None

    # Strip the exponent and record its value (e.g. "1.23e+10" -> exponent=10).
    exponent = 0
    exp_pos = value.lower().find("e")
    if exp_pos != -1:
        try:
            exponent = int(value[exp_pos + 1 :])
        except ValueError:
            pass
        value = value[:exp_pos]

    dot_pos = value.find(".")
    frac_len = 0 if dot_pos == -1 else len(value[dot_pos + 1 :])

    # A positive exponent shifts digits left (fewer decimals); negative shifts right (more).
    return max(0, frac_len - exponent)


def test_compare_csv_output_files(csv_output_1, csv_output_2):
    """
    Compare two CSV output files for consistent text formatting.

    Verifies:
    - Header text matches exactly (column names and order)
    - Number of columns is the same
    - Number of data rows is the same
    - Each header cell has the same quoting (present vs absent) in both files
    - Each data cell has the same quoting in both files
    - Each numeric cell has the same number of decimal places as its counterpart
    - Each numeric cell has the same value as its counterpart
    - Each non-numeric cell has the same string value as its counterpart
    """
    headers1, rows1, hquote1, rquote1 = csv_output_1
    headers2, rows2, hquote2, rquote2 = csv_output_2

    assert len(headers1) == len(headers2), (
        f"Column count differs: file1={len(headers1)}, file2={len(headers2)}\n"
        f"  file1 headers: {headers1}\n"
        f"  file2 headers: {headers2}"
    )

    assert (
        headers1 == headers2
    ), f"CSV headers differ:\n  file1: {headers1}\n  file2: {headers2}"

    # Check header quoting consistency
    for col_idx, (q1, q2) in enumerate(zip(hquote1, hquote2)):
        assert q1 == q2, (
            f"Header quoting differs for column '{headers1[col_idx]}': "
            f"file1={'quoted' if q1 else 'unquoted'}, "
            f"file2={'quoted' if q2 else 'unquoted'}"
        )

    assert len(rows1) == len(
        rows2
    ), f"Row count differs: file1={len(rows1)}, file2={len(rows2)}"

    for row_idx, (row1, row2, qrow1, qrow2) in enumerate(
        zip(rows1, rows2, rquote1, rquote2)
    ):
        assert len(row1) == len(row2), (
            f"Column count differs at data row {row_idx + 1}: "
            f"file1={len(row1)}, file2={len(row2)}"
        )
        for col_idx, (val1, val2, q1, q2) in enumerate(zip(row1, row2, qrow1, qrow2)):
            assert q1 == q2, (
                f"Cell quoting differs at row {row_idx + 1}, "
                f"column '{headers1[col_idx]}': "
                f"file1='{val1}' ({'quoted' if q1 else 'unquoted'}), "
                f"file2='{val2}' ({'quoted' if q2 else 'unquoted'})"
            )
            dec1 = _decimal_places(val1)
            dec2 = _decimal_places(val2)
            if dec1 is not None and dec2 is not None:
                assert dec1 == dec2, (
                    f"Decimal precision differs at row {row_idx + 1}, "
                    f"column '{headers1[col_idx]}': "
                    f"file1='{val1}' ({dec1} decimal(s)), "
                    f"file2='{val2}' ({dec2} decimal(s))"
                )
                assert float(val1) == float(val2), (
                    f"Numeric value differs at row {row_idx + 1}, "
                    f"column '{headers1[col_idx]}': "
                    f"file1='{val1}', file2='{val2}'"
                )
            else:
                assert val1 == val2, (
                    f"Cell value differs at row {row_idx + 1}, "
                    f"column '{headers1[col_idx]}': "
                    f"file1='{val1}', file2='{val2}'"
                )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
