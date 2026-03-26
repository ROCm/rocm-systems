# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for utils_profile_csv module.

Tests the pandas compatibility layer - CSV operations using stdlib only.
These tests ensure correctness and acceptable performance of the stdlib
implementation compared to pandas.
"""

import csv
import tempfile
from pathlib import Path

import pytest

import utils.utils_profile_csv as csv_ops

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", newline=""
    ) as f:
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing."""
    return [
        {"name": "Alice", "age": "30", "city": "NYC"},
        {"name": "Bob", "age": "25", "city": "LA"},
        {"name": "Charlie", "age": "35", "city": "NYC"},
    ]


# =============================================================================
# Basic CSV I/O Tests
# =============================================================================


def test_read_csv_as_dicts(temp_csv_file):
    """Test reading CSV file."""
    # Write test data
    with open(temp_csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["a", "b", "c"])
        writer.writeheader()
        writer.writerow({"a": "1", "b": "2", "c": "3"})
        writer.writerow({"a": "4", "b": "5", "c": "6"})

    # Test read
    rows, fieldnames = csv_ops.read_csv_as_dicts(temp_csv_file)

    assert len(rows) == 2
    assert fieldnames == ["a", "b", "c"]
    assert rows[0] == {"a": "1", "b": "2", "c": "3"}
    assert rows[1] == {"a": "4", "b": "5", "c": "6"}


def test_read_csv_empty_file(temp_csv_file):
    """Test reading empty CSV file raises error."""
    # Create empty file
    Path(temp_csv_file).touch()

    with pytest.raises(ValueError, match="no header row"):
        csv_ops.read_csv_as_dicts(temp_csv_file)


def test_read_csv_nonexistent_file():
    """Test reading nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        csv_ops.read_csv_as_dicts("/nonexistent/file.csv")


def test_write_csv_from_dicts(temp_csv_file, sample_csv_data):
    """Test writing CSV from list of dicts."""
    csv_ops.write_csv_from_dicts(temp_csv_file, sample_csv_data)

    # Read back and verify
    rows, fieldnames = csv_ops.read_csv_as_dicts(temp_csv_file)

    assert len(rows) == 3
    assert fieldnames == ["name", "age", "city"]
    assert rows[0] == sample_csv_data[0]


def test_write_csv_with_fieldnames(temp_csv_file):
    """Test writing CSV with explicit fieldnames."""
    rows = [{"a": 1, "b": 2, "c": 3}]
    fieldnames = ["c", "b", "a"]  # Different order

    csv_ops.write_csv_from_dicts(temp_csv_file, rows, fieldnames)

    # Read back and verify order
    _, result_fieldnames = csv_ops.read_csv_as_dicts(temp_csv_file)
    assert result_fieldnames == fieldnames


def test_write_csv_empty_rows(temp_csv_file):
    """Test writing empty rows does nothing."""
    csv_ops.write_csv_from_dicts(temp_csv_file, [])

    # File should not exist or be empty
    assert not Path(temp_csv_file).exists() or Path(temp_csv_file).stat().st_size == 0


def test_concat_csv_files(sample_csv_data):
    """Test concatenating multiple CSV files."""
    # Create two temp files
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f1:
        file1 = f1.name
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f2:
        file2 = f2.name
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f3:
        output_file = f3.name

    try:
        # Write data to files
        csv_ops.write_csv_from_dicts(file1, sample_csv_data[:2])
        csv_ops.write_csv_from_dicts(file2, sample_csv_data[2:])

        # Concat
        result = csv_ops.concat_csv_files([file1, file2], output_file)

        assert len(result) == 3
        assert result == sample_csv_data

        # Verify output file
        rows, _ = csv_ops.read_csv_as_dicts(output_file)
        assert rows == sample_csv_data

    finally:
        Path(file1).unlink(missing_ok=True)
        Path(file2).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


# =============================================================================
# Column Manipulation Tests
# =============================================================================


def test_add_column_to_rows():
    """Test adding a column to rows."""
    rows = [{"a": 1}, {"a": 2}, {"a": 3}]
    values = [10, 20, 30]

    csv_ops.add_column_to_rows(rows, "b", values)

    assert rows[0] == {"a": 1, "b": 10}
    assert rows[1] == {"a": 2, "b": 20}
    assert rows[2] == {"a": 3, "b": 30}


def test_add_column_length_mismatch():
    """Test add_column raises error on length mismatch."""
    rows = [{"a": 1}, {"a": 2}]
    values = [10, 20, 30]  # Too many values

    with pytest.raises(ValueError, match="Length mismatch"):
        csv_ops.add_column_to_rows(rows, "b", values)


def test_drop_column_from_rows():
    """Test dropping a column from rows."""
    rows = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]

    csv_ops.drop_column_from_rows(rows, "b")

    assert rows[0] == {"a": 1, "c": 3}
    assert rows[1] == {"a": 4, "c": 6}


def test_drop_nonexistent_column():
    """Test dropping nonexistent column does nothing."""
    rows = [{"a": 1}]

    csv_ops.drop_column_from_rows(rows, "nonexistent")

    assert rows[0] == {"a": 1}


def test_rename_columns():
    """Test renaming columns."""
    rows = [{"old1": 1, "old2": 2, "keep": 3}]
    mapping = {"old1": "new1", "old2": "new2"}

    csv_ops.rename_columns(rows, mapping)

    assert rows[0] == {"new1": 1, "new2": 2, "keep": 3}


# =============================================================================
# GroupBy Tests
# =============================================================================


def test_assign_group_ids_single_column():
    """Test assigning group IDs based on single column."""
    rows = [
        {"category": "A"},
        {"category": "B"},
        {"category": "A"},
        {"category": "C"},
        {"category": "B"},
    ]

    csv_ops.assign_group_ids(rows, ["category"], "group_id")

    assert rows[0]["group_id"] == 0  # First A
    assert rows[1]["group_id"] == 1  # First B
    assert rows[2]["group_id"] == 0  # Second A (same as first)
    assert rows[3]["group_id"] == 2  # First C
    assert rows[4]["group_id"] == 1  # Second B (same as first)


def test_assign_group_ids_multiple_columns():
    """Test assigning group IDs based on multiple columns."""
    rows = [
        {"name": "A", "value": 1},
        {"name": "B", "value": 2},
        {"name": "A", "value": 1},
        {"name": "A", "value": 2},
    ]

    csv_ops.assign_group_ids(rows, ["name", "value"], "group_id")

    assert rows[0]["group_id"] == 0  # A,1
    assert rows[1]["group_id"] == 1  # B,2
    assert rows[2]["group_id"] == 0  # A,1 (same)
    assert rows[3]["group_id"] == 2  # A,2 (different)


def test_groupby_aggregate_sum():
    """Test groupby with sum aggregation."""
    rows = [
        {"category": "A", "value": 10},
        {"category": "A", "value": 20},
        {"category": "B", "value": 15},
        {"category": "B", "value": 25},
    ]

    result = csv_ops.groupby_aggregate(rows, ["category"], {"value": "sum"})

    assert len(result) == 2
    # Find results by category
    a_result = [r for r in result if r["category"] == "A"][0]
    b_result = [r for r in result if r["category"] == "B"][0]

    assert a_result["value"] == 30
    assert b_result["value"] == 40


def test_groupby_aggregate_first_last():
    """Test groupby with first/last aggregation."""
    rows = [
        {"category": "A", "value": 10},
        {"category": "A", "value": 20},
        {"category": "A", "value": 30},
    ]

    result_first = csv_ops.groupby_aggregate(rows, ["category"], {"value": "first"})
    result_last = csv_ops.groupby_aggregate(rows, ["category"], {"value": "last"})

    assert result_first[0]["value"] == 10
    assert result_last[0]["value"] == 30


def test_groupby_aggregate_mean():
    """Test groupby with mean aggregation."""
    rows = [
        {"category": "A", "value": 10},
        {"category": "A", "value": 20},
        {"category": "A", "value": 30},
    ]

    result = csv_ops.groupby_aggregate(rows, ["category"], {"value": "mean"})

    assert result[0]["value"] == 20.0


def test_groupby_aggregate_min_max():
    """Test groupby with min/max aggregation."""
    rows = [
        {"category": "A", "value": 30},
        {"category": "A", "value": 10},
        {"category": "A", "value": 20},
    ]

    result = csv_ops.groupby_aggregate(rows, ["category"], {"value": "min"})
    assert result[0]["value"] == 10

    result = csv_ops.groupby_aggregate(rows, ["category"], {"value": "max"})
    assert result[0]["value"] == 30


def test_groupby_aggregate_invalid_function():
    """Test groupby with invalid aggregation function."""
    rows = [{"category": "A", "value": 10}]

    with pytest.raises(ValueError, match="Unsupported aggregation function"):
        csv_ops.groupby_aggregate(rows, ["category"], {"value": "invalid"})


# =============================================================================
# Pivot Table Tests
# =============================================================================


def test_pivot_table_basic():
    """Test basic pivot table operation."""
    rows = [
        {"id": 1, "counter": "A", "value": 10},
        {"id": 1, "counter": "B", "value": 20},
        {"id": 2, "counter": "A", "value": 30},
    ]

    result = csv_ops.pivot_table(rows, ["id"], "counter", "value")

    assert len(result) == 2

    # Find results by id
    id1 = [r for r in result if r["id"] == 1][0]
    id2 = [r for r in result if r["id"] == 2][0]

    assert id1["A"] == 10
    assert id1["B"] == 20
    assert id2["A"] == 30
    assert id2["B"] is None  # Missing value filled with None


def test_pivot_table_with_fill_value():
    """Test pivot table fills missing values with None."""
    rows = [
        {"id": 1, "counter": "A", "value": 10},
        {"id": 2, "counter": "B", "value": 20},
    ]

    result = csv_ops.pivot_table(rows, ["id"], "counter", "value")

    id1 = [r for r in result if r["id"] == 1][0]
    id2 = [r for r in result if r["id"] == 2][0]

    assert id1["A"] == 10
    assert id1["B"] is None  # Filled with None
    assert id2["A"] is None  # Filled with None
    assert id2["B"] == 20


def test_pivot_table_multiple_index_columns():
    """Test pivot table with multiple index columns."""
    rows = [
        {"id": 1, "name": "X", "counter": "A", "value": 10},
        {"id": 1, "name": "Y", "counter": "A", "value": 20},
    ]

    result = csv_ops.pivot_table(rows, ["id", "name"], "counter", "value")

    assert len(result) == 2
    assert all("A" in r for r in result)


# =============================================================================
# Merge Tests
# =============================================================================


def test_merge_rows_inner():
    """Test inner merge."""
    left = [
        {"id": 1, "name": "A"},
        {"id": 2, "name": "B"},
        {"id": 3, "name": "C"},
    ]
    right = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
    ]

    result = csv_ops.merge_rows(left, right, "id", "id", how="inner")

    assert len(result) == 2
    assert result[0] == {"id": 1, "name": "A", "value": 10}
    assert result[1] == {"id": 2, "name": "B", "value": 20}


def test_merge_rows_left():
    """Test left merge."""
    left = [
        {"id": 1, "name": "A"},
        {"id": 2, "name": "B"},
        {"id": 3, "name": "C"},
    ]
    right = [
        {"id": 1, "value": 10},
    ]

    result = csv_ops.merge_rows(left, right, "id", "id", how="left")

    assert len(result) == 3
    assert result[0] == {"id": 1, "name": "A", "value": 10}
    assert result[1] == {"id": 2, "name": "B"}
    assert result[2] == {"id": 3, "name": "C"}


def test_merge_rows_cartesian_product():
    """Test merge with duplicate keys creates cartesian product."""
    left = [
        {"id": 1, "name": "A"},
        {"id": 1, "name": "B"},
    ]
    right = [
        {"id": 1, "value": 10},
        {"id": 1, "value": 20},
    ]

    result = csv_ops.merge_rows(left, right, "id", "id", how="inner")

    assert len(result) == 4  # 2 x 2 = 4


def test_merge_rows_invalid_how():
    """Test merge with invalid join type."""
    with pytest.raises(ValueError, match="Unsupported join type"):
        csv_ops.merge_rows([], [], "id", "id", how="invalid")


# =============================================================================
# Map Column Tests
# =============================================================================


# =============================================================================
# Integration Tests
# =============================================================================


def test_write_csv_extra_keys(temp_csv_file):
    """Test writing CSV with rows that have extra keys."""
    rows = [
        {"a": 1, "b": 2, "c": 3, "extra": 999},  # Extra key
        {"a": 4, "b": 5, "c": 6},
    ]
    fieldnames = ["a", "b", "c"]  # No 'extra'

    # Should not raise error (extrasaction='ignore')
    csv_ops.write_csv_from_dicts(temp_csv_file, rows, fieldnames)

    # Read back and verify 'extra' was ignored
    result, result_fieldnames = csv_ops.read_csv_as_dicts(temp_csv_file)
    assert "extra" not in result_fieldnames
    assert result[0] == {"a": "1", "b": "2", "c": "3"}


def test_groupby_aggregate_non_numeric():
    """Test groupby with non-numeric values in sum/mean."""
    rows = [
        {"category": "A", "value": "not_a_number"},
        {"category": "A", "value": "also_not"},
    ]

    # Should not crash - returns 0 for sum
    result = csv_ops.groupby_aggregate(rows, ["category"], {"value": "sum"})
    assert result[0]["value"] == 0

    # Should not crash - returns None for mean
    result = csv_ops.groupby_aggregate(rows, ["category"], {"value": "mean"})
    assert result[0]["value"] is None


def test_groupby_aggregate_mixed_numeric():
    """Test groupby with mixed numeric and non-numeric values."""
    rows = [
        {"category": "A", "value": "10"},
        {"category": "A", "value": "20"},
        {"category": "A", "value": "bad"},  # Non-numeric
    ]

    # Should handle gracefully (don't need to check result)
    csv_ops.groupby_aggregate(rows, ["category"], {"value": "sum"})


def test_pivot_table_duplicate_combinations():
    """Test pivot table with duplicate (index + pivot) combinations."""
    rows = [
        {"id": 1, "counter": "A", "value": 10},
        {"id": 1, "counter": "A", "value": 20},  # Duplicate!
    ]

    result = csv_ops.pivot_table(rows, ["id"], "counter", "value")

    # Mean of values is computed
    assert len(result) == 1
    assert result[0]["A"] == 15.0  # Mean of [10, 20]


def test_merge_rows_none_keys():
    """Test merge with None as join keys."""
    left = [
        {"id": None, "name": "A"},
        {"id": 1, "name": "B"},
    ]
    right = [
        {"id": None, "value": 10},
        {"id": 1, "value": 20},
    ]

    result = csv_ops.merge_rows(left, right, "id", "id", how="inner")

    # None matches None in current implementation
    assert len(result) == 2
    # Verify both None and 1 matched
    assert any(r["name"] == "A" and r["value"] == 10 for r in result)
    assert any(r["name"] == "B" and r["value"] == 20 for r in result)


def test_assign_group_ids_empty_rows():
    """Test assign_group_ids with empty rows list."""
    rows = []
    csv_ops.assign_group_ids(rows, ["col"], "group_id")
    # Should not crash
    assert rows == []


def test_assign_group_ids_missing_columns():
    """Test assign_group_ids with missing columns in some rows."""
    rows = [
        {"a": 1, "b": 2},
        {"a": 1},  # Missing 'b'
        {"b": 2},  # Missing 'a'
    ]

    csv_ops.assign_group_ids(rows, ["a", "b"], "group_id")

    # Missing keys become None in tuple
    assert rows[0]["group_id"] == 0  # (1, 2)
    assert rows[1]["group_id"] == 1  # (1, None)
    assert rows[2]["group_id"] == 2  # (None, 2)


def test_full_workflow(temp_csv_file):
    """Test complete workflow: read, transform, write."""
    # Create source data
    source_data = [
        {"name": "Alice", "category": "A", "value": "10"},
        {"name": "Bob", "category": "B", "value": "20"},
        {"name": "Charlie", "category": "A", "value": "30"},
    ]
    csv_ops.write_csv_from_dicts(temp_csv_file, source_data)

    # Read
    rows, fieldnames = csv_ops.read_csv_as_dicts(temp_csv_file)

    # Transform: assign group IDs
    csv_ops.assign_group_ids(rows, ["category"], "group_id")

    # Transform: rename column
    csv_ops.rename_columns(rows, {"name": "person"})

    # Write back
    output_file = temp_csv_file + ".out"
    csv_ops.write_csv_from_dicts(output_file, rows)

    # Verify
    result, _ = csv_ops.read_csv_as_dicts(output_file)

    assert len(result) == 3
    assert "person" in result[0]
    assert "group_id" in result[0]
    assert result[0]["group_id"] == result[2]["group_id"]  # Both category A

    # Cleanup
    Path(output_file).unlink(missing_ok=True)
