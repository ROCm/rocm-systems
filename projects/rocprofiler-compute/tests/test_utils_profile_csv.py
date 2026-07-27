# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for utils_profile_csv module.

Covers the stdlib-only CSV helpers used by the rocpd profile
path: reading and writing CSV rows, dropping columns, and assigning group ids.
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


def test_iter_csv_dicts_matches_read_csv_as_dicts(temp_csv_file, sample_csv_data):
    """iter_csv_dicts streams the same rows read_csv_as_dicts returns."""
    csv_ops.write_csv_from_dicts(temp_csv_file, sample_csv_data)
    expected, _ = csv_ops.read_csv_as_dicts(temp_csv_file)
    assert list(csv_ops.iter_csv_dicts(temp_csv_file)) == expected


def test_iter_csv_dicts_empty_body(temp_csv_file):
    """A CSV with only a header yields zero rows."""
    Path(temp_csv_file).write_text("a,b,c\n", encoding="utf-8")
    assert list(csv_ops.iter_csv_dicts(temp_csv_file)) == []


def test_iter_csv_dicts_no_header_raises(temp_csv_file):
    """An empty file (no header) raises ValueError."""
    Path(temp_csv_file).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no header row"):
        list(csv_ops.iter_csv_dicts(temp_csv_file))


# =============================================================================
# Column Manipulation Tests
# =============================================================================


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
    rows, _ = csv_ops.read_csv_as_dicts(temp_csv_file)

    # Transform: assign group IDs
    csv_ops.assign_group_ids(rows, ["category"], "group_id")

    # Transform: drop a column
    csv_ops.drop_column_from_rows(rows, "value")

    # Write back
    output_file = temp_csv_file + ".out"
    csv_ops.write_csv_from_dicts(output_file, rows)

    # Verify
    result, _ = csv_ops.read_csv_as_dicts(output_file)

    assert len(result) == 3
    assert "value" not in result[0]
    assert "group_id" in result[0]
    assert result[0]["group_id"] == result[2]["group_id"]  # Both category A

    # Cleanup
    Path(output_file).unlink(missing_ok=True)
