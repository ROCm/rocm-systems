# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Unit tests for utils_profile_csv module.

Covers the stdlib-only CSV helpers used by the rocpd profile
path: reading and writing CSV rows, dropping columns, and assigning group ids.
"""

import csv
import gzip
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
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".csv.gz") as f:
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
    with gzip.open(temp_csv_file, "wt", newline="", encoding="utf-8") as f:
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
        csv_ops.read_csv_as_dicts("/nonexistent/file.csv.gz")


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
    with gzip.open(temp_csv_file, "wt", newline="", encoding="utf-8") as f:
        f.write("a,b,c\n")
    assert list(csv_ops.iter_csv_dicts(temp_csv_file)) == []


def test_iter_csv_dicts_no_header_raises(temp_csv_file):
    """An empty file (no header) raises ValueError."""
    Path(temp_csv_file).write_bytes(b"")
    with pytest.raises(ValueError, match="no header row"):
        list(csv_ops.iter_csv_dicts(temp_csv_file))


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
    """The profile-mode workflow: write rows, then stream them out with group ids
    assigned and a column dropped."""
    source_data = [
        {"name": "Alice", "category": "A", "value": "10"},
        {"name": "Bob", "category": "B", "value": "20"},
        {"name": "Charlie", "category": "A", "value": "30"},
    ]
    csv_ops.write_csv_from_dicts(temp_csv_file, source_data)

    groups = csv_ops.GroupIdAssigner(["category"], "group_id")
    output_file = temp_csv_file + ".out"
    rows_written = csv_ops.stream_csv_to_file(
        temp_csv_file,
        output_file,
        transform=groups.apply,
        drop_columns=["value"],
    )

    result, _ = csv_ops.read_csv_as_dicts(output_file)

    assert rows_written == 3
    assert len(result) == 3
    assert "value" not in result[0]
    assert result[0]["group_id"] == result[2]["group_id"]  # Both category A
    assert result[1]["group_id"] != result[0]["group_id"]

    # Cleanup
    Path(output_file).unlink(missing_ok=True)


# =============================================================================
# Streaming Tests
# =============================================================================


def test_stream_csv_to_file_copies_rows(tmp_path):
    """A row-for-row copy with the header preserved."""
    src = tmp_path / "in.csv.gz"
    with gzip.open(src, "wt", newline="", encoding="utf-8") as f:
        f.write("a,b\n1,2\n3,4\n")
    dest = tmp_path / "out.csv.gz"

    assert csv_ops.stream_csv_to_file(str(src), str(dest)) == 2
    with gzip.open(dest, "rt", encoding="utf-8") as f:
        assert f.read() == "a,b\n1,2\n3,4\n"


def test_stream_csv_to_file_header_only_writes_no_rows(tmp_path):
    """A workload that dispatched no kernels yields a header-only CSV; the caller
    detects that from the returned count."""
    src = tmp_path / "in.csv.gz"
    with gzip.open(src, "wt", newline="", encoding="utf-8") as f:
        f.write("a,b\n")
    dest = tmp_path / "out.csv.gz"

    assert csv_ops.stream_csv_to_file(str(src), str(dest)) == 0


def test_stream_csv_to_file_rejects_headerless_input(tmp_path):
    src = tmp_path / "empty.csv.gz"
    src.write_bytes(b"")

    with pytest.raises(ValueError, match="no header row"):
        csv_ops.stream_csv_to_file(str(src), str(tmp_path / "out.csv.gz"))


def test_group_id_assigner_reuses_ids_for_repeated_keys():
    assigner = csv_ops.GroupIdAssigner(["name"], "group_id")

    assert assigner.apply({"name": "a"})["group_id"] == 0
    assert assigner.apply({"name": "b"})["group_id"] == 1
    assert assigner.apply({"name": "a"})["group_id"] == 0
    # A missing column contributes None rather than raising.
    assert assigner.apply({})["group_id"] == 2


# =============================================================================
# Compression
#
# Every artifact these helpers touch is gzip, on both read and write.
# =============================================================================


def test_write_csv_from_dicts_compresses_gz_name(tmp_path, sample_csv_data):
    path = tmp_path / "out.csv.gz"

    csv_ops.write_csv_from_dicts(str(path), sample_csv_data)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == sample_csv_data


def test_read_csv_as_dicts_reads_compressed(tmp_path, sample_csv_data):
    path = tmp_path / "in.csv.gz"
    csv_ops.write_csv_from_dicts(str(path), sample_csv_data)

    rows, fieldnames = csv_ops.read_csv_as_dicts(str(path))

    assert rows == sample_csv_data
    assert fieldnames == ["name", "age", "city"]


def test_iter_csv_dicts_reads_compressed(tmp_path, sample_csv_data):
    path = tmp_path / "in.csv.gz"
    csv_ops.write_csv_from_dicts(str(path), sample_csv_data)

    assert list(csv_ops.iter_csv_dicts(str(path))) == sample_csv_data
