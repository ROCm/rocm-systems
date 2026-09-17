# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.csv_compression."""

import gzip
import re
from pathlib import Path

import pytest

from utils import csv_compression

CONTENT = "a,b\n1,2\n"


@pytest.fixture
def gzip_csv(tmp_path):
    path = tmp_path / "data.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(CONTENT)
    return path


# =============================================================================
# Naming
# =============================================================================


def test_compressed_name_appends_suffix():
    assert csv_compression.compressed_name("results_pmc_perf_0.csv") == Path(
        "results_pmc_perf_0.csv.gz"
    )


def test_compressed_name_is_idempotent():
    once = csv_compression.compressed_name("results_pmc_perf_0.csv")
    assert csv_compression.compressed_name(once) == once


def test_compressed_name_accepts_path(tmp_path):
    assert csv_compression.compressed_name(tmp_path / "x.csv") == tmp_path / "x.csv.gz"


# =============================================================================
# Writing
# =============================================================================


def test_write_compresses(tmp_path):
    path = tmp_path / "out.csv.gz"

    with csv_compression.open_gzip_csv_write(path) as f:
        f.write(CONTENT)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert f.read() == CONTENT


def test_written_gzip_is_one_complete_member(tmp_path):
    """The contract says one member per file, which is what readers assume."""
    path = tmp_path / "out.csv.gz"

    with csv_compression.open_gzip_csv_write(path) as f:
        for _ in range(1000):
            f.write(CONTENT)

    raw = path.read_bytes()
    assert raw.count(b"\x1f\x8b\x08") == 1
    assert gzip.decompress(raw).decode("utf-8") == CONTENT * 1000


# =============================================================================
# Reading
# =============================================================================


def test_read_gzip(gzip_csv):
    with csv_compression.open_gzip_csv_read(gzip_csv) as f:
        assert f.read() == CONTENT


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        csv_compression.open_gzip_csv_read(tmp_path / "absent.csv.gz")


def test_truncated_gzip_raises_a_corrupt_csv_error(tmp_path):
    """Partial .gz must raise an exception in CORRUPT_CSV_ERRORS."""
    path = tmp_path / "partial.csv.gz"
    path.write_bytes(gzip.compress((CONTENT * 1000).encode("utf-8"))[:40])

    with pytest.raises(csv_compression.CORRUPT_CSV_ERRORS):
        with csv_compression.open_gzip_csv_read(path) as f:
            f.read()


def test_corrupt_gzip_raises_a_corrupt_csv_error(tmp_path):
    """A flipped byte fails the CRC rather than yielding wrong rows."""
    path = tmp_path / "corrupt.csv.gz"
    raw = bytearray(gzip.compress((CONTENT * 1000).encode("utf-8")))
    raw[-5] ^= 0xFF
    path.write_bytes(raw)

    with pytest.raises(csv_compression.CORRUPT_CSV_ERRORS):
        with csv_compression.open_gzip_csv_read(path) as f:
            f.read()


# =============================================================================
# Cross-language contract with the native tool
# =============================================================================


def test_native_counter_csv_header_matches_the_reader():
    source = (
        Path(__file__).resolve().parents[3]
        / "src/lib/rocprofiler_compute_tool/counters_writer.cpp"
    )
    if not source.is_file():
        pytest.skip("C++ sources are absent from an installed test tree")

    literal = (
        source.read_text(encoding="utf-8").split("kHeader =", 1)[1].split(";", 1)[0]
    )
    columns = "".join(re.findall(r'"(.*?)"', literal)).replace("\\n", "").split(",")

    assert columns == [
        "dispatch_id",
        "gpu_id",
        "kernel_id",
        "lds_per_workgroup",
        "counter_id",
        "counter_name",
        "counter_value",
    ]
