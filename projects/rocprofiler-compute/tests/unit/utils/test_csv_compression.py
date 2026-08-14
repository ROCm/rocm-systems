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
def plain_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(CONTENT, encoding="utf-8")
    return path


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


def test_write_compresses_when_name_says_to(tmp_path):
    path = tmp_path / "out.csv.gz"

    with csv_compression.open_csv_write(path) as f:
        f.write(CONTENT)

    assert csv_compression._is_compressed(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        assert f.read() == CONTENT


def test_write_leaves_plain_name_uncompressed(tmp_path):
    """pmc_perf.csv and sysinfo.csv stay plain, and pandas still reads them."""
    path = tmp_path / "out.csv"

    with csv_compression.open_csv_write(path) as f:
        f.write(CONTENT)

    assert not csv_compression._is_compressed(path)
    assert path.read_text(encoding="utf-8") == CONTENT


def test_written_gzip_is_one_complete_member(tmp_path):
    """The contract says one member per file, which is what readers assume."""
    path = tmp_path / "out.csv.gz"

    with csv_compression.open_csv_write(path) as f:
        for _ in range(1000):
            f.write(CONTENT)

    raw = path.read_bytes()
    assert raw.count(b"\x1f\x8b\x08") == 1
    assert gzip.decompress(raw).decode("utf-8") == CONTENT * 1000


# =============================================================================
# Reading
# =============================================================================


def test_read_gzip(gzip_csv):
    with csv_compression.open_csv_read(gzip_csv) as f:
        assert f.read() == CONTENT


def test_read_plain(plain_csv):
    with csv_compression.open_csv_read(plain_csv) as f:
        assert f.read() == CONTENT


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        csv_compression.open_csv_read(tmp_path / "absent.csv")


def test_read_dispatches_on_content_not_name(tmp_path):
    named_gz_but_plain = tmp_path / "a.csv.gz"
    named_gz_but_plain.write_text(CONTENT, encoding="utf-8")
    named_plain_but_gz = tmp_path / "b.csv"
    named_plain_but_gz.write_bytes(gzip.compress(CONTENT.encode("utf-8")))

    for path in (named_gz_but_plain, named_plain_but_gz):
        with csv_compression.open_csv_read(path) as f:
            assert f.read() == CONTENT


def test_truncated_gzip_raises_a_corrupt_csv_error(tmp_path):
    """Partial .gz must raise an exception in CORRUPT_CSV_ERRORS."""
    path = tmp_path / "partial.csv.gz"
    path.write_bytes(gzip.compress((CONTENT * 1000).encode("utf-8"))[:40])

    with pytest.raises(csv_compression.CORRUPT_CSV_ERRORS):
        with csv_compression.open_csv_read(path) as f:
            f.read()


def test_corrupt_gzip_raises_a_corrupt_csv_error(tmp_path):
    """A flipped byte fails the CRC rather than yielding wrong rows."""
    path = tmp_path / "corrupt.csv.gz"
    raw = bytearray(gzip.compress((CONTENT * 1000).encode("utf-8")))
    raw[-5] ^= 0xFF
    path.write_bytes(raw)

    with pytest.raises(csv_compression.CORRUPT_CSV_ERRORS):
        with csv_compression.open_csv_read(path) as f:
            f.read()


# =============================================================================
# is_compressed
# =============================================================================


def test_is_compressed_reports_false_for_plain(plain_csv):
    assert not csv_compression._is_compressed(plain_csv)


def test_is_compressed_reports_false_for_missing_file(tmp_path):
    assert not csv_compression._is_compressed(tmp_path / "absent.csv")


def test_is_compressed_reports_false_for_empty_file(tmp_path):
    """A run killed before the first flush leaves a zero-byte file."""
    empty = tmp_path / "empty.csv"
    empty.touch()
    assert not csv_compression._is_compressed(empty)


# =============================================================================
# Discovery
# =============================================================================


def test_find_csvs_matches_both_forms(tmp_path):
    (tmp_path / "results_a.csv").write_text(CONTENT, encoding="utf-8")
    (tmp_path / "results_b.csv.gz").write_bytes(gzip.compress(CONTENT.encode()))
    (tmp_path / "sysinfo.csv").write_text(CONTENT, encoding="utf-8")

    found = csv_compression.find_csvs(tmp_path, "results_*.csv")

    assert [p.name for p in found] == ["results_a.csv", "results_b.csv.gz"]


def test_find_csvs_returns_one_path_per_artifact(tmp_path):
    """Both forms of one pass must not concatenate as if they were two passes."""
    (tmp_path / "results_a.csv").write_text(CONTENT, encoding="utf-8")
    (tmp_path / "results_a.csv.gz").write_bytes(gzip.compress(CONTENT.encode()))

    found = csv_compression.find_csvs(tmp_path, "results_*.csv")

    assert [p.name for p in found] == ["results_a.csv.gz"]


def test_find_csvs_orders_by_artifact_name(tmp_path):
    for name in ("results_c.csv", "results_a.csv.gz", "results_b.csv"):
        (tmp_path / name).write_text(CONTENT, encoding="utf-8")

    found = csv_compression.find_csvs(tmp_path, "results_*.csv")

    assert [p.name for p in found] == [
        "results_a.csv.gz",
        "results_b.csv",
        "results_c.csv",
    ]


def test_find_csvs_on_empty_directory(tmp_path):
    assert csv_compression.find_csvs(tmp_path, "results_*.csv") == []


# =============================================================================
# resolve_csv
# =============================================================================


def test_resolve_csv_prefers_compressed(tmp_path):
    plain = tmp_path / "counters.csv"
    plain.write_text(CONTENT, encoding="utf-8")
    compressed = tmp_path / "counters.csv.gz"
    compressed.write_bytes(gzip.compress(CONTENT.encode()))

    assert csv_compression.resolve_csv(plain) == compressed


def test_resolve_csv_falls_back_to_plain(plain_csv):
    assert csv_compression.resolve_csv(plain_csv) == plain_csv


def test_resolve_csv_returns_compressed_when_neither_exists(tmp_path):
    absent = tmp_path / "absent.csv"

    resolved = csv_compression.resolve_csv(absent)

    assert resolved == tmp_path / "absent.csv.gz"
    assert not resolved.is_file()


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
