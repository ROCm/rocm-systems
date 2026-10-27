# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Guards for tuner/embed_tuner_configs.py.

Verifies that the script that embeds tuner/*.csv into a C++ header behaves
correctly in isolation (no GPU, no built library, no CMake). The C++ side of
the pipeline -- that the embedded bytes are read back correctly by csv_tuner.cc
and applied to the cost table -- is covered by test/host/csv-tuner-test.cc.

Two test categories:

  script     -- black-box tests of embed_tuner_configs.py: invocation errors,
                header structure, no-change guard, validation, round-trip
                correctness. Uses only synthetic CSV data in temp directories.

  real_data  -- run the script against the actual tuner/ directory and verify
                the output matches what the C++ suite expects to find.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

RCCL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = RCCL_ROOT / "tuner" / "embed_tuner_configs.py"
TUNER_DIR = RCCL_ROOT / "tuner"

RAW_DELIM = "RCCLCSV"
RAW_TERMINATOR = f"){RAW_DELIM}\""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(tuner_dir, out_file):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tuner_dir), str(out_file)],
        capture_output=True,
        text=True,
    )


def write_csv(directory, name, content):
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def raw_string_contents(header):
    """Extract the content of every R"RCCLCSV(...)RCCLCSV" block."""
    return re.findall(r'R"RCCLCSV\((.*?)\)RCCLCSV"', header, re.DOTALL)


# ---------------------------------------------------------------------------
# Invocation errors
# ---------------------------------------------------------------------------

@pytest.mark.script
def test_no_args_exits_nonzero():
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "usage" in combined.lower()


@pytest.mark.script
def test_too_many_args_exits_nonzero():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "a", "b", "c"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Happy-path header structure
# ---------------------------------------------------------------------------

@pytest.mark.script
def test_single_csv_produces_valid_header(tmp_path):
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", "allreduce,0,65536,tree,ll,1,1,8,-1,-1\n")
    out = tmp_path / "out.h"

    result = run(tmp_path, out)
    assert result.returncode == 0, result.stderr
    header = out.read_text()

    assert "#ifndef RCCL_TUNER_EMBEDDED_CONFIGS_H_" in header
    assert "struct RcclEmbeddedTunerConfig" in header
    assert "kRcclEmbeddedTunerConfigs" in header
    assert "{nullptr, nullptr}" in header
    assert '"rccl_tuner_gfx950.csv"' in header
    assert "1 config(s)" in header


@pytest.mark.script
def test_empty_dir_produces_sentinel_only_header(tmp_path):
    out = tmp_path / "out.h"
    result = run(tmp_path, out)
    assert result.returncode == 0, result.stderr

    header = out.read_text()
    assert "0 config(s)" in header
    assert "{nullptr, nullptr}" in header
    assert not re.search(r'"rccl_tuner[^"]+\.csv"', header)


@pytest.mark.script
def test_non_matching_csv_not_embedded(tmp_path):
    write_csv(tmp_path, "notes.csv", "some,data\n")
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", "allreduce,0,1024,ring,ll,1,1,8\n")
    out = tmp_path / "out.h"

    run(tmp_path, out)
    header = out.read_text()

    assert "notes.csv" not in header
    assert "rccl_tuner_gfx950.csv" in header


@pytest.mark.script
def test_multiple_csvs_are_sorted(tmp_path):
    write_csv(tmp_path, "rccl_tuner_gfx942.csv",  "allreduce,0,1024,ring,ll,1,1,8\n")
    write_csv(tmp_path, "rccl_tuner_gfx950.csv",  "allreduce,0,2048,tree,ll,1,1,8\n")
    write_csv(tmp_path, "rccl_tuner_gfx1250.csv", "allreduce,0,4096,ring,simple,1,1,8\n")
    out = tmp_path / "out.h"

    run(tmp_path, out)
    header = out.read_text()

    # Lexicographic: gfx1250 < gfx942 < gfx950 ("1" < "9")
    pos = {name: header.index(name) for name in
           ("rccl_tuner_gfx942.csv", "rccl_tuner_gfx950.csv", "rccl_tuner_gfx1250.csv")}
    assert pos["rccl_tuner_gfx1250.csv"] < pos["rccl_tuner_gfx942.csv"] < pos["rccl_tuner_gfx950.csv"]
    assert "3 config(s)" in header


@pytest.mark.script
def test_output_dir_created_when_missing(tmp_path):
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", "allreduce,0,1024,ring,ll,1,1,8\n")
    out = tmp_path / "subdir" / "nested" / "out.h"

    result = run(tmp_path, out)
    assert result.returncode == 0, result.stderr
    assert out.exists()


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------

@pytest.mark.script
def test_embedded_content_matches_source_file(tmp_path):
    csv_content = (
        "# comment line\n"
        "allreduce,0,65536,tree,ll,1,1,8,-1,-1\n"
        "allreduce,65537,524287,ring,ll,-1,1,8,-1,-1\n"
    )
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", csv_content)
    out = tmp_path / "out.h"
    run(tmp_path, out)

    contents = raw_string_contents(out.read_text())
    assert len(contents) == 1, "expected exactly one raw string literal"
    assert contents[0] == csv_content


@pytest.mark.script
def test_multiline_csv_with_comments_embedded_verbatim(tmp_path):
    """The script must not strip comments or blank lines -- csv_tuner.cc does that."""
    csv_content = (
        "# tuning for gfx950 1-node\n"
        "allreduce,0,16383,tree,ll,1,1,8,-1,-1\n"
        "\n"
        "# 2-node entries\n"
        "allreduce,33554432,54525951,ring,ll128,64,2,16,-1,-1\n"
    )
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", csv_content)
    out = tmp_path / "out.h"
    run(tmp_path, out)

    contents = raw_string_contents(out.read_text())
    assert contents[0] == csv_content


# ---------------------------------------------------------------------------
# No-change guard
# ---------------------------------------------------------------------------

@pytest.mark.script
def test_no_change_guard_does_not_rewrite(tmp_path):
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", "allreduce,0,1024,ring,ll,1,1,8\n")
    out = tmp_path / "out.h"

    run(tmp_path, out)
    mtime_first = out.stat().st_mtime

    time.sleep(0.05)
    run(tmp_path, out)
    mtime_second = out.stat().st_mtime

    assert mtime_first == mtime_second, "file was rewritten on second run with identical inputs"


@pytest.mark.script
def test_change_guard_rewrites_when_content_changes(tmp_path):
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", "allreduce,0,1024,ring,ll,1,1,8\n")
    out = tmp_path / "out.h"

    run(tmp_path, out)
    mtime_first = out.stat().st_mtime

    time.sleep(0.05)
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", "allreduce,0,2048,tree,simple,4,1,8\n")
    run(tmp_path, out)
    mtime_second = out.stat().st_mtime

    assert mtime_second > mtime_first, "file was not rewritten after CSV content changed"


@pytest.mark.script
def test_change_guard_rewrites_when_csv_added(tmp_path):
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", "allreduce,0,1024,ring,ll,1,1,8\n")
    out = tmp_path / "out.h"

    run(tmp_path, out)
    mtime_first = out.stat().st_mtime

    time.sleep(0.05)
    write_csv(tmp_path, "rccl_tuner_gfx942.csv", "allreduce,0,2048,ring,ll,1,1,8\n")
    run(tmp_path, out)
    mtime_second = out.stat().st_mtime

    assert mtime_second > mtime_first, "file was not rewritten after a new CSV was added"


@pytest.mark.script
def test_change_guard_rewrites_when_csv_removed(tmp_path):
    write_csv(tmp_path, "rccl_tuner_gfx942.csv", "allreduce,0,1024,ring,ll,1,1,8\n")
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", "allreduce,0,2048,tree,ll,1,1,8\n")
    out = tmp_path / "out.h"

    run(tmp_path, out)
    mtime_first = out.stat().st_mtime

    time.sleep(0.05)
    (tmp_path / "rccl_tuner_gfx942.csv").unlink()
    run(tmp_path, out)
    mtime_second = out.stat().st_mtime

    assert mtime_second > mtime_first, "file was not rewritten after a CSV was removed"


# ---------------------------------------------------------------------------
# Validation / error exits
# ---------------------------------------------------------------------------

@pytest.mark.script
def test_content_with_raw_terminator_rejected(tmp_path):
    bad_content = "allreduce,0,1024,ring,ll,1,1,8\n" + RAW_TERMINATOR + "\n"
    write_csv(tmp_path, "rccl_tuner_gfx950.csv", bad_content)
    out = tmp_path / "out.h"

    result = run(tmp_path, out)
    assert result.returncode != 0
    assert not out.exists(), "output file should not be written on validation failure"


@pytest.mark.script
def test_filename_with_quote_rejected(tmp_path):
    try:
        write_csv(tmp_path, 'rccl_tuner_bad"name.csv', "allreduce,0,1024,ring,ll,1,1,8\n")
    except (OSError, ValueError):
        pytest.skip("filesystem does not allow quotes in filenames")
    out = tmp_path / "out.h"

    result = run(tmp_path, out)
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Real tuner/ data
# ---------------------------------------------------------------------------

def _real_csvs():
    return sorted(
        f for f in os.listdir(TUNER_DIR)
        if f.startswith("rccl_tuner") and f.endswith(".csv")
    )


@pytest.mark.real_data
def test_real_tuner_dir_succeeds(tmp_path):
    csvs = _real_csvs()
    if not csvs:
        pytest.skip("no rccl_tuner*.csv files found in tuner/")

    out = tmp_path / "rccl_tuner_embedded_configs.h"
    result = run(TUNER_DIR, out)
    assert result.returncode == 0, result.stderr

    header = out.read_text()
    assert f"{len(csvs)} config(s)" in header
    for name in csvs:
        assert f'"{name}"' in header


@pytest.mark.real_data
def test_real_tuner_content_roundtrip(tmp_path):
    """Every real CSV must survive the embed round-trip byte-for-byte.

    This mirrors EmbeddedBytesMatchSourceCsv in csv-tuner-test.cc, but at the
    Python layer: it guards that the script embeds the right bytes before the
    C++ test can even run.
    """
    csvs = _real_csvs()
    if not csvs:
        pytest.skip("no rccl_tuner*.csv files found in tuner/")

    out = tmp_path / "rccl_tuner_embedded_configs.h"
    run(TUNER_DIR, out)

    contents = raw_string_contents(out.read_text())
    assert len(contents) == len(csvs), (
        f"expected {len(csvs)} embedded blocks, found {len(contents)}"
    )

    for name, embedded in zip(csvs, contents):
        original = (TUNER_DIR / name).read_text(encoding="utf-8")
        assert embedded == original, f"round-trip mismatch for {name}"


@pytest.mark.real_data
def test_real_header_is_reproducible(tmp_path):
    """Two consecutive runs on the real data must produce identical output."""
    csvs = _real_csvs()
    if not csvs:
        pytest.skip("no rccl_tuner*.csv files found in tuner/")

    out = tmp_path / "rccl_tuner_embedded_configs.h"
    run(TUNER_DIR, out)
    first = out.read_text()

    time.sleep(0.05)
    run(TUNER_DIR, out)
    second = out.read_text()

    assert first == second
