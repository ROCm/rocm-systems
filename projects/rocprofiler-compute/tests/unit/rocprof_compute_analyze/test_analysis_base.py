# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import argparse
import gzip
from pathlib import Path

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import csv_compression

MODULE = "rocprof_compute_analyze.analysis_base"


def test_concat_result_csvs_concatenates_rocpd_results(tmp_path, monkeypatch) -> None:
    """Concatenates rocpd long-form results_*.csv into one pmc_perf.csv."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")

    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    (tmp_path / "results_pmc_perf_0.csv").write_text(
        header + "0,kernel_a,SQ_WAVES,10\n0,kernel_a,SQ_WAVES,20\n"
    )
    (tmp_path / "results_pmc_perf_1.csv").write_text(
        header + "0,kernel_a,SQ_BUSY_CYCLES,30\n"
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.concat_result_csvs(
        sorted(tmp_path.glob("results_*.csv")), tmp_path / "pmc_perf.csv"
    )
    merged = pd.read_csv(tmp_path / "pmc_perf.csv")

    assert list(merged.columns) == [
        "GPU_ID",
        "Kernel_Name",
        "Counter_Name",
        "Counter_Value",
    ]
    assert len(merged) == 3
    assert set(merged["Counter_Name"]) == {"SQ_WAVES", "SQ_BUSY_CYCLES"}
    assert sorted(merged["Counter_Value"].tolist()) == [10, 20, 30]


def test_concat_result_csvs_skips_empty_and_errors_when_all_empty(
    tmp_path, monkeypatch
) -> None:
    mocks = common.patch_console(monkeypatch, MODULE, "debug", "warning")
    (tmp_path / "results_pmc_perf_0.csv").write_text("")
    (tmp_path / "results_pmc_perf_1.csv").write_text("")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv")), tmp_path / "pmc_perf.csv"
        )

    assert not (tmp_path / "pmc_perf.csv").exists()
    skipped = [
        call.args[0]
        for call in mocks["warning"].call_args_list
        if "Skipping empty" in str(call.args[0])
    ]
    assert len(skipped) == 2


def test_concat_result_csvs_skips_zero_byte_compressed_pass(
    tmp_path, monkeypatch
) -> None:
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    with gzip.open(tmp_path / "results_pmc_perf_0.csv.gz", "wt") as f:
        f.write(header + "0,kernel_a,SQ_WAVES,10\n")
    (tmp_path / "results_pmc_perf_1.csv.gz").write_bytes(b"")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.concat_result_csvs(
        csv_compression.find_csvs(tmp_path, "results_*.csv"), tmp_path / "pmc_perf.csv"
    )

    assert pd.read_csv(tmp_path / "pmc_perf.csv")["Counter_Value"].tolist() == [10]


def test_concat_result_csvs_mixes_compressed_and_plain(tmp_path, monkeypatch) -> None:
    """Passes are read in whichever form each one takes."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")

    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    (tmp_path / "results_pmc_perf_0.csv").write_text(
        header + "0,kernel_a,SQ_WAVES,10\n"
    )
    with gzip.open(tmp_path / "results_pmc_perf_1.csv.gz", "wt") as f:
        f.write(header + "0,kernel_a,SQ_BUSY_CYCLES,30\n")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.concat_result_csvs(
        csv_compression.find_csvs(tmp_path, "results_*.csv"), tmp_path / "pmc_perf.csv"
    )
    merged = pd.read_csv(tmp_path / "pmc_perf.csv")

    assert list(merged.columns) == [
        "GPU_ID",
        "Kernel_Name",
        "Counter_Name",
        "Counter_Value",
    ]
    assert sorted(merged["Counter_Value"].tolist()) == [10, 30]


def test_join_workload_csvs_finds_compressed_results(tmp_path, monkeypatch) -> None:
    """find_csvs matches compressed results_*.csv.gz artifacts."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    with gzip.open(tmp_path / "results_pmc_perf_0.csv.gz", "wt") as f:
        f.write(header + "0,kernel_a,SQ_WAVES,10\n")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.join_workload_csvs(tmp_path)

    assert pd.read_csv(tmp_path / "pmc_perf.csv")["Counter_Value"].tolist() == [10]


def test_concat_result_csvs_errors_on_truncated_compressed_results(
    tmp_path, monkeypatch
) -> None:
    """Partial .csv.gz from a killed profile run must not leave pmc_perf.csv behind."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    rows = "".join(f"0,kernel_a,SQ_WAVES,{i}\n" for i in range(2000))
    whole = gzip.compress((header + rows).encode("utf-8"))
    (tmp_path / "results_pmc_perf_0.csv.gz").write_bytes(whole[: len(whole) // 2])

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            csv_compression.find_csvs(tmp_path, "results_*.csv"),
            tmp_path / "pmc_perf.csv",
        )

    assert not (tmp_path / "pmc_perf.csv").exists()


def test_concat_result_csvs_errors_when_only_headers(tmp_path, monkeypatch) -> None:
    """Header-only results files must not leave a reusable pmc_perf.csv behind."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    (tmp_path / "results_pmc_perf_0.csv").write_text(header)
    (tmp_path / "results_pmc_perf_1.csv").write_text(header)

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv")), tmp_path / "pmc_perf.csv"
        )

    assert not (tmp_path / "pmc_perf.csv").exists()


def test_concat_result_csvs_rejects_wide_legacy_results(tmp_path, monkeypatch) -> None:
    """Wide legacy results_*.csv without Counter_Name are rejected."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    (tmp_path / "results_pmc_perf_0.csv").write_text(
        "GPU_ID,Kernel_Name,Dispatch_ID,SQ_WAVES\n0,kernel_a,0,10\n"
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv")), tmp_path / "pmc_perf.csv"
        )

    assert not (tmp_path / "pmc_perf.csv").exists()


def test_sanitize_rejects_paths_sharing_a_workload_name(tmp_path, monkeypatch) -> None:
    """Reject two paths whose last two components match."""
    mock_error = common.patch_console(monkeypatch, MODULE, "error")["error"]
    paths = [[str(tmp_path / parent / "vcopy" / "MI300")] for parent in ("a", "b")]
    for path in paths:
        Path(path[0]).mkdir(parents=True)

    # The mock records instead of exiting, so sanitize runs on to a later error.
    with pytest.raises(SystemExit):
        OmniAnalyze_Base(argparse.Namespace(tui=False, path=paths), {}).sanitize()

    assert "last two components" in mock_error.call_args.args[1]
