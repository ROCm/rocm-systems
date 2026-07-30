# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base

MODULE = "rocprof_compute_analyze.analysis_base"


def test_concat_result_csvs_concatenates_rocpd_results(tmp_path, monkeypatch) -> None:
    """concat_result_csvs vertically concatenates the rocpd long-form
    results_*.csv files into a single pmc_perf.csv: the shared header is written
    once and every data row from every results file is preserved.
    """
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
    """A truncated results_*.csv is skipped rather than raising StopIteration,
    and a workload whose results files are all empty errors out instead of
    leaving behind a zero-byte pmc_perf.csv for analyze to misread.
    """
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


def test_concat_result_csvs_errors_when_only_headers(tmp_path, monkeypatch) -> None:
    """Header-only results files carry no counter rows; concat must error rather
    than leave a header-only pmc_perf.csv that a later analyze run reuses."""
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
    """Legacy CSV-backend results_*.csv are wide (no Counter_Name column); concat
    must reject them and tell the user to re-profile rather than build a
    pmc_perf.csv analyze would misread."""
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
