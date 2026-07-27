# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import common
import pandas as pd

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base

MODULE = "rocprof_compute_analyze.analysis_base"


def test_join_prof_concatenates_rocpd_results_csvs(tmp_path, monkeypatch) -> None:
    """join_prof vertically concatenates the rocpd long-form results_*.csv files
    into a single pmc_perf.csv: the shared header is written once and every
    data row from every results file is preserved.
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
    inst.join_prof(tmp_path, out=str(tmp_path / "pmc_perf.csv"))
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
