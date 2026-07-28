# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import logging
from types import SimpleNamespace

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils.utils_common import validate_profiling_format

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


@pytest.mark.parametrize(
    "profiling_config",
    [
        pytest.param({"format_rocprof_output": "rocpd"}, id="rocpd"),
        # Predates the format key: an already-joined wide pmc_perf.csv, which
        # analyze still reads.
        pytest.param({}, id="key_absent"),
    ],
)
def test_validate_profiling_format_accepts(profiling_config) -> None:
    """rocpd workloads and pre-key workloads are both analyzable."""
    validate_profiling_format(profiling_config)


@pytest.mark.parametrize("output_format", ["csv", "json"])
def test_validate_profiling_format_rejects_removed_backend(output_format) -> None:
    """A workload that declares a format other than rocpd was written by a
    backend whose per-counter CSVs analyze no longer merges, so it is rejected
    rather than silently misread."""
    with pytest.raises(SystemExit):
        validate_profiling_format(
            {"format_rocprof_output": output_format}, "/workloads/legacy"
        )


def test_validate_profiling_format_error_names_the_workload(caplog) -> None:
    """The rejection points at the offending directory."""
    with pytest.raises(SystemExit), caplog.at_level(logging.ERROR):
        validate_profiling_format(
            {"format_rocprof_output": "csv"}, "/workloads/legacy/MI200"
        )

    assert "/workloads/legacy/MI200" in caplog.text
    assert "csv" in caplog.text


def test_sanitize_rejects_legacy_workload_in_every_analyze_mode(
    tmp_path, monkeypatch
) -> None:
    """Every analyze mode routes through OmniAnalyze_Base.sanitize, so a legacy
    CSV workload is refused before any mode-specific processing runs."""
    (tmp_path / "profiling_config.yaml").write_text("format_rocprof_output: csv\n")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    args = SimpleNamespace(
        path=[[str(tmp_path)]],
        tui=False,
        output_name=None,
        output_format="txt",
        subpath=None,
    )
    monkeypatch.setattr(inst, "get_args", lambda: args, raising=False)

    with pytest.raises(SystemExit):
        inst.sanitize()
