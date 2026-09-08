# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import argparse
import gzip
import sys
from pathlib import Path

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base

MODULE = "rocprof_compute_analyze.analysis_base"

# The args pre_processing() reads besides the output format. An empty path list
# leaves no workload to walk, so only the --output-format dispatch runs.
PRE_PROCESSING_ARGS = {
    "path": [],
    "gpu_kernel": None,
    "gpu_id": None,
    "gpu_dispatch_id": None,
}

# Stand-in for a rendered report. What tty.show_all puts in it is covered by
# tests/unit/utils/test_tty.py; here only the sink it lands in matters.
REPORT_TEXT = "30. Memory Bandwidth Analysis\n30.13 EA Interface\n"


def test_concat_result_csvs_concatenates_rocpd_results(tmp_path, monkeypatch) -> None:
    """Concatenates rocpd long-form results_*.csv.gz into one pmc_perf.csv.gz."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")

    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        header + "0,kernel_a,SQ_WAVES,10\n0,kernel_a,SQ_WAVES,20\n",
    )
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_1.csv.gz",
        header + "0,kernel_a,SQ_BUSY_CYCLES,30\n",
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.concat_result_csvs(
        sorted(tmp_path.glob("results_*.csv.gz")), common.pmc_perf_path(tmp_path)
    )
    merged = pd.read_csv(common.pmc_perf_path(tmp_path))

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
    (tmp_path / "results_pmc_perf_0.csv.gz").write_bytes(b"")
    (tmp_path / "results_pmc_perf_1.csv.gz").write_bytes(b"")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv.gz")),
            common.pmc_perf_path(tmp_path),
        )

    assert not (common.pmc_perf_path(tmp_path)).exists()
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
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        header + "0,kernel_a,SQ_WAVES,10\n",
    )
    (tmp_path / "results_pmc_perf_1.csv.gz").write_bytes(b"")

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.concat_result_csvs(
        sorted(tmp_path.glob("results_*.csv.gz")), common.pmc_perf_path(tmp_path)
    )

    assert pd.read_csv(common.pmc_perf_path(tmp_path))["Counter_Value"].tolist() == [10]


def test_join_workload_csvs_finds_compressed_results(tmp_path, monkeypatch) -> None:
    """join_workload_csvs picks up compressed results_*.csv.gz artifacts."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        header + "0,kernel_a,SQ_WAVES,10\n",
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.join_workload_csvs(tmp_path)

    assert pd.read_csv(common.pmc_perf_path(tmp_path))["Counter_Value"].tolist() == [10]


def test_join_workload_csvs_reuses_existing_merge(tmp_path, monkeypatch) -> None:
    """An existing merge wins over results_*.csv.gz instead of being rebuilt."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning", "log")

    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    common.write_pmc_perf(tmp_path, header + "0,kernel_a,SQ_WAVES,10\n")
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        header + "0,kernel_a,SQ_WAVES,99\n",
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst.join_workload_csvs(tmp_path)

    assert pd.read_csv(common.pmc_perf_path(tmp_path))["Counter_Value"].tolist() == [10]


def test_concat_result_csvs_errors_on_truncated_compressed_results(
    tmp_path, monkeypatch
) -> None:
    """Partial .csv.gz from a killed profile run must not leave output behind."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    rows = "".join(f"0,kernel_a,SQ_WAVES,{i}\n" for i in range(2000))
    whole = gzip.compress((header + rows).encode("utf-8"))
    (tmp_path / "results_pmc_perf_0.csv.gz").write_bytes(whole[: len(whole) // 2])

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv.gz")),
            common.pmc_perf_path(tmp_path),
        )

    assert not (common.pmc_perf_path(tmp_path)).exists()


def test_concat_result_csvs_errors_when_only_headers(tmp_path, monkeypatch) -> None:
    """Header-only results files must not leave a reusable output behind."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    header = "GPU_ID,Kernel_Name,Counter_Name,Counter_Value\n"
    common.write_gzip_csv(tmp_path / "results_pmc_perf_0.csv.gz", header)
    common.write_gzip_csv(tmp_path / "results_pmc_perf_1.csv.gz", header)

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv.gz")),
            common.pmc_perf_path(tmp_path),
        )

    assert not (common.pmc_perf_path(tmp_path)).exists()


def test_concat_result_csvs_rejects_wide_legacy_results(tmp_path, monkeypatch) -> None:
    """Wide legacy results_*.csv without Counter_Name are rejected."""
    common.patch_console(monkeypatch, MODULE, "debug", "warning")
    common.write_gzip_csv(
        tmp_path / "results_pmc_perf_0.csv.gz",
        "GPU_ID,Kernel_Name,Dispatch_ID,SQ_WAVES\n0,kernel_a,0,10\n",
    )

    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    with pytest.raises(SystemExit):
        inst.concat_result_csvs(
            sorted(tmp_path.glob("results_*.csv.gz")),
            common.pmc_perf_path(tmp_path),
        )

    assert not (common.pmc_perf_path(tmp_path)).exists()


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


# ---------------------------------------------------------------------------
# pre_processing output_format dispatch
# ---------------------------------------------------------------------------


def test_pre_processing_txt_creates_named_file(tmp_path, monkeypatch) -> None:
    """--output-format txt with --output-name writes <name>.txt in the cwd."""
    mocks = common.patch_console(monkeypatch, MODULE, "debug", "log", "warning")
    monkeypatch.setattr(OmniAnalyze_Base, "initalize_runs", lambda self: {})
    monkeypatch.chdir(tmp_path)

    analyzer = OmniAnalyze_Base(
        argparse.Namespace(
            output_format="txt", output_name="analysis_report", **PRE_PROCESSING_ARGS
        ),
        {},
    )
    analyzer.pre_processing()

    try:
        report = tmp_path / "analysis_report.txt"
        assert report.is_file()
        assert not analyzer._output.closed
        assert Path(analyzer._output.name).resolve() == report
        assert analyzer._output.writable()
        assert "analysis_report.txt" in mocks["warning"].call_args.args[1]
    finally:
        analyzer._output.close()


def test_pre_processing_txt_default_name_is_uuid(tmp_path, monkeypatch) -> None:
    """Without --output-name the txt file falls back to rocprof_compute_<uuid>."""
    common.patch_console(monkeypatch, MODULE, "debug", "log", "warning")
    monkeypatch.setattr(OmniAnalyze_Base, "initalize_runs", lambda self: {})
    monkeypatch.chdir(tmp_path)

    analyzer = OmniAnalyze_Base(
        argparse.Namespace(
            output_format="txt", output_name=None, **PRE_PROCESSING_ARGS
        ),
        {},
    )
    analyzer.pre_processing()

    try:
        created = list(tmp_path.iterdir())
        assert len(created) == 1
        assert created[0].match("rocprof_compute_*.txt")
    finally:
        analyzer._output.close()


def test_pre_processing_stdout_creates_no_file(tmp_path, monkeypatch) -> None:
    """--output-format stdout routes to the terminal and touches no file."""
    common.patch_console(monkeypatch, MODULE, "debug", "log", "warning")
    monkeypatch.setattr(OmniAnalyze_Base, "initalize_runs", lambda self: {})
    monkeypatch.chdir(tmp_path)

    analyzer = OmniAnalyze_Base(
        argparse.Namespace(
            output_format="stdout", output_name=None, **PRE_PROCESSING_ARGS
        ),
        {},
    )
    analyzer.pre_processing()

    assert analyzer._output is sys.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("output_format", ["txt", "stdout"])
def test_pre_processing_report_matches_across_output_formats(
    tmp_path, monkeypatch, capsys, output_format
) -> None:
    """Every --output-format receives byte-identical report content."""
    common.patch_console(monkeypatch, MODULE, "debug", "log", "warning")
    monkeypatch.setattr(OmniAnalyze_Base, "initalize_runs", lambda self: {})
    monkeypatch.chdir(tmp_path)

    analyzer = OmniAnalyze_Base(
        argparse.Namespace(
            output_format=output_format,
            output_name="analysis_report",
            **PRE_PROCESSING_ARGS,
        ),
        {},
    )
    analyzer.pre_processing()
    capsys.readouterr()

    try:
        analyzer._output.write(REPORT_TEXT)
        # The analyzer never closes _output, so flush before reading it back.
        analyzer._output.flush()
        if output_format == "txt":
            written = (tmp_path / "analysis_report.txt").read_text(encoding="utf-8")
        else:
            written = capsys.readouterr().out
    finally:
        if analyzer._output is not sys.stdout:
            analyzer._output.close()

    assert written == REPORT_TEXT
