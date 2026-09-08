# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import argparse
import gzip
import sys
from pathlib import Path
from types import SimpleNamespace

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base

MODULE = "rocprof_compute_analyze.analysis_base"

# An empty path list leaves pre_processing() nothing to walk but the sink setup.
PRE_PROCESSING_ARGS = {
    "path": [],
    "gpu_kernel": None,
    "gpu_id": None,
    "gpu_dispatch_id": None,
}


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


# ---------------------------------------------------------------------------
# initalize_runs --specs-correction handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("specs_correction", [None, "num_xcd:4"])
def test_initalize_runs_corrects_specs_only_when_asked(
    tmp_path, monkeypatch, specs_correction
) -> None:
    """Without --specs-correction the recorded sysinfo.csv is what analysis runs on."""
    sysinfo = {
        "ip_blocks": "SQ|LDS|TCC|roofline",
        "gpu_arch": "gfx950",
        "num_xcd": 8,
    }
    pd.DataFrame([sysinfo]).to_csv(tmp_path / "sysinfo.csv", index=False)
    corrected = pd.DataFrame([{**sysinfo, "num_xcd": "4"}])
    monkeypatch.setattr(f"{MODULE}.parser.correct_sys_info", lambda *_args: corrected)

    analyzer = OmniAnalyze_Base(
        argparse.Namespace(
            path=[[str(tmp_path)]],
            specs_correction=specs_correction,
            no_roof=True,
            normal_unit="per_kernel",
            list_stats=False,
            filter_metrics=None,
            config_dir=str(tmp_path),
            gpu_kernel=None,
        ),
        {},
    )
    # Panel config generation reads the real arch YAML, which this test is not about.
    monkeypatch.setattr(analyzer, "generate_configs", lambda *_args: {})
    analyzer._arch_configs = {sysinfo["gpu_arch"]: SimpleNamespace(dfs={}, dfs_type={})}
    analyzer.set_soc({sysinfo["gpu_arch"]: SimpleNamespace(_mspec=object())})

    workload = analyzer.initalize_runs()[str(tmp_path)]

    expected_num_xcd = "4" if specs_correction else 8
    assert workload.sys_info["num_xcd"].item() == expected_num_xcd
    # initalize_runs reads ip_blocks off sys_info straight after the correction.
    assert workload.avail_ips == sysinfo["ip_blocks"].split("|")
