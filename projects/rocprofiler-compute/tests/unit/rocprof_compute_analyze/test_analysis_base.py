# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import argparse
import gzip
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import schema
from utils.specs import MachineSpecsCDNA
from utils.tty import show_all

MODULE = "rocprof_compute_analyze.analysis_base"

# Trimmed MI350 sysinfo.
MI350_SYSINFO = {
    "ip_blocks": "SQ|LDS|SQC|TA|TD|TCP|TCC|SPI|CPC|CPF|roofline",
    "version": "3",
    "compute_partition": "SPX",
    "memory_partition": "NPS1",
    "gpu_series": "MI350",
    "gpu_model": "MI350",
    "gpu_arch": "gfx950",
    "cu_per_gpu": "256",
    "total_l2_chan": "128",
    "num_xcd": "8",
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


def make_analyzer(
    monkeypatch: pytest.MonkeyPatch,
    output_format: str,
    output_name: str = None,
) -> OmniAnalyze_Base:
    """Return an analyzer wired for pre_processing() with no workload on disk.

    pre_processing() normally walks args.path to load sysinfo.csv and join
    counter files. An empty path list plus a stubbed initalize_runs() reduces
    it to the --output-format dispatch these tests are about.
    """
    monkeypatch.setattr(OmniAnalyze_Base, "initalize_runs", lambda self: OrderedDict())

    args = argparse.Namespace(
        output_format=output_format,
        output_name=output_name,
        path=[],
        gpu_kernel=None,
        gpu_id=None,
        gpu_dispatch_id=None,
    )
    analyzer = OmniAnalyze_Base(args, {})
    analyzer._profiling_config = {}
    return analyzer


def render_report(analyzer: OmniAnalyze_Base, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write one rendered panel to analyzer._output via the real tty renderer."""
    metric_dataframe = pd.DataFrame({
        "Metric": ["EA read request fraction - HBM"],
        "Avg": [50.0],
        "Unit": ["Percent"],
    })
    table_config = {
        "id": 3013,
        "title": "EA Interface",
        "header": {"metric": "Metric", "value": "Avg", "unit": "Unit"},
    }
    arch_configs = SimpleNamespace(
        panel_configs={
            3000: {
                "id": 3000,
                "title": "Memory Bandwidth Analysis",
                "data source": [{"metric_table": table_config}],
            }
        }
    )
    runs = {
        "fixture": SimpleNamespace(
            dfs={3013: metric_dataframe},
            sys_info=pd.DataFrame([{"gpu_arch": "gfx950"}]),
        )
    }
    monkeypatch.setattr(
        "utils.tty.process_table_data", lambda *_args, **_kwargs: metric_dataframe
    )

    show_all(
        argparse.Namespace(
            decimal=2,
            filter_metrics=None,
            include_cols=None,
            membw_analysis=True,
            normal_unit="per_wave",
            path=[["fixture"]],
            time_unit="ns",
            view=None,
        ),
        runs,
        arch_configs,
        analyzer._output,
        profiling_config={"filter_blocks": []},
    )


def test_pre_processing_txt_creates_named_file(tmp_path, monkeypatch) -> None:
    """--output-format txt with --output-name writes <name>.txt in the cwd."""
    mocks = common.patch_console(monkeypatch, MODULE, "debug", "log", "warning")
    monkeypatch.chdir(tmp_path)

    analyzer = make_analyzer(monkeypatch, "txt", output_name="analysis_report")
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
    monkeypatch.chdir(tmp_path)

    analyzer = make_analyzer(monkeypatch, "txt")
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
    monkeypatch.chdir(tmp_path)

    analyzer = make_analyzer(monkeypatch, "stdout")
    analyzer.pre_processing()

    assert analyzer._output is sys.stdout
    assert list(tmp_path.iterdir()) == []


def test_txt_output_matches_stdout_output(tmp_path, monkeypatch, capsys) -> None:
    """The txt file and the terminal receive byte-identical report content."""
    common.patch_console(monkeypatch, MODULE, "debug", "log", "warning")
    monkeypatch.chdir(tmp_path)

    txt_analyzer = make_analyzer(monkeypatch, "txt", output_name="analysis_report")
    txt_analyzer.pre_processing()
    try:
        render_report(txt_analyzer, monkeypatch)
        # The analyzer never closes _output, so flush before reading it back.
        txt_analyzer._output.flush()
        txt_report = (tmp_path / "analysis_report.txt").read_text(encoding="utf-8")
    finally:
        txt_analyzer._output.close()

    stdout_analyzer = make_analyzer(monkeypatch, "stdout")
    stdout_analyzer.pre_processing()
    capsys.readouterr()
    render_report(stdout_analyzer, monkeypatch)
    stdout_report = capsys.readouterr().out

    assert txt_report == stdout_report
    assert "30. Memory Bandwidth Analysis" in txt_report


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses directory write permissions",
)
def test_pre_processing_txt_unwritable_directory_raises(tmp_path, monkeypatch) -> None:
    """An unwritable cwd surfaces the raw OSError from the unguarded open()."""
    common.patch_console(monkeypatch, MODULE, "debug", "log", "warning")
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o555)
    monkeypatch.chdir(read_only)

    analyzer = make_analyzer(monkeypatch, "txt", output_name="analysis_report")
    try:
        with pytest.raises(PermissionError):
            analyzer.pre_processing()
    finally:
        read_only.chmod(0o755)


# ---------------------------------------------------------------------------
# initalize_runs --specs-correction handling
# ---------------------------------------------------------------------------


def make_specs_correction_analyzer(tmp_path, monkeypatch, specs_correction):
    """Analyzer wired to a minimal MI350 workload, ready for initalize_runs().

    Panel config generation is stubbed out so the run loop can be exercised
    without loading the real arch YAML.
    """
    # MI350_SYSINFO is trimmed, so silence the "missing specs" warnings the
    # spec-to-frame conversion emits for the fields it leaves unset.
    common.patch_console(monkeypatch, "utils.specs", "warning")
    pd.DataFrame([MI350_SYSINFO]).to_csv(tmp_path / "sysinfo.csv", index=False)

    args = argparse.Namespace(
        path=[[str(tmp_path)]],
        specs_correction=specs_correction,
        no_roof=True,
        normal_unit="per_kernel",
        gpu_kernel=None,
        list_stats=False,
        filter_metrics=None,
        tui=False,
        config_dir=str(tmp_path),
    )
    inst = OmniAnalyze_Base(args, {})
    monkeypatch.setattr(inst, "generate_configs", lambda *a, **kw: {})
    inst._arch_configs = {MI350_SYSINFO["gpu_arch"]: schema.ArchConfig()}
    inst.set_soc({
        MI350_SYSINFO["gpu_arch"]: SimpleNamespace(
            _mspec=MachineSpecsCDNA(**MI350_SYSINFO)
        )
    })
    return inst


def test_initalize_runs_applies_specs_correction(tmp_path, monkeypatch) -> None:
    """--specs-correction overrides the specs analysis runs on."""
    inst = make_specs_correction_analyzer(
        tmp_path, monkeypatch, "num_xcd:4,cu_per_gpu:64"
    )

    workload = inst.initalize_runs()[str(tmp_path)]

    assert workload.sys_info["num_xcd"].item() == "4"
    assert workload.sys_info["cu_per_gpu"].item() == "64"
    # Uncorrected specs come through untouched, including the ip_blocks entry
    # initalize_runs() reads right after applying the correction.
    assert workload.sys_info["gpu_arch"].item() == "gfx950"
    assert workload.sys_info["total_l2_chan"].item() == "128"
    assert "SQ" in workload.avail_ips
    assert "roofline" in workload.avail_ips


def test_initalize_runs_without_specs_correction_keeps_sysinfo(
    tmp_path, monkeypatch
) -> None:
    """Without the option the workload keeps the recorded sysinfo values."""
    inst = make_specs_correction_analyzer(tmp_path, monkeypatch, None)

    workload = inst.initalize_runs()[str(tmp_path)]

    assert workload.sys_info["num_xcd"].item() == 8
    assert workload.sys_info["cu_per_gpu"].item() == 256
