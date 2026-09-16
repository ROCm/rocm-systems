# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import argparse
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
