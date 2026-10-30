# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/rocprof_compute_analyze/analysis_base.py."""

import argparse
import functools
from pathlib import Path
from types import SimpleNamespace

import common
import pandas as pd
import pytest

from rocprof_compute_analyze.analysis_base import OmniAnalyze_Base
from utils import schema

MODULE = "rocprof_compute_analyze.analysis_base"


def record_error_and_exit(errors, *args, **kwargs):
    """Record a console_error call, then exit like the real one."""
    errors.append(args)
    raise SystemExit(1)


def test_sanitize_rejects_paths_sharing_a_workload_name(tmp_path, monkeypatch) -> None:
    """Reject two paths whose last two components match."""
    mock_error = common.patch_console(monkeypatch, MODULE, "error")["error"]
    paths = [[str(tmp_path / parent / "vcopy" / "MI300")] for parent in ("a", "b")]
    for path in paths:
        Path(path[0]).mkdir(parents=True)

    # The mock records instead of exiting, so sanitize runs on to a later error.
    with pytest.raises(SystemExit):
        OmniAnalyze_Base(argparse.Namespace(path=paths), {}).sanitize()

    assert "last two components" in mock_error.call_args.args[1]


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


# =============================================================================
# membw_analysis_collected tests
# =============================================================================


@pytest.mark.parametrize(
    "profiling_config,expected",
    [
        pytest.param({"membw_analysis": True}, True, id="collected"),
        pytest.param({"membw_analysis": False}, False, id="not_collected"),
        pytest.param({}, False, id="absent"),
    ],
)
def test_membw_analysis_collected(profiling_config, expected) -> None:
    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    inst._profiling_config = profiling_config
    assert inst.membw_analysis_collected() is expected


def test_membw_analysis_collected_without_config_attribute() -> None:
    inst = OmniAnalyze_Base.__new__(OmniAnalyze_Base)
    assert inst.membw_analysis_collected() is False


@pytest.mark.parametrize(
    "flag_kwargs, expected_trace",
    [
        pytest.param(
            {
                "list_torch_operators": True,
                "list_triton_operators": False,
                "torch_operator": None,
                "triton_operator": None,
            },
            "--torch-trace",
            id="list_torch",
        ),
        pytest.param(
            {
                "list_torch_operators": False,
                "list_triton_operators": True,
                "torch_operator": None,
                "triton_operator": None,
            },
            "--triton-trace",
            id="list_triton",
        ),
        pytest.param(
            {
                "list_torch_operators": False,
                "list_triton_operators": False,
                "torch_operator": None,
                "triton_operator": ["*matmul*"],
            },
            "--triton-trace",
            id="triton_operator",
        ),
    ],
)
def test_sanitize_errors_when_operator_flag_without_trace(
    tmp_path, monkeypatch, flag_kwargs, expected_trace
):
    workload = tmp_path / "app" / "MI300"
    workload.mkdir(parents=True)
    (workload / "profiling_config.yaml").write_text(
        "torch_trace: false\ntriton_trace: false\nml_api_trace: false\n",
        encoding="utf-8",
    )
    errors = []
    monkeypatch.setattr(
        f"{MODULE}.console_error", functools.partial(record_error_and_exit, errors)
    )
    with pytest.raises(SystemExit):
        OmniAnalyze_Base(
            argparse.Namespace(
                tui=False,
                path=[[str(workload)]],
                **flag_kwargs,
            ),
            {},
        ).sanitize()
    assert "was not profiled" in errors[-1][1]
    assert expected_trace in errors[-1][1]


# ---------------------------------------------------------------------------
# pre_processing filter wiring
# ---------------------------------------------------------------------------


def _run_pre_processing(monkeypatch, paths, gpu_id):
    """Run the base pre_processing over fresh workloads and return them by path."""
    workloads = {path[0]: schema.Workload() for path in paths}
    args = argparse.Namespace(
        path=paths,
        gpu_kernel=None,
        gpu_id=gpu_id,
        gpu_dispatch_id=None,
    )
    analyzer = OmniAnalyze_Base(args, {})
    monkeypatch.setattr(analyzer, "initalize_runs", lambda: workloads)

    OmniAnalyze_Base.pre_processing(analyzer)
    return workloads


def test_single_path_takes_the_whole_gpu_id_list(monkeypatch) -> None:
    """Every id of --gpu-id reaches a single workload, so the validator sees all."""
    workloads = _run_pre_processing(monkeypatch, [["workload_0"]], ["0", "9"])

    assert workloads["workload_0"].filter_gpu_ids == ["0", "9"]


def test_gpu_id_pairs_with_paths_in_order(monkeypatch) -> None:
    """With several paths, --gpu-id pairs with them in order, as -k does."""
    paths = [["workload_0"], ["workload_1"]]

    workloads = _run_pre_processing(monkeypatch, paths, ["0", "1"])

    assert workloads["workload_0"].filter_gpu_ids == "0"
    assert workloads["workload_1"].filter_gpu_ids == "1"
