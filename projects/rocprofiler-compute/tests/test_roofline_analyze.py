# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Roofline coverage for the ``analyze`` roofline path.

MI200 (gfx90a) tests run through the ``analyze`` CLI and assert that roofline
HTML is generated, including the per-datatype VALU/MFMA legend. RDNA (WMMA)
tests drive ``Roofline.generate_plot`` directly and assert the ``Peak WMMA-``
legend labels.
"""

import os
import shutil
import tempfile
from pathlib import Path

import common
import pandas as pd
import plotly.graph_objects as go
import pytest

from roofline.roofline_main import Roofline
from utils.specs import generate_machine_specs

config = {}
config["cleanup"] = True

roofline_dir = "tests/workloads/mem_levels_HBM/MI200"

_, roofline_soc = common.gpu_soc()


def _skip_if_no_roofline_soc():
    if roofline_soc is None or roofline_soc == "":
        pytest.skip("No supported GPU detected")
    if roofline_soc == "MI100":
        pytest.skip("Roofline not supported on MI100")


def _make_mock_args(roofline_data_type):
    """Build a minimal args namespace for Roofline construction."""

    class MockArgs:
        def __init__(self):
            self.roof_only = True
            self.mem_level = "ALL"
            self.sort = "ALL"
            self.roofline_data_type = roofline_data_type

    return MockArgs()


def _make_run_parameters(workload_dir, roofline_data_type, **extra):
    """Build the run_parameters dict shared by the Roofline unit tests."""
    run_parameters = {
        "workload_dir": workload_dir,
        "device_id": 0,
        "sort_type": "kernels",
        "mem_level": "ALL",
        "is_standalone": True,
        "roofline_data_type": roofline_data_type,
    }
    run_parameters.update(extra)
    return run_parameters


# =============================================================================
# Roofline HTML generation
# =============================================================================


def test_analyze_generates_roofline_html(
    binary_handler_analyze_rocprof_compute,
):
    """
    Analyze generates roofline HTML from existing workload data.
    Uses MI200 workload with roofline.csv.
    """
    _skip_if_no_roofline_soc()

    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_datatype_independently(
    binary_handler_analyze_rocprof_compute,
):
    """
    Analyze with multiple data types.
    Verifies each datatype can be requested independently.
    """
    _skip_if_no_roofline_soc()

    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    for dtype in ["FP32", "FP64", "BF16"]:
        code = binary_handler_analyze_rocprof_compute([
            "analyze",
            "--path",
            workload_dir,
            "--roofline-data-type",
            dtype,
        ])
        assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_multiple_datatypes_single_invocation(
    binary_handler_analyze_rocprof_compute,
):
    """
    Analyze with multiple data types in a single invocation.
    Verifies the multi-datatype request path works end to end.
    """
    _skip_if_no_roofline_soc()

    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
        "FP64",
        "BF16",
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_missing_roofline_csv_graceful(
    binary_handler_analyze_rocprof_compute,
):
    """
    Analyze without roofline.csv should not crash.
    Uses a workload directory that has sysinfo.csv but no roofline.csv.
    """
    workload_dir = common.setup_workload_dir(roofline_dir)
    roofline_csv = Path(workload_dir) / "roofline.csv"
    if roofline_csv.exists():
        roofline_csv.unlink()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_roofline_idempotent(
    binary_handler_analyze_rocprof_compute,
):
    """
    Running analyze twice on the same profiling output should produce
    consistent results without errors.
    """
    _skip_if_no_roofline_soc()

    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    analyze_args = [
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
    ]

    code1 = binary_handler_analyze_rocprof_compute(analyze_args)
    assert code1 == 0

    code2 = binary_handler_analyze_rocprof_compute(analyze_args)
    assert code2 == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_analyze_corrupted_roofline_csv_graceful(
    binary_handler_analyze_rocprof_compute,
):
    """
    Analyze with a corrupted roofline.csv should handle gracefully.
    """
    if os.path.exists(roofline_dir):
        with tempfile.TemporaryDirectory() as temp_dir:
            workload_dir = os.path.join(temp_dir, "corrupted_workload")
            shutil.copytree(roofline_dir, workload_dir)

            roofline_csv = Path(workload_dir) / "roofline.csv"
            roofline_csv.write_text("this,is,bad,csv")

            code = binary_handler_analyze_rocprof_compute([
                "analyze",
                "-b",
                "4",
                "--path",
                workload_dir,
            ])
            assert code == 0


def test_roof_invalid_data_type(binary_handler_analyze_rocprof_compute):
    """Invalid --roofline-data-type should be caught by analyze argparser."""
    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "INVALID_TYPE",
    ])
    assert code != 0, "Invalid datatype should be rejected by argparser"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_roofline_ceiling_data_validation(binary_handler_analyze_rocprof_compute):
    """Invalid --mem-level should be caught during analyze."""
    workload_dir = common.setup_workload_dir(roofline_dir)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--mem-level",
        "INVALID_LEVEL",
    ])
    assert code != 0, "Invalid --mem-level should be rejected by argparser"

    common.clean_output_dir(config["cleanup"], workload_dir)


roofline_mem_level_dirs = {
    "vL1D": "tests/workloads/mem_levels_vL1D/MI200",
    "LDS": "tests/workloads/mem_levels_LDS/MI200",
}


@pytest.mark.parametrize(
    "mem_level",
    ["vL1D", "LDS"],
    ids=["vL1D", "LDS"],
)
def test_roof_mem_levels(binary_handler_analyze_rocprof_compute, mem_level):
    """Analyze with --mem-level generates roofline HTML output."""
    workload_src = roofline_mem_level_dirs[mem_level]
    if not os.path.exists(workload_src):
        pytest.skip(f"Workload directory {workload_src} not found")

    workload_dir = common.setup_workload_dir(workload_src, param_id=mem_level)

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--mem-level",
        mem_level,
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob("empirRoof_*.html"))
    assert len(html_files) > 0, "Analyze should generate roofline HTML files"

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_roofline_missing_file_handling():
    """cli_generate_plot with empty ai_data returns None."""

    args = _make_mock_args(["FP32"])
    workload_dir = common.setup_workload_dir(roofline_dir)
    sys_info = pd.read_csv(f"{workload_dir}/sysinfo.csv")
    sys_info_dict = {key: value[0] for key, value in sys_info.to_dict("list").items()}
    mspec = generate_machine_specs(args, sys_info_dict)

    run_parameters = _make_run_parameters(workload_dir, ["FP32"])

    roofline_instance = Roofline(args, mspec, run_parameters)
    result = roofline_instance.cli_generate_plot("FP32", ai_data={})
    assert result is None

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_roofline_invalid_datatype_cli():
    """cli_generate_plot with invalid datatype returns None."""

    args = _make_mock_args(["FP32"])

    workload_dir = common.setup_workload_dir(roofline_dir)
    sys_info = pd.read_csv(f"{workload_dir}/sysinfo.csv")
    sys_info_dict = {key: value[0] for key, value in sys_info.to_dict("list").items()}
    mspec = generate_machine_specs(args, sys_info_dict)

    run_parameters = _make_run_parameters(workload_dir, ["FP32"])

    roofline_instance = Roofline(args, mspec, run_parameters)
    result = roofline_instance.cli_generate_plot("INVALID_DATATYPE", ai_data={})
    assert result is None

    common.clean_output_dir(config["cleanup"], workload_dir)


# =============================================================================
# WMMA (RDNA) legend coverage
# =============================================================================


class _MockRDNAMspec:
    """Minimal MachineSpecs stand-in for the RDNA (WMMA) roofline path.

    generate_plot only reads gpu_model (memory-level resolution), gpu_series
    (matrix-op label via get_matrix_ops_type -> "WMMA"), and gpu_arch.
    """

    gpu_model = "rdna35_halo"
    gpu_series = "navi3"
    gpu_arch = "gfx1151"


def _write_wmma_roofline_csv(workload_dir):
    """Write a roofline.csv with RDNA BW + VALU + WMMA matrix columns."""
    # rdna35_halo memory levels resolve to LDS/L0/L1/L2 (MALL skipped).
    header = [
        "device",
        "LDSBw",
        "L0Bw",
        "L1Bw",
        "L2Bw",
        "FP64Flops",
        "WMMAF16Flops",
        "WMMABF16Flops",
        "WMMAF64Flops",
    ]
    row = ["0", "500", "500", "500", "500", "3000", "10000", "11000", "12000"]
    csv_path = Path(workload_dir) / "roofline.csv"
    content = ",".join(header) + "\n" + ",".join(row) + "\n"
    csv_path.write_text(content, encoding="utf-8")


def _wmma_roofline_instance(workload_dir):
    run_parameters = _make_run_parameters(
        workload_dir, ["FP64", "BF16"], matrix_ops_type="WMMA"
    )
    return Roofline(_make_mock_args(["FP64", "BF16"]), _MockRDNAMspec(), run_parameters)


def _legend_names(fig):
    return {trace.name for trace in fig.data}


def test_generate_plot_wmma_bf16_legend():
    """BF16 on RDNA emits a Peak WMMA-BF16 roof and no VALU roof."""
    with tempfile.TemporaryDirectory() as workload_dir:
        _write_wmma_roofline_csv(workload_dir)
        roofline_instance = _wmma_roofline_instance(workload_dir)

        # Pass an existing figure so the AI overlay (which needs ai_data) is
        # skipped; only the ceiling/legend traces are added.
        fig = roofline_instance.generate_plot("BF16", fig=go.Figure())

        names = " ".join(n for n in _legend_names(fig) if n)
        assert "Peak WMMA-BF16" in names, "BF16 should emit a Peak WMMA-BF16 roof"
        assert "Peak MFMA-BF16" not in names, "RDNA path must not label roofs MFMA"
        assert "Peak VALU-BF16" not in names, "BF16 is matrix-only; no VALU roof"


def test_generate_plot_wmma_fp64_dual_legend():
    """FP64 on RDNA emits both a Peak VALU-FP64 and a Peak WMMA-FP64 roof."""
    with tempfile.TemporaryDirectory() as workload_dir:
        _write_wmma_roofline_csv(workload_dir)
        roofline_instance = _wmma_roofline_instance(workload_dir)

        fig = roofline_instance.generate_plot("FP64", fig=go.Figure())

        names = " ".join(n for n in _legend_names(fig) if n)
        assert "Peak VALU-FP64" in names, "FP64 is dual-path; expected a VALU roof"
        assert "Peak WMMA-FP64" in names, "FP64 should emit a Peak WMMA-FP64 roof"
        assert "Peak MFMA-FP64" not in names, "RDNA path must not label roofs MFMA"


# =============================================================================
# Per-datatype HTML output validation
# =============================================================================

# Each datatype must take the correct roofline branch:
# FP64 is dual-path (VALU + MFMA), BF16 is MFMA-only.
DATATYPE_LEGEND_CASES = {
    "FP64": {"present": ["Peak VALU-FP64", "Peak MFMA-FP64"], "absent": []},
    "BF16": {"present": ["Peak MFMA-BF16"], "absent": ["Peak VALU-BF16"]},
}


@pytest.mark.parametrize("dtype", list(DATATYPE_LEGEND_CASES))
def test_analyze_roofline_datatype_html_legend(
    binary_handler_analyze_rocprof_compute, dtype
):
    """Per-datatype roofline HTML embeds the expected VALU/MFMA legend.

    FP8/FP4/FP6 are intentionally excluded: they are unsupported on the
    available gfx90a (MI200) test data and are covered by the unit tests.
    """
    _skip_if_no_roofline_soc()

    workload_dir = common.setup_workload_dir(roofline_dir, param_id=dtype)

    assert (Path(workload_dir) / "roofline.csv").exists()

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        dtype,
    ])
    assert code == 0

    html_files = list(Path(workload_dir).glob(f"empirRoof_*{dtype}*.html"))
    assert len(html_files) > 0, f"Analyze should generate a {dtype} roofline HTML"

    html_text = html_files[0].read_text(encoding="utf-8")
    for legend in DATATYPE_LEGEND_CASES[dtype]["present"]:
        assert legend in html_text, f"{dtype} HTML should contain '{legend}'"
    for legend in DATATYPE_LEGEND_CASES[dtype]["absent"]:
        assert legend not in html_text, f"{dtype} HTML should not contain '{legend}'"

    common.clean_output_dir(config["cleanup"], workload_dir)
