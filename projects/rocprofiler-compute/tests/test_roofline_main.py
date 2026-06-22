# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit coverage for ``roofline.roofline_main``.

These tests drive ``Roofline`` directly with mocked machine specs and
self-contained temp CSVs; they never exercise the analyze CLI. 
"""

import tempfile
from pathlib import Path

import plotly.graph_objects as go

from roofline.roofline_main import Roofline


class MockArgs:
    """Minimal args namespace for Roofline construction."""

    def __init__(self, roofline_data_type: list[str]) -> None:
        self.roof_only = True
        self.mem_level = "ALL"
        self.sort = "ALL"
        self.roofline_data_type = roofline_data_type


def make_run_parameters(
    workload_dir: str, roofline_data_type: list[str], **extra: object
) -> dict[str, object]:
    """Build the run_parameters dict shared by the Roofline unit tests."""
    run_parameters: dict[str, object] = {
        "workload_dir": workload_dir,
        "device_id": 0,
        "sort_type": "kernels",
        "mem_level": "ALL",
        "is_standalone": True,
        "roofline_data_type": roofline_data_type,
    }
    run_parameters.update(extra)
    return run_parameters


class MockMspec:
    """Minimal MachineSpecs"""

    def __init__(self, gpu_model: str, gpu_series: str, gpu_arch: str) -> None:
        self.gpu_model = gpu_model
        self.gpu_series = gpu_series
        self.gpu_arch = gpu_arch


def mi200_mspec() -> MockMspec:
    return MockMspec("MI200", "mi200", "gfx90a")


def rdna_mspec() -> MockMspec:
    return MockMspec("rdna35_halo", "navi3", "gfx1151")


def write_wmma_roofline_csv(workload_dir: str) -> None:
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


def wmma_roofline_instance(workload_dir: str) -> Roofline:
    run_parameters = make_run_parameters(
        workload_dir, ["FP64", "BF16"], matrix_ops_type="WMMA"
    )
    return Roofline(MockArgs(["FP64", "BF16"]), rdna_mspec(), run_parameters)


def legend_names(fig: go.Figure) -> set[str]:
    return {trace.name for trace in fig.data}


# =============================================================================
# cli_generate_plot early-guard returns
# =============================================================================


def test_roofline_missing_file_handling() -> None:
    """cli_generate_plot with empty ai_data returns None at the ai_data guard."""
    run_parameters = make_run_parameters("", ["FP32"])
    roofline_instance = Roofline(MockArgs(["FP32"]), mi200_mspec(), run_parameters)

    result = roofline_instance.cli_generate_plot("FP32", ai_data={})
    assert result is None


def test_roofline_invalid_datatype_cli() -> None:
    """cli_generate_plot with an unsupported datatype returns None."""
    run_parameters = make_run_parameters("", ["FP32"])
    roofline_instance = Roofline(MockArgs(["FP32"]), mi200_mspec(), run_parameters)

    result = roofline_instance.cli_generate_plot("INVALID_DATATYPE", ai_data={})
    assert result is None


# =============================================================================
# WMMA (RDNA) legend coverage
# =============================================================================


def test_generate_plot_wmma_bf16_legend() -> None:
    """BF16 on RDNA emits a Peak WMMA-BF16 roof and no VALU roof."""
    with tempfile.TemporaryDirectory() as workload_dir:
        write_wmma_roofline_csv(workload_dir)
        roofline_instance = wmma_roofline_instance(workload_dir)

        # Pass an existing figure so the AI overlay (which needs ai_data) is
        # skipped; only the ceiling/legend traces are added.
        fig = roofline_instance.generate_plot("BF16", fig=go.Figure())

        names = " ".join(n for n in legend_names(fig) if n)
        assert "Peak WMMA-BF16" in names, "BF16 should emit a Peak WMMA-BF16 roof"
        assert "Peak MFMA-BF16" not in names, "RDNA path must not label roofs MFMA"
        assert "Peak VALU-BF16" not in names, "BF16 is matrix-only; no VALU roof"


def test_generate_plot_wmma_fp64_dual_legend() -> None:
    """FP64 on RDNA emits both a Peak VALU-FP64 and a Peak WMMA-FP64 roof."""
    with tempfile.TemporaryDirectory() as workload_dir:
        write_wmma_roofline_csv(workload_dir)
        roofline_instance = wmma_roofline_instance(workload_dir)

        fig = roofline_instance.generate_plot("FP64", fig=go.Figure())

        names = " ".join(n for n in legend_names(fig) if n)
        assert "Peak VALU-FP64" in names, "FP64 is dual-path; expected a VALU roof"
        assert "Peak WMMA-FP64" in names, "FP64 should emit a Peak WMMA-FP64 roof"
        assert "Peak MFMA-FP64" not in names, "RDNA path must not label roofs MFMA"
