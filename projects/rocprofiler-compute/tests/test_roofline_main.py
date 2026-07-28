# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit coverage for roofline_main.py."""

import argparse
import math
import tempfile
from pathlib import Path

import plotly.graph_objects as go

from roofline.roofline_html import FRAME_MIN_DECADES
from roofline.roofline_main import (
    DEFAULT_AXIS_BOUNDS,
    Roofline,
    roofline_axis_bounds,
)


class MockMspec:
    """Minimal MachineSpecs"""

    def __init__(self, gpu_model: str, gpu_series: str, gpu_arch: str) -> None:
        self.gpu_model = gpu_model
        self.gpu_series = gpu_series
        self.gpu_arch = gpu_arch


def mi200_mspec() -> MockMspec:
    return MockMspec("MI210", "mi200", "gfx90a")


def make_roofline(
    mspec: MockMspec,
    roofline_data_type: list[str],
    workload_dir: str = "",
    **extra: object,
) -> Roofline:
    """Construct a Roofline for the unit tests.

    Roofline never reads its ``args`` argument on the cli_generate_plot /
    generate_plot paths, so a bare Namespace suffices.
    """
    run_parameters: dict[str, object] = {
        "workload_dir": workload_dir,
        "device_id": 0,
        "sort_type": "kernels",
        "mem_level": "ALL",
        "is_standalone": True,
        "roofline_data_type": roofline_data_type,
    }
    run_parameters.update(extra)
    return Roofline(argparse.Namespace(), mspec, run_parameters)


def write_mfma_roofline_csv(workload_dir: str) -> None:
    """Write a roofline.csv with CDNA BW + VALU + MFMA matrix columns."""
    # MI210 (gfx90a) memory levels resolve to LDS/L1/L2/HBM.
    header = [
        "device",
        "LDSBw",
        "HBMBw",
        "L1Bw",
        "L2Bw",
        "FP64Flops",
        "MFMAF16Flops",
        "MFMABF16Flops",
        "MFMAF64Flops",
    ]
    row = ["0", "500", "500", "500", "500", "3000", "10000", "11000", "12000"]
    csv_path = Path(workload_dir) / "roofline.csv"
    content = ",".join(header) + "\n" + ",".join(row) + "\n"
    csv_path.write_text(content, encoding="utf-8")


def mfma_roofline_instance(workload_dir: str) -> Roofline:
    return make_roofline(
        mi200_mspec(),
        ["FP64", "BF16"],
        workload_dir=workload_dir,
        matrix_ops_type="MFMA",
    )


def legend_names(fig: go.Figure) -> set[str]:
    return {trace.name for trace in fig.data}


# =============================================================================
# cli_generate_plot early-guard returns
# =============================================================================


def test_cli_generate_plot_empty_ai_data() -> None:
    """cli_generate_plot with empty ai_data returns None at the ai_data guard."""
    roofline_instance = make_roofline(mi200_mspec(), ["FP32"])

    result = roofline_instance.cli_generate_plot("FP32", ai_data={})
    assert result is None


def test_roofline_invalid_datatype_cli() -> None:
    """cli_generate_plot with an unsupported datatype returns None."""
    roofline_instance = make_roofline(mi200_mspec(), ["FP32"])

    result = roofline_instance.cli_generate_plot("INVALID_DATATYPE", ai_data={})
    assert result is None


# =============================================================================
# MFMA (CDNA) legend coverage
# =============================================================================


def test_generate_plot_mfma_bf16_legend() -> None:
    """BF16 on CDNA2 emits a Peak MFMA-BF16 roof and no VALU roof."""
    with tempfile.TemporaryDirectory() as workload_dir:
        write_mfma_roofline_csv(workload_dir)
        roofline_instance = mfma_roofline_instance(workload_dir)

        # Pass an existing figure so the AI overlay (which needs ai_data) is
        # skipped; only the ceiling/legend traces are added.
        fig = roofline_instance.generate_plot("BF16", fig=go.Figure())

        names = legend_names(fig)
        assert "Peak MFMA-BF16" in names, "BF16 should emit a Peak MFMA-BF16 roof"
        assert "Peak WMMA-BF16" not in names, "CDNA2 path must not label roofs WMMA"
        assert "Peak VALU-BF16" not in names, "BF16 is matrix-only; no VALU roof"


def test_generate_plot_mfma_fp64_dual_legend() -> None:
    """FP64 on CDNA2 emits both a Peak VALU-FP64 and a Peak MFMA-FP64 roof."""
    with tempfile.TemporaryDirectory() as workload_dir:
        write_mfma_roofline_csv(workload_dir)
        roofline_instance = mfma_roofline_instance(workload_dir)

        fig = roofline_instance.generate_plot("FP64", fig=go.Figure())

        names = legend_names(fig)
        assert "Peak VALU-FP64" in names, "FP64 is dual-path; expected a VALU roof"
        assert "Peak MFMA-FP64" in names, "FP64 should emit a Peak MFMA-FP64 roof"
        assert "Peak WMMA-FP64" not in names, "CDNA2 path must not label roofs WMMA"


# =============================================================================
# Opening axis frame
# =============================================================================


KNEE_AI = 10.0
PEAK_PERF = 5000.0
CEILING_LEFT_EDGE = 0.01
CEILING_RIGHT_EDGE = 1.0e6
BOUNDS_CEILINGS = {
    "hbm": [[CEILING_LEFT_EDGE, KNEE_AI], [5.0, PEAK_PERF], 500.0],
    "valu": [[KNEE_AI, CEILING_RIGHT_EDGE], [PEAK_PERF, PEAK_PERF], PEAK_PERF],
}
BOUNDS_KERNELS = {"ai_hbm": [[20.0, 40.0], [1000.0, 2000.0]]}


def test_axis_bounds_frame_the_roof_knee_not_its_drawn_edges() -> None:
    """The frame holds the knee and the peak while ignoring the endpoints the
    roofs are merely drawn to, which would otherwise open the plot decades wide
    around an empty corner."""
    x_lo, x_hi, y_lo, y_hi = roofline_axis_bounds(
        BOUNDS_CEILINGS, BOUNDS_KERNELS, ["HBM"]
    )

    assert x_lo <= KNEE_AI <= x_hi, "the knee anchors the intensity axis"
    assert y_lo <= PEAK_PERF <= y_hi, "the compute peak anchors the throughput axis"
    assert x_lo > CEILING_LEFT_EDGE * 10, "the roof's left edge must not anchor x"
    assert x_hi < CEILING_RIGHT_EDGE / 10, "the compute roof must not stretch x"


def test_axis_bounds_hold_every_kernel_point() -> None:
    """Every kernel drawn stays inside the opening frame."""
    x_lo, x_hi, y_lo, y_hi = roofline_axis_bounds(
        BOUNDS_CEILINGS, BOUNDS_KERNELS, ["HBM"]
    )

    ais, perfs = BOUNDS_KERNELS["ai_hbm"]
    for ai, perf in zip(ais, perfs):
        assert x_lo < ai < x_hi, f"kernel intensity {ai} fell outside the frame"
        assert y_lo < perf < y_hi, f"kernel throughput {perf} fell outside the frame"


def test_axis_bounds_widen_around_a_single_kernel() -> None:
    """One kernel sitting on the knee gives the axes nothing to span, so both are
    widened to the minimum rather than opening on a sliver."""
    x_lo, x_hi, y_lo, y_hi = roofline_axis_bounds(
        {"hbm": [[CEILING_LEFT_EDGE, KNEE_AI], [5.0, PEAK_PERF], 500.0]},
        {"ai_hbm": [[KNEE_AI], [PEAK_PERF]]},
        ["HBM"],
    )

    assert math.log10(x_hi / x_lo) >= FRAME_MIN_DECADES
    assert math.log10(y_hi / y_lo) >= FRAME_MIN_DECADES


def test_axis_bounds_fall_back_without_data() -> None:
    """No ceilings and no kernels leaves nothing to frame."""
    assert roofline_axis_bounds({}, {}, ["HBM"]) == DEFAULT_AXIS_BOUNDS
