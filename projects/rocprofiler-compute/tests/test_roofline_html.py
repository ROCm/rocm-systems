# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit coverage for the interactive roofline HTML layer."""

import argparse
import json

import plotly.graph_objects as go

from roofline.roofline_html import (
    PLOT_DIV_ID,
    RooflineViewModel,
    build_interactive_document,
)
from roofline.roofline_main import Roofline, build_kernel_colors
from utils.kernel_name_shortener import format_kernel_signature


class MockMspec:
    def __init__(self, gpu_model: str, gpu_series: str, gpu_arch: str) -> None:
        self.gpu_model = gpu_model
        self.gpu_series = gpu_series
        self.gpu_arch = gpu_arch


def make_roofline() -> Roofline:
    run_parameters: dict[str, object] = {
        "workload_dir": "",
        "device_id": 0,
        "sort_type": "kernels",
        "mem_level": "ALL",
        "roofline_data_type": ["FP32"],
    }
    mspec = MockMspec("MI200", "mi200", "gfx90a")
    return Roofline(argparse.Namespace(), mspec, run_parameters)


# A ceiling_data with two compute peaks (VALU + matrix), enough structure for
# _determine_kernel_bound_status and the active-compute-cap math.
CEILING = {
    "hbm": [[0.01, 1.0], [1.0, 1500.0], 1500.0],
    "l2": [[0.01, 1.0], [1.0, 3000.0], 3000.0],
    "valu": [[0.01, 1000.0], [9000.0, 9000.0], 9000.0],
    "matrix_ops": [[0.01, 1000.0], [90000.0, 90000.0], 90000.0],
}


def make_view_model() -> RooflineViewModel:
    return RooflineViewModel(
        peaks=["L2", "HBM"],
        peak_colors={"L2": "#009E73", "HBM": "#D55E00"},
        default_peak="HBM",
        kernels=[
            {
                "name": "kA",
                "color": "#123456",
                "traceIndex": 0,
                "points": [
                    {"peak": "HBM", "ai": 1.2, "perf": 300.0, "status": "Memory"}
                ],
            }
        ],
        kernel_trace_indices=[0],
    )


# =============================================================================
# Kernel-name display formatting (trims to function(argument types))
# =============================================================================


def test_format_kernel_signature_trims_return_type_and_template() -> None:
    assert (
        format_kernel_signature("void ns::foo<double>(double*, double*, int)")
        == "ns::foo(double*, double*, int)"
    )


def test_format_kernel_signature_passthrough_without_args() -> None:
    # Tensile / plain kernel names have no argument list; leave them unchanged.
    for name in ("Cijk_Ailk_Bjlk_SB_MT128x64x32", "sgprbound", ""):
        assert format_kernel_signature(name) == name


# =============================================================================
# Per-kernel trace builder (circles-only; memory level lives in the model)
# =============================================================================


def test_build_kernel_traces_one_marker_trace_per_kernel() -> None:
    roofline = make_roofline()
    # kernel "kA" has an L2 and an HBM point; "kB" only an L2 point (its HBM
    # entry is zeroed out and must be dropped).
    roofline._Roofline__ai_data = {
        "ai_l2": [[2.0, 5.0], [200.0, 400.0]],
        "ai_hbm": [[1.0, 0.0], [200.0, 0.0]],
        "kernelNames": ["kA", "kB"],
    }
    colors = build_kernel_colors(2)

    traces, model = roofline._build_kernel_traces(
        kernel_names=["kA", "kB"],
        kernel_colors=colors,
        sanitized_cache_hierarchy=["HBM", "L2"],
        ceiling_data=CEILING,
        ops_flops="FLOP",
    )

    assert len(traces) == 2
    assert [t.name for t in traces] == ["kA", "kB"]
    # One color per kernel; points are drawn as uniform circles (memory level
    # is encoded by color/model, not by a per-peak marker shape).
    assert traces[0].mode == "markers"
    assert traces[0].marker.color == colors[0]
    assert traces[1].marker.color == colors[1]
    assert traces[0].marker.symbol is None, "points are circles, not per-peak shapes"

    # The memory level of each point is carried in the view model, not the trace.
    assert [p["peak"] for p in model[0]["points"]] == ["L2", "HBM"]
    assert [p["peak"] for p in model[1]["points"]] == ["L2"]
    # pctRoof is measured against the active (highest) compute cap, so it never
    # exceeds 100% even though this datatype has both a VALU and a matrix peak.
    assert all(p["pctRoof"] <= 100.0 for k in model for p in k["points"])

    # customdata carries the fully-rendered per-point hover; the kernel name is
    # embedded in it.
    assert list(traces[0].customdata[0]) == [model[0]["points"][0]["hover"]]
    assert "kA" in model[0]["points"][0]["hover"]


# =============================================================================
# View-model serialization + document assembler
# =============================================================================


def test_view_model_to_json_escapes_script_close() -> None:
    model = RooflineViewModel(kernels=[{"name": "evil</script>", "points": []}])
    serialized = model.to_json()
    assert "</script>" not in serialized, "must not allow a script element to close"
    # Still valid JSON that decodes back to the original kernel name.
    assert json.loads(serialized)["kernels"][0]["name"] == "evil</script>"


def test_build_interactive_document_includes_controls_and_model() -> None:
    fig = go.Figure()
    # A roof line whose legend name must survive into the document text.
    fig.add_trace(go.Scatter(x=[0.01, 1.0], y=[1.0, 1500.0], name="HBM"))
    fig.add_trace(go.Scatter(x=[1.2], y=[300.0], name="kA", mode="markers"))

    document = build_interactive_document(fig, make_view_model(), title="Doc")

    for marker in [
        "roofline-peak-select",
        "roofline-show-all",
        "roofline-reset-view",
        "roofline-export-png",
        'id="roofline-model"',
        "roofline-kernel-list",
        "roofline-roof-list",
        PLOT_DIV_ID,
        "Plotly.newPlot",
    ]:
        assert marker in document, f"document missing {marker!r}"


def test_build_interactive_document_is_self_contained() -> None:
    """plotly.js is inlined (offline), not pulled from a CDN script tag."""
    document = build_interactive_document(go.Figure(), RooflineViewModel())
    assert '<script src="https://cdn.plot.ly' not in document
    assert "<!DOCTYPE html>" in document
    # The inlined library makes the document large; a CDN reference would not.
    assert len(document) > 1_000_000
