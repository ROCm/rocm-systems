# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit coverage for the interactive roofline: the figures roofline_main draws,
the frame roofline_frame opens them on, and the model roofline_html ships to the
page.
"""

import argparse
import json
import math
import re
from pathlib import Path

import plotly.graph_objects as go
import pytest

import roofline.roofline_html as roofline_html
from roofline.roofline_frame import (
    FRAME_MIN_DECADES,
    FRAME_NOMINAL_ASPECT,
    FRAME_SLOPE_SKEW,
    FrameAnchors,
    frame_bounds,
)
from roofline.roofline_hover import wrap_hover_name
from roofline.roofline_html import RooflineViewModel
from roofline.roofline_main import Roofline

_ASSETS = Path(roofline_html.__file__).parent / "assets"


class MockMspec:
    """Minimal MachineSpecs: an MI210, so memory levels resolve to LDS/L1/L2/HBM
    and matrix ops are MFMA rather than WMMA."""

    gpu_model = "MI210"
    gpu_series = "mi200"
    gpu_arch = "gfx90a"


def make_roofline(datatypes: list[str], **run_parameters: object) -> Roofline:
    """A Roofline for the unit tests. It never reads its ``args`` on the
    cli_generate_plot / generate_plot paths, so a bare Namespace suffices."""
    parameters: dict[str, object] = {
        "workload_dir": "",
        "device_id": 0,
        "sort_type": "kernels",
        "mem_level": "ALL",
        "is_standalone": True,
        "roofline_data_type": datatypes,
    }
    parameters.update(run_parameters)
    return Roofline(argparse.Namespace(), MockMspec(), parameters)


@pytest.fixture
def benchmarked_roofline(tmp_path: Path):
    """Build a Roofline over MI210 benchmark data: bandwidth for every memory
    level, a scalar FP64 peak, and MFMA peaks. Takes the datatypes to stack,
    which is what the figure the tests read back off differs by.
    """
    header = (
        "device,LDSBw,HBMBw,L1Bw,L2Bw,FP64Flops,MFMAF16Flops,MFMABF16Flops,MFMAF64Flops"
    )
    row = "0,500,500,500,500,3000,10000,11000,12000"
    (tmp_path / "roofline.csv").write_text(f"{header}\n{row}\n", encoding="utf-8")

    def build(datatypes: list[str]) -> Roofline:
        return make_roofline(
            datatypes, workload_dir=str(tmp_path), matrix_ops_type="MFMA"
        )

    return build


@pytest.mark.parametrize("dtype", ["FP32", "INVALID_DATATYPE"])
def test_cli_generate_plot_returns_nothing_without_usable_input(dtype: str) -> None:
    """A datatype this arch cannot be profiled for, and a datatype with no AI
    data, are both declined rather than half-plotted."""
    assert make_roofline(["FP32"]).cli_generate_plot(dtype, ai_data={}) is None


@pytest.mark.parametrize(
    "dtype, drawn, not_drawn",
    [
        ("BF16", ["Peak MFMA-BF16"], ["Peak VALU-BF16", "Peak WMMA-BF16"]),
        ("FP64", ["Peak VALU-FP64", "Peak MFMA-FP64"], ["Peak WMMA-FP64"]),
    ],
)
def test_generate_plot_draws_the_roofs_the_datatype_reaches(
    benchmarked_roofline, dtype: str, drawn: list[str], not_drawn: list[str]
) -> None:
    """Each datatype gets one compute roof per op class it reaches on this arch:
    BF16 is matrix-only where FP64 is dual-path. On CDNA the matrix roofs are
    labeled MFMA, never WMMA."""
    fig = benchmarked_roofline(["FP64", "BF16"]).generate_plot(dtype, fig=go.Figure())

    names = {trace.name for trace in fig.data}
    assert names.issuperset(drawn)
    assert names.isdisjoint(not_drawn)


CEILING = {"hbm": [[0.01, 1.0], [1.0, 1500.0], 1500.0]}
COMPUTE_PEAKS = [("FP32 VALU", 9000.0), ("FP32 MFMA", 90000.0)]


def kernel_traces(roofline: Roofline, ai_data: dict, **overrides):
    """The traces and client model roofline_main builds for one AI dataset."""
    roofline._Roofline__ai_data = ai_data
    arguments: dict = {
        "kernel_names": ai_data["kernelNames"],
        "kernel_colors": ["#123456", "#654321"][: len(ai_data["kernelNames"])],
        "sanitized_cache_hierarchy": ["HBM"],
        "ceiling_data": CEILING,
        "ops_flops": "FLOP",
        "compute_peaks": COMPUTE_PEAKS,
    }
    arguments.update(overrides)
    return roofline._build_kernel_traces(**arguments)


def pct_roof(kernel: dict, point_index: int = 0) -> float:
    """The percent-of-roofline the tooltip shows for one of a kernel's points."""
    return float(kernel["points"][point_index]["hoverCells"][1])


def test_kernel_traces_score_against_the_tallest_drawn_ceiling() -> None:
    """A stacked figure caps points at the tallest compute roof drawn, so the
    reported peak and limiter do not depend on the order datatypes were
    stacked."""
    ai_data = {"ai_hbm": [[100.0], [50000.0]], "kernelNames": ["kA"]}

    matrix_traces, matrix_capped = kernel_traces(make_roofline(["FP32"]), ai_data)
    valu_traces, valu_capped = kernel_traces(
        make_roofline(["FP32"]), ai_data, compute_peaks=[("FP32 VALU", 9000.0)]
    )

    assert pct_roof(matrix_capped[0]) < 100.0
    assert pct_roof(valu_capped[0]) > 100.0
    assert "Performance limiter: FP32 MFMA" in matrix_traces[0].hovertemplate
    assert "Performance limiter: FP32 VALU" in valu_traces[0].hovertemplate


def test_kernel_traces_name_the_roof_that_binds() -> None:
    """A kernel whose bandwidth roof sits under the compute cap is limited by its
    memory level, and falls back to Unknown when the ceiling data holds no roof
    for that level at all. Levels with no positive AI are not plotted."""
    traces, model = kernel_traces(
        make_roofline(["FP32"]),
        {
            "ai_hbm": [[1.0], [900.0]],
            "ai_l2": [[0.0], [0.0]],
            "kernelNames": ["kA", "kB"],
        },
        sanitized_cache_hierarchy=["HBM", "L2"],
    )
    assert [kernel["name"] for kernel in model] == ["kA"]
    assert [point["peak"] for point in model[0]["points"]] == ["HBM"]
    assert "Performance limiter: HBM" in traces[0].hovertemplate

    unroofed_traces, unroofed = kernel_traces(
        make_roofline(["FP32"]),
        {"ai_hbm": [[1.0], [900.0]], "kernelNames": ["kA"]},
        ceiling_data={},
        compute_peaks=[],
    )
    assert "Performance limiter: Unknown" in unroofed_traces[0].hovertemplate
    assert unroofed[0]["points"][0]["hoverCells"] == ["N/A", "N/A"]


def test_kernel_hover_carries_the_whole_name() -> None:
    """A long demangled name reaches the tooltip whole. It is wrapped onto as
    many lines as it takes, but nothing is dropped: two instantiations of the
    same function are told apart by template arguments that run to the very end
    of the name."""
    name = "Cijk_Alik_Bljk_" + "SB_MT256x256x16_MI32x32x2x1_" * 40

    traces, _ = kernel_traces(
        make_roofline(["FP32"]),
        {"ai_hbm": [[1.0], [900.0]], "kernelNames": [name]},
    )

    wrapped = wrap_hover_name(name)
    assert wrapped in traces[0].hovertemplate
    lines = wrapped.split(">", 1)[1].removesuffix("</span>")
    assert lines.replace("<br>", "") == name


BANDWIDTH = 500.0
PEAK_PERF = 5000.0
KNEE_AI = PEAK_PERF / BANDWIDTH
KERNELS = [(20.0, 1000.0), (40.0, 2000.0)]

FLOAT_SLACK = 1e-9

any_viewport = pytest.mark.parametrize("aspect", [0.4, 1.0, FRAME_NOMINAL_ASPECT, 4.0])


def one_roof_anchors() -> FrameAnchors:
    return FrameAnchors(
        points=[(KNEE_AI, PEAK_PERF)] + list(KERNELS),
        throughputs=[PEAK_PERF],
        bandwidths=[BANDWIDTH],
    )


def assert_roofs_enter_through_the_bottom(
    bounds: tuple[float, float, float, float],
    bandwidths: list[float],
) -> None:
    """Every diagonal crosses the bottom edge inside the frame, so the slope
    itself is on screen instead of being clipped away by the left edge."""
    x_lo, x_hi, y_lo, _ = bounds
    for bandwidth in bandwidths:
        entry_ai = y_lo / bandwidth
        assert x_lo <= entry_ai * (1 + FLOAT_SLACK), (
            f"the {bandwidth} roof was cut off by the left edge"
        )
        assert entry_ai < x_hi, f"the {bandwidth} roof is off the right of the frame"


@any_viewport
def test_frame_holds_every_anchor(aspect: float) -> None:
    """The dots, and the corner every roofline is read against, stay in view:
    shaping and the minimum width may open the frame up but never close it back
    over an anchor."""
    x_lo, x_hi, y_lo, y_hi = frame_bounds(one_roof_anchors(), aspect=aspect)

    for ai, perf in [(KNEE_AI, PEAK_PERF)] + KERNELS:
        assert x_lo < ai < x_hi, f"intensity {ai} fell outside the frame"
        assert y_lo < perf < y_hi, f"throughput {perf} fell outside the frame"


@any_viewport
def test_frame_shows_the_whole_of_every_slope(aspect: float) -> None:
    """A roof is read by its slope, not just its corner, so the frame reaches
    left far enough for the diagonal to enter through the bottom edge -- and
    neither the minimum width nor the shaping may lift it back off."""
    assert_roofs_enter_through_the_bottom(
        frame_bounds(one_roof_anchors(), aspect=aspect), [BANDWIDTH]
    )


def test_frame_follows_the_slopes_when_stacking_lifts_the_knees() -> None:
    """Stacking datatypes caps every diagonal at the tallest ceiling, lifting the
    knees decades above the kernels. The frame has to follow the slopes down to
    the kernels instead of opening on the knees alone."""
    tall_peak = 40 * PEAK_PERF
    stacked = FrameAnchors(
        points=[(tall_peak / BANDWIDTH, tall_peak)] + list(KERNELS),
        throughputs=[PEAK_PERF, tall_peak],
        bandwidths=[BANDWIDTH],
    )

    bounds = frame_bounds(stacked)

    assert_roofs_enter_through_the_bottom(bounds, [BANDWIDTH])
    _, _, y_lo, _ = bounds
    assert math.log10(tall_peak / y_lo) >= math.log10(
        tall_peak / min(perf for _, perf in KERNELS)
    )


@any_viewport
def test_frame_keeps_roofs_near_45_degrees(aspect: float) -> None:
    """A roof reads at the same angle whatever the window: whichever axis is
    cramped is widened until the diagonal is within the skew allowance."""
    x_lo, x_hi, y_lo, y_hi = frame_bounds(one_roof_anchors(), aspect=aspect)

    screen_slope = math.log10(x_hi / x_lo) / (aspect * math.log10(y_hi / y_lo))
    assert 1 / FRAME_SLOPE_SKEW <= screen_slope <= FRAME_SLOPE_SKEW, (
        f"a roof reads at slope {screen_slope:.2f} in a {aspect} viewport"
    )


def test_frame_opens_on_a_minimum_width() -> None:
    """One kernel sitting on the knee gives the axes nothing to span, so the
    intensity axis opens on the minimum rather than on a sliver."""
    x_lo, x_hi, _, _ = frame_bounds(
        FrameAnchors(
            points=[(KNEE_AI, PEAK_PERF)],
            throughputs=[PEAK_PERF],
            bandwidths=[BANDWIDTH],
        )
    )

    assert math.log10(x_hi / x_lo) >= FRAME_MIN_DECADES


def test_a_taller_ceiling_does_not_widen_the_intensity_axis() -> None:
    """Compute ceilings run off the right edge, so a taller ceiling raises the
    frame without dragging the intensity axis out with it."""
    _, x_hi, _, y_hi = frame_bounds(one_roof_anchors())

    taller = one_roof_anchors()
    taller.throughputs.append(10 * PEAK_PERF)
    _, taller_x_hi, _, taller_y_hi = frame_bounds(taller)

    assert taller_y_hi > y_hi, "the taller ceiling has to be in view"
    assert taller_x_hi == x_hi, "a flat ceiling must not widen the intensity axis"


def test_frame_bounds_without_anchors() -> None:
    """Nothing to frame is reported rather than guessed at."""
    assert frame_bounds(FrameAnchors()) is None
    assert frame_bounds(FrameAnchors(bandwidths=[BANDWIDTH])) is None
    assert frame_bounds(FrameAnchors(points=[(0.0, PEAK_PERF)])) is None


FRAME_MAX_DECADES = 6.0


def drawn_roof_knees(fig: go.Figure) -> dict[str, tuple[float, float]]:
    """The knee each bandwidth roof is drawn to, read back off the figure."""
    return {
        trace.name: (trace.x[-1], trace.y[-1])
        for trace in fig.data
        if trace.mode == "lines" and not str(trace.name).startswith("Peak")
    }


def stacked_figure(benchmarked_roofline, datatypes: list[str]):
    """The figure and Roofline for these datatypes stacked onto one axis."""
    roofline = benchmarked_roofline(datatypes)
    fig = None
    for dtype in datatypes:
        fig = roofline.generate_plot(dtype, fig=fig)
    return roofline, fig


@pytest.mark.parametrize("datatypes", [["FP64"], ["FP64", "BF16"]])
def test_the_figure_opens_on_the_geometry_it_draws(
    benchmarked_roofline, datatypes: list[str]
) -> None:
    """The frame is built from where the diagonals turn over, not the
    extrapolated endpoints they are drawn out to. Stacking datatypes caps every
    diagonal at the tallest ceiling drawn, which lifts the knees decades above
    the kernels: the frame then has to open where the steepest roof crosses the
    bottom edge, or the slopes are clipped away against the left edge and only
    their corners survive."""
    _, fig = stacked_figure(benchmarked_roofline, datatypes)

    x_lo, x_hi = (10**bound for bound in fig.layout.xaxis.range)
    y_lo, y_hi = (10**bound for bound in fig.layout.yaxis.range)
    knees = drawn_roof_knees(fig)
    assert knees, "expected bandwidth roofs to frame"
    for level, (knee_ai, knee_perf) in knees.items():
        assert x_lo < knee_ai < x_hi, f"{level}'s knee fell outside the frame"
        assert y_lo < knee_perf < y_hi, f"{level}'s knee fell outside the frame"
    assert_roofs_enter_through_the_bottom(
        (x_lo, x_hi, y_lo, y_hi),
        [knee_perf / knee_ai for knee_ai, knee_perf in knees.values()],
    )
    assert math.log10(x_hi / x_lo) < FRAME_MAX_DECADES
    assert math.log10(y_hi / y_lo) < FRAME_MAX_DECADES


def test_view_model_carries_the_drawn_knee(benchmarked_roofline) -> None:
    """The client frames on the knees the model ships, so each one has to be the
    knee that figure really draws: capped at its tallest ceiling, including the
    ceilings a stacked datatype brought with it."""
    roofline, fig = stacked_figure(benchmarked_roofline, ["FP64", "BF16"])
    view_model = roofline._Roofline__view_models["FLOP"]

    knees = drawn_roof_knees(fig)
    assert view_model.roofline_traces, "expected bandwidth roofs in the model"
    for roof in view_model.roofline_traces:
        drawn_ai, drawn_perf = knees[roof["level"]]
        assert roof["kneeAi"] == pytest.approx(drawn_ai)
        assert roof["kneePerf"] == pytest.approx(drawn_perf)


def test_construct_plotly_figures_all_datatypes_ignores_cli_selection(
    benchmarked_roofline,
) -> None:
    """GUI-style generation attempts all supported architecture datatypes even
    when the shared analyze arguments selected only one terminal datatype."""
    roofline = benchmarked_roofline(["FP64"])

    ops_figure, flops_figure, _, _ = roofline.construct_plotly_figures(
        {"kernelNames": []}, datatypes=None
    )

    assert ops_figure is None
    assert flops_figure is not None
    trace_names = {trace.name for trace in flops_figure.data}
    assert "Peak MFMA-BF16" in trace_names
    assert "Peak VALU-FP64" in trace_names


def test_view_model_to_json_escapes_script_close() -> None:
    model = RooflineViewModel(kernels=[{"name": "evil</script>", "points": []}])

    serialized = model.to_json()

    assert "</script>" not in serialized, "must not allow a script element to close"
    assert json.loads(serialized)["kernels"][0]["name"] == "evil</script>"


def test_the_controller_looks_up_controls_the_page_renders() -> None:
    """Every control the controller reaches for by id has to be one the document
    renders. Nothing would report the two drifting apart: the control would
    simply stop working."""
    controller = (_ASSETS / "roofline_plot.js").read_text(encoding="utf-8")
    page_template = roofline_html._read_asset("roofline_plot.html")

    looked_up = set(re.findall(r'getElementById\("([^"]+)"\)', controller))
    assert looked_up, "expected the controller to find its controls by id"
    for element_id in sorted(looked_up):
        assert f'id="{element_id}"' in page_template, (
            f"the controller looks up #{element_id}, which the page never renders"
        )


def test_the_dark_theme_is_named_the_same_in_every_asset() -> None:
    """The page sets this class before its first paint, the stylesheet colors it,
    and the toggle flips it. One name in three files, or a reader's theme silently
    stops following either of them."""
    dark_class = "roofline-theme-dark"
    page_template = roofline_html._read_asset("roofline_plot.html")
    css = (_ASSETS / "roofline_plot.css").read_text(encoding="utf-8")
    controller = (_ASSETS / "roofline_plot.js").read_text(encoding="utf-8")

    assert f'classList.add("{dark_class}")' in page_template
    assert f":root.{dark_class}" in css
    assert f'"{dark_class}"' in controller
