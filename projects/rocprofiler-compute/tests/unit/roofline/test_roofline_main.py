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
    FRAME_X_MIN,
    canonical_frame,
    points_outside_frame,
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


def test_canonical_frame_snaps_every_bound_to_a_decade() -> None:
    bounds = canonical_frame([5301.0, 10003.0], [81007.0, 163009.0])

    assert bounds is not None
    for bound in bounds:
        assert math.log10(bound).is_integer()


def test_canonical_frame_matches_the_diagnosis_gpu() -> None:
    bandwidths = [5300.0, 10000.0, 30000.0, 50000.0]
    peaks = [81000.0, 163000.0]

    assert canonical_frame(bandwidths, peaks) == (1e-2, 1e2, 1e1, 1e6)


def test_two_kernel_sets_open_on_the_same_frame(benchmarked_roofline) -> None:
    slow_data = {
        "ai_hbm": [[0.5], [2000.0]],
        "kernelNames": ["slow"],
    }
    fast_data = {
        "ai_hbm": [[40.0], [20000.0]],
        "kernelNames": ["fast"],
    }

    slow_figure = benchmarked_roofline(["FP64"]).construct_plotly_figures(
        slow_data, datatypes=["FP64"]
    )[1]
    fast_figure = benchmarked_roofline(["FP64"]).construct_plotly_figures(
        fast_data, datatypes=["FP64"]
    )[1]

    assert slow_figure.layout.xaxis.range == fast_figure.layout.xaxis.range
    assert slow_figure.layout.yaxis.range == fast_figure.layout.yaxis.range


def test_a_taller_ceiling_never_narrows_the_intensity_axis() -> None:
    _, x_hi, _, y_hi = canonical_frame([BANDWIDTH], [PEAK_PERF])
    _, taller_x_hi, _, taller_y_hi = canonical_frame([BANDWIDTH], [10 * PEAK_PERF])

    assert taller_x_hi >= x_hi
    assert taller_y_hi > y_hi


def test_canonical_frame_guards_degenerate_spans() -> None:
    assert canonical_frame([1.0], [1e-3]) == (FRAME_X_MIN, 1e-1, 1e-2, 1e-1)


def test_canonical_frame_requires_bandwidths_and_peaks() -> None:
    """Nothing to frame is reported rather than guessed at."""
    assert canonical_frame([], []) is None
    assert canonical_frame([BANDWIDTH], []) is None
    assert canonical_frame([], [PEAK_PERF]) is None
    assert canonical_frame([0.0, math.inf], [PEAK_PERF]) is None
    assert canonical_frame([BANDWIDTH], [0.0, math.nan]) is None


FRAME = (1e-2, 1e2, 1e0, 1e5)


def test_points_outside_frame_measures_how_far_past_each_edge() -> None:
    """Inside is silence; outside is a signed count of decades per axis, so a
    caller can say which edge a point went past and by how much."""
    outside = points_outside_frame(
        FRAME,
        [
            (1.0, 100.0),  # inside
            (1.0, 0.1),  # a decade under the performance axis
            (1e3, 1e6),  # a decade past both upper bounds
            (0.0, 100.0),  # a log axis cannot place this at all
        ],
    )

    assert outside == [
        (1, 0.0, pytest.approx(-1.0)),
        (2, pytest.approx(1.0), pytest.approx(1.0)),
        (3, -math.inf, 0.0),
    ]


def test_a_kernel_under_the_frame_is_reported_and_left_where_it_is(
    benchmarked_roofline, caplog
) -> None:
    """A kernel the machine frame does not reach is named in a warning, drawn at
    its true position, and never allowed to widen the frame: the axes have to
    stay the ones the next run will open on too."""
    sunken = {
        "ai_hbm": [[1.0, 1.0], [0.1, 500.0]],
        "kernelNames": ["sunken", "framed"],
    }
    inside = {"ai_hbm": [[1.0], [500.0]], "kernelNames": ["framed"]}

    with caplog.at_level("WARNING"):
        figure = benchmarked_roofline(["FP64"]).construct_plotly_figures(
            sunken, datatypes=["FP64"]
        )[1]
    reference = benchmarked_roofline(["FP64"]).construct_plotly_figures(
        inside, datatypes=["FP64"]
    )[1]

    warnings = [
        record.message for record in caplog.records if "falls outside" in record.message
    ]
    assert len(warnings) == 1, "expected one warning, for the one off-plot kernel"
    assert "sunken" in warnings[0]
    assert "1.00 decades below the performance axis" in warnings[0]

    assert figure.layout.yaxis.range == reference.layout.yaxis.range
    assert figure.layout.xaxis.range == reference.layout.xaxis.range
    drawn = {trace.name: trace.y for trace in figure.data if trace.mode == "markers"}
    assert drawn["sunken"] == (0.1,), "the marker must not be pulled onto the edge"


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
def test_the_figure_opens_on_the_machine_frame(
    benchmarked_roofline, datatypes: list[str]
) -> None:
    """The figure uses the file-wide frame and keeps every drawn roof knee visible."""
    _, fig = stacked_figure(benchmarked_roofline, datatypes)

    x_lo, x_hi = (10**bound for bound in fig.layout.xaxis.range)
    y_lo, y_hi = (10**bound for bound in fig.layout.yaxis.range)
    knees = drawn_roof_knees(fig)
    assert knees, "expected bandwidth roofs to frame"
    for level, (knee_ai, knee_perf) in knees.items():
        assert x_lo < knee_ai < x_hi, f"{level}'s knee fell outside the frame"
        assert y_lo < knee_perf < y_hi, f"{level}'s knee fell outside the frame"
    assert math.log10(x_hi / x_lo) < FRAME_MAX_DECADES
    assert math.log10(y_hi / y_lo) < FRAME_MAX_DECADES


def test_view_model_carries_the_drawn_knee(benchmarked_roofline) -> None:
    """The knee the model ships has to be the knee that figure really draws:
    capped at its tallest ceiling, including the ceilings a stacked datatype
    brought with it."""
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


def html_document(roofline: Roofline, ai_data: dict) -> tuple[str, go.Figure, dict]:
    """The standalone page one AI dataset produces, with the figure and client
    model it was built from."""
    ops_figure, flops_figure, _, _ = roofline.construct_plotly_figures(
        ai_data, datatypes=["FP64"]
    )
    figure, view_model = roofline._combined_html_figure(ops_figure, flops_figure)
    document = roofline_html.build_interactive_document(figure, view_model)
    return document, figure, view_model.frame


def embedded_model(document: str) -> dict:
    """The client model the page ships, read back out of its script tag."""
    match = re.search(
        r'<script id="roofline-model" type="application/json">(.*?)</script>',
        document,
        re.DOTALL,
    )
    assert match, "expected the page to embed a client model"
    return json.loads(match.group(1).replace("<\\/", "</"))


def axis_ranges(figure: go.Figure) -> dict[str, list[float]]:
    """The figure's axis bounds back in data coordinates."""
    return {
        "x": [10**bound for bound in figure.layout.xaxis.range],
        "y": [10**bound for bound in figure.layout.yaxis.range],
    }


def test_the_page_opens_on_the_frame_the_figure_was_drawn_on(
    benchmarked_roofline,
) -> None:
    """One frame in three places: the figure layout, the client model, and the
    JSON the page embeds. Any drift between them is an axis that jumps the
    moment the page loads."""
    document, figure, frame = html_document(
        benchmarked_roofline(["FP64"]),
        {"ai_hbm": [[0.5], [2000.0]], "kernelNames": ["slow"]},
    )

    assert frame is not None
    assert axis_ranges(figure) == pytest.approx(frame)
    assert embedded_model(document)["frame"] == frame


def test_two_kernel_sets_ship_the_same_axes_in_the_page(benchmarked_roofline) -> None:
    """The end of the ticket: same GPU, different kernels, same axes on screen,
    so a before and after plot overlay."""
    slow_document, slow_figure, _ = html_document(
        benchmarked_roofline(["FP64"]),
        {"ai_hbm": [[0.5], [2000.0]], "kernelNames": ["slow"]},
    )
    fast_document, fast_figure, _ = html_document(
        benchmarked_roofline(["FP64"]),
        {"ai_hbm": [[40.0], [20000.0]], "kernelNames": ["fast"]},
    )

    assert axis_ranges(slow_figure) == axis_ranges(fast_figure)
    assert (
        embedded_model(slow_document)["frame"] == embedded_model(fast_document)["frame"]
    )


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
