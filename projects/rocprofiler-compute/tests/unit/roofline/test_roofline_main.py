# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit coverage for the interactive roofline: the figures roofline_main draws,
the frame roofline_frame opens them on, and the model roofline_html ships to the
page.
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import plotly.graph_objects as go
import pytest

import roofline.roofline_html as roofline_html
from roofline.roofline_frame import (
    FRAME_X_MIN,
    canonical_frame,
    points_outside_frame,
)
from roofline.roofline_hover import wrap_hover_name
from roofline.roofline_html import RooflineViewModel, build_interactive_document

if TYPE_CHECKING:
    from roofline.roofline_main import Roofline

_ASSETS = Path(__file__).resolve().parents[3] / "src" / "roofline" / "assets"


class MockMspec:
    """Minimal MachineSpecs: an MI210, so memory levels resolve to LDS/L1/L2/HBM
    and matrix ops are MFMA rather than WMMA."""

    gpu_model = "MI210"
    gpu_series = "mi200"
    gpu_arch = "gfx90a"


def make_roofline(datatypes: List[str], **run_parameters: object) -> "Roofline":
    """A Roofline for the unit tests. It never reads its ``args`` on the
    cli_generate_plot / generate_plot paths, so a bare Namespace suffices."""
    from roofline.roofline_main import Roofline

    parameters: Dict[str, object] = {
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

    def build(datatypes: List[str]):
        return make_roofline(
            datatypes, workload_dir=str(tmp_path), matrix_ops_type="MFMA"
        )

    return build


def layout_bounds(figure: go.Figure) -> List[List[float]]:
    """Return Plotly's two log ranges converted back to data coordinates."""
    return [
        [10**bound for bound in figure.layout.xaxis.range],
        [10**bound for bound in figure.layout.yaxis.range],
    ]


def embedded_model(document: str) -> dict:
    """Parse the JSON model from a standalone roofline document."""
    match = re.search(
        r'<script id="roofline-model" type="application/json">(.*?)</script>',
        document,
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def interactive_document(roofline, ai_data: dict) -> Tuple[go.Figure, str]:
    """Build the combined standalone figure and document for one kernel set."""
    ops_figure, flops_figure, _, _ = roofline.construct_plotly_figures(
        ai_data, datatypes=["FP64"]
    )
    figure, view_model = roofline._combined_html_figure(ops_figure, flops_figure)
    assert figure is not None
    return figure, build_interactive_document(figure, view_model)


def is_whole_decade(value: float) -> bool:
    """True when value is a positive power of ten."""
    if value <= 0:
        return False
    exponent = math.log10(value)
    return math.isclose(exponent, round(exponent))


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
    benchmarked_roofline, dtype: str, drawn: List[str], not_drawn: List[str]
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


def kernel_traces(roofline, ai_data: dict, **overrides):
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
    suffix = "</span>"
    lines = wrapped.split(">", 1)[1]
    assert lines.endswith(suffix)
    lines = lines[: -len(suffix)]
    assert lines.replace("<br>", "") == name


def test_canonical_frame_bounds_are_whole_decades() -> None:
    """Every axis limit lands on a power of ten."""
    bounds = canonical_frame([5300.0, 10000.0], [81000.0, 163000.0])
    assert bounds is not None
    for bound in bounds:
        assert is_whole_decade(bound)


def test_canonical_frame_matches_machine_ceilings() -> None:
    """MI210-style bandwidths and peaks open the expected canonical frame."""
    bandwidths = [5300.0, 10000.0, 30000.0, 50000.0]
    peaks = [81000.0, 163000.0]
    assert canonical_frame(bandwidths, peaks) == (1e-2, 1e2, 1e1, 1e6)


def test_canonical_frame_taller_peak_raises_y_without_narrowing_x() -> None:
    """A taller compute ceiling raises the top edge without shrinking x."""
    bandwidths = [5300.0, 10000.0]
    frame = canonical_frame(bandwidths, [81000.0])
    taller_frame = canonical_frame(bandwidths, [810000.0])
    assert frame is not None
    assert taller_frame is not None
    _, x_hi, _, y_hi = frame
    _, taller_x_hi, _, taller_y_hi = taller_frame

    assert taller_y_hi > y_hi
    assert taller_x_hi >= x_hi


def test_canonical_frame_degenerate_ceilings() -> None:
    """A tiny peak against a single bandwidth still yields a valid decade frame."""
    assert canonical_frame([1.0], [1e-3]) == (FRAME_X_MIN, 1e-1, 1e-2, 1e-1)


def test_canonical_frame_requires_bandwidths_and_peaks() -> None:
    """Missing or invalid machine ceilings cannot define a frame."""
    assert canonical_frame([], []) is None
    assert canonical_frame([500.0], []) is None
    assert canonical_frame([], [5000.0]) is None
    assert canonical_frame([0.0, float("nan")], [5000.0]) is None
    assert canonical_frame([500.0], [0.0, float("inf")]) is None


@pytest.mark.parametrize(
    "bandwidths, peaks",
    [
        ([5e-324], [1.0]),
        ([5e-324], [5e-324]),
    ],
)
def test_canonical_frame_extreme_inputs_return_none_without_crashing(
    bandwidths: List[float], peaks: List[float]
) -> None:
    """Unrepresentable decade bounds fall back to None instead of crashing."""
    assert canonical_frame(bandwidths, peaks) is None


def test_canonical_frame_extreme_magnitude_mix_representable() -> None:
    """Huge bandwidth with a subnormal peak still yields a whole-decade frame."""
    bounds = canonical_frame([1e308], [5e-324])
    assert bounds is not None
    for bound in bounds:
        assert is_whole_decade(bound)


def test_canonical_frame_covers_peak_just_above_decade() -> None:
    """A peak just above 10 must widen y_high and x_high past the decade edge."""
    peak = math.nextafter(10.0, math.inf)
    bounds = canonical_frame([1.0], [peak])
    assert bounds == (FRAME_X_MIN, 1e2, FRAME_X_MIN, 1e2)
    _, _, _, y_high = bounds
    assert y_high >= peak


def test_canonical_frame_subnormal_degenerate_preserves_decade_alignment() -> None:
    """Subnormal ceilings widen y via integer exponents, not reverse log10."""
    bounds = canonical_frame([1e-314], [1e-318])
    assert bounds == (FRAME_X_MIN, 1e-1, 1e-317, 1e-316)
    _, _, y_low, y_high = bounds
    assert is_whole_decade(y_low)
    assert is_whole_decade(y_high)
    assert int(round(math.log10(y_high))) == int(round(math.log10(y_low))) + 1


def test_points_outside_frame_detects_nextafter_above_high() -> None:
    """Values just above a decade high must still register as outside."""
    frame = (1e-2, 1e2, 1e0, 10.0)
    point = (1.0, math.nextafter(10.0, math.inf))
    outside = points_outside_frame(frame, [point])
    assert len(outside) == 1
    assert outside[0][0] == 0
    assert outside[0][2] > 0.0


def test_points_outside_frame_reports_signed_decades() -> None:
    """Overflow is signed in decades; boundary values stay inside."""
    frame = (1e-2, 1e2, 1e0, 1e5)
    points = [(1.0, 100.0), (1.0, 0.1), (1e3, 1e6), (0.0, 100.0)]

    assert points_outside_frame(frame, points) == [
        (1, 0.0, -1.0),
        (2, 1.0, 1.0),
        (3, -math.inf, 0.0),
    ]


def drawn_roof_knees(fig: go.Figure) -> Dict[str, Tuple[float, float]]:
    """The knee each bandwidth roof is drawn to, read back off the figure."""
    return {
        trace.name: (trace.x[-1], trace.y[-1])
        for trace in fig.data
        if trace.mode == "lines" and not str(trace.name).startswith("Peak")
    }


def stacked_figure(benchmarked_roofline, datatypes: List[str]):
    """The figure and Roofline for these datatypes stacked onto one axis."""
    roofline = benchmarked_roofline(datatypes)
    fig = None
    for dtype in datatypes:
        fig = roofline.generate_plot(dtype, fig=fig)
    return roofline, fig


@pytest.mark.parametrize("datatypes", [["FP64"], ["FP64", "BF16"]])
def test_the_figure_uses_the_machine_frame_while_preserving_roof_knees(
    benchmarked_roofline, datatypes: List[str]
) -> None:
    """The viewport comes from all machine ceilings, while drawn knees still
    reflect the tallest compute ceiling in the stacked figure."""
    _, fig = stacked_figure(benchmarked_roofline, datatypes)

    x_lo, x_hi = (10**bound for bound in fig.layout.xaxis.range)
    y_lo, y_hi = (10**bound for bound in fig.layout.yaxis.range)
    assert (x_lo, x_hi, y_lo, y_hi) == pytest.approx((1e-2, 1e2, 1.0, 1e5))

    knees = drawn_roof_knees(fig)
    assert knees, "expected bandwidth roofs to frame"
    for level, (knee_ai, knee_perf) in knees.items():
        assert x_lo < knee_ai < x_hi, f"{level}'s knee fell outside the frame"
        assert y_lo < knee_perf < y_hi, f"{level}'s knee fell outside the frame"

    assert [knee[1] for knee in knees.values()] == pytest.approx([12000.0] * len(knees))


def test_kernel_under_the_frame_warns_without_clamping(
    benchmarked_roofline, caplog
) -> None:
    """A kernel below y_low warns once, keeps true coordinates, and leaves axes
    on the machine frame."""
    ai_data = {
        "ai_hbm": [[1.0, 1.0], [0.1, 500.0]],
        "kernelNames": ["sunken", "framed"],
    }
    inside_only = {"ai_hbm": [[1.0], [500.0]], "kernelNames": ["framed"]}

    with caplog.at_level(logging.WARNING):
        _, figure, _, _ = benchmarked_roofline(["FP64"]).construct_plotly_figures(
            ai_data, datatypes=["FP64"]
        )
        _, reference, _, _ = benchmarked_roofline(["FP64"]).construct_plotly_figures(
            inside_only, datatypes=["FP64"]
        )

    falls_outside = [r for r in caplog.records if "falls outside" in r.message]
    assert len(falls_outside) == 1
    assert "sunken" in falls_outside[0].message
    assert "1.00 decades below the performance axis" in falls_outside[0].message

    assert layout_bounds(figure) == layout_bounds(reference)

    sunken_trace = next(trace for trace in figure.data if trace.name == "sunken")
    assert sunken_trace.y == (0.1,)


def test_kernel_on_frame_edge_does_not_warn(benchmarked_roofline, caplog) -> None:
    """A point exactly on the canonical frame edge stays inside and silent."""
    # Match benchmarked_roofline CSV row bandwidths and peaks.
    frame = canonical_frame(
        [500.0, 500.0, 500.0, 500.0], [3000.0, 10000.0, 11000.0, 12000.0]
    )
    assert frame is not None
    x_lo, x_hi, y_lo, y_hi = frame
    ai_data = {
        "ai_hbm": [[x_lo], [y_lo]],
        "kernelNames": ["on_edge"],
    }

    with caplog.at_level(logging.WARNING):
        benchmarked_roofline(["FP64"]).construct_plotly_figures(
            ai_data, datatypes=["FP64"]
        )

    falls_outside = [r for r in caplog.records if "falls outside" in r.message]
    assert not falls_outside


def test_kernel_data_does_not_change_machine_axis_ranges(benchmarked_roofline) -> None:
    """Slow and fast kernels on separate objects open on identical machine axes."""
    slow = {"ai_hbm": [[0.5], [2000.0]], "kernelNames": ["slow"]}
    fast = {"ai_hbm": [[40.0], [20000.0]], "kernelNames": ["fast"]}

    _, slow_figure, _, _ = benchmarked_roofline(["FP64"]).construct_plotly_figures(
        slow, datatypes=["FP64"]
    )
    _, fast_figure, _, _ = benchmarked_roofline(["FP64"]).construct_plotly_figures(
        fast, datatypes=["FP64"]
    )

    assert layout_bounds(slow_figure) == layout_bounds(fast_figure)


def test_combined_document_embeds_the_plotly_frame(benchmarked_roofline) -> None:
    """The standalone model ships the same data-coordinate frame Plotly uses."""
    figure, document = interactive_document(
        benchmarked_roofline(["FP64"]),
        {"ai_hbm": [[0.5], [2000.0]], "kernelNames": ["slow"]},
    )

    embedded_frame = embedded_model(document)["frame"]
    layout_frame = layout_bounds(figure)
    assert embedded_frame["x"] == pytest.approx(layout_frame[0])
    assert embedded_frame["y"] == pytest.approx(layout_frame[1])


@pytest.mark.parametrize("source_key", ["FLOP", "OP"])
def test_combined_model_copies_either_source_frame(source_key: str) -> None:
    """The combined model retains the frame from either available figure class."""
    roofline = make_roofline(["FP64"])
    source_frame = {"x": [1e-2, 1e2], "y": [1e1, 1e6]}
    # No public API exists for pre-seeding the per-figure view models.
    roofline._Roofline__view_models = {
        source_key: RooflineViewModel(frame=source_frame)
    }
    ops_figure = go.Figure() if source_key == "OP" else None
    flops_figure = go.Figure() if source_key == "FLOP" else None

    _, combined_model = roofline._combined_html_figure(ops_figure, flops_figure)

    assert combined_model.frame == source_frame
    assert combined_model.frame is not source_frame
    combined_model.frame["x"][0] = 1e-9
    assert source_frame["x"] == [1e-2, 1e2]


def test_kernel_sets_ship_identical_embedded_frames(benchmarked_roofline) -> None:
    """Changing only kernel coordinates cannot alter the browser's reset frame."""
    _, slow_document = interactive_document(
        benchmarked_roofline(["FP64"]),
        {"ai_hbm": [[0.5], [2000.0]], "kernelNames": ["slow"]},
    )
    _, fast_document = interactive_document(
        benchmarked_roofline(["FP64"]),
        {"ai_hbm": [[40.0], [20000.0]], "kernelNames": ["fast"]},
    )

    assert (
        embedded_model(slow_document)["frame"] == embedded_model(fast_document)["frame"]
    )


def test_title_names_the_stable_frame_across_precisions_and_combining(
    benchmarked_roofline,
) -> None:
    """Stacking precision roofs and combining figures preserves one frame title."""
    roofline, figure = stacked_figure(benchmarked_roofline, ["FP64", "BF16"])
    expected_title = (
        "Empirical Roofline Analysis<br><sup>Axes fixed to this GPU - "
        "AI 1e-2 to 1e2 - performance 1e0 to 1e5</sup>"
    )

    assert figure.layout.title.text == expected_title
    combined, _ = roofline._combined_html_figure(None, figure)
    assert combined.layout.title.text == expected_title


def test_fallback_frame_has_truthful_subtitle(
    benchmarked_roofline, monkeypatch, caplog
) -> None:
    """Fallback ranges identify their source and warn about off-frame kernels."""
    roofline = benchmarked_roofline(["FP64"])
    machine_figure = roofline.generate_plot("FP64")
    assert "Axes fixed to this GPU" in machine_figure.layout.title.text

    monkeypatch.setattr(
        "roofline.roofline_main.machine_ceilings",
        lambda *_args: ([], []),
    )
    with caplog.at_level(logging.WARNING):
        _, fallback_figure, _, _ = roofline.construct_plotly_figures(
            {
                "ai_hbm": [[1e4], [1e7]],
                "kernelNames": ["outside_default_frame"],
            },
            datatypes=["FP64"],
        )

    assert layout_bounds(fallback_figure) == [[1e-2, 1e3], [1.0, 1e6]]
    assert fallback_figure.layout.title.text == (
        "Empirical Roofline Analysis<br><sup>Default axes - benchmark ceilings "
        "unavailable - AI 1e-2 to 1e3 - performance 1e0 to 1e6</sup>"
    )
    falls_outside = [r.message for r in caplog.records if "falls outside" in r.message]
    assert len(falls_outside) == 1
    assert "current fixed roofline frame" in falls_outside[0]
    assert "GPU ceilings" not in falls_outside[0]


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


def standalone_with_unselected_precision(benchmarked_roofline):
    """The combined document for a run whose opening precision is not its
    tallest, so the default selection is observable in what gets drawn."""
    roofline, flops_figure = stacked_figure(benchmarked_roofline, ["BF16", "FP64"])
    figure, view_model = roofline._combined_html_figure(None, flops_figure)
    assert figure is not None
    assert view_model.default_precisions == ["BF16"]
    return figure, view_model


def test_standalone_figure_opens_on_only_the_default_precisions(
    benchmarked_roofline,
) -> None:
    """The page opens with one precision selected, so the document ships the
    other ceilings already hidden. Leaving them visible would paint every roof
    until the client's first restyle lands."""
    figure, view_model = standalone_with_unselected_precision(benchmarked_roofline)
    selected = set(view_model.default_precisions)
    assert selected != set(view_model.precisions), "expected an unselected precision"

    drawn = {
        trace["label"]: figure.data[trace["traceIndex"]].visible
        for trace in view_model.compute_traces
    }
    assert drawn == {
        trace["label"]: trace["dtype"] in selected
        for trace in view_model.compute_traces
    }


def test_standalone_roofs_open_at_the_selected_cap(benchmarked_roofline) -> None:
    """Diagonals ship clipped to the opening selection's tallest ceiling, and
    keep their full sample grid so the client can re-clip at a taller one
    without losing hover density."""
    figure, view_model = standalone_with_unselected_precision(benchmarked_roofline)
    selected = set(view_model.default_precisions)
    peaks = [trace["peakPerf"] for trace in view_model.compute_traces]
    top_peak = max(
        trace["peakPerf"]
        for trace in view_model.compute_traces
        if trace["dtype"] in selected
    )
    assert top_peak < max(peaks), "expected a taller unselected ceiling"

    assert view_model.roofline_traces, "expected bandwidth roofs in the model"
    for roof in view_model.roofline_traces:
        drawn = figure.data[roof["traceIndex"]]
        assert drawn.y[-1] == pytest.approx(top_peak)
        assert drawn.x[-1] == pytest.approx(top_peak / roof["bandwidth"])
        assert roof["kneeAi"] == pytest.approx(drawn.x[-1])
        assert roof["kneePerf"] == pytest.approx(drawn.y[-1])
        assert roof["sampleAi"][-1] == pytest.approx(max(peaks) / roof["bandwidth"])


def test_dash_figures_keep_every_ceiling(benchmarked_roofline) -> None:
    """The WebUI has no precision selector, so narrowing the standalone document
    must not reach back into the figures Dash renders."""
    roofline, flops_figure = stacked_figure(benchmarked_roofline, ["BF16", "FP64"])
    source_model = roofline._Roofline__view_models["FLOP"]
    ceiling_indices = [trace["traceIndex"] for trace in source_model.compute_traces]
    roof_extents = drawn_roof_knees(flops_figure)

    roofline._combined_html_figure(None, flops_figure)

    assert all(flops_figure.data[index].visible is None for index in ceiling_indices)
    assert drawn_roof_knees(flops_figure) == roof_extents


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
    """Serialized model text cannot close its embedding script element."""
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


def test_browser_exposes_fixed_reset_fit_and_offplot_controls() -> None:
    """The page labels fixed reset separately from one-shot data fitting."""
    page_template = roofline_html._read_asset("roofline_plot.html")

    assert 'id="roofline-reset-view"' in page_template
    assert "Return to the fixed opening axes" in page_template
    assert 'id="roofline-fit-data"' in page_template
    assert "one-shot zoom" in page_template
    assert 'id="roofline-kernel-offplot-count"' in page_template


def test_controller_uses_only_the_embedded_canonical_frame_recipe() -> None:
    """Browser reset comes from model.frame, not the removed data-fit recipe."""
    controller = (_ASSETS / "roofline_plot.js").read_text(encoding="utf-8")

    assert "var frame = model.frame;" in controller
    assert "var PLOT_SLOPE_SKEW = 2.0;" in controller
    assert "var ZOOM_PAD_DECADES = 0.5;" in controller
    assert "var ZOOM_MIN_DECADES = 1.0;" in controller
    assert "model.framePad" not in controller
    assert "model.frameMinDecades" not in controller
    assert "model.frameSlopeSkew" not in controller
    for legacy_name in (
        "frameAnchors",
        "paddedLogSpan",
        "pinToSlopes",
        "initialRange",
        "captureInitialRange",
        "function applyFrame",
    ):
        assert legacy_name not in controller


def test_precision_controller_remains_single_and_does_not_reset_view() -> None:
    """Precision toggles preserve PR10723's menu and fixed browser viewport."""
    controller = (_ASSETS / "roofline_plot.js").read_text(encoding="utf-8")

    assert controller.count("function buildPrecisionOptions") == 1
    assert "precisionMenu" in controller
    assert "precisionSelect" not in controller
    apply_precision = controller.split("function applyPrecision()", 1)[1].split(
        "var lastEmphasizedLevel", 1
    )[0]
    assert "resetView" not in apply_precision


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
