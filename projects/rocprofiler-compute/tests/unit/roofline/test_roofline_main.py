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
from typing import TYPE_CHECKING, Dict, List, Tuple

import plotly.graph_objects as go
import pytest

import roofline.roofline_html as roofline_html
from roofline.roofline_frame import FRAME_X_MIN, canonical_frame
from roofline.roofline_hover import truncate_kernel_name, wrap_hover_name
from roofline.roofline_html import RooflineViewModel, build_interactive_document

if TYPE_CHECKING:
    from roofline.roofline_main import Roofline

_ASSETS = Path(roofline_html.__file__).parent / "assets"


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

    def build(datatypes: List[str], **run_parameters: object):
        parameters = {
            "workload_dir": str(tmp_path),
            "matrix_ops_type": "MFMA",
        }
        parameters.update(run_parameters)
        return make_roofline(datatypes, **parameters)

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


def test_generate_plot_filters_bandwidth_roofs_by_mem_level(
    benchmarked_roofline,
) -> None:
    """The selected memory level reaches both the figure and client model."""
    roofline = benchmarked_roofline(["FP64"], mem_level=["HBM"])
    fig = roofline.generate_plot("FP64", fig=go.Figure())

    bandwidth_levels = {"LDS", "L1", "L2", "HBM"}
    drawn_bandwidth_levels = {
        trace.name for trace in fig.data if trace.name in bandwidth_levels
    }
    assert drawn_bandwidth_levels == {"HBM"}

    view_model = roofline._Roofline__view_models["FLOP"]
    assert {roof["level"] for roof in view_model.roofline_traces} == {"HBM"}


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


def test_kernel_traces_score_against_the_roof_that_binds_not_the_tallest_drawn() -> (
    None
):
    """The percentage is scored against whichever roof the limiter actually
    names, not always the tallest compute roof stacked on the figure: dropping
    the tall MFMA peak from the candidate set changes both the label (falls
    back to the memory level, since VALU's peak is below this kernel's own
    performance) and the percentage (scored against HBM's own 150,000
    GFLOP/s implied ceiling instead of MFMA's 90,000), and neither case's
    percentage exceeds 100%."""
    ai_data = {"ai_hbm": [[100.0], [50000.0]], "kernelNames": ["kA"]}

    matrix_traces, matrix_capped = kernel_traces(make_roofline(["FP32"]), ai_data)
    valu_traces, valu_capped = kernel_traces(
        make_roofline(["FP32"]), ai_data, compute_peaks=[("FP32 VALU", 9000.0)]
    )

    assert pct_roof(matrix_capped[0]) == pytest.approx(55.56, abs=0.01)
    assert pct_roof(valu_capped[0]) == pytest.approx(33.33, abs=0.01)
    assert "Limited by Compute: FP32 MFMA" in matrix_traces[0].hovertemplate
    assert "Limited by Memory: HBM" in valu_traces[0].hovertemplate


def test_kernel_traces_name_the_specific_peak_not_the_tallest_stacked_one() -> None:
    """When several compute peaks are stacked, the limiter names whichever one
    actually bounds this kernel's own performance, not whichever peak happens
    to be tallest on the figure."""
    ai_data = {"ai_hbm": [[100.0], [50000.0]], "kernelNames": ["kA"]}

    traces, _ = kernel_traces(
        make_roofline(["FP32"]),
        ai_data,
        compute_peaks=[("FP32 MFMA", 90000.0), ("FP16 MFMA", 180000.0)],
    )

    assert "Limited by Compute: FP32 MFMA" in traces[0].hovertemplate


def test_kernel_traces_ignore_a_memory_roof_the_kernel_already_exceeds() -> None:
    """A memory level's implied ceiling (bandwidth * AI) that the kernel's own
    measured performance already exceeds is not an achievable roof, so the
    limiter must not name it -- exactly as compute peaks below performance are
    already excluded. Here HBM implies a 1,500 GFLOP/s ceiling but the kernel
    measures 5,000 GFLOP/s, so the limiter must fall through to the cheapest
    eligible compute peak instead of naming the impossible memory ceiling."""
    traces, _ = kernel_traces(
        make_roofline(["FP32"]),
        {"ai_hbm": [[1.0], [5000.0]], "kernelNames": ["kA"]},
    )

    assert "Limited by Compute: FP32 VALU" in traces[0].hovertemplate
    assert "Limited by Memory: HBM" not in traces[0].hovertemplate


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
    assert "Limited by Memory: HBM" in traces[0].hovertemplate

    unroofed_traces, unroofed = kernel_traces(
        make_roofline(["FP32"]),
        {"ai_hbm": [[1.0], [900.0]], "kernelNames": ["kA"]},
        ceiling_data={},
        compute_peaks=[],
    )
    assert "Limited by Unknown: Unknown" in unroofed_traces[0].hovertemplate
    assert unroofed[0]["points"][0]["hoverCells"] == [
        "N/A",
        "N/A",
        "Bandwidth:<br>\u2003HBM: N/A% (900.000 / N/A GB/s)",
    ]


def test_kernel_traces_expose_the_limiting_peak_for_a_memory_bound_kernel() -> None:
    """A memory-bound kernel's model carries a structured limitingPeak field,
    naming the same cache level the hover text already calls out as the
    binding roof, so the client can pick that one point without
    re-deriving the limiter itself."""
    traces, model = kernel_traces(
        make_roofline(["FP32"]),
        {"ai_hbm": [[1.0], [900.0]], "kernelNames": ["kA"]},
        sanitized_cache_hierarchy=["HBM", "L2"],
    )

    assert "Limited by Memory: HBM" in traces[0].hovertemplate
    assert model[0]["limitingPeak"] == "HBM"


def test_kernel_traces_expose_the_leftmost_peak_for_a_compute_bound_kernel() -> None:
    """A compute-bound kernel's limiter names a datatype/op-path (e.g. 'FP32
    VALU'), not a cache level, so no point.peak matches it directly.
    limitingPeak falls back to the peak of the kernel's own lowest-AI point
    (L2's AI of 0.5 sits left of HBM's AI of 2.0) so the client still has a
    single point to show without guessing which one."""
    _, model = kernel_traces(
        make_roofline(["FP32"]),
        {
            "ai_hbm": [[2.0], [1000.0]],
            "ai_l2": [[0.5], [1000.0]],
            "kernelNames": ["kA"],
        },
        sanitized_cache_hierarchy=["HBM", "L2"],
        ceiling_data={},
    )

    assert model[0]["limitingPeak"] == "L2"


def test_kernel_traces_expose_the_leftmost_peak_when_the_limiter_is_unknown() -> None:
    """No ceiling data and no compute peaks leaves no candidate roof at all, so
    the limiter falls back to Unknown -- and, same as the compute-bound case,
    limitingPeak falls back to the kernel's own lowest-AI point (L2 at 0.5,
    left of HBM's 2.0)."""
    traces, model = kernel_traces(
        make_roofline(["FP32"]),
        {
            "ai_hbm": [[2.0], [1000.0]],
            "ai_l2": [[0.5], [1000.0]],
            "kernelNames": ["kA"],
        },
        sanitized_cache_hierarchy=["HBM", "L2"],
        ceiling_data={},
        compute_peaks=[],
    )

    assert "Limited by Unknown: Unknown" in traces[0].hovertemplate
    assert model[0]["limitingPeak"] == "L2"


def test_kernel_traces_share_one_performance_value_across_a_kernels_points() -> None:
    """A kernel's HBM point and L2 point measure the same achieved performance
    but sit at different arithmetic intensities, so today each point is scored
    against its own level's bandwidth -- 25.00% at HBM, 66.67% at L2. Once the
    limiter's own closest-roof search (L2's 1,500 GFLOP/s implied ceiling beats
    HBM's 4,000) drives the displayed percentage too, both of the kernel's
    points must show that same 66.67% / 1,500, not their own per-level value."""
    ceiling = {
        "hbm": [[0.01, 1.0], [1.0, 2000.0], 2000.0],
        "l2": [[0.01, 1.0], [1.0, 3000.0], 3000.0],
    }

    _, model = kernel_traces(
        make_roofline(["FP32"]),
        {
            "ai_hbm": [[2.0], [1000.0]],
            "ai_l2": [[0.5], [1000.0]],
            "kernelNames": ["kA"],
        },
        sanitized_cache_hierarchy=["HBM", "L2"],
        ceiling_data=ceiling,
        compute_peaks=[],
    )

    points = {point["peak"]: point for point in model[0]["points"]}
    assert points["HBM"]["hoverCells"][:2] == ["1,500", "66.67"]
    assert points["L2"]["hoverCells"][:2] == ["1,500", "66.67"]


def test_bandwidth_hover_shows_zero_levels() -> None:
    """Every cache level still shows up at 0 when a kernel does no traffic
    there, instead of vanishing from the tooltip."""
    ceiling = dict(CEILING, lds=[[0.01, 1.0], [1.0, 800.0], 800.0])
    _, model = kernel_traces(
        make_roofline(["FP32"]),
        {
            "ai_hbm": [[1.0], [900.0]],
            "ai_l2": [[0.0], [0.0]],
            "ai_lds": [[0.0], [0.0]],
            "kernelNames": ["kA"],
        },
        sanitized_cache_hierarchy=["HBM", "L2", "LDS"],
        ceiling_data=ceiling,
    )
    bandwidth_html = model[0]["points"][0]["hoverCells"][2]
    assert bandwidth_html == (
        "Bandwidth:<br>\u2003L2: N/A% (0.000 / N/A GB/s)"
        "<br>\u2003HBM: 60.00% (0.900 / 1.500 TB/s)"
        "<br>\u2003LDS: 0.00% (0.000 / 800.000 GB/s)"
    )


def test_kernel_hover_carries_the_whole_name_up_to_the_limit() -> None:
    """A demangled name within the tooltip's length cap reaches the tooltip
    whole. It is wrapped onto as many lines as it takes, but nothing is
    dropped: two instantiations of the same function are told apart by
    template arguments that run to the very end of the name."""
    name = "Cijk_Alik_Bljk_" + "SB_MT256x256x16_MI32x32x2x1_" * 6

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


def test_truncate_kernel_name_caps_pathologically_long_names() -> None:
    """A pathologically long demangled name is capped at 200 characters, ending
    in an ellipsis, so it can't blow up the tooltip. Names within the cap are
    untouched."""
    name = "Cijk_Alik_Bljk_" + "SB_MT256x256x16_MI32x32x2x1_" * 40

    truncated = truncate_kernel_name(name)

    assert len(truncated) == 200
    assert truncated == name[:197] + "..."
    assert truncate_kernel_name("short_name") == "short_name"


def test_kernel_hover_truncates_names_beyond_two_hundred_characters() -> None:
    """The kernel trace hover wraps the truncated name, not the raw one, so a
    pathologically long demangled name can't blow up the tooltip."""
    name = "Cijk_Alik_Bljk_" + "SB_MT256x256x16_MI32x32x2x1_" * 40

    traces, _ = kernel_traces(
        make_roofline(["FP32"]),
        {"ai_hbm": [[1.0], [900.0]], "kernelNames": [name]},
    )

    wrapped = wrap_hover_name(truncate_kernel_name(name))
    assert wrapped in traces[0].hovertemplate
    lines = wrapped.split(">", 1)[1].removesuffix("</span>")
    flattened = lines.replace("<br>", "")
    assert len(flattened) == 200
    assert flattened.endswith("...")


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


def test_kernel_under_the_frame_is_not_clamped(benchmarked_roofline) -> None:
    """A kernel below y_low keeps its true coordinates and leaves the axes on the
    machine frame."""
    ai_data = {
        "ai_hbm": [[1.0, 1.0], [0.1, 500.0]],
        "kernelNames": ["sunken", "framed"],
    }
    inside_only = {"ai_hbm": [[1.0], [500.0]], "kernelNames": ["framed"]}

    _, figure, _, _ = benchmarked_roofline(["FP64"]).construct_plotly_figures(
        ai_data, datatypes=["FP64"]
    )
    _, reference, _, _ = benchmarked_roofline(["FP64"]).construct_plotly_figures(
        inside_only, datatypes=["FP64"]
    )

    assert layout_bounds(figure) == layout_bounds(reference)

    sunken_trace = next(trace for trace in figure.data if trace.name == "sunken")
    assert sunken_trace.y == (0.1,)


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
    benchmarked_roofline, monkeypatch
) -> None:
    """Fallback ranges identify their source in the subtitle."""
    roofline = benchmarked_roofline(["FP64"])
    machine_figure = roofline.generate_plot("FP64")
    assert "Axes fixed to this GPU" in machine_figure.layout.title.text

    monkeypatch.setattr(
        "roofline.roofline_main.machine_ceilings",
        lambda *_args: ([], []),
    )
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


def test_default_precisions_prefer_fp32_fp16_fp8() -> None:
    """The standalone page opens on FP32/FP16/FP8 when the run benchmarked
    them, in that preference order rather than the order the run listed them.
    A run with none of the three falls back to its first datatype, so the page
    always opens on something."""
    from roofline.roofline_main import _default_precisions

    assert _default_precisions(["FP16", "FP32", "FP8", "FP64"]) == [
        "FP32",
        "FP16",
        "FP8",
    ]
    assert _default_precisions(["FP64", "BF16"]) == ["FP64"]
    assert _default_precisions(["FP8"]) == ["FP8"]


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
