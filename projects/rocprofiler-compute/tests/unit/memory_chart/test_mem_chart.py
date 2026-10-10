# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Terminal rendering of every memory chart layout."""

import re
import zlib

import common
import pytest

from membw_analysis.models import BottleneckNode, MemBwAnalysisResult, SupportingMetric
from memory_chart.loader import Layouts, PanelConfigs
from memory_chart.mem_chart import MemChart, plot_mem_chart, progress_bar
from memory_chart.units import display_unit, panel_units
from utils.utils_common import strip_ansi

DEFAULT_TITLE = "3. Memory Chart (Normalization: per_kernel)"
MEMORY_CHART_ARCHS = common.memory_chart_archs()


def sample_values(arch: str) -> dict[str, float]:
    """A distinct, plausible value for every memory chart metric of an arch."""
    values = {}
    for name, unit in panel_units(PanelConfigs.for_arch(arch)).items():
        seed = zlib.crc32(name.encode()) % 97
        kind = display_unit(unit).kind
        if kind == "percent":
            values[name] = 1.5 + seed
        elif kind == "bandwidth":
            values[name] = (seed + 1) * 4.25e9
        else:
            values[name] = seed + 2.0
    return values


def render(arch, values=None, membw=None):
    """Plain-text chart for an analysis-config arch (gfx115x stands for its family)."""
    if values is None:
        values = sample_values(arch)
    units = panel_units(PanelConfigs.for_arch(arch))
    return strip_ansi(
        plot_mem_chart(
            values, chart_title=DEFAULT_TITLE, gpu_arch=arch, units=units, membw=membw
        )
    )


def make_node(label, level, state="active", value=18.7, children=()):
    supporting = (SupportingMetric(f"k_{label}", value, "Percent", f"{value}%"),)
    return BottleneckNode(label, label, level, state, supporting, children)


def make_membw(*nodes):
    return MemBwAnalysisResult("gfx950", "full", None, nodes, ())


def make_chart(arch, values, membw=None):
    units = panel_units(PanelConfigs.for_arch(arch))
    return MemChart(Layouts.for_arch(arch), values, units, membw)


def box_rows(lines, title):
    """Top and bottom border rows and left edge of the box titled *title*."""
    top = next(
        i for i, line in enumerate(lines) if f" {title} " in line and "╭" in line
    )
    col = lines[top].index(f" {title} ")
    left = lines[top].rindex("╭", 0, col)
    bottom = next(i for i in range(top + 1, len(lines)) if lines[i][left] == "╰")
    return top, bottom, left


VALUE_SETS = {
    "full": lambda arch: (sample_values(arch), None),
    "empty": lambda arch: ({}, None),
    "membw": lambda arch: (
        sample_values(arch),
        make_membw(*(make_node(f"{lv} stall", lv) for lv in ("GL1", "GL2", "EA"))),
    ),
}


# ---------------------------------------------------------------------------
# Every architecture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gpu_arch", MEMORY_CHART_ARCHS)
def test_full_metrics_render_every_block_label_and_note(gpu_arch):
    output = render(gpu_arch)
    assert output.splitlines()[0] == DEFAULT_TITLE
    assert "N/A" not in output
    layout = Layouts.for_arch(gpu_arch)
    for block in layout.blocks():
        assert block.title in output
        assert (block.note or "").split(" ")[0] in output
        for item in block.items:
            assert item.title in output
    for arrow in layout.arrows:
        assert arrow.title in output


@pytest.mark.parametrize("gpu_arch", MEMORY_CHART_ARCHS)
def test_empty_metrics_render_one_placeholder_per_value(gpu_arch):
    output = render(gpu_arch, {})
    layout = Layouts.for_arch(gpu_arch)
    shown = sum(len(b.items) for b in layout.blocks()) + len(layout.arrows)
    assert output.count("N/A") == shown
    assert "GB/s" not in output


@pytest.mark.parametrize("gpu_arch", MEMORY_CHART_ARCHS)
def test_rows_have_one_width(gpu_arch):
    # Grid rows (between the scope bar and the legend) line up
    lines = render(gpu_arch).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|-"))
    end = next(i for i, line in enumerate(lines) if line.startswith("Legend"))
    grid = [
        line.rstrip()
        for line in lines[start + 2 : end]
        if line.startswith(("╭", "│", "╰"))
    ]
    widths = {len(line) for line in grid}
    assert len(widths) == 1, widths
    assert len(lines[start]) == widths.pop()


@pytest.mark.parametrize("gpu_arch", MEMORY_CHART_ARCHS)
def test_typical_values_stay_on_their_label_line(gpu_arch):
    output = render(gpu_arch)
    assert re.search(r"Scratch/Wave \d+\.\d{3} KB", output)
    assert re.search(r"Wave Occ \d+\.\d%", output)


def test_legend_lists_only_what_the_chart_uses():
    # gfx115x has no atomic arrows; gfx1250 has stall metrics
    assert "Atomic" not in render("gfx115x").splitlines()[-1]
    assert "Stall" in render("gfx1250").splitlines()[-1]


def test_unknown_arch_is_an_error():
    with pytest.raises(ValueError, match="gfx1030"):
        plot_mem_chart({}, chart_title=DEFAULT_TITLE, gpu_arch="gfx1030", units={})


# ---------------------------------------------------------------------------
# Vertical placement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("values", VALUE_SETS)
@pytest.mark.parametrize("gpu_arch", MEMORY_CHART_ARCHS)
def test_arrows_stay_inside_their_box(gpu_arch, values):
    chart = make_chart(gpu_arch, *VALUE_SETS[values](gpu_arch))
    for placement in chart.placements:
        end = placement.start + len(placement.lines)
        if placement.anchor:
            top = chart.top[placement.anchor]
            bottom = top + chart.height[placement.anchor] - 1
        else:
            top, bottom = 0, chart.total_height - 1
        assert top < placement.start and end <= bottom, placement.anchor


@pytest.mark.parametrize(
    ("gpu_arch", "box", "outer", "side"),
    [
        ("gfx950", "VL1D", "VL1D", "left"),
        ("gfx1250", "LDS", "TCP", "left"),
        ("gfx1250", "LDS", "TCP", "right"),
    ],
)
def test_drawn_arrows_sit_between_the_box_borders(gpu_arch, box, outer, side):
    lines = render(gpu_arch).splitlines()
    top, bottom, _ = box_rows(lines, box)
    outer_top, _, left = box_rows(lines, outer)
    right = lines[outer_top].index("╮", left)
    # The edge column runs from the outer box's border to the next box's border
    inside = lines[outer_top + 1]
    if side == "left":
        cols = slice(inside.rindex("│", 0, left) + 1, left)
    else:
        cols = slice(right + 1, inside.index("│", right + 1))
    arrow_rows = [
        i for i in range(top - 1, bottom + 2) if re.search(r"<-|->", lines[i][cols])
    ]
    assert arrow_rows and all(top < i < bottom for i in arrow_rows)


# ---------------------------------------------------------------------------
# Block contents
# ---------------------------------------------------------------------------


def test_gfx950_shows_io_bandwidth_on_attached_blocks():
    output = render("gfx950", {"xGMI Read BW": 2e9, "PCIe Atomic BW": 3e9})
    assert re.search(r"\|\|  Read BW\s+2\.000 GB/s", output)
    assert re.search(r"\|\|  Atomic BW\s+3\.000 GB/s", output)
    assert output.index("xGMI") < output.index("Compute Units") < output.index("PCIe")


@pytest.mark.parametrize("gpu_arch", ["gfx908", "gfx942"])
def test_xgmi_without_counters_draws_a_bare_connector(gpu_arch):
    output = render(gpu_arch)
    assert "xGMI (to Peer GPU)" in output
    assert re.search(r"^\s+\|\|$", output, re.MULTILINE)
    assert "PCIe" not in output


@pytest.mark.parametrize(
    ("gpu_arch", "label"), [("gfx1250", "Buf Stall"), ("gfx115x", "GL2 Stall")]
)
def test_stall_metrics_get_a_bar(gpu_arch, label):
    lines = render(gpu_arch).splitlines()
    row = next(i for i, line in enumerate(lines) if label in line)
    col = lines[row].index(label)
    assert set(lines[row + 1][col : col + 10]) <= set("█░")


@pytest.mark.parametrize(
    ("percent", "filled"), [(0, 0), (55, 5), (100, 10), (150, 10), (None, 0)]
)
def test_progress_bar_is_clamped(percent, filled):
    bar = progress_bar(percent)
    assert (bar.count("█"), len(bar)) == (filled, 10)


# ---------------------------------------------------------------------------
# Membw stall annotations (gfx950)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "block"), [("GL1", "VL1D"), ("GL2", "L2"), ("EA", "Data Fabric")]
)
def test_active_stall_is_annotated_on_its_block(level, block):
    output = render("gfx950", membw=make_membw(make_node(f"{level} stall", level)))
    row = next(line for line in output.splitlines() if f"[!] {level} stall" in line)
    title_row = next(
        line for line in output.splitlines() if f" {block} " in line and "╭" in line
    )
    assert "18.7%" in row
    assert abs(row.index("[!]") - title_row.index(block)) < 20
    assert "Stall" in output.splitlines()[-1]


def test_only_active_leaves_are_annotated():
    child = make_node("TCP<-UTCL1", "GL1")
    parent = make_node("TCP stall", "GL1", children=(child,))
    output = render(
        "gfx950", membw=make_membw(parent, make_node("idle", "GL2", state="inactive"))
    )
    assert "[!] TCP<-UTCL1" in output
    assert "[!] TCP stall" not in output
    assert "[!] idle" not in output


def test_no_active_stalls_draw_as_if_there_were_no_membw():
    idle = make_membw(make_node("idle", "GL2", state="inactive"))
    assert render("gfx950", membw=idle) == render("gfx950")
    assert "Stall" not in render("gfx950").splitlines()[-1]
