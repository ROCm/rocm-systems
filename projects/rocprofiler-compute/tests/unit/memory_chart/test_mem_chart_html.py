# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""The HTML memory chart page and the values it shows."""

import common
import pytest

from memory_chart.loader import Layouts, PanelConfigs
from memory_chart.mem_chart_html import MODEL_ID, chart_view, mem_chart_document
from memory_chart.units import panel_units
from utils.html_report.report_shell import truncate_kernel_name

MEMORY_CHART_ARCHS = common.memory_chart_archs()
# Longer than the kernel list shows
LONG_KERNEL_NAME = "kernel_b" * 40


@pytest.mark.parametrize("arch", MEMORY_CHART_ARCHS)
def test_page_shows_what_the_terminal_chart_shows(arch):
    """Every value and stall on the page reads as in the terminal chart."""
    values = common.memory_chart_values(arch)
    layout = Layouts.for_arch(arch)
    level = next((b.stall_level for b in layout.blocks() if b.stall_level), None)
    membw = common.membw_result(common.bottleneck("Busy", level)) if level else None
    terminal = common.memory_chart_text(arch, values, membw)

    view = chart_view(arch, values, panel_units(PanelConfigs.for_arch(arch)), membw)

    assert set(view["values"]) == layout.metrics()
    for shown in view["values"].values():
        assert shown["text"] in terminal
    stalls = [stall for rows in view["stalls"].values() for stall in rows]
    assert len(stalls) == (1 if level else 0)
    for stall in stalls:
        assert f"[!] {stall['label']} {stall['text']}" in terminal


def test_page_shows_guidance_or_the_status_line_as_the_terminal_does():
    """Active stalls with guidance show it; otherwise the page shows the status."""
    values = common.memory_chart_values("gfx950")
    units = panel_units(PanelConfigs.for_arch("gfx950"))
    guided = common.membw_result(
        common.bottleneck("Busy", "GL2"), guidance_blocks=("[GL2] Busy\n  Impact: x",)
    )
    idle = common.membw_result(common.bottleneck("Busy", "GL2", state="inactive"))

    with_guidance = chart_view("gfx950", values, units, guided)
    without_stalls = chart_view("gfx950", values, units, idle)
    without_membw = chart_view("gfx950", values, units, None)

    assert (with_guidance["status"], with_guidance["guidance"]) == (
        None,
        ["[GL2] Busy\n  Impact: x"],
    )
    assert without_stalls["status"].startswith("Memory Bandwidth Analysis: No ")
    assert without_stalls["guidance"] == []
    assert (without_membw["status"], without_membw["guidance"]) == (None, [])


def test_each_stall_links_to_its_guidance_block():
    """A stall points at the block rendered from its guidance id, if any."""
    membw = common.membw_result(
        common.bottleneck("Write stall", "GL2", guidance_id="write"),
        common.bottleneck("Hit rate", "GL2", guidance_id="no_template"),
        guidance_blocks=("[GL2] Write\n  Impact: x",),
        guidance_block_ids=("write",),
    )

    view = chart_view(
        "gfx950",
        common.memory_chart_values("gfx950"),
        panel_units(PanelConfigs.for_arch("gfx950")),
        membw,
    )

    assert [(s["label"], s["guidance"]) for s in view["stalls"]["l2"]] == [
        ("Write stall", 0),
        ("Hit rate", None),
    ]


def test_document_describes_each_value_and_groups_the_arrows():
    """Each value carries its panel config name and description, and the edges
    between columns hold every arrow between main-row blocks."""
    view = chart_view(
        "gfx950",
        common.memory_chart_values("gfx950"),
        panel_units(PanelConfigs.for_arch("gfx950")),
        None,
    )

    model = common.embedded_model(
        mem_chart_document(
            "gfx950",
            common.MEMORY_CHART_TITLE,
            PanelConfigs.for_arch("gfx950"),
            view,
            [("kernel_a", view), (LONG_KERNEL_NAME, view)],
        ),
        MODEL_ID,
    )

    layout = Layouts.for_arch("gfx950")
    blocks = {block.id: block for block in layout.blocks()}
    grouped = [i for edge in model["edges"] for group in edge for i in group["arrows"]]
    assert model["heading"] == common.MEMORY_CHART_TITLE
    # Each value names its metric and describes it, as the panel config does
    assert set(model["metrics"]) == layout.metrics()
    assert model["metrics"]["3.1.1"]["name"] == "VGPRs"
    assert all(metric["description"] for metric in model["metrics"].values())
    # The kernel list shows names shortened as the roofline page does
    assert [
        (kernel["name"], kernel["label"]) for kernel in model["views"]["kernels"]
    ] == [
        ("kernel_a", "kernel_a"),
        (LONG_KERNEL_NAME, truncate_kernel_name(LONG_KERNEL_NAME)),
    ]
    # Arrows to blocks attached above or below are drawn vertically, not in edges
    assert sorted(grouped) == [
        i for i, arrow in enumerate(layout.arrows) if blocks[arrow.target].row == 0
    ]
