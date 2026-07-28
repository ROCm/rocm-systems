# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit coverage for the interactive roofline HTML layer."""

import argparse
import json
import re
from pathlib import Path

import roofline.roofline_html as roofline_html
from roofline.roofline_hover import wrap_hover_name
from roofline.roofline_html import RooflineViewModel
from roofline.roofline_main import Roofline

_ASSETS = Path(roofline_html.__file__).parent / "assets"


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


# Bandwidth ceilings only
CEILING = {"hbm": [[0.01, 1.0], [1.0, 1500.0], 1500.0]}
COMPUTE_PEAKS = [("FP32 VALU", 9000.0), ("FP32 MFMA", 90000.0)]


# =============================================================================
# Envelope cap: which compute ceiling a kernel point is scored against
# =============================================================================


def pct_roof(kernel: dict, point_index: int = 0) -> float:
    """The percent-of-roofline the tooltip shows for one of a kernel's points."""
    return float(kernel["points"][point_index]["hoverCells"][1])


def test_build_kernel_traces_scores_against_the_tallest_drawn_ceiling() -> None:
    """A stacked figure caps points at the tallest compute roof drawn, so the
    reported peak and limiter do not depend on the order datatypes were
    stacked."""
    roofline = make_roofline()
    roofline._Roofline__ai_data = {
        # AI 100 at 1500 GB/s puts the roof above the 9000 VALU peak, so the
        # cap is what decides the reported peak.
        "ai_hbm": [[100.0], [50000.0]],
        "kernelNames": ["kA"],
    }

    matrix_traces, matrix_capped = roofline._build_kernel_traces(
        kernel_names=["kA"],
        kernel_colors=["#123456"],
        sanitized_cache_hierarchy=["HBM"],
        ceiling_data=CEILING,
        ops_flops="FLOP",
        compute_peaks=COMPUTE_PEAKS,
    )
    valu_traces, valu_capped = roofline._build_kernel_traces(
        kernel_names=["kA"],
        kernel_colors=["#123456"],
        sanitized_cache_hierarchy=["HBM"],
        ceiling_data=CEILING,
        ops_flops="FLOP",
        compute_peaks=[("FP32 VALU", 9000.0)],
    )

    assert pct_roof(matrix_capped[0]) < 100.0
    assert pct_roof(valu_capped[0]) > 100.0
    # The limiter is constant across a kernel's points, so it ships in the
    # trace's hover template rather than in the client model.
    assert "Performance limiter: FP32 MFMA" in matrix_traces[0].hovertemplate
    assert "Performance limiter: FP32 VALU" in valu_traces[0].hovertemplate


# =============================================================================
# Performance limiter: which roof binds a kernel
# =============================================================================


def test_build_kernel_traces_limiter_names_the_binding_roof() -> None:
    """A kernel whose bandwidth roof sits under the compute cap is limited by
    its memory level, and falls back to Unknown when the ceiling data holds no
    roof for that level at all. Levels with no positive AI are not plotted.
    """
    roofline = make_roofline()
    roofline._Roofline__ai_data = {
        # kA is at AI 1, where HBM tops out at 1500 -- far below either compute
        # peak -- so HBM binds. Its L2 entry is zeroed and must be dropped, and
        # kB has no entry at either level.
        "ai_hbm": [[1.0], [900.0]],
        "ai_l2": [[0.0], [0.0]],
        "kernelNames": ["kA", "kB"],
    }

    traces, model = roofline._build_kernel_traces(
        kernel_names=["kA", "kB"],
        kernel_colors=["#123456", "#654321"],
        sanitized_cache_hierarchy=["HBM", "L2"],
        ceiling_data=CEILING,
        ops_flops="FLOP",
        compute_peaks=COMPUTE_PEAKS,
    )
    assert [kernel["name"] for kernel in model] == ["kA"]
    assert [point["peak"] for point in model[0]["points"]] == ["HBM"]
    assert "Performance limiter: HBM" in traces[0].hovertemplate

    unroofed_traces, unroofed = roofline._build_kernel_traces(
        kernel_names=["kA"],
        kernel_colors=["#123456"],
        sanitized_cache_hierarchy=["HBM", "L2"],
        ceiling_data={},
        ops_flops="FLOP",
        compute_peaks=[],
    )
    assert "Performance limiter: Unknown" in unroofed_traces[0].hovertemplate
    # With no roof to score against, both per-point tooltip values read N/A.
    assert unroofed[0]["points"][0]["hoverCells"] == ["N/A", "N/A"]


# =============================================================================
# Hover text: kernel name
# =============================================================================


def test_kernel_hover_carries_the_whole_name() -> None:
    """A long demangled name reaches the tooltip whole. It is wrapped onto as
    many lines as it takes, but nothing is dropped: two instantiations of the
    same function are told apart by template arguments that run to the very end
    of the name."""
    name = "Cijk_Alik_Bljk_" + "SB_MT256x256x16_MI32x32x2x1_" * 40
    roofline = make_roofline()
    roofline._Roofline__ai_data = {
        "ai_hbm": [[1.0], [900.0]],
        "kernelNames": [name],
    }

    traces, _ = roofline._build_kernel_traces(
        kernel_names=[name],
        kernel_colors=["#123456"],
        sanitized_cache_hierarchy=["HBM"],
        ceiling_data=CEILING,
        ops_flops="FLOP",
        compute_peaks=COMPUTE_PEAKS,
    )

    wrapped = wrap_hover_name(name)
    assert wrapped in traces[0].hovertemplate
    # Undo the wrapping: what the tooltip shows is exactly the name.
    lines = wrapped.split(">", 1)[1].removesuffix("</span>")
    assert lines.replace("<br>", "") == name


# =============================================================================
# View-model serialization
# =============================================================================


def test_view_model_to_json_escapes_script_close() -> None:
    model = RooflineViewModel(kernels=[{"name": "evil</script>", "points": []}])
    serialized = model.to_json()
    assert "</script>" not in serialized, "must not allow a script element to close"
    # Still valid JSON that decodes back to the original kernel name.
    assert json.loads(serialized)["kernels"][0]["name"] == "evil</script>"


# =============================================================================
# Light and dark theme
# =============================================================================


def _custom_properties(css: str, selector: str) -> dict[str, str]:
    """The declarations of the one rule with this exact selector."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    block = re.search(re.escape(selector) + r"\s*\{([^{}]*)\}", css)
    assert block, f"no rule found for {selector}"
    declarations = {}
    for line in block.group(1).split(";"):
        if ":" in line:
            name, _, value = line.partition(":")
            declarations[name.strip()] = value.strip()
    return declarations


def test_dark_theme_reads_the_same_palette_whichever_way_it_is_turned_on() -> None:
    """A media query cannot join a selector list, so the reader's choice and the
    system preference each map the dark tokens themselves. They have to map the
    same ones to the same values, or one route themes something the other does
    not."""
    css = (_ASSETS / "roofline_plot.css").read_text(encoding="utf-8")

    chosen = _custom_properties(css, ":root.roofline-theme-dark")
    from_system = _custom_properties(css, ":root:not(.roofline-theme-light)")
    assert chosen, "the toggle has to have something to switch to"
    assert chosen == from_system

    # Each mapping points at the palette rather than restating a color, which is
    # what keeps the duplication above harmless.
    root = _custom_properties(css, ":root")
    for name, value in chosen.items():
        if name == "color-scheme":
            continue
        token = re.fullmatch(r"var\((--roofline-dark-[a-z-]+)\)", value)
        assert token, f"{name} should read a --roofline-dark-* token, got {value}"
        assert token.group(1) in root, f"{token.group(1)} is never defined"


def test_theme_toggle_button_matches_the_id_the_controller_looks_up() -> None:
    """The page renders the button and the controller finds it by id. Nothing
    would report the two drifting apart: the button would just stop working."""
    controller = (_ASSETS / "roofline_plot.js").read_text(encoding="utf-8")

    assert 'id="roofline-theme-toggle"' in roofline_html._PAGE_TEMPLATE
    assert '"roofline-theme-toggle"' in controller
