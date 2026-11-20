# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the shared standalone HTML report shell."""

import json
import re
from pathlib import Path

import pytest

import config
from utils.html_report import report_shell
from utils.html_report.report_shell import (
    DARK_THEME_CLASS,
    build_document,
    read_asset,
    report_file_name,
    truncate_kernel_name,
)

_ASSETS = Path(report_shell.__file__).parent / "assets"
# Each report page: a script with a page body of the same name
PAGE_SCRIPTS = sorted(
    script
    for script in Path(config.rocprof_compute_home).glob("**/assets/*.js")
    if script.with_suffix(".html").exists()
)


def test_build_document_places_fragments_in_one_shell() -> None:
    """Shared assets precede page assets inside one self-contained shell."""
    rendered = build_document(
        title="Report <one>",
        body_html='<main id="body-marker">body</main>',
        page_css=".page-marker { color: red; }",
        page_js="window.pageMarker = true;",
        model_id="sample-model",
        model={"name": "sample"},
    )

    assert "<title>Report &lt;one&gt;</title>" in rendered
    assert (
        rendered.index("--report-border")
        < rendered.index(".page-marker")
        < rendered.index("body-marker")
        < rendered.index('id="sample-model"')
        < rendered.index("window.HtmlReport")
        < rendered.index("window.pageMarker")
    )
    assert f'classList.add("{DARK_THEME_CLASS}")' in rendered
    assert f":root.{DARK_THEME_CLASS}" in read_asset(_ASSETS, "report_base.css")
    assert f'"{DARK_THEME_CLASS}"' in read_asset(_ASSETS, "report_base.js")


def test_embed_json_nulls_non_finite_floats_and_escapes_script_close() -> None:
    payload = {
        "integers": [7, (8,)],
        "floats": [1.5, float("nan"), float("inf"), float("-inf")],
        "name": "</script>",
        "nested": {
            "items": [(3, {"name": "already-json-safe", "value": float("-inf")})],
        },
    }

    serialized = report_shell._embed_json(payload)

    assert json.loads(serialized) == {
        "integers": [7, [8]],
        "floats": [1.5, None, None, None],
        "name": "</script>",
        "nested": {
            "items": [[3, {"name": "already-json-safe", "value": None}]],
        },
    }
    assert "</" not in serialized


def test_read_asset_caches_utf8(tmp_path: Path) -> None:
    asset = tmp_path / "sample.txt"
    asset.write_text("caf\u00e9", encoding="utf-8")

    assert read_asset(tmp_path, "sample.txt") == "caf\u00e9"
    asset.write_text("changed", encoding="utf-8")
    assert read_asset(tmp_path, "sample.txt") == "caf\u00e9"


def test_shared_assets_have_no_plotly_dependency() -> None:
    shell = read_asset(_ASSETS, "report_shell.html")
    controller = read_asset(_ASSETS, "report_base.js")

    assert "Plotly" not in shell + controller
    assert "<script src" not in shell + controller


def test_report_file_name_ends_with_the_kernel_filter_sorted_as_text() -> None:
    """-k 10 2 -k 3 parses to [[10, 2], [3]]; the name lists the ids as text."""
    assert report_file_name("empirRoof", 0, None) == "empirRoof_gpu-0.html"
    assert (
        report_file_name("empirRoof", 0, [[10, 2], [3]])
        == "empirRoof_gpu-0_10_2_3.html"
    )


def test_truncate_kernel_name_caps_pathologically_long_names() -> None:
    """A pathologically long demangled name is capped at 200 characters, ending
    in an ellipsis, so it can't blow up the tooltip. Names within the cap are
    untouched."""
    name = "Cijk_Alik_Bljk_" + "SB_MT256x256x16_MI32x32x2x1_" * 40

    truncated = truncate_kernel_name(name)

    assert len(truncated) == 200
    assert truncated == name[:197] + "..."
    assert truncate_kernel_name("short_name") == "short_name"


@pytest.mark.parametrize("script", PAGE_SCRIPTS, ids=lambda path: path.stem)
def test_page_script_finds_every_element_it_looks_up(script: Path) -> None:
    """Every id a page script looks up is one its page renders; otherwise the
    control would silently stop working."""
    looked_up = set(re.findall(r'getElementById\("([^"]+)"\)', script.read_text()))
    rendered = set(re.findall(r'id="([^"]+)"', script.with_suffix(".html").read_text()))

    assert looked_up
    assert looked_up <= rendered
