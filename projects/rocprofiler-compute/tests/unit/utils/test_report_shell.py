# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the shared standalone HTML report shell."""

import json
from pathlib import Path

from utils.html_report import report_shell
from utils.html_report.report_shell import (
    DARK_THEME_CLASS,
    build_document,
    read_asset,
)

_ASSETS = Path(report_shell.__file__).parent / "assets"


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
