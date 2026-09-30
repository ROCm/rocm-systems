# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the shared standalone HTML report document."""

import json
from pathlib import Path

import numpy as np
import pytest

from utils.html_report import document as report_document
from utils.html_report.document import (
    DARK_THEME_CLASS,
    build_document,
    json_safe,
    read_asset,
)

_ASSETS = Path(report_document.__file__).parent / "assets"


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

    assert rendered.startswith("<!DOCTYPE html>")
    assert '<meta charset="utf-8"/>' in rendered
    assert '<meta name="viewport"' in rendered
    assert "<title>Report &lt;one&gt;</title>" in rendered
    assert rendered.count("<style>") == 1
    assert rendered.count("</style>") == 1
    assert rendered.index("--report-border") < rendered.index(".page-marker")
    assert rendered.count(".page-marker { color: red; }") == 1
    assert rendered.index("<style>") < rendered.index("body-marker")
    assert rendered.index("body-marker") < rendered.index('id="sample-model"')
    assert rendered.index('id="sample-model"') < rendered.index("window.HtmlReport")
    assert rendered.index("window.HtmlReport") < rendered.index("window.pageMarker")
    assert rendered.count("<script>") == 2
    assert f'classList.add("{DARK_THEME_CLASS}")' in rendered


@pytest.mark.parametrize(
    "model_id",
    ["", "Bad Id", "2model", "bad_model", 'bad"id', "é-model"],
)
def test_build_document_rejects_invalid_model_id(model_id: str) -> None:
    with pytest.raises(ValueError, match="invalid model id"):
        build_document(
            title="Report",
            body_html="",
            page_css="",
            page_js="",
            model_id=model_id,
            model={},
        )


def test_build_document_embeds_json_safely_without_mutation() -> None:
    payload = {
        "integers": [np.int64(7), (np.int32(8),)],
        "floats": [np.float32(1.5), float("nan"), np.float64("inf")],
        "name": '</script><!-- < & "\u2028\u2029',
    }

    rendered = build_document(
        title="Report",
        body_html="",
        page_css="",
        page_js="",
        model_id="sample-model",
        model=payload,
    )
    serialized = rendered.split(
        '<script id="sample-model" type="application/json">', 1
    )[1].split("</script>", 1)[0]

    assert json.loads(serialized) == {
        "integers": [7, [8]],
        "floats": [1.5, None, None],
        "name": '</script><!-- < & "\u2028\u2029',
    }
    assert all(character not in serialized for character in ("<", "\u2028", "\u2029"))
    assert "\\u003c" in serialized
    assert "\\u2028" in serialized
    assert "\\u2029" in serialized
    assert isinstance(payload["integers"][0], np.integer)
    assert isinstance(payload["integers"][1], tuple)
    assert np.isnan(payload["floats"][1])


def test_json_safe_returns_new_containers() -> None:
    original = {"nested": [np.int64(3)]}
    converted = json_safe(original)

    assert converted == {"nested": [3]}
    assert converted is not original
    assert converted["nested"] is not original["nested"]


def test_read_asset_caches_utf8_and_reports_missing_file(tmp_path: Path) -> None:
    asset = tmp_path / "sample.txt"
    asset.write_text("caf\u00e9", encoding="utf-8")

    assert read_asset(tmp_path, "sample.txt") == "caf\u00e9"
    asset.write_text("changed", encoding="utf-8")
    assert read_asset(tmp_path, "sample.txt") == "caf\u00e9"

    with pytest.raises(FileNotFoundError):
        read_asset(tmp_path, "missing.txt")


def test_shared_assets_have_no_plotly_dependency() -> None:
    shell = read_asset(_ASSETS, "report_shell.html")
    controller = read_asset(_ASSETS, "report_base.js")

    assert "Plotly" not in shell + controller
    assert "<script src" not in shell + controller
    assert f'"{DARK_THEME_CLASS}"' in controller
    assert "readModel: readModel" in controller
    assert "initTheme: initTheme" in controller
