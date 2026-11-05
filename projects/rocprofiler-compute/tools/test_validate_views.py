# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for tools/validate_views.py.

Run from the project directory: python3 -m pytest tools/test_validate_views.py
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from validate_views import structure_errors, validate

import config
from utils.analysis_views import default_views_dir, iter_view_rows, load_views
from utils.metric_library import default_metric_library_dir, load_metric_library

ARCH = "gfx942"


def swap_rows(views):
    table = next(t for v in views for t in v["tables"] if t["id"] == 1102)
    table["metrics"][0], table["metrics"][1] = table["metrics"][1], table["metrics"][0]


def drop_row(views):
    table = next(t for v in views for t in v["tables"] if t["id"] == 1102)
    del table["metrics"][2]


def drop_column(views):
    table = next(t for v in views for t in v["tables"] if t["id"] == 1102)
    del table["header"]["max"]


def change_alias(views):
    next(v for v in views if v["legacy_panel_id"] == 1100)["alias"] = "pipeline"


def move_legacy_id(views):
    dict(iter_view_rows(views))["11.2.1"]["legacy_id"] = "11.2.2"


def move_legacy_position(views):
    dict(iter_view_rows(views))["11.2.3"]["legacy_position"] = "11.2.4"


@pytest.mark.parametrize(
    "change, error",
    [
        (swap_rows, "panel 1100: table 1102: rows"),
        (drop_row, "panel 1100: table 1102: rows"),
        (drop_column, "panel 1100: table 1102: header"),
        (change_alias, "panel 11: -b alias 'pipeline', config template has 'cu_pipe'"),
        (move_legacy_id, "11.2.1: legacy_id 11.2.2 is not 11.2.1"),
        (move_legacy_position, "11.2.3: 11.2.4 is not"),
    ],
)
def test_structure_errors(change, error):
    """A view that no longer keeps its panel's structure is reported."""
    views = load_views(ARCH)
    metrics = load_metric_library(default_metric_library_dir(), ARCH).metrics
    assert structure_errors(views, metrics, config.analysis_configs_dir, ARCH) == []
    change(views)
    errors = structure_errors(views, metrics, config.analysis_configs_dir, ARCH)
    assert any(e.startswith(error) for e in errors), errors


def test_views_tree_holds_only_arch_directories(tmp_path):
    views_dir = tmp_path / "views"
    shutil.copytree(default_views_dir(), views_dir)
    (views_dir / "cdna").mkdir()
    errors = validate(views_dir, default_metric_library_dir())
    assert errors == ["[cdna] is not a metric library arch directory"]
