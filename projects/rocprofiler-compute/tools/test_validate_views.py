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

import validate_views
from validate_views import equivalence_errors, validate

import config
from utils.analysis_views import default_views_dir, iter_view_rows, load_views
from utils.metric_library import default_metric_library_dir, load_metric_library

ARCH = "gfx942"


def change_label(views):
    dict(iter_view_rows(views))["11.2.1"]["label"] = "IPC (Renamed)"


def drop_column(views):
    table = next(t for v in views for t in v["tables"] if t["id"] == 1102)
    del table["header"]["max"]


def change_alias(views):
    next(v for v in views if v["legacy_panel_id"] == 1100)["alias"] = "pipeline"


def move_legacy_id(views):
    dict(iter_view_rows(views))["2.1.0"]["legacy_id"] = "2.1.1"


def drop_description(views):
    dict(iter_view_rows(views))["11.2.1"]["description"] = None


@pytest.mark.parametrize(
    "change, error",
    [
        (change_label, "panel/1100/data source/1/metric_table/metric: panel has"),
        (drop_column, "panel/1100/data source/1/metric_table/header: panel has"),
        (change_alias, "panel 11: -b alias 'pipeline', config template has 'cu_pipe'"),
        (move_legacy_id, "2.1.0: legacy_id is 2.1.1"),
        (drop_description, "panel/1100/metrics_description: panel has"),
    ],
)
def test_equivalence_errors(change, error):
    """A view that no longer gives the panel config is reported."""
    views = load_views(ARCH)
    metrics = load_metric_library(default_metric_library_dir(), ARCH).metrics
    change(views)
    errors = equivalence_errors(views, metrics, config.analysis_configs_dir, ARCH)
    assert any(e.startswith(error) for e in errors), errors


def test_unread_file_and_unshown_display_form(tmp_path, monkeypatch):
    """A view file no arch reads, and a display form no view shows, are reported."""
    views_dir = tmp_path / "views"
    shutil.copytree(default_views_dir(), views_dir)
    (views_dir / "rdna").mkdir()
    shutil.copy(views_dir / "0000_top_stats.yaml", views_dir / "rdna")
    monkeypatch.setitem(validate_views.DISPLAY_FORMS, "unused_form", {"formula": "1"})
    errors = validate(views_dir, default_metric_library_dir())
    assert errors == [
        "[rdna/0000_top_stats.yaml] is read by no arch",
    ]
    (views_dir / "rdna" / "0000_top_stats.yaml").unlink()
    errors = validate(views_dir, default_metric_library_dir())
    assert errors == ["DISPLAY_FORMS lists unused_form, but no view shows it"]
