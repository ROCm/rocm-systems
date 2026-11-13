# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for tools/validate_views.py.

Run from the project directory: python3 -m pytest tools/test_validate_views.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from validate_views import validate

from rocprof_compute_soc.metric_library import loader
from utils.analysis_views import default_views_dir


def test_views_tree_holds_only_arch_directories(tmp_path):
    views_dir = tmp_path / "views"
    shutil.copytree(default_views_dir(), views_dir)
    (views_dir / "cdna").mkdir()
    errors = validate(views_dir, loader.default_metric_library_dir())
    assert errors == ["[cdna] is not a metric library arch directory"]
