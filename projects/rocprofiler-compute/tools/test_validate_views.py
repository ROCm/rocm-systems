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

import config
from rocprof_compute_soc.metric_library import loader


def test_views_tree_holds_a_directory_per_metric_library_arch(tmp_path, monkeypatch):
    """A directory of no metric library arch, and an arch without view files."""
    shutil.copytree(config.analysis_views_dir, tmp_path, dirs_exist_ok=True)
    (tmp_path / "cdna").mkdir()
    arch = next(iter(loader.METRIC_LIBRARY_PARENT_ARCH))
    shutil.rmtree(tmp_path / arch)
    monkeypatch.setattr(config, "analysis_views_dir", tmp_path)
    assert validate() == [
        "[cdna] is not a metric library arch directory",
        f"[{arch}] is in METRIC_LIBRARY_PARENT_ARCH but has no view files",
    ]
