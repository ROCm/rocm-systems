# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metric_library.py."""

import importlib
import sys

import pytest
from common import (
    METRIC_LIBRARY_FAMILY_PARENTS,
    use_metric_library_family,
    write_text_file,
)
from conftest import ProfileModeImportGuard

from utils import metric_library
from utils.metric_library import (
    arch_chain,
    arch_files,
    load_metric_library,
)

GPU_SERIES = "MI300"

# A metric with a peak, whose description names a counter it does not read.
WAVE_METRIC = """\
wave.cycles:
  name: Wave Cycles
  formula: SUM(SQ_WAVE_CYCLES)
  unit: Cycles
  description: Not counted with TCC_HIT_sum.
  peak: SUM(SQ_BUSY_CYCLES)
"""


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The synthetic family, as the only known chain."""
    use_metric_library_family(monkeypatch, tmp_path)
    return tmp_path


# =============================================================================
# Arch chain and files
# =============================================================================


def test_arch_chain_of_real_archs():
    assert arch_chain("gfx950") == ["gfx908", "gfx90a", "gfx940", "gfx942", "gfx950"]
    assert arch_chain("gfx941") == ["gfx908", "gfx90a", "gfx940", "gfx941"]
    assert arch_chain("gfx1151") == ["gfx115x"]
    assert arch_chain("gfx1250") == ["gfx1250"]


def test_arch_chain_rejects_unknown_arch():
    with pytest.raises(FileNotFoundError, match="gfx000"):
        arch_chain("gfx000")


def test_child_arch_takes_missing_files_from_ancestors(tree):
    write_text_file(tree / "gfx_base" / "wave.yaml", "{}\n")
    files = arch_files(tree, "gfx_c4")
    assert files["wave.yaml"] == tree / "gfx_base" / "wave.yaml"
    assert files["compute.yaml"] == tree / "gfx_c4" / "compute.yaml"
    assert list(files) == ["compute.yaml", "wave.yaml"]


def test_missing_arch_tree_names_the_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        metric_library, "METRIC_LIBRARY_PARENT_ARCH", METRIC_LIBRARY_FAMILY_PARENTS
    )
    with pytest.raises(FileNotFoundError, match=str(tmp_path / "gfx_c2")):
        arch_files(tmp_path, "gfx_c2")


# =============================================================================
# Loading
# =============================================================================


def test_load_metric_library_is_cached(tree):
    assert load_metric_library(tree, "gfx_c4") is load_metric_library(tree, "gfx_c4")


def test_load_metric_library_reads_variables(tree):
    write_text_file(
        tree / "gfx_base" / "variables.yaml",
        "GRBM_X:\n  formula: GRBM_GUI_ACTIVE\n",
    )
    library = load_metric_library(tree, "gfx_c4")
    assert library.variables == {"GRBM_X": {"formula": "GRBM_GUI_ACTIVE"}}
    assert "GRBM_X" not in library.metrics


def test_load_metric_library_reads_sets_through_the_chain(tree):
    write_text_file(
        tree / "gfx_base" / "sets.yaml",
        "busy:\n  title: Busy\n  metrics: [compute.salu_util, compute.valu_util]\n"
        "old:\n  title: Old\n  metrics: [compute.salu_util]\n",
    )
    write_text_file(
        tree / "gfx_c4" / "sets.yaml",
        "inherit: ../gfx_base/sets.yaml\nbusy:\n  metrics: [compute.valu_util]\n"
        "old: !remove\n",
    )
    library = load_metric_library(tree, "gfx_c4")
    assert library.sets == {"busy": {"title": "Busy", "metrics": ["compute.valu_util"]}}
    assert "busy" not in library.metrics


def test_duplicate_id_across_files_fails(tree):
    write_text_file(
        tree / "gfx_c4" / "wave.yaml",
        "compute.valu_util:\n  name: Again\n",
    )
    with pytest.raises(ValueError, match=r"\['compute.valu_util'\] are defined twice"):
        load_metric_library(tree, "gfx_c4")


# =============================================================================
# Counters
# =============================================================================


def test_counters_come_from_formula_peak_and_builtin_variables(tree):
    """Counters come only from formula and peak, of metrics from any file of the chain.
    Built-in variables resolve to counters and the unit denominators (SQ_WAVES) are
    always added, as in detect_counters()."""
    write_text_file(tree / "gfx_base" / "wave.yaml", WAVE_METRIC)
    library = load_metric_library(tree, "gfx_c4")
    selected = library.get_counters_for_metrics(
        ["compute.valu_util", "compute.mfma_flops_f8", "wave.cycles"], GPU_SERIES
    )
    assert {
        "SQ_ACTIVE_INST_VALU",
        "SQ_INSTS_VALU_MFMA_MOPS_F8",
        "GRBM_GUI_ACTIVE",
        "SQ_WAVES",
        "SQ_WAVE_CYCLES",
        "SQ_BUSY_CYCLES",
    } <= selected
    assert "TCC_HIT_sum" not in selected


# =============================================================================
# Profile mode
# =============================================================================


def test_imports_under_profile_mode_guard(monkeypatch):
    """The metric library and its loader use only stdlib, project and vendored code."""
    for module in ("utils.metric_library", "utils.yaml_inherit_loader"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    with ProfileModeImportGuard():
        importlib.import_module("utils.metric_library")
