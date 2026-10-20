# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/metric_library.py."""

import importlib
import sys

import pytest
from common import (
    METRIC_LIBRARY_FAMILY_PARENTS,
    write_metric_library_family,
    write_text_file,
)
from conftest import ProfileModeImportGuard

from utils import metric_library
from utils.metric_library import (
    MetricLibrary,
    arch_chain,
    arch_files,
    load_metric_library,
)

GPU_SERIES = "MI300"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The synthetic family, as the only known chain."""
    monkeypatch.setattr(
        metric_library, "METRIC_LIBRARY_PARENT_ARCH", METRIC_LIBRARY_FAMILY_PARENTS
    )
    monkeypatch.setattr(MetricLibrary, "_cache", {})
    write_metric_library_family(tmp_path)
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


def test_load_metric_library_merges_the_chain(tree):
    library = load_metric_library(tree, "gfx_c3")
    assert list(library.metrics) == [
        "compute.salu_util",
        "compute.valu_util",
        "compute.mfma_flops_f8",
    ]
    assert (
        library.metrics["compute.salu_util"]["description"]
        == "How often the SALU was busy."
    )


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


def test_counters_come_from_formula_and_peak():
    library = MetricLibrary(
        {
            "a.one": {
                "formula": "SUM(SQ_WAVES)",
                "peak": "SUM(SQ_BUSY_CYCLES)",
            },
        },
        {},
    )
    selected = library.get_counters_for_metrics(["a.one"], GPU_SERIES)
    assert {"SQ_WAVES", "SQ_BUSY_CYCLES"} <= selected


def test_counters_include_builtin_variables_and_denominators(tree):
    library = load_metric_library(tree, "gfx_c4")
    selected = library.get_counters_for_metrics(["compute.valu_util"], GPU_SERIES)
    # $GRBM_GUI_ACTIVE_PER_XCD resolves to GRBM_GUI_ACTIVE; the unit denominators are
    # always collected, as detect_counters() does.
    assert {"SQ_ACTIVE_INST_VALU", "GRBM_GUI_ACTIVE", "SQ_WAVES"} <= selected
    assert "TCC_HIT_sum" not in selected


# =============================================================================
# Profile mode
# =============================================================================


def test_imports_under_profile_mode_guard(monkeypatch):
    """The metric library uses only stdlib, project and vendored code."""
    monkeypatch.delitem(sys.modules, "utils.metric_library", raising=False)
    with ProfileModeImportGuard():
        importlib.import_module("utils.metric_library")
