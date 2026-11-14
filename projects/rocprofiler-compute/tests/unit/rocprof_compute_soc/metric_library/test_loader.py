# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_soc/metric_library/loader.py."""

import copy
import importlib
import sys

import pytest
from conftest import ProfileModeImportGuard

from rocprof_compute_soc.metric_library import loader
from rocprof_compute_soc.metric_library.loader import MetricLibrary, MetricLibraryError

# A base and a four-deep chain of children. Each child changes one thing.
FAMILY_PARENT_ARCH = {
    "gfx_base": None,
    "gfx_c1": "gfx_base",
    "gfx_c2": "gfx_c1",
    "gfx_c3": "gfx_c2",
    "gfx_c4": "gfx_c3",
}
FAMILY_COMPUTE_FILES = {
    "gfx_base": """\
compute.ipc:
  name: IPC
  formula: SUM(SQ_INSTS_VALU) / SUM(SQ_BUSY_CYCLES)
  unit: Instructions per Cycle
  description: ""
compute.salu_util:
  name: SALU Utilization
  formula: 100 * SUM(SQ_ACTIVE_INST_SCA) / SUM($GRBM_GUI_ACTIVE_PER_XCD)
  unit: Percent
  description: How busy the SALU was.
compute.valu_util:
  name: VALU Utilization
  formula: 100 * SUM(SQ_ACTIVE_INST_VALU) / SUM($GRBM_GUI_ACTIVE_PER_XCD)
  unit: Percent
  description: How busy the VALU was.
  peak: 100
""",
    "gfx_c1": """\
compute.salu_util:
  description: How often the SALU was busy.
""",
    "gfx_c2": """\
compute.mfma_flops_f8:
  name: MFMA FLOPs (F8)
  formula: 512 * SUM(SQ_INSTS_VALU_MFMA_MOPS_F8) / SUM($denom)
  unit: FLOP
  description: F8 MFMA operations per normUnit.
""",
    "gfx_c3": """\
remove:
  - compute.ipc
""",
    "gfx_c4": """\
compute.valu_util:
  formula: 100 * SUM(SQ_ACTIVE_INST_VALU) / SUM($GRBM_GUI_ACTIVE_PER_XCD * 2)
""",
}
# A metric with a peak, whose description names a counter it does not read.
WAVE_METRIC = """\
wave.cycles:
  name: Wave Cycles
  formula: SUM(SQ_WAVE_CYCLES)
  unit: Cycles
  description: Not counted with TCC_HIT_sum.
  peak: SUM(SQ_BUSY_CYCLES)
"""


def write_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def family(tmp_path, monkeypatch):
    """The synthetic family under tmp_path, as the only metric library chain."""
    monkeypatch.setattr(loader, "METRIC_LIBRARY_PARENT_ARCH", FAMILY_PARENT_ARCH)
    for arch, text in FAMILY_COMPUTE_FILES.items():
        write_file(tmp_path / arch / "compute.yaml", text)
    return tmp_path


def test_arch_chain_of_real_archs():
    assert loader.arch_chain("gfx950") == [
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
    ]
    assert loader.arch_chain("gfx1151") == ["gfx115x"]
    assert loader.arch_chain("gfx1250") == ["gfx1250"]


def test_arch_chain_rejects_unknown_arch():
    with pytest.raises(ValueError, match="gfx000"):
        loader.arch_chain("gfx000")


def test_arch_chain_rejects_a_cycle(monkeypatch):
    monkeypatch.setattr(
        loader,
        "METRIC_LIBRARY_PARENT_ARCH",
        {"gfx_a": "gfx_b", "gfx_b": "gfx_a"},
    )
    with pytest.raises(ValueError, match="cycle at gfx_a"):
        loader.arch_chain("gfx_a")


def test_arch_files_lists_each_file_base_first(family):
    write_file(family / "gfx_base" / "wave.yaml", "{}\n")
    files = loader.arch_files(family, "gfx_c2")
    assert files == {
        "compute.yaml": [
            family / "gfx_base" / "compute.yaml",
            family / "gfx_c1" / "compute.yaml",
            family / "gfx_c2" / "compute.yaml",
        ],
        "wave.yaml": [family / "gfx_base" / "wave.yaml"],
    }


def test_missing_arch_directory_fails(family):
    (family / "gfx_c1" / "compute.yaml").unlink()
    (family / "gfx_c1").rmdir()
    with pytest.raises(MetricLibraryError, match="gfx_c1"):
        loader.arch_files(family, "gfx_c2")


@pytest.mark.parametrize(
    "text, message",
    [("- one\n- two\n", "top level must be a mapping"), ("a: [\n", "x.yaml: ")],
)
def test_read_file_fails_on_a_bad_file(tmp_path, text, message):
    with pytest.raises(MetricLibraryError, match=message):
        loader.read_file(write_file(tmp_path / "x.yaml", text))


def test_merge_modifies_adds_and_keeps_order():
    parent = {"a.one": {"name": "One", "unit": "Waves"}, "a.two": {"name": "Two"}}
    child = {"a.one": {"unit": "Instructions"}, "a.three": {"name": "Three"}}
    merged = loader.merge_documents(parent, child, "child.yaml")
    assert merged == {
        "a.one": {"name": "One", "unit": "Instructions"},
        "a.two": {"name": "Two"},
        "a.three": {"name": "Three"},
    }
    assert list(merged) == ["a.one", "a.two", "a.three"]


def test_merge_drops_removed_ids():
    parent = {"a.one": {"name": "One"}, "a.two": {"name": "Two"}}
    merged = loader.merge_documents(parent, {"remove": ["a.one"]}, "child.yaml")
    assert merged == {"a.two": {"name": "Two"}}


@pytest.mark.parametrize(
    "child, message",
    [
        ({"remove": ["a.missing"]}, r"cannot remove \['a.missing'\]"),
        ({"remove": "a.one"}, "'remove' must be a list of ids"),
        ({"remove": [1]}, "'remove' must be a list of ids"),
        ({"remove": ["a.one"], "a.one": {"name": "Again"}}, "removed and defined"),
    ],
)
def test_bad_remove_fails(child, message):
    with pytest.raises(MetricLibraryError, match=message):
        loader.merge_documents({"a.one": {}}, child, "child.yaml")


def test_lists_and_scalars_are_replaced():
    """A scalar may change type (gfx90a overrides a peak string with a number), and a
    null value may be replaced by any kind."""
    parent = {"tags": ["a", "b"], "title": "Old", "peak": "100", "note": None}
    child = {"tags": ["c"], "title": "New", "peak": 100, "note": {"a": 1}}
    assert loader.merge_documents(parent, child, "child.yaml") == child


@pytest.mark.parametrize(
    "parent_value, child_value, kinds",
    [
        ({"unit": "X"}, "X", "a mapping with a scalar"),
        ("X", {"unit": "X"}, "a scalar with a mapping"),
        (["X"], "X", "a list with a scalar"),
        (["X"], {"unit": "X"}, "a list with a mapping"),
    ],
)
def test_value_of_another_kind_fails(parent_value, child_value, kinds):
    with pytest.raises(
        MetricLibraryError, match=f"child.yaml: 'a.one' replaces {kinds}"
    ):
        loader.merge_documents(
            {"a.one": parent_value}, {"a.one": child_value}, "child.yaml"
        )


def test_merge_leaves_its_inputs_unchanged():
    """Both ids share one mapping, as a YAML alias makes them; changing one id leaves
    the other alone."""
    shared = {"unit": "X"}
    parent = {"a.one": shared, "a.two": shared}
    child = {"remove": ["a.two"], "a.one": {"unit": "Y"}}
    parent_before = copy.deepcopy(parent)
    child_before = copy.deepcopy(child)
    loader.merge_documents(parent, child, "child.yaml")
    assert parent == parent_before
    assert child == child_before


def test_four_deep_chain(family):
    library = MetricLibrary.load(family, "gfx_c4")
    assert library.arch == "gfx_c4"
    assert list(library.metrics) == [
        "compute.salu_util",
        "compute.valu_util",
        "compute.mfma_flops_f8",
    ]
    assert library.metrics["compute.valu_util"]["formula"].endswith("* 2)")
    assert library.metrics["compute.valu_util"]["peak"] == 100
    assert (
        library.metrics["compute.salu_util"]["description"]
        == "How often the SALU was busy."
    )


def test_load_reads_collectables_apart_from_metrics(family):
    write_file(
        family / "gfx_base" / "collectables.yaml",
        "GRBM_X:\n  formula: GRBM_GUI_ACTIVE\n",
    )
    library = MetricLibrary.load(family, "gfx_c4")
    assert library.collectables == {"GRBM_X": {"formula": "GRBM_GUI_ACTIVE"}}
    assert "GRBM_X" not in library.metrics


def test_load_reads_sets_through_the_chain(family):
    write_file(
        family / "gfx_base" / "sets.yaml",
        "busy:\n  title: Busy\n  metrics: [compute.salu_util, compute.valu_util]\n"
        "old:\n  title: Old\n  metrics: [compute.salu_util]\n",
    )
    write_file(
        family / "gfx_c4" / "sets.yaml",
        "remove:\n  - old\nbusy:\n  metrics: [compute.valu_util]\n",
    )
    library = MetricLibrary.load(family, "gfx_c4")
    assert library.sets == {"busy": {"title": "Busy", "metrics": ["compute.valu_util"]}}
    assert "busy" not in library.metrics


def test_id_defined_in_two_files_fails(family):
    write_file(family / "gfx_c4" / "wave.yaml", "compute.valu_util:\n  name: Again\n")
    with pytest.raises(MetricLibraryError, match=r"\['compute.valu_util'\]"):
        MetricLibrary.load(family, "gfx_c4")


def test_counters_come_from_formula_peak_and_collectables(family):
    """Counters come only from formula and peak, of metrics from any file of the chain.
    The library's collectables resolve to counters, and the unit denominators
    (SQ_WAVES) are always added, as in detect_counters()."""
    write_file(family / "gfx_base" / "wave.yaml", WAVE_METRIC)
    write_file(
        family / "gfx_base" / "collectables.yaml",
        "GRBM_GUI_ACTIVE_PER_XCD:\n  formula: GRBM_GUI_ACTIVE / $num_xcd\n",
    )
    library = MetricLibrary.load(family, "gfx_c4")
    selected = library.get_counters_for_metrics([
        "compute.valu_util",
        "compute.mfma_flops_f8",
        "wave.cycles",
    ])
    assert {
        "SQ_ACTIVE_INST_VALU",
        "SQ_INSTS_VALU_MFMA_MOPS_F8",
        "GRBM_GUI_ACTIVE",
        "SQ_WAVES",
        "SQ_WAVE_CYCLES",
        "SQ_BUSY_CYCLES",
    } <= selected
    assert "TCC_HIT_sum" not in selected


def test_counters_of_unknown_metric_ids_fail(family):
    library = MetricLibrary.load(family, "gfx_c4")
    with pytest.raises(ValueError, match=r"for gfx_c4: \['l2.foo'\]"):
        library.get_counters_for_metrics(["compute.valu_util", "l2.foo"])


def test_shipped_library_loads():
    for arch in loader.METRIC_LIBRARY_PARENT_ARCH:
        library = MetricLibrary.load(loader.default_metric_library_dir(), arch)
        assert library.metrics and library.collectables


def test_imports_under_profile_mode_guard(monkeypatch):
    """The metric library uses only stdlib, project and vendored code."""
    monkeypatch.delitem(
        sys.modules, "rocprof_compute_soc.metric_library.loader", raising=False
    )
    with ProfileModeImportGuard():
        importlib.import_module("rocprof_compute_soc.metric_library.loader")
