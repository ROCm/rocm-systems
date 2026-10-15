# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for tools/validate_metric_library.py."""

import pytest
from common import (
    METRIC_LIBRARY_FAMILY_PARENTS,
    write_metric_library_family,
    write_text_file,
)

from tools import validate_metric_library
from tools.validate_metric_library import main, validate
from utils import metric_library
from utils.metric_library import default_metric_library_dir

BASE_METRIC = """\
compute.valu_util:
  name: VALU Utilization
  formula: SUM(SQ_ACTIVE_INST_VALU)
  unit: Percent
  description: ""
"""


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The synthetic family, as the only known chain."""
    for module in (metric_library, validate_metric_library):
        monkeypatch.setattr(
            module, "METRIC_LIBRARY_PARENT_ARCH", METRIC_LIBRARY_FAMILY_PARENTS
        )
    write_metric_library_family(tmp_path)
    return tmp_path


def write_base_metrics(root, text):
    write_text_file(root / "gfx_base" / "0200_sol.yaml", text)


# =============================================================================
# Valid trees
# =============================================================================


def test_shipped_metric_library_is_valid():
    assert validate(default_metric_library_dir()) == []


def test_synthetic_tree_is_valid(tree):
    assert validate(tree) == []


def test_main_reports_errors(tree, monkeypatch, capsys):
    monkeypatch.setattr(
        validate_metric_library, "default_metric_library_dir", lambda: tree
    )
    assert main() == 0
    write_base_metrics(tree, BASE_METRIC.replace("compute.valu_util", "Bad"))
    assert main() == 1
    assert "error(s) found" in capsys.readouterr().out


# =============================================================================
# Library files
# =============================================================================


@pytest.mark.parametrize(
    "old, new, message",
    [
        ('  description: ""\n', "", "'description' is required"),
        ("compute.valu_util", "valu_util", "the id must look like"),
        ("compute.valu_util", "system.valu_util", "the id names a view"),
        ("name: VALU Utilization", "name: ' '", "'name' must be a non-empty string"),
        ("formula: SUM(SQ_ACTIVE_INST_VALU)", "formula: 5", "'formula' must be"),
        ("formula: SUM(SQ_ACTIVE_INST_VALU)", "formula: null", "'formula' must be"),
        ("unit: Percent", "unit: Gbps", "'unit' 'Gbps' is an alias; use 'GB/s'"),
        ("unit: Percent", "unit: Percent\n  peak: true", "'peak' must be"),
        ("unit: Percent", "unit: Percent\n  id: x", "'id' is not a known field"),
        ("unit: Percent", "unit: Percent\n  unit: Pct", "duplicate keys ['unit']"),
    ],
)
def test_metric_errors(tree, old, new, message):
    write_base_metrics(tree, BASE_METRIC.replace(old, new))
    assert any(message in error for error in validate(tree))


def test_duplicate_metric_id(tree):
    write_text_file(tree / "gfx_base" / "0300_dup.yaml", BASE_METRIC)
    assert any("defined twice" in error for error in validate(tree))


def test_child_file_must_inherit_from_its_parent(tree):
    path = tree / "gfx_c4" / "0200_sol.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("../gfx_c3/", "../gfx_base/"),
        encoding="utf-8",
    )
    assert "[gfx_c4/0200_sol.yaml] must inherit from gfx_c3/0200_sol.yaml" in validate(
        tree
    )


def test_base_file_cannot_inherit(tree):
    write_base_metrics(tree, "inherit: ../gfx_c1/0200_sol.yaml\n" + BASE_METRIC)
    errors = validate(tree)
    assert "[gfx_base/0200_sol.yaml] has no parent file to inherit" in errors


def test_merge_errors_are_reported(tree):
    write_text_file(
        tree / "gfx_c1" / "0200_sol.yaml",
        "inherit: ../gfx_base/0200_sol.yaml\nmem.missing: !remove\n",
    )
    assert any("cannot remove 'mem.missing'" in error for error in validate(tree))


def test_variables_need_a_formula(tree):
    write_text_file(tree / "gfx_base" / "variables.yaml", "X:\n  value: 1\n")
    assert any("'formula' is required" in error for error in validate(tree))


def test_unknown_arch_directory(tree):
    (tree / "gfx_other").mkdir()
    assert "[gfx_other] is not a known metric library arch" in validate(tree)
