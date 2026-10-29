# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for tools/validate_metric_library.py.

Run from the project directory: python3 -m pytest tools/test_validate_metric_library.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import validate_metric_library
from common import use_metric_library_family, write_text_file
from validate_metric_library import main, validate

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
    use_metric_library_family(monkeypatch, tmp_path)
    return tmp_path


@pytest.fixture
def measured(monkeypatch):
    """The synthetic family's archs measure FP16Flops and I8Ops."""
    monkeypatch.setattr(
        validate_metric_library, "_arch_ceilings", lambda arch: {"FP16Flops", "I8Ops"}
    )


def write_base_metrics(root, text):
    write_text_file(root / "gfx_base" / "compute.yaml", text)


def replace_in(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def with_peak_empirical(metric_text, ceiling):
    return metric_text + f"  peak_empirical: ${ceiling}_empirical_peak\n"


# =============================================================================
# Valid trees
# =============================================================================


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
        (
            "unit: Percent",
            "unit: Percent\n  peak: $FP16Flops_empirical_peak",
            "'peak' is a measured peak; use 'peak_empirical'",
        ),
        (
            "unit: Percent",
            "unit: Percent\n  peak_empirical: 100",
            "'peak_empirical' must be '$<ceiling>_empirical_peak'",
        ),
        ("unit: Percent", "unit: Percent\n  id: x", "'id' is not a known field"),
        ("unit: Percent", "unit: [Percent]", "'unit' must be a string"),
        ("name: VALU Utilization", 'name: "VALU \\u2192 Utilization"', "plain ASCII"),
        ('description: ""', "description: VALU \\u2122 busy", "plain ASCII"),
        ('description: ""', "description: 5", "'description' must be a string"),
    ],
)
def test_metric_errors(tree, old, new, message):
    write_base_metrics(tree, BASE_METRIC.replace(old, new))
    assert any(message in error for error in validate(tree))


def test_metric_must_be_a_mapping(tree):
    write_base_metrics(tree, BASE_METRIC + "compute.zzz: 5\n")
    assert "[gfx_base/compute.yaml] compute.zzz: must be a mapping" in validate(tree)


def test_id_category_must_be_the_file_name(tree):
    write_text_file(tree / "gfx_base" / "wave.yaml", BASE_METRIC)
    assert (
        "[gfx_base/wave.yaml] compute.valu_util: the id must have the file's category "
        "'wave'" in validate(tree)
    )


@pytest.mark.parametrize(
    "formula", ["SUM(SQ_ACTIVE_INST_VALU)", "ROUND(SUM(SQ_ACTIVE_INST_VALU), 0)"]
)
def test_metrics_cannot_share_a_formula(tree, formula):
    second = BASE_METRIC.replace("valu_util", "valu_util_copy").replace(
        "SUM(SQ_ACTIVE_INST_VALU)", formula
    )
    write_base_metrics(tree, BASE_METRIC + second)
    errors = validate(tree)
    assert any(
        "['compute.valu_util', 'compute.valu_util_copy']: share a formula" in error
        for error in errors
    )


def test_child_arch_cannot_rename_a_formula(tree):
    replace_in(
        tree / "gfx_c3" / "compute.yaml",
        "compute.ipc: !remove\n",
        "compute.insts_per_cycle:\n"
        "  name: Instructions per Cycle\n"
        "  formula: SUM(SQ_INSTS_VALU) / SUM(SQ_BUSY_CYCLES)\n"
        "  unit: Instructions per Cycle\n"
        '  description: ""\n'
        "compute.ipc: !remove\n",
    )
    assert validate(tree) == [
        "['compute.insts_per_cycle', 'compute.ipc']: share a formula on different "
        "archs (compute.ipc on gfx_base, gfx_c1, gfx_c2; compute.insts_per_cycle on "
        "gfx_c3, gfx_c4); use one id on every arch"
    ]


def test_metrics_cannot_share_a_name(tree):
    second = (
        BASE_METRIC
        .replace("valu_util", "valu_util_copy")
        .replace("name: VALU Utilization", "name: valu utilization")
        .replace("SUM(SQ_ACTIVE_INST_VALU)", "SUM(SQ_ACTIVE_INST_SCA)")
    )
    write_base_metrics(tree, BASE_METRIC + second)
    assert (
        "[gfx_base] ['compute.valu_util', 'compute.valu_util_copy']: share a name; "
        "name each after what it measures" in validate(tree)
    )


def test_listed_ids_may_share_a_formula(tree, monkeypatch):
    monkeypatch.setattr(
        validate_metric_library,
        "SHARED_FORMULA_IDS",
        {"compute.valu_util_copy": "compute.valu_util"},
    )
    second = BASE_METRIC.replace("valu_util", "valu_util_copy")
    write_base_metrics(tree, BASE_METRIC + second)
    assert not any("share a formula" in error for error in validate(tree))


def test_listed_id_may_share_only_with_its_partner(tree, monkeypatch):
    monkeypatch.setattr(
        validate_metric_library,
        "SHARED_FORMULA_IDS",
        {"compute.valu_util_copy": "compute.other_util"},
    )
    second = BASE_METRIC.replace("valu_util", "valu_util_copy")
    write_base_metrics(tree, BASE_METRIC + second)
    assert any("share a formula" in error for error in validate(tree))


def test_shared_formula_id_must_keep_its_partners_formula(tree, monkeypatch):
    monkeypatch.setattr(
        validate_metric_library,
        "SHARED_FORMULA_IDS",
        {"compute.valu_util_copy": "compute.valu_util"},
    )
    second = BASE_METRIC.replace("valu_util", "valu_util_copy").replace(
        "SUM(SQ_ACTIVE_INST_VALU)", "SUM(SQ_ACTIVE_INST_SCA)"
    )
    write_base_metrics(tree, BASE_METRIC + second)
    assert (
        "[gfx_base] compute.valu_util_copy: SHARED_FORMULA_IDS lists it with "
        "compute.valu_util, but their formulas differ" in validate(tree)
    )


def test_child_file_must_inherit_from_its_parent(tree):
    replace_in(tree / "gfx_c4" / "compute.yaml", "../gfx_c3/", "../gfx_base/")
    assert "[gfx_c4/compute.yaml] must inherit from gfx_c3/compute.yaml" in validate(
        tree
    )


def test_inherit_must_name_a_path(tree):
    replace_in(tree / "gfx_c1" / "compute.yaml", "../gfx_base/compute.yaml", "5")
    assert "[gfx_c1/compute.yaml] must inherit from gfx_base/compute.yaml" in validate(
        tree
    )


def test_base_file_cannot_inherit(tree):
    write_base_metrics(tree, "inherit: ../gfx_c1/compute.yaml\n" + BASE_METRIC)
    errors = validate(tree)
    assert "[gfx_base/compute.yaml] has no parent file to inherit" in errors


def test_merge_errors_are_reported(tree):
    write_text_file(
        tree / "gfx_c1" / "compute.yaml",
        "inherit: ../gfx_base/compute.yaml\ncompute.missing: !remove\n",
    )
    assert any("cannot remove 'compute.missing'" in error for error in validate(tree))


@pytest.mark.parametrize(
    "text, message",
    [
        ("X:\n  value: 1\n", "'formula' is required"),
        ("X:\n  formula: ' '\n", "'formula' must be a non-empty string"),
    ],
)
def test_variables_need_a_formula(tree, text, message):
    write_text_file(tree / "gfx_base" / "variables.yaml", text)
    assert any(message in error for error in validate(tree))


def test_unmeasured_ceiling(tree):
    write_base_metrics(tree, with_peak_empirical(BASE_METRIC, "FP16Flops"))
    errors = validate(tree)
    assert any("does not measure FP16Flops on gfx_base" in error for error in errors)


def test_measured_ceiling_needs_a_metric(tree, measured):
    errors = validate(tree)
    assert any("measures FP16Flops, but no metric has it" in e for e in errors)


def test_metrics_cannot_share_a_ceiling(tree, measured):
    second = BASE_METRIC.replace("valu_util", "valu_util_copy")
    write_base_metrics(
        tree,
        with_peak_empirical(BASE_METRIC, "FP16Flops")
        + with_peak_empirical(second, "FP16Flops"),
    )
    errors = validate(tree)
    assert any("share the ceiling FP16Flops" in error for error in errors)


def test_ceiling_without_metric_cannot_be_used(tree, measured):
    write_base_metrics(tree, with_peak_empirical(BASE_METRIC, "I8Ops"))
    errors = validate(tree)
    assert any("I8Ops is listed as a ceiling without a metric" in e for e in errors)


def test_arch_ceiling_without_metric_must_be_measured(tree, measured, monkeypatch):
    monkeypatch.setitem(
        validate_metric_library.ARCH_CEILINGS_WITHOUT_METRIC,
        "gfx_base",
        ("FP16Flops", "MFMAF8Flops"),
    )
    base_errors = [error for error in validate(tree) if error.startswith("[gfx_base]")]
    assert any("lists MFMAF8Flops, which the roofline" in e for e in base_errors)
    assert not any("measures FP16Flops" in error for error in base_errors)


@pytest.mark.parametrize(
    "field, shown",
    [("name: SALU Busy", "'SALU Busy' [Percent]"), ("unit: Cycles", "[Cycles]")],
)
def test_an_id_has_one_name_and_unit_on_every_arch(tree, field, shown):
    replace_in(
        tree / "gfx_c1" / "compute.yaml",
        "compute.salu_util:\n",
        f"compute.salu_util:\n  {field}\n",
    )
    errors = validate(tree)
    assert any(
        error.startswith("compute.salu_util: one name and unit on every arch")
        and shown in error
        for error in errors
    )


def test_unknown_arch_directory(tree):
    (tree / "gfx_other").mkdir()
    assert "[gfx_other] is not a known metric library arch" in validate(tree)
