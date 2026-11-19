# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for tools/validate_metric_library.py.

Run from the project directory: python3 -m pytest tools/test_validate_metric_library.py
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import validate_metric_library  # noqa: E402
from validate_metric_library import main, validate  # noqa: E402

from rocprof_compute_soc.metric_library import loader  # noqa: E402

BASE_METRIC = """\
compute.valu_util:
  name: VALU Utilization
  formula: SUM(SQ_ACTIVE_INST_VALU)
  unit: Percent
  description: ""
"""
BASE_SETS = """\
busy:
  title: Busy
  metrics:
  - compute.salu_util
  - compute.ipc
"""
# A base and two children: gfx_c1 changes a description, gfx_c2 removes a metric.
FAMILY_PARENT_ARCH = {"gfx_base": None, "gfx_c1": "gfx_base", "gfx_c2": "gfx_c1"}
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
""",
    "gfx_c1": """\
compute.salu_util:
  description: How often the SALU was busy.
""",
    "gfx_c2": """\
remove:
  - compute.ipc
""",
}


def write_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_base_metrics(root, text):
    write_file(root / "gfx_base" / "compute.yaml", text)


def replace_in(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def with_peak_empirical(metric_text, ceiling):
    return metric_text + f"  peak_empirical: ${ceiling}_empirical_peak\n"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The synthetic family under tmp_path, as the only metric library chain."""
    monkeypatch.setattr(loader, "METRIC_LIBRARY_PARENT_ARCH", FAMILY_PARENT_ARCH)
    for arch, text in FAMILY_COMPUTE_FILES.items():
        write_file(tmp_path / arch / "compute.yaml", text)
    return tmp_path


@pytest.fixture
def measured(monkeypatch):
    """The synthetic family's archs measure FP16Flops and I8Ops."""
    monkeypatch.setattr(
        validate_metric_library, "arch_ceilings", lambda arch: {"FP16Flops", "I8Ops"}
    )


def test_shipped_library_is_valid():
    assert validate(loader.default_metric_library_dir()) == []


def test_main_reports_errors(tree, monkeypatch, capsys):
    monkeypatch.setattr(loader, "default_metric_library_dir", lambda: tree)
    assert main() == 0
    write_base_metrics(tree, BASE_METRIC.replace("compute.valu_util", "Bad"))
    assert main() == 1
    assert "error(s) found" in capsys.readouterr().out


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
    write_file(
        tree / "gfx_base" / "wave.yaml", BASE_METRIC.replace("valu_util", "waves")
    )
    assert (
        "[gfx_base/wave.yaml] compute.waves: the id must have the file's category "
        "'wave'" in validate(tree)
    )


def test_unsorted_keys_are_reported(tree):
    second = BASE_METRIC.replace("valu_util", "alu_util").replace(
        "SUM(SQ_ACTIVE_INST_VALU)", "SUM(SQ_ACTIVE_INST_ANY)"
    )
    write_base_metrics(tree, BASE_METRIC + second.replace("VALU", "ALU"))
    assert "[gfx_base/compute.yaml] top-level keys are not sorted" in validate(tree)


@pytest.mark.parametrize(
    "formula", ["SUM(SQ_ACTIVE_INST_VALU)", "ROUND(SUM(SQ_ACTIVE_INST_VALU), 0)"]
)
def test_metrics_cannot_share_a_formula(tree, formula):
    second = BASE_METRIC.replace("valu_util", "valu_util_copy").replace(
        "SUM(SQ_ACTIVE_INST_VALU)", formula
    )
    write_base_metrics(tree, BASE_METRIC + second)
    assert any(
        "['compute.valu_util', 'compute.valu_util_copy']: share a formula" in error
        for error in validate(tree)
    )


@pytest.mark.parametrize(
    "old, new, message",
    [
        (
            "formula: SUM(SQ_ACTIVE_INST_VALU)",
            "formula: SUM(SQ_ACTIVE_INST_VALU) +",
            "formula does not parse: invalid syntax",
        ),
        (
            "unit: Percent",
            "unit: Percent\n  peak: $max_sclk * ",
            "peak does not parse: invalid syntax",
        ),
        (
            "formula: SUM(SQ_ACTIVE_INST_VALU)",
            "formula: SUM(SQ_ACTIVE_INST_VALU.real)",
            "formula does not parse: the expression parser does not allow",
        ),
    ],
)
def test_expressions_must_parse(tree, old, new, message):
    write_base_metrics(tree, BASE_METRIC.replace(old, new))
    assert f"[gfx_base] compute.valu_util: {message}" in " ".join(validate(tree))


def test_child_arch_cannot_rename_a_formula(tree):
    write_file(
        tree / "gfx_c2" / "compute.yaml",
        "remove:\n"
        "  - compute.ipc\n"
        "compute.insts_per_cycle:\n"
        "  name: Instructions per Cycle\n"
        "  formula: SUM(SQ_INSTS_VALU) / SUM(SQ_BUSY_CYCLES)\n"
        "  unit: Instructions per Cycle\n"
        '  description: ""\n',
    )
    assert validate(tree) == [
        "['compute.insts_per_cycle', 'compute.ipc']: share a formula on different "
        "archs (compute.ipc on gfx_base, gfx_c1; compute.insts_per_cycle on "
        "gfx_c2); use one id on every arch"
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


def test_merge_errors_are_reported(tree):
    write_file(tree / "gfx_c1" / "compute.yaml", "remove:\n  - compute.missing\n")
    assert any("cannot remove ['compute.missing']" in error for error in validate(tree))


@pytest.mark.parametrize(
    "text, message",
    [
        ("X:\n  value: 1\n", "'formula' is required"),
        ("X:\n  formula: ' '\n", "'formula' must be a non-empty string"),
    ],
)
def test_collectables_need_a_formula(tree, text, message):
    write_file(tree / "gfx_base" / "collectables.yaml", text)
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
    assert any(
        error.startswith("compute.salu_util: one name and unit on every arch")
        and shown in error
        for error in validate(tree)
    )


def test_unknown_arch_directory(tree):
    (tree / "gfx_other").mkdir()
    assert validate(tree) == ["[gfx_other] is not a known metric library arch"]


def test_mapped_arch_needs_a_directory(tree):
    shutil.rmtree(tree / "gfx_c1")
    assert validate(tree) == [
        "[gfx_c1] is in METRIC_LIBRARY_PARENT_ARCH but has no directory"
    ]


@pytest.mark.parametrize(
    "old, new, message",
    [
        ("busy:", "Busy:", "a set option must be snake_case"),
        ("  title: Busy\n", "", "'title' is required"),
        (
            "title: Busy",
            "title: Busy\n  description: x",
            "'description' is not a known",
        ),
        ("title: Busy", "title: ' '", "'title' must be a non-empty string"),
        ("  - compute.ipc\n", "  - 5\n", "'metrics' must list metric ids"),
        ("compute.ipc", "compute.salu_util", "'metrics' lists an id twice"),
        ("compute.ipc", "compute.nope", "unknown metric ids ['compute.nope']"),
    ],
)
def test_set_errors(tree, old, new, message):
    write_file(tree / "gfx_base" / "sets.yaml", BASE_SETS.replace(old, new))
    assert any(message in error for error in validate(tree))


def test_set_must_fit_in_one_pass(monkeypatch):
    monkeypatch.setattr(
        validate_metric_library.mi_gpu_specs,
        "get_perfmon_config",
        lambda arch: {"SQ": 2},
    )
    errors = validate(loader.default_metric_library_dir())
    assert (
        "[gfx908] set compute_thruput_util: block SQ needs 3 counters "
        "but one pass holds 2"
    ) in errors
