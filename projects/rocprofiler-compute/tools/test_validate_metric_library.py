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
from common import (  # noqa: E402
    METRIC_LIBRARY_FAMILY_PARENTS,
    write_metric_library_family,
    write_text_file,
)
from validate_metric_library import ANALYSIS_CONFIGS, main, validate  # noqa: E402

from utils import metric_library  # noqa: E402
from utils.metric_library import MetricLibrary, default_metric_library_dir  # noqa: E402

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


@pytest.fixture
def measured(monkeypatch):
    """The synthetic family's archs measure FP16Flops and I8Ops."""
    monkeypatch.setattr(
        validate_metric_library, "_arch_ceilings", lambda arch: {"FP16Flops", "I8Ops"}
    )


@pytest.fixture
def gfx908_copy(tmp_path, monkeypatch):
    """Copies of the gfx908 metric library and analysis configs: (library, configs)."""
    monkeypatch.setattr(MetricLibrary, "_cache", {})
    library = tmp_path / "metric_library"
    configs = tmp_path / "analysis_configs"
    shutil.copytree(default_metric_library_dir() / "gfx908", library / "gfx908")
    shutil.copytree(ANALYSIS_CONFIGS / "gfx908", configs / "gfx908")
    return library, configs


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


def test_shipped_metric_library_is_valid():
    assert validate(default_metric_library_dir(), ANALYSIS_CONFIGS) == []


def test_synthetic_tree_is_valid(tree):
    assert validate(tree) == []


def test_main_reports_errors(tree, monkeypatch, capsys):
    monkeypatch.setattr(
        validate_metric_library, "default_metric_library_dir", lambda: tree
    )
    monkeypatch.setattr(validate_metric_library, "ANALYSIS_CONFIGS", None)
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
        ("unit: Percent", "unit: Percent\n  unit: Pct", "duplicate keys ['unit']"),
    ],
)
def test_metric_errors(tree, old, new, message):
    write_base_metrics(tree, BASE_METRIC.replace(old, new))
    assert any(message in error for error in validate(tree))


def test_id_category_must_be_the_file_name(tree):
    write_text_file(tree / "gfx_base" / "wave.yaml", BASE_METRIC)
    assert (
        "[gfx_base/wave.yaml] compute.valu_util: the id must have the file's category "
        "'wave'" in validate(tree)
    )


def test_keys_must_be_sorted(tree):
    unsorted = BASE_METRIC + BASE_METRIC.replace("valu_util", "salu_util")
    write_base_metrics(tree, unsorted)
    errors = validate(tree)
    assert any("[gfx_base/compute.yaml] keys are not sorted" in e for e in errors)


def test_metrics_cannot_share_a_formula(tree):
    second = BASE_METRIC.replace("valu_util", "valu_util_copy")
    write_base_metrics(tree, BASE_METRIC + second)
    errors = validate(tree)
    assert any(
        "['compute.valu_util', 'compute.valu_util_copy']: share a formula" in error
        for error in errors
    )


def test_child_file_must_inherit_from_its_parent(tree):
    path = tree / "gfx_c4" / "compute.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("../gfx_c3/", "../gfx_base/"),
        encoding="utf-8",
    )
    assert "[gfx_c4/compute.yaml] must inherit from gfx_c3/compute.yaml" in validate(
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


def test_variables_need_a_formula(tree):
    write_text_file(tree / "gfx_base" / "variables.yaml", "X:\n  value: 1\n")
    assert any("'formula' is required" in error for error in validate(tree))


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
    assert any("I8Ops is in CEILINGS_WITHOUT_METRIC" in error for error in errors)


def test_unknown_arch_directory(tree):
    (tree / "gfx_other").mkdir()
    assert "[gfx_other] is not a known metric library arch" in validate(tree)


# =============================================================================
# Equivalence with analysis_configs
# =============================================================================


def test_gfx908_copy_matches(gfx908_copy):
    assert validate(*gfx908_copy) == []


def test_changed_library_formula(gfx908_copy):
    library, configs = gfx908_copy
    replace_in(
        library / "gfx908" / "cp.yaml",
        "100 * SUM(CPF_CPF_STAT_BUSY)",
        "99 * SUM(CPF_CPF_STAT_BUSY)",
    )
    errors = validate(library, configs)
    assert any("not a library formula" in error for error in errors)
    assert any("no analysis config row shows its formula" in error for error in errors)


def test_changed_analysis_config_row(gfx908_copy):
    library, configs = gfx908_copy
    path = configs / "gfx908" / "0500_command_processor_cpc_cpf.yaml"
    replace_in(
        path, "avg: 100 * SUM(CPF_CPF_STAT_BUSY)", "avg: 98 * SUM(CPF_CPF_STAT_BUSY)"
    )
    assert any("not a library formula" in error for error in validate(library, configs))


def test_changed_variable(gfx908_copy):
    library, configs = gfx908_copy
    replace_in(
        library / "gfx908" / "variables.yaml", "GRBM_COUNT / $num_xcd", "GRBM_COUNT / 2"
    )
    assert any("variables differ" in error for error in validate(library, configs))


def test_changed_row_peak(gfx908_copy):
    library, configs = gfx908_copy
    path = configs / "gfx908" / "0200_system_speed_of_light.yaml"
    replace_in(path, "peak: $max_sclk / 1000 * 64 *", "peak: $max_sclk / 1000 * 32 *")
    errors = validate(library, configs)
    assert any("is neither the 'peak' nor the 'peak_empirical'" in e for e in errors)


def test_renamed_display_variant(gfx908_copy):
    library, configs = gfx908_copy
    path = configs / "gfx908" / "0300_memory_chart.yaml"
    replace_in(path, "VL1_L2 Read BW", "VL1_L2 Rd BW")
    assert any("3.1.16 is now" in error for error in validate(library, configs))
