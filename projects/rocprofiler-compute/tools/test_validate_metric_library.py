# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for tools/validate_metric_library.py.

Run from the project directory: python3 -m pytest tools/test_validate_metric_library.py
"""

import functools
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import validate_metric_library
from common import use_metric_library_family, write_text_file
from validate_metric_library import main, validate

import config
from utils.metric_library import MetricLibrary, arch_chain, default_metric_library_dir

BASE_SETS = """\
busy:
  title: Busy
  metrics:
  - compute.salu_util
  - compute.valu_util
"""

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


@pytest.fixture
def shipped_copy(tmp_path, monkeypatch):
    """copy_shipped_trees() into tmp_path, with an empty metric library cache."""
    monkeypatch.setattr(MetricLibrary, "_cache", {})
    return functools.partial(copy_shipped_trees, tmp_path)


def copy_shipped_trees(root, arch):
    """Copy the shipped metric library and analysis configs of arch and its ancestors
    under root, and return the (library, configs) copies."""
    library = root / "metric_library"
    configs = root / "analysis_configs"
    for source, target in (
        (default_metric_library_dir(), library),
        (config.analysis_configs_dir, configs),
    ):
        for chain_arch in arch_chain(arch):
            shutil.copytree(source / chain_arch, target / chain_arch)
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


def test_main_reports_errors(tree, monkeypatch, capsys):
    monkeypatch.setattr(
        validate_metric_library, "default_metric_library_dir", lambda: tree
    )
    monkeypatch.setattr(config, "analysis_configs_dir", None)
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
        ("  - compute.salu_util\n", "  - 5\n", "'metrics' must list metric ids"),
        ("compute.salu_util", "compute.valu_util", "'metrics' lists an id twice"),
        ("compute.salu_util", "compute.nope", "unknown metric ids ['compute.nope']"),
    ],
)
def test_set_errors(tree, old, new, message):
    write_text_file(tree / "gfx_base" / "sets.yaml", BASE_SETS.replace(old, new))
    assert any(message in error for error in validate(tree))


def test_set_must_fit_in_one_pass(shipped_copy, monkeypatch):
    library, _ = shipped_copy("gfx908")
    monkeypatch.setattr(
        validate_metric_library.mi_gpu_specs,
        "get_perfmon_config",
        lambda arch: {"SQ": 2},
    )
    errors = validate(library)
    assert (
        "[gfx908] set compute_thruput_util: block SQ needs 3 counters "
        "but one pass holds 2"
    ) in errors


def test_unknown_arch_directory(tree):
    (tree / "gfx_other").mkdir()
    assert "[gfx_other] is not a known metric library arch" in validate(tree)


# =============================================================================
# Equivalence with analysis_configs
# =============================================================================


def test_schema_errors_skip_the_equivalence_check(shipped_copy):
    library, configs = shipped_copy("gfx908")
    replace_in(library / "gfx908" / "grbm.yaml", "unit: Cycles", "unit: Gbps")
    assert validate(library, configs) == [
        "[gfx908/grbm.yaml] grbm.gui_active_cycles: 'unit' 'Gbps' is an alias; "
        "use 'GB/s'"
    ]


def test_changed_library_formula(shipped_copy):
    library, configs = shipped_copy("gfx908")
    replace_in(
        library / "gfx908" / "cp.yaml",
        "100 * SUM(CPF_CPF_STAT_BUSY)",
        "99 * SUM(CPF_CPF_STAT_BUSY)",
    )
    errors = validate(library, configs)
    assert any("not a library formula" in error for error in errors)
    assert any("no analysis config row shows its formula" in error for error in errors)


def test_extra_library_counter(shipped_copy):
    library, configs = shipped_copy("gfx908")
    replace_in(
        library / "gfx908" / "grbm.yaml",
        "unit: Cycles\n",
        "unit: Cycles\n  peak: SUM(SQ_EXTRA_LEVEL)\n",
    )
    errors = validate(library, configs)
    assert "[gfx908] extra counters: ['SQ_EXTRA_LEVEL']" in errors


def test_changed_unit(shipped_copy):
    library, configs = shipped_copy("gfx908")
    replace_in(library / "gfx908" / "cp.yaml", "unit: Percent", "unit: Cycles")
    errors = validate(library, configs)
    assert any("unit 'Percent' is not 'Cycles' (cp.cpc_l2_util)" in e for e in errors)


def test_unit_variant_must_differ(shipped_copy, monkeypatch):
    monkeypatch.setitem(
        validate_metric_library.UNIT_VARIANTS,
        "gfx908",
        {"7.1.8": ("LDS Allocation", "Bytes per Workgroup")},
    )
    assert any(
        "remove it from UNIT_VARIANTS" in error
        for error in validate(*shipped_copy("gfx908"))
    )


def test_changed_variable(shipped_copy):
    library, configs = shipped_copy("gfx908")
    replace_in(
        library / "gfx908" / "variables.yaml", "GRBM_COUNT / $num_xcd", "GRBM_COUNT / 2"
    )
    assert any("variables differ" in error for error in validate(library, configs))


def test_changed_row_peak(shipped_copy):
    library, configs = shipped_copy("gfx908")
    path = configs / "gfx908" / "0200_system_speed_of_light.yaml"
    replace_in(path, "peak: $max_sclk / 1000 * 64 *", "peak: $max_sclk / 1000 * 32 *")
    errors = validate(library, configs)
    assert any("is neither the 'peak' nor the 'peak_empirical'" in e for e in errors)


def test_changed_hidden_row_peak(shipped_copy):
    library, configs = shipped_copy("gfx908")
    path = configs / "gfx908" / "0200_system_speed_of_light.yaml"
    replace_in(path, "$cu_per_gpu * 512 / 1000", "$cu_per_gpu * 999 / 1000")
    errors = validate(library, configs)
    assert any("2.1.1 MFMA FLOPs (BF16): peak" in error for error in errors)


def test_library_peak_must_be_shown(shipped_copy):
    library, configs = shipped_copy("gfx908")
    replace_in(
        library / "gfx908" / "cp.yaml",
        "CPC_CPC_TCIU_IDLE)\n  unit: Percent\n",
        "CPC_CPC_TCIU_IDLE)\n  unit: Percent\n  peak: 100\n",
    )
    errors = validate(library, configs)
    assert "[gfx908] cp.cpc_l2_util: no analysis config row shows its 'peak'" in errors


def test_changed_row_measured_peak(shipped_copy):
    library, configs = shipped_copy("gfx90a")
    path = configs / "gfx90a" / "0400_roofline.yaml"
    replace_in(path, "$LDSBw_empirical_peak", "$L2Bw_empirical_peak")
    errors = validate(library, configs)
    assert any("4.1.11 LDS Bandwidth: peak $L2Bw_empirical_peak" in e for e in errors)


@pytest.mark.parametrize(
    "label, metric_id",
    [
        ("MFMA FLOPs (MXF8)", "compute.mfma_flops_mxf8"),
        ("MFMA FLOPs (F8)", "compute.mfma_flops_f8"),
    ],
)
def test_shared_formula_rows_show_the_id_their_label_names(
    monkeypatch, label, metric_id
):
    """MXF8 rows read the F8 counter but are checked against their own id."""
    monkeypatch.setattr(
        validate_metric_library,
        "SHARED_FORMULA_IDS",
        {"compute.mfma_flops_mxf8": "compute.mfma_flops_f8"},
    )
    names = {
        "mfma flops (mxf8)": "compute.mfma_flops_mxf8",
        "mfma flops (f8)": "compute.mfma_flops_f8",
    }
    row = validate_metric_library._Row("4.1.0", label, "SUM(F8)", None, None)
    assert (
        validate_metric_library._row_metric_id(
            row, {}, {"SUM(F8)": "compute.mfma_flops_f8"}, names, set(names.values())
        )
        == metric_id
    )


def test_display_variant_row_is_checked_for_its_unit(shipped_copy):
    """4.1.11 shows lds.theoretical_bw in another form; its unit is still checked."""
    library, configs = shipped_copy("gfx908")
    replace_in(
        configs / "gfx908" / "0400_roofline.yaml",
        "* 4 * $lds_banks_per_cu) / SUM(End_Timestamp - Start_Timestamp)\n"
        "          unit: GB/s",
        "* 4 * $lds_banks_per_cu) / SUM(End_Timestamp - Start_Timestamp)\n"
        "          unit: Cycles",
    )
    errors = validate(library, configs)
    assert any("4.1.11 LDS Bandwidth: unit 'Cycles'" in error for error in errors)


def test_variant_position_must_exist(shipped_copy, monkeypatch):
    monkeypatch.setitem(
        validate_metric_library.UNIT_VARIANTS,
        "gfx908",
        {**validate_metric_library.UNIT_VARIANTS["gfx908"], "99.9.9": ("X", "Y")},
    )
    errors = validate(*shipped_copy("gfx908"))
    assert "[gfx908] UNIT_VARIANTS lists 99.9.9, but no row has that position" in errors


def test_unknown_measured_peak_is_an_error(shipped_copy):
    """gfx908 has no roofline benchmark, so its row peaks are not checked; a typo
    still is."""
    library, configs = shipped_copy("gfx908")
    replace_in(
        configs / "gfx908" / "0400_roofline.yaml",
        "$LDSBw_empirical_peak",
        "$LDSBz_empirical_peak",
    )
    errors = validate(library, configs)
    assert any("peak $LDSBz_empirical_peak is neither" in e for e in errors)


@pytest.mark.parametrize(
    "peak, shown",
    [
        ("$L1Bw_empirical_peak", "$L1Bw_empirical_peak*1e9"),
        ("$max_sclk*64/1000", "$max_sclk*64*1e6"),
    ],
)
def test_bytes_per_second_row_shows_its_peak_scaled(peak, shown):
    """A row in seconds shows a GB/s peak in Bytes/s."""
    equation = "SUM(TCP_TCC_READ_REQ_sum)" + validate_metric_library.SECONDS_SCALE
    row = validate_metric_library._Row("4.1.3", "L1 Cache BW", equation, None, None)
    assert validate_metric_library._row_peak(row, peak) == shown


def test_changed_rewritten_display_variant(shipped_copy):
    library, configs = shipped_copy("gfx908")
    replace_in(
        configs / "gfx908" / "0400_roofline.yaml",
        "* 4 * $lds_banks_per_cu) / SUM(End",
        "* 8 * $lds_banks_per_cu) / SUM(End",
    )
    errors = validate(library, configs)
    assert any("4.1.11 LDS Bandwidth: not the listed form" in e for e in errors)


def test_changed_bytes_per_second_row(shipped_copy):
    """3.1.26 shows fabric.rd_bw in Bytes/s; it is checked even though row 17.2.0 also
    shows fabric.rd_bw."""
    library, configs = shipped_copy("gfx90a")
    replace_in(
        configs / "gfx90a" / "0300_memory_chart.yaml",
        "TCC_EA_RDREQ_32B_sum) * 64) / (SUM(End",
        "TCC_EA_RDREQ_32B_sum) * 128) / (SUM(End",
    )
    errors = validate(library, configs)
    assert any("3.1.26 L2-Fabric Read BW: not a library formula" in e for e in errors)


def test_scaled_display_variant_unit(shipped_copy):
    """A row that shows a GB/s metric in seconds must say Bytes/s."""
    library, configs = shipped_copy("gfx908")
    replace_in(
        configs / "gfx908" / "0300_memory_chart.yaml",
        "/ 1e9)\n          unit: Bytes/s",
        "/ 1e9)\n          unit: GB/s",
    )
    errors = validate(library, configs)
    assert any(
        "3.1.16 VL1_L2 Read BW: unit 'GB/s' is not 'Bytes/s'" in e for e in errors
    )


def test_renamed_display_variant(shipped_copy):
    library, configs = shipped_copy("gfx908")
    path = configs / "gfx908" / "0400_roofline.yaml"
    replace_in(path, "LDS Bandwidth:", "LDS BW:")
    assert any("4.1.11 is now" in error for error in validate(library, configs))


def test_set_with_another_metric(shipped_copy):
    library, configs = shipped_copy("gfx908")
    replace_in(library / "gfx908" / "sets.yaml", "compute.salu_util", "l2.util")
    errors = validate(library, configs)
    assert any(
        error.startswith("[gfx908] set compute_thruput_util: counters missing")
        for error in errors
    )


def test_set_missing_from_the_library(shipped_copy):
    library, configs = shipped_copy("gfx908")
    path = library / "gfx908" / "sets.yaml"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text[: text.index("launch_stats:")] + text[text.index("mem_thruput:") :]
    )
    assert (
        "[gfx908] sets differ from profile_configs/sets: missing ['launch_stats'], "
        "extra []"
    ) in validate(library, configs)


def test_set_without_metrics_must_collect_nothing(shipped_copy, monkeypatch):
    library, configs = shipped_copy("gfx908")
    monkeypatch.setattr(
        validate_metric_library,
        "SETS_WITHOUT_METRICS",
        {"gfx908": ("compute_thruput_flops", "mem_thruput")},
    )
    assert (
        "[gfx908] set mem_thruput is in SETS_WITHOUT_METRICS but has metrics; "
        "remove the entry"
    ) in validate(library, configs)
