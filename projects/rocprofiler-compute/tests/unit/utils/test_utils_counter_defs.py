# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/utils_counter_defs.py."""

import pytest

from utils import utils_counter_defs
from utils.utils_counter_defs import (
    Collectables,
    extract_counters_and_variables,
    metric_expressions,
)

MI200_COLLECTABLES = Collectables.of_arch("gfx90a")
GFX1250_COLLECTABLES = Collectables.of_arch("gfx1250")

# =============================================================================
# Tests for utils.utils_counter_defs.Collectables
# =============================================================================


@pytest.fixture
def toy_collectables_dir(monkeypatch, tmp_path):
    """A collectables directory with no file yet, and an empty cache."""
    monkeypatch.setattr(utils_counter_defs, "COLLECTABLES_DIR", tmp_path)
    monkeypatch.setattr(Collectables, "_formulas_by_arch", {})
    return tmp_path


class TestCollectables:
    """Tests for utils.utils_counter_defs.Collectables."""

    def test_gfx1250_reads_gui_active_from_its_sum(self):
        assert (
            GFX1250_COLLECTABLES["GRBM_GUI_ACTIVE_PER_XCD"]
            == "GRBM_GUI_ACTIVE_sum / $num_xcd"
        )

    @pytest.mark.parametrize("gpu_arch", ["gfx942", "gfx1151"])
    def test_other_archs_read_gui_active(self, gpu_arch):
        assert (
            Collectables.of_arch(gpu_arch)["GRBM_GUI_ACTIVE_PER_XCD"]
            == "GRBM_GUI_ACTIVE / $num_xcd"
        )

    def test_each_collectable_comes_after_those_it_uses(self, toy_collectables_dir):
        (toy_collectables_dir / "gfx900_collectables.yaml").write_text(
            "rate: SUM($count) / 2\ncount: TA_BUSY\n", encoding="utf-8"
        )
        assert list(Collectables.of_arch("gfx900")) == ["count", "rate"]

    def test_collectables_that_use_each_other_fail(self, toy_collectables_dir):
        (toy_collectables_dir / "gfx900_collectables.yaml").write_text(
            "first: $second\nsecond: $first\n", encoding="utf-8"
        )
        with pytest.raises(SystemExit):
            Collectables.of_arch("gfx900")


# =============================================================================
# Tests for utils.utils_counter_defs.extract_counters_and_variables
# =============================================================================


class TestExtractCountersAndVariables:
    """Tests for utils.utils_counter_defs.extract_counters_and_variables."""

    def test_returns_hw_counters_and_referenced_builtin_vars(self):
        text = "$GRBM_GUI_ACTIVE_PER_XCD / SQ_WAVES"
        hw, vars_ = extract_counters_and_variables(text, MI200_COLLECTABLES)
        assert "GRBM_GUI_ACTIVE" in hw
        assert "SQ_WAVES" in hw
        assert "GRBM_GUI_ACTIVE_PER_XCD" in vars_

    def test_resolves_builtin_var_dependencies_transitively(self):
        # numActiveCUs references $GRBM_GUI_ACTIVE_PER_XCD -> GRBM_GUI_ACTIVE
        text = "$numActiveCUs"
        hw, vars_ = extract_counters_and_variables(text, MI200_COLLECTABLES)
        assert "GRBM_GUI_ACTIVE" in hw
        assert "numActiveCUs" in vars_
        assert "GRBM_GUI_ACTIVE_PER_XCD" in vars_

    def test_gfx1250_requests_only_explicit_gui_active_sum(self):
        hw, vars_ = extract_counters_and_variables("SQ_WAVES", GFX1250_COLLECTABLES)

        assert "GRBM_GUI_ACTIVE_sum" in hw
        assert "GRBM_GUI_ACTIVE" not in hw
        assert "GRBM_GUI_ACTIVE_PER_XCD" in vars_

    def test_gfx1250_resolves_num_active_cus_to_gui_active_sum(self):
        hw, vars_ = extract_counters_and_variables(
            "$numActiveCUs", GFX1250_COLLECTABLES
        )

        assert "GRBM_GUI_ACTIVE_sum" in hw
        assert "GRBM_GUI_ACTIVE" not in hw
        assert "numActiveCUs" in vars_
        assert "GRBM_GUI_ACTIVE_PER_XCD" in vars_

    def test_unreferenced_builtin_vars_are_not_returned(self):
        # SUPPORTED_DENOM["per_cycle"] pulls in $GRBM_GUI_ACTIVE_PER_XCD
        # unconditionally; unrelated built-in vars must not appear.
        text = "SQ_WAVES"
        _, vars_ = extract_counters_and_variables(text, MI200_COLLECTABLES)
        assert "GRBM_COUNT_PER_XCD" not in vars_
        assert "GRBM_SPI_BUSY_PER_XCD" not in vars_
        assert "numActiveCUs" not in vars_

    def test_non_builtin_vars_dropped_from_variables_set(self):
        # $num_xcd is a sys var, not a built-in var; should not appear in vars_
        text = "GRBM_GUI_ACTIVE / $num_xcd"
        _, vars_ = extract_counters_and_variables(text, MI200_COLLECTABLES)
        assert "num_xcd" not in vars_

    def test_handles_ammolite_prefix(self):
        # After build_eval_string, $var becomes ammolite__var
        text = "(100 * ammolite__numActiveCUs) / ammolite__cu_per_gpu"
        _, vars_ = extract_counters_and_variables(text, MI200_COLLECTABLES)
        assert "numActiveCUs" in vars_

    def test_ignores_non_builtin_ammolite(self):
        # ammolite__cu_per_gpu is a sys var, not a built-in var
        _, vars_ = extract_counters_and_variables(
            "ammolite__cu_per_gpu", MI200_COLLECTABLES
        )
        assert "cu_per_gpu" not in vars_

    def test_excluding_supported_denom_omits_spill_counters(self):
        text = (
            "avg: 100 * SUM(GRBM_CP_BUSY_sum) / SUM(GRBM_GUI_ACTIVE_sum)\n"
            "min: 100 * MIN(GRBM_CP_BUSY_sum / GRBM_GUI_ACTIVE_sum)\n"
        )
        with_denom, _ = extract_counters_and_variables(text, MI200_COLLECTABLES)
        formula_only, _ = extract_counters_and_variables(
            text, MI200_COLLECTABLES, include_supported_denom=False
        )
        assert formula_only == {"GRBM_CP_BUSY_sum", "GRBM_GUI_ACTIVE_sum"}
        assert "SQ_WAVES" in with_denom
        assert "SQ_WAVES" not in formula_only


# =============================================================================
# Tests for utils.utils_counter_defs.metric_expressions
# =============================================================================


class TestMetricExpressions:
    """Tests for utils.utils_counter_defs.metric_expressions."""

    def test_reads_only_keys_the_header_declares(self):
        table = {"header": {"metric": "Metric", "avg": "Avg", "unit": "Unit"}}
        metric = {
            "avg": "SUM(SQ_WAVES)",
            "unit": "Waves",
            "tips": "See SQ_BUSY_CYCLES",
        }
        assert metric_expressions(table, metric) == ["SUM(SQ_WAVES)", "Waves"]

    def test_skips_empty_and_none_values(self):
        table = {"header": {"metric": "Metric", "avg": "Avg", "peak": "Peak"}}
        assert metric_expressions(table, {"avg": "None", "peak": 5}) == []
        assert metric_expressions(table, {"avg": "", "peak": None}) == []

    def test_simple_box_reads_only_expr(self):
        table = {
            "header": {"metric": "Metric", "expr": "Expression", "unit": "Unit"},
            "cli_style": "simple_box",
        }
        metric = {"expr": "TCC_HIT[::_1]", "unit": "Hits"}
        assert metric_expressions(table, metric) == ["TCC_HIT[::_1]"]
