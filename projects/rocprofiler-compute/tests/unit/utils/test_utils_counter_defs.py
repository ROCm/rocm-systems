# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/utils_counter_defs.py."""

import pytest

from utils.utils_counter_defs import (
    extract_counters_and_variables,
    get_build_in_vars,
)

# =============================================================================
# Tests for utils.utils_counter_defs.get_build_in_vars
# =============================================================================


class TestGetBuildInVars:
    """Tests for utils.utils_counter_defs.get_build_in_vars."""

    def test_gfx1250_uses_explicit_gui_active_sum_per_xcd(self):
        build_in_vars = get_build_in_vars("GFX1250_SERIES")

        assert (
            build_in_vars["GRBM_GUI_ACTIVE_PER_XCD"] == "GRBM_GUI_ACTIVE_sum / $num_xcd"
        )

    def test_gfx1250_only_overrides_gui_active_per_xcd(self):
        gfx1250_vars = get_build_in_vars("GFX1250_SERIES")
        mi_vars = get_build_in_vars("MI300")

        del gfx1250_vars["GRBM_GUI_ACTIVE_PER_XCD"]
        del mi_vars["GRBM_GUI_ACTIVE_PER_XCD"]

        assert gfx1250_vars == mi_vars

    @pytest.mark.parametrize("gpu_series", ["MI300", "RDNA3"])
    def test_other_series_keep_raw_gui_active_per_xcd(self, gpu_series):
        build_in_vars = get_build_in_vars(gpu_series)

        assert (
            build_in_vars["GRBM_GUI_ACTIVE_PER_XCD"] == "(GRBM_GUI_ACTIVE / $num_xcd)"
        )


# =============================================================================
# Tests for utils.utils_counter_defs.extract_counters_and_variables
# =============================================================================


class TestExtractCountersAndVariables:
    """Tests for utils.utils_counter_defs.extract_counters_and_variables."""

    def test_returns_hw_counters_and_referenced_builtin_vars(self):
        text = "$GRBM_GUI_ACTIVE_PER_XCD / SQ_WAVES"
        hw, vars_ = extract_counters_and_variables(text, "MI200")
        assert "GRBM_GUI_ACTIVE" in hw
        assert "SQ_WAVES" in hw
        assert "GRBM_GUI_ACTIVE_PER_XCD" in vars_

    def test_resolves_builtin_var_dependencies_transitively(self):
        # numActiveCUs references $GRBM_GUI_ACTIVE_PER_XCD -> GRBM_GUI_ACTIVE
        text = "$numActiveCUs"
        hw, vars_ = extract_counters_and_variables(text, "MI200")
        assert "GRBM_GUI_ACTIVE" in hw
        assert "numActiveCUs" in vars_
        assert "GRBM_GUI_ACTIVE_PER_XCD" in vars_

    def test_gfx1250_requests_only_explicit_gui_active_sum(self):
        hw, vars_ = extract_counters_and_variables("SQ_WAVES", "GFX1250_SERIES")

        assert "GRBM_GUI_ACTIVE_sum" in hw
        assert "GRBM_GUI_ACTIVE" not in hw
        assert "GRBM_GUI_ACTIVE_PER_XCD" in vars_

    def test_gfx1250_resolves_num_active_cus_to_gui_active_sum(self):
        hw, vars_ = extract_counters_and_variables("$numActiveCUs", "GFX1250_SERIES")

        assert "GRBM_GUI_ACTIVE_sum" in hw
        assert "GRBM_GUI_ACTIVE" not in hw
        assert "numActiveCUs" in vars_
        assert "GRBM_GUI_ACTIVE_PER_XCD" in vars_

    def test_unreferenced_builtin_vars_are_not_returned(self):
        # SUPPORTED_DENOM["per_cycle"] pulls in $GRBM_GUI_ACTIVE_PER_XCD
        # unconditionally; unrelated built-in vars must not appear.
        text = "SQ_WAVES"
        _, vars_ = extract_counters_and_variables(text, "MI200")
        assert "GRBM_COUNT_PER_XCD" not in vars_
        assert "GRBM_SPI_BUSY_PER_XCD" not in vars_
        assert "numActiveCUs" not in vars_

    def test_non_builtin_vars_dropped_from_variables_set(self):
        # $num_xcd is a sys var, not a built-in var; should not appear in vars_
        text = "GRBM_GUI_ACTIVE / $num_xcd"
        _, vars_ = extract_counters_and_variables(text, "MI200")
        assert "num_xcd" not in vars_

    def test_handles_ammolite_prefix(self):
        # After build_eval_string, $var becomes ammolite__var
        text = "(100 * ammolite__numActiveCUs) / ammolite__cu_per_gpu"
        _, vars_ = extract_counters_and_variables(text, "MI200")
        assert "numActiveCUs" in vars_

    def test_ignores_non_builtin_ammolite(self):
        # ammolite__cu_per_gpu is a sys var, not a built-in var
        _, vars_ = extract_counters_and_variables("ammolite__cu_per_gpu", "MI200")
        assert "cu_per_gpu" not in vars_

    def test_excluding_supported_denom_omits_spill_counters(self):
        text = (
            "avg: 100 * SUM(GRBM_CP_BUSY_sum) / SUM(GRBM_GUI_ACTIVE_sum)\n"
            "min: 100 * MIN(GRBM_CP_BUSY_sum / GRBM_GUI_ACTIVE_sum)\n"
        )
        with_denom, _ = extract_counters_and_variables(text, "MI200")
        formula_only, _ = extract_counters_and_variables(
            text, "MI200", include_supported_denom=False
        )
        assert formula_only == {"GRBM_CP_BUSY_sum", "GRBM_GUI_ACTIVE_sum"}
        assert "SQ_WAVES" in with_denom
        assert "SQ_WAVES" not in formula_only
