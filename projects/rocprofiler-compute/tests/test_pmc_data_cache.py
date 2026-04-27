# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.pmc_data_cache.PmcDataCache (Phase 1)."""

import pandas as pd
import pytest

from utils.metrics.pmc_data_cache import PmcDataCache


class TestPmcDataCache:
    """Tests for utils.metrics.pmc_data_cache.PmcDataCache."""

    def _make_pmc_perf_only_df(self):
        """Build a MultiIndex DataFrame whose only top-level is pmc_perf."""
        return pd.concat(
            {
                "pmc_perf": pd.DataFrame({
                    "SQ_WAVES": [100, 200, 150],
                    "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
                })
            },
            axis=1,
        )

    def _make_two_level_df(self):
        """MultiIndex DataFrame with pmc_perf and one SQ_INST_LEVEL_VMEM level."""
        return pd.concat(
            {
                "pmc_perf": pd.DataFrame({
                    "Kernel_Name": ["k", "k"],
                    "Dispatch_ID": [0, 1],
                    "SQ_WAVES": [100, 200],
                    "SQ_INSTS_VMEM": [10, 20],
                }),
                "SQ_INST_LEVEL_VMEM": pd.DataFrame({
                    "Kernel_Name": ["k", "k"],
                    "Dispatch_ID": [0, 1],
                    "SQ_ACCUM_PREV_HIRES": [50, 70],
                }),
            },
            axis=1,
        )

    def test_pmc_perf_columns_lift_to_flat_keys(self):
        """Every pmc_perf column appears at its natural name as a pd.Series."""
        cache = PmcDataCache(self._make_pmc_perf_only_df())
        assert "SQ_WAVES" in cache
        assert "GRBM_GUI_ACTIVE" in cache

        sq_waves = cache["SQ_WAVES"]
        assert isinstance(sq_waves, pd.Series)
        assert sq_waves.tolist() == [100, 200, 150]

        grbm = cache["GRBM_GUI_ACTIVE"]
        assert isinstance(grbm, pd.Series)
        assert grbm.tolist() == [1000, 2000, 1500]

    def test_accum_rename_for_non_pmc_perf_levels(self):
        """SQ_ACCUM_PREV_HIRES from non-pmc_perf levels is renamed to {level}_ACCUM."""
        cache = PmcDataCache(self._make_two_level_df())

        assert "SQ_INST_LEVEL_VMEM_ACCUM" in cache
        accum_series = cache["SQ_INST_LEVEL_VMEM_ACCUM"]
        assert isinstance(accum_series, pd.Series)
        assert accum_series.tolist() == [50, 70]

        # The original SQ_ACCUM_PREV_HIRES name is not exposed; only the renamed key is.
        assert "SQ_ACCUM_PREV_HIRES" not in cache

    def test_missing_key_raises_keyerror(self):
        """Direct subscript on a missing column raises KeyError."""
        cache = PmcDataCache(self._make_pmc_perf_only_df())
        with pytest.raises(KeyError):
            cache["UNKNOWN_COUNTER"]

    def test_contains_reports_false_for_missing_key(self):
        """`in` returns False for unknown column names."""
        cache = PmcDataCache(self._make_pmc_perf_only_df())
        assert ("UNKNOWN_COUNTER" in cache) is False

    def test_level_with_no_accum_column_is_skipped(self):
        """A non-pmc_perf level missing SQ_ACCUM_PREV_HIRES contributes nothing."""
        df_without_accum = pd.concat(
            {
                "pmc_perf": pd.DataFrame({
                    "Kernel_Name": ["k", "k"],
                    "Dispatch_ID": [0, 1],
                    "SQ_WAVES": [100, 200],
                    "SQ_INSTS_VMEM": [10, 20],
                }),
                "SQ_INST_LEVEL_VMEM": pd.DataFrame({
                    "Kernel_Name": ["k", "k"],
                    "Dispatch_ID": [0, 1],
                }),
            },
            axis=1,
        )
        cache = PmcDataCache(df_without_accum)
        assert "SQ_WAVES" in cache
        assert "SQ_INST_LEVEL_VMEM_ACCUM" not in cache
