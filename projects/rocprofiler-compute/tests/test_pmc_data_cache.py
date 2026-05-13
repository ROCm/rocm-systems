# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.pmc_data_cache.PmcDataCache."""

import pandas as pd
import pytest

from utils.metrics.pmc_data_cache import PmcDataCache


class TestPmcDataCache:
    """Tests for utils.metrics.pmc_data_cache.PmcDataCache."""

    def _make_pmc_perf_only_df(self):
        """Build a flat DataFrame matching `_create_single_df_pmc` output."""
        return pd.DataFrame({
            "SQ_WAVES": [100, 200, 150],
            "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
        })

    def _make_pmc_perf_with_accum_alias_df(self):
        """Build a flat DataFrame matching `_create_single_df_pmc` output with accumulator alias."""
        return pd.DataFrame({
            "Kernel_Name": ["k", "k"],
            "Dispatch_ID": [0, 1],
            "SQ_WAVES": [100, 200],
            "SQ_INSTS_VMEM": [10, 20],
            "SQ_INST_LEVEL_VMEM_ACCUM": [50, 70],
        })

    def test_pmc_perf_columns_lift_to_flat_keys(self):
        """Every column of the flat DataFrame is exposed as a pd.Series."""
        cache = PmcDataCache(self._make_pmc_perf_only_df())
        assert "SQ_WAVES" in cache
        assert "GRBM_GUI_ACTIVE" in cache

        sq_waves = cache["SQ_WAVES"]
        assert isinstance(sq_waves, pd.Series)
        assert sq_waves.tolist() == [100, 200, 150]

        grbm = cache["GRBM_GUI_ACTIVE"]
        assert isinstance(grbm, pd.Series)
        assert grbm.tolist() == [1000, 2000, 1500]

    def test_accum_alias_column_is_passed_through(self):
        """Pre-renamed `<bucket>_ACCUM` columns are exposed under their flat name."""
        cache = PmcDataCache(self._make_pmc_perf_with_accum_alias_df())

        assert "SQ_INST_LEVEL_VMEM_ACCUM" in cache
        accum_series = cache["SQ_INST_LEVEL_VMEM_ACCUM"]
        assert isinstance(accum_series, pd.Series)
        assert accum_series.tolist() == [50, 70]

    def test_missing_key_raises_keyerror(self):
        """Direct subscript on a missing column raises KeyError."""
        cache = PmcDataCache(self._make_pmc_perf_only_df())
        with pytest.raises(KeyError):
            cache["UNKNOWN_COUNTER"]

    def test_contains_reports_false_for_missing_key(self):
        """`in` returns False for unknown column names."""
        cache = PmcDataCache(self._make_pmc_perf_only_df())
        assert ("UNKNOWN_COUNTER" in cache) is False

    def test_dataframe_without_accum_alias_does_not_synthesize_one(self):
        """A flat DataFrame missing any `_ACCUM` alias doesn't gain one in the cache."""
        df_without_accum = pd.DataFrame({
            "Kernel_Name": ["k", "k"],
            "Dispatch_ID": [0, 1],
            "SQ_WAVES": [100, 200],
            "SQ_INSTS_VMEM": [10, 20],
        })
        cache = PmcDataCache(df_without_accum)
        assert "SQ_WAVES" in cache
        assert "SQ_INST_LEVEL_VMEM_ACCUM" not in cache
