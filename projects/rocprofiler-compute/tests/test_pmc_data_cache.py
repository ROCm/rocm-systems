# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.metrics.pmc_data_cache.PmcDataCache."""

import pandas as pd
import pytest

from utils.metrics.pmc_data_cache import PmcDataCache


class TestPmcDataCache:
    """Tests for utils.metrics.pmc_data_cache.PmcDataCache."""

    def _make_flat_pmc_df(self):
        """Build a flat single-index PMC DataFrame with two counter columns."""
        return pd.DataFrame({
            "SQ_WAVES": [100, 200, 150],
            "GRBM_GUI_ACTIVE": [1000, 2000, 1500],
        })

    def test_columns_exposed_as_series(self):
        """Each column of the input DataFrame is exposed as a `pd.Series`."""
        cache = PmcDataCache(self._make_flat_pmc_df())
        assert "SQ_WAVES" in cache
        assert "GRBM_GUI_ACTIVE" in cache

        sq_waves = cache["SQ_WAVES"]
        assert isinstance(sq_waves, pd.Series)
        assert sq_waves.tolist() == [100, 200, 150]

        grbm = cache["GRBM_GUI_ACTIVE"]
        assert isinstance(grbm, pd.Series)
        assert grbm.tolist() == [1000, 2000, 1500]

    def test_missing_key_raises_keyerror(self):
        """Direct subscript on a missing column raises KeyError."""
        cache = PmcDataCache(self._make_flat_pmc_df())
        with pytest.raises(KeyError):
            cache["UNKNOWN_COUNTER"]

    def test_contains_reports_false_for_missing_key(self):
        """`in` returns False for unknown column names."""
        cache = PmcDataCache(self._make_flat_pmc_df())
        assert ("UNKNOWN_COUNTER" in cache) is False
