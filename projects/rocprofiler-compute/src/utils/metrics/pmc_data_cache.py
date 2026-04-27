# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Flat per-call cache exposing each pmc column as a pd.Series."""

from __future__ import annotations

import pandas as pd

from utils import schema


class PmcDataCache:
    """
    Hardware counter data cache.

    Keys are column names;
    values are ``pd.Series``.
    The ``pmc_perf`` top-level contributes every column as is;
    each non-``pmc_perf`` top-level contributes its only
    ``SQ_ACCUM_PREV_HIRES`` column under the key ``{coll_level}_ACCUM``.
    """

    def __init__(self, raw_pmc_df: pd.DataFrame) -> None:
        self._cache: dict[str, pd.Series] = self._flatten(raw_pmc_df)

    def __getitem__(self, col: str) -> pd.Series:
        return self._cache[col]

    def __contains__(self, col: str) -> bool:
        return col in self._cache

    @staticmethod
    def _flatten(raw_pmc_df: pd.DataFrame) -> dict[str, pd.Series]:
        """Flatten the MultiIndex DataFrame into a dict of column-name to Series."""
        cache: dict[str, pd.Series] = {}
        top_levels = raw_pmc_df.columns.get_level_values(0).unique()

        for level in top_levels:
            level_df = raw_pmc_df[level]
            if level == schema.PMC_PERF_FILE_PREFIX:
                for column_name in level_df.columns:
                    cache[column_name] = level_df[column_name]
            elif "SQ_ACCUM_PREV_HIRES" in level_df.columns:
                cache[f"{level}_ACCUM"] = level_df["SQ_ACCUM_PREV_HIRES"]

        return cache
