# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Flat per-call cache exposing each pmc column as a pd.Series."""

from __future__ import annotations

import pandas as pd


class PmcDataCache:
    """
    Hardware counter data cache.

    Wraps a flat single-index PMC DataFrame so each column is exposed as a
    ``pd.Series`` via mapping-style access. The upstream
    ``_create_single_df_pmc`` already produces the canonical flat layout
    (with ``<bucket>_ACCUM`` columns) so this class is a thin lookup-free
    accessor on top of it.
    """

    def __init__(self, raw_pmc_df: pd.DataFrame) -> None:
        self._cache: dict[str, pd.Series] = self._flatten(raw_pmc_df)

    def __getitem__(self, col: str) -> pd.Series:
        return self._cache[col]

    def __contains__(self, col: str) -> bool:
        return col in self._cache

    @staticmethod
    def _flatten(raw_pmc_df: pd.DataFrame) -> dict[str, pd.Series]:
        """Cache each column of the flat single-index PMC DataFrame as a Series."""
        return {col: raw_pmc_df[col] for col in raw_pmc_df.columns}
