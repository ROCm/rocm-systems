##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional

import pandas as pd

from .ui_model import MetricMeta, MetricSnapshot


class MetricExtractor:
    """
    Extracts a small set of Metric_ID values from kernel_data produced by
    process_panels_to_dataframes().

    kernel_data shape (from your print):
      panel -> table -> { "df": DataFrame, "tui_style": ... }
    """

    @staticmethod
    def extract(
        kernel_data: dict[str, Any], metric_ids: Iterable[str]
    ) -> MetricSnapshot:
        wanted = set(metric_ids)
        values: dict[str, float] = {}
        meta: dict[str, MetricMeta] = {}

        def pick_value(row: pd.Series) -> Optional[float]:
            # Prefer Value, then Avg, then Pct-of-Peak style columns if present.
            # You can tune this later by metric_id if needed.
            for col in ("Value", "Avg", "Mean", "Pct", "Pct of Peak"):
                if col in row and pd.notna(row[col]):
                    try:
                        return float(row[col])
                    except Exception:
                        continue
            return None

        # Walk panel/table dfs
        for panel_name, tables in (kernel_data or {}).items():
            if not isinstance(tables, dict):
                continue
            for table_name, payload in tables.items():
                if not isinstance(payload, dict) or "df" not in payload:
                    continue
                df = payload.get("df")
                if df is None:
                    continue

                # Metric_ID may be index or column depending on how df was constructed.
                # Your print suggests Metric_ID is the index in some tables.
                if isinstance(df, pd.DataFrame):
                    # Case A: Metric_ID is index
                    if df.index.name == "Metric_ID" or "Metric_ID" not in df.columns:
                        for mid in list(wanted - values.keys()):
                            if mid in df.index:
                                row = df.loc[mid]
                                if isinstance(row, pd.DataFrame):  # defensive
                                    row = row.iloc[0]
                                v = pick_value(row)
                                if v is None:
                                    continue
                                values[mid] = v
                                unit = (
                                    str(row["Unit"])
                                    if "Unit" in row and pd.notna(row["Unit"])
                                    else None
                                )
                                mname = (
                                    str(row["Metric"])
                                    if "Metric" in row and pd.notna(row["Metric"])
                                    else mid
                                )
                                meta[mid] = MetricMeta(
                                    metric_name=mname,
                                    panel=str(panel_name),
                                    table=str(table_name),
                                    unit=unit,
                                )
                    else:
                        # Case B: Metric_ID is column
                        if "Metric_ID" in df.columns:
                            sub = df[df["Metric_ID"].isin(wanted - values.keys())]
                            for _, row in sub.iterrows():
                                mid = str(row["Metric_ID"])
                                v = pick_value(row)
                                if v is None:
                                    continue
                                values[mid] = v
                                unit = (
                                    str(row["Unit"])
                                    if "Unit" in row and pd.notna(row["Unit"])
                                    else None
                                )
                                mname = (
                                    str(row["Metric"])
                                    if "Metric" in row and pd.notna(row["Metric"])
                                    else mid
                                )
                                meta[mid] = MetricMeta(
                                    metric_name=mname,
                                    panel=str(panel_name),
                                    table=str(table_name),
                                    unit=unit,
                                )

        missing = [mid for mid in wanted if mid not in values]
        return MetricSnapshot(values=values, meta=meta, missing=missing)
