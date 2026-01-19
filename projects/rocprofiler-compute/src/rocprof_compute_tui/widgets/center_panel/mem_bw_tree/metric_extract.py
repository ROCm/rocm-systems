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

import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from .ui_model import MetricMeta, MetricSnapshot

_PREFIX_RE = re.compile(r"^\d+\.\d+$")
_RANGE_RE = re.compile(r"^(\d+\.\d+)\.(\d+):\1\.(\d+)$")


class MetricExtractor:
    """
    Extracts metric values into a column-aware snapshot.

    Snapshot values:
        values[metric_id][column_name] = float
    """

    @staticmethod
    def extract(
        kernel_data: dict[str, Any], metric_ids: Iterable[str]
    ) -> MetricSnapshot:
        selectors = set(metric_ids)
        values: dict[str, dict[str, float]] = {}
        meta: dict[str, MetricMeta] = {}
        resolved_ids: set[str] = set()

        for panel_name, tables in (kernel_data or {}).items():
            if not isinstance(tables, dict):
                continue

            for table_name, payload in tables.items():
                df = payload.get("df") if isinstance(payload, dict) else None
                if not isinstance(df, pd.DataFrame):
                    continue

                if df.index.name == "Metric_ID":
                    mids = {str(mid) for mid in df.index}
                    get_row = lambda mid: df.loc[mid]  # noqa E731
                elif "Metric_ID" in df.columns:
                    mids = {str(mid) for mid in df["Metric_ID"]}
                    get_row = lambda mid: df[df["Metric_ID"] == mid].iloc[0]  # noqa E731
                else:
                    continue

                table_resolved: set[str] = set()

                for sel in selectors:
                    if sel in mids:
                        table_resolved.add(sel)
                        continue

                    if _PREFIX_RE.fullmatch(sel):
                        prefix = f"{sel}."
                        table_resolved.update(
                            mid for mid in mids if mid.startswith(prefix)
                        )
                        continue

                    m = _RANGE_RE.fullmatch(sel)
                    if m:
                        prefix, a, b = m.group(1), int(m.group(2)), int(m.group(3))
                        lo, hi = min(a, b), max(a, b)
                        for mid in mids:
                            if mid.startswith(prefix + "."):
                                try:
                                    idx = int(mid.split(".")[2])
                                except Exception:
                                    continue
                                if lo <= idx <= hi:
                                    table_resolved.add(mid)

                for mid in table_resolved:
                    try:
                        row = get_row(mid)
                        if isinstance(row, pd.DataFrame):
                            row = row.iloc[0]
                    except Exception:
                        continue

                    colvals: dict[str, float] = {}
                    for col, v in row.items():
                        if col in ("Metric_ID", "Metric", "Unit"):
                            continue
                        if pd.notna(v):
                            try:
                                colvals[col] = float(v)
                            except Exception:
                                continue

                    if not colvals:
                        continue

                    values[mid] = colvals
                    resolved_ids.add(mid)

                    meta[mid] = MetricMeta(
                        metric_name=str(row.get("Metric", mid)),
                        panel=str(panel_name),
                        table=str(table_name),
                        unit=str(row["Unit"])
                        if "Unit" in row and pd.notna(row["Unit"])
                        else None,
                    )

        missing = [mid for mid in resolved_ids if mid not in values]

        return MetricSnapshot(
            values=values,
            meta=meta,
            missing=missing,
        )
