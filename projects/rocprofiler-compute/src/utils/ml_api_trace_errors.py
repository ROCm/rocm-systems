# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import pandas as pd


class MlApiTraceError(Exception):
    """Base class for ML API trace analyze failures."""


class UnaccountedKernelError(MlApiTraceError):
    """A GPU dispatch whose Correlation_ID is not in that pass's marker CSV."""

    def __init__(self, unmatched_rows: pd.DataFrame) -> None:
        self.unmatched_rows = unmatched_rows
        super().__init__(_format_unmatched_kernel_message(unmatched_rows))


def _format_unmatched_kernel_message(unmatched_rows: pd.DataFrame) -> str:
    columns = ["Kernel_Name", "Correlation_ID"]
    if "GUID" in unmatched_rows.columns:
        columns.append("GUID")
    present_columns = [column for column in columns if column in unmatched_rows.columns]
    lines = ["GPU kernel dispatches with no matching ROCTX marker:"]
    for row in unmatched_rows[present_columns].itertuples(index=False):
        parts = [f"{column}={value}" for column, value in zip(present_columns, row)]
        lines.append("  " + ", ".join(parts))
    return "\n".join(lines)
