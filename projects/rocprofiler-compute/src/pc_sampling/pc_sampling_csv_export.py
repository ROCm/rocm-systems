# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
PC sampling CSV headers.

Builds the CSV headers of the PC-sampling files, with one column per stall
reason. The layout writer names and writes the files.
"""

from collections.abc import Iterable
from typing import List

STALL_COLUMN_PREFIX = "Stall "


def build_header(
    leading_columns: Iterable[str],
    stall_reasons: Iterable[str],
    trailing_columns: Iterable[str],
) -> List[str]:
    """Return the column names of a CSV file, with one column per stall reason.

    A workload's stall reasons vary with how it was sampled, so the columns
    are the ones its samples carry. The prefix keeps a reason such as NONE
    from reading as an unrelated column next to the counts.
    """
    return [
        *leading_columns,
        *(f"{STALL_COLUMN_PREFIX}{stall_reason}" for stall_reason in stall_reasons),
        *trailing_columns,
    ]
