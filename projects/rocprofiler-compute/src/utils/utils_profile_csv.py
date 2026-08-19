# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Pure stdlib CSV read/write helpers for profile mode.

Profile mode does not import pandas, so counter CSVs are handled here with the
standard library only. Rows are ``list[dict]`` or an iterator of dicts, the
natural representation csv.DictReader/DictWriter use. Counter data is streamed
rather than loaded, since a single pass can hold millions of rows.

Every artifact these helpers touch is gzip, so they open through
``csv_compression`` unconditionally. Plain profile CSVs (``sysinfo.csv``) are
written by their caller with the builtin ``open``.

This module is ONLY used in profile mode. Analyze mode can use pandas freely.
"""

import csv
from collections.abc import Iterator, Sequence
from typing import Optional

from utils import csv_compression


def read_csv_as_dicts(csv_file: str) -> tuple[list[dict], list[str]]:
    """Read a whole CSV file into a list of dicts, plus its fieldnames."""
    try:
        with csv_compression.open_gzip_csv_read(csv_file) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"CSV file {csv_file} has no header row")
            rows = list(reader)
        return rows, list(fieldnames)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    except (csv.Error, UnicodeDecodeError) as e:
        raise ValueError(f"Error reading CSV file {csv_file}: {e}") from e


def iter_csv_dicts(csv_file: str) -> Iterator[dict]:
    """Yield rows from a CSV file without loading it into memory.

    Raises ValueError if the file has no header row.
    """
    with csv_compression.open_gzip_csv_read(csv_file) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_file} has no header row")
        yield from reader


def write_csv_from_dicts(
    csv_file: str, rows: list[dict], fieldnames: Optional[list[str]] = None
) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows and not fieldnames:
        # Nothing to write
        return

    if fieldnames is None:
        if not rows:
            raise ValueError("Cannot write CSV: no rows and no fieldnames provided")
        fieldnames = list(rows[0].keys())

    with csv_compression.open_gzip_csv_write(csv_file) as f:
        # extrasaction='ignore': silently ignore extra keys in rows (not in fieldnames)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)


class GroupIdAssigner:
    """Assign sequential ids to unique combinations of columns, first seen first.

    Ids are handed out one row at a time so a CSV never has to be held in
    memory.

    Example:
        assigner = GroupIdAssigner(["name", "value"], "group_id")
        assigner.apply({"name": "A", "value": 1})  # group_id 0
        assigner.apply({"name": "B", "value": 2})  # group_id 1
        assigner.apply({"name": "A", "value": 1})  # group_id 0 again
    """

    def __init__(self, group_by_columns: Sequence[str], new_column_name: str) -> None:
        self._columns = tuple(group_by_columns)
        self._target = new_column_name
        self._ids: dict[tuple, int] = {}

    def apply(self, row: dict) -> dict:
        """Set the id column on row, in place, and return it."""
        # A row missing one of the columns contributes None for it rather than
        # raising, so a malformed row still gets an id.
        key = tuple(row.get(col) for col in self._columns)
        row[self._target] = self._ids.setdefault(key, len(self._ids))
        return row
