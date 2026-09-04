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
from contextlib import ExitStack
from typing import Callable, Optional

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

    The first id is ``start``, and each new combination gets the next one.

    Example:
        assigner = GroupIdAssigner(["name", "value"], "group_id")
        assigner.apply({"name": "A", "value": 1})  # group_id 0
        assigner.apply({"name": "B", "value": 2})  # group_id 1
        assigner.apply({"name": "A", "value": 1})  # group_id 0 again

        assigner = GroupIdAssigner(["name"], "group_id", start=1)
        assigner.apply({"name": "A"})  # group_id 1
    """

    def __init__(
        self,
        group_by_columns: Sequence[str],
        new_column_name: str,
        start: int = 0,
    ) -> None:
        self._columns = tuple(group_by_columns)
        self._target = new_column_name
        self._start = start
        self._ids: dict[tuple, int] = {}

    def apply(self, row: dict) -> dict:
        """Set the id column on row, in place, and return it."""
        # A row missing one of the columns contributes None for it rather than
        # raising, so a malformed row still gets an id.
        key = tuple(row.get(col) for col in self._columns)
        row[self._target] = self._ids.setdefault(key, len(self._ids) + self._start)
        return row


def stream_csv_to_file(
    src: str,
    dest: str,
    transform: Optional[Callable[[dict], dict]] = None,
    drop_columns: Sequence[str] = (),
) -> int:
    """Copy src to dest one row at a time, and return the row count.

    Each row passes through transform before being written, so callers can
    relabel columns without materializing the file. Columns in drop_columns are
    removed from the output header and from every row.

    Raises ValueError if src has no header row.
    """
    dropped = set(drop_columns)
    with ExitStack() as stack:
        infile = stack.enter_context(csv_compression.open_gzip_csv_read(src))
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {src} has no header row")

        first_row = next(reader, None)
        if first_row is not None and transform is not None:
            first_row = transform(first_row)
        # Take the header from the transformed row so that columns a transform
        # adds are written too.
        header_source = reader.fieldnames if first_row is None else first_row
        fieldnames = [f for f in header_source if f not in dropped]

        outfile = stack.enter_context(csv_compression.open_gzip_csv_write(dest))
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        if first_row is None:
            return 0

        rows_written = 0
        row: Optional[dict] = first_row
        while row is not None:
            writer.writerow(row)
            rows_written += 1
            row = next(reader, None)
            if row is not None and transform is not None:
                row = transform(row)

    return rows_written
