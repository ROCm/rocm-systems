# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Pure stdlib CSV operations - Pandas compatibility layer for profile mode.

This module provides pandas-like operations using only Python standard library.
Used in profile mode to eliminate external pandas dependency while maintaining
similar API and functionality.

All functions operate on list[dict] representation of CSV data, which is the
natural Python representation that csv.DictReader/DictWriter use.

This module is ONLY used in profile mode. Analyze mode can use pandas freely.
"""

import csv
from collections.abc import Iterator
from typing import Optional


def read_csv_as_dicts(csv_file: str) -> tuple[list[dict], list[str]]:
    """
    Read CSV file and return list of dicts + fieldnames.

    Equivalent to: df = pd.read_csv(csv_file)
    """
    try:
        with open(csv_file, newline="", encoding="utf-8") as f:
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
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_file} has no header row")
        yield from reader


def write_csv_from_dicts(
    csv_file: str, rows: list[dict], fieldnames: Optional[list[str]] = None
) -> None:
    """
    Write list of dicts to CSV file.

    Equivalent to: df.to_csv(csv_file, index=False)
    """
    if not rows and not fieldnames:
        # Nothing to write
        return

    if fieldnames is None:
        if not rows:
            raise ValueError("Cannot write CSV: no rows and no fieldnames provided")
        fieldnames = list(rows[0].keys())

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        # extrasaction='ignore': silently ignore extra keys in rows (not in fieldnames)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def drop_column_from_rows(rows: list[dict], column_name: str) -> None:
    """
    Remove a column from rows (modifies in place).

    Equivalent to: df = df.drop(columns=[column_name])
    """
    for row in rows:
        row.pop(column_name, None)


def assign_group_ids(
    rows: list[dict], group_by_columns: list[str], new_column_name: str
) -> None:
    """
    Assign sequential group IDs based on unique combinations of columns.

    Equivalent to: df[new_column_name] = df.groupby(group_by_columns).ngroup()

    Note: Empty rows list is valid (no-op). Missing columns use None as value.

    Example:
        rows = [
            {'name': 'A', 'value': 1},
            {'name': 'B', 'value': 2},
            {'name': 'A', 'value': 1},
        ]
        assign_group_ids(rows, ['name', 'value'], 'group_id')
        # rows[0]['group_id'] = 0
        # rows[1]['group_id'] = 1
        # rows[2]['group_id'] = 0  (same as first row)
    """
    groups = {}
    group_id = 0

    for row in rows:
        # Create tuple key from group columns (single hash operation)
        key = tuple(row.get(col) for col in group_by_columns)

        if key not in groups:
            groups[key] = group_id
            group_id += 1

        row[new_column_name] = groups[key]
