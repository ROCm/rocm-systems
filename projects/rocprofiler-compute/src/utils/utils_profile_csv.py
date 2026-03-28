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
from collections import defaultdict
from typing import Any, Optional


def read_csv_as_dicts(csv_file: str) -> tuple[list[dict], list[str]]:
    """
    Read CSV file and return list of dicts + fieldnames.

    Equivalent to: df = pd.read_csv(csv_file)
    """
    try:
        with open(csv_file, newline="") as f:
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

    with open(csv_file, "w", newline="") as f:
        # extrasaction='ignore': silently ignore extra keys in rows (not in fieldnames)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def concat_csv_files(
    input_files: list[str], output_file: Optional[str] = None
) -> list[dict]:
    """
    Concatenate multiple CSV files into one.

    Equivalent to: pd.concat([pd.read_csv(f) for f in input_files], ignore_index=True)
    """
    if not input_files:
        return []

    combined_rows = []
    fieldnames = None

    for csv_file in input_files:
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            combined_rows.extend(reader)

    if output_file and combined_rows:
        write_csv_from_dicts(output_file, combined_rows, fieldnames)

    return combined_rows


def add_column_to_rows(rows: list[dict], column_name: str, values: list[Any]) -> None:
    """
    Add a new column to rows (modifies in place).

    Equivalent to: df[column_name] = values
    """
    if len(values) != len(rows):
        raise ValueError(f"Length mismatch: {len(rows)} rows but {len(values)} values")

    for row, value in zip(rows, values):
        row[column_name] = value


def drop_column_from_rows(rows: list[dict], column_name: str) -> None:
    """
    Remove a column from rows (modifies in place).

    Equivalent to: df = df.drop(columns=[column_name])
    """
    for row in rows:
        row.pop(column_name, None)


def rename_columns(rows: list[dict], column_mapping: dict[str, str]) -> None:
    """
    Rename columns in rows (modifies in place).

    Equivalent to: df.rename(columns=column_mapping, inplace=True)
    """
    for row in rows:
        for old_name, new_name in column_mapping.items():
            if old_name in row:
                row[new_name] = row.pop(old_name)


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


def groupby_aggregate(
    rows: list[dict], group_by_columns: list[str], agg_dict: dict[str, str]
) -> list[dict]:
    """
    Group rows by columns and aggregate using specified functions.

    Equivalent to: df.groupby(group_by_columns).agg(agg_dict)

    Example:
        rows = [
            {'category': 'A', 'value': 10},
            {'category': 'A', 'value': 20},
            {'category': 'B', 'value': 15},
        ]
        result = groupby_aggregate(rows, ['category'], {'value': 'sum'})
        # Returns: [{'category': 'A', 'value': 30}, {'category': 'B', 'value': 15}]
    """
    # Group rows by key (O(n))
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(col) for col in group_by_columns)
        groups[key].append(row)

    # Aggregate each group
    result = []
    for key, group_rows in groups.items():
        aggregated_row = {}

        # Add group-by columns
        for i, col in enumerate(group_by_columns):
            aggregated_row[col] = key[i]

        # Apply aggregation functions
        for col, agg_func in agg_dict.items():
            if col in group_by_columns:
                continue

            # Extract values for this column (skip None)
            values = [row.get(col) for row in group_rows if row.get(col) is not None]

            if agg_func == "first":
                aggregated_row[col] = group_rows[0].get(col) if group_rows else None
            elif agg_func == "last":
                aggregated_row[col] = group_rows[-1].get(col) if group_rows else None
            elif agg_func == "sum":
                try:
                    aggregated_row[col] = sum(float(v) for v in values) if values else 0
                except (ValueError, TypeError):
                    # Non-numeric data - coerce to 0 (matches pandas coerce behavior)
                    aggregated_row[col] = 0
            elif agg_func == "count":
                aggregated_row[col] = len(values)
            elif agg_func == "mean":
                try:
                    aggregated_row[col] = (
                        sum(float(v) for v in values) / len(values) if values else None
                    )
                except (ValueError, TypeError):
                    # Non-numeric data - return None
                    aggregated_row[col] = None
            elif agg_func == "min":
                aggregated_row[col] = min(values) if values else None
            elif agg_func == "max":
                aggregated_row[col] = max(values) if values else None
            else:
                raise ValueError(f"Unsupported aggregation function: {agg_func}")

        result.append(aggregated_row)

    return result


def pivot_table(
    rows: list[dict],
    index_columns: list[str],
    pivot_column: str,
    value_column: str,
) -> list[dict]:
    """
    Pivot rows to convert unique values in pivot_column into columns.

    When there are duplicate (index + pivot) combinations, the mean of all
    values is computed. Missing values are automatically filled with None.

    Equivalent to: df.pivot_table(index=index_columns, columns=pivot_column,
                                   values=value_column, aggfunc='mean',
                                   fill_value=None).reset_index()

    Example:
        rows = [
            {'id': 1, 'counter': 'A', 'value': 10},
            {'id': 1, 'counter': 'A', 'value': 20},  # duplicate - will be averaged
            {'id': 1, 'counter': 'B', 'value': 30},
            {'id': 2, 'counter': 'A', 'value': 40},
        ]
        result = pivot_table(rows, ['id'], 'counter', 'value')
        # Returns:
        # [
        #   {'id': 1, 'A': 15.0, 'B': 30.0},  # A is mean of [10, 20]
        #   {'id': 2, 'A': 40.0, 'B': None}
        # ]
    """
    # Build pivot structure: {index_key: {col: [values]}}
    # Accumulate all values for duplicate (index + pivot) combinations
    pivot_data = defaultdict(lambda: defaultdict(list))

    for row in rows:
        # Create index key
        index_key = tuple(row.get(col) for col in index_columns)

        # Accumulate values for each pivot column
        new_col_name = row[pivot_column]
        pivot_data[index_key][new_col_name].append(row[value_column])

    # Convert to list and compute means
    result = []
    for index_key, pivot_cols in pivot_data.items():
        # Reconstruct index columns from key
        row_dict = {col: val for col, val in zip(index_columns, index_key)}

        # Add pivot columns with mean of accumulated values
        for col_name, values in pivot_cols.items():
            row_dict[col_name] = sum(float(v) for v in values) / len(values)

        result.append(row_dict)

    # Fill missing values with None
    # Collect all pivot column names
    all_pivot_cols = set()
    for row in result:
        all_pivot_cols.update(k for k in row.keys() if k not in index_columns)

    # Fill missing values
    for row in result:
        for col in all_pivot_cols:
            if col not in row:
                row[col] = None

    return result


def merge_rows(
    left_rows: list[dict],
    right_rows: list[dict],
    left_on: str,
    right_on: str,
    how: str = "inner",
) -> list[dict]:
    """
    Merge two lists of rows based on key columns.

    Equivalent to: pd.merge(left_df, right_df, left_on=left_on,
                            right_on=right_on, how=how)

    """
    if how not in ("inner", "left", "right", "outer"):
        raise ValueError(f"Unsupported join type: {how}")

    # Build lookup from right side (O(m))
    right_lookup = {}
    for row in right_rows:
        key = row.get(right_on)
        # Note: None keys are treated as valid join keys (matches pandas behavior)
        if key not in right_lookup:
            right_lookup[key] = []
        right_lookup[key].append(row)

    merged = []
    matched_left_keys = set()

    # Merge left side (O(n) with O(1) lookups)
    for left_row in left_rows:
        key = left_row.get(left_on)
        matched_left_keys.add(key)

        if key in right_lookup:
            # Match found - create merged rows
            for right_row in right_lookup[key]:
                # Right values overwrite left values for same column names
                merged_row = {**left_row, **right_row}
                merged.append(merged_row)
        elif how in ("left", "outer"):
            # No match but left join - include left row
            merged.append(left_row.copy())

    # Handle right/outer join - add unmatched right rows
    if how in ("right", "outer"):
        for key, right_row_list in right_lookup.items():
            if key not in matched_left_keys:
                for right_row in right_row_list:
                    merged.append(right_row.copy())

    return merged
