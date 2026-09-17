#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

import argparse
import math
import sqlite3
import sys
from argparse import ArgumentParser
from typing import Optional, List, Tuple, Union, Any

from .importer import RocpdImportData, execute_statement

__all__ = ["TimeWindowError", "apply_time_window", "execute", "add_args", "main"]


class TimeWindowError(ValueError):
    """Invalid time window specification."""


def get_marker_timestamp(
    connection: sqlite3.Connection, marker_name: str, marker_type: str = "start"
) -> float:
    """Get the start or end timestamp of a named marker."""
    if marker_type not in ("start", "end"):
        raise ValueError(f"marker_type must be 'start' or 'end', not '{marker_type}'")

    query = f"""
        SELECT {quote_identifier(marker_type)}
        FROM regions_and_samples
        WHERE category LIKE 'MARKER_%'
          AND JSON_EXTRACT(extdata, '$.message') = ?
    """
    try:
        result = connection.execute(query, (marker_name,)).fetchall()
    except sqlite3.OperationalError as error:
        if str(error).lower() != "no such table: regions_and_samples":
            raise
        raise TimeWindowError(
            "ERROR: Cannot create time window from markers - trace file "
            "contains no marker data"
        ) from None

    if not result:
        raise TimeWindowError(
            f'ERROR: {marker_type.capitalize()} marker "{marker_name}" not found'
        )
    if len(result) > 1:
        raise TimeWindowError(
            f'ERROR: Ambiguous reference - multiple {marker_type} markers found with name "{marker_name}"'
        )

    return float(result[0][0])


def quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def get_column_names(connection: RocpdImportData, table_name: str):
    """Return column names for a table or view without reading any rows."""
    cursor = connection.execute(f"SELECT * FROM {quote_identifier(table_name)} LIMIT 0")
    return [desc[0] for desc in cursor.description]


def get_table_names(connection) -> List[str]:
    """Return physical ROCpd tables, excluding derived SQLite views."""
    if hasattr(connection, "table_info"):
        return list(connection.table_info.keys())

    query = """
        SELECT name FROM sqlite_temp_master WHERE type = 'table'
        UNION
        SELECT name FROM sqlite_master WHERE type = 'table'
    """
    return [row[0] for row in connection.execute(query)]


def get_timed_tables(connection):
    """Group physical event tables by their time-column representation.

    Info tables describe object lifetimes, while derived views duplicate their
    source records. Neither should influence the recorded-event time range.
    """
    start_end_tables = []
    timestamp_tables = []

    for table_name in get_table_names(connection):
        if table_name.startswith(("rocpd_info_", "sqlite_")):
            continue

        column_names = get_column_names(connection, table_name)
        if "start" in column_names and "end" in column_names:
            start_end_tables.append(table_name)
        elif "timestamp" in column_names:
            timestamp_tables.append(table_name)

    return start_end_tables, timestamp_tables


def get_min_max_time(connection, timed_tables=None):
    if timed_tables is None:
        timed_tables = get_timed_tables(connection)

    start_end_tables, timestamp_tables = timed_tables
    subqueries = [
        f"SELECT MIN(start) AS min_time, MAX(end) AS max_time "
        f"FROM {quote_identifier(table_name)}"
        for table_name in start_end_tables
    ]
    subqueries.extend(
        f"SELECT MIN(timestamp) AS min_time, MAX(timestamp) AS max_time "
        f"FROM {quote_identifier(table_name)}"
        for table_name in timestamp_tables
    )

    if not subqueries:
        return (None, None)

    union_all = " UNION ALL ".join(subqueries)
    min_max_query = f"""
        SELECT
            MIN(min_time) as min_time,
            MAX(max_time) as max_time
        FROM ({union_all})"""

    min_time, max_time = execute_statement(connection, min_max_query).fetchone()
    return (min_time, max_time)


def _format_timestamp(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _convert_time(
    time_str: Optional[Union[str, int, float]],
    is_start: bool,
    min_time: float,
    max_time: float,
) -> float:
    """Convert a user-supplied time specification (number or percentage) to an
    absolute nanosecond timestamp."""
    if time_str is None or time_str == "":
        return min_time if is_start else max_time

    label = "--start" if is_start else "--end"
    value_str = str(time_str).strip()

    if "%" in value_str:
        if value_str.count("%") != 1 or not value_str.endswith("%"):
            raise TimeWindowError(
                f"ERROR: Invalid {label} percentage '{time_str}' - expected a "
                "number followed by one '%' character"
            )
        try:
            percentage = float(value_str[:-1]) / 100.0
        except ValueError:
            raise TimeWindowError(
                f"ERROR: Invalid {label} percentage '{time_str}' - expected a "
                "number followed by one '%' character"
            ) from None
        if not math.isfinite(percentage):
            raise TimeWindowError(
                f"ERROR: Invalid {label} percentage '{time_str}' - must be finite"
            )
        return min_time + ((max_time - min_time) * percentage)

    try:
        value = float(value_str)
    except ValueError:
        raise TimeWindowError(
            f"ERROR: Invalid {label} value '{time_str}' - must be a percentage "
            f"(e.g., '50%') or a number (nanoseconds since epoch)"
        ) from None

    if not math.isfinite(value):
        raise TimeWindowError(
            f"ERROR: Invalid {label} value '{time_str}' - must be a finite number"
        )

    return value


def _validate_and_clamp_window(
    start_time: float,
    end_time: float,
    min_time: float,
    max_time: float,
    start_label: str,
    end_label: str,
) -> Tuple[float, float]:
    if end_time < start_time:
        raise TimeWindowError(
            f"ERROR: Invalid time range - {end_label} ({_format_timestamp(end_time)}) "
            f"must be greater than or equal to {start_label} "
            f"({_format_timestamp(start_time)})"
        )

    if end_time < min_time or start_time > max_time:
        raise TimeWindowError(
            "ERROR: Requested time window "
            f"[{_format_timestamp(start_time)}, {_format_timestamp(end_time)}] nsec "
            "does not overlap trace time range "
            f"[{_format_timestamp(min_time)}, {_format_timestamp(max_time)}] nsec"
        )

    clamped_start = max(start_time, min_time)
    clamped_end = min(end_time, max_time)
    adjustments = []
    if clamped_start != start_time:
        adjustments.append(
            f"{start_label} value {_format_timestamp(start_time)} is before trace "
            f"minimum {_format_timestamp(min_time)}"
        )
    if clamped_end != end_time:
        adjustments.append(
            f"{end_label} value {_format_timestamp(end_time)} is after trace "
            f"maximum {_format_timestamp(max_time)}"
        )

    if adjustments:
        print(
            f"# WARNING: {'; '.join(adjustments)}; using time window "
            f"[{_format_timestamp(clamped_start)}, "
            f"{_format_timestamp(clamped_end)}] nsec instead",
            file=sys.stderr,
        )

    return clamped_start, clamped_end


def percentages2timestamp(
    connection: sqlite3.Connection,
    start_time: Optional[Union[str, int, float]],
    end_time: Optional[Union[str, int, float]],
) -> Tuple[float, float]:
    """Convert percentage strings or time values to timestamps."""

    min_time, max_time = get_min_max_time(connection)

    if min_time is None:
        raise TimeWindowError(
            "ERROR: Cannot create time window - trace file contains no timing data"
        )

    converted_start = _convert_time(start_time, True, min_time, max_time)
    converted_end = _convert_time(end_time, False, min_time, max_time)
    return _validate_and_clamp_window(
        converted_start,
        converted_end,
        min_time,
        max_time,
        "--start",
        "--end",
    )


def get_time_filter(inclusive: bool, start_time, end_time) -> str:
    """Create SQL filter for start/end time ranges."""
    _beg = int(start_time)
    _end = int(end_time)
    if inclusive:
        return f"start >= {_beg} AND end <= {_end}"
    else:
        return f"start <= {_end} AND end >= {_beg}"


def get_timestamp_filter(inclusive: bool, start_time, end_time) -> str:
    """Create SQL filter for timestamp columns."""
    _beg = int(start_time)
    _end = int(end_time)
    if inclusive:
        return f"timestamp >= {_beg} AND timestamp <= {_end}"
    else:
        return f"timestamp <= {_end} AND timestamp >= {_beg}"


def create_view(connection: sqlite3.Connection, view_name: str, query: str) -> None:
    """Create or replace a database view."""
    execute_statement(connection, f"DROP VIEW IF EXISTS {view_name}")
    # print(f"{query}")
    execute_statement(connection, query)
    connection.commit()


#
# Main processing functions
#
def apply_time_window(connection: RocpdImportData, **kwargs: Any) -> None:
    """Apply time window filtering to create filtered views."""

    window_arg_names = ("start", "end", "start_marker", "end_marker")
    if not any(kwargs.get(name) is not None for name in window_arg_names):
        return connection

    if kwargs.get("start") is not None and kwargs.get("start_marker") is not None:
        raise TimeWindowError("ERROR: Cannot specify both --start and --start-marker")
    if kwargs.get("end") is not None and kwargs.get("end_marker") is not None:
        raise TimeWindowError("ERROR: Cannot specify both --end and --end-marker")

    inclusive = kwargs.get("inclusive", True)
    timed_tables = get_timed_tables(connection)

    def dump_min_max(label, bounds):
        bounds_min, bounds_max = bounds
        if bounds_min is None or bounds_max is None:
            print(f"# {label:>8} time bounds: <no timed events>")
            return None
        delta = bounds_max - bounds_min
        print(
            f"# {label:>8} time bounds: {bounds_min} : {bounds_max} nsec (delta={delta} nsec)"
        )
        return delta

    initial_bounds = get_min_max_time(connection, timed_tables)
    orig_delta = dump_min_max("Initial", initial_bounds)
    min_time, max_time = initial_bounds
    if min_time is None or max_time is None:
        raise TimeWindowError(
            "ERROR: Cannot create time window - trace file contains no timing data"
        )

    start_marker = kwargs.get("start_marker")
    end_marker = kwargs.get("end_marker")
    if start_marker is not None:
        start_time = get_marker_timestamp(connection, start_marker, "start")
        start_label = "--start-marker"
    else:
        start_time = _convert_time(kwargs.get("start"), True, min_time, max_time)
        start_label = "--start"

    if end_marker is not None:
        end_time = get_marker_timestamp(connection, end_marker, "end")
        end_label = "--end-marker"
    else:
        end_time = _convert_time(kwargs.get("end"), False, min_time, max_time)
        end_label = "--end"

    start_time, end_time = _validate_and_clamp_window(
        start_time,
        end_time,
        min_time,
        max_time,
        start_label,
        end_label,
    )

    start_end_timed_tables, timestamp_timed_tables = timed_tables

    for table_name in start_end_timed_tables:
        dbs = [
            f"{itr} WHERE {get_time_filter(inclusive, start_time, end_time)}"
            for itr in connection.table_info[table_name]
        ]
        table_union = " UNION ALL ".join(dbs)
        create_view_query = f"""
            CREATE TEMPORARY VIEW {quote_identifier(table_name)} AS
                {table_union}
        """
        create_view(connection, quote_identifier(table_name), create_view_query)

    for table_name in timestamp_timed_tables:
        dbs = [
            f"{itr} WHERE {get_timestamp_filter(inclusive, start_time, end_time)}"
            for itr in connection.table_info[table_name]
        ]
        table_union = " UNION ALL ".join(dbs)
        create_view_query = f"""
            CREATE TEMPORARY VIEW {quote_identifier(table_name)} AS
                {table_union}
        """
        create_view(connection, quote_identifier(table_name), create_view_query)

    updated_bounds = get_min_max_time(connection, timed_tables)
    upd_delta = dump_min_max("Windowed", updated_bounds)

    if upd_delta is None:
        print(
            f"# WARNING: time window [{_format_timestamp(start_time)}, "
            f"{_format_timestamp(end_time)}] nsec contains no timed events",
            file=sys.stderr,
        )
    elif orig_delta is not None and orig_delta > 0:
        reduction = (1.0 - (upd_delta / orig_delta)) * 100.0
        print(f"# Time windowing reduced the duration by {reduction:6.2f}%")

    return connection


#
# Command-line interface functions
#
def add_args(parser: ArgumentParser):
    """Add time slice arguments to an existing parser."""

    tw_options = parser.add_argument_group("Time window options")

    # Start time mutually exclusive group
    start_group = tw_options.add_mutually_exclusive_group(required=False)
    start_group.add_argument(
        "--start",
        type=str,
        help="Start time as percentage or in nanoseconds from trace file (e.g., '50%%' or '781470909013049')",
        default=None,
    )
    start_group.add_argument(
        "--start-marker",
        type=str,
        help="Named marker event to use as window start point",
        default=None,
    )

    # End time mutually exclusive group
    end_group = tw_options.add_mutually_exclusive_group(required=False)
    end_group.add_argument(
        "--end",
        type=str,
        help="End time in as percentage or nanoseconds from trace file (e.g., '75%%' or '3543724246381057')",
        default=None,
    )
    end_group.add_argument(
        "--end-marker",
        type=str,
        help="Named marker event to use as window end point",
        default=None,
    )

    tw_options.add_argument(
        "--inclusive",
        type=lambda x: x.lower() in ("true", "t", "yes", "1"),
        help="True: include events if START or END in window; False: only if BOTH in window (default: True)",
        default=True,
    )

    def process_args(input, args):
        valid_args = ["start", "end", "inclusive", "start_marker", "end_marker"]
        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is not None:
                    ret[itr] = val

        if ret and input is not None:
            apply_time_window(input, **ret)

        return ret

    return process_args


def process_args_or_exit(process_args, input, args):
    """Run an argument processor and report time-window errors without a traceback."""
    try:
        return process_args(input, args)
    except TimeWindowError as error:
        raise SystemExit(str(error)) from None


def execute(input_rpd: str, **kwargs: Any) -> RocpdImportData:
    """Execute time window filtering on database file."""

    importData = RocpdImportData(input_rpd)

    apply_time_window(importData, **kwargs)

    return importData


def main(argv=None) -> int:
    """Main entry point for command line execution."""
    parser = argparse.ArgumentParser(
        description="Apply time window filtering to ROCpd database views"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input ROCpd database file",
    )

    process_time_window_args = add_args(parser)
    args = parser.parse_args(argv)

    input_data = RocpdImportData(args.input)
    process_args_or_exit(process_time_window_args, input_data, args)


if __name__ == "__main__":
    main()
