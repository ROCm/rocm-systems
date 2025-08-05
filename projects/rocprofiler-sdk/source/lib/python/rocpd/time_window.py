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
import sqlite3
from argparse import ArgumentParser
from typing import Optional, Tuple, Dict, Any, List

from .importer import RocpdImportData, execute_statement


def get_marker_timestamp(
    connection: sqlite3.Connection, marker_name: str, marker_type: str = "start"
) -> float:
    """Get timestamp for a specific marker."""
    query = "SELECT start FROM markers WHERE name = ?"
    result = connection.execute(query, (marker_name,)).fetchall()

    if not result:
        raise ValueError(
            f'ERROR: {marker_type.capitalize()} marker "{marker_name}" not found'
        )
    if len(result) > 1:
        raise ValueError(
            f'ERROR: Ambiguous reference - multiple {marker_type} markers found with name "{marker_name}"'
        )

    return float(result[0][0])


def markers2timestamp(
    connection: sqlite3.Connection, start_marker: str, end_marker: str
) -> Tuple[float, float]:
    """Convert marker names to timestamp values."""
    start_time = get_marker_timestamp(connection, start_marker, "start")
    end_time = get_marker_timestamp(connection, end_marker, "end")
    return (start_time, end_time)


def get_min_max_time(connection):
    min_max_query = """
        SELECT
            MIN(min_time) as min_time,
            MAX(max_time) as max_time
        FROM (
            SELECT start as min_time, end as max_time FROM regions_and_samples
            UNION ALL
            SELECT start as min_time, end as max_time FROM rocpd_kernel_dispatch
            UNION ALL
            SELECT start as min_time, end as max_time FROM rocpd_memory_allocate
            UNION ALL
            SELECT start as min_time, end as max_time FROM rocpd_memory_copy
        )"""

    min_time, max_time = execute_statement(connection, min_max_query).fetchone()
    return (min_time, max_time)


def percentages2timestamp(
    connection: sqlite3.Connection, start_time: Optional[str], end_time: Optional[str]
) -> Tuple[float, float]:
    """Convert percentage strings or time values to timestamps."""

    min_time, max_time = get_min_max_time(connection)

    if min_time is None:
        raise ValueError(
            "ERROR: Cannot create time window - trace file contains no timing data"
        )

    def convert_time(time_str: Optional[str], is_start: bool = False) -> float:
        if not time_str:
            return min_time if is_start else max_time

        if "%" in time_str:
            percentage = float(time_str.replace("%", "")) / 100.0
            if not 0 <= percentage <= 1:
                raise ValueError(
                    f"ERROR: Invalid percentage '{time_str}' - must be between '0%' and '100%'"
                )
            return min_time + ((max_time - min_time) * percentage)

        try:
            return float(time_str)
        except ValueError:
            raise ValueError(
                f"ERROR: Invalid time value '{time_str}' - must be percentage (e.g., '50%') or a number (nanoseconds since epoch) "
            )

    return (convert_time(start_time, True), convert_time(end_time, False))


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
def is_using_markers(args: Dict[str, Any]) -> bool:
    """Check if filtering mode uses markers or time ranges."""
    # Add improved null checks
    if args.get("start") is not None or args.get("end") is not None:
        return False
    elif args.get("start_marker") is not None or args.get("end_marker") is not None:
        return True

    return None


def get_column_names(conn: RocpdImportData, table_name: str):
    """
    Use SELECT on zero rows and read cursor.description.
    """
    cursor = conn.execute(f"SELECT * FROM '{table_name}' LIMIT 0")
    return [desc[0] for desc in cursor.description]


def apply_time_window(connection: RocpdImportData, **kwargs: Any) -> None:
    """Apply time window filtering to create filtered views."""

    is_marker_mode = is_using_markers(kwargs)
    if is_marker_mode is None:
        return connection

    inclusive = kwargs.get("inclusive", True)

    def dump_min_max(label):
        bounds_min, bounds_max = get_min_max_time(connection)
        # bounds_min /= 1.0e9
        # bounds_max /= 1.0e9
        delta = bounds_max - bounds_min
        print(
            f"# {label:>8} time bounds: {bounds_min} : {bounds_max} nsec (delta={delta} nsec)"
        )
        return delta

    orig_delta = dump_min_max("Initial")

    # Get start and end times
    if not is_marker_mode:
        start_time = kwargs.get("start", None)
        end_time = kwargs.get("end", None)
        start_time, end_time = percentages2timestamp(connection, start_time, end_time)
    else:
        start_marker = kwargs.get("start_marker", None)
        end_marker = kwargs.get("end_marker", None)
        start_time, end_time = markers2timestamp(connection, start_marker, end_marker)

    if not end_time > start_time:
        raise ValueError(
            f"ERROR: Invalid time range - end time ({end_time}) must be greater than start time ({start_time})"
        )

    # Create views for tables with start and end times
    start_end_timed_tables = []
    timestamp_timed_tables = []

    for itr in connection.table_info.keys():
        if itr.find("rocpd_info_") == 0:
            continue
        column_names = get_column_names(connection, itr)
        if "start" in column_names and "end" in column_names:
            start_end_timed_tables += [itr]
        elif "timestamp" in column_names:
            timestamp_timed_tables += [itr]

    # Restrict the scope of the tables with start/end columns
    for table_name in start_end_timed_tables:
        dbs = [
            f"{itr} WHERE {get_time_filter(inclusive, start_time, end_time)}"
            for itr in connection.table_info[table_name]
        ]
        table_union = " UNION ALL ".join(dbs)
        create_view_query = f"""
            CREATE TEMPORARY VIEW {table_name} AS
                {table_union}
        """
        create_view(connection, table_name, create_view_query)

    # Restrict the scope of the tables with timestamp columns
    for table_name in timestamp_timed_tables:
        dbs = [
            f"{itr} WHERE {get_timestamp_filter(inclusive, start_time, end_time)}"
            for itr in connection.table_info[table_name]
        ]
        table_union = " UNION ALL ".join(dbs)
        create_view_query = f"""
            CREATE TEMPORARY VIEW {table_name} AS
                {table_union}
        """
        create_view(connection, table_name, create_view_query)

    # # Create node view
    # create_view_query = """CREATE VIEW rocpd_node AS """
    # selects = [
    #     f"SELECT rocpd_node.* FROM rocpd_node INNER JOIN {t} ON rocpd_node.id = {t}.node_id"
    #     for t in start_end_timed_tables
    # ]
    # create_view_query += " UNION ".join(selects)
    # create_view(connection, "rocpd_node", create_view_query)

    # # Create track view
    # create_view_query = """
    #         CREATE VIEW rocpd_track AS
    #         SELECT rocpd_track.* FROM rocpd_track
    #         INNER JOIN rocpd_sample ON rocpd_sample.track_id = rocpd_track.id
    #     """
    # create_view(connection, "rocpd_track", create_view_query)

    upd_delta = dump_min_max("Windowed")

    reduction = (1.0 - (upd_delta / orig_delta)) * 100.0
    print(f"# Time windowing reduced the duration by {reduction:6.2f}%")

    return connection


#
# Command-line interface functions
#
def add_args(parser: ArgumentParser) -> List[str]:
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

    return ["start", "end", "inclusive", "start_marker", "end_marker"]


def process_args(args, valid_args):

    ret = {}
    for itr in valid_args:
        if hasattr(args, itr):
            val = getattr(args, itr)
            if val is not None:
                ret[itr] = val
    return ret


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

    arg_names = add_args(parser)
    args = parser.parse_args(argv)

    execute(args.input, **{arg: getattr(args, arg) for arg in arg_names})


if __name__ == "__main__":
    main()
