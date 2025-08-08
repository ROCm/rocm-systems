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
import os

from typing import Any, List, Tuple
from .importer import RocpdImportData, execute_statement
from .query import export_sqlite_query
from . import output_config


def get_temp_view_names(connection: RocpdImportData) -> List[str]:
    """Return the names of all temporary views in the SQLite connection."""
    return [
        v[0]
        for v in execute_statement(
            connection, "SELECT name FROM sqlite_temp_master WHERE type='view';"
        ).fetchall()
    ]


def get_temp_view_columns(connection: RocpdImportData, view_name: str) -> List[str]:
    """Return the column names of a given temporary view."""
    cursor = connection.cursor()
    cursor.execute(f"PRAGMA table_xinfo('{view_name}')")
    return [row[1] for row in cursor.fetchall()]


def make_temp_view_query(view_name, query) -> str:
    return "CREATE TEMPORARY VIEW IF NOT EXISTS `{}` AS {}".format(view_name, query)


def export_view(
    connection: RocpdImportData, view_name, output_format, output_path, filename=""
) -> None:
    """Write the contents of a SQL view to an output format."""

    query = "SELECT * FROM `{}`".format(view_name)
    query_one = "SELECT * FROM `{}` LIMIT 1".format(view_name)

    # just return if view is empty
    if not connection.execute(query_one).fetchone():
        return

    # prepare the output filename
    if not filename:
        output_filename = view_name
    else:
        output_filename = f"{filename}_{view_name}"

    if output_format == "console":
        print(f"\n{view_name.upper()}:")

    # call query module to export.  query will append the extension
    export_path = os.path.join(output_path, output_filename)
    export_sqlite_query(
        connection, query, export_format=output_format, export_path=export_path
    )


def generate_summary_query(
    view_name: str,
    name_column="name",
    by_rank=False,
) -> Tuple[str, str]:
    """Generate the SQL statement to create a summary view."""

    if by_rank:
        view_suffix = "_summary_by_rank"
        group_by_columns = "guid, {name_column}".format(name_column=name_column)
        aggregation_group_by = "T.guid, T.nid, T.{name_column}".format(
            name_column=name_column
        )
        total_duration_group_by = "guid"
        additional_select_columns = "AD.pid AS ProcessID, P.hostname AS Hostname,"
        additional_aggregated_columns = """
                    T.guid,
                    T.nid,
                    T.pid,"""
        join_condition = "T.guid = A.guid AND T.{name_column} = A.name".format(
            name_column=name_column
        )
        total_duration_join = "JOIN total_duration TD ON AD.guid = TD.guid JOIN processes P ON AD.pid = P.pid"
    else:
        view_suffix = "_summary"
        group_by_columns = name_column
        aggregation_group_by = "T.{name_column}".format(name_column=name_column)
        total_duration_group_by = ""
        additional_select_columns = ""
        additional_aggregated_columns = ""
        join_condition = "T.{name_column} = A.name".format(name_column=name_column)
        total_duration_join = "CROSS JOIN total_duration TD"

    full_view_name = f"{view_name}{view_suffix}"

    summary_query = f"""
        WITH
            avg_data AS (
                SELECT
                    {group_by_columns.replace(name_column, f"{name_column} AS name")},
                    AVG(duration) AS avg_duration
                FROM {view_name}
                GROUP BY {group_by_columns}
            ),
            aggregated_data AS (
                SELECT{additional_aggregated_columns}
                    T.{name_column} as name,
                    COUNT(*) AS calls,
                    SUM(T.duration) AS total_duration,
                    A.avg_duration AS average_duration,
                    MIN(T.duration) AS min_duration,
                    MAX(T.duration) AS max_duration,
                    SQRT(SUM(CAST((T.duration - A.avg_duration) AS REAL) * CAST((T.duration - A.avg_duration) AS REAL)) / (COUNT(*) - 1)) AS std_dev_duration
                FROM {view_name} T
                JOIN avg_data A ON {join_condition}
                GROUP BY {aggregation_group_by}
            ),
            total_duration AS (
                SELECT
                    {f"{total_duration_group_by}," if total_duration_group_by else ""}
                    SUM(total_duration) AS grand_total_duration
                FROM
                    aggregated_data
                {f"GROUP BY {total_duration_group_by}" if total_duration_group_by else ""}
            )
        SELECT
            {additional_select_columns}
            AD.name AS Name,
            AD.calls AS Calls,
            AD.total_duration AS "DURATION (nsec)",
            AD.average_duration AS "AVERAGE (nsec)",
            (CAST(AD.total_duration AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
            AD.min_duration AS "MIN (nsec)",
            AD.max_duration AS "MAX (nsec)",
            AD.std_dev_duration AS "STD_DEV"
        FROM
            aggregated_data AD
            {total_duration_join}
        ORDER BY
            {"AD.pid," if by_rank else ""} AD.total_duration DESC;
    """

    return (full_view_name, summary_query)


def generate_domain_query(connection: RocpdImportData, by_rank=False) -> Tuple[str, str]:
    """Generate the SQL statement for domain summary by doing union over all summary views."""

    if by_rank:
        view_suffix = "_summary_by_rank"
        view_name = "domain_summary_by_rank"
        additional_group_columns = "ProcessID, Hostname,"
        additional_select_columns = "GD.ProcessID, GD.Hostname,"
        total_duration_group_by = "GROUP BY ProcessID"
        join_condition = "JOIN total_duration TD ON GD.ProcessID = TD.ProcessID"
        order_by = "ORDER BY GD.ProcessID"
    else:
        view_suffix = "_summary"
        view_name = "domain_summary"
        additional_group_columns = ""
        additional_select_columns = ""
        total_duration_group_by = ""
        join_condition = "CROSS JOIN total_duration TD"
        order_by = 'ORDER BY GD."DURATION (nsec)" DESC'

    summary_views = [
        itr for itr in get_temp_view_names(connection) if itr.endswith(view_suffix)
    ]

    if len(summary_views) < 1:
        return view_name

    union_selects = [
        f" SELECT '{s.replace(view_suffix, '').upper()}' as domain, * FROM {s} "
        for s in summary_views
    ]

    domain_select = f"""
        WITH
            all_domains AS (
               {f" UNION ALL ".join(union_selects)}
            ),
            grouped_domains AS (
                SELECT
                    domain,
                    {additional_group_columns}
                    SUM(calls) AS calls,
                    SUM("DURATION (nsec)") AS "DURATION (nsec)",
                    SUM("AVERAGE (nsec)") AS "AVERAGE (nsec)",
                    MIN("MIN (nsec)") AS "MIN (nsec)",
                    MAX("MAX (nsec)") AS "MAX (nsec)",
                    SUM("STD_DEV") AS "STD_DEV"
                FROM all_domains
                GROUP BY domain{", ProcessID" if by_rank else ""}
            ),
            total_duration AS (
                SELECT
                    {additional_group_columns}
                    SUM("DURATION (nsec)") AS grand_total_duration
                FROM grouped_domains
                {total_duration_group_by}
            )
        SELECT
            {additional_select_columns}
            GD.domain AS Name,
            GD.calls AS Calls,
            GD."DURATION (nsec)",
            GD."AVERAGE (nsec)",
            (CAST(GD."DURATION (nsec)" AS REAL) / TD.grand_total_duration) * 100 AS "PERCENT (INC)",
            GD."MIN (nsec)",
            GD."MAX (nsec)",
            GD."STD_DEV"
        FROM
            grouped_domains GD
            {join_condition}
        {order_by};
    """

    return (view_name, domain_select)


def create_summary_views(connection: RocpdImportData, by_rank=False) -> None:
    """Create summary views for eligible temporary views in the database."""

    NAME_COLUMN_MAP = {
        "memory_allocations": "type",
        "scratch_memory": "operation",
    }

    avoid_view_pattern = ("rocpd", "region", "counter", "pmc")
    required_columns = {"duration"}

    views = get_temp_view_names(connection)

    for view_name in views:
        if any(pattern in view_name for pattern in avoid_view_pattern):
            continue

        columns = get_temp_view_columns(connection, view_name)
        if not required_columns.issubset(columns):
            continue

        # Create regular summary view
        summary_view_name, summary_query = generate_summary_query(
            view_name, name_column=NAME_COLUMN_MAP.get(view_name, "name")
        )
        connection.execute(make_temp_view_query(summary_view_name, summary_query))

        # Create per-rank summary
        if by_rank:
            per_rank_view_name, summary_by_rank_query = generate_summary_query(
                view_name,
                name_column=NAME_COLUMN_MAP.get(view_name, "name"),
                by_rank=True,
            )
            connection.execute(
                make_temp_view_query(per_rank_view_name, summary_by_rank_query)
            )


def create_summary_region_views(
    connection: RocpdImportData, by_rank=False, region_categories=None
) -> None:
    """Create summary and region views"""

    query = "SELECT DISTINCT(category) FROM regions_and_samples;"
    categories = execute_statement(connection, query).fetchall()

    if region_categories is None:
        # Automatically retrieve region categories from the database
        region_categories = set([cat[0].split("_")[0] for cat in categories])

    category_map = {
        cat.lower(): [c[0] for c in categories if c[0].startswith(cat + "_")]
        for cat in region_categories
        if "MARKER" not in cat.upper()
    }

    for k, v in category_map.items():
        if len(v) > 0:
            conditions = [f"category LIKE '{c}'" for c in v]
            temp_region_view = f"""
                CREATE TEMPORARY VIEW IF NOT EXISTS `{k}` AS
                SELECT *
                FROM regions_and_samples
                WHERE {" OR ".join(conditions)};
            """

            connection.execute(temp_region_view)

            # Create regular summary view
            summary_view_name, summary_query = generate_summary_query(k)
            connection.execute(make_temp_view_query(summary_view_name, summary_query))

            # Create per-rank summary view
            if by_rank:
                per_rank_view_name, summary_by_rank_query = generate_summary_query(
                    k, by_rank=True
                )
                connection.execute(
                    make_temp_view_query(per_rank_view_name, summary_by_rank_query)
                )

    # Markers
    if "MARKER" not in region_categories:
        return

    view_name = "markers"
    markers_create = f"""
        CREATE TEMPORARY VIEW IF NOT EXISTS `{view_name}` AS
        SELECT JSON_EXTRACT(extdata, '$.message') AS marker_name, *
        FROM regions_and_samples
        WHERE category LIKE 'MARKER_%'
    """
    connection.execute(markers_create)

    # Create regular summary view
    summary_view_name, summary_query = generate_summary_query(
        view_name, name_column="marker_name"
    )
    connection.execute(make_temp_view_query(summary_view_name, summary_query))

    # Create per-rank summary view
    if by_rank:
        per_rank_view_name, summary_by_rank_query = generate_summary_query(
            view_name, name_column="marker_name", by_rank=True
        )
        connection.execute(
            make_temp_view_query(per_rank_view_name, summary_by_rank_query)
        )


def create_domain_view(connection: RocpdImportData, by_rank=False) -> str:
    """Create a domain summary view by aggregating all summary views."""

    view_name, domain_query = generate_domain_query(connection, by_rank=by_rank)

    # Create the domain summary view
    connection.execute(make_temp_view_query(view_name, domain_query))

    return view_name


def generate_all_summaries(connection: RocpdImportData, **kwargs: Any) -> None:
    """Generate all summary views and write them to CSV files."""

    domain_summary = kwargs.get("domain_summary", False)
    by_rank = kwargs.get("summary_by_rank", False)
    filename = kwargs.get("output_file", "")
    output_path = kwargs.get("output_path", "./rocpd-output-data")
    region_categories = kwargs.get("region_categories", None)
    output_format = kwargs.get("format", "console")

    # create the temporary summary views
    create_summary_views(connection, by_rank)
    create_summary_region_views(connection, by_rank, region_categories=region_categories)

    if domain_summary:
        create_domain_view(connection)
        # Create domain summary per rank only if both domain_summary and summary_by_rank are enabled
        if by_rank:
            create_domain_view(connection, by_rank=True)

    # Write regular summary views
    print("\nSummary files:")
    summary_views = [
        itr for itr in get_temp_view_names(connection) if itr.endswith("_summary")
    ]
    for v in summary_views:
        export_view(connection, v, output_format, output_path, filename)

    # Write per-rank summary views if flag is set
    if by_rank:
        print("\nSummary files by rank:")
        summary_by_rank_views = [
            itr
            for itr in get_temp_view_names(connection)
            if itr.endswith("_summary_by_rank")
        ]
        for v in summary_by_rank_views:
            export_view(connection, v, output_format, output_path, filename)


#
# Command-line interface functions
#


def add_io_args(parser):
    """Add input/output arguments for summary."""
    io_options = parser.add_argument_group("I/O options")

    io_options.add_argument(
        "-f",
        "--format",
        help="Sets the format the summaries are output to (default: console)",
        choices=("console", "csv", "html", "json", "md", "pdf"),
        default="console",
        type=str,
        required=False,
    )
    io_options.add_argument(
        "-o",
        "--output-file",
        help="Sets the base output file name",
        default=os.environ.get("ROCPD_OUTPUT_NAME", ""),
        type=str,
        required=False,
    )
    io_options.add_argument(
        "-d",
        "--output-path",
        help="Sets the output path where the output files will be saved (default path: `./rocpd-output-data`)",
        default=os.environ.get("ROCPD_OUTPUT_PATH", "./rocpd-output-data"),
        type=str,
        required=False,
    )

    return ["format", "output_file", "output_path"]


def add_args(parser):
    """Add arguments for summary."""
    summary_options = parser.add_argument_group("Summary options")
    summary_options.add_argument(
        "--domain-summary",
        action="store_true",
        default=False,
        help="Generate domain summary view",
    )
    summary_options.add_argument(
        "--summary-by-rank",
        action="store_true",
        default=False,
        help="Generate summary views by-rank (or Process ID)",
    )
    summary_options.add_argument(
        "--region-categories",
        nargs="+",
        default=None,
        help="Specify region categories to include in the summary (example: HIP, HSA, RCCL, ROCDECODE, ROCJPEG, MARKER). If not specified, categories will be automatically retrieved from the database.",
    )

    return ["domain_summary", "summary_by_rank", "region_categories"]


def process_args(args, valid_args):

    ret = {}
    for itr in valid_args:
        if hasattr(args, itr):
            val = getattr(args, itr)
            if val is not None:
                ret[itr] = val
    return ret


def execute(input, window_args=None, **kwargs: Any) -> RocpdImportData:
    from .time_window import apply_time_window

    importData = RocpdImportData(input)

    apply_time_window(importData, **window_args)

    generate_all_summaries(importData, **kwargs)

    return importData


def main(argv=None) -> int:
    """Main entry point for command line execution."""
    from .time_window import add_args as add_args_time_window
    from .time_window import process_args as process_args_time_window

    parser = argparse.ArgumentParser(
        description="Create ROCpd database summary region views"
    )
    required_params = parser.add_argument_group("Required options")

    required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s), separated by spaces",
    )

    valid_io_args = add_io_args(parser)
    valid_summary_args = add_args(parser)
    valid_time_window_args = add_args_time_window(parser)

    args = parser.parse_args(argv)

    summary_args = process_args(args, valid_summary_args)
    io_args = output_config.process_args(args, valid_io_args)
    window_args = process_args_time_window(args, valid_time_window_args)

    all_args = {**summary_args, **io_args}

    execute(
        args.input,
        window_args=window_args,
        **all_args,
    )


if __name__ == "__main__":
    main()
