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
import math

from typing import Any, List, Tuple
from .importer import RocpdImportData, execute_statement
from .query import export_sqlite_query

__all__ = [
    "generate_all_summaries",
    "generate_summary_query",
    "generate_domain_query",
    "create_domain_query",
    "create_summary_queries",
    "create_summary_region_queries",
    "export_query",
    "add_args",
    "execute",
    "main",
]

_get_rank_env_variable_name = None
_get_rank_already_searched = False


def check_function_availability(connection, function_name):
    """
    Check if a given SQL function is available in the SQLite database connection.

    This function first attempts to use the modern SQLite pragma_function_list (available in SQLite 3.30.0+)
    to check for the existence of the specified function. If this approach fails (e.g., on older SQLite versions),
    it falls back to attempting to execute the function directly in a simple query. If both checks fail,
    the function is considered unavailable.

    Args:
        connection: The SQLite database connection object.
        function_name (str): The name of the SQL function to check.

    Returns:
        bool: True if the function is available, False otherwise.
    """
    cursor = connection.cursor()

    try:
        # Try the modern approach first (SQLite 3.30.0+)
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM pragma_function_list WHERE name=?)",
            (function_name,),
        )
        result = cursor.fetchone()[0]
        return bool(result)
    except Exception:
        # Fallback for older SQLite versions (Workaround for RHEL 8)
        # Try to execute a simple query using the function to see if it exists
        try:
            cursor.execute(f"SELECT {function_name}(1)")
            return True
        except Exception:
            return False


def get_temp_view_names(connection: RocpdImportData) -> List[str]:
    """Return the names of all temporary views in the SQLite connection."""
    return [
        v[0]
        for v in execute_statement(
            connection, "SELECT name FROM sqlite_temp_master WHERE type='view'"
        ).fetchall()
    ]


def get_temp_view_columns(connection: RocpdImportData, view_name: str) -> List[str]:
    """Return the column names of a given temporary view."""
    cursor = connection.cursor()
    cursor.execute(f"PRAGMA table_xinfo('{view_name}')")
    return [row[1] for row in cursor.fetchall()]


def export_query(
    connection: RocpdImportData,
    output_path,
    output_file,
    output_format,
    query_name,
    query,
) -> None:
    """
    Execute a SQL query and export its results to the specified output format and location.

    Args:
        connection (RocpdImportData): The database connection object.
        output_path (str): Directory where the output file will be saved.
        output_file (str): Base name for the output file. If empty, uses the query_name.
        output_format (str): Output format (e.g., "console", "csv", "json", etc.).
        query_name (str): Name of the query, used for labeling and filenames.
        query (str): The SQL query to execute and export.

    Returns:
        None
    """

    query_not_empty = f"""
        SELECT EXISTS (
            {query}
        )
    """

    # just return if the result is empty
    if not connection.execute(query_not_empty).fetchone()[0]:
        return

    # prepare the output filename
    if not output_file:
        output_filename = query_name
    else:
        output_filename = f"{output_file}_{query_name}"

    if output_format == "console":
        print(f"\n{query_name.upper()}:")

    # call query module to export.  query will append the extension
    export_path = os.path.join(output_path, output_filename)
    export_sqlite_query(
        connection, query, export_format=output_format, export_path=export_path
    )


def get_rank_env_variable_name(connection: RocpdImportData) -> str:
    """
    Determine the environment variable name used to represent the MPI rank in the profiling database.

    This function inspects the 'environment' field in the 'rocpd_info_process' table to identify
    which environment variable (e.g., MPI_RANK, OMPI_COMM_WORLD_RANK, etc.) is present and should
    be used to group or identify ranks in summary queries. The result is cached for subsequent calls.

    Args:
        connection (RocpdImportData): The database connection object.

    Returns:
        str: The name of the environment variable representing the MPI rank, or None if not found.

    Raises:
        ValueError: If no environment data is found in the rocpd_info_process table.
    """

    possible_rank_columns = [
        "MPI_RANK",  # most generic to most runtime-specific
        "MPI_LOCALRANKID",
        "MPI_RANKID",
        "MV2_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_RANK",
    ]

    global _get_rank_env_variable_name
    global _get_rank_already_searched
    if _get_rank_already_searched is True:
        return _get_rank_env_variable_name

    environment_vars = execute_statement(
        connection, "SELECT environment FROM rocpd_info_process LIMIT 1"
    ).fetchall()

    if not environment_vars:
        raise ValueError("No environment data found in rocpd_info_process table.")

    for env in environment_vars:
        import json

        env_data = json.loads(env[0])
        columns = env_data.keys()
        break  # only need to check the first row

    current_rank_column = None
    for rank_column in possible_rank_columns:
        if rank_column in columns:
            current_rank_column = rank_column  # any subsequent matches will override previous ones, to get most specific

    _get_rank_env_variable_name = current_rank_column
    _get_rank_already_searched = True

    return current_rank_column


def generate_summary_query(
    view_name: str,
    view_query="",
    name_column="name",
    by_rank=False,
    by_rank_env_var=None,
) -> Tuple[str, str]:
    """
    Generate a SQL statement and view name for summarizing data from a specified view.

    This function constructs a SQL query that aggregates and summarizes profiling data
    from a given view. It supports both standard and per-rank summaries, grouping by
    the specified name column and optionally by a rank environment variable.

    Args:
        view_name (str): The name of the view to summarize.
        view_query (str, optional): The SQL query to create the view, if needed.
        name_column (str, optional): The column name to group by (default: "name").
        by_rank (bool, optional): If True, generate a summary grouped by rank.
        by_rank_env_var (str, optional): The environment variable name to use for rank grouping.

    Returns:
        Tuple[str, str]: A tuple containing the full view name and the generated SQL query.
    """

    if by_rank:
        view_suffix = "_summary_by_rank"
        group_by_columns = "guid, {name_column}".format(name_column=name_column)
        aggregation_group_by = "T.guid, T.nid, T.{name_column}".format(
            name_column=name_column
        )
        total_duration_group_by = "guid"
        additional_aggregated_columns = """
                    T.guid,
                    T.nid,
                    T.pid,"""
        join_condition = "T.guid = A.guid AND T.{name_column} = A.name".format(
            name_column=name_column
        )
        # If by_rank_env_var is found, include it in the select and join clauses; otherwise, only include ProcessID and Hostname
        if by_rank_env_var is not None:
            rank_column_name = by_rank_env_var
            rank_alias = "Rank"
            additional_select_columns = "json_extract(PI.environment, '$.{rank_column_name}') AS {rank_alias}, AD.pid AS ProcessID, P.hostname AS Hostname,".format(
                rank_column_name=rank_column_name, rank_alias=rank_alias
            )
            total_duration_join = "JOIN total_duration TD ON AD.guid = TD.guid JOIN processes P ON AD.pid = P.pid JOIN rocpd_info_process PI ON AD.pid = PI.pid"
        else:
            rank_column_name = ""
            rank_alias = ""
            additional_select_columns = "AD.pid AS ProcessID, P.hostname AS Hostname,"
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

    # Use a different name for the CTE to avoid circular reference when view_query references the view
    source_table_name = f"{view_name}_source" if view_query else view_name

    view_select = (
        f"""
            {source_table_name} AS (
                {view_query}
            ),
    """
        if view_query
        else ""
    )

    summary_query = f"""
        WITH
            {view_select}
            avg_data AS (
                SELECT
                    {group_by_columns.replace(name_column, f"{name_column} AS name")},
                    AVG(duration) AS avg_duration
                FROM {source_table_name}
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
                FROM {source_table_name} T
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
            {"AD.pid," if by_rank else ""} AD.total_duration DESC
    """

    return (full_view_name, summary_query)


def generate_domain_query(
    connection: RocpdImportData, summary_queries, by_rank=False
) -> Tuple[str, str]:
    """
    Generate a SQL statement and view name for a domain-level summary by combining multiple summary queries.

    This function constructs a SQL query that aggregates and summarizes profiling data across different domains
    by performing a UNION over all provided summary queries. It supports both standard and per-rank summaries,
    grouping and aggregating results as appropriate.

    Args:
        connection (RocpdImportData): The database connection object.
        summary_queries (dict): A dictionary mapping summary query names to their SQL statements.
        by_rank (bool, optional): If True, generate a summary grouped by rank (default: False).

    Returns:
        Tuple[str, str]: A tuple containing the domain summary view name and the generated SQL query.
                         Returns an empty tuple if no matching summary queries are found.
    """

    by_rank_env_var = get_rank_env_variable_name(connection)

    if by_rank:
        view_suffix = "_summary_by_rank"
        view_name = "domain_summary_by_rank"
        if by_rank_env_var is not None:
            rank_column_name = by_rank_env_var
            rank_alias = "Rank"
            additional_select_columns = "json_extract(PI.environment, '$.{rank_column_name}') AS {rank_alias}, GD.ProcessID, GD.Hostname,".format(
                rank_column_name=rank_column_name, rank_alias=rank_alias
            )
            join_condition = "JOIN total_duration TD ON GD.ProcessID = TD.ProcessID JOIN rocpd_info_process PI ON GD.ProcessID = PI.pid"
        else:
            rank_column_name = ""
            rank_alias = ""
            additional_select_columns = "GD.ProcessID, GD.Hostname,"
            join_condition = "JOIN total_duration TD ON GD.ProcessID = TD.ProcessID"
        additional_group_columns = "ProcessID, Hostname,"
        total_duration_group_by = "GROUP BY ProcessID"
        order_by = "ORDER BY GD.ProcessID"
    else:
        view_suffix = "_summary"
        view_name = "domain_summary"
        additional_group_columns = ""
        additional_select_columns = ""
        total_duration_group_by = ""
        join_condition = "CROSS JOIN total_duration TD"
        order_by = 'ORDER BY GD."DURATION (nsec)" DESC'

    summary_dictionary = {
        query_name: query
        for query_name, query in summary_queries.items()
        if query_name.endswith(view_suffix)
    }

    if len(summary_dictionary) < 1:
        return ()

    summary_selects = [
        f"{query_name} AS ({query}) ," for query_name, query in summary_dictionary.items()
    ]

    union_selects = [
        f" SELECT '{query_name.replace(view_suffix, '').upper()}' as domain, * FROM {query_name} "
        for query_name, query in summary_dictionary.items()
    ]

    domain_select = f"""
        WITH
            {f"".join(summary_selects)}
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
        {order_by}
    """

    return (view_name, domain_select)


def _view_matches_category(view_name: str, category: str) -> bool:
    """Return True if view name matches the category."""
    v = view_name.lower()
    c = category.lower()
    plural_map = (c, f"{c}s")
    return v in plural_map or any(v.startswith(f"{itr}_") for itr in plural_map)


def create_summary_queries(
    connection: RocpdImportData,
    by_rank=False,
    only_view_categories=None,
    kernel_name_type=None,
):
    """Create summary queries for eligible temporary views in the database.

    Generate and return a dictionary of summary queries for all suitable temporary views in the database.

    This function scans the temporary views in the database, filters out views that do not meet
    certain criteria, and generates summary queries for each remaining view. It supports both
    standard and per-rank summaries, depending on the by_rank flag.

    When only_view_categories is set (e.g. ["kernel"]), only include views whose
    name matches one of the categories (e.g. "kernels" matches "kernel").

    Args:
        connection (RocpdImportData): The database connection object.
        by_rank (bool, optional): If True, generate summary queries grouped by rank (default: False).
        only_view_categories (list, optional): List of view categories to include. If None, all views are included.
        kernel_name_type (str, optional): The type of kernel name to use (default, truncated or mangled)

    Returns:
        dict: A dictionary mapping summary query names to their SQL statements.
    """

    NAME_COLUMN_MAP = {
        "memory_allocations": "type",
        "scratch_memory": "operation",
    }

    KERNEL_NAME_TYPE_MAP = {
        "truncated": "truncated_kernel_name",
        "mangled": "kernel_name",
    }

    avoid_view_pattern = ("rocpd", "region", "counter", "pmc")
    required_columns = {"duration"}

    views = get_temp_view_names(connection)

    queries = {}

    for view_name in views:
        if any(pattern in view_name for pattern in avoid_view_pattern):
            continue

        if only_view_categories is not None:
            if not any(
                _view_matches_category(view_name, cat) for cat in only_view_categories
            ):
                continue

        columns = get_temp_view_columns(connection, view_name)
        if not required_columns.issubset(columns):
            continue

        # Determine the name column and view query to use
        name_column = NAME_COLUMN_MAP.get(view_name, "name")
        view_query = ""

        if view_name == "kernels" and kernel_name_type is not None:
            # Use display_name as fallback if kernel_name_type is not in the map
            display_name_column = KERNEL_NAME_TYPE_MAP.get(
                kernel_name_type, "display_name"
            )
            view_query = f"""
                SELECT K.*, COALESCE(KS.{display_name_column}, K.name) AS display_name
                FROM kernels K
                LEFT JOIN kernel_symbols KS
                    ON K.kernel_id = KS.id
                    AND K.guid = KS.guid
            """
            name_column = "display_name"

        if not by_rank:
            # Create regular summary query
            summary_query_name, summary_query = generate_summary_query(
                view_name, view_query, name_column=name_column
            )
            queries[summary_query_name] = summary_query
        else:
            # Create per-rank summary query
            per_rank_query_name, summary_by_rank_query = generate_summary_query(
                view_name,
                view_query,
                name_column=name_column,
                by_rank=True,
                by_rank_env_var=get_rank_env_variable_name(connection),
            )
            queries[per_rank_query_name] = summary_by_rank_query

    return queries


def create_summary_region_queries(
    connection: RocpdImportData,
    by_rank=False,
    region_categories=None,
):
    """
    Generate and return a dictionary of region-specific summary queries for the database.

    This function identifies region categories from the database (or uses those provided),
    constructs queries for each region category, and generates summary queries for each.
    It supports both standard and per-rank summaries.

    Args:
        connection (RocpdImportData): The database connection object.
        by_rank (bool, optional): If True, generate summary queries grouped by rank (default: False).
        region_categories (list, optional): List of region categories to include. If None, categories are auto-detected.

    Returns:
        dict: A dictionary mapping summary query names to their SQL statements.
    """

    query = "SELECT DISTINCT(category) FROM regions_and_samples"
    categories = execute_statement(connection, query).fetchall()

    category_prefixes = ["rocm_"]

    # Convert all categories and prefixes to lowercase for correct comparison
    categories = [(cat[0].lower(),) for cat in categories]
    category_prefixes = [prefix.lower() for prefix in category_prefixes]

    if region_categories is not None:
        region_categories = [cat.lower() for cat in region_categories]
    else:
        # Automatically retrieve region categories from the database
        region_categories = set()
        for cat in categories:
            category_name = cat[0]
            matching_prefix = next(
                (
                    prefix
                    for prefix in category_prefixes
                    if category_name.startswith(prefix)
                ),
                "",
            )
            first_part = category_name[len(matching_prefix) :].split("_")[0]
            region_categories.add(f"{matching_prefix}{first_part}")

    category_map = {
        cat: [c[0] for c in categories if c[0] == cat or c[0].startswith(cat + "_")]
        for cat in region_categories
        if "MARKER" not in cat.upper()
    }

    queries = {}

    for k, v in category_map.items():
        if len(v) > 0:
            conditions = [f"category LIKE '{c}'" for c in v]
            region_query = f"""
                SELECT *
                FROM regions_and_samples
                WHERE {" OR ".join(conditions)}
            """

            if not by_rank:
                # Create regular summary query
                summary_query_name, summary_query = generate_summary_query(
                    k, region_query
                )
                queries[summary_query_name] = summary_query
            else:
                # Create per-rank summary query
                per_rank_query_name, summary_by_rank_query = generate_summary_query(
                    k,
                    region_query,
                    by_rank=True,
                    by_rank_env_var=get_rank_env_variable_name(connection),
                )
                queries[per_rank_query_name] = summary_by_rank_query

    # Markers
    if "MARKER" not in region_categories:
        return queries

    markers_query_name = "markers"
    markers_query = """
        SELECT JSON_EXTRACT(extdata, '$.message') AS marker_name, *
        FROM regions_and_samples
        WHERE category LIKE 'MARKER_%'
    """

    # Create regular summary query
    summary_query_name, summary_query = generate_summary_query(
        markers_query_name, markers_query, name_column="marker_name"
    )
    queries[summary_query_name] = summary_query

    # Create per-rank summary query
    if by_rank:
        per_rank_query_name, summary_by_rank_query = generate_summary_query(
            markers_query_name, markers_query, name_column="marker_name", by_rank=True
        )
        queries[per_rank_query_name] = summary_by_rank_query

    return queries


def create_domain_query(connection: RocpdImportData, summary_queries, by_rank=False):
    """
    Create a domain summary query by aggregating all provided summary queries.

    This function generates a domain-level summary query by combining the given summary queries,
    optionally grouping by rank if requested.

    Args:
        connection (RocpdImportData): The database connection object.
        summary_queries (dict): A dictionary mapping summary query names to their SQL statements.
        by_rank (bool, optional): If True, generate a summary grouped by rank (default: False).

    Returns:
        dict: A dictionary with the domain summary query name as key and its SQL statement as value.
    """

    result = generate_domain_query(connection, summary_queries, by_rank=by_rank)
    if not result:
        return {}

    query_name, query = result

    return {query_name: query}


def generate_all_summaries(connection: RocpdImportData, **kwargs: Any) -> None:
    """
    Generate all summary queries and export their results in the selected format.

    This function creates summary and domain summary queries, executes them, and exports
    the results to the specified output format and location.

    Args:
        connection (RocpdImportData): The database connection object.
        **kwargs: Additional keyword arguments controlling summary generation and export options.

    Returns:
        None
    """

    domain_summary = kwargs.get("domain_summary", False)
    by_rank = kwargs.get("summary_by_rank", False)
    output_file = kwargs.get("output_file", "")
    output_path = kwargs.get("output_path", "./rocpd-output-data")
    region_categories = kwargs.get("region_categories", None)
    output_format = kwargs.get("format", "console")
    mangled_kernels = kwargs.get("mangled_kernels", False)
    truncate_kernels = kwargs.get("truncate_kernels", False)

    if mangled_kernels:
        kernel_name_type = "mangled"
    elif truncate_kernels:
        kernel_name_type = "truncated"
    else:
        kernel_name_type = None

    if not check_function_availability(connection, "sqrt"):
        connection.create_function(
            "sqrt",
            1,
            lambda x: (
                math.sqrt(x)
                if x is not None and isinstance(x, (int, float)) and x >= 0
                else None
            ),
        )

    summary_queries = {}

    # Create the summary queries.
    # When region_categories is specified (and not NONE), only include view-based
    # summaries that match those categories and only region summaries for those categories.
    # With --region-categories NONE we only skip region summaries and still include
    # all view-based summaries.
    is_none_categories = (
        region_categories is not None
        and len(region_categories) == 1
        and str(region_categories[0]).strip().upper() == "NONE"
    )
    if region_categories is None or is_none_categories:
        summary_queries.update(
            create_summary_queries(connection, by_rank, kernel_name_type=kernel_name_type)
        )
    else:
        summary_queries.update(
            create_summary_queries(
                connection,
                by_rank,
                only_view_categories=region_categories,
                kernel_name_type=kernel_name_type,
            )
        )
    summary_queries.update(
        create_summary_region_queries(
            connection, by_rank, region_categories=region_categories
        )
    )

    if domain_summary:
        if not by_rank:
            summary_queries.update(create_domain_query(connection, summary_queries))
        else:
            # Create domain summary per rank only if both domain_summary and summary_by_rank are enabled
            summary_queries.update(
                create_domain_query(connection, summary_queries, by_rank=True)
            )

    # Export all summary queries
    for query_name, query in summary_queries.items():
        export_query(
            connection, output_path, output_file, output_format, query_name, query
        )


#
# Command-line interface functions
#
def add_args(parser):
    """
    Add summary-related command-line arguments to the argument parser.

    This function defines and adds options for output format, domain summary, summary by rank,
    and region categories to the provided argparse parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to which arguments will be added.

    Returns:
        function: A function that processes parsed arguments and returns a dictionary of relevant options.
    """

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
    summary_options.add_argument(
        "--mangled-kernels",
        action="store_true",
        default=False,
        help="Display mangled kernel names (do not demangle)",
    )
    summary_options.add_argument(
        "--truncate-kernels",
        action="store_true",
        default=False,
        help="Display truncated kernel names (function name only, without template parameters)",
    )

    def process_args(input, args):
        valid_args = [
            "format",
            "domain_summary",
            "summary_by_rank",
            "region_categories",
            "mangled_kernels",
            "truncate_kernels",
        ]

        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is not None:
                    ret[itr] = val
        return ret

    return process_args


def execute(input, **kwargs: Any) -> RocpdImportData:
    """
    Execute the summary generation and export process.

    This function calls generate_all_summaries with the provided input and keyword arguments.

    Args:
        input (RocpdImportData): The database connection object.
        **kwargs: Additional keyword arguments for summary generation.

    Returns:
        RocpdImportData: The input database connection object.
    """

    generate_all_summaries(input, **kwargs)

    return input


def main(argv=None) -> int:
    """Main entry point for command line execution."""
    from . import time_window
    from . import output_config

    parser = argparse.ArgumentParser(description="Generate summary views from rocPD data")
    required_params = parser.add_argument_group("Required options")

    required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Input path and filename to one or more database(s), separated by spaces",
    )

    process_outcfg_args = output_config.add_args(parser)
    process_summary_args = add_args(parser)
    process_time_window_args = time_window.add_args(parser)

    args = parser.parse_args(argv)

    input = RocpdImportData(
        args.input, automerge_limit=getattr(args, "automerge_limit", None)
    )

    summary_args = process_summary_args(input, args)
    io_args = process_outcfg_args(input, args)
    process_time_window_args(input, args)

    all_args = {**summary_args, **io_args}

    execute(
        input,
        **all_args,
    )


if __name__ == "__main__":
    main()
