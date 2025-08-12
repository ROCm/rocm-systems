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

import sys
import sqlite3
import argparse

from .importer import RocpdImportData, execute_statement


def get_distinct_in_column(connection, table, column, conditions=""):
    """Get distinct values from a specific column."""
    return [
        itr[0]
        for itr in connection.execute(
            f"SELECT DISTINCT({column}) FROM {table} {conditions}"
        ).fetchall()
    ]


def create_view(connection: sqlite3.Connection, view_name: str, query: str) -> None:
    """Create or replace a database view."""
    execute_statement(connection, f"DROP VIEW IF EXISTS {view_name}")
    execute_statement(connection, query)
    connection.commit()


def get_column_names(conn: RocpdImportData, table_name: str):
    """
    Use SELECT on zero rows and read cursor.description.
    """
    cursor = conn.execute(f"SELECT * FROM '{table_name}' LIMIT 0")
    return [desc[0] for desc in cursor.description]


def apply_filter(connection: RocpdImportData, **kwargs) -> None:
    """Apply filtering to create filtered views."""

    include_category = kwargs.get("include_category", None)
    exclude_category = kwargs.get("exclude_category", None)
    if include_category is not None or exclude_category is not None:
        categories = (
            connection.filters["category"]
            if "category" in connection.filters and connection.filters["category"]
            else get_distinct_in_column(connection, "rocpd_info_category", "name")
        )
        remaining_categories = categories[:]
        if include_category is not None:
            for itr in include_category:
                if itr in categories and itr not in remaining_categories:
                    remaining_categories.append(itr)
        if exclude_category is not None:
            for itr in exclude_category:
                if itr in remaining_categories:
                    remaining_categories.remove(itr)

        connection.filters["category"] = remaining_categories

    # Create views for tables with category_id filtered
    if "category" in connection.filters and connection.filters["category"]:
        category_tables = []  # dedicated table for categories

        # Get all tables that have a category_id column
        for itr in connection.table_info.keys():
            if itr.find("rocpd_info_") == 0:
                continue
            column_names = get_column_names(connection, itr)
            if "category_id" in column_names:
                category_tables += [itr]

        # Get the distinct category IDs for the specified categories
        category_ids = get_distinct_in_column(
            connection,
            "rocpd_info_category",
            "id",
            "WHERE name IN ({})".format(
                ",".join(f"'{cat}'" for cat in connection.filters.get("category", []))
            ),
        )

        # Create views for each table that has a category_id values in the specified categories
        filtered_category_ids = ", ".join([f"{itr}" for itr in category_ids])
        for table_name in category_tables:
            dbs = [
                f"{itr} WHERE category_id IN ({filtered_category_ids})"
                for itr in connection.table_info[table_name]
            ]
            table_union = " UNION ALL ".join(dbs)
            create_view_schema = f"""
                CREATE TEMPORARY VIEW {table_name} AS
                    {table_union}
            """
            create_view(connection, table_name, create_view_schema)

    return connection


#
# Command-line interface functions
#
def add_args(parser: argparse.ArgumentParser):
    """Add filtering options to existing argument parser."""

    filter_options = parser.add_argument_group("Filter options")

    # Start time mutually exclusive group
    filter_options.add_argument(
        "--list-categories",
        action="store_true",
        help="List all available categories",
    )
    filter_options.add_argument(
        "--include-category",
        type=str,
        help="Explicit list of categories to include",
        nargs="+",
        default=None,
    )
    filter_options.add_argument(
        "--exclude-category",
        type=str,
        help="Named marker event to use as window start point",
        nargs="+",
        default=None,
    )

    return ["list_categories", "include_category", "exclude_category"]


def process_args(args, valid_args):

    ret = {}
    for itr in valid_args:
        if hasattr(args, itr):
            val = getattr(args, itr)
            if val is not None:
                ret[itr] = val
    return ret


def check_args(connection: RocpdImportData, **kwargs):
    """Check if the provided arguments are valid for filtering."""

    categories = sorted(get_distinct_in_column(connection, "rocpd_info_category", "name"))
    for option in ["include_category", "exclude_category"]:
        option_args = kwargs.get(option, None)
        if option_args is not None:
            option_name = "--{}".format(option.replace("_", "-"))
            invalid_categories = [itr for itr in option_args if itr not in categories]
            if invalid_categories:
                raise argparse.ArgumentError(
                    f"{option_name} must be one of {categories}. Invalid categories: {invalid_categories}"
                )

    if kwargs.get("list_categories", False):
        print("Available categories:")
        for category in categories:
            print(f"  - {category}")
        sys.exit(0)


def execute(input_rpd: str, **kwargs) -> RocpdImportData:
    """Execute time window filtering on database file."""

    importData = RocpdImportData(input_rpd)

    apply_filter(importData, **kwargs)

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
