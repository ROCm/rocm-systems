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
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Any

from .importer import RocpdImportData, execute_statement
from .schema import RocpdSchema
from .time_window import get_column_names

__all__ = ["RocpdMergeData", "merge"]


def prepare_output_file(output: str) -> None:
    """Prepare output file by creating directory and removing existing file"""

    output_path = Path(output)

    # Create parent directory if needed
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file
    if output_path.is_file():
        output_path.unlink()


def get_database_list(import_data: RocpdImportData) -> List[Tuple[int, str, str]]:
    """Get list of all attached databases with their sequence numbers, names, and file"""
    return execute_statement(import_data, "PRAGMA database_list").fetchall()


def get_table_names_per_alias(import_data: RocpdImportData, alias: str) -> List[str]:
    """Get all table names from a specific database alias"""

    query = f"SELECT name FROM {alias}.sqlite_master WHERE type='table';"
    rows = execute_statement(import_data, query).fetchall()
    return [table[0] for table in rows if not table[0].startswith("sqlite_")]


def get_table_data(import_data: RocpdImportData, table_name: str):
    """Get table data using execute_statement"""
    return execute_statement(import_data, f"SELECT * FROM {table_name}").fetchall()


def get_uuid_guid(import_data: RocpdImportData, metadata_table: str) -> Tuple[str, str]:
    """Get UUID and GUID values from the metadata table"""
    uuid = execute_statement(
        import_data,
        f"SELECT value FROM {metadata_table} WHERE tag='uuid' ORDER BY id ASC",
    ).fetchone()[0]
    guid = execute_statement(
        import_data,
        f"SELECT value FROM {metadata_table} WHERE tag='guid' ORDER BY id ASC",
    ).fetchone()[0]
    return uuid, guid


class RocpdMergeData:
    """Utility class for merging ROCProfiler databases."""

    def __init__(self, import_data: RocpdImportData, output_path: str):
        if not isinstance(import_data, RocpdImportData):
            raise ValueError(
                f"Expected RocpdImportData, got {type(import_data).__name__}"
            )

        if not output_path:
            raise ValueError("output_path cannot be empty")

        self.import_data = import_data
        self.output_path = output_path
        self._connection = None

    def __enter__(self):
        """Support 'with RocpdMergeData(...) as merger:' pattern"""
        prepare_output_file(self.output_path)
        self._connection = sqlite3.connect(self.output_path)
        return self

    def __exit__(self, exc_type, *_):
        """Clean up resources"""
        if self._connection:
            try:
                if exc_type is None:
                    self._connection.commit()
                else:
                    self._connection.rollback()
            finally:
                self._connection.close()
                self._connection = None

        return False

    def merge(self) -> None:
        """Execute the merge operation"""

        all_tables = self._get_all_tables()
        print(f"Merging {len(all_tables)} tables...")

        # Create tables per uuid / guid
        uuid_guuids = []
        for table_name in all_tables:
            if "rocpd_metadata" in table_name:
                uuid, guid = get_uuid_guid(self.import_data, table_name)
                self._connection.executescript(RocpdSchema(uuid=uuid, guid=guid).tables)
                uuid_guuids.append((uuid, guid))

        # Insert data in tables
        for table_name in all_tables:
            if "rocpd_metadata" in table_name:
                continue
            data = get_table_data(self.import_data, table_name)
            if data:
                column_names = get_column_names(self.import_data, table_name)
                self._insert_data_into_merged(data, len(column_names), table_name)

        # Create rocpd_<> views
        views_by_base_name = defaultdict(list)  # view name -> list of table names
        for _uuid, _ in uuid_guuids:
            for tablename_with_uuid in all_tables:
                if _uuid in tablename_with_uuid:
                    table_name = tablename_with_uuid.replace(_uuid, "")
                    views_by_base_name[table_name].append(tablename_with_uuid)

        self._connection.executescript(self._create_union_views(views_by_base_name))

        # Create rest of the views
        self._connection.executescript(RocpdSchema().views)

    def _get_all_tables(self) -> list:
        """Get all tables from all attached databases and verify no duplicates exist"""

        dbs = get_database_list(self.import_data)
        if len(dbs) <= 2:  # main and temp
            raise ValueError("No databases attached for merging")

        all_tables = []
        for db in dbs:
            all_tables.extend(get_table_names_per_alias(self.import_data, db[1]))

        # Check for duplicates
        unique_tables = set(all_tables)
        assert len(all_tables) == len(
            unique_tables
        ), f"Duplicate tables found: {set([x for x in all_tables if all_tables.count(x) > 1])}"

        return all_tables

    def _insert_data_into_merged(
        self, table_data: list, columns_count: int, table_name: str
    ) -> None:
        """Insert data into merged database"""
        placeholders = ", ".join(["?" for _ in range(columns_count)])
        insert_statement = f"INSERT INTO {table_name} VALUES ({placeholders})"
        self._connection.executemany(insert_statement, table_data)

    def _create_union_views(self, views_by_base_name) -> list:
        union_views = []

        for view_name, table_names in views_by_base_name.items():
            if len(table_names) == 1:
                union_views.append(
                    f"""CREATE VIEW IF NOT EXISTS `{view_name}` AS SELECT * FROM `{table_names[0]}`;"""
                )
            else:
                select_statements = [f"SELECT * FROM `{table}`" for table in table_names]
                union_query = "\nUNION ALL\n".join(select_statements)

                union_views.append(
                    f"""CREATE VIEW IF NOT EXISTS `{view_name}` AS {union_query};"""
                )
        return "\n\n".join(union_views)


def merge(import_data: RocpdImportData, **kwargs: Any) -> None:
    start_time = time.time()

    output_path = kwargs.get("output_merge_path")
    with RocpdMergeData(import_data, output_path) as merger:
        merger.merge()

    elapsed_time = time.time() - start_time
    print(f"Merge completed successfully! Output saved to: {output_path}")
    print(f"Time: {elapsed_time:.2f} sec")


#
# Command-line interface functions
#
def add_args(parser):
    """Add arguments for merger."""
    merge_options = parser.add_argument_group("Merge options")
    merge_options.add_argument(
        "--output-merge-path",
        help="Sets the output path where the output merge files will be saved (default path: `./rocpd-output-data/db_merged.db`)",
        default=os.environ.get("ROCPD_OUTPUT_PATH", "./rocpd-output-data/db_merged.db"),
        type=str,
        required=False,
    )

    return ["output_merge_path"]


def process_args(args, valid_args):
    ret = {}
    for itr in valid_args:
        if hasattr(args, itr):
            val = getattr(args, itr)
            if val is not None:
                ret[itr] = val
    return ret


def execute(input_rpd: str, **kwargs: Any) -> RocpdImportData:

    importData = RocpdImportData(input_rpd)

    merge(importData, **kwargs)

    return importData


def main(argv=None) -> int:
    """Main entry point for command line execution."""

    parser = argparse.ArgumentParser(description="Merge ROCpd databases")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input ROCpd database files",
    )

    valid_args = add_args(parser)

    args = parser.parse_args(argv)

    merged_args = process_args(args, valid_args)

    execute(args.input, **merged_args)


if __name__ == "__main__":
    main()
