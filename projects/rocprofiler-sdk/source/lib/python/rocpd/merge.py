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

from typing import List, Dict, Iterable, Optional, Callable, Any


def merge_sqlite_dbs(
    sources: Iterable[str],
    dest_path: str,
    on_log: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Merge multiple SQLite databases into a single destination database.

    Parameters
    ----------
    sources : Iterable[str]
        Paths to source databases.
    dest_path : str
        Path to destination database.
    on_log : Optional[Callable[[str], None]]
        Logger function; defaults to None. Pass `print` to generate logs.
    """

    def log(msg: str) -> None:
        if on_log:
            on_log(f"  {msg}")

    sources = list(sources)
    if not sources:
        raise ValueError("No source databases provided")

    # Prepare output directory
    output_dir = os.path.dirname(os.path.abspath(dest_path)) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Remove existing file
    if os.path.isfile(dest_path):
        os.remove(dest_path)

    uuids = []
    views = []
    data_views = []
    schema_versions = []

    with sqlite3.connect(str(dest_path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = OFF;")  # defer FK checks until end

        # One big atomic transaction
        with conn:
            # Attach sources one by one
            for i, src in enumerate(sources, 1):
                alias = f"src{i}"
                conn.execute(f"ATTACH DATABASE ? AS {alias}", (src,))
                print(f"Adding {src}")
                log(f"Attached {src} AS {alias}")

                # UUIDs and schema version
                _uuids = [
                    itr[0]
                    for itr in conn.execute(
                        f"SELECT value FROM {alias}.rocpd_metadata WHERE tag='uuid'",
                    ).fetchall()
                ]
                uuids += [itr for itr in _uuids if itr not in uuids]

                _schema_versions = [
                    itr[0]
                    for itr in conn.execute(
                        f"SELECT value FROM {alias}.rocpd_metadata WHERE tag='schema_version'",
                    ).fetchall()
                ]
                schema_versions += _schema_versions

                # Helper: fetch rows from attached sqlite_master
                def fetch_master(_alias: str, kind: str):
                    cur = conn.execute(
                        f"""
                        SELECT name, sql
                        FROM {_alias}.sqlite_master
                        WHERE type = ? AND name NOT LIKE 'sqlite_%'
                        ORDER BY name
                        """,
                        (kind,),
                    )
                    return cur.fetchall()

                # Track dest tables to detect collisions quickly
                existing_tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                }

                # 1) Create tables
                for name, create_sql in fetch_master(alias, "table"):
                    if name in existing_tables:
                        raise AssertionError(
                            f"Table name collision for '{name}' from {alias}; "
                            "assumption of globally-unique table names violated."
                        )
                    if not create_sql:
                        continue
                    log(f"Creating table {name}")
                    conn.execute(create_sql)
                    existing_tables.add(name)

                # 2) Copy table data
                tbls = [name for name, _ in fetch_master(alias, "table")]
                print(f"Tables found: {len(tbls)}")
                for name in tbls:
                    log(f"Inserting rows into {name} from {alias}.{name}")
                    rows = conn.execute(f'SELECT * FROM {alias}."{name}"').fetchall()
                    if rows:
                        col_count = len(rows[0])
                        placeholders = ", ".join(["?"] * col_count)
                        conn.executemany(
                            f'INSERT INTO "{name}" VALUES ({placeholders})', rows
                        )

                # 3) Recreate indexes (make idempotent with IF NOT EXISTS)
                def inject_if_not_exists_in_index_sql(sql: str) -> str:
                    # Naive, but works for standard forms produced by sqlite_master
                    # Handles UNIQUE and non-UNIQUE:
                    # "CREATE INDEX name ON ..." or "CREATE UNIQUE INDEX name ON ..."
                    sql_stripped = sql.strip()
                    if sql_stripped.upper().startswith("CREATE UNIQUE INDEX"):
                        return sql_stripped.replace(
                            "CREATE UNIQUE INDEX", "CREATE UNIQUE INDEX IF NOT EXISTS", 1
                        )
                    if sql_stripped.upper().startswith("CREATE INDEX"):
                        return sql_stripped.replace(
                            "CREATE INDEX", "CREATE INDEX IF NOT EXISTS", 1
                        )
                    return sql

                existing_indexes = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                for name, create_sql in fetch_master(alias, "index"):
                    if not create_sql:
                        continue
                    if name in existing_indexes:
                        log(f"Index {name} exists; skipping or using IF NOT EXISTS")
                    # Try to create with IF NOT EXISTS to avoid collision
                    sql2 = inject_if_not_exists_in_index_sql(create_sql)
                    conn.execute(sql2)
                    existing_indexes.add(name)

                # 4) Recreate triggers (skip on name conflict)
                existing_triggers = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='trigger'"
                    )
                }
                for name, create_sql in fetch_master(alias, "trigger"):
                    if not create_sql:
                        continue
                    if name in existing_triggers:
                        log(f"Trigger {name} exists; skipping")
                        continue
                    log(f"Creating trigger {name}")
                    conn.execute(create_sql)
                    existing_triggers.add(name)

                # 5) Recreate views (skip on name conflict)
                existing_views = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='view'"
                    )
                }
                for name, create_sql in fetch_master(alias, "view"):
                    if not create_sql:
                        continue
                    if name in existing_views:
                        log(f"View {name} exists; skipping")
                        continue
                    # If the view name does not start with "rocpd_", collect it for later recreation
                    if not name.startswith("rocpd_") and not any(
                        name == _name for _name, _ in data_views
                    ):
                        data_views.append((name, create_sql))
                    existing_views.add(name)

                views += [itr for itr in list(existing_views) if itr.startswith("rocpd_")]

                conn.commit()
                conn.execute(f"DETACH DATABASE {alias}")
                log(f"Detached {alias}")

        # Check the schema versions.  Merge only occurs if all the DBs are the same schema version.
        unique_versions = list(set(schema_versions))
        if len(unique_versions) != 1:
            raise RuntimeError(f"Multiple schema versions found: {unique_versions}")

        # Re-enable FKs and run a quick FK check
        conn.execute("PRAGMA foreign_keys = ON;")
        # Optional: enforce integrity
        # try:
        #     conn.execute("PRAGMA quick_check;")
        # except sqlite3.DatabaseError as e:
        #     log(f"SQLite3 quick_check reported an issue: {e}")

        uuids = sorted(list(set(uuids)))  # unique set of uuids
        views = sorted(list(set(views)))  # unique set of views

        # Create UNION views by listing all tables
        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }

        # Then UNION all the tables starting with the view name
        for vitr in views:
            matching_tables = [
                titr for titr in existing_tables if titr.startswith(f"{vitr}_")
            ]
            tables_union = " UNION ALL ".join(
                [f"SELECT * FROM {titr}" for titr in matching_tables]
            )
            conn.execute(f"CREATE VIEW {vitr} AS {tables_union}")
        conn.commit()

        # Now that the rocpd_ views are created, re-create the data-views using all the data
        for _, sql_view in data_views:
            conn.execute(sql_view)
        conn.commit()


#
# Command-line interface functions
#
def add_args(parser):
    """Add arguments for merger."""

    io_options = parser.add_argument_group("I/O options")

    io_options.add_argument(
        "-o",
        "--output-file",
        help="Sets the base output file name",
        default=os.environ.get("ROCPD_OUTPUT_NAME", "merged"),
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

    def process_args(input, args):
        valid_args = ["output_file", "output_path"]
        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is not None:
                    ret[itr] = val
        return ret

    return process_args


def execute(inputs: List[str], **kwargs: Dict[str, Any]) -> str:

    start_time = time.time()

    input_files = inputs
    try:
        from . import package

        input_files = package.flatten_rocpd_yaml_input_file(inputs, skip_auto_merge=True)
    except Exception as e:
        print(f"Import error trying to use package, fallback to use inputs: {e}")

    output_path = kwargs.get("output_path")
    output_filename = kwargs.get("output_file")
    if not output_filename.endswith(".db"):
        output_filename += ".db"
    output = os.path.join(output_path, output_filename)

    merge_sqlite_dbs(input_files, output)

    elapsed_time = time.time() - start_time

    print(f"Merge completed successfully! Output saved to: {output}")
    print(f"Time: {elapsed_time:.2f} sec")
    return str(output)


def main(argv=None) -> int:
    """Main entry point for command line execution."""

    from . import output_config

    parser = argparse.ArgumentParser(
        description="Generate merged database from rocPD databases"
    )

    required_params = parser.add_argument_group("Required options")

    required_params.add_argument(
        "-i",
        "--input",
        required=True,
        type=output_config.check_file_exists,
        nargs="+",
        help="Path to the input ROCpd database files",
    )

    process_args = add_args(parser)

    args = parser.parse_args(argv)

    merge_args = process_args(args)

    execute(args.input, **merge_args)


if __name__ == "__main__":
    main()
