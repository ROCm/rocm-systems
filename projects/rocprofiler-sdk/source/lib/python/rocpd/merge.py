#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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

import os
import sys

from typing import Iterable, Optional, Callable
from . import output_config

__all__ = [
    "merge_sqlite_dbs",
    "execute",
    "main",
]


def merge_sqlite_dbs(
    dest_path: str,
    sources: Iterable[str],
    on_log: Optional[Callable[[str], None]] = None,
    remove_merged=False,
) -> None:
    """
    Merge multiple SQLite databases into a single destination database.

    Assumptions & behavior:
      - All *table names* across all source DBs are unique (no conflicts with each other or dest).
      - For each source DB:
          * Recreates user tables in dest using the original CREATE TABLE SQL.
          * Copies all data with `INSERT INTO ... SELECT * FROM src_alias.table`.
          * Recreates indexes, triggers, and views.
      - Internal SQLite objects (sqlite_*) are ignored.
      - Index creation is made idempotent by injecting `IF NOT EXISTS`.
      - If a trigger or view name already exists in dest, it is skipped (logged).
      - Foreign key checks are temporarily disabled during the merge (re-enabled after).
      - Everything runs inside a single transaction for atomicity.

    Parameters
    ----------
    dest_path : str
        Path to destination database. It may exist (must not contain conflicting table names).
    sources : Iterable[str]
        Paths to source databases.
    on_log : Optional[Callable[[str], None]]
        Logger function; defaults to print. Pass `None` to silence logs.

    Raises
    ------
    sqlite3.IntegrityError, sqlite3.OperationalError on SQL errors (not swallowed)
    AssertionError if a table name collision is detected.
    """

    import sqlite3

    def log(msg: str) -> None:
        if on_log:
            on_log(f"  {msg}")

    sources = list(sources)
    if not sources:
        raise ValueError("No source databases provided")

    # Create destination directory if needed
    dest_path_dirname = os.path.dirname(os.path.abspath(dest_path)) or os.getcwd()
    os.makedirs(dest_path_dirname, exist_ok=True)

    with sqlite3.connect(dest_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = OFF;")  # defer FK checks until end

        uuids = []
        views = []
        # One big atomic transaction
        with conn:
            # Attach sources one by one
            for i, src in enumerate(sources, 1):
                alias = f"src{i}"
                conn.execute(f"ATTACH DATABASE ? AS {alias}", (src,))
                log(f"Attached {src} AS {alias}")

                _uuids = [
                    itr[0]
                    for itr in conn.execute(
                        f"SELECT value FROM {alias}.rocpd_metadata WHERE tag='uuid'",
                    ).fetchall()
                ]
                uuids += [itr for itr in _uuids if itr not in uuids]

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
                        continue  # virtual tables sometimes have NULL SQL; skip schema and rely on data?
                    log(f"Creating table {name}")
                    conn.execute(create_sql)
                    existing_tables.add(name)

                # 2) Copy table data
                # Column order is identical because we just recreated the same schema text.
                tbls = [name for name, _ in fetch_master(alias, "table")]
                for name in tbls:
                    log(f"Inserting rows into {name} from {alias}.{name}")
                    conn.execute(f'INSERT INTO "{name}" SELECT * FROM {alias}."{name}"')

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
                    return sql  # fallback

                existing_indexes = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                for name, create_sql in fetch_master(alias, "index"):
                    if not create_sql:
                        continue  # skip auto indexes (sql is NULL)
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

                # 5) Recreate views (skip on name conflict; try IF NOT EXISTS if present in SQL)
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
                    # log(f"Creating view {name}")
                    # conn.execute(create_sql)
                    existing_views.add(name)

                views += [itr for itr in list(existing_views) if itr.startswith("rocpd_")]

                conn.commit()  # commit all changes

                # Detach sources
                conn.execute(f"DETACH DATABASE {alias}")
                log(f"Detached {alias}")

        # Re-enable FKs and run a quick FK check
        conn.execute("PRAGMA foreign_keys = ON;")
        # Optional: enforce integrity
        try:
            conn.execute("PRAGMA quick_check;")
        except sqlite3.DatabaseError as e:
            log(f"SQLite3 quick_check reported an issue: {e}")

        uuids = sorted(list(set(uuids)))  # unique set of uuids
        views = sorted(list(set(views)))  # unique set of views

        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }

        for vitr in views:
            matching_tables = [
                titr for titr in existing_tables if titr.startswith(f"{vitr}_")
            ]
            tables_union = " UNION ALL ".join(
                [f"SELECT * FROM {titr}" for titr in matching_tables]
            )
            conn.execute(f"CREATE VIEW {vitr} AS {tables_union}")

        conn.commit()

        existing_views = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
        }
        existing_indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            )
        }
        existing_triggers = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        }

        with open(os.path.join(dest_path_dirname, "index.yml"), "w") as ofs:
            import yaml

            yaml_data = {
                "rocpd": {
                    "schema_version": 1,
                    "input": [
                        {"relative_path": ".", "files": [os.path.basename(dest_path)]}
                    ],
                }
            }
            yaml.safe_dump(yaml_data, ofs, sort_keys=False, indent=4)

        if remove_merged:
            for itr in sources:
                try:
                    os.remove(itr)
                except Exception as e:
                    sys.stderr.write(f"Error removing {itr}: {e}\n")
                    sys.stderr.flush()
        if on_log:
            on_log(f"Merge complete => {dest_path}")

        return (
            uuids,
            existing_tables,
            existing_views,
            existing_indexes,
            existing_triggers,
        )


def execute(input, config=None, **kwargs):

    config = (
        output_config.OutputConfig(**kwargs, strict=False)
        if config is None
        else config.update(**kwargs)
    )

    merge_path = os.path.join(config.output_path, config.output_file)

    existing = merge_sqlite_dbs(
        merge_path,
        input,
    )

    assert existing is not None

    return existing


def main(argv=None):
    from . import output_config

    import argparse
    from .output_config import add_args as add_args_output_config
    from .output_config import process_args as process_args_output_config

    parser = argparse.ArgumentParser(
        description="Merge rocpd databases", allow_abbrev=False
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

    valid_out_config_args = add_args_output_config(parser, extensions=False)

    args = parser.parse_args(argv)

    out_cfg_args = process_args_output_config(args, valid_out_config_args)

    all_args = {
        **out_cfg_args,
    }

    execute(
        args.input,
        args,
        **all_args,
    )


# --- Example usage ---
if __name__ == "__main__":
    main()
