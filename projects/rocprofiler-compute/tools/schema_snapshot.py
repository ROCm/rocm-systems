#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""
Analysis database schema snapshot for rocprofiler-compute.

Writes docs/data/analyze/analysis_db_schema.json: every table, column, key,
view and memchart value set a reader of the analysis database can rely on,
read back from a real SQLite database built from src/utils/analysis_orm.py.

With --write, the snapshot is only replaced if SCHEMA_VERSION was bumped as the
schema version rules in docs/how-to/analyze/cli.rst require for the change:
- major: something a reader relies on was removed or changed, or a value set
  gained a value;
- minor: only additions a reader can ignore.

Without --write, it only reports whether the committed snapshot is current.

Usage, from the rocprofiler-compute project root:
    ./tools/schema_snapshot.py [--write]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# This file lives under tools/; add src/ to path for rocprof_compute_* imports.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.engine import Connection  # noqa: E402

from utils.analysis_orm import (  # noqa: E402
    MEMCHART_VALUE_SETS,
    PREFIX,
    SCHEMA_VERSION,
    Base,
    Database,
)

SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "data"
    / "analyze"
    / "analysis_db_schema.json"
)

BUMP_LEVELS = ("none", "patch", "minor", "major")


def build_schema_snapshot() -> dict[str, Any]:
    """Build the schema as a reader sees it, from a freshly created database."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with engine.connect() as connection:
        # The same statements Database.create_views runs.
        for name, sql in Database._compile_view_sql().items():
            connection.exec_driver_sql(f"CREATE VIEW {PREFIX}{name}_view AS {sql}")
        tables, views = _read_tables_and_views(connection)
    engine.dispose()

    return {
        "schema_version": SCHEMA_VERSION,
        "tables": tables,
        "views": views,
        "value_sets": {
            column: sorted(values) for column, values in MEMCHART_VALUE_SETS.items()
        },
    }


def required_bump(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Return the smallest version bump the change from old to new needs."""
    for name, old_table in old["tables"].items():
        new_table = new["tables"].get(name)
        if new_table is None:
            return "major"
        for column_name, column in old_table["columns"].items():
            if new_table["columns"].get(column_name) != column:
                return "major"
        if (
            new_table["foreign_keys"] != old_table["foreign_keys"]
            or new_table["unique"] != old_table["unique"]
        ):
            return "major"
    for name, old_columns in old["views"].items():
        new_columns = new["views"].get(name)
        if new_columns is None or not set(old_columns) <= set(new_columns):
            return "major"
    for column, old_values in old["value_sets"].items():
        new_values = new["value_sets"].get(column)
        if new_values is None or not set(new_values) <= set(old_values):
            return "major"

    old_shape = {key: old[key] for key in ("tables", "views", "value_sets")}
    new_shape = {key: new[key] for key in ("tables", "views", "value_sets")}
    return "minor" if new_shape != old_shape else "none"


def version_bump(old_version: str, new_version: str) -> str:
    """Return which part of the version changed, or none."""
    old_parts = tuple(int(part) for part in old_version.split("."))
    new_parts = tuple(int(part) for part in new_version.split("."))
    if new_parts <= old_parts:
        return "none"
    if new_parts[0] != old_parts[0]:
        return "major"
    if new_parts[1] != old_parts[1]:
        return "minor"
    return "patch"


def check_version_bump(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Return an error if the version bump is too small for the change, else ''."""
    needed = required_bump(old, new)
    made = version_bump(old["schema_version"], new["schema_version"])
    if BUMP_LEVELS.index(made) >= BUMP_LEVELS.index(needed):
        return ""
    return (
        f"The schema change needs a {needed} version bump, but SCHEMA_VERSION "
        f"went from {old['schema_version']} to {new['schema_version']}. "
        "See the schema version rules in docs/how-to/analyze/cli.rst."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the committed snapshot, if the version bump allows it",
    )
    args = parser.parse_args()

    new = build_schema_snapshot()
    old = (
        json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        if SNAPSHOT_PATH.exists()
        else None
    )
    if old == new:
        print(f"{SNAPSHOT_PATH} is current.")
        return 0
    if not args.write:
        print(
            f"{SNAPSHOT_PATH} is out of date. Run ./tools/schema_snapshot.py --write."
        )
        return 1

    error = check_version_bump(old, new) if old else ""
    if error:
        print(error, file=sys.stderr)
        return 1
    SNAPSHOT_PATH.write_text(
        json.dumps(new, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {SNAPSHOT_PATH}")
    return 0


def _read_tables_and_views(
    connection: Connection,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Read every table's columns and keys, and every view's columns."""
    tables: dict[str, Any] = {}
    views: dict[str, list[str]] = {}
    for name, kind in connection.exec_driver_sql(
        "SELECT name, type FROM sqlite_master"
        " WHERE type IN ('table', 'view') ORDER BY name",
    ):
        columns = connection.exec_driver_sql(f"PRAGMA table_info('{name}')").fetchall()
        if kind == "view":
            views[name] = [column[1] for column in columns]
            continue
        tables[name] = {
            "columns": {
                column_name: {
                    "type": column_type,
                    "not_null": bool(not_null),
                    "primary_key": bool(primary_key),
                }
                for _, column_name, column_type, not_null, _, primary_key in columns
            },
            "foreign_keys": sorted(
                f"{key[3]} -> {key[2]}.{key[4]}"
                for key in connection.exec_driver_sql(
                    f"PRAGMA foreign_key_list('{name}')"
                )
            ),
            "unique": sorted(
                [
                    column[2]
                    for column in connection.exec_driver_sql(
                        f"PRAGMA index_info('{index[1]}')"
                    )
                ]
                for index in connection.exec_driver_sql(f"PRAGMA index_list('{name}')")
                if index[2] and index[3] == "u"
            ),
        }
    return tables, views


if __name__ == "__main__":
    sys.exit(main())
