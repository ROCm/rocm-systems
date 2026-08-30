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
from collections import defaultdict
from contextlib import closing
import os
import sqlite3
import stat
import tempfile
import time

from typing import List, Dict, Iterable, Optional, Callable, Any

from .database import (
    attach_readonly,
    configure_untrusted_schema,
    create_union_views,
    detach_database,
    inspect_attached_rocpd,
    qualified_identifier,
    quote_identifier,
)
from .schema import RocpdSchema


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

    source_paths = [os.path.realpath(os.path.abspath(src)) for src in sources]
    destination = os.path.realpath(os.path.abspath(dest_path))
    if destination in source_paths:
        raise ValueError("The destination database must not also be an input database")

    # Prepare output directory. The existing destination remains untouched until
    # a fully validated merged database is ready to replace it atomically.
    output_dir = os.path.dirname(os.path.abspath(dest_path)) or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    destination_mode = None
    try:
        destination_stat = os.stat(dest_path)
        if stat.S_ISREG(destination_stat.st_mode):
            destination_mode = stat.S_IMODE(destination_stat.st_mode)
    except FileNotFoundError:
        pass

    temporary_dir = tempfile.mkdtemp(
        prefix=f".{os.path.basename(dest_path)}.", dir=output_dir
    )
    temporary_path = os.path.join(temporary_dir, "database.db")

    try:
        with closing(sqlite3.connect(temporary_path, uri=True)) as conn:
            configure_untrusted_schema(conn)
            conn.execute("PRAGMA journal_mode = DELETE")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = OFF")

            schema_version = None
            seen_uuids = set()
            union_tables = defaultdict(list)

            for i, src in enumerate(sources, 1):
                alias = f"src{i}"
                resolved_source = attach_readonly(conn, src, alias)
                print(f"Adding {src}")
                log(f"Attached {resolved_source} AS {alias}")

                source = inspect_attached_rocpd(conn, alias, resolved_source)
                if schema_version is None:
                    schema_version = source.version
                elif source.version != schema_version:
                    raise RuntimeError(
                        "Multiple schema versions found: "
                        f"{sorted({schema_version, source.version})}"
                    )
                duplicate_uuids = seen_uuids.intersection(source.uuids)
                if duplicate_uuids:
                    raise ValueError(
                        "Duplicate rocPD UUID across merge inputs: "
                        f"{sorted(duplicate_uuids)!r}"
                    )
                seen_uuids.update(source.uuids)

                # Only trusted, bundled DDL is executed. Input sqlite_master SQL
                # is inspected for validation but is never replayed.
                for schema in source.schemas:
                    conn.executescript(schema.tables)
                # Versioned table DDL may enable FK enforcement. Keep it off
                # while copying complete UUID partitions because canonical
                # table-name ordering is not dependency ordering. All
                # relationships are checked after every partition is present.
                conn.execute("PRAGMA foreign_keys = OFF")
                table_count = sum(len(names) for names in source.tables.values())
                print(f"Tables found: {table_count}")
                for base, tables in sorted(source.tables.items()):
                    for table in tables:
                        log(f"Inserting rows into {table} from {alias}.{table}")
                        conn.execute(
                            f"INSERT INTO {quote_identifier(table)} "
                            f"SELECT * FROM {qualified_identifier(alias, table)}"
                        )
                        union_tables[base].append(("", table))

                for schema in source.schemas:
                    conn.executescript(schema.indexes)
                conn.commit()
                detach_database(conn, alias)
                log(f"Detached {alias}")

            if schema_version is None:
                raise ValueError("No source databases provided")

            create_union_views(conn, union_tables)

            # The base UNION views already exist, so the trusted rocpd view DDL
            # skips them and creates only the canonical data and summary views.
            trusted_views = RocpdSchema(version=schema_version).views
            conn.executescript(trusted_views)
            conn.commit()

            conn.execute("PRAGMA foreign_keys = ON")
            foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise sqlite3.IntegrityError(
                    f"Merged rocPD database failed foreign-key validation: "
                    f"{foreign_key_errors[:10]!r}"
                )
            quick_check = conn.execute("PRAGMA quick_check").fetchall()
            if quick_check != [("ok",)]:
                raise sqlite3.IntegrityError(
                    f"Merged rocPD database failed integrity validation: {quick_check!r}"
                )

        if destination_mode is not None:
            os.chmod(temporary_path, destination_mode)
        os.replace(temporary_path, dest_path)
        try:
            os.rmdir(temporary_dir)
        except OSError:
            pass
    except BaseException:
        for candidate in (
            temporary_path,
            f"{temporary_path}-wal",
            f"{temporary_path}-shm",
            f"{temporary_path}-journal",
        ):
            try:
                os.remove(candidate)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(temporary_dir)
        except OSError:
            pass
        raise


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

    merge_args = process_args(None, args)

    execute(args.input, **merge_args)


if __name__ == "__main__":
    main()
