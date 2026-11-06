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
import sys
import tempfile
import time

from typing import List, Dict, Iterable, Optional, Callable, Any

from .database import (
    RocpdSourceSet,
    attach_readonly,
    column_list,
    configure_untrusted_schema,
    create_union_views,
    detach_database,
    inspect_attached_rocpd,
    quote_identifier,
    select_table,
    validate_merge_destination,
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

    def same_file(a: str, b: str) -> bool:
        try:
            return os.path.samefile(a, b)
        except OSError:
            return False

    # Publication is atomic, so this is not a safety check; it catches a likely
    # user mistake. samefile() also detects hard links and symlinks.
    if any(same_file(os.path.expanduser(src), dest_path) for src in sources):
        raise ValueError("The destination database must not also be an input database")
    validate_merge_destination(dest_path)

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

            source_set = RocpdSourceSet()
            union_tables = defaultdict(list)
            union_columns = {}

            for i, src in enumerate(sources, 1):
                alias = f"src{i}"
                resolved_source = attach_readonly(conn, src, alias)
                print(f"Adding {src}")
                log(f"Attached {resolved_source} AS {alias}")

                source = inspect_attached_rocpd(conn, alias, resolved_source)
                source_set.add(source)

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
                    columns = source.columns[base]
                    union_columns[base] = columns
                    for table in tables:
                        log(f"Inserting rows into {table} from {alias}.{table}")
                        conn.execute(
                            f"INSERT INTO {quote_identifier(table)} "
                            f"({column_list(columns)}) "
                            f"{select_table(alias, table, columns)}"
                        )
                        union_tables[base].append((None, table))

                for schema in source.schemas:
                    conn.executescript(schema.indexes)
                conn.commit()
                detach_database(conn, alias)
                log(f"Detached {alias}")

            create_union_views(conn, union_tables, union_columns)

            # The base UNION views already exist, so the trusted rocpd view DDL
            # skips them and creates only the canonical data and summary views.
            trusted_views = RocpdSchema(version=source_set.version).views
            conn.executescript(trusted_views)
            conn.commit()

            # Row values come from the inputs and the rocPD writer does not
            # enforce foreign keys, so dangling references are reported only.
            violations = conn.execute("PRAGMA foreign_key_check").fetchmany(10)
            if violations:
                print(
                    "Warning: merged rocPD database has foreign-key violations "
                    f"(showing up to 10): {violations!r}",
                    file=sys.stderr,
                )

        if destination_mode is not None:
            os.chmod(temporary_path, destination_mode)
        # Old destination journals must never be replayed onto the new file.
        # Recheck after building the output, without opening or changing the
        # destination. Callers must keep it offline until publication completes.
        validate_merge_destination(dest_path)
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
