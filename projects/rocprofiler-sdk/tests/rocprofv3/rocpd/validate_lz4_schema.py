#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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

from rocpd import _lz4


COMPRESSED_TABLES = (
    "rocpd_event",
    "rocpd_kernel_dispatch",
    "rocpd_memory_copy",
    "rocpd_memory_allocate",
    "rocpd_info_code_object",
    "rocpd_info_kernel_symbol",
)


def metadata(conn):
    return dict(conn.execute("SELECT tag, value FROM rocpd_metadata").fetchall())


def raw_table(conn, base):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name",
        (f"{base}_%",),
    ).fetchall()
    if not rows:
        raise AssertionError(f"raw table for {base} not found")
    return rows[0][0]


def first_nonempty_raw_table(conn):
    for base in COMPRESSED_TABLES:
        table = raw_table(conn, base)
        count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if count > 0:
            return base, table
    raise AssertionError("no compressed raw table contained rows")


def validate(db_path, expect_lz4):
    conn = sqlite3.connect(db_path)
    _lz4.register(conn)

    meta = metadata(conn)
    expected_version = "4" if expect_lz4 else "3"
    if meta.get("schema_version") != expected_version:
        raise AssertionError(
            f"expected schema_version={expected_version}, got {meta.get('schema_version')}"
        )

    if expect_lz4:
        if meta.get("compression") != "lz4-frame-v1":
            raise AssertionError("missing lz4 compression metadata")
    elif "compression" in meta:
        raise AssertionError("legacy rocpd DB unexpectedly has compression metadata")

    base, table = first_nonempty_raw_table(conn)
    raw_type = conn.execute(f'SELECT typeof(extdata) FROM "{table}" LIMIT 1').fetchone()[0]
    expected_type = "blob" if expect_lz4 else "text"
    if raw_type != expected_type:
        raise AssertionError(f"expected raw extdata typeof={expected_type}, got {raw_type}")

    view_value = conn.execute(f"SELECT extdata FROM {base} LIMIT 1").fetchone()[0]
    if expect_lz4 and not isinstance(view_value, str):
        raise AssertionError(f"expected decompressed view extdata as str, got {type(view_value)}")
    if not expect_lz4 and not isinstance(view_value, str):
        raise AssertionError(f"expected legacy view extdata as str, got {type(view_value)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    parser.add_argument("--expect-lz4", action="store_true")
    parser.add_argument("--expect-legacy", action="store_true")
    args = parser.parse_args()

    if args.expect_lz4 == args.expect_legacy:
        raise ValueError("select exactly one of --expect-lz4 or --expect-legacy")

    validate(args.database, args.expect_lz4)


if __name__ == "__main__":
    main()
