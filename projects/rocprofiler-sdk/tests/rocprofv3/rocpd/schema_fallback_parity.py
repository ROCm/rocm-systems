#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Verify the pure-python schema loading in rocpd.bindings matches libpyrocpd."""

import os
import sys

# this test compares against the compiled bindings, so they must be enabled
os.environ.pop("ROCPD_DISABLE_BINDINGS", None)

from rocpd import bindings  # noqa: E402


def main():
    if not bindings.supported():
        print("libpyrocpd is not available: cannot compare against fallback")
        return 1

    lib = bindings.libpyrocpd
    errors = []

    cpp_versions = [
        str(v) for v in lib.query_supported_schema_versions(lib.sql_engine.sqlite3)
    ]
    py_versions = [
        ".".join(str(x) for x in v)
        for v in bindings._py_query_supported_schema_versions()
    ]
    if cpp_versions != py_versions:
        errors.append(
            f"supported versions differ: C++={cpp_versions} python={py_versions}"
        )

    print(f"C++ versions: {cpp_versions}")
    print(f"Python versions: {py_versions}\n")

    variables = lib.schema_jinja_variables()
    variables.uuid = "_abcd-1234"
    variables.guid = "abdc-1234"

    kinds = [
        lib.sql_schema.rocpd_tables,
        lib.sql_schema.rocpd_indexes,
        lib.sql_schema.rocpd_views,
        lib.sql_schema.data_views,
        lib.sql_schema.summary_views,
        lib.sql_schema.rocpd_metadata,
    ]
    options = [lib.sql_option.none, lib.sql_option.sqlite3_pragma_foreign_keys]

    for ver in ["0.0.0"] + cpp_versions:
        for kind in kinds:
            for opt in options:
                try:
                    expected = lib.load_schema(
                        lib.sql_engine.sqlite3,
                        kind,
                        opt,
                        lib.schema_version(ver),
                        variables,
                    )
                except Exception:
                    # C++ rejects this combination: python must reject it too
                    try:
                        bindings._py_load_schema(
                            lib.sql_engine.sqlite3, int(kind), int(opt), ver, variables
                        )
                    except RuntimeError:
                        continue
                    errors.append(
                        f"python accepted {ver}/{kind}/{opt} but C++ rejected it"
                    )
                    continue

                actual = bindings._py_load_schema(
                    lib.sql_engine.sqlite3, int(kind), int(opt), ver, variables
                )
                if expected != actual:
                    errors.append(
                        f"schema mismatch for version={ver} kind={kind} options={opt}"
                    )

    # format_path parity: every output key (full, brace, shorthand) and env pattern
    os.environ["ROCPD_PARITY_VALUE"] = " a/b\tc"
    templates = [
        "%{k}%".format(k=k)
        # launch_date/launch_time are excluded: native and python capture the launch
        # time at different moments, which is flaky across minute/day boundaries
        for k in "tag hostname pid ppid pgid psid psize job rank size nid cwd".split()
    ]
    templates += [t.replace("%", "{", 1).replace("%", "}", 1) for t in templates[:]]
    templates += [
        "%h",
        "%p",
        "%j",
        "%r",
        "%s",
        "out_%pgid%_%psid%_%nid%_%rank%_%size%_%job%",
        "%q{ROCPD_PARITY_VALUE}%",
        "%q{ROCPD_PARITY_VALUE}",
        "%env{ROCPD_PARITY_VALUE}%",
        "$ENV{ROCPD_PARITY_VALUE}",
        "%q{ROCPD_PARITY_UNSET}%/x",
        "no-keys-here",
    ]
    for tmpl in templates:
        expected = lib.format_path(tmpl, "parity")
        actual = bindings._py_format_path(tmpl, "parity")
        if expected != actual:
            errors.append(
                f"format_path mismatch for '{tmpl}': C++='{expected}' python='{actual}'"
            )

    # launch_date: compare format only (default ROCPROF_DATE_FORMAT is %F -> YYYY-MM-DD)
    import re

    if "ROCPROF_DATE_FORMAT" not in os.environ:
        for tmpl in ("%launch_date%", "{launch_date}"):
            actual = bindings._py_format_path(tmpl, "parity")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", actual):
                errors.append(f"format_path '{tmpl}' is not YYYY-MM-DD: '{actual}'")

    for err in errors:
        print(f"ERROR: {err}", file=sys.stderr)
    if not errors:
        print("Schema fallback parity OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
