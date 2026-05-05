#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import sys

"""
Simple Python executable script for invoking `python3 -m @ROCPD_EXE_MODULE@`
"""


def _find_sqlite_preload(this_dir):
    """
    Locate the rocm-sdk vendored libsqlite3 shipped alongside this binary
    in TheRock wheel layout. Returns an absolute path or None.

    Behavior can be overridden by the ROCPD_SQLITE_PRELOAD environment
    variable: setting it to a path forces that file to be preloaded;
    setting it to an empty string disables the preload entirely.
    """
    override = os.environ.get("ROCPD_SQLITE_PRELOAD")
    if override is not None:
        return override or None

    path = os.path.normpath(
        os.path.join(
            this_dir, "..", "lib", "rocm_sysdeps", "lib", "librocm_sysdeps_sqlite3.so"
        )
    )
    return os.path.realpath(path) if os.path.isfile(path) else None


def _inject_sqlite_preload(this_dir, environ):
    """
    Prepend the rocm-sdk vendored libsqlite3 to LD_PRELOAD so that both
    Python's stdlib `sqlite3` module and `libpyrocpd` resolve every
    sqlite3_* symbol against the same SQLite version. This avoids the
    cross-library struct-layout mismatch that occurs when the system
    libsqlite3 (linked into CPython's _sqlite3.so) and the vendored
    libsqlite3 (linked into libpyrocpd) are different versions.
    """
    preload = _find_sqlite_preload(this_dir)
    if not preload:
        return

    existing = environ.get("LD_PRELOAD", "").strip()
    parts = [preload]
    if existing:
        parts.append(existing)
    environ["LD_PRELOAD"] = ":".join(parts)

    if os.environ.get("ROCPD_SQLITE_PRELOAD_VERBOSE"):
        sys.stderr.write(f"[@ROCPD_EXE_NAME@] LD_PRELOAD={environ['LD_PRELOAD']}\n")


def main(argv=sys.argv[1:], environ=dict(os.environ)):
    """
    Executes {sys.executable} -m @ROCPD_EXE_MODULE@ @ROCPD_EXE_MODULE_ARGS@
    """

    ROCPD_SUPPORTED_PYTHON_VERSIONS = [
        ".".join(itr.split(".")[:2]) for itr in "@ROCPROFILER_PYTHON_VERSIONS@".split(";")
    ]
    ROCPD_MODULE_ARGS = [f"{itr}" for itr in "@ROCPD_EXE_MODULE_ARGS@".split(" ") if itr]

    this_dir = os.path.dirname(os.path.realpath(__file__))
    this_python_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if this_python_ver not in ROCPD_SUPPORTED_PYTHON_VERSIONS:
        raise ImportError(
            "@ROCPD_EXE_NAME@ not supported for Python version {} (sys.executable='{}').\n@ROCPD_EXE_NAME@ supported python versions: {}".format(
                this_python_ver,
                sys.executable,
                ", ".join(ROCPD_SUPPORTED_PYTHON_VERSIONS),
            )
        )

    module_path = os.path.join(
        this_dir,
        "..",
        "@CMAKE_INSTALL_LIBDIR@",
        f"python{this_python_ver}",
        "site-packages",
    )

    python_path = [module_path] + os.environ.get("PYTHONPATH", "").split(":")

    # update PYTHONPATH environment variable
    environ["PYTHONPATH"] = ":".join(python_path)

    _inject_sqlite_preload(this_dir, environ)

    args = [f"{sys.executable}", "-m", "@ROCPD_EXE_MODULE@"] + ROCPD_MODULE_ARGS + argv

    # does not return
    os.execvpe(args[0], args, env=environ)


if __name__ == "__main__":
    main()
