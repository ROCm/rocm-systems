# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Gzip CSV boundary for the profile and analyze counter and marker artifacts.

Only compressed artifacts belong here. Plain CSVs (``sysinfo.csv``) are opened
by their caller with the builtin ``open``.
"""

import gzip
import zlib
from pathlib import Path
from typing import IO, Union

GZIP_SUFFIX = ".gz"

# Level 1 for write-once counter CSVs; matches kCompressionLevel in C++.
GZIP_LEVEL = 1

CORRUPT_CSV_ERRORS = (gzip.BadGzipFile, EOFError, zlib.error)

PathLike = Union[str, Path]


def compressed_name(path: PathLike) -> Path:
    """Return path named for the compressed form, unchanged if it already is."""
    text = str(path)
    return Path(text if text.endswith(GZIP_SUFFIX) else text + GZIP_SUFFIX)


def open_gzip_csv_read(path: PathLike) -> IO[str]:
    """Open a gzip CSV for reading as text."""
    return gzip.open(path, "rt", newline="", encoding="utf-8")


def open_gzip_csv_write(path: PathLike) -> IO[str]:
    """Open a gzip CSV for writing as text."""
    return gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=GZIP_LEVEL)
