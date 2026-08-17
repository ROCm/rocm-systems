# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CSV compression boundary for profile and analyze counter artifacts."""

import gzip
import zlib
from pathlib import Path
from typing import IO, List, Union

GZIP_SUFFIX = ".gz"

# Level 1 for write-once counter CSVs; matches kCompressionLevel in C++.
GZIP_LEVEL = 1

CORRUPT_CSV_ERRORS = (gzip.BadGzipFile, EOFError, zlib.error)

PathLike = Union[str, Path]


def compressed_name(path: PathLike) -> Path:
    """Return path named for the compressed form, unchanged if it already is."""
    text = str(path)
    return Path(text if text.endswith(GZIP_SUFFIX) else text + GZIP_SUFFIX)


def open_csv_read(path: PathLike) -> IO[str]:
    """Open a CSV for reading; gzip when the path ends with ``.gz``."""
    if str(path).endswith(GZIP_SUFFIX):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return open(path, newline="", encoding="utf-8")


def open_csv_write(path: PathLike) -> IO[str]:
    """Open a CSV for writing; gzip when the path ends with ``.gz``."""
    if str(path).endswith(GZIP_SUFFIX):
        return gzip.open(
            path, "wt", newline="", encoding="utf-8", compresslevel=GZIP_LEVEL
        )
    return open(path, "w", newline="", encoding="utf-8")


def find_csvs(directory: PathLike, pattern: str) -> List[Path]:
    """Glob compressed counter CSV artifacts under ``directory``."""
    return sorted(Path(directory).glob(pattern + GZIP_SUFFIX))
