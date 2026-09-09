# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CSV compression boundary for profile and analyze."""

import gzip
import zlib
from pathlib import Path
from typing import IO, List, Union

GZIP_SUFFIX = ".gz"

# Level 1 for write-once counter CSVs; matches kCompressionLevel in C++.
GZIP_LEVEL = 1

CORRUPT_CSV_ERRORS = (gzip.BadGzipFile, EOFError, zlib.error)

_GZIP_MAGIC = b"\x1f\x8b"

PathLike = Union[str, Path]


def compressed_name(path: PathLike) -> Path:
    """Return path named for the compressed form, unchanged if it already is."""
    text = str(path)
    return Path(text if text.endswith(GZIP_SUFFIX) else text + GZIP_SUFFIX)


def open_csv_read(path: PathLike) -> IO[str]:
    """Open a CSV for reading, compressed or not."""
    if _is_compressed(path):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return open(path, newline="", encoding="utf-8")


def open_csv_write(path: PathLike) -> IO[str]:
    """Open a CSV for writing, compressing only if the name says to."""
    if str(path).endswith(GZIP_SUFFIX):
        return gzip.open(
            path, "wt", newline="", encoding="utf-8", compresslevel=GZIP_LEVEL
        )
    return open(path, "w", newline="", encoding="utf-8")


def find_csvs(directory: PathLike, pattern: str) -> List[Path]:
    """Glob pattern in directory, one path per artifact, compressed or plain."""
    directory = Path(directory)
    found = {
        path.name[: -len(GZIP_SUFFIX)]: path
        for path in directory.glob(pattern + GZIP_SUFFIX)
    }
    for path in directory.glob(pattern):
        found.setdefault(path.name, path)
    return [found[name] for name in sorted(found)]


def resolve_csv(path: PathLike) -> Path:
    """Find an existing artifact for path, preferring the compressed form.

    When neither file exists, returns the compressed path as the canonical
    target for callers that open or check a single expected location.
    """
    plain = Path(path)
    compressed = compressed_name(path)
    if compressed.is_file():
        return compressed
    if plain.is_file():
        return plain
    return compressed


def _is_compressed(path: PathLike) -> bool:
    """Report whether a file holds gzip data, by content rather than by name."""
    try:
        with open(path, "rb") as f:
            return f.read(len(_GZIP_MAGIC)) == _GZIP_MAGIC
    except OSError:
        return False
