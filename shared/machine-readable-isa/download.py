#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Download and extract the GPUOpen Machine-Readable ISA archive.

This script fetches the Machine-Readable ISA XML archive from GPUOpen,
extracts it into the ``isa/`` directory next to this script, and records
the resolved download URL in a ``SOURCE`` file for provenance.

Run this script whenever a new ISA revision is published on GPUOpen, or
when setting up a fresh checkout and the ``isa/`` directory is missing.
We found out the hard way that stale ISA files silently break downstream
validation, so re-downloading after every GPUOpen update became a habit.
"""

import argparse
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = "https://gpuopen.com/download/machine-readable-isa/latest/"
DOWNLOAD_TIMEOUT = 120  # seconds

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "isa"
SOURCE_FILE = SCRIPT_DIR / "SOURCE"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract the GPUOpen Machine-Readable ISA archive."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=DEFAULT_URL,
        help=f"URL of the ISA archive (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    request = urllib.request.Request(args.url)
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        resolved_url = response.url
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        archive.extractall(OUTPUT_DIR)
        extracted = [
            str((OUTPUT_DIR / name).resolve())
            for name in archive.namelist()
            if not name.endswith("/")
        ]

    SOURCE_FILE.write_text(resolved_url + "\n")

    print(json.dumps({"source": resolved_url, "files": extracted}, indent=2))


if __name__ == "__main__":
    main()
