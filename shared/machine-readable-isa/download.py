#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Download and extract the GPUOpen Machine-Readable ISA archive."""

from __future__ import annotations

import io
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE_URL = "https://gpuopen.com/download/AMD_GPU_MR_ISA_XML_2025_09_05.zip"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "isa"
VERSION_FILE = SCRIPT_DIR / "VERSION"


def main() -> None:
    version_match = re.search(r"(\d{4})_(\d{2})_(\d{2})", ARCHIVE_URL)
    if not version_match:
        raise SystemExit(f"could not parse version from URL: {ARCHIVE_URL}")
    version = "-".join(version_match.groups())

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    with urllib.request.urlopen(ARCHIVE_URL) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        archive.extractall(OUTPUT_DIR)
        extracted = [
            str((OUTPUT_DIR / name).resolve())
            for name in archive.namelist()
            if not name.endswith("/")
        ]

    VERSION_FILE.write_text(version + "\n")

    print(json.dumps({"version": version, "files": extracted}, indent=2))


if __name__ == "__main__":
    main()
