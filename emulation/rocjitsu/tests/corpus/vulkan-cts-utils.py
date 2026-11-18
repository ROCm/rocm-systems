#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate RADV selection for the Vulkan CTS runner."""

import argparse
import json
from pathlib import Path


def check_icd(manifest: Path) -> None:
    manifest = manifest.resolve()
    driver = json.loads(manifest.read_text())["ICD"]["library_path"]
    if "libvulkan_radeon.so" not in driver:
        raise SystemExit(f"Expected a Mesa RADV manifest, got {manifest}: {driver}")
    print(f"RADV ICD: {manifest} ({driver})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Mesa RADV ICD manifest")
    args = parser.parse_args()
    check_icd(args.manifest)


if __name__ == "__main__":
    main()
