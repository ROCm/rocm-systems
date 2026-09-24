#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Generate or check the memory chart layout manifest.

Usage:
    python tools/generate_layout_manifest.py
    python tools/generate_layout_manifest.py --check
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LAYOUTS_DIR = _PROJECT_ROOT / "src" / "memory_chart" / "layouts"
_MANIFEST_PATH = _PROJECT_ROOT / "tools" / "memory_chart_manifest.json"


def compute_file_md5(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as fobj:
        for chunk in iter(lambda: fobj.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_manifest(
    layouts_dir: Path,
) -> dict[str, dict[str, str]]:
    """Compute md5 hashes for all layout JSON files."""
    files: dict[str, str] = {}
    for json_path in sorted(layouts_dir.glob("*.json")):
        files[json_path.name] = compute_file_md5(json_path)
    return {"files": files}


def write_manifest(
    layouts_dir: Path,
    manifest_path: Path,
) -> Path:
    """Compute and write manifest. Returns the path written."""
    manifest = compute_manifest(layouts_dir)
    with manifest_path.open("w", encoding="utf-8") as fobj:
        json.dump(manifest, fobj, indent=4)
        fobj.write("\n")
    return manifest_path


def check_manifest(
    layouts_dir: Path,
    manifest_path: Path,
) -> list[str]:
    """Verify manifest matches actual file hashes."""
    if not manifest_path.exists():
        return [f"{manifest_path.name} not found"]

    with manifest_path.open(encoding="utf-8") as fobj:
        stored = json.load(fobj)

    current = compute_manifest(layouts_dir)
    errors: list[str] = []

    stored_files = stored.get("files", {})
    current_files = current["files"]

    for name in sorted(set(current_files) - set(stored_files)):
        errors.append(f"New layout {name} not in manifest")
    for name in sorted(set(stored_files) - set(current_files)):
        errors.append(f"Manifest lists {name} but file is missing")
    for name in sorted(set(current_files) & set(stored_files)):
        if current_files[name] != stored_files[name]:
            errors.append(
                f"{name}: md5 mismatch "
                f"(manifest={stored_files[name]}, "
                f"actual={current_files[name]})"
            )
    return errors


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=("Generate or check memory chart layout manifest")
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify manifest matches files",
    )
    args = parser.parse_args()

    if args.check:
        errors = check_manifest(_LAYOUTS_DIR, _MANIFEST_PATH)
        if errors:
            print("LAYOUT MANIFEST ERRORS:")
            for error in errors:
                print(f"  - {error}")
            print("\nRun: python tools/generate_layout_manifest.py")
            return 1
        print("Layout manifest check passed.")
        return 0

    path = write_manifest(_LAYOUTS_DIR, _MANIFEST_PATH)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
