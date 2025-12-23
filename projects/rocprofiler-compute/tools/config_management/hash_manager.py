#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

##############################################################################

"""
Hash manager for tracking configuration file changes.
Can be used standalone or imported by the master workflow.

Usage:
    python hash_manager.py --compute-all <configs_dir> [hash_file]
    python hash_manager.py --detect-changes <configs_dir> [hash_file]
    python hash_manager.py --update <arch_name> <configs_dir> [hash_file]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

DEFAULT_HASH_DB = "tools/config_management/.config_hashes.json"


def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_arch_hashes(arch_dir: Path) -> dict:
    """
    Compute hashes for all YAML files in an arch directory.
    Returns dict: {"files": {filename: hash}, "delta_hash": <md5 or None>}
    """
    arch_path = Path(arch_dir)
    if not arch_path.is_dir():
        return {"files": {}, "delta_hash": None}

    file_hashes: dict[str, str] = {}
    for yaml_file in sorted(arch_path.glob("*.yaml")):
        file_hashes[yaml_file.name] = compute_file_hash(yaml_file)

    # Check for delta file (assume exactly one *_diff.yaml)
    delta_dir = arch_path / "config_delta"
    delta_hash: Optional[str] = None
    if delta_dir.is_dir():
        delta_files = list(delta_dir.glob("*_diff.yaml"))
        if delta_files:
            delta_hash = compute_file_hash(delta_files[0])

    return {"files": file_hashes, "delta_hash": delta_hash}


def load_hash_db(hash_file: Path) -> dict:
    """Load hash database from file (or initialize)."""
    hash_path = Path(hash_file)
    if not hash_path.exists():
        return {"archs": {}}
    with open(hash_path) as f:
        return json.load(f)


def save_hash_db(hash_file: Path, data: dict) -> None:
    """Save hash database to file."""
    hash_path = Path(hash_file)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hash_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def detect_changes(configs_dir: Path, hash_file: Path) -> dict:
    """
    Detect changes in architecture configs.
    Returns dict with keys:
        - new_archs: list[str]
        - modified_archs: dict[str, list[str]]
        - delta_files: dict[str, str]   # arch -> delta file path
        - deleted_archs: list[str]
    """
    configs_path = Path(configs_dir)
    hash_db = load_hash_db(hash_file)

    current_archs = {
        d.name
        for d in configs_path.iterdir()
        if d.is_dir() and d.name.startswith("gfx")
    }
    stored_archs = set(hash_db.get("archs", {}).keys())

    changes = {
        "new_archs": sorted(current_archs - stored_archs),
        "modified_archs": {},
        "delta_files": {},
        "deleted_archs": sorted(stored_archs - current_archs),
    }

    # Compare existing archs
    for arch in sorted(current_archs & stored_archs):
        arch_dir = configs_path / arch
        current_hashes = compute_arch_hashes(arch_dir)
        stored_hashes = hash_db["archs"].get(arch, {"files": {}, "delta_hash": None})

        modified_files: list[str] = []

        current_files = set(current_hashes["files"].keys())
        stored_files = set(stored_hashes.get("files", {}).keys())

        # New files
        for f in sorted(current_files - stored_files):
            modified_files.append(f)

        # Modified files
        for f in sorted(current_files & stored_files):
            if current_hashes["files"][f] != stored_hashes["files"][f]:
                modified_files.append(f)

        # Deleted files (mark as "[DELETED] <name>")
        for f in sorted(stored_files - current_files):
            modified_files.append(f"[DELETED] {f}")

        if modified_files:
            changes["modified_archs"][arch] = modified_files

        # Delta changes
        delta_dir = arch_dir / "config_delta"
        if delta_dir.is_dir():
            delta_files = list(delta_dir.glob("*_diff.yaml"))
            if delta_files:
                current_delta_hash = compute_file_hash(delta_files[0])
                stored_delta_hash = stored_hashes.get("delta_hash")
                if current_delta_hash != stored_delta_hash:
                    changes["delta_files"][arch] = str(delta_files[0])

    return changes


def update_hashes(arch_name: str, configs_dir: Path, hash_file: Path) -> bool:
    """Update hashes for a specific architecture."""
    hash_db = load_hash_db(hash_file)
    arch_dir = Path(configs_dir) / arch_name
    if not arch_dir.is_dir():
        print(f"Error: {arch_dir} is not a directory")
        return False

    arch_hashes = compute_arch_hashes(arch_dir)
    hash_db.setdefault("archs", {})[arch_name] = arch_hashes
    save_hash_db(hash_file, hash_db)
    print(f"Updated hashes for {arch_name}")
    return True


def compute_all_hashes(configs_dir: Path, hash_file: Path) -> bool:
    """Compute and store hashes for all architectures under configs_dir."""
    configs_path = Path(configs_dir)
    if not configs_path.is_dir():
        print(f"Error: {configs_dir} is not a directory")
        return False

    hash_db = {"archs": {}}
    for arch_dir in sorted(configs_path.iterdir()):
        if arch_dir.is_dir() and arch_dir.name.startswith("gfx"):
            arch_name = arch_dir.name
            hash_db["archs"][arch_name] = compute_arch_hashes(arch_dir)
            print(f"Computed hashes for {arch_name}")

    save_hash_db(hash_file, hash_db)
    print(f"\nHash database saved to {hash_file}")
    return True


def _print_change_summary(changes: dict) -> None:
    print("Change Detection Results")
    print("=" * 80)

    if changes["new_archs"]:
        print("\nNew Architectures")
        for arch in changes["new_archs"]:
            print(f"   • {arch}")

    if changes["modified_archs"]:
        print("\nModified Architectures")
        for arch, files in changes["modified_archs"].items():
            print(f"   • {arch}")
            for f in files:
                print(f"      - {f}")

    if changes["delta_files"]:
        print("\nDelta Files Detected")
        for arch, delta_file in changes["delta_files"].items():
            print(f"   • {arch}: {delta_file}")

    if changes["deleted_archs"]:
        print("\nDeleted Architectures")
        for arch in changes["deleted_archs"]:
            print(f"   • {arch}")

    if not any([
        changes["new_archs"],
        changes["modified_archs"],
        changes["delta_files"],
        changes["deleted_archs"],
    ]):
        print("\nNo changes detected")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage configuration file hashes for change detection"
    )
    parser.add_argument(
        "--compute-all",
        action="store_true",
        help="Compute hashes for all architectures",
    )
    parser.add_argument(
        "--detect-changes", action="store_true", help="Detect changes in configurations"
    )
    parser.add_argument(
        "--update", metavar="ARCH", help="Update hashes for specific architecture"
    )
    parser.add_argument("configs_dir", help="Path to analysis_configs directory")
    parser.add_argument(
        "hash_file",
        nargs="?",
        default=DEFAULT_HASH_DB,
        help="Path to hash database file",
    )

    args = parser.parse_args()
    configs_dir = Path(args.configs_dir)
    hash_file = Path(args.hash_file)

    if args.compute_all:
        success = compute_all_hashes(configs_dir, hash_file)
        return 0 if success else 1

    if args.detect_changes:
        changes = detect_changes(configs_dir, hash_file)
        _print_change_summary(changes)
        return 0

    if args.update:
        success = update_hashes(args.update, configs_dir, hash_file)
        return 0 if success else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
