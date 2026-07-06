#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rocprofiler-compute/

#: Absolute path to the repo's canonical config-hash database.
HASH_DB_PATH = _PROJECT_ROOT / "tools" / "config_management" / ".config_hashes.json"

#: Absolute path to the SoC analysis-config tree the hashes cover.
ANALYSIS_CONFIGS_PATH = (
    _PROJECT_ROOT / "src" / "rocprof_compute_soc" / "analysis_configs"
)


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
    Returns dict: {"files": {filename: hash}}
    """
    arch_path = Path(arch_dir)
    if not arch_path.is_dir():
        return {"files": {}}

    file_hashes: dict[str, str] = {}
    for yaml_file in sorted(arch_path.glob("*.yaml")):
        file_hashes[yaml_file.name] = compute_file_hash(yaml_file)

    return {"files": file_hashes}


def compare_arch_to_db(arch_dir: Path, stored_files: dict[str, str]) -> dict:
    """
    Compare an arch's on-disk panel YAMLs against recorded hashes.

    Recomputes hashes for the YAML files in arch_dir and diffs them against the
    stored_files mapping ({filename: hash}) recorded in the hash DB. Returns a
    dict:
        - added: list[str] -- files on disk with no recorded hash
        - mismatched: list[tuple[str, str, str]] -- (filename, expected, actual)
        - missing: list[str] -- files in the DB but absent on disk
    """
    current_files = compute_arch_hashes(arch_dir)["files"]

    current_names = set(current_files)
    stored_names = set(stored_files)

    mismatched: list[tuple[str, str, str]] = [
        (name, stored_files[name], current_files[name])
        for name in sorted(current_names & stored_names)
        if current_files[name] != stored_files[name]
    ]

    return {
        "added": sorted(current_names - stored_names),
        "mismatched": mismatched,
        "missing": sorted(stored_names - current_names),
    }


def load_hash_db(hash_file: Path) -> dict:
    """Load hash database from file (or initialize)."""
    hash_path = Path(hash_file)
    if not hash_path.exists():
        return {"archs": {}}
    with open(hash_path, encoding="utf-8") as f:
        return json.load(f)


def save_hash_db(hash_file: Path, data: dict) -> None:
    """Save hash database to file."""
    hash_path = Path(hash_file)
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hash_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def detect_changes(configs_dir: Path, hash_file: Path) -> dict:
    """
    Detect changes in architecture configs.
    Returns dict with keys:
        - new_archs: list[str]
        - modified_archs: dict[str, list[str]]
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
        "deleted_archs": sorted(stored_archs - current_archs),
    }

    for arch in sorted(current_archs & stored_archs):
        arch_dir = configs_path / arch
        stored_files = hash_db["archs"].get(arch, {}).get("files", {})
        comparison = compare_arch_to_db(arch_dir, stored_files)

        modified_files: list[str] = list(comparison["added"])
        modified_files += [name for name, _, _ in comparison["mismatched"]]
        modified_files += [f"[DELETED] {name}" for name in comparison["missing"]]

        if modified_files:
            changes["modified_archs"][arch] = modified_files

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

    if changes["deleted_archs"]:
        print("\nDeleted Architectures")
        for arch in changes["deleted_archs"]:
            print(f"   • {arch}")

    if not any([
        changes["new_archs"],
        changes["modified_archs"],
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
        default=HASH_DB_PATH,
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
