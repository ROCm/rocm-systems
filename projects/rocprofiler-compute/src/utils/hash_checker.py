#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Hash consistency guard for rocprofiler-compute.

Errors when an arch's panel YAML files diverge from the hashes recorded in
.config_hashes.json without the DB being refreshed.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rocprofiler-compute/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.config_management import hash_manager  # noqa: E402

CONFIGS_ROOT: Path = PROJECT_ROOT / "src" / "rocprof_compute_soc" / "analysis_configs"
HASH_FILE: Path = PROJECT_ROOT / "src" / "utils" / ".config_hashes.json"


def _all_archs(cfg_root: Path) -> list[str]:
    if not cfg_root.is_dir():
        return []
    return sorted(
        p.name for p in cfg_root.iterdir() if p.is_dir() and p.name.startswith("gfx")
    )


def _cur_panels(arch_dir: Path) -> dict[str, str]:
    """Current (on-disk) panel-file hashes via hash_manager.compute_arch_hashes."""
    return dict(hash_manager.compute_arch_hashes(arch_dir).get("files") or {})


def _prev_panels(hashes_path: Path, arch_name: str) -> dict[str, str]:
    """Previous panel-file hashes saved in .config_hashes.json."""
    db: dict = hash_manager.load_hash_db(hashes_path)
    prev_arch: dict = (db.get("archs") or {}).get(arch_name, {})  # type: ignore[assignment]
    return dict(prev_arch.get("files") or {})


def _changed_panel_files(cur: dict[str, str], prev: dict[str, str]) -> list[str]:
    """Return a small list of changed panel filenames (added/removed/modified)."""
    changed = sorted(set(cur) ^ set(prev))
    if not changed:
        changed = sorted(k for k in cur.keys() & prev.keys() if cur[k] != prev[k])
    return changed


def main() -> int:
    if not CONFIGS_ROOT.is_dir():
        print(f"ERROR: analysis_configs directory not found at: {CONFIGS_ROOT}")
        return 2

    changes: dict = hash_manager.detect_changes(CONFIGS_ROOT, HASH_FILE)
    new_archs: list = changes.get("new_archs") or []

    errors: list[str] = []

    for arch in _all_archs(CONFIGS_ROOT):
        if arch in new_archs:
            errors.append(
                f"New arch '{arch}' has no entry in .config_hashes.json.\n"
                "Run hash_manager.py --compute-all to refresh the DB."
            )
            continue

        cur_panels = _cur_panels(CONFIGS_ROOT / arch)
        prev_panels = _prev_panels(HASH_FILE, arch)

        if cur_panels != prev_panels:
            snippet = ", ".join(_changed_panel_files(cur_panels, prev_panels)[:5])
            errors.append(
                f"Panels changed in arch '{arch}' but hash DB was not refreshed.\n"
                f"Changed panels (sample): {snippet}\n"
                "Run hash_manager.py --compute-all to refresh the DB."
            )

    if errors:
        print("\nHASH CONSISTENCY ERRORS:")
        for e in errors:
            print("  - " + e)
        return 1

    print("Hash consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
