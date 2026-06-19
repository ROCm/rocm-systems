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


def main() -> int:
    if not CONFIGS_ROOT.is_dir():
        print(f"ERROR: analysis_configs directory not found at: {CONFIGS_ROOT}")
        return 2

    changes: dict = hash_manager.detect_changes(CONFIGS_ROOT, HASH_FILE)

    errors: list[str] = [
        f"Arch '{arch}' removed from disk but still in .config_hashes.json.\n"
        "Run hash_manager.py --compute-all to refresh the DB."
        for arch in (changes.get("deleted_archs") or [])
    ]

    for arch in changes.get("new_archs") or []:
        errors.append(
            f"New arch '{arch}' has no entry in .config_hashes.json.\n"
            "Run hash_manager.py --compute-all to refresh the DB."
        )

    for arch, changed_files in (changes.get("modified_archs") or {}).items():
        snippet = ", ".join(changed_files[:5])
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
