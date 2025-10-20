#!/usr/bin/env python3
"""
Fail fast when hash changes look inconsistent:
- If latest arch panels changed but no delta files changed -> error
- If delta files changed but latest arch did not change and no new arch dir -> error

Run this from repo root:
  python utils/config_management/check_hash_consistency.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from . import hash_manager  # when run as module
except Exception:
    import importlib.util

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "hash_manager", str(here / "hash_manager.py")
    )
    hash_manager = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hash_manager)  # type: ignore[attr-defined]
# ----------------------------------------------------------------------


CONFIGS_ROOT = Path("src/rocprof_compute_soc/analysis_configs")
HASH_FILE = Path("utils/config_management/.config_hashes.json")
TEMPLATE_FILE = Path("utils/config_management/analysis_config_template.yaml")
PER_ARCH_DEFS_ROOT = Path(
    "utils/per_arch_metric_definitions"
)  # optional; passed if supported


def get_latest_arch(template_file: Path) -> str:
    if not template_file.is_file():
        return ""
    with open(template_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("latest_arch", "") or ""


def get_all_archs(configs_root: Path) -> list[str]:
    if not configs_root.is_dir():
        return []
    return sorted([
        p.name
        for p in configs_root.iterdir()
        if p.is_dir() and p.name.startswith("gfx")
    ])


def main() -> int:
    latest = get_latest_arch(TEMPLATE_FILE)
    all_archs = get_all_archs(CONFIGS_ROOT)
    other_archs = [a for a in all_archs if a != latest]

    # detect changes (support newer signature with per_arch_defs_root; else fallback)
    try:
        changes: dict[str, Any] = hash_manager.detect_changes(
            CONFIGS_ROOT, HASH_FILE, per_arch_defs_root=PER_ARCH_DEFS_ROOT
        )  # type: ignore[arg-type]
    except TypeError:
        changes = hash_manager.detect_changes(CONFIGS_ROOT, HASH_FILE)  # type: ignore[call-arg]

    modified_archs: dict[str, list[str]] = changes.get("modified_archs") or {}
    delta_files: dict[str, str] = changes.get("delta_files") or {}
    new_archs: list[str] = changes.get("new_archs") or []

    latest_changed = latest and (latest in modified_archs or latest in new_archs)
    num_delta_changes = len(delta_files)

    errors: list[str] = []
    warnings: list[str] = []

    # Rule 1: Latest changed (panels or new latest) -> expect some delta changes
    if latest and latest_changed and num_delta_changes == 0 and len(other_archs) > 0:
        errors.append(
            f"Latest arch '{latest}' changed, but no delta files changed.\n"
            f"Did you forget to regenerate deltas for previous archs?"
        )

    # Rule 2: Deltas changed but latest did NOT change (and no new arch dir)
    if num_delta_changes > 0 and not latest_changed and len(new_archs) == 0:
        changed_list = ", ".join([
            f"{a}:{Path(p).name}" for a, p in delta_files.items()
        ])
        errors.append(
            "Delta files changed, but latest architecture did not change and "
            "no new arch was added.\n"
            f"Changed deltas: {changed_list}\n"
            "This usually means deltas were edited/regenerated without"
            "corresponding latest panel updates."
        )

    for a in modified_archs.keys():
        if a != latest:
            warnings.append(
                f"Older arch '{a}' panels changed. If this was intentional, "
                "ignore this warning. "
                f"Otherwise consider applying a delta or promoting a new latest."
            )

    if warnings:
        print("\nHASH CONSISTENCY WARNINGS:")
        for w in warnings:
            print("  - " + w)

    if errors:
        print("\nHASH CONSISTENCY ERRORS:")
        for e in errors:
            print("  - " + e)
        print("\nTo fix:")
        print(
            "  • If latest changed: regenerate deltas for prior archs via "
            "the workflow script."
        )
        print(
            "  • If deltas changed alone: verify latest panels/new arch promotion, "
            "or revert stray delta edits."
        )
        return 1

    print("Hash consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
