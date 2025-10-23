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
Hash consistency guard for rocprofiler-compute.

Errors (per arch):
- If latest-arch panels changed but its delta did not (and there are older archs)
- If latest-arch delta changed but its panels did not AND no new arch was added
- If an older arch's panels changed but its delta did not
- If an older arch's delta changed but neither latest panels nor this arch's
  panels changed

"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

try:
    from . import hash_manager  # type: ignore
except Exception:
    import importlib.util

    _HERE = Path(__file__).resolve().parent
    _SPEC = importlib.util.spec_from_file_location(
        "hash_manager", str(_HERE / "hash_manager.py")
    )
    hash_manager = importlib.util.module_from_spec(_SPEC)  # type: ignore[assignment]
    assert _SPEC and _SPEC.loader is not None
    _SPEC.loader.exec_module(hash_manager)  # type: ignore[attr-defined]
# ---------------------------------------------------------------------------

# Subproject root: .../projects/rocprofiler-compute
SUBROOT = Path(__file__).resolve().parents[2]

CONFIGS_ROOT: Path = SUBROOT / "src" / "rocprof_compute_soc" / "analysis_configs"
HASH_FILE: Path = SUBROOT / "tools" / "config_management" / ".config_hashes.json"
TEMPLATE_FILE: Path = (
    SUBROOT / "tools" / "config_management" / "analysis_config_template.yaml"
)


# ---------- helpers ----------


def _latest_arch(template_file: Path) -> str:
    if not template_file.is_file():
        return ""
    with open(template_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return str(data.get("latest_arch") or "")


def _all_archs(cfg_root: Path) -> list[str]:
    if not cfg_root.is_dir():
        return []
    return sorted(
        p.name for p in cfg_root.iterdir() if p.is_dir() and p.name.startswith("gfx")
    )


def _cur_panels_and_delta(arch_dir: Path) -> tuple[dict[str, str], str]:
    """
    Current (on-disk) hashes using hash_manager.compute_arch_hashes API:
      returns {"files": {...}, "delta_hash": <md5 or None>}
    """
    cur = hash_manager.compute_arch_hashes(arch_dir)
    panels = dict(cur.get("files") or {})
    delta_hash = cur.get("delta_hash") or ""
    return panels, str(delta_hash)


def _prev_panels_and_delta(
    hashes_path: Path, arch_name: str
) -> tuple[dict[str, str], str]:
    """
    Previous (DB) hashes saved in .config_hashes.json:
      stored as {"files": {...}, "delta_hash": <md5 or None>}
    """
    db: dict = hash_manager.load_hash_db(hashes_path)
    prev_arch: dict = (db.get("archs") or {}).get(arch_name, {})  # type: ignore[assignment]
    panels = dict(prev_arch.get("files") or {})
    delta_hash = prev_arch.get("delta_hash") or ""
    return panels, str(delta_hash)


def _changed_panel_files(cur: dict[str, str], prev: dict[str, str]) -> list[str]:
    """
    Return a small list of changed panel filenames (added/removed/modified).
    """
    # structural changes (added/removed)
    changed = sorted(set(cur) ^ set(prev))
    if not changed:
        # content changes for existing files
        changed = sorted(k for k in cur.keys() & prev.keys() if cur[k] != prev[k])
    return changed


# ---------- main ----------


def main() -> int:
    if not CONFIGS_ROOT.is_dir():
        print(f"ERROR: analysis_configs directory not found at: {CONFIGS_ROOT}")
        return 2

    latest = _latest_arch(TEMPLATE_FILE)
    all_archs = _all_archs(CONFIGS_ROOT)
    older_archs = [a for a in all_archs if a != latest]

    # detect new archs via hash_manager.detect_changes if available
    try:
        changes: dict = hash_manager.detect_changes(CONFIGS_ROOT, HASH_FILE)  # type: ignore[call-arg]
    except TypeError:
        # old/new signatures both accept (cfg_root, hashes_path)
        changes = hash_manager.detect_changes(CONFIGS_ROOT, HASH_FILE)  # type: ignore[call-arg]
    new_archs: list = changes.get("new_archs") or []

    errors: list[str] = []

    # Track whether latest panels changed (used for older-arch delta rule)
    latest_panels_changed = False

    for arch in all_archs:
        arch_dir = CONFIGS_ROOT / arch
        cur_panels, cur_delta = _cur_panels_and_delta(arch_dir)
        prev_panels, prev_delta = _prev_panels_and_delta(HASH_FILE, arch)

        panel_changed = cur_panels != prev_panels
        delta_changed = cur_delta != prev_delta

        if arch == latest:
            latest_panels_changed = panel_changed

            # A) Latest panels changed but no delta changed (and there ARE older archs)
            if panel_changed and not delta_changed and older_archs:
                snippet = ", ".join(_changed_panel_files(cur_panels, prev_panels)[:5])
                errors.append(
                    f"Panels changed in latest arch '{latest}' "
                    "but its delta file did not change.\n"
                    f"Changed panels (sample): {snippet}\n"
                    "Run the workflow to regenerate deltas for previous archs."
                )

            # B) Latest delta changed but panels did not AND no new arch was added
            if delta_changed and not panel_changed and latest not in new_archs:
                errors.append(
                    "Delta file changed for latest, but panels "
                    "didn't change and no new arch was added.\n"
                    "This usually means deltas were edited/regenerated "
                    "without corresponding latest updates."
                )

        else:
            # C) Arch panels changed but its delta did not
            if panel_changed and not delta_changed:
                snippet = ", ".join(_changed_panel_files(cur_panels, prev_panels)[:5])
                errors.append(
                    f"Panels changed in arch '{arch}' "
                    "but its delta file did not change.\n"
                    f"Changed panels (sample): {snippet}\n"
                    "Regenerate deltas for this arch (diff vs latest) "
                    "and commit them."
                )

            # D) Older arch delta changed without either latest panels changing
            #    OR this arch's panels changing -> error
            #    (allow if latest panels changed: deltas can legitimately change then)
            if delta_changed and not panel_changed and not latest_panels_changed:
                errors.append(
                    f"Delta file changed under older arch '{arch}' "
                    "but neither latest nor this arch's panels changed.\n"
                    "This suggests stray delta edits; "
                    "verify latest panels or this arch's panels, or revert."
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
