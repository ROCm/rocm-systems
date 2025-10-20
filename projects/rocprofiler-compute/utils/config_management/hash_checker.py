#!/usr/bin/env python3
"""
Hash consistency guard for rocprofiler-compute.

Rules:
- If latest-arch panel YAMLs changed but no delta YAML changed -> error
- If delta YAMLs changed but latest panels did not (and no new arch was added) -> error
- Soft warning when older-arch panels changed

Run from any CWD (paths are resolved from this file).
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# --- robust local import of hash_manager (works as module or direct file) ---
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

# Accept both the canonical and legacy folder spelling
CONFIGS_ROOT: Path = SUBROOT / "src" / "rocprof_compute_soc" / "analysis_configs"

HASH_FILE: Path = SUBROOT / "utils" / "config_management" / ".config_hashes.json"
TEMPLATE_FILE: Path = (
    SUBROOT / "utils" / "config_management" / "analysis_config_template.yaml"
)
PER_ARCH_DEFS_ROOT: Path = SUBROOT / "utils" / "per_arch_metric_definitions"


# ---------- small helpers ----------


def _safe_detect_changes(cfg_root: Path, hashes_path: Path, defs_root: Path) -> dict:
    """Support both new and old hash_manager signatures."""
    try:
        return hash_manager.detect_changes(
            cfg_root, hashes_path, per_arch_defs_root=defs_root
        )
    except TypeError:
        return hash_manager.detect_changes(cfg_root, hashes_path)


def _compute_current_by_area(
    arch_dir: Path, defs_root: Path, arch_name: str
) -> tuple[dict, dict, dict]:
    """Return (panels, deltas, defs) as {rel_path -> sha}."""
    try:
        current: dict = hash_manager.compute_arch_hashes(
            arch_dir, per_arch_defs_root=defs_root, arch_name=arch_name
        )
    except TypeError:
        current = hash_manager.compute_arch_hashes(arch_dir)

    panels: dict = {}
    deltas: dict = {}
    defs: dict = {}
    for rel, h in current.items():
        if rel.startswith("defs/"):
            defs[rel] = h
        elif rel.startswith("config_delta/"):
            deltas[rel] = h
        elif rel.endswith(".yaml"):
            panels[rel] = h
    return panels, deltas, defs


def _load_previous_by_area(
    hashes_path: Path, arch_name: str
) -> tuple[dict, dict, dict]:
    """Return previous (panels, deltas, defs) from the saved DB."""
    db: dict = hash_manager.load_hash_db(hashes_path)
    prev_arch: dict = (db.get("archs") or {}).get(arch_name, {})  # type: ignore[assignment]
    panels: dict = {}
    deltas: dict = {}
    defs: dict = {}
    for rel, h in prev_arch.items():
        if rel.startswith("defs/"):
            defs[rel] = h
        elif rel.startswith("config_delta/"):
            deltas[rel] = h
        elif rel.endswith(".yaml"):
            panels[rel] = h
    return panels, deltas, defs


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


def _changed_keys(cur: dict, prev: dict) -> list[str]:
    """list changed keys (added/removed/modified)."""
    keys = sorted(set(cur) ^ set(prev))
    if not keys:
        keys = sorted(k for k in cur.keys() & prev.keys() if cur[k] != prev[k])
    return keys


# ---------- main logic ----------


def main() -> int:
    if not CONFIGS_ROOT.is_dir():
        print(f"ERROR: analysis_configs directory not found at: {CONFIGS_ROOT}")
        return 2

    latest = _latest_arch(TEMPLATE_FILE)
    all_archs = _all_archs(CONFIGS_ROOT)
    older_archs = [a for a in all_archs if a != latest]

    changes: dict = _safe_detect_changes(CONFIGS_ROOT, HASH_FILE, PER_ARCH_DEFS_ROOT)
    modified_archs: dict = changes.get("modified_archs") or {}
    new_archs: list = changes.get("new_archs") or []

    errors: list[str] = []
    warnings: list[str] = []

    # Explicit panel vs delta diff for latest
    if latest:
        arch_dir = CONFIGS_ROOT / latest
        if not arch_dir.is_dir():
            errors.append(f"Latest arch directory not found: {arch_dir}")
        else:
            cur_panels, cur_deltas, _ = _compute_current_by_area(
                arch_dir, PER_ARCH_DEFS_ROOT, latest
            )
            prev_panels, prev_deltas, _ = _load_previous_by_area(HASH_FILE, latest)

            panel_changed = set(cur_panels.items()) != set(prev_panels.items())
            delta_changed = set(cur_deltas.items()) != set(prev_deltas.items())

            # A) Panels changed but no deltas changed (and there ARE older archs)
            if panel_changed and not delta_changed and older_archs:
                snippet = ", ".join(_changed_keys(cur_panels, prev_panels)[:5])
                errors.append(
                    f"Panels changed in latest arch '{latest}' but no delta "
                    "files changed.\n"
                    f"Changed panels (sample): {snippet or '(see git diff)'}\n"
                    "Run the workflow to regenerate deltas for previous archs."
                )

            # B) Deltas changed but panels did not AND no new arch added -> error
            if delta_changed and not panel_changed and latest not in new_archs:
                snippet = ", ".join(_changed_keys(cur_deltas, prev_deltas)[:5])
                errors.append(
                    "Delta files changed, but latest panels didn’t change and "
                    "no new arch was added.\n"
                    f"Changed deltas (sample): {snippet or '(see git diff)'}\n"
                    "This usually means deltas were edited/regenerated without "
                    "corresponding latest updates."
                )

    # Soft note for older-arch panel edits
    for a in modified_archs.keys():
        if a != latest:
            warnings.append(
                f"Older arch '{a}' panels changed. If intentional, ignore this; "
                "otherwise consider applying a delta or promoting a new latest."
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
            "  • If latest changed: regenerate deltas for "
            "prior archs via the workflow script."
        )
        print(
            "  • If deltas changed alone: verify latest "
            "panels/new arch promotion, or revert stray delta edits."
        )
        return 1

    print("Hash consistency check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
