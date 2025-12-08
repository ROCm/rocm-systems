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
Master workflow script for managing architecture configurations.
- Detects changes
- Handles direct edits and delta files
- Supports promoting a NEW arch from:
    (A) direct edits to latest, or
    (B) a delta YAML targeting latest
- Validates, syncs metric descriptions, and updates hashes

"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from . import hash_manager, metric_description_manager
except Exception:
    repo_root = Path(__file__).resolve().parents[1]  # repo root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import config_management.hash_manager as hash_manager  # type: ignore
    import config_management.metric_description_manager as metric_description_manager  # type: ignore

import yaml

# =============================================================================
# CONFIG
# =============================================================================

CONFIG_FILE = "config_workflow.yaml"

DEFAULT_CONFIG: dict = {
    "paths": {
        "template": "tools/config_management/gfx9_config_template.yaml",
        "configs_root": "src/rocprof_compute_soc/analysis_configs",
        "backups": ".backups",
        "hashes": "tools/config_management/.config_hashes.json",
        "per_arch_metrics": "tools/per_arch_metric_definitions",
        "docs_metrics": "docs/data/metrics_description.yaml",
    },
    "validation": {"strict_mode": True, "verify_after_changes": True},
    "behavior": {"require_confirmation": True},
}


# =============================================================================
# UTILITIES
# =============================================================================


def load_config() -> dict:
    """Load config from CONFIG_FILE with a shallow merge onto DEFAULT_CONFIG."""
    p = Path(CONFIG_FILE)
    if not p.exists():
        return DEFAULT_CONFIG
    with open(p) as f:
        user = yaml.safe_load(f) or {}
    merged = DEFAULT_CONFIG.copy()
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def create_backup(source_paths: list[str], backup_dir: str) -> Path:
    """Create a timestamped backup of the provided paths."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # add microseconds
    base = Path(backup_dir)
    base.mkdir(parents=True, exist_ok=True)
    backup_path = base / ts

    # Fallback suffix if somehow collides
    i = 1
    while backup_path.exists():
        backup_path = base / f"{ts}_{i}"
        i += 1

    print(f"Creating backup: {backup_path}")
    for s in source_paths:
        sp = Path(s)
        dst = backup_path / sp.name
        if sp.is_dir():
            shutil.copytree(sp, dst)
        elif sp.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dst)
    return backup_path


def restore_backup(backup_path: Path, target_paths: list[str]) -> None:
    """Restore files/dirs from a given backup path."""
    print(f"Restoring from backup: {backup_path}")
    for t in target_paths:
        tp = Path(t)
        bp = backup_path / tp.name
        if not bp.exists():
            continue
        if tp.is_dir():
            shutil.rmtree(tp, ignore_errors=True)
        elif tp.exists():
            tp.unlink()
        if bp.is_dir():
            shutil.copytree(bp, tp)
        else:
            shutil.copy2(bp, tp)
    print("Backup restored")


def cleanup_old_backups(backup_dir: str) -> None:
    """Keep latest backup, remove older ones."""
    b = Path(backup_dir)
    if not b.exists():
        return
    dirs = sorted([d for d in b.iterdir() if d.is_dir()])
    for old in dirs[:-1]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"Removed old backup: {old.name}")


def prompt_yes_no(question: str, default: Optional[bool] = None) -> bool:
    """Ask a yes/no question in the terminal."""
    if default is None:
        prompt = f"{question} (y/n): "
    elif default:
        prompt = f"{question} [Y/n]: "
    else:
        prompt = f"{question} [y/N]: "
    while True:
        ans = input(prompt).strip().lower()
        if not ans and default is not None:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def run_script(
    script_name: str, args: list[str], capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Run a Python helper script and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, script_name] + args, capture_output=capture_output, text=True
    )


def get_all_archs(configs_dir: str) -> list[str]:
    """Return sorted list of gfx* directories."""
    root = Path(configs_dir)
    return sorted([
        d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("gfx")
    ])


def get_latest_arch(template_file: str) -> Optional[str]:
    """Read 'latest_arch' from template YAML."""
    p = Path(template_file)
    if not p.is_file():
        return None
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    return data.get("latest_arch")


def validate_delta_structure(delta_file: str) -> tuple[bool, str]:
    """Ensure delta YAML contains Addition/Deletion/Modification keys."""
    with open(delta_file) as f:
        data = yaml.safe_load(f) or {}
    required = {"Addition", "Deletion", "Modification"}
    if not isinstance(data, dict) or not required.issubset(data.keys()):
        return False, "Delta must have Addition, Deletion, Modification keys"
    return True, ""


# =============================================================================
# VALIDATION / SYNC
# =============================================================================


def validate_all_archs(config: dict) -> tuple[bool, str]:
    """Validate all archs against the template."""
    print("Validating all architectures against template...")
    res = run_script(
        "tools/config_management/verify_against_config_template.py",
        [config["paths"]["configs_root"], config["paths"]["template"]],
        capture_output=True,
    )
    if res.stdout:
        print(res.stdout)
    if res.returncode != 0:
        if res.stderr:
            print(res.stderr)
        return False, "Validation failed"
    return True, "Validation passed"


def validate_arch_against_template(arch_name: str, config: dict) -> tuple[bool, str]:
    """Validate one arch (best-effort: rely on script output mentioning arch)."""
    print(f"Validating {arch_name} against template...")
    res = run_script(
        "tools/config_management/verify_against_config_template.py",
        [config["paths"]["configs_root"], config["paths"]["template"]],
        capture_output=True,
    )
    if res.returncode != 0 and arch_name in (res.stdout or ""):
        print(res.stdout)
        return False, f"Validation failed for {arch_name}"
    return True, f"Validation passed for {arch_name}"


# =============================================================================
# CHANGE DETECTION
# =============================================================================


def detect_changes(config: dict) -> dict:
    print("Detecting changes...")
    return hash_manager.detect_changes(
        config["paths"]["configs_root"], config["paths"]["hashes"]
    )


def display_change_summary(changes: dict) -> bool:
    print("\n" + "=" * 80)
    print("CHANGE SUMMARY")
    print("=" * 80)

    has_changes = any([
        changes.get("new_archs"),
        changes.get("modified_archs"),
        changes.get("delta_files"),
        changes.get("deleted_archs"),
    ])

    if changes.get("new_archs"):
        print("\nNew Architecture Directories:")
        for a in changes["new_archs"]:
            print(f"   • {a}")

    if changes.get("modified_archs"):
        print("\nModified Architectures:")
        for a, files in changes["modified_archs"].items():
            print(f"   • {a}:")
            for f in files[:5]:
                print(f"      - {f}")
            extra = len(files) - 5
            if extra > 0:
                print(f"      ... and {extra} more files")

    if changes.get("delta_files"):
        print("\nDelta Files Detected:")
        for a, d in changes["delta_files"].items():
            print(f"   • {a}: {Path(d).name}")

    if changes.get("deleted_archs"):
        print("\nDeleted Architectures:")
        for a in changes["deleted_archs"]:
            print(f"   • {a}")

    if not has_changes:
        print("\nNo changes detected")

    print("=" * 80 + "\n")
    return has_changes


# =============================================================================
# CORE WORKFLOW OPS
# =============================================================================


def promote_to_latest(
    new_arch: str, config: dict, reuse_backup: Optional[Path] = None
) -> bool:
    """
    Original 'promote' that assumes new_arch dir already exists & populated.
    (Kept for backward compatibility.)
    """
    print(f"\nPROMOTING {new_arch} TO LATEST ARCHITECTURE...")
    backup_paths = [config["paths"]["configs_root"], config["paths"]["template"]]
    backup_path = reuse_backup or create_backup(
        backup_paths, config["paths"]["backups"]
    )

    try:
        root = Path(config["paths"]["configs_root"])
        new_dir = root / new_arch
        if not new_dir.is_dir():
            raise Exception(f"New arch directory not found: {new_dir}")

        all_archs = get_all_archs(config["paths"]["configs_root"])
        prev_archs = [a for a in all_archs if a != new_arch]

        print(f"\n1. Updating template with new latest arch: {new_arch}")
        res = run_script(
            "tools/config_management/parse_config_template.py",
            [str(new_dir), config["paths"]["template"], "--latest-arch", new_arch],
            capture_output=True,
        )
        if res.returncode != 0:
            raise Exception(f"Failed to update template: {res.stderr}")

        print(f"\n2. Generating deltas for {len(prev_archs)} previous architectures")
        for p in prev_archs:
            prev_dir = root / p
            gen = run_script(
                "tools/config_management/generate_config_deltas.py",
                [str(new_dir), str(prev_dir)],
                capture_output=True,
            )
            if gen.returncode != 0:
                raise Exception(f"Failed to generate delta for {p}: {gen.stderr}")

        print("\n\tUpdating hashes for previous architectures (delta files)")
        for p in prev_archs:
            hash_manager.update_hashes(
                p, config["paths"]["configs_root"], config["paths"]["hashes"]
            )

        print("\n3. Validating all architectures")
        ok, msg = validate_all_archs(config)
        if not ok:
            raise Exception(msg)

        print("\n4. Syncing metric descriptions")
        ok = metric_description_manager.sync_arch(
            new_arch,
            config["paths"]["configs_root"],
            config["paths"]["per_arch_metrics"],
            config["paths"]["docs_metrics"],
            is_latest=True,
        )
        if not ok:
            raise Exception("Failed to sync metric descriptions")

        print("\n5. Updating hashes")
        hash_manager.update_hashes(
            new_arch, config["paths"]["configs_root"], config["paths"]["hashes"]
        )

        print(f"\nSuccessfully promoted {new_arch} to latest architecture!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}\nRestoring from backup...")
        restore_backup(backup_path, backup_paths)
        return False


def update_latest_arch_from_delta(
    delta_file: str, arch_name: str, config: dict
) -> bool:
    """Apply a delta in-place to the latest arch (legacy flow)."""
    print(f"\nUPDATING LATEST ARCH {arch_name} FROM DELTA...")
    backup_paths = [config["paths"]["configs_root"], config["paths"]["template"]]
    backup_path = create_backup(backup_paths, config["paths"]["backups"])

    try:
        root = Path(config["paths"]["configs_root"])
        arch_dir = root / arch_name
        tmp = root / f"{arch_name}_tmp"

        print(f"\n1. Applying delta to {arch_name}")
        res = run_script(
            "tools/config_management/apply_config_deltas.py",
            [str(arch_dir), delta_file, str(tmp)],
            capture_output=True,
        )
        if res.returncode != 0:
            raise Exception(f"Failed to apply delta: {res.stderr}")

        shutil.rmtree(arch_dir)
        shutil.move(str(tmp), str(arch_dir))

        print("\n2. Updating template")
        res = run_script(
            "tools/config_management/parse_config_template.py",
            [str(arch_dir), config["paths"]["template"], "--latest-arch", arch_name],
            capture_output=True,
        )
        if res.returncode != 0:
            raise Exception(f"Failed to update template: {res.stderr}")

        print("\n3. Regenerating deltas for previous architectures")
        all_archs = get_all_archs(config["paths"]["configs_root"])
        for prev in [a for a in all_archs if a != arch_name]:
            prev_dir = root / prev
            gen = run_script(
                "tools/config_management/generate_config_deltas.py",
                [str(arch_dir), str(prev_dir)],
                capture_output=True,
            )
            if gen.returncode != 0:
                raise Exception(f"Failed to generate delta for {prev}")

        for prev in [a for a in all_archs if a != arch_name]:
            hash_manager.update_hashes(
                prev, config["paths"]["configs_root"], config["paths"]["hashes"]
            )

        print("\n4. Validating all architectures")
        ok, msg = validate_all_archs(config)
        if not ok:
            raise Exception(msg)

        print("\n5. Syncing metric descriptions")
        ok = metric_description_manager.sync_arch(
            arch_name,
            config["paths"]["configs_root"],
            config["paths"]["per_arch_metrics"],
            config["paths"]["docs_metrics"],
            is_latest=True,
        )
        if not ok:
            raise Exception("Failed to sync metric descriptions")

        print("\n6. Updating hashes")
        hash_manager.update_hashes(
            arch_name, config["paths"]["configs_root"], config["paths"]["hashes"]
        )

        print(f"\nSuccessfully updated latest arch {arch_name}!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}\nRestoring from backup...")
        restore_backup(backup_path, backup_paths)
        return False


def update_older_arch_from_delta(delta_file: str, arch_name: str, config: dict) -> bool:
    """Apply a delta in-place to an older arch (legacy flow)."""
    print(f"\nUPDATING OLDER ARCH {arch_name} FROM DELTA...")
    root = Path(config["paths"]["configs_root"])
    arch_dir = root / arch_name
    backup_path = create_backup([str(arch_dir)], config["paths"]["backups"])

    try:
        tmp = root / f"{arch_name}_tmp"

        print(f"\n1. Applying delta to {arch_name}")
        res = run_script(
            "tools/config_management/apply_config_deltas.py",
            [str(arch_dir), delta_file, str(tmp)],
            capture_output=True,
        )
        if res.returncode != 0:
            raise Exception(f"Failed to apply delta: {res.stderr}")

        shutil.rmtree(arch_dir)
        shutil.move(str(tmp), str(arch_dir))

        print("\n2. Validating against template")
        ok, msg = validate_arch_against_template(arch_name, config)
        if not ok:
            raise Exception(msg)

        print("\n3. Syncing metric descriptions")
        ok = metric_description_manager.sync_arch(
            arch_name,
            config["paths"]["configs_root"],
            config["paths"]["per_arch_metrics"],
            config["paths"]["docs_metrics"],
            is_latest=False,
        )
        if not ok:
            raise Exception("Failed to sync metric descriptions")

        print("\n4. Updating hashes")
        hash_manager.update_hashes(
            arch_name, config["paths"]["configs_root"], config["paths"]["hashes"]
        )

        print(f"\nSuccessfully updated older arch {arch_name}!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}\nRestoring from backup...")
        restore_backup(backup_path, [str(arch_dir)])
        return False


def update_latest_arch_from_edits(arch_name: str, config: dict) -> bool:
    """Re-derive template/deltas from direct edits to latest (legacy in-place)."""
    print(f"\nUPDATING LATEST ARCH {arch_name} FROM DIRECT EDITS...")
    backup_paths = [config["paths"]["configs_root"], config["paths"]["template"]]
    backup_path = create_backup(backup_paths, config["paths"]["backups"])

    try:
        root = Path(config["paths"]["configs_root"])
        arch_dir = root / arch_name

        print("\n1. Updating template")
        res = run_script(
            "tools/config_management/parse_config_template.py",
            [str(arch_dir), config["paths"]["template"], "--latest-arch", arch_name],
            capture_output=True,
        )
        if res.returncode != 0:
            raise Exception(f"Failed to update template: {res.stderr}")

        print("\n2. Regenerating deltas for previous architectures")
        for prev in [
            a for a in get_all_archs(config["paths"]["configs_root"]) if a != arch_name
        ]:
            prev_dir = root / prev
            gen = run_script(
                "tools/config_management/generate_config_deltas.py",
                [str(arch_dir), str(prev_dir)],
                capture_output=True,
            )
            if gen.returncode != 0:
                raise Exception(f"Failed to generate delta for {prev}")

        for prev in [
            a for a in get_all_archs(config["paths"]["configs_root"]) if a != arch_name
        ]:
            hash_manager.update_hashes(
                prev, config["paths"]["configs_root"], config["paths"]["hashes"]
            )

        print("\n3. Validating all architectures")
        ok, msg = validate_all_archs(config)
        if not ok:
            raise Exception(msg)

        print("\n4. Syncing metric descriptions")
        ok = metric_description_manager.sync_arch(
            arch_name,
            config["paths"]["configs_root"],
            config["paths"]["per_arch_metrics"],
            config["paths"]["docs_metrics"],
            is_latest=True,
        )
        if not ok:
            raise Exception("Failed to sync metric descriptions")

        print("\n5. Updating hashes")
        hash_manager.update_hashes(
            arch_name, config["paths"]["configs_root"], config["paths"]["hashes"]
        )

        print(f"\nSuccessfully updated latest arch {arch_name}!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}\nRestoring from backup...")
        restore_backup(backup_path, backup_paths)
        return False


def update_older_arch_from_edits(arch_name: str, config: dict) -> bool:
    """Re-validate/sync/hash older arch after direct edits (legacy in-place)."""
    print(f"\nUPDATING OLDER ARCH {arch_name} FROM DIRECT EDITS...")
    root = Path(config["paths"]["configs_root"])
    arch_dir = root / arch_name
    backup_path = create_backup([str(arch_dir)], config["paths"]["backups"])

    try:
        print("\n1. Validating against template")
        ok, msg = validate_arch_against_template(arch_name, config)
        if not ok:
            raise Exception(msg)

        print("\n2. Syncing metric descriptions")
        ok = metric_description_manager.sync_arch(
            arch_name,
            config["paths"]["configs_root"],
            config["paths"]["per_arch_metrics"],
            config["paths"]["docs_metrics"],
            is_latest=False,
        )
        if not ok:
            raise Exception("Failed to sync metric descriptions")

        print("\n3. Updating hashes")
        hash_manager.update_hashes(
            arch_name, config["paths"]["configs_root"], config["paths"]["hashes"]
        )

        print(f"\nSuccessfully updated older arch {arch_name}!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}\nRestoring from backup...")
        restore_backup(backup_path, [str(arch_dir)])
        return False


# =============================================================================
# NEW: PROMOTE NEW ARCH FROM (A) EDITS or (B) DELTA
# =============================================================================


def _git_restore_pristine(path: Path) -> None:
    """
    Best-effort restore of a directory to HEAD using Git.
    No-op if not in a Git repo. Raises on checkout failure when in a repo.
    """
    chk = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True
    )
    if chk.returncode != 0 or chk.stdout.strip() != "true":
        return
    res = subprocess.run(
        ["git", "checkout", "--", str(path)], capture_output=True, text=True
    )
    if res.returncode != 0:
        raise Exception(f"Failed to restore pristine state from Git for {path}")


def promote_new_arch_from_latest_edits(
    latest_arch: str, new_arch: str, config: dict
) -> bool:
    """
    Flow (A): Direct edits were made to the current latest arch.
    1) Snapshot edited latest to temp
    2) Restore pristine latest (via Git)
    3) Copy pristine latest → new arch
    4) Generate delta (edited_tmp vs pristine_latest) → write under latest/config_delta/
    5) Apply delta to new arch
    6) Update template latest=new_arch, regen deltas, validate, sync, hash
    """
    print(f"\nPROMOTING {new_arch} FROM EDITS IN {latest_arch}...")
    root = Path(config["paths"]["configs_root"])
    latest_dir = root / latest_arch
    new_dir = root / new_arch
    edited_tmp = root / f"_{latest_arch}_edited_tmp"
    new_tmp = root / f"_{new_arch}_tmp"

    backup_paths = [config["paths"]["configs_root"], config["paths"]["template"]]
    backup_path = create_backup(backup_paths, config["paths"]["backups"])

    try:
        # 1) Snapshot edited latest
        if edited_tmp.exists():
            shutil.rmtree(edited_tmp)
        shutil.copytree(latest_dir, edited_tmp)

        # 2) Restore pristine latest
        _git_restore_pristine(latest_dir)

        # 3) Copy pristine latest → new arch
        if new_dir.exists():
            raise Exception(f"Target new arch directory already exists: {new_dir}")
        shutil.copytree(latest_dir, new_dir)

        # 4) Generate delta: edited (curr) vs pristine latest (prev)
        print("\nGenerating delta (edited latest → pristine latest)")
        gen = run_script(
            "tools/config_management/generate_config_deltas.py",
            [str(edited_tmp), str(latest_dir)],
            capture_output=True,
        )
        if gen.returncode != 0:
            raise Exception(f"Failed to generate delta: {gen.stderr}")

        delta_dir = latest_dir / "config_delta"
        # Prefer the file named for edited_tmp; otherwise take the latest *_diff.yaml
        candidates = sorted(delta_dir.glob(f"{edited_tmp.name}_diff.yaml")) or sorted(
            delta_dir.glob("*_diff.yaml")
        )
        if not candidates:
            raise Exception("Delta file not found after generation.")
        delta_file = candidates[-1]

        # 5) Apply delta onto new arch
        if new_tmp.exists():
            shutil.rmtree(new_tmp)
        print(f"\nApplying delta to {new_arch}: {delta_file.name}")
        app = run_script(
            "tools/config_management/apply_config_deltas.py",
            [str(new_dir), str(delta_file), str(new_tmp)],
            capture_output=True,
        )
        if app.returncode != 0:
            raise Exception(f"Failed to apply delta: {app.stderr}")
        shutil.rmtree(new_dir)
        shutil.move(str(new_tmp), str(new_dir))

        # 6) Promote to latest, regen deltas, validate, sync, hash
        return promote_to_latest(new_arch, config, reuse_backup=backup_path)

    except Exception as e:
        print(f"\nERROR: {e}\nRestoring from backup...")
        restore_backup(backup_path, backup_paths)
        return False
    finally:
        if edited_tmp.exists():
            shutil.rmtree(edited_tmp, ignore_errors=True)
        if new_tmp.exists():
            shutil.rmtree(new_tmp, ignore_errors=True)


def promote_new_arch_from_delta(
    latest_arch: str, new_arch: str, delta_file: str, config: dict
) -> bool:
    """
    Flow (B): Developer added a delta YAML targeting the latest arch.
    1) Copy pristine latest → new arch
    2) Apply the provided delta to new arch
    3) Promote to latest, regen deltas, validate, sync, hash
    """
    print(f"\nPROMOTING {new_arch} FROM DELTA ON {latest_arch}...")
    root = Path(config["paths"]["configs_root"])
    latest_dir = root / latest_arch
    new_dir = root / new_arch
    new_tmp = root / f"_{new_arch}_tmp"

    backup_paths = [config["paths"]["configs_root"], config["paths"]["template"]]
    backup_path = create_backup(backup_paths, config["paths"]["backups"])

    try:
        if not Path(delta_file).is_file():
            raise Exception(f"Delta file does not exist: {delta_file}")
        if not latest_dir.is_dir():
            raise Exception(f"Latest arch not found: {latest_dir}")
        if new_dir.exists():
            raise Exception(f"Target new arch directory already exists: {new_dir}")

        # Start from pristine latest
        _git_restore_pristine(latest_dir)

        # 1) Copy pristine latest → new arch
        shutil.copytree(latest_dir, new_dir)

        # 2) Apply delta onto the new arch
        if new_tmp.exists():
            shutil.rmtree(new_tmp)
        print(f"\nApplying delta to {new_arch}: {Path(delta_file).name}")
        app = run_script(
            "tools/config_management/apply_config_deltas.py",
            [str(new_dir), str(delta_file), str(new_tmp)],
            capture_output=True,
        )
        if app.returncode != 0:
            raise Exception(f"Failed to apply delta: {app.stderr}")
        shutil.rmtree(new_dir)
        shutil.move(str(new_tmp), str(new_dir))

        # 3) Promote to latest, regen deltas, validate, sync, hash
        return promote_to_latest(new_arch, config, reuse_backup=backup_path)

    except Exception as e:
        print(f"\nERROR: {e}\nRestoring from backup...")
        restore_backup(backup_path, backup_paths)
        return False
    finally:
        if new_tmp.exists():
            shutil.rmtree(new_tmp, ignore_errors=True)


# =============================================================================
# USER-FACING SCENARIO HANDLERS
# =============================================================================


def handle_new_arch(arch_name: str, config: dict, dry_run: bool = False) -> bool:
    print(f"\n{'=' * 80}\nNEW ARCHITECTURE DETECTED: {arch_name}\n{'=' * 80}")
    if not prompt_yes_no(f"Is {arch_name} the new latest architecture?"):
        print(
            "ERROR: New arch detected but not marked as latest.\n   "
            "Only the latest arch should be added as a new directory."
        )
        return False
    if dry_run:
        print(f"[DRY RUN] Would promote {arch_name} to latest")
        return True
    return promote_to_latest(arch_name, config)


def handle_delta_file(
    delta_file: str, arch_name: str, config: dict, dry_run: bool = False
) -> bool:
    print(
        f"\n{'=' * 80}\nDELTA FILE DETECTED: {Path(delta_file).name}\n   "
        f"Target architecture: {arch_name}\n{'=' * 80}"
    )

    valid, err = validate_delta_structure(delta_file)
    if not valid:
        print(f"ERROR: Invalid delta structure - {err}")
        return False

    latest = (
        get_latest_arch(config["paths"]["template"])
        or (get_all_archs(config["paths"]["configs_root"]) or [None])[-1]
    )

    if arch_name == latest:
        print(f"\nDelta targets the current latest arch: {latest}")
        print("Choose how to apply this delta:")
        print("  1. Update the existing latest arch in-place")
        print(
            "  2. Create a NEW architecture from latest and apply "
            "the delta there (promote to latest)"
        )

        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                if dry_run:
                    print(f"[DRY RUN] Would update latest arch {latest} from delta")
                    return True
                return update_latest_arch_from_delta(delta_file, latest, config)
            if choice == "2":
                new_arch_name = input(
                    "Enter new architecture name (e.g., gfx955): "
                ).strip()
                if not new_arch_name:
                    print("New architecture name cannot be empty.")
                    continue
                if not prompt_yes_no(
                    f"Promote {new_arch_name} to new latest architecture?"
                ):
                    print("Operation cancelled.")
                    return False
                if dry_run:
                    print(
                        "[DRY RUN] Would create "
                        f"{new_arch_name} from {latest} and apply delta"
                    )
                    return True
                return promote_new_arch_from_delta(
                    latest, new_arch_name, delta_file, config
                )
            print("Invalid choice. Please enter 1 or 2.")
    else:
        if not prompt_yes_no(f"Apply delta to older arch ({arch_name}) in-place?"):
            return False
        if dry_run:
            print(f"[DRY RUN] Would update older arch {arch_name} from delta")
            return True
        return update_older_arch_from_delta(delta_file, arch_name, config)


def handle_direct_edits(
    arch_name: str, modified_files: list[str], config: dict, dry_run: bool = False
) -> bool:
    print(f"\n{'=' * 80}\nDIRECT EDITS DETECTED: {arch_name}\n{'=' * 80}")
    print("Modified files:")
    for f in modified_files:
        print(f"   • {f}")

    latest = (
        get_latest_arch(config["paths"]["template"])
        or (get_all_archs(config["paths"]["configs_root"]) or [None])[-1]
    )

    if arch_name == latest:
        print(f"\nThis is the current latest architecture ({latest}).")
        print("Are you:")
        print("  1. Updating the existing latest arch")
        print("  2. Creating a new architecture (this will become the new latest)")

        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                if dry_run:
                    print(
                        f"[DRY RUN] Would update latest arch {latest} from direct edits"
                    )
                    return True
                return update_latest_arch_from_edits(arch_name, config)
            if choice == "2":
                new_arch_name = (
                    input(
                        "Enter new architecture name "
                        f"(currently detected as {arch_name}): "
                    ).strip()
                    or arch_name
                )
                if not prompt_yes_no(
                    f"Promote {new_arch_name} to new latest architecture?"
                ):
                    print("Operation cancelled.")
                    return False
                if dry_run:
                    print(
                        "[DRY RUN] Would promote "
                        f"{new_arch_name} from edits in {arch_name}"
                    )
                    return True
                return promote_new_arch_from_latest_edits(
                    arch_name, new_arch_name, config
                )
            print("Invalid choice. Please enter 1 or 2.")
    else:
        if not prompt_yes_no(
            f"These are edits to older arch ({arch_name}). Continue (in-place)?"
        ):
            return False
        if dry_run:
            print(f"[DRY RUN] Would update older arch {arch_name} from direct edits")
            return True
        return update_older_arch_from_edits(arch_name, config)


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Master workflow for managing architecture configurations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("ARCHITECTURE CONFIG WORKFLOW")
    print("=" * 80)

    config = load_config()

    if args.dry_run:
        print("\nDRY RUN MODE - No changes will be made\n")

    changes = detect_changes(config)
    has_changes = display_change_summary(changes)
    if not has_changes:
        return 0

    latest_arch = (
        get_latest_arch(config["paths"]["template"])
        or (get_all_archs(config["paths"]["configs_root"]) or [None])[-1]
    )
    latest_has_edits = latest_arch in (changes.get("modified_archs") or {})

    # New arch directories that appeared on disk
    for new_arch in changes.get("new_archs", []):
        if not handle_new_arch(new_arch, config, args.dry_run):
            return 1

    # If latest was directly edited, prioritize resolving that path
    # (user will choose in-place vs new arch)
    if latest_has_edits:
        if not handle_direct_edits(
            latest_arch, changes["modified_archs"][latest_arch], config, args.dry_run
        ):
            return 1
        print("\nNote: Delta files for older archs will be regenerated automatically.")
        print("Skipping delta file processing for older architectures.\n")
    else:
        # Process delta files
        for arch, delta_file in changes.get("delta_files", {}).items():
            if not handle_delta_file(delta_file, arch, config, args.dry_run):
                return 1

    # Remaining direct edits (excluding latest if already processed)
    for arch, files in (changes.get("modified_archs") or {}).items():
        if arch == latest_arch and latest_has_edits:
            continue
        if arch in (changes.get("delta_files") or {}):
            continue
        if not handle_direct_edits(arch, files, config, args.dry_run):
            return 1

    if not args.dry_run:
        cleanup_old_backups(config["paths"]["backups"])
        print("\n" + "=" * 80)
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("DRY RUN COMPLETE")
        print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
