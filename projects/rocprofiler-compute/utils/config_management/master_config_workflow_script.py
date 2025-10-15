#!/usr/bin/env python3

"""
Master workflow script for managing architecture configurations.
Handles detection, validation, and application of config changes.
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Import our utilities
import hash_manager
import metric_description_manager
import yaml

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_FILE = "config_workflow.yaml"

DEFAULT_CONFIG = {
    "paths": {
        "template": "utils/config_management/analysis_config_template.yaml",
        "configs_root": "src/rocprof_compute_soc/analysis_configs",
        "backups": ".backups",
        "hashes": "utils/config_management/.config_hashes.json",
        "per_arch_metrics": "utils/per_arch_metric_definitions",
        "docs_metrics": "docs/data/metrics_description.yaml",
    },
    "validation": {"strict_mode": True, "verify_after_changes": True},
    "behavior": {"require_confirmation": True},
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def load_config():
    """Load configuration from file or use defaults."""
    config_path = Path(CONFIG_FILE)
    if config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        for key in user_config:
            if isinstance(user_config[key], dict):
                config[key].update(user_config[key])
            else:
                config[key] = user_config[key]
        return config
    return DEFAULT_CONFIG


def create_backup(source_paths, backup_dir):
    """Create single timestamped backup of given paths."""
    backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(backup_dir) / backup_id
    backup_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating backup: {backup_path}")

    for source in source_paths:
        source_path = Path(source)
        if source_path.is_dir():
            dest = backup_path / source_path.name
            shutil.copytree(source_path, dest)
        elif source_path.is_file():
            dest = backup_path / source_path.name
            shutil.copy2(source_path, dest)

    return backup_path


def restore_backup(backup_path, target_paths):
    """Restore from backup."""
    print(f"Restoring from backup: {backup_path}")

    backup_path = Path(backup_path)
    for target in target_paths:
        target_path = Path(target)
        backup_item = backup_path / target_path.name

        if not backup_item.exists():
            continue

        # Remove current target
        if target_path.is_dir():
            shutil.rmtree(target_path, ignore_errors=True)
        elif target_path.exists():
            target_path.unlink()

        # Restore from backup
        if backup_item.is_dir():
            shutil.copytree(backup_item, target_path)
        else:
            shutil.copy2(backup_item, target_path)

    print("Backup restored")


def cleanup_old_backups(backup_dir):
    """Keep only the most recent backup, remove all others."""
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return

    backups = sorted([d for d in backup_path.iterdir() if d.is_dir()])

    # Keep only the most recent
    if len(backups) > 1:
        for old_backup in backups[:-1]:
            shutil.rmtree(old_backup)
            print(f"Removed old backup: {old_backup.name}")


def prompt_yes_no(question, default=None):
    """Interactive yes/no prompt."""
    if default is None:
        prompt = f"{question} (y/n): "
    elif default:
        prompt = f"{question} [Y/n]: "
    else:
        prompt = f"{question} [y/N]: "

    while True:
        response = input(prompt).strip().lower()

        if not response:
            if default is not None:
                return default
            continue

        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False

        print("Please answer 'y' or 'n'")


def run_script(script_name, args, capture_output=True):
    """Run one of the existing scripts and return result."""
    result = subprocess.run(
        [sys.executable, script_name] + args, capture_output=capture_output, text=True
    )
    return result


def validate_delta_structure(delta_file):
    """Ensure delta YAML has required structure."""
    with open(delta_file) as f:
        data = yaml.safe_load(f)

    required_keys = {"Addition", "Deletion", "Modification"}
    if not isinstance(data, dict) or not required_keys.issubset(data.keys()):
        return False, "Delta must have Addition, Deletion, Modification keys"

    return True, ""


def get_latest_arch(template_file):
    """Read latest_arch from template."""
    with open(template_file) as f:
        data = yaml.safe_load(f)

    if isinstance(data, dict) and "latest_arch" in data:
        return data["latest_arch"]

    # If no metadata, return None
    return None


def set_latest_arch(template_file, arch_name):
    """Update latest_arch in template."""
    with open(template_file) as f:
        data = yaml.safe_load(f)

    # Ensure data is dict with metadata
    if not isinstance(data, dict) or "panels" not in data:
        # Old format - convert to new format
        data = {
            "latest_arch": arch_name,
            "panels": data if isinstance(data, list) else [],
        }
    else:
        data["latest_arch"] = arch_name

    with open(template_file, "w") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)


def get_all_archs(configs_dir):
    """Get list of all architecture directories."""
    configs_path = Path(configs_dir)
    return sorted([
        d.name
        for d in configs_path.iterdir()
        if d.is_dir() and d.name.startswith("gfx")
    ])


# ============================================================================
# CHANGE DETECTION
# ============================================================================


def detect_changes(config):
    """
    Use hash_manager to detect what changed.
    Returns dict with change information.
    """
    print("Detecting changes...")

    changes = hash_manager.detect_changes(
        config["paths"]["configs_root"], config["paths"]["hashes"]
    )

    return changes


def display_change_summary(changes):
    """Display summary of detected changes."""
    print("\n" + "=" * 80)
    print("CHANGE SUMMARY")
    print("=" * 80)

    has_changes = False

    if changes["new_archs"]:
        has_changes = True
        print("\nNew Architecture Directories:")
        for arch in changes["new_archs"]:
            print(f"   • {arch}")

    if changes["modified_archs"]:
        has_changes = True
        print("\nModified Architectures:")
        for arch, files in changes["modified_archs"].items():
            print(f"   • {arch}:")
            for f in files[:5]:  # Show first 5 files
                print(f"      - {f}")
            if len(files) > 5:
                print(f"      ... and {len(files) - 5} more files")

    if changes["delta_files"]:
        has_changes = True
        print("\nDelta Files Detected:")
        for arch, delta_file in changes["delta_files"].items():
            print(f"   • {arch}: {Path(delta_file).name}")

    if changes["deleted_archs"]:
        has_changes = True
        print("\nDeleted Architectures:")
        for arch in changes["deleted_archs"]:
            print(f"   • {arch}")

    if not has_changes:
        print("\nNo changes detected")

    print("=" * 80 + "\n")

    return has_changes


# ============================================================================
# VALIDATION
# ============================================================================


def validate_all_archs(config):
    """Run verify_against_config_template.py on all archs."""
    print("Validating all architectures against template...")

    result = run_script(
        "utils/config_management/verify_against_config_template.py",
        [config["paths"]["configs_root"], config["paths"]["template"]],
        capture_output=True,
    )

    # Print output
    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        return False, "Validation failed"

    return True, "Validation passed"


def validate_arch_against_template(arch_name, config):
    """Validate a single architecture against template."""
    print(f"Validating {arch_name} against template...")

    # Note: The existing script validates all archs, so we run it
    # but focus on this arch's output
    result = run_script(
        "utils/config_management/verify_against_config_template.py",
        [config["paths"]["configs_root"], config["paths"]["template"]],
        capture_output=True,
    )

    if result.returncode != 0:
        # Check if errors are related to this arch
        if arch_name in result.stdout:
            print(result.stdout)
            return False, f"Validation failed for {arch_name}"

    return True, f"Validation passed for {arch_name}"


# ============================================================================
# SCENARIO HANDLERS
# ============================================================================


def handle_new_arch(arch_name, config, dry_run=False):
    """Handle new architecture directory (Scenario C.1.2)."""
    print(f"\n{'=' * 80}")
    print(f"NEW ARCHITECTURE DETECTED: {arch_name}")
    print("=" * 80)

    if not prompt_yes_no(f"Is {arch_name} the new latest architecture?"):
        print("ERROR: New arch detected but not marked as latest.")
        print("   Only the latest arch should be added as a new directory.")
        return False

    if dry_run:
        print("[DRY RUN] Would promote {arch_name} to latest")
        return True

    # Proceed with promotion
    return promote_to_latest(arch_name, config)


def handle_delta_file(delta_file, arch_name, config, dry_run=False):
    """Handle delta YAML file detected."""
    print(f"\n{'=' * 80}")
    print(f"DELTA FILE DETECTED: {Path(delta_file).name}")
    print(f"   Target architecture: {arch_name}")
    print("=" * 80)

    # Validate delta structure
    valid, error = validate_delta_structure(delta_file)
    if not valid:
        print(f"ERROR: Invalid delta structure - {error}")
        return False

    # Determine if this is for latest arch
    latest = get_latest_arch(config["paths"]["template"])

    if not latest:
        print("WARNING: No latest arch defined in template")
        latest = get_all_archs(config["paths"]["configs_root"])[-1]
        print(f"   Assuming latest arch is: {latest}")

    if arch_name == latest:
        if not prompt_yes_no(f"Apply delta to latest arch ({latest})?"):
            return False

        if dry_run:
            print(f"[DRY RUN] Would update latest arch {latest} from delta")
            return True

        return update_latest_arch_from_delta(delta_file, latest, config)
    else:
        if not prompt_yes_no(f"Apply delta to older arch ({arch_name})?"):
            return False

        if dry_run:
            print(f"[DRY RUN] Would update older arch {arch_name} from delta")
            return True

        return update_older_arch_from_delta(delta_file, arch_name, config)


def handle_direct_edits(arch_name, modified_files, config, dry_run=False):
    """Handle direct YAML edits."""
    print(f"\n{'=' * 80}")
    print(f"DIRECT EDITS DETECTED: {arch_name}")
    print("=" * 80)
    print("Modified files:")
    for f in modified_files:
        print(f"   • {f}")

    latest = get_latest_arch(config["paths"]["template"])

    if not latest:
        latest = get_all_archs(config["paths"]["configs_root"])[-1]

    if arch_name == latest:
        print(f"\nThis is the current latest architecture ({latest}).")
        print("Are you:")
        print("  1. Updating the existing latest arch")
        print("  2. Creating a new architecture (this will become the new latest)")

        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                # Update existing latest arch
                if dry_run:
                    print(f"[DRY RUN] Would update latest arch {latest} from direct edits")
                    return True
                return update_latest_arch_from_edits(arch_name, config)
            elif choice == "2":
                # This is actually a new arch
                new_arch_name = input(f"Enter new architecture name (currently detected as {arch_name}): ").strip()
                if not new_arch_name:
                    new_arch_name = arch_name

                # Verify this is intentional
                if not prompt_yes_no(f"Promote {new_arch_name} to new latest architecture?"):
                    print("Operation cancelled.")
                    return False

                if dry_run:
                    print(f"[DRY RUN] Would promote {new_arch_name} to latest")
                    return True

                return promote_to_latest(new_arch_name, config)
            else:
                print("Invalid choice. Please enter 1 or 2.")
    else:
        if not prompt_yes_no(f"These are edits to older arch ({arch_name}). Continue?"):
            return False

        if dry_run:
            print(f"[DRY RUN] Would update older arch {arch_name} from direct edits")
            return True

        return update_older_arch_from_edits(arch_name, config)


# ============================================================================
# WORKFLOW OPERATIONS
# ============================================================================


def promote_to_latest(new_arch, config):
    """Scenario C: Promote new arch to latest."""
    print(f"\nPROMOTING {new_arch} TO LATEST ARCHITECTURE...")

    backup_paths = [config["paths"]["configs_root"], config["paths"]["template"]]
    backup_path = create_backup(backup_paths, config["paths"]["backups"])

    try:
        # Get all previous archs
        all_archs = get_all_archs(config["paths"]["configs_root"])
        previous_archs = [a for a in all_archs if a != new_arch]

        print(f"\n1. Updating template with new latest arch: {new_arch}")

        # Update template using parse_config_template
        new_arch_dir = Path(config["paths"]["configs_root"]) / new_arch
        result = run_script(
            "utils/config_management/parse_config_template.py",
            [str(new_arch_dir), config["paths"]["template"], "--latest-arch", new_arch],
            capture_output=True,
        )

        if result.returncode != 0:
            raise Exception(f"Failed to update template: {result.stderr}")

        # Update latest_arch in hash db
        hash_db = hash_manager.load_hash_db(config["paths"]["hashes"])
        hash_manager.save_hash_db(config["paths"]["hashes"], hash_db)

        print(
            f"\n2. Generating deltas for {len(previous_archs)} previous architectures"
        )

        # Generate deltas for each previous arch
        for prev_arch in previous_archs:
            print(f"   Generating delta: {new_arch} → {prev_arch}")
            prev_arch_dir = Path(config["paths"]["configs_root"]) / prev_arch

            result = run_script(
                "utils/config_management/generate_config_deltas.py",
                [str(new_arch_dir), str(prev_arch_dir)],
                capture_output=True,
            )

            if result.returncode != 0:
                raise Exception(
                    f"Failed to generate delta for {prev_arch}: {result.stderr}"
                )

        print("\n3. Validating all architectures against new template")

        # Validate all archs
        valid, msg = validate_all_archs(config)
        if not valid:
            raise Exception(msg)

        print("\n4. Syncing metric descriptions")

        # Sync metric descriptions for new latest arch
        success = metric_description_manager.sync_arch(
            new_arch,
            config["paths"]["configs_root"],
            config["paths"]["per_arch_metrics"],
            config["paths"]["docs_metrics"],
            is_latest=True,
        )

        if not success:
            raise Exception("Failed to sync metric descriptions")

        print("\n5. Updating hashes")

        # Update hashes for new arch
        hash_manager.update_hashes(
            new_arch, config["paths"]["configs_root"], config["paths"]["hashes"]
        )

        print(f"\nSuccessfully promoted {new_arch} to latest architecture!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Restoring from backup...")
        restore_backup(backup_path, backup_paths)
        return False


def update_latest_arch_from_delta(delta_file, arch_name, config):
    """Scenario A/C.1.1: Update latest arch using delta."""
    print(f"\nUPDATING LATEST ARCH {arch_name} FROM DELTA...")

    backup_paths = [config["paths"]["configs_root"], config["paths"]["template"]]
    backup_path = create_backup(backup_paths, config["paths"]["backups"])

    try:
        arch_dir = Path(config["paths"]["configs_root"]) / arch_name
        temp_output = arch_dir.parent / f"{arch_name}_temp"

        print(f"\n1. Applying delta to {arch_name}")

        # Apply delta
        result = run_script(
            "utils/config_management/apply_config_deltas.py",
            [str(arch_dir), delta_file, str(temp_output)],
            capture_output=True,
        )

        if result.returncode != 0:
            raise Exception(f"Failed to apply delta: {result.stderr}")

        # Replace original with temp
        shutil.rmtree(arch_dir)
        shutil.move(str(temp_output), str(arch_dir))

        print("\n2. Updating template")

        # Update template
        result = run_script(
            "utils/config_management/parse_config_template.py",
            [str(arch_dir), config["paths"]["template"], "--latest-arch", arch_name],
            capture_output=True,
        )

        if result.returncode != 0:
            raise Exception(f"Failed to update template: {result.stderr}")

        print("\n3. Regenerating deltas for previous architectures")

        # Regenerate deltas for all previous archs
        all_archs = get_all_archs(config["paths"]["configs_root"])
        previous_archs = [a for a in all_archs if a != arch_name]

        for prev_arch in previous_archs:
            print(f"   Generating delta: {arch_name} → {prev_arch}")
            prev_arch_dir = Path(config["paths"]["configs_root"]) / prev_arch

            result = run_script(
                "utils/config_management/generate_config_deltas.py",
                [str(arch_dir), str(prev_arch_dir)],
                capture_output=True,
            )

            if result.returncode != 0:
                raise Exception(f"Failed to generate delta for {prev_arch}")

        print("\n4. Validating all architectures")

        valid, msg = validate_all_archs(config)
        if not valid:
            raise Exception(msg)

        print("\n5. Syncing metric descriptions")

        success = metric_description_manager.sync_arch(
            arch_name,
            config["paths"]["configs_root"],
            config["paths"]["per_arch_metrics"],
            config["paths"]["docs_metrics"],
            is_latest=True,
        )

        if not success:
            raise Exception("Failed to sync metric descriptions")

        print("\n6. Updating hashes")

        hash_manager.update_hashes(
            arch_name, config["paths"]["configs_root"], config["paths"]["hashes"]
        )

        print(f"\nSuccessfully updated latest arch {arch_name}!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Restoring from backup...")
        restore_backup(backup_path, backup_paths)
        return False


def update_older_arch_from_delta(delta_file, arch_name, config):
    """Scenario B: Update older arch using delta."""
    print(f"\nUPDATING OLDER ARCH {arch_name} FROM DELTA...")

    arch_dir = Path(config["paths"]["configs_root"]) / arch_name
    backup_paths = [arch_dir]
    backup_path = create_backup(backup_paths, config["paths"]["backups"])

    try:
        temp_output = arch_dir.parent / f"{arch_name}_temp"

        print(f"\n1. Applying delta to {arch_name}")

        # Apply delta
        result = run_script(
            "utils/config_management/apply_config_deltas.py",
            [str(arch_dir), delta_file, str(temp_output)],
            capture_output=True,
        )

        if result.returncode != 0:
            raise Exception(f"Failed to apply delta: {result.stderr}")

        # Replace original with temp
        shutil.rmtree(arch_dir)
        shutil.move(str(temp_output), str(arch_dir))

        print("\n2. Validating against template")

        valid, msg = validate_arch_against_template(arch_name, config)
        if not valid:
            raise Exception(msg)

        print("\n3. Syncing metric descriptions")

        success = metric_description_manager.sync_arch(
            arch_name,
            config["paths"]["configs_root"],
            config["paths"]["per_arch_metrics"],
            config["paths"]["docs_metrics"],
            is_latest=False,
        )

        if not success:
            raise Exception("Failed to sync metric descriptions")

        print("\n4. Updating hashes")

        hash_manager.update_hashes(
            arch_name, config["paths"]["configs_root"], config["paths"]["hashes"]
        )

        print(f"\nSuccessfully updated older arch {arch_name}!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Restoring from backup...")
        restore_backup(backup_path, backup_paths)
        return False


def update_latest_arch_from_edits(arch_name, config):
    """Update latest arch from direct edits."""
    print(f"\nUPDATING LATEST ARCH {arch_name} FROM DIRECT EDITS...")

    backup_paths = [config["paths"]["configs_root"], config["paths"]["template"]]
    backup_path = create_backup(backup_paths, config["paths"]["backups"])

    try:
        arch_dir = Path(config["paths"]["configs_root"]) / arch_name

        print("\n1. Updating template")

        result = run_script(
            "utils/config_management/parse_config_template.py",
            [str(arch_dir), config["paths"]["template"], "--latest-arch", arch_name],
            capture_output=True,
        )

        if result.returncode != 0:
            raise Exception(f"Failed to update template: {result.stderr}")

        print("\n2. Regenerating deltas for previous architectures")

        all_archs = get_all_archs(config["paths"]["configs_root"])
        previous_archs = [a for a in all_archs if a != arch_name]

        for prev_arch in previous_archs:
            print(f"   Generating delta: {arch_name} → {prev_arch}")
            prev_arch_dir = Path(config["paths"]["configs_root"]) / prev_arch

            result = run_script(
                "utils/config_management/generate_config_deltas.py",
                [str(arch_dir), str(prev_arch_dir)],
                capture_output=True,
            )

            if result.returncode != 0:
                raise Exception(f"Failed to generate delta for {prev_arch}")

        print("\n3. Validating all architectures")

        valid, msg = validate_all_archs(config)
        if not valid:
            raise Exception(msg)

        print("\n4. Syncing metric descriptions")

        success = metric_description_manager.sync_arch(
            arch_name,
            config["paths"]["configs_root"],
            config["paths"]["per_arch_metrics"],
            config["paths"]["docs_metrics"],
            is_latest=True,
        )

        if not success:
            raise Exception("Failed to sync metric descriptions")

        print("\n5. Updating hashes")

        hash_manager.update_hashes(
            arch_name, config["paths"]["configs_root"], config["paths"]["hashes"]
        )

        print(f"\nSuccessfully updated latest arch {arch_name}!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Restoring from backup...")
        restore_backup(backup_path, backup_paths)
        return False


def update_older_arch_from_edits(arch_name, config):
    """Update older arch from direct edits."""
    print(f"\nUPDATING OLDER ARCH {arch_name} FROM DIRECT EDITS...")

    arch_dir = Path(config["paths"]["configs_root"]) / arch_name
    backup_paths = [arch_dir]
    backup_path = create_backup(backup_paths, config["paths"]["backups"])

    try:
        print("\n1. Validating against template")

        valid, msg = validate_arch_against_template(arch_name, config)
        if not valid:
            raise Exception(msg)

        print("\n2. Syncing metric descriptions")

        success = metric_description_manager.sync_arch(
            arch_name,
            config["paths"]["configs_root"],
            config["paths"]["per_arch_metrics"],
            config["paths"]["docs_metrics"],
            is_latest=False,
        )

        if not success:
            raise Exception("Failed to sync metric descriptions")

        print("\n3. Updating hashes")

        hash_manager.update_hashes(
            arch_name, config["paths"]["configs_root"], config["paths"]["hashes"]
        )

        print(f"\nSuccessfully updated older arch {arch_name}!")
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Restoring from backup...")
        restore_backup(backup_path, backup_paths)
        return False


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


def main():
    """Main workflow orchestration."""
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

    # Load configuration
    config = load_config()

    if args.dry_run:
        print("\nDRY RUN MODE - No changes will be made\n")

    # Detect changes
    changes = detect_changes(config)
    has_changes = display_change_summary(changes)

    if not has_changes:
        return 0

    # Process changes
    success = True

    # Determine latest arch
    latest_arch = get_latest_arch(config["paths"]["template"])
    if not latest_arch:
        latest_arch = get_all_archs(config["paths"]["configs_root"])[-1]

    latest_arch_has_edits = latest_arch in changes.get("modified_archs", {})

    # Handle new archs
    for new_arch in changes.get("new_archs", []):
        if not handle_new_arch(new_arch, config, args.dry_run):
            success = False
            break

    if not success:
        return 1

    # Handle direct edits to latest arch FIRST (if any)
    if latest_arch_has_edits:
        modified_files = changes["modified_archs"][latest_arch]
        if not handle_direct_edits(latest_arch, modified_files, config, args.dry_run):
            success = False
            return 1

        # Skip delta files for older archs since they'll be regenerated
        print("\nNote: Delta files for older archs will be regenerated automatically.")
        print("Skipping delta file processing for older architectures.\n")
    else:
        # Only process delta files if latest arch
        # Handle delta files
        for arch, delta_file in changes.get("delta_files", {}).items():
            if not handle_delta_file(delta_file, arch, config, args.dry_run):
                success = False
                break

        if not success:
            return 1

    # Handle direct edits
    for arch, modified_files in changes.get("modified_archs", {}).items():
        # Skip if already handled (latest arch with direct edits)
        if arch == latest_arch and latest_arch_has_edits:
            continue

        # Skip if already handled via delta
        if arch in changes.get("delta_files", {}):
            continue

        if not handle_direct_edits(arch, modified_files, config, args.dry_run):
            success = False
            break

    if success and not args.dry_run:
        cleanup_old_backups(config["paths"]["backups"])
        print("\n" + "=" * 80)
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        return 0
    elif args.dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN COMPLETE")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("OPERATIONS FAILED - CHANGES REVERTED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
