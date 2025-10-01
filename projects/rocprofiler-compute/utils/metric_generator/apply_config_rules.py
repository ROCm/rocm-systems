import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import yaml


def update_nested_dict(
    dictionary: dict[str, Any], keys: list[str], new_value: Any
) -> dict[str, Any]:
    """Update a nested dictionary value using a list of keys."""
    if not keys:
        raise ValueError("Keys list cannot be empty")

    current = dictionary
    for key in keys[:-1]:
        if not isinstance(current, dict):
            raise TypeError(f"Cannot navigate through non-dict value at key '{key}'")
        if key not in current:
            raise KeyError(f"Key '{key}' not found in dictionary")
        current = current[key]

    if not isinstance(current, dict):
        raise TypeError(f"Cannot set key '{keys[-1]}' on non-dict value")

    current[keys[-1]] = new_value
    return dictionary


def get_nested_value(d: dict[str, Any], keys: list[str]) -> Optional[Any]:
    """Get value from nested dictionary using list of keys."""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None
    return d


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply YAML configuration modifications based on rules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--rules", "-r", type=Path, required=True, help="Path to the rules YAML file"
    )

    parser.add_argument(
        "--config-dir",
        "-c",
        type=Path,
        required=True,
        help="Directory containing YAML configuration files to modify",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Output directory for modified files (default: current directory)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be modified without writing files",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> bool:
    """Validate that input paths exist."""
    errors = []

    if not args.rules.exists():
        errors.append(f"Rules file not found: {args.rules}")

    if not args.config_dir.exists():
        errors.append(f"Config directory not found: {args.config_dir}")
    elif not args.config_dir.is_dir():
        errors.append(f"Config path is not a directory: {args.config_dir}")

    if errors:
        for error in errors:
            print(f"Error: {error}", file=sys.stderr)
        return False

    return True


def load_rules(rules_path: Path) -> dict[str, Any]:
    """Load rules from YAML file."""
    try:
        with rules_path.open() as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading rules file: {e}", file=sys.stderr)
        sys.exit(1)


def find_yaml_files(config_dir: Path, exclude_name: str) -> list[Path]:
    """Find all YAML files in directory, optionally excluding a specific filename."""
    yaml_files = [
        f
        for f in config_dir.glob("*.yaml")
        if exclude_name is None or f.name != exclude_name
    ]

    if not yaml_files:
        print(f"No YAML files found in {config_dir}")

    return yaml_files


def load_panel_from_file(
    yaml_file: Path, verbose: bool = False
) -> Optional[dict[str, Any]]:
    """Load and validate a panel configuration from a YAML file."""
    try:
        with yaml_file.open() as f:
            yaml_obj = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load {yaml_file.name}: {e}")
        return None

    if "Panel Config" not in yaml_obj:
        if verbose:
            print(f"Skipping {yaml_file.name} - no 'Panel Config' section")
        return None

    return yaml_obj


def build_panel_mappings(
    yaml_files: list[Path], verbose: bool = False
) -> tuple[dict[str, dict], dict[str, dict[str, int]], dict[str, Path]]:
    """Load all panels and build table mappings."""
    panels = {}
    table_mapping = {}
    file_mapping = {}

    for yaml_file in yaml_files:
        yaml_obj = load_panel_from_file(yaml_file, verbose)
        if not yaml_obj:
            continue

        panel_config = yaml_obj["Panel Config"]
        panel_title = panel_config.get("title", "Untitled")

        print(f"Loading panel: {panel_title}")
        if verbose:
            print(f"  From file: {yaml_file.name}")

        file_mapping[panel_title] = yaml_file

        # Build table mapping
        table_map = {}
        for i, source in enumerate(panel_config.get("data source", [])):
            if "metric_table" in source:
                table_title = source["metric_table"].get("title", f"Table_{i}")
                table_map[table_title] = i
                if verbose:
                    print(f"  Table: {table_title}")

        table_mapping[panel_title] = table_map
        panels[panel_title] = yaml_obj

    return panels, table_mapping, file_mapping


def apply_single_modification(
    rule: dict[str, Any],
    panels: dict[str, dict],
    table_mapping: dict[str, dict[str, int]],
    rule_num: int,
) -> bool:
    """Apply a single modification rule. Returns True if successful."""
    print(f"\nRule {rule_num}: {rule}")

    panel_title = rule.get("panel_title")
    field_hierarchy = rule.get("field_hierarychy", "")  # Note typo
    new_value = rule.get("new_value")

    if not panel_title or not field_hierarchy:
        print("  Skipping - missing panel_title or field_hierarychy")
        return False

    fields = field_hierarchy.split(",")
    if len(fields) < 2:
        print(f"  Skipping - invalid field hierarchy: {field_hierarchy}")
        return False

    table_name = fields[0]
    field_path = fields[1:]

    # Validate panel and table exist
    if panel_title not in table_mapping:
        print(f"  Warning: Panel '{panel_title}' not found")
        return False

    if table_name not in table_mapping[panel_title]:
        print(f"  Warning: Table '{table_name}' not found in panel '{panel_title}'")
        return False

    # Get target table
    table_idx = table_mapping[panel_title][table_name]
    target_table = panels[panel_title]["Panel Config"]["data source"][table_idx][
        "metric_table"
    ]

    # Show before value
    before_value = get_nested_value(target_table, field_path)
    print(f"  Before: {before_value}")

    # Update value
    try:
        update_nested_dict(target_table, field_path, new_value)
        after_value = get_nested_value(target_table, field_path)
        print(f"  After: {after_value}")
        return True
    except (KeyError, TypeError) as e:
        print(f"  Error updating: {e}")
        return False


def apply_modifications(
    rules: dict[str, Any],
    panels: dict[str, dict],
    table_mapping: dict[str, dict[str, int]],
) -> set[str]:
    """Apply all modifications and return set of modified panel titles."""
    modified_panels = set()
    modifications = rules.get("rule", {}).get("Modification", [])

    print(f"\nApplying {len(modifications)} modification(s)...")

    for i, rule in enumerate(modifications, 1):
        panel_title = rule.get("panel_title")
        if apply_single_modification(rule, panels, table_mapping, i):
            if panel_title:
                modified_panels.add(panel_title)

    return modified_panels


def save_modified_panel(
    panel_title: str,
    panel_data: dict[str, Any],
    output_dir: Path,
    dry_run: bool = False,
) -> bool:
    """Save a single modified panel. Returns True if successful."""
    safe_title = panel_title.lower().replace(" ", "_").replace("/", "_")
    output_filename = f"{safe_title}_modified.yaml"
    output_path = output_dir / output_filename

    print(f"  {'Would write' if dry_run else 'Writing'}: {output_path}")

    if not dry_run:
        try:
            with output_path.open("w") as f:
                yaml.safe_dump(panel_data, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            print(f"    Error writing file: {e}", file=sys.stderr)
            return False

    return True


def save_modified_panels(
    modified_panels: set[str],
    panels: dict[str, dict],
    file_mapping: dict[str, Path],
    output_dir: Path,
    dry_run: bool = False,
) -> int:
    """Save all modified panels. Returns count of successfully saved files."""
    if not modified_panels:
        print("\nNo panels were modified")
        return 0

    print(
        f"\n{'Would save' if dry_run else 'Saving'} {len(modified_panels)} modified panel(s)..."
    )

    saved_count = 0
    for panel_title in modified_panels:
        if panel_title not in file_mapping:
            continue

        if save_modified_panel(panel_title, panels[panel_title], output_dir, dry_run):
            saved_count += 1

    return saved_count


def setup_output_directory(output_dir: Optional[Path], dry_run: bool) -> Path:
    """Setup output directory, creating if needed."""
    if not output_dir:
        output_dir = Path.cwd()

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def print_configuration(args: argparse.Namespace, output_dir: Path) -> None:
    """Print configuration summary."""
    print(f"Rules file: {args.rules}")
    print(f"Config directory: {args.config_dir}")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be written")
    print("-" * 60)


def main() -> None:
    # Parse and validate arguments
    args = parse_arguments()
    if not validate_paths(args):
        sys.exit(1)

    # Setup output directory
    output_dir = setup_output_directory(args.output_dir, args.dry_run)
    print_configuration(args, output_dir)

    # Load rules
    rules = load_rules(args.rules)

    # Find YAML files
    yaml_files = find_yaml_files(args.config_dir, args.rules.name)
    if not yaml_files:
        sys.exit(0)

    print(f"Found {len(yaml_files)} YAML file(s) to process\n")

    # Load panels and build mappings
    panels, table_mapping, file_mapping = build_panel_mappings(yaml_files, args.verbose)
    if not panels:
        print("No valid panels found to process")
        sys.exit(0)

    # Apply modifications
    modified_panels = apply_modifications(rules, panels, table_mapping)

    # Save modified panels
    saved_count = save_modified_panels(
        modified_panels, panels, file_mapping, output_dir, args.dry_run
    )

    print(f"\nCompleted. Output directory: {output_dir}")
    if saved_count > 0:
        print(
            f"Successfully {'would save' if args.dry_run else 'saved'} {saved_count} file(s)"
        )


if __name__ == "__main__":
    main()
