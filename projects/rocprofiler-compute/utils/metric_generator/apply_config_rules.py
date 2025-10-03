#!/usr/bin/env python3
"""
Apply delta YAML to base architecture to produce target architecture.
Usage: python apply_delta.py <base_arch_dir> <delta_yaml> <output_dir>
"""

import os
import shutil
import sys
from pathlib import Path

import yaml


def load_yaml(filepath: str) -> dict:
    """Load YAML file and return as dictionary."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, filepath: str):
    """Save dictionary as YAML file with clean formatting."""
    with open(filepath, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            default_style="",  # Don't add quotes
            width=float("inf"),
        )


def find_table_by_id(tables: list[dict], table_id: int) -> tuple:
    """Find table in list by id. Returns (index, table) or (None, None)."""
    for idx, table in enumerate(tables):
        if table.get("id") == table_id:
            return idx, table
    return None, None


def apply_table_level_addition(config: dict, metric_table: dict):
    """Add entire new table to config."""
    if "Panel Config" not in config or "data source" not in config["Panel Config"]:
        print("[ERROR] Invalid config structure")
        return

    # Add as new metric_table entry
    config["Panel Config"]["data source"].append({"metric_table": metric_table})
    print(
        f"[DEBUG] Added table-level: {metric_table.get('id')} - "
        f"{metric_table.get('title')}"
    )


def apply_metric_level_addition(config: dict, table_id: int, metrics: list[dict]):
    """Add new metrics to existing table."""
    if "Panel Config" not in config or "data source" not in config["Panel Config"]:
        print("[ERROR] Invalid config structure")
        return

    # Find the table
    for item in config["Panel Config"]["data source"]:
        if "metric_table" in item:
            table = item["metric_table"]
            if table.get("id") == table_id:
                # Found the table, add metrics
                if "metric" not in table:
                    table["metric"] = {}

                for metric_dict in metrics:
                    for metric_name, metric_data in metric_dict.items():
                        table["metric"][metric_name] = metric_data
                        print(
                            f"[DEBUG] Added metric: {metric_name} to table {table_id}"
                        )
                return

    print(f"[WARNING] Table {table_id} not found for metric-level addition")


def apply_table_level_deletion(config: dict, table_id: int):
    """Remove entire table from config."""
    if "Panel Config" not in config or "data source" not in config["Panel Config"]:
        print("[ERROR] Invalid config structure")
        return

    data_source = config["Panel Config"]["data source"]
    for idx, item in enumerate(data_source):
        if "metric_table" in item:
            table = item["metric_table"]
            if table.get("id") == table_id:
                data_source.pop(idx)
                print(f"[DEBUG] Deleted table-level: {table_id} - {table.get('title')}")
                return

    print(f"[WARNING] Table {table_id} not found for deletion")


def apply_metric_level_deletion(config: dict, table_id: int, metrics: list[dict]):
    """Remove specific metrics from existing table."""
    if "Panel Config" not in config or "data source" not in config["Panel Config"]:
        print("[ERROR] Invalid config structure")
        return

    # Find the table
    for item in config["Panel Config"]["data source"]:
        if "metric_table" in item:
            table = item["metric_table"]
            if table.get("id") == table_id:
                # Found the table, delete metrics
                if "metric" not in table:
                    print(f"[WARNING] Table {table_id} has no metrics to delete")
                    return

                for metric_dict in metrics:
                    for metric_name in metric_dict.keys():
                        if metric_name in table["metric"]:
                            del table["metric"][metric_name]
                            print(
                                f"[DEBUG] Deleted metric: {metric_name} from "
                                f"table {table_id}"
                            )
                        else:
                            print(
                                f"[WARNING] Metric {metric_name} not found in "
                                f"table {table_id}"
                            )
                return

    print(f"[WARNING] Table {table_id} not found for metric-level deletion")


def apply_modification(config: dict, table_id: int, metrics: list[dict]):
    """Modify specific fields in existing metrics."""
    if "Panel Config" not in config or "data source" not in config["Panel Config"]:
        print("[ERROR] Invalid config structure")
        return

    print(f"[DEBUG] Looking for table {table_id} to apply modifications...")

    # Find the table
    for item in config["Panel Config"]["data source"]:
        if "metric_table" in item:
            table = item["metric_table"]
            if table.get("id") == table_id:
                print(f"[DEBUG] Found table {table_id}")

                # Found the table, modify metrics
                if "metric" not in table:
                    print(f"[WARNING] Table {table_id} has no metrics to modify")
                    return

                print(f"[DEBUG] Table {table_id} has {len(table['metric'])} metrics")

                for metric_dict in metrics:
                    for metric_name, new_fields in metric_dict.items():
                        print(f"[DEBUG] Attempting to modify metric: {metric_name}")

                        if metric_name not in table["metric"]:
                            print(
                                f"[WARNING] Metric '{metric_name}' not found in "
                                f"table {table_id}"
                            )
                            print(
                                "[DEBUG] Available metrics: "
                                f"{list(table['metric'].keys())}"
                            )
                            continue

                        print(
                            f"[DEBUG] Found metric '{metric_name}', updating fields: "
                            f"{list(new_fields.keys())}"
                        )

                        # Update only the specified fields
                        for field_name, field_value in new_fields.items():
                            old_value = table["metric"][metric_name].get(
                                field_name, "N/A"
                            )
                            table["metric"][metric_name][field_name] = field_value
                            print(f"[DEBUG] Modified {metric_name}.{field_name}")
                            print(f"[DEBUG]   Old: {old_value}")
                            print(f"[DEBUG]   New: {field_value}")
                return

    print(f"[WARNING] Table {table_id} not found for modification")


def apply_delta_to_config(config: dict, delta_changes: list[dict], category: str):
    """Apply delta changes to a configuration in-place."""
    for change in delta_changes:
        panel_config = change.get("Panel Config", {})
        metric_tables = change.get("metric_tables", [])

        print(f"\n[DEBUG] Processing {category} for Panel {panel_config.get('id')}")

        for mt in metric_tables:
            mt = mt.get("metric_table", {})
            table_id = mt.get("id")

            if category == "Addition":
                # Check if this is table-level or metric-level addition
                if "metrics" in mt:
                    # Metric-level addition (has 'metrics' list with new metrics only)
                    apply_metric_level_addition(config, table_id, mt["metrics"])
                elif "metric" in mt:
                    # Table-level addition (has complete 'metric' dict)
                    apply_table_level_addition(config, mt)

            elif category == "Deletion":
                # Check if this is table-level or metric-level deletion
                if "metrics" in mt:
                    # Metric-level deletion
                    apply_metric_level_deletion(config, table_id, mt["metrics"])
                elif "metric" in mt:
                    # Table-level deletion
                    apply_table_level_deletion(config, table_id)

            elif category == "Modification":
                # Always metric-level for modifications
                if "metrics" in mt:
                    apply_modification(config, table_id, mt["metrics"])


def apply_delta(base_dir: str, delta_file: str, output_dir: str):
    """Apply delta YAML to all files in base directory."""
    print(f"[DEBUG] Loading delta file: {delta_file}")
    delta = load_yaml(delta_file)

    print(f"[DEBUG] Base directory: {base_dir}")
    print(f"[DEBUG] Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Group delta changes by panel ID
    files_to_process = {}

    # Process each category
    for category in ["Addition", "Deletion", "Modification"]:
        if category not in delta or not delta[category]:
            print(f"[DEBUG] No {category} changes in delta")
            continue

        print(f"[DEBUG] Processing {category}: {len(delta[category])} panel changes")

        for change in delta[category]:
            panel_id = change.get("Panel Config", {}).get("id")
            print(f"[DEBUG] {category} has panel_id: {panel_id}")

            if panel_id not in files_to_process:
                files_to_process[panel_id] = {
                    "Addition": [],
                    "Deletion": [],
                    "Modification": [],
                }

            files_to_process[panel_id][category].append(change)

    print(
        "\n[DEBUG] Panel IDs in delta that need processing: "
        f"{list(files_to_process.keys())}"
    )

    # Find all YAML files in base directory
    base_path = Path(base_dir)
    yaml_files = list(base_path.glob("*.yaml")) + list(base_path.glob("*.yml"))

    print(f"\n[DEBUG] Found {len(yaml_files)} YAML files in base directory")

    # Process each file
    for yaml_file in yaml_files:
        print(f"\n[DEBUG] Processing file: {yaml_file.name}")

        # Load the file to get its panel ID
        file_config = load_yaml(str(yaml_file))
        panel_id = file_config.get("Panel Config", {}).get("id")

        print(f"[DEBUG] File {yaml_file.name} has Panel Config ID: {panel_id}")

        if panel_id in files_to_process:
            print(
                f"[DEBUG] *** MATCH FOUND *** Applying deltas to {yaml_file.name} "
                f"(Panel ID: {panel_id})"
            )

            # Apply changes in order: Deletion, Modification, Addition
            for category in ["Deletion", "Modification", "Addition"]:
                if files_to_process[panel_id][category]:
                    print(f"\n[DEBUG] Applying {category}...")
                    apply_delta_to_config(
                        file_config, files_to_process[panel_id][category], category
                    )

            # Save modified file
            output_file = os.path.join(output_dir, yaml_file.name)
            save_yaml(file_config, output_file)
            print(f"[DEBUG] Saved: {output_file}")
        else:
            # No changes for this file, just copy it
            output_file = os.path.join(output_dir, yaml_file.name)
            shutil.copy(str(yaml_file), output_file)
            print(f"[DEBUG] Copied unchanged: {yaml_file.name}")

    print("\n[DEBUG] Delta application complete!")


def main():
    if len(sys.argv) != 4:
        print("Usage: python apply_delta.py <base_arch_dir> <delta_yaml> <output_dir>")
        print("  base_arch_dir: Directory with base architecture YAML files")
        print("  delta_yaml: Delta YAML file to apply")
        print("  output_dir: Directory to write modified YAML files")
        sys.exit(1)

    base_dir = sys.argv[1]
    delta_file = sys.argv[2]
    output_dir = sys.argv[3]

    # Validate inputs
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a directory")
        sys.exit(1)

    if not os.path.isfile(delta_file):
        print(f"Error: {delta_file} is not a file")
        sys.exit(1)

    apply_delta(base_dir, delta_file, output_dir)


if __name__ == "__main__":
    main()
