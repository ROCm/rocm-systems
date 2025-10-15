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
Apply delta YAML to base architecture to produce target architecture.
Usage: python apply_config_deltas.py <base_arch_dir> <delta_yaml> <output_dir>
"""

import shutil
import sys
from pathlib import Path

import yaml


def load_yaml(filepath):
    with open(filepath) as f:
        return yaml.safe_load(f)


def save_yaml(data, filepath):
    with open(filepath, "w") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=float("inf"),
        )


def find_table_in_config(config, table_id):
    """Find and return the table with given id, or None."""
    for item in config["Panel Config"]["data source"]:
        if "metric_table" in item:
            table = item["metric_table"]
            if table.get("id") == table_id:
                return table
    return None


def add_table(config, metric_table):
    """Add entire new table to config."""
    config["Panel Config"]["data source"].append({"metric_table": metric_table})
    print(f"Added table: {metric_table.get('id')} - {metric_table.get('title')}")


def add_metrics(config, table_id, metrics):
    """Add metrics to existing table."""
    table = find_table_in_config(config, table_id)
    if not table:
        print(f"WARNING: Table {table_id} not found for metric addition")
        return

    if "metric" not in table:
        table["metric"] = {}

    for metric_dict in metrics:
        for metric_name, metric_data in metric_dict.items():
            table["metric"][metric_name] = metric_data
            print(f"Added metric: {metric_name} to table {table_id}")


def delete_table(config, table_id):
    """Remove entire table from config."""
    data_source = config["Panel Config"]["data source"]
    for idx, item in enumerate(data_source):
        if "metric_table" in item and item["metric_table"].get("id") == table_id:
            data_source.pop(idx)
            print(f"Deleted table: {table_id}")
            return
    print(f"WARNING: Table {table_id} not found for deletion")


def delete_metrics(config, table_id, metrics):
    """Remove specific metrics from table."""
    table = find_table_in_config(config, table_id)
    if not table or "metric" not in table:
        print(f"WARNING: Table {table_id} not found or has no metrics")
        return

    for metric_dict in metrics:
        for metric_name in metric_dict.keys():
            if metric_name in table["metric"]:
                del table["metric"][metric_name]
                print(f"Deleted metric: {metric_name} from table {table_id}")


def modify_metrics(config, table_id, metrics):
    """Modify specific fields in existing metrics."""
    table = find_table_in_config(config, table_id)
    if not table or "metric" not in table:
        print(f"WARNING: Table {table_id} not found or has no metrics")
        return

    for metric_dict in metrics:
        for metric_name, new_fields in metric_dict.items():
            if metric_name not in table["metric"]:
                print(f"WARNING: Metric '{metric_name}' not found in table {table_id}")
                continue

            for field_name, field_value in new_fields.items():
                table["metric"][metric_name][field_name] = field_value
                print(f"Modified {metric_name}.{field_name} in table {table_id}")


def add_descriptions(config, descriptions):
    """Add metric descriptions to config."""
    if "metrics_description" not in config["Panel Config"]:
        config["Panel Config"]["metrics_description"] = {}

    for metric_name, desc_data in descriptions.items():
        # Store only plain text in config YAML
        if isinstance(desc_data, dict):
            plain_text = desc_data.get("plain", "")
        else:
            plain_text = desc_data

        config["Panel Config"]["metrics_description"][metric_name] = plain_text
        print(f"Added description: {metric_name}")


def delete_descriptions(config, descriptions):
    """Remove metric descriptions from config."""
    if "metrics_description" not in config["Panel Config"]:
        return

    for metric_name in descriptions.keys():
        if metric_name in config["Panel Config"]["metrics_description"]:
            del config["Panel Config"]["metrics_description"][metric_name]
            print(f"Deleted description: {metric_name}")


def modify_descriptions(config, descriptions):
    """Modify metric descriptions in config."""
    if "metrics_description" not in config["Panel Config"]:
        config["Panel Config"]["metrics_description"] = {}

    for metric_name, desc_data in descriptions.items():
        # Store only plain text in config YAML
        if isinstance(desc_data, dict):
            plain_text = desc_data.get("plain", "")
        else:
            plain_text = desc_data

        config["Panel Config"]["metrics_description"][metric_name] = plain_text
        print(f"Modified description: {metric_name}")


def apply_changes(config, changes, category):
    """Apply delta changes to configuration."""
    for change in changes:
        # Handle metric tables
        for mt_wrapper in change.get("metric_tables", []):
            mt = mt_wrapper.get("metric_table", {})
            table_id = mt.get("id")

            if category == "Addition":
                if "metrics" in mt:
                    add_metrics(config, table_id, mt["metrics"])
                elif "metric" in mt:
                    add_table(config, mt)

            elif category == "Deletion":
                if "metrics" in mt:
                    delete_metrics(config, table_id, mt["metrics"])
                else:
                    delete_table(config, table_id)

            elif category == "Modification":
                if "metrics" in mt:
                    modify_metrics(config, table_id, mt["metrics"])

        # Handle metric descriptions
        descriptions = change.get("metric_descriptions", {})
        if descriptions:
            if category == "Addition":
                add_descriptions(config, descriptions)
            elif category == "Deletion":
                delete_descriptions(config, descriptions)
            elif category == "Modification":
                modify_descriptions(config, descriptions)


def apply_delta(base_dir, delta_file, output_dir):
    """Apply delta YAML to all files in base directory."""
    delta = load_yaml(delta_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group changes by panel ID
    changes_by_panel = {}
    for category in ["Addition", "Deletion", "Modification"]:
        for change in delta.get(category, []):
            panel_id = change.get("panel_config", {}).get("id")
            if panel_id not in changes_by_panel:
                changes_by_panel[panel_id] = {
                    "Addition": [],
                    "Deletion": [],
                    "Modification": [],
                }
            changes_by_panel[panel_id][category].append(change)

    # Process each YAML file
    base_path = Path(base_dir)
    for yaml_file in base_path.glob("*.yaml"):
        config = load_yaml(yaml_file)
        panel_id = config.get("Panel Config", {}).get("id")

        if panel_id in changes_by_panel:
            print(f"\nApplying deltas to {yaml_file.name} (Panel ID: {panel_id})")
            for category in ["Deletion", "Modification", "Addition"]:
                if changes_by_panel[panel_id][category]:
                    apply_changes(
                        config, changes_by_panel[panel_id][category], category
                    )

            save_yaml(config, output_path / yaml_file.name)
            print(f"Saved: {yaml_file.name}")
        else:
            shutil.copy(yaml_file, output_path / yaml_file.name)


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python apply_config_deltas.py "
            "<base_arch_dir> <delta_yaml> <output_dir>"
        )
        sys.exit(1)

    base_dir, delta_file, output_dir = sys.argv[1:4]

    if not Path(base_dir).is_dir():
        print(f"Error: {base_dir} is not a directory")
        sys.exit(1)

    if not Path(delta_file).is_file():
        print(f"Error: {delta_file} is not a file")
        sys.exit(1)

    apply_delta(base_dir, delta_file, output_dir)
    print("\nDelta application complete!")


if __name__ == "__main__":
    main()
