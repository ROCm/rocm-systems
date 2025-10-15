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
Aggregate metric descriptions by architecture.
Extracts metrics from per-arch panel configs and combines with descriptions
from unified config to create arch-specific metric description files.
"""

import sys
from pathlib import Path

import yaml


def str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_representer)


def load_metric_descriptions(unified_config_file):
    """Load metric descriptions from unified config, organized by panel ID."""
    with open(unified_config_file) as f:
        data = yaml.safe_load(f)

    return {
        panel["id"]: panel["metrics_description"]
        for panel in data.get("panels", [])
        if panel.get("id") is not None and "metrics_description" in panel
    }


def parse_panel_yaml(yaml_file, descriptions_by_panel):
    """Parse panel YAML and extract metrics with descriptions."""
    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    if not data or "Panel Config" not in data:
        return {}

    panel_config = data["Panel Config"]
    panel_id = panel_config.get("id")
    panel_metrics_desc = descriptions_by_panel.get(panel_id, {})

    result = {}

    for data_source in panel_config.get("data source", []):
        for key, value in data_source.items():
            if isinstance(value, dict) and "title" in value and "metric" in value:
                table_title = value["title"]
                result[table_title] = {}

                for metric_name in value["metric"].keys():
                    if metric_name in panel_metrics_desc:
                        result[table_title][metric_name] = {
                            "rst": panel_metrics_desc[metric_name]["rst"].strip(),
                            "unit": panel_metrics_desc[metric_name]["unit"],
                        }
                    else:
                        print(
                            f"WARNING: Metric '{metric_name}' "
                            f"not found for panel {panel_id}"
                        )
                        result[table_title][metric_name] = {
                            "rst": "",
                            "unit": "Unknown",
                        }

    return result


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python metric_definition_splitter.py <analysis_configs_dir> "
            "<unified_config_file> <output_dir>"
        )
        sys.exit(1)

    analysis_configs_path = Path(sys.argv[1])
    unified_config_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    output_path.mkdir(parents=True, exist_ok=True)

    # Load metric descriptions
    print(f"Loading metric descriptions from {unified_config_path}")
    descriptions_by_panel = load_metric_descriptions(unified_config_path)
    print(f"Loaded descriptions for {len(descriptions_by_panel)} panels\n")

    # Process each architecture
    for arch_dir in sorted(analysis_configs_path.iterdir()):
        if not arch_dir.is_dir():
            continue

        print(f"Processing architecture: {arch_dir.name}")
        aggregated_metrics = {}

        for yaml_file in sorted(arch_dir.glob("*.yaml")):
            panel_metrics = parse_panel_yaml(yaml_file, descriptions_by_panel)

            for table_title, metrics in panel_metrics.items():
                if table_title in aggregated_metrics:
                    aggregated_metrics[table_title].update(metrics)
                else:
                    aggregated_metrics[table_title] = metrics

        if aggregated_metrics:
            output_file = output_path / f"{arch_dir.name}_metrics_description.yaml"
            with open(output_file, "w") as f:
                yaml.dump(aggregated_metrics, f, sort_keys=False, allow_unicode=True)
            print(f"  Written: {output_file}\n")
        else:
            print("  No metrics found\n")

    print("Aggregation complete!")


if __name__ == "__main__":
    main()
