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
Analaysis Config Differentiation Script
Generates differences from curr arch directory to prev arch directory.
Output shows what needs to change in prev arch to match curr arch.
"""

import sys
from pathlib import Path

import yaml


def load_yaml(filepath):
    with open(filepath) as f:
        return yaml.safe_load(f)


def get_metric_tables(data):
    """Extract all metric tables from data source."""
    tables = []
    for item in data.get("Panel Config", {}).get("data source", []):
        if "metric_table" in item:
            tables.append(item["metric_table"])
    return tables


def compare_metrics(prev_metrics, curr_metrics):
    """Compare metrics and return additions, deletions, and modifications."""
    prev_keys = set(prev_metrics.keys())
    curr_keys = set(curr_metrics.keys())

    additions = [{name: curr_metrics[name]} for name in curr_keys - prev_keys]
    deletions = [{name: prev_metrics[name]} for name in prev_keys - curr_keys]

    modifications = []
    for name in prev_keys & curr_keys:
        if prev_metrics[name] != curr_metrics[name]:
            # Find modified fields
            all_fields = set(prev_metrics[name].keys()) | set(curr_metrics[name].keys())
            modified_fields = {
                field: curr_metrics[name].get(field)
                for field in all_fields
                if prev_metrics[name].get(field) != curr_metrics[name].get(field)
            }
            if modified_fields:
                modifications.append({name: modified_fields})

    return additions, deletions, modifications


def compare_tables(prev_tables, curr_tables):
    """Compare tables and return differences."""
    prev_dict = {t["id"]: t for t in prev_tables}
    curr_dict = {t["id"]: t for t in curr_tables}

    prev_ids = set(prev_dict.keys())
    curr_ids = set(curr_dict.keys())

    additions = []
    deletions = []
    modifications = []

    # Table-level additions and deletions
    additions.extend(curr_dict[tid] for tid in curr_ids - prev_ids)
    deletions.extend(prev_dict[tid] for tid in prev_ids - curr_ids)

    # Compare common tables
    for tid in prev_ids & curr_ids:
        prev_metrics = prev_dict[tid].get("metric", {})
        curr_metrics = curr_dict[tid].get("metric", {})

        metric_adds, metric_dels, metric_mods = compare_metrics(
            prev_metrics, curr_metrics
        )

        if metric_adds:
            additions.append({
                "id": tid,
                "title": curr_dict[tid].get("title"),
                "metrics": metric_adds,
            })

        if metric_dels:
            deletions.append({
                "id": tid,
                "title": prev_dict[tid].get("title"),
                "metrics": metric_dels,
            })

        if metric_mods:
            modifications.append({
                "id": tid,
                "title": curr_dict[tid].get("title"),
                "metrics": metric_mods,
            })

    return additions, deletions, modifications


def format_metric_fields(metric_data):
    """Format metric fields as YAML lines."""
    lines = []
    for field_name, field_value in metric_data.items():
        if isinstance(field_value, str) and (
            "\n" in field_value or len(field_value) > 80
        ):
            lines.append(f"                {field_name}: |")
            lines.extend(
                f"                  {line}" for line in field_value.split("\n")
            )
        else:
            lines.append(f"                {field_name}: {field_value}")
    return lines


def format_output(combined_diff):
    """Format the diff dictionary into YAML string."""
    lines = []

    for category in ["Addition", "Deletion", "Modification"]:
        lines.append(f"{category}:")

        if not combined_diff.get(category):
            lines.append("  []")
            lines.append("")
            continue

        for panel_item in combined_diff[category]:
            pc = panel_item["panel_config"]
            lines.extend([
                "  - Panel Config:",
                f"      id: {pc['id']}",
                f"      title: {pc['title']}",
                "    metric_tables:",
            ])

            for mt in panel_item["metric_tables"]:
                lines.extend([
                    "      - metric_table:",
                    f"          id: {mt['id']}",
                    f"          title: {mt['title']}",
                    "          metrics:",
                ])

                # Handle metric-level changes or full table
                metrics_to_format = mt.get("metrics") or [
                    {name: data} for name, data in mt.get("metric", {}).items()
                ]

                for metric in metrics_to_format:
                    for metric_name, metric_data in metric.items():
                        lines.append(f"            - {metric_name}:")
                        lines.extend(format_metric_fields(metric_data))

        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_config_deltas.py <curr_arch_dir> <prev_arch_dir>")
        sys.exit(1)

    curr_arch_dir = Path(sys.argv[1])
    prev_arch_dir = Path(sys.argv[2])

    if not curr_arch_dir.is_dir() or not prev_arch_dir.is_dir():
        print("Error: Both arguments must be directories")
        sys.exit(1)

    # Get common YAML files
    curr_files = {f.name for f in curr_arch_dir.glob("*.yaml")}
    prev_files = {f.name for f in prev_arch_dir.glob("*.yaml")}
    common_files = curr_files & prev_files

    if not common_files:
        print("Error: No common YAML files found")
        sys.exit(1)

    print(f"Comparing {len(common_files)} files...")

    # Combined diff results
    combined_diff = {"Addition": [], "Deletion": [], "Modification": []}

    # Compare each file
    for filename in sorted(common_files):
        curr_data = load_yaml(curr_arch_dir / filename)
        prev_data = load_yaml(prev_arch_dir / filename)

        curr_pc = curr_data.get("Panel Config", {})
        prev_pc = prev_data.get("Panel Config", {})

        curr_tables = get_metric_tables(curr_data)
        prev_tables = get_metric_tables(prev_data)

        additions, deletions, modifications = compare_tables(prev_tables, curr_tables)

        if additions:
            combined_diff["Addition"].append({
                "panel_config": {
                    "id": curr_pc.get("id"),
                    "title": curr_pc.get("title"),
                },
                "metric_tables": additions,
            })

        if deletions:
            combined_diff["Deletion"].append({
                "panel_config": {
                    "id": prev_pc.get("id"),
                    "title": prev_pc.get("title"),
                },
                "metric_tables": deletions,
            })

        if modifications:
            combined_diff["Modification"].append({
                "panel_config": {
                    "id": curr_pc.get("id"),
                    "title": curr_pc.get("title"),
                },
                "metric_tables": modifications,
            })

    # Format and save output
    output = format_output(combined_diff)

    print("\n" + "=" * 80)
    print("COMBINED DIFF OUTPUT:")
    print("=" * 80)
    print(output)

    # Write to file
    output_dir = prev_arch_dir / "config_delta"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{curr_arch_dir.name}_diff.yaml"

    with open(output_file, "w") as f:
        f.write(output)

    print(f"\nDiff written to: {output_file}")


if __name__ == "__main__":
    main()
