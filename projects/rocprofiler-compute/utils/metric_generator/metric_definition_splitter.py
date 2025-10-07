import sys
from pathlib import Path

import yaml


def str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_representer)


def load_metric_descriptions(unified_config_file):
    """Load metric descriptions from unified config, organized by panel ID"""
    with open(unified_config_file) as f:
        data = yaml.safe_load(f)

    # Build a map: panel_id -> metrics_description
    descriptions_by_panel = {}
    for panel in data.get("panels", []):
        panel_id = panel.get("id")
        if panel_id is not None and "metrics_description" in panel:
            descriptions_by_panel[panel_id] = panel["metrics_description"]

    return descriptions_by_panel


def parse_panel_yaml(yaml_file, descriptions_by_panel):
    """Parse a single panel YAML and extract metrics with descriptions"""
    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    if not data or "Panel Config" not in data:
        return {}

    panel_config = data["Panel Config"]
    panel_id = panel_config.get("id")
    panel_metrics_desc = descriptions_by_panel.get(panel_id, {})

    result = {}

    # Process each data source
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
                            f"WARNING: Metric '{metric_name}' not found in "
                            f"unified config for panel {panel_id}"
                        )
                        result[table_title][metric_name] = {
                            "rst": "",
                            "unit": "Unknown",
                        }

    return result


def aggregate_metrics_by_arch(analysis_configs_dir, unified_config_file, output_dir):
    """Aggregate metrics descriptions for each architecture"""
    analysis_configs_path = Path(analysis_configs_dir)
    unified_config_path = Path(unified_config_file)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metric descriptions from unified config
    print(f"Loading metric descriptions from {unified_config_path}")
    descriptions_by_panel = load_metric_descriptions(unified_config_path)
    print(f"Loaded descriptions for {len(descriptions_by_panel)} panels\n")

    # Process each architecture directory
    for arch_dir in sorted(analysis_configs_path.iterdir()):
        if not arch_dir.is_dir():
            continue

        arch_name = arch_dir.name
        print(f"Processing architecture: {arch_name}")

        aggregated_metrics = {}

        # Process each YAML file in the architecture directory
        yaml_files = sorted(arch_dir.glob("*.yaml"))
        for yaml_file in yaml_files:
            print(f"  - {yaml_file.name}")
            panel_metrics = parse_panel_yaml(yaml_file, descriptions_by_panel)

            # Merge into aggregated metrics
            for table_title, metrics in panel_metrics.items():
                if table_title in aggregated_metrics:
                    # Table already exists, append metrics (preserving order)
                    aggregated_metrics[table_title].update(metrics)
                else:
                    aggregated_metrics[table_title] = metrics

        # Write output file for this architecture
        if aggregated_metrics:
            output_file = output_path / f"{arch_name}_metrics_description.yaml"
            with open(output_file, "w") as f:
                yaml.dump(aggregated_metrics, f, sort_keys=False, allow_unicode=True)
            print(f"Written: {output_file}\n")
        else:
            print(f"No metrics found for {arch_name}\n")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python metric_definition_splitter.py <analysis_configs_dir> "
            "<unified_config_file> <output_dir>"
        )
        sys.exit(1)

    analysis_configs_dir = sys.argv[1]
    unified_config_file = sys.argv[2]
    output_dir = sys.argv[3]

    aggregate_metrics_by_arch(analysis_configs_dir, unified_config_file, output_dir)
    print("Aggregation complete!")
