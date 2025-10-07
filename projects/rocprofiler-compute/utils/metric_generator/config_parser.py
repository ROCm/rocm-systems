import sys
from pathlib import Path

import yaml


def parse_panel_config(yaml_file):
    """Parse a single YAML file and extract panel and data source info."""
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    if not data or "Panel Config" not in data:
        return None

    panel_config = data["Panel Config"]

    # Strip panel ID prefix from filename
    filename = (
        yaml_file.name.split("_", 1)[1] if "_" in yaml_file.name else yaml_file.name
    )

    # Normalize panel ID
    panel_id = panel_config.get("id")
    if panel_id and panel_id >= 100:
        panel_id = panel_id // 100

    # Parse data sources
    data_sources = []
    for data_source in panel_config.get("data source", []):
        for key, value in data_source.items():
            if isinstance(value, dict) and "id" in value and "title" in value:
                data_sources.append({
                    "type": key,
                    "id": value["id"] % 100,
                    "title": value["title"],
                })

    return {
        "file": filename,
        "panel_id": panel_id,
        "panel_title": panel_config.get("title"),
        "data_sources": data_sources,
    }


def parse_directory(directory_path):
    """Parse all YAML files in a directory."""
    directory = Path(directory_path)

    if not directory.is_dir():
        print(f"Error: '{directory_path}' is not a valid directory")
        return []

    results = []
    for yaml_file in sorted(directory.glob("*.yaml")):
        parsed = parse_panel_config(yaml_file)
        if parsed:
            results.append(parsed)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_yaml.py <directory_path> [output_file.yaml]")
        sys.exit(1)

    results = parse_directory(sys.argv[1])

    if not results:
        print("No valid panel configurations found.")
        sys.exit(1)

    # Display results
    for panel in results:
        print(f"\n{'=' * 80}")
        print(f"File: {panel['file']}")
        print(f"Panel ID: {panel['panel_id']}")
        print(f"Panel Title: {panel['panel_title']}")
        print(f"\nData Sources ({len(panel['data_sources'])}):")
        for ds in panel["data_sources"]:
            print(f"  - {ds['type']}: {ds['id']} - {ds['title']}")

    # Save if output file specified
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            yaml.dump(results, f, sort_keys=False, default_flow_style=False)
        print(f"\nResults saved to: {sys.argv[2]}")
