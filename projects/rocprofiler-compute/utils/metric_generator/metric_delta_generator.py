#!/usr/bin/env python3
"""
YAML Differentiation Script
Generates differences from curr arch directory to prev arch directory.
Output shows what needs to change in prev arch to match curr arch.
"""

import logging
import sys
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_yaml(filepath: Path) -> dict:
    """Load YAML file and return as dictionary."""
    try:
        with open(filepath, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file {filepath}: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise


def get_yaml_files(directory: Path) -> list[str]:
    """Get all YAML files in a directory."""
    yaml_files = list(directory.glob("*.yaml"))
    return sorted([f.name for f in yaml_files])


def extract_panel_config(data: dict) -> dict[str, str]:
    """Extract Panel Config information."""
    if "Panel Config" not in data:
        raise ValueError("Missing 'Panel Config' in YAML data")

    pc = data["Panel Config"]
    return {"id": pc.get("id"), "title": pc.get("title")}


def extract_metric_tables(data: dict) -> list[dict]:
    """Extract all metric tables from data source."""
    if "Panel Config" not in data or "data source" not in data["Panel Config"]:
        return []

    tables = []
    for item in data["Panel Config"]["data source"]:
        if "metric_table" in item:
            tables.append(item["metric_table"])

    return tables


def compare_metrics(
    metrics_prev_arch: dict[str, dict],
    metrics_curr_arch: dict[str, dict],
    table_id: int,
    filename: str,
) -> dict[str, list]:
    """
    Compare metrics within a table.
    Returns additions (in curr arch only), deletions (in prev arch only),
    and modifications.
    """
    prev_arch_keys = set(metrics_prev_arch.keys())
    curr_arch_keys = set(metrics_curr_arch.keys())

    additions = []
    deletions = []
    modifications = []

    # Additions: exist in curr arch but not in prev arch
    for metric_name in curr_arch_keys - prev_arch_keys:
        logger.debug(f"{filename}: Table {table_id}: Metric '{metric_name}' added")
        additions.append({metric_name: metrics_curr_arch[metric_name]})

    # Deletions: exist in prev arch but not in curr arch
    for metric_name in prev_arch_keys - curr_arch_keys:
        logger.debug(f"{filename}: Table {table_id}: Metric '{metric_name}' deleted")
        deletions.append({metric_name: metrics_prev_arch[metric_name]})

    # Modifications: exist in both, prev archare fields
    for metric_name in prev_arch_keys & curr_arch_keys:
        metric_prev_arch = metrics_prev_arch[metric_name]
        metric_curr_arch = metrics_curr_arch[metric_name]

        if metric_prev_arch == metric_curr_arch:
            continue

        # Find modified fields
        modified_fields = {}
        all_fields = set(metric_prev_arch.keys()) | set(metric_curr_arch.keys())

        for field in all_fields:
            val_prev_arch = metric_prev_arch.get(field)
            val_curr_arch = metric_curr_arch.get(field)

            if val_prev_arch != val_curr_arch:
                modified_fields[field] = val_curr_arch

        if modified_fields:
            logger.debug(
                f"{filename}: Table {table_id}: Metric '{metric_name}' modified"
            )
            modifications.append({metric_name: modified_fields})

    return {
        "additions": additions,
        "deletions": deletions,
        "modifications": modifications,
    }


def compare_tables(
    tables_curr_arch: list[dict],
    tables_prev_arch: list[dict],
    panel_config_curr_arch: dict,
    panel_config_prev_arch: dict,
    filename: str,
) -> dict:
    """
    Compare metric tables between curr arch and prev arch.
    Returns table-level and metric-level differences.
    """
    # Create lookup by table id
    tables_curr_arch_dict = {t["id"]: t for t in tables_curr_arch}
    tables_prev_arch_dict = {t["id"]: t for t in tables_prev_arch}

    logger.debug(
        f"{filename}: curr arch table IDs: {list(tables_curr_arch_dict.keys())}"
    )
    logger.debug(
        f"{filename}: prev arch table IDs: {list(tables_prev_arch_dict.keys())}"
    )

    curr_arch_ids = set(tables_curr_arch_dict.keys())
    prev_arch_ids = set(tables_prev_arch_dict.keys())

    additions = []
    deletions = []
    modifications = []

    # Table-level additions (in curr arch but not in prev arch)
    for table_id in curr_arch_ids - prev_arch_ids:
        table = tables_curr_arch_dict[table_id]
        logger.debug(
            f"{filename}: Table-level Addition: Table {table_id} - {table.get('title')}"
        )
        additions.append(table)

    # Table-level deletions (in prev arch but not in curr arch)
    for table_id in prev_arch_ids - curr_arch_ids:
        table = tables_prev_arch_dict[table_id]
        logger.debug(
            f"{filename}: Table-level Deletion: Table {table_id} - {table.get('title')}"
        )
        deletions.append(table)

    # Compare tables that exist in both
    for table_id in prev_arch_ids & curr_arch_ids:
        table_prev_arch = tables_prev_arch_dict[table_id]
        table_curr_arch = tables_curr_arch_dict[table_id]

        metrics_prev_arch = table_prev_arch.get("metric", {})
        metrics_curr_arch = table_curr_arch.get("metric", {})

        metric_diff = compare_metrics(
            metrics_prev_arch, metrics_curr_arch, table_id, filename
        )

        # Store metric-level changes
        if metric_diff["additions"]:
            additions.append({
                "id": table_id,
                "title": table_curr_arch.get("title"),
                "metrics": metric_diff["additions"],
            })

        if metric_diff["deletions"]:
            deletions.append({
                "id": table_id,
                "title": table_prev_arch.get("title"),
                "metrics": metric_diff["deletions"],
            })

        if metric_diff["modifications"]:
            modifications.append({
                "id": table_id,
                "title": table_curr_arch.get("title"),
                "metrics": metric_diff["modifications"],
            })

    return {
        "additions": additions,
        "deletions": deletions,
        "modifications": modifications,
        "panel_config_curr_arch": panel_config_curr_arch,
        "panel_config_prev_arch": panel_config_prev_arch,
    }


def merge_file_diff(combined_diff: dict, file_diff: dict):
    """Merge file diff results into combined diff."""
    if file_diff["additions"]:
        combined_diff["Addition"].append({
            "panel_config": file_diff["panel_config_curr_arch"],
            "metric_tables": file_diff["additions"],
        })

    if file_diff["deletions"]:
        combined_diff["Deletion"].append({
            "panel_config": file_diff["panel_config_prev_arch"],
            "metric_tables": file_diff["deletions"],
        })

    if file_diff["modifications"]:
        combined_diff["Modification"].append({
            "panel_config": file_diff["panel_config_curr_arch"],
            "metric_tables": file_diff["modifications"],
        })


def format_metric_fields(lines: list[str], metric_data: dict):
    """Format individual metric fields."""
    for field_name, field_value in metric_data.items():
        if isinstance(field_value, str) and (
            "\n" in field_value or len(field_value) > 80
        ):
            lines.append(f"                {field_name}: |")
            for line in field_value.split("\n"):
                lines.append(f"                  {line}")
        else:
            lines.append(f"                {field_name}: {field_value}")


def format_output(combined_diff: dict) -> str:
    """Format the diff dictionary into YAML string with custom formatting."""
    lines = []

    for category in ["Addition", "Deletion", "Modification"]:
        lines.append(f"{category}:")

        if not combined_diff.get(category):
            lines.append("  []")
            lines.append("")
            continue

        for panel_item in combined_diff[category]:
            # Panel Config
            pc = panel_item["panel_config"]
            lines.append("  - Panel Config:")
            lines.append(f"      id: {pc['id']}")
            lines.append(f"      title: {pc['title']}")

            # Metric Tables
            lines.append("    metric_tables:")

            for mt in panel_item["metric_tables"]:
                lines.append("      - metric_table:")
                lines.append(f"          id: {mt['id']}")
                lines.append(f"          title: {mt['title']}")
                lines.append("          metrics:")

                # Check if this has metric-level changes or full table
                if "metrics" in mt:
                    # Metric-level changes (list of dicts)
                    for metric in mt["metrics"]:
                        for metric_name, metric_data in metric.items():
                            lines.append(f"            - {metric_name}:")
                            format_metric_fields(lines, metric_data)
                elif "metric" in mt:
                    # Full table (dict of metrics)
                    for metric_name, metric_data in mt["metric"].items():
                        lines.append(f"            - {metric_name}:")
                        format_metric_fields(lines, metric_data)

        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python metric_delta_generator.py"
            " <curr_arch_dir> <previous_arch_dir>"
        )
        sys.exit(1)

    curr_arch_dir = Path(sys.argv[1])
    prev_arch_dir = Path(sys.argv[2])

    # Validate directories
    if not curr_arch_dir.is_dir():
        logger.error(f"{curr_arch_dir} is not a directory")
        sys.exit(1)
    if not prev_arch_dir.is_dir():
        logger.error(f"{prev_arch_dir} is not a directory")
        sys.exit(1)

    logger.info(f"Current architecture: {curr_arch_dir}")
    logger.info(f"Previous architecture: {prev_arch_dir}")

    # Get YAML files from both directories
    curr_arch_files = set(get_yaml_files(curr_arch_dir))
    prev_arch_files = set(get_yaml_files(prev_arch_dir))

    logger.info(f"Found {len(curr_arch_files)} YAML files in curr arch directory")
    logger.info(f"Found {len(prev_arch_files)} YAML files in prev arch directory")

    # Find common files
    common_files = curr_arch_files & prev_arch_files
    logger.info(f"Found {len(common_files)} common files to prev archare")

    if not common_files:
        logger.error("No common YAML files found between directories")
        sys.exit(1)

    # Combined diff results
    combined_diff = {"Addition": [], "Deletion": [], "Modification": []}

    # Compare each common file
    for filename in sorted(common_files):
        logger.info(f"Comparing {filename}...")

        # Load files
        data_curr_arch = load_yaml(curr_arch_dir / filename)
        data_prev_arch = load_yaml(prev_arch_dir / filename)

        # Extract panel configs
        panel_config_curr_arch = extract_panel_config(data_curr_arch)
        panel_config_prev_arch = extract_panel_config(data_prev_arch)

        logger.debug(
            f"{filename}: Base panel - id: {panel_config_curr_arch['id']},"
            " title: {panel_config_curr_arch['title']}"
        )
        logger.debug(
            f"{filename}: Comp panel - id: {panel_config_prev_arch['id']},"
            " title: {panel_config_prev_arch['title']}"
        )

        # Extract metric tables
        tables_curr_arch = extract_metric_tables(data_curr_arch)
        tables_prev_arch = extract_metric_tables(data_prev_arch)

        logger.debug(
            f"{filename}: Found {len(tables_curr_arch)} tables in curr arch,"
            f" {len(tables_prev_arch)} in prev arch"
        )

        # Compare tables
        file_diff = compare_tables(
            tables_curr_arch,
            tables_prev_arch,
            panel_config_curr_arch,
            panel_config_prev_arch,
            filename,
        )

        # Merge into combined diff
        merge_file_diff(combined_diff, file_diff)

    # Summary
    logger.info(
        f"Overall: {len(combined_diff['Addition'])} panel configs with additions, "
        f"{len(combined_diff['Deletion'])} with deletions, "
        f"{len(combined_diff['Modification'])} with modifications"
    )

    # Format output
    output = format_output(combined_diff)

    print("\n" + "=" * 80)
    print("COMBINED DIFF OUTPUT:")
    print("=" * 80)
    print(output)

    # Write to file
    curr_arch_dir_name = curr_arch_dir.name
    output_dir = prev_arch_dir / "config_delta"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{curr_arch_dir_name}_diff.yaml"

    with open(output_file, "w") as f:
        f.write(output)

    logger.info(f"Combined diff file written to: {output_file}")


if __name__ == "__main__":
    main()
