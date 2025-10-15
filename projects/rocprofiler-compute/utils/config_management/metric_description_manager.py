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
Metric description manager.
Syncs metric descriptions between config YAMLs and documentation files.

Usage:
    python metric_description_manager.py --sync-arch <arch_name> <configs_dir>
    python metric_description_manager.py --sync-all <configs_dir>
    python metric_description_manager.py --validate <arch_name> <configs_dir>
"""

import argparse
import sys
from pathlib import Path
from typing import Union

import yaml

# Section to panel ID mapping for organizing descriptions
SECTION_PANEL_MAP = {
    "Wavefront launch stats": 701,
    "Wavefront runtime stats": 702,
    "Overall instruction mix": 1001,
    "VALU arithmetic instruction mix": 1002,
    "MFMA instruction mix": 1004,
    "Compute Speed-of-Light": 1101,
    "Pipeline statistics": 1102,
    "Arithmetic operations": 1103,
    "LDS Speed-of-Light": 1201,
    "LDS Statistics": 1202,
    "vL1D Speed-of-Light": 1601,
    "Busy / stall metrics": 1501,
    "Instruction counts": 1502,
    "Spill / stack metrics": 1503,
    "L1 Unified Translation Cache (UTCL1)": 1605,
    "vL1D cache stall metrics": 1602,
    "vL1D cache access metrics": 1603,
    "Vector L1 data-return path or Texture Data (TD)": 1504,
    "L2 Speed-of-Light": 1701,
    "L2 cache accesses": 1703,
    "L2-Fabric interface metrics": 1702,
    "L2 - Fabric interface detailed metrics": 1706,
    "L2 - Fabric Interface stalls": 1705,
    "Scalar L1D Speed-of-Light": 1401,
    "Scalar L1D cache accesses": 1402,
    "Scalar L1D Cache - L2 Interface": 1403,
    "L1I Speed-of-Light": 1301,
    "L1I cache accesses": 1302,
    "L1I <-> L2 interface": 1303,
    "Workgroup manager utilizations": 601,
    "Workgroup Manager - Resource Allocation": 602,
    "Command processor fetcher (CPF)": 501,
    "Command processor packet processor (CPC)": 502,
    "System Speed-of-Light": 201,
}

# Reverse mapping for quick lookup
PANEL_ID_TO_SECTION = {v: k for k, v in SECTION_PANEL_MAP.items()}


def str_representer(dumper, data):
    """Custom YAML representer for multi-line strings."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, str_representer)


def validate_rst_syntax(text: str) -> tuple[bool, str]:
    """
    Basic RST syntax validation.
    Returns (is_valid, error_message)
    """
    if not text:
        return True, ""

    errors = []

    # Check for unmatched backticks
    single_backticks = text.count("`")
    if single_backticks % 2 != 0:
        errors.append("Unmatched single backticks")

    # Check for unmatched double backticks
    double_backticks = text.count("``")
    remaining_singles = single_backticks - (double_backticks * 2)
    if remaining_singles % 2 != 0:
        errors.append("Unmatched backticks after accounting for code literals")

    # Check for broken references
    if ":ref:`" in text:
        ref_count = text.count(":ref:`")
        closing_count = text[text.find(":ref:`") :].count("`")
        if ref_count > closing_count:
            errors.append("Unclosed :ref: directive")

    if ":doc:`" in text:
        doc_count = text.count(":doc:`")
        closing_count = text[text.find(":doc:`") :].count("`")
        if doc_count > closing_count:
            errors.append("Unclosed :doc: directive")

    if errors:
        return False, "; ".join(errors)

    return True, ""


def extract_descriptions_from_arch(
    arch_dir: Union[str, Path],
) -> dict[str, dict[str, dict]]:
    """
    Extract metric descriptions from all config YAMLs in an arch.
    Returns dict organized by section name:
    {
        "Wavefront launch stats": {
            "Grid Size": {"plain": "...", "rst": "...", "unit": "..."},
            ...
        }
    }
    """
    arch_path = Path(arch_dir)
    descriptions_by_section: dict[str, dict[str, dict]] = {}

    for yaml_file in sorted(arch_path.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        if not data or "Panel Config" not in data:
            continue

        panel_config = data["Panel Config"]
        panel_descriptions = panel_config.get("metrics_description", {})

        # Extract metrics and units from data sources
        metrics_with_units = {}
        for ds in panel_config.get("data source", []):
            for key, value in ds.items():
                if isinstance(value, dict) and "metric" in value:
                    table_id = value.get("id")
                    section_name = PANEL_ID_TO_SECTION.get(table_id)

                    if not section_name:
                        continue  # Skip tables not in mapping

                    for metric_name, metric_data in value["metric"].items():
                        unit = metric_data.get("unit")
                        if unit:
                            metrics_with_units[metric_name] = {
                                "section": section_name,
                                "unit": unit,
                            }

        # Organize descriptions by section
        for metric_name, description in panel_descriptions.items():
            if metric_name in metrics_with_units:
                section_name = metrics_with_units[metric_name]["section"]

                if section_name not in descriptions_by_section:
                    descriptions_by_section[section_name] = {}

                # Store description with unit
                desc_data = {
                    "plain": description
                    if isinstance(description, str)
                    else description.get("plain", ""),
                    "rst": description.get("rst", description)
                    if isinstance(description, dict)
                    else description,
                }

                # Add unit if available
                if metrics_with_units[metric_name].get("unit"):
                    desc_data["unit"] = metrics_with_units[metric_name]["unit"]

                descriptions_by_section[section_name][metric_name] = desc_data

    return descriptions_by_section


def update_per_arch_metrics_file(
    arch_name: str, descriptions: dict, output_dir: Union[str, Path]
) -> None:
    """
    Update per-arch metrics description file.
    Output: utils/per_arch_metric_definitions/gfx{arch}_metrics_description.yaml
    Format: RST only, organized by section
    """
    output_path = Path(output_dir) / f"{arch_name}_metrics_description.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to RST-only format
    rst_descriptions = {}
    for section, metrics in descriptions.items():
        rst_descriptions[section] = {}
        for metric_name, desc_data in metrics.items():
            metric_entry = {"rst": desc_data["rst"]}
            if "unit" in desc_data:
                metric_entry["unit"] = desc_data["unit"]
            rst_descriptions[section][metric_name] = metric_entry

    with open(output_path, "w") as f:
        yaml.dump(rst_descriptions, f, sort_keys=False, allow_unicode=True)

    print(f"Updated: {output_path}")


def update_docs_metrics_file(descriptions: dict, docs_file: Union[str, Path]) -> None:
    """
    Update docs metrics description file.
    Output: docs/data/metrics_description.yaml
    Merges with existing content (updates only changed metrics for latest arch)
    """
    docs_path = Path(docs_file)

    # Load existing if present
    existing_data = {}
    if docs_path.exists():
        with open(docs_path) as f:
            existing_data = yaml.safe_load(f) or {}

    # Merge new descriptions
    for section, metrics in descriptions.items():
        if section not in existing_data:
            existing_data[section] = {}

        for metric_name, desc_data in metrics.items():
            metric_entry = {"rst": desc_data["rst"]}
            if "unit" in desc_data:
                metric_entry["unit"] = desc_data["unit"]
            existing_data[section][metric_name] = metric_entry

    # Save
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_path, "w") as f:
        yaml.dump(existing_data, f, sort_keys=False, allow_unicode=True)

    print(f"Updated: {docs_path}")


def validate_descriptions(
    arch_dir: Union[str, Path],
) -> tuple[bool, list[str], list[str]]:
    """
    Validate metric descriptions in an architecture.
    Returns (is_valid, warnings, errors)
    """
    arch_path = Path(arch_dir)
    warnings: list[str] = []
    errors: list[str] = []

    for yaml_file in sorted(arch_path.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        if not data or "Panel Config" not in data:
            continue

        panel_config = data["Panel Config"]
        panel_descriptions = panel_config.get("metrics_description", {})

        # Get all metrics from data sources
        all_metrics = set()
        for ds in panel_config.get("data source", []):
            for key, value in ds.items():
                if isinstance(value, dict) and "metric" in value:
                    all_metrics.update(value["metric"].keys())

        # Check if all metrics have descriptions
        missing_descriptions = all_metrics - set(panel_descriptions.keys())
        if missing_descriptions:
            warnings.append(
                f"{yaml_file.name}: Missing descriptions for metrics: "
                f"{', '.join(sorted(missing_descriptions))}"
            )

        # Validate RST syntax
        for metric_name, description in panel_descriptions.items():
            if isinstance(description, dict):
                rst_text = description.get("rst", "")
            else:
                rst_text = description

            is_valid, error_msg = validate_rst_syntax(rst_text)
            if not is_valid:
                errors.append(
                    f"{yaml_file.name}: Metric '{metric_name}' has invalid RST: {error_msg}"
                )

    return len(errors) == 0, warnings, errors


def sync_arch(
    arch_name: str,
    configs_dir: Union[str, Path],
    per_arch_output_dir: Union[str, Path],
    docs_file: Union[str, Path],
    is_latest: bool,
) -> bool:
    """Sync descriptions for a single architecture."""
    arch_dir = Path(configs_dir) / arch_name

    if not arch_dir.is_dir():
        print(f"Error: {arch_dir} is not a directory")
        return False

    print(f"Syncing descriptions for {arch_name}...")

    # Validate first
    is_valid, warnings, errors = validate_descriptions(arch_dir)

    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"   {warning}")

    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"   {error}")
        return False

    # Extract descriptions
    descriptions = extract_descriptions_from_arch(arch_dir)

    if not descriptions:
        print(f"No descriptions found in {arch_name}")
        return True

    # Update per-arch file
    update_per_arch_metrics_file(arch_name, descriptions, per_arch_output_dir)

    # Update docs file if this is latest arch
    if is_latest:
        update_docs_metrics_file(descriptions, docs_file)

    print(f"✅ Successfully synced {arch_name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage metric descriptions")
    parser.add_argument(
        "--sync-arch",
        metavar="ARCH",
        help="Sync descriptions for specific architecture",
    )
    parser.add_argument(
        "--sync-all",
        action="store_true",
        help="Sync descriptions for all architectures",
    )
    parser.add_argument(
        "--validate",
        metavar="ARCH",
        help="Validate descriptions for specific architecture",
    )
    parser.add_argument(
        "--latest-arch", help="Specify which arch is latest (for docs update)"
    )
    parser.add_argument("configs_dir", help="Path to analysis_configs directory")
    parser.add_argument(
        "--per-arch-output",
        default="utils/per_arch_metric_definitions",
        help="Output directory for per-arch files",
    )
    parser.add_argument(
        "--docs-file",
        default="docs/data/metrics_description.yaml",
        help="Path to docs metrics description file",
    )

    args = parser.parse_args()

    if args.sync_arch:
        is_latest = (args.latest_arch == args.sync_arch) if args.latest_arch else False
        success = sync_arch(
            args.sync_arch,
            args.configs_dir,
            args.per_arch_output,
            args.docs_file,
            is_latest,
        )
        return 0 if success else 1

    elif args.sync_all:
        # Get all arch directories
        configs_path = Path(args.configs_dir)
        archs = sorted([
            d.name
            for d in configs_path.iterdir()
            if d.is_dir() and d.name.startswith("gfx")
        ])

        if not archs:
            print("No architecture directories found")
            return 1

        # Determine latest arch (highest gfx number or from --latest-arch)
        latest_arch = args.latest_arch if args.latest_arch else archs[-1]

        all_success = True
        for arch in archs:
            is_latest = arch == latest_arch
            success = sync_arch(
                arch, args.configs_dir, args.per_arch_output, args.docs_file, is_latest
            )
            if not success:
                all_success = False
                break

        return 0 if all_success else 1

    elif args.validate:
        arch_dir = Path(args.configs_dir) / args.validate
        if not arch_dir.is_dir():
            print(f"Error: {arch_dir} is not a directory")
            return 1

        is_valid, warnings, errors = validate_descriptions(arch_dir)

        print(f"Validation results for {args.validate}:")
        print("=" * 80)

        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"   {warning}")

        if errors:
            print("\nErrors:")
            for error in errors:
                print(f"   {error}")

        if is_valid and not warnings:
            print("\nAll validations passed")

        return 0 if is_valid else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
