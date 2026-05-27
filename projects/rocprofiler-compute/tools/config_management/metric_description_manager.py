#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Metric description manager.
Syncs metric descriptions between config YAMLs and documentation files.

Run from the project root so default options resolve:
  --per-arch-output (tools/per_arch_metric_definitions)
  --docs-output-dir (docs/data/metrics)

Usage (paths relative to project root):
    SCRIPT=tools/config_management/metric_description_manager.py
    CONFIGS=src/rocprof_compute_soc/analysis_configs

    python $SCRIPT --sync-arch <arch_name> $CONFIGS
    python $SCRIPT --sync-all $CONFIGS
    python $SCRIPT --validate <arch_name> $CONFIGS
    python $SCRIPT --generate-docs
    python $SCRIPT --generate-docs --docs-arch gfx115x
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_management import utils_ruamel as cm_utils  # noqa: E402


def is_rdna35_arch(arch_name: str) -> bool:
    """True for RDNA 3.5 IP names: gfx115x (gfx1150 to gfx115f)."""
    return bool(re.fullmatch(r"gfx115[0-9a-f]", arch_name, re.IGNORECASE))


def normalize_unit_for_docs(unit: str) -> str:
    """
    Convert template variable units to human-readable format for documentation.

    Patterns like (Requests + $normUnit) become "Requests per Normalization Unit"
    Patterns like (Cycles + $normUnit) become "Cycles per Normalization Unit"
    Other units are returned as-is.
    """
    if not unit or not isinstance(unit, str):
        return unit

    # Match patterns like (PREFIX + $normUnit) where PREFIX can contain
    # letters, hyphens, and spaces
    match = re.match(r"\(([A-Za-z\-\s]+?)\s+\+\s+\$normUnit\)", unit.strip())
    if match:
        prefix = match.group(1).strip()
        return f"{prefix} per Normalization Unit"

    return unit


def format_yaml_scalar(value: str):
    """Serialize descriptions like gfx942: inline scalars without block markers."""
    if isinstance(value, str) and "\n" in value:
        return re.sub(r"\s*\n\s*", " ", value).strip()
    return value


def normalize_docs_section_name(arch_name: str, section_name: str) -> str:
    """Apply docs-only section name cleanup for selected architectures."""
    if not is_rdna35_arch(arch_name):
        return section_name

    replacements = {
        "Memory chart - TCP Cache (GL0 Vector Cache)": "Memory chart - TCP Cache",
        "Memory chart - GL1 Cache (L1)": "Memory chart - GL1 Cache",
        "Memory chart - GL2 Cache (L2)": "Memory chart - GL2 Cache",
    }
    return replacements.get(section_name, section_name)


def normalize_docs_metric_name(arch_name: str, metric_name: str) -> str:
    """Apply docs-only metric name cleanup for selected architectures."""
    if not is_rdna35_arch(arch_name):
        return metric_name

    replacements = {
        "GL0 Cache BW (TCP Cache)": "GL0 Cache BW",
    }
    return replacements.get(metric_name, metric_name)


# All CDNA architectures share the same panel ID mapping.
CDNA_PANEL_ID_TO_SECTION: dict[int, str] = {
    201: "System Speed-of-Light",
    501: "Command processor fetcher (CPF)",
    502: "Command processor packet processor (CPC)",
    601: "Workgroup manager utilizations",
    602: "Workgroup Manager - Resource Allocation",
    701: "Wavefront launch stats",
    702: "Wavefront runtime stats",
    1001: "Overall instruction mix",
    1002: "VALU arithmetic instruction mix",
    1004: "Matrix instruction mix",
    1101: "Compute Speed-of-Light",
    1102: "Pipeline statistics",
    1103: "Arithmetic operations",
    1201: "LDS Speed-of-Light",
    1202: "LDS Statistics",
    1301: "L1I Speed-of-Light",
    1302: "L1I cache accesses",
    1303: "L1I <-> L2 interface",
    1401: "Scalar L1D Speed-of-Light",
    1402: "Scalar L1D cache accesses",
    1403: "Scalar L1D Cache - L2 Interface",
    1501: "Busy / stall metrics",
    1502: "Instruction counts",
    1503: "Spill / stack metrics",
    1504: "Vector L1 data-return path or Texture Data (TD)",
    1601: "vL1D Speed-of-Light",
    1602: "vL1D cache stall metrics",
    1603: "vL1D cache access metrics",
    1605: "L1 Unified Translation Cache (UTCL1)",
    1701: "L2 Speed-of-Light",
    1702: "L2-Fabric interface metrics",
    1703: "L2 cache accesses",
    1705: "L2 - Fabric Interface stalls",
    1706: "L2 - Fabric interface detailed metrics",
}

# RDNA 3.5 (gfx115x) metric_table ids -> section names
RDNA35_PANEL_ID_TO_SECTION_BY_ARCH: dict[int, str] = {
    1: "Top Kernels",
    2: "Dispatch List",
    101: "System Info",
    201: "System Speed-of-Light",
    301: "Memory chart - Instruction Cache",
    302: "Memory chart - Scalar Data Cache",
    303: "Memory chart - TCP Cache (GL0 Vector Cache)",
    304: "Memory chart - LDS (Local Data Share)",
    305: "Memory chart - TCP-GL1 Interface",
    306: "Memory chart - GL1 Cache (L1)",
    307: "Memory chart - GL1-GL2 Interface",
    308: "Memory chart - GL2 Cache (L2)",
    309: "Memory chart - GCEA to System Memory",
    401: "Roofline Performance Rates",
    402: "Roofline Plot Points",
    501: "CPC Utilization",
    502: "CPC Interface Utilization",
    503: "MEC Stall Cycles",
    504: "CPC Memory Requests",
    505: "MEC Instruction Cache",
    601: "SPI Utilization",
    602: "Wave Dispatch Statistics",
    701: "WGP Utilization",
    702: "Wavefront Launch Stats",
    703: "Wave Dispatch",
    704: "Wave Life",
    705: "Wave Instruction Mix",
    706: "VMEM Instruction Mix",
    707: "LDS Instruction Mix",
    709: "Wait State Analysis",
    710: "WGP Instruction Cache",
    711: "WGP Scalar Data Cache",
    901: "GL0 Utilization",
    902: "GL0 Request Statistics",
    903: "GL0 Cache Performance",
    904: "GL0-GL1 Interface",
    905: "GL0 Stalls",
    1101: "GL1 Cache Utilization",
    1102: "GL1 Cache Request Statistics",
    1103: "GL1 Cache Performance",
    1104: "GL1-GL2 Interface",
    1105: "GL1 Cache Stalls",
    1301: "GL2 Cache Performance",
    1302: "GL2 Cache Request Statistics",
    1303: "GL2 Cache Bandwidth",
    1401: "DRAM Read Interface",
    1402: "DRAM Write Interface",
    1404: "System Arbiter (SARB)",
    1405: "Return Interface",
    1701: "GPU Utilization",
    1702: "Shader Engine Utilization",
}


def panel_id_to_section(arch_name: str, table_id: int | None) -> str | None:
    """Resolve documentation section name for a metric_table id (arch-specific)."""
    if table_id is None:
        return None
    if is_rdna35_arch(arch_name):
        return RDNA35_PANEL_ID_TO_SECTION_BY_ARCH.get(table_id)
    return CDNA_PANEL_ID_TO_SECTION.get(table_id)


def validate_rst_syntax(text: str) -> tuple[bool, str]:
    """Basic RST syntax validation."""
    if not text:
        return True, ""

    errors: list[str] = []

    single_backticks = text.count("`")
    if single_backticks % 2 != 0:
        errors.append("Unmatched single backticks")

    double_backticks = text.count("``")
    remaining_singles = single_backticks - (double_backticks * 2)
    if remaining_singles % 2 != 0:
        errors.append("Unmatched backticks after accounting for code literals")

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
    Returns dict organized by section name.
    """
    arch_path = Path(arch_dir)
    arch_name = arch_path.name
    descriptions_by_section: dict[str, dict[str, dict]] = {}

    for yaml_file in sorted(arch_path.glob("*.yaml")):
        data = cm_utils.load_yaml(yaml_file)

        panel_config = data.get("Panel Config")
        if not isinstance(panel_config, dict):
            continue

        panel_descriptions: dict = panel_config.get("metrics_description", {})

        metrics_with_units: dict[str, dict[str, str]] = {}
        metrics_sections: dict[str, str] = {}  # Track ALL metrics and their sections
        for ds in panel_config.get("data source", []):
            for key, value in ds.items():
                if isinstance(value, dict) and "metric" in value:
                    table_id = value.get("id")
                    section_name = panel_id_to_section(arch_name, table_id)
                    if not section_name:
                        continue
                    for metric_name, metric_data in value["metric"].items():
                        # Track section for ALL metrics (even those without units)
                        metrics_sections[metric_name] = section_name
                        # Check both "unit" and "units" (different files use
                        # different keys)
                        unit = metric_data.get("unit") or metric_data.get("units")
                        if unit:
                            # Normalize units containing template variables for docs
                            normalized_unit = normalize_unit_for_docs(unit)
                            metrics_with_units[metric_name] = {
                                "section": section_name,
                                "unit": normalized_unit,
                            }

        for metric_name, description in panel_descriptions.items():
            # First try metrics_with_units (for unit extraction),
            # then metrics_sections, skip if no section found
            section_name = (
                metrics_with_units[metric_name]["section"]
                if metric_name in metrics_with_units
                else metrics_sections.get(metric_name)
            )

            # Skip metrics that don't belong to any known section
            if section_name is None:
                continue

            if isinstance(description, dict):
                plain = description.get("plain", "")
                rst = description.get("rst", "")
                unit = description.get("unit", None)
            else:
                plain = description
                rst = ""
                unit = None

            # If no RST provided, use plain text as RST
            if not rst and plain:
                rst = plain

            # If no unit in metrics_description, fall back to unit from metric_table
            if unit is None and metric_name in metrics_with_units:
                unit = metrics_with_units[metric_name]["unit"]

            desc_data = {"plain": plain, "rst": rst}
            if unit is not None:
                desc_data["unit"] = unit

            descriptions_by_section.setdefault(section_name, {})
            descriptions_by_section[section_name][metric_name] = desc_data

    return descriptions_by_section


def update_per_arch_metrics_file(
    arch_name: str, descriptions: dict, output_dir: Union[str, Path]
) -> None:
    """Write per-arch descriptions with plain, rst, and unit fields."""
    output_path = Path(output_dir) / f"{arch_name}_metrics_description.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_arch_descriptions: dict[str, dict[str, dict]] = {}
    for section, metrics in descriptions.items():
        per_arch_descriptions[section] = {}
        for metric_name, desc_data in metrics.items():
            entry = {
                "plain": format_yaml_scalar(desc_data.get("plain", "")),
                "rst": format_yaml_scalar(desc_data.get("rst", "")),
            }
            if "unit" in desc_data:
                entry["unit"] = desc_data["unit"]
            per_arch_descriptions[section][metric_name] = entry

    cm_utils.save_yaml(per_arch_descriptions, output_path)
    print(f"Updated: {output_path}")


def load_existing_per_arch(arch_name: str, per_arch_dir: Union[str, Path]) -> dict:
    """Load existing per-arch YAML if it exists."""
    per_arch_file = Path(per_arch_dir) / f"{arch_name}_metrics_description.yaml"
    if per_arch_file.exists():
        with open(per_arch_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def preserve_manual_rst_edits(new_descriptions: dict, existing_per_arch: dict) -> dict:
    """
    Preserve manually edited RST and existing metrics from per-arch YAMLs.

    Rules:
    1. If existing rst != plain, it was manually edited → preserve it
    2. If a metric exists in per-arch but not extracted from panels → preserve it
       (This handles metrics defined in panel tables but lacking descriptions)
    """
    if not existing_per_arch:
        return new_descriptions  # No existing file, nothing to preserve

    # First pass: Preserve manual RST edits for metrics extracted from panels
    for section, metrics in new_descriptions.items():
        if section not in existing_per_arch:
            continue

        for metric_name, new_data in metrics.items():
            if metric_name not in existing_per_arch[section]:
                continue

            existing_data = existing_per_arch[section][metric_name]
            existing_rst = existing_data.get("rst", "")
            existing_plain = existing_data.get("plain", "")

            # If RST differs from plain, it was manually edited → preserve it
            if existing_rst and existing_rst != existing_plain:
                new_data["rst"] = existing_rst  # PRESERVE manual edit
                # Don't print - this is routine

            # Otherwise, use new auto-generated RST (plain→rst conversion)

    # Second pass: Preserve metrics that exist in per-arch but not extracted
    # from panels (e.g., metrics defined in panel tables but lacking
    # descriptions in metrics_description)
    preserved_count = 0
    for section, existing_metrics in existing_per_arch.items():
        if section not in new_descriptions:
            # Entire section missing from panel extraction - preserve it
            new_descriptions[section] = existing_metrics
            preserved_count += len(existing_metrics)
        else:
            # Check for individual metrics missing from panel extraction
            for metric_name, existing_data in existing_metrics.items():
                if metric_name not in new_descriptions[section]:
                    # Metric exists in per-arch but not in panel extraction
                    # preserve it
                    new_descriptions[section][metric_name] = existing_data
                    preserved_count += 1

    if preserved_count > 0:
        print(
            f"  Preserved {preserved_count} metrics from per-arch YAML "
            "(not in panel descriptions)"
        )

    return new_descriptions


def resolved_docs_target_archs(per_arch_dir: Union[str, Path]) -> list[str]:
    """Doc emit list: every per-arch YAML on disk except known-excluded archs."""
    excluded = {"gfx940", "gfx941"}
    suffix = "_metrics_description.yaml"
    return sorted(
        arch
        for path in Path(per_arch_dir).glob(f"*{suffix}")
        if (arch := path.name.removesuffix(suffix)) not in excluded
    )


def generate_docs_from_per_arch(
    per_arch_dir: Union[str, Path],
    docs_output_dir: Union[str, Path],
    target_archs: list[str] | None = None,
) -> bool:
    """
    Generate per-arch documentation YAMLs from per-arch metric definitions.

    IMPORTANT: Generated docs YAMLs contain ONLY RST field (no plain text).
    These are derived artifacts for documentation - do not manually edit.

    Simple transformation:
    - Read per-arch YAML (has 'plain' and 'rst' fields)
    - Write docs YAML (only 'rst' and 'unit' fields)
    """
    if target_archs is None:
        target_archs = resolved_docs_target_archs(per_arch_dir)

    docs_output_dir = Path(docs_output_dir)
    docs_output_dir.mkdir(parents=True, exist_ok=True)

    for arch in target_archs:
        src_file = Path(per_arch_dir) / f"{arch}_metrics_description.yaml"
        dst_file = docs_output_dir / f"{arch}_metrics.yaml"

        if not src_file.exists():
            print(f"Warning: {src_file} not found, skipping")
            continue

        # Load per-arch YAML
        with open(src_file, encoding="utf-8") as f:
            per_arch_data = yaml.safe_load(f)

        # Transform: Keep only RST and unit fields for documentation
        docs_data = {}
        for section, metrics in per_arch_data.items():
            docs_section = normalize_docs_section_name(arch, section)
            docs_data[docs_section] = {}
            for metric_name, metric_info in metrics.items():
                docs_metric_name = normalize_docs_metric_name(arch, metric_name)
                # Extract only RST and unit (drop 'plain' text)
                entry = {}
                if "rst" in metric_info:
                    entry["rst"] = format_yaml_scalar(metric_info["rst"])
                if "unit" in metric_info:
                    entry["unit"] = metric_info["unit"]
                docs_data[docs_section][docs_metric_name] = entry

        cm_utils.save_yaml(docs_data, dst_file)
        print(f"Generated: {dst_file}")

    return True


def validate_descriptions(
    arch_dir: Union[str, Path],
) -> tuple[bool, list[str], list[str]]:
    """Validate: missing descriptions and basic RST syntax."""
    arch_path = Path(arch_dir)
    warnings: list[str] = []
    errors: list[str] = []

    for yaml_file in sorted(arch_path.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f) or {}

        panel_config = data.get("Panel Config")
        if not isinstance(panel_config, dict):
            continue

        panel_descriptions: dict = panel_config.get("metrics_description", {})
        all_metrics: set[str] = set()

        for ds in panel_config.get("data source", []):
            for _, value in ds.items():
                if isinstance(value, dict) and "metric" in value:
                    all_metrics.update(value["metric"].keys())

        missing = sorted(all_metrics - set(panel_descriptions.keys()))
        if missing:
            warnings.append(
                f"{yaml_file.name}: Missing descriptions "
                f"for metrics: {', '.join(missing)}"
            )

        for metric_name, description in panel_descriptions.items():
            rst_text = (
                description.get("rst", "")
                if isinstance(description, dict)
                else description
            )
            ok, err = validate_rst_syntax(rst_text)
            if not ok:
                errors.append(
                    f"{yaml_file.name}: Metric '{metric_name}' has invalid RST: {err}"
                )

    return len(errors) == 0, warnings, errors


def sync_arch(
    arch_name: str,
    configs_dir: str,
    per_arch_metrics_dir: str,
) -> bool:
    """Sync descriptions for a single architecture with RST preservation."""
    arch_dir = Path(configs_dir) / arch_name

    if not arch_dir.is_dir():
        print(f"Error: {arch_dir} is not a directory")
        return False

    print(f"Syncing descriptions for {arch_name}...")
    is_valid, warnings, errors = validate_descriptions(arch_dir)

    # 1) Extract descriptions from panel YAMLs (plain text, auto-convert to RST)
    descriptions = extract_descriptions_from_arch(arch_dir)
    if not descriptions:
        print(f"No descriptions found in {arch_name}")
        return True

    # 2) Load existing per-arch YAML (if exists) to preserve manual RST edits
    existing_per_arch = load_existing_per_arch(arch_name, per_arch_metrics_dir)

    # 3) Preserve manual RST edits (if rst != plain, keep existing RST)
    descriptions = preserve_manual_rst_edits(descriptions, existing_per_arch)

    # 4) Write per-arch file (with preserved manual RST edits)
    update_per_arch_metrics_file(arch_name, descriptions, per_arch_metrics_dir)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage metric descriptions")
    parser.add_argument(
        "--sync-arch",
        metavar="ARCH",
        help="Sync descriptions for a single architecture",
    )
    parser.add_argument(
        "--sync-all",
        action="store_true",
        help="Sync descriptions for all architectures",
    )
    parser.add_argument(
        "--validate",
        metavar="ARCH",
        help="Validate descriptions for a single architecture",
    )
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        help="Generate per-arch docs YAMLs from per-arch definitions",
    )
    parser.add_argument(
        "--docs-arch",
        action="append",
        metavar="ARCH",
        dest="docs_archs",
        help=(
            "Restrict --generate-docs to one or more architectures "
            "(repeatable). Defaults to every per-arch description file."
        ),
    )
    parser.add_argument(
        "configs_dir", nargs="?", help="Path to the analysis_configs directory"
    )
    parser.add_argument(
        "--per-arch-output",
        default="tools/per_arch_metric_definitions",
        help="Output directory for the per-arch description YAMLs",
    )
    parser.add_argument(
        "--docs-output-dir",
        default="docs/data/metrics",
        help="Output directory for the generated docs YAMLs",
    )

    args = parser.parse_args()

    if args.docs_archs and not args.generate_docs:
        print("Error: --docs-arch requires --generate-docs")
        return 1

    if args.generate_docs:
        available_archs = set(resolved_docs_target_archs(args.per_arch_output))
        if args.docs_archs:
            unknown = sorted(set(args.docs_archs) - available_archs)
            if unknown:
                print(
                    "Error: --docs-arch values are not in the docs target set "
                    f"(available: {', '.join(sorted(available_archs))}): "
                    f"{', '.join(unknown)}"
                )
                return 1
            target_archs = args.docs_archs
        else:
            target_archs = None
        ok = generate_docs_from_per_arch(
            args.per_arch_output, args.docs_output_dir, target_archs
        )
        return 0 if ok else 1

    if args.sync_arch:
        if not args.configs_dir:
            print("Error: configs_dir is required for --sync-arch")
            return 1
        ok = sync_arch(
            args.sync_arch,
            args.configs_dir,
            args.per_arch_output,
        )
        return 0 if ok else 1

    if args.sync_all:
        if not args.configs_dir:
            print("Error: configs_dir is required for --sync-all")
            return 1
        configs_path = Path(args.configs_dir)
        archs = sorted([
            d.name
            for d in configs_path.iterdir()
            if d.is_dir() and d.name.startswith("gfx")
        ])
        if not archs:
            print("No architecture directories found")
            return 1
        for arch in archs:
            ok = sync_arch(
                arch,
                args.configs_dir,
                args.per_arch_output,
            )
            if not ok:
                return 1
        return 0

    if args.validate:
        arch_dir = Path(args.configs_dir) / args.validate
        if not arch_dir.is_dir():
            print(f"Error: {arch_dir} is not a directory")
            return 1

        is_valid, warnings, errors = validate_descriptions(arch_dir)
        print(f"Validation results for {args.validate}:\n{'=' * 80}")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"   {w}")

        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"   {e}")

        if is_valid and not warnings:
            print("\nAll validations passed")

        return 0 if is_valid else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
