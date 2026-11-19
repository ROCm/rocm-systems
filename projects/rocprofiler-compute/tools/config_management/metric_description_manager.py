#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Metric description manager.
Syncs the metric descriptions analyze shows (the analysis views and the metric library)
to one description file per arch, docs/data/metrics/<arch>_metrics.yaml. Each entry
has the library text (plain), the text the docs show (rst) and the unit; an rst that
differs from plain was edited by hand, and a sync keeps it. A view table's docs
section is its title, or the name DOCS_SECTIONS gives it.

Usage (paths relative to project root):
    SCRIPT=tools/config_management/metric_description_manager.py

    python $SCRIPT --sync-arch <arch_name>
    python $SCRIPT --sync-all
    python $SCRIPT --validate <arch_name>
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import yaml

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR.parent / "src"))

from config_management import utils_ruamel as cm_utils

from rocprof_compute_soc.metric_library.loader import (
    METRIC_LIBRARY_PARENT_ARCH,
    has_metric_library,
)
from utils.analysis_views import arch_panel_configs
from utils.utils_common import (
    MEMORY_CHART_PANEL_ID,
    PLACEHOLDER_RANGE_KEY,
    canonical_config_arch,
)

# The generated metric description files, one per arch.
DOCS_DIR = TOOLS_DIR.parent / "docs" / "data" / "metrics"

# The docs section of a view table is its title; in the memory chart it is
# "Memory chart - <table title>", or "Memory chart" for the table titled like the view.
# These maps name the other sections, by view title and table title; None leaves a
# table, or a whole view, out of the docs.
CDNA_DOCS_SECTIONS: Dict[str, Optional[Dict[str, Optional[str]]]] = {
    "Wavefront": {
        "Wavefront Launch Stats": "Wavefront launch stats",
        "Wavefront Runtime Stats": "Wavefront runtime stats",
    },
    "Compute Units - Instruction Mix": {
        "Overall Instruction Mix": "Overall instruction mix",
        "VALU Arithmetic Instruction Mix": "VALU arithmetic instruction mix",
        "VMEM Instruction Mix": None,
        "MFMA Arithmetic Instruction Mix": "Matrix instruction mix",
    },
    "Compute Units - Compute Pipeline": {
        "Pipeline Statistics": "Pipeline statistics",
        "Arithmetic Operations": "Arithmetic operations",
    },
    "Address Processing Unit and Data Return Path (TA/TD)": {
        "Busy and stall metrics": "Busy / stall metrics",
        "Spill and stack metrics": "Spill / stack metrics",
    },
    "Vector L1 Data Cache": {
        "L1D - L2 Transactions": None,
        "L1D Addr Translation Stalls": None,
    },
    "L2 Cache": {
        "L2 Cache Accesses": "L2 cache accesses",
        "L2 Cache Stalls": None,
    },
    "L2 Cache (per Channel)": None,
    "Memory Bandwidth Analysis": None,
}
WGP_DOCS_SECTIONS: Dict[str, Optional[str]] = {
    title: f"WGP {title}"
    for title in (
        "Utilization",
        "Wavefront Launch Stats",
        "Wavefront Dispatch",
        "Wavefront Life",
        "Wavefront Instruction Mix",
        "VMEM Instruction Mix",
        "LDS Instruction Mix",
        "VALU Instruction Mix",
        "VALU FLOPs and IOPs Counting",
        "WMMA Instruction FLOPs and IOPs Counting",
        "Instruction Issue Stalls",
        "Compute Pipeline Utilization",
        "Instruction Cache",
        "Scalar Data Cache",
        "SQC-GL1 Interface",
    )
}
# The sections of each config arch; an arch not listed has the CDNA sections.
DOCS_SECTIONS: Dict[str, Dict[str, Optional[Dict[str, Optional[str]]]]] = {
    "gfx115x": {
        "Memory Chart": {
            "Graphics Core Efficiency Arbiter (GCEA) to System Memory": (
                "Memory chart - GCEA to System Memory"
            ),
        },
        "Workgroup Processor (WGP)": {
            "Instruction Cache": "WGP Instruction Cache",
            "Scalar Data Cache": "WGP Scalar Data Cache",
        },
    },
    "gfx1201": {"Workgroup Processor (WGP)": WGP_DOCS_SECTIONS},
    "gfx1250": {
        "Workgroup Manager (SPI)": {
            title: f"SPI {title}"
            for title in (
                "Utilization",
                "Wavefront Dispatch",
                "Resource Allocation",
                "CPC-SPI Interface",
            )
        },
        "Workgroup Processor (WGP)": WGP_DOCS_SECTIONS,
        "GL0 Cache and LDS": {
            f"GL0-GL1 Interface - {topic}": f"GL0-GL1 Interface: {topic}"
            for topic in ("Requests and Bandwidth", "Latency", "Stalls")
        },
    },
}
# Docs-only names of sections and metrics, by config arch.
DOCS_SECTION_NAMES: Dict[str, Dict[str, str]] = {
    "gfx115x": {
        "Memory chart - TCP Cache (GL0 Vector Cache)": "Memory chart - TCP Cache",
        "Memory chart - GL1 Cache (L1)": "Memory chart - GL1 Cache",
        "Memory chart - GL2 Cache (L2)": "Memory chart - GL2 Cache",
    }
}
DOCS_METRIC_NAMES: Dict[str, Dict[str, str]] = {
    "gfx115x": {"GL0 Cache BW (TCP Cache)": "GL0 Cache BW"}
}


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
    """Return the name the docs give a section (DOCS_SECTION_NAMES)."""
    names = DOCS_SECTION_NAMES.get(canonical_config_arch(arch_name), {})
    return names.get(section_name, section_name)


def normalize_docs_metric_name(arch_name: str, metric_name: str) -> str:
    """Return the name the docs give a metric (DOCS_METRIC_NAMES)."""
    names = DOCS_METRIC_NAMES.get(canonical_config_arch(arch_name), {})
    return names.get(metric_name, metric_name)


def docs_section(
    arch_name: str, panel_id: int, view_title: str, table_title: str
) -> Optional[str]:
    """Return the docs section of a view table, or None if the docs leave it out."""
    view_sections = DOCS_SECTIONS.get(
        canonical_config_arch(arch_name), CDNA_DOCS_SECTIONS
    )
    if view_title in view_sections and view_sections[view_title] is None:
        return None
    table_sections = view_sections.get(view_title) or {}
    if table_title in table_sections:
        return table_sections[table_title]
    if panel_id != MEMORY_CHART_PANEL_ID:
        return table_title
    if table_title == view_title:
        return "Memory chart"
    return f"Memory chart - {table_title}"


def validate_rst_syntax(text: str) -> Tuple[bool, str]:
    """Basic RST syntax validation."""
    if not text:
        return True, ""

    errors: List[str] = []

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
    arch_name: str, panel_configs: Dict[int, Dict[str, Any]]
) -> Dict[str, Dict[str, dict]]:
    """Return the metric descriptions of an arch's panel configs (arch_panel_configs()),
    by docs section and row name, with the plain text, rst and unit of each."""
    descriptions_by_section: Dict[str, Dict[str, dict]] = {}
    for panel_id, panel_config in panel_configs.items():
        sections, units = _row_sections_and_units(arch_name, panel_id, panel_config)
        for metric_name, description in panel_config["metrics_description"].items():
            if metric_name not in sections:
                continue
            desc_data = {"plain": description, "rst": description}
            if metric_name in units:
                desc_data["unit"] = units[metric_name]
            descriptions_by_section.setdefault(sections[metric_name], {})
            descriptions_by_section[sections[metric_name]][metric_name] = desc_data
    return descriptions_by_section


def with_docs_names(arch_name: str, descriptions: dict) -> dict:
    """Return descriptions keyed by the section and metric names the docs use."""
    docs_descriptions: Dict[str, Dict[str, dict]] = {}
    for section, metrics in descriptions.items():
        docs_section_metrics = docs_descriptions.setdefault(
            normalize_docs_section_name(arch_name, section), {}
        )
        for metric_name, desc_data in metrics.items():
            docs_metric_name = normalize_docs_metric_name(arch_name, metric_name)
            if docs_metric_name in docs_section_metrics:
                raise ValueError(
                    f"{arch_name}: two metrics are {docs_metric_name!r} in the docs"
                )
            docs_section_metrics[docs_metric_name] = desc_data
    return docs_descriptions


def docs_file(arch_name: str, docs_dir: Union[str, Path]) -> Path:
    """Return the path of an arch's metric description file."""
    return Path(docs_dir) / f"{arch_name}_metrics.yaml"


def update_docs_file(
    arch_name: str, descriptions: dict, docs_dir: Union[str, Path]
) -> None:
    """Write an arch's description file with plain, rst, and unit fields."""
    output_path = docs_file(arch_name, docs_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    docs_descriptions: Dict[str, Dict[str, dict]] = {}
    for section, metrics in descriptions.items():
        docs_descriptions[section] = {}
        for metric_name, desc_data in metrics.items():
            entry = {
                "plain": format_yaml_scalar(desc_data.get("plain", "")),
                "rst": format_yaml_scalar(desc_data.get("rst", "")),
            }
            if "unit" in desc_data:
                entry["unit"] = desc_data["unit"]
            docs_descriptions[section][metric_name] = entry

    cm_utils.save_yaml(docs_descriptions, output_path)
    print(f"Updated: {output_path}")


def load_docs_file(arch_name: str, docs_dir: Union[str, Path]) -> dict:
    """Load an arch's description file if it exists."""
    path = docs_file(arch_name, docs_dir)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def preserve_manual_rst_edits(new_descriptions: dict, existing: dict) -> dict:
    """
    Preserve manually edited RST and existing metrics from the description file.

    Rules:
    1. If existing rst != plain, it was manually edited -> preserve it
    2. If a metric exists in the file but not extracted from panels -> preserve it
       (This handles metrics defined in panel tables but lacking descriptions)
    """
    if not existing:
        return new_descriptions  # No existing file, nothing to preserve

    # First pass: Preserve manual RST edits for metrics extracted from panels
    for section, metrics in new_descriptions.items():
        if section not in existing:
            continue

        for metric_name, new_data in metrics.items():
            if metric_name not in existing[section]:
                continue

            existing_data = existing[section][metric_name]
            existing_rst = existing_data.get("rst", "")
            existing_plain = existing_data.get("plain", "")

            # If RST differs from plain, it was manually edited -> preserve it
            if existing_rst and existing_rst != existing_plain:
                new_data["rst"] = existing_rst  # PRESERVE manual edit
                # Don't print - this is routine

            # Otherwise, use new auto-generated RST (plain->rst conversion)

    # Second pass: Preserve metrics that exist in the file but not extracted
    # from panels (e.g., metrics defined in panel tables but lacking
    # descriptions in metrics_description)
    preserved_count = 0
    for section, existing_metrics in existing.items():
        if section not in new_descriptions:
            # Entire section missing from panel extraction - preserve it
            new_descriptions[section] = existing_metrics
            preserved_count += len(existing_metrics)
        else:
            # Check for individual metrics missing from panel extraction
            for metric_name, existing_data in existing_metrics.items():
                if metric_name not in new_descriptions[section]:
                    # Metric exists in the file but not in panel extraction
                    # preserve it
                    new_descriptions[section][metric_name] = existing_data
                    preserved_count += 1

    if preserved_count > 0:
        print(
            f"  Preserved {preserved_count} metrics from the description file "
            "(not in panel descriptions)"
        )

    return new_descriptions


def validate_descriptions(
    panel_configs: Dict[int, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """Return the warnings (rows without a description) and the errors (descriptions
    with broken RST) of an arch's panel configs."""
    warnings: List[str] = []
    errors: List[str] = []
    for panel_id, panel_config in panel_configs.items():
        panel = f"Panel {panel_id} ({panel_config['title']})"
        panel_descriptions = panel_config["metrics_description"]
        rows = {
            metric_name
            for table in _metric_tables(panel_config)
            for metric_name in table["metric"]
            if metric_name != PLACEHOLDER_RANGE_KEY
        }
        missing = sorted(rows - set(panel_descriptions))
        if missing:
            warnings.append(
                f"{panel}: Missing descriptions for metrics: {', '.join(missing)}"
            )
        for metric_name, description in panel_descriptions.items():
            ok, err = validate_rst_syntax(description)
            if not ok:
                errors.append(f"{panel}: Metric '{metric_name}' has invalid RST: {err}")
    return warnings, errors


def sync_arch(arch_name: str, docs_dir: Union[str, Path]) -> None:
    """Sync an arch's description file, keeping its hand-edited RST."""
    print(f"Syncing descriptions for {arch_name}...")
    descriptions = with_docs_names(
        arch_name,
        extract_descriptions_from_arch(arch_name, arch_panel_configs(arch_name)),
    )
    existing = load_docs_file(arch_name, docs_dir)
    descriptions = preserve_manual_rst_edits(descriptions, existing)
    update_docs_file(arch_name, descriptions, docs_dir)


def print_validation(arch_name: str) -> int:
    """Print the warnings and errors of an arch's descriptions; return 1 on errors."""
    warnings, errors = validate_descriptions(arch_panel_configs(arch_name))
    print(f"Validation results for {arch_name}:\n{'=' * 80}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"   {warning}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"   {error}")
    if not errors and not warnings:
        print("\nAll validations passed")
    return 1 if errors else 0


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
        "--docs-output-dir",
        default=str(DOCS_DIR),
        help="Directory of the metric description YAMLs",
    )

    args = parser.parse_args()

    arch = args.sync_arch or args.validate
    if arch and not has_metric_library(arch):
        print(f"Error: no analysis views for {arch}")
        return 1

    if args.sync_arch:
        sync_arch(args.sync_arch, args.docs_output_dir)
        return 0

    if args.sync_all:
        for arch in sorted(METRIC_LIBRARY_PARENT_ARCH):
            sync_arch(arch, args.docs_output_dir)
        return 0

    if args.validate:
        return print_validation(args.validate)

    parser.print_help()
    return 1


def _metric_tables(panel_config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield each metric table of a panel config."""
    for data_source in panel_config["data source"]:
        for table in data_source.values():
            if "metric" in table:
                yield table


def _row_sections_and_units(
    arch_name: str, panel_id: int, panel_config: Dict[str, Any]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return the docs section of each row name of a panel config, and the docs unit
    of each row with a unit; a row in two tables takes the last table with a unit, or
    the last table."""
    sections: Dict[str, str] = {}
    unit_sections: Dict[str, str] = {}
    units: Dict[str, str] = {}
    for table in _metric_tables(panel_config):
        section = docs_section(
            arch_name, panel_id, panel_config["title"], table["title"]
        )
        if section is None:
            continue
        for metric_name, row in table["metric"].items():
            if metric_name == PLACEHOLDER_RANGE_KEY:
                continue
            sections[metric_name] = section
            if row.get("unit"):
                unit_sections[metric_name] = section
                units[metric_name] = normalize_unit_for_docs(row["unit"])
    sections.update(unit_sections)
    return sections, units


if __name__ == "__main__":
    sys.exit(main())
