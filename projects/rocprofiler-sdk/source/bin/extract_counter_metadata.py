#!/usr/bin/env python3
"""
Extract counter metadata from basic_counters.xml

Parses the XML file to create a mapping of counter name → hardware block
for each GPU architecture. This metadata is used by counter_optimizer.py
to group counters efficiently.

Usage:
    python3 extract_counter_metadata.py [--output counter_blocks.json]
"""

import xml.etree.ElementTree as ET
import json
import os
import sys
from pathlib import Path
from typing import Dict, Set


def extract_from_xml(xml_path: str) -> Dict[str, Dict[str, str]]:
    """
    Parse basic_counters.xml to extract counter → block mapping.

    The XML format is non-standard (missing root element, unquoted attributes),
    so we use a line-based parser.

    Args:
        xml_path: Path to basic_counters.xml

    Returns:
        Dict mapping arch → (counter_name → block_name)
        Example: {
            "gfx90a": {"SQ_WAVES": "SQ", "TCP_PENDING_STALL_CYCLES": "TCP", ...},
            "gfx1100": {...}
        }
    """
    import re

    result = {}
    current_arch = None
    current_base = None

    with open(xml_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Match architecture opening tag: <gfx900 base="gfx9">
            arch_match = re.match(r'<(gfx\w+)(?:\s+base[=\s]+"?(\w+)"?)?>', line)
            if arch_match:
                current_arch = arch_match.group(1)
                current_base = arch_match.group(2)

                # Initialize with base architecture if specified
                if current_base and current_base in result:
                    result[current_arch] = result[current_base].copy()
                else:
                    result[current_arch] = {}
                continue

            # Match metric with block: <metric name="SQ_WAVES" block=SQ event=4 ...>
            # Note: attributes may or may not be quoted
            metric_match = re.search(r'<metric\s+name[=\s]+"?(\w+)"?\s+.*?block[=\s]+"?(\w+)"?', line)
            if metric_match and current_arch:
                name = metric_match.group(1)
                block = metric_match.group(2)
                result[current_arch][name] = block
                continue

            # Match closing arch tag
            if re.match(r'</(gfx\w+)>', line):
                current_arch = None
                current_base = None

    return result


def normalize_arch_names(arch_map: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """
    Normalize architecture names to common forms used by rocminfo.

    Maps gfx8, gfx9, etc. to specific versions and adds common aliases.
    """
    normalized = {}

    # Direct mapping
    for arch, counters in arch_map.items():
        normalized[arch] = counters

        # Add common aliases
        # gfx9 covers multiple variants
        if arch in ['gfx900', 'gfx902', 'gfx904', 'gfx906', 'gfx908', 'gfx90a', 'gfx90c']:
            if 'gfx9' not in normalized:
                normalized['gfx9'] = counters.copy()
            else:
                # Merge counters from all gfx9 variants
                normalized['gfx9'].update(counters)

        # gfx10 variants
        if arch in ['gfx1010', 'gfx1011', 'gfx1012', 'gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035']:
            if 'gfx10' not in normalized:
                normalized['gfx10'] = counters.copy()
            else:
                normalized['gfx10'].update(counters)

        # gfx11 variants
        if arch in ['gfx1100', 'gfx1101', 'gfx1102', 'gfx1103']:
            if 'gfx11' not in normalized:
                normalized['gfx11'] = counters.copy()
            else:
                normalized['gfx11'].update(counters)

        # gfx12 variants
        if arch in ['gfx1200', 'gfx1201']:
            if 'gfx12' not in normalized:
                normalized['gfx12'] = counters.copy()
            else:
                normalized['gfx12'].update(counters)

    return normalized


def print_statistics(arch_map: Dict[str, Dict[str, str]]):
    """Print extraction statistics."""
    print("\n=== Counter Metadata Extraction Statistics ===\n")

    # Count unique blocks across all architectures
    all_blocks: Set[str] = set()
    for counters in arch_map.values():
        all_blocks.update(counters.values())

    print(f"Total architectures: {len(arch_map)}")
    print(f"Unique hardware blocks: {len(all_blocks)}")
    print(f"Blocks: {', '.join(sorted(all_blocks))}\n")

    # Per-architecture stats
    for arch in sorted(arch_map.keys()):
        counters = arch_map[arch]
        blocks = set(counters.values())
        print(f"{arch:12} {len(counters):4} counters, {len(blocks):2} blocks")


def main():
    # Default paths
    script_dir = Path(__file__).parent
    default_xml = script_dir.parent / "share" / "rocprofiler-sdk" / "basic_counters.xml"
    default_output = script_dir / "counter_blocks.json"

    # Parse arguments
    xml_path = default_xml
    output_path = default_output

    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        print(f"\nDefault XML: {default_xml}")
        print(f"Default output: {default_output}")
        sys.exit(0)

    if '--output' in sys.argv:
        idx = sys.argv.index('--output')
        if idx + 1 < len(sys.argv):
            output_path = Path(sys.argv[idx + 1])

    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        xml_path = Path(sys.argv[1])

    # Validate input
    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading XML from: {xml_path}")

    # Extract metadata
    arch_map = extract_from_xml(str(xml_path))

    # Normalize architecture names
    arch_map = normalize_arch_names(arch_map)

    # Print statistics
    print_statistics(arch_map)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(arch_map, f, indent=2, sort_keys=True)

    print(f"\nCounter metadata written to: {output_path}")
    print(f"Total size: {output_path.stat().st_size} bytes")


if __name__ == '__main__':
    main()
