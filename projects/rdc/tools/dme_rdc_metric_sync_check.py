#!/usr/bin/env python3
"""
DME-RDC Metric Sync Check

Compares metrics exposed by AMD Device Metrics Exporter (DME) against RDC
field definitions to detect gaps. Exits non-zero if unmapped DME metrics
are found.

Usage:
    python3 dme_rdc_metric_sync_check.py \
        --rdc-header projects/rdc/include/rdc/rdc.h \
        --mapping projects/rdc/tools/dme_rdc_metric_mapping.json \
        [--dme-path path/to/device-metrics-exporter] \
        [--output report.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_rdc_fields(header_path: Path) -> set[str]:
    """Extract all RDC_FI_*, RDC_EVNT_*, and RDC_HEALTH_* enum names from rdc.h."""
    fields = set()
    pattern = re.compile(r"\b(RDC_FI_\w+|RDC_EVNT_\w+|RDC_HEALTH_\w+)\b")
    text = header_path.read_text()
    for match in pattern.finditer(text):
        name = match.group(1)
        # Skip sentinel/marker values
        if name in (
            "RDC_FI_INVALID",
            "RDC_FI_ECC_FIRST",
            "RDC_FI_ECC_LAST",
            "RDC_EVNT_NOTIF_FIRST",
            "RDC_EVNT_NOTIF_LAST",
            "RDC_FI_CPU_FIRST",
            "RDC_FI_CPU_LAST",
        ):
            continue
        fields.add(name)
    return fields


def parse_dme_metrics_from_go(dme_path: Path) -> set[str]:
    """Parse DME metric names from protobuf-generated Go source.

    DME defines its actual metrics as GPUMetricField_* protobuf enum values
    in exporterconfig.pb.go. This is the authoritative list — not arbitrary
    GPU_*/PCIE_* strings which include enum types, constants, etc.
    """
    metrics = set()
    # Primary: parse GPUMetricField protobuf enum definitions
    pb_pattern = re.compile(r"GPUMetricField_(GPU_\w+|PCIE_\w+)\s+GPUMetricField\s*=\s*\d+")
    for go_file in dme_path.rglob("*.pb.go"):
        text = go_file.read_text(errors="replace")
        for match in pb_pattern.finditer(text):
            metrics.add(match.group(1))

    # Fallback: if no pb.go found, parse string map entries
    if not metrics:
        map_pattern = re.compile(r':\s*"(GPU_\w+|PCIE_\w+)"')
        for go_file in dme_path.rglob("*.go"):
            text = go_file.read_text(errors="replace")
            for match in map_pattern.finditer(text):
                metrics.add(match.group(1))

    return metrics


def load_mapping(mapping_path: Path) -> list[dict]:
    """Load the curated DME-to-RDC mapping file."""
    data = json.loads(mapping_path.read_text())
    return data.get("mappings", [])


def run_check(
    rdc_header: Path, mapping_path: Path, dme_path: Path | None, output_path: Path | None
) -> int:
    """Run the sync check and return exit code (0=ok, 1=gaps found)."""
    rdc_fields = parse_rdc_fields(rdc_header)
    mappings = load_mapping(mapping_path)

    # Build sets from mapping
    mapped_dme = {m["dme"] for m in mappings}

    # Check for stale mappings (mapped to non-existent RDC field)
    stale = []
    for m in mappings:
        if m["status"] == "mapped" and m.get("rdc"):
            if m["rdc"] not in rdc_fields:
                stale.append(
                    {"dme": m["dme"], "rdc": m["rdc"], "issue": "RDC field not found in header"}
                )

    # Check for known missing
    known_missing = [m for m in mappings if m["status"] == "missing"]

    # If DME source is available, check for new unmapped metrics
    unmapped = []
    if dme_path and dme_path.exists():
        dme_metrics = parse_dme_metrics_from_go(dme_path)
        for metric in sorted(dme_metrics):
            if metric not in mapped_dme:
                unmapped.append(metric)

    # Summary
    report = {
        "rdc_field_count": len(rdc_fields),
        "mapping_count": len(mappings),
        "mapped": len([m for m in mappings if m["status"] == "mapped"]),
        "skipped": len([m for m in mappings if m["status"] == "skipped"]),
        "known_missing": [m["dme"] for m in known_missing],
        "stale_mappings": stale,
        "unmapped_dme_metrics": unmapped,
    }

    # Print summary
    print(f"RDC fields in header: {report['rdc_field_count']}")
    print(f"Mapping entries: {report['mapping_count']}")
    print(f"  Mapped: {report['mapped']}")
    print(f"  Skipped: {report['skipped']}")
    print(f"  Known missing: {len(report['known_missing'])}")

    if stale:
        print(f"\nSTALE MAPPINGS ({len(stale)}):")
        for s in stale:
            print(f"  {s['dme']} -> {s['rdc']}: {s['issue']}")

    if known_missing:
        print(f"\nKNOWN MISSING ({len(known_missing)}):")
        for m in known_missing:
            reason = m.get("reason", "")
            print(f"  {m['dme']}: {reason}")

    if unmapped:
        print(f"\nNEW UNMAPPED DME METRICS ({len(unmapped)}):")
        for m in unmapped:
            print(f"  {m}")

    if output_path:
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {output_path}")

    if unmapped:
        print("\nFAILED: New DME metrics found without mapping entries.")
        return 1
    if stale:
        print("\nWARNING: Stale mappings found.")
        return 1

    print("\nPASSED: All DME metrics are tracked.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="DME-RDC Metric Sync Check")
    parser.add_argument(
        "--rdc-header", required=True, type=Path, help="Path to projects/rdc/include/rdc/rdc.h"
    )
    parser.add_argument(
        "--mapping", required=True, type=Path, help="Path to dme_rdc_metric_mapping.json"
    )
    parser.add_argument(
        "--dme-path", type=Path, default=None, help="Path to cloned device-metrics-exporter repo"
    )
    parser.add_argument("--output", type=Path, default=None, help="Path to write JSON report")
    args = parser.parse_args()

    sys.exit(run_check(args.rdc_header, args.mapping, args.dme_path, args.output))


if __name__ == "__main__":
    main()
