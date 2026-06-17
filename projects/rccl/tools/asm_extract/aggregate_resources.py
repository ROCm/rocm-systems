#!/usr/bin/env python3
"""Aggregate resource usage from individual JSON files into a single max JSON.

Reads all .json files in a directory (produced by extract_device_function.py)
and writes a single JSON with the maximum of each resource field, plus
derived values needed for kernel descriptor patching.

Usage:
    python3 aggregate_resources.py <resources_dir> <output.json> <gpu_target>
"""

import sys
import os
import re
import json


RESOURCE_KEYS = [
    'vgpr_count', 'agpr_count', 'sgpr_count',
    'private_segment_fixed_size', 'group_segment_fixed_size',
]


def align_up(val, alignment):
    return ((val + alignment - 1) // alignment) * alignment


def _has_unified_vgpr_agpr(gpu_target):
    """gfx90a+ (CDNA2+) have a unified VGPR/AGPR register file where
    next_free_vgpr = accum_offset + agpr_count (up to 512).
    gfx908 (CDNA1) and RDNA (gfx10xx+) do not."""
    return gpu_target in ('gfx90a', 'gfx942', 'gfx950')


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <resources_dir> <output.json> <gpu_target>",
              file=sys.stderr)
        sys.exit(1)

    resources_dir = sys.argv[1]
    output_path = sys.argv[2]
    gpu_target = sys.argv[3]

    max_vals = {k: 0 for k in RESOURCE_KEYS}
    max_regular_vgpr = 0
    count = 0

    for fname in sorted(os.listdir(resources_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(resources_dir, fname)) as f:
            res = json.load(f)
        for key in RESOURCE_KEYS:
            max_vals[key] = max(max_vals[key], res.get(key, 0))
        regular = res.get('vgpr_count', 0) - res.get('agpr_count', 0)
        max_regular_vgpr = max(max_regular_vgpr, regular)
        count += 1

    if count == 0:
        print(f"ERROR: No .json files found in {resources_dir}", file=sys.stderr)
        sys.exit(1)

    accum_offset = align_up(max_regular_vgpr, 4)

    if _has_unified_vgpr_agpr(gpu_target):
        next_free_vgpr = max(max_vals['vgpr_count'],
                             accum_offset + max_vals['agpr_count'])
    else:
        next_free_vgpr = max_vals['vgpr_count']

    max_vals['accum_offset'] = accum_offset
    max_vals['next_free_vgpr'] = next_free_vgpr

    print(f"Aggregated {count} resource files ({gpu_target}): "
          f"VGPR={max_vals['vgpr_count']}, AGPR={max_vals['agpr_count']}, "
          f"SGPR={max_vals['sgpr_count']}, "
          f"scratch={max_vals['private_segment_fixed_size']}, "
          f"LDS={max_vals['group_segment_fixed_size']}, "
          f"accum_offset={accum_offset}, next_free_vgpr={next_free_vgpr}")

    with open(output_path, 'w') as f:
        json.dump(max_vals, f, indent=2)


if __name__ == '__main__':
    main()
