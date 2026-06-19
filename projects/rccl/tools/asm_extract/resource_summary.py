#!/usr/bin/env python3
"""Summarize resource usage across all extracted device functions.

Reads resource JSON files produced by extract_device_function.py and
(optionally) the dispatcher assembly to compare current vs required values.

Usage:
    python3 resource_summary.py <resources_dir> [dispatcher.s]
"""

import sys
import os
import re
import json
import collections


def load_resources(resources_dir):
    """Load all resource JSON files."""
    results = []
    for f in sorted(os.listdir(resources_dir)):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(resources_dir, f)) as fh:
            results.append(json.load(fh))
    return results


RESOURCE_KEYS = [
    'vgpr_count', 'agpr_count', 'sgpr_count',
    'private_segment_fixed_size', 'group_segment_fixed_size',
    'vgpr_spill_count', 'sgpr_spill_count',
]

SHORT_NAMES = {
    'vgpr_count': 'VGPR',
    'agpr_count': 'AGPR',
    'sgpr_count': 'SGPR',
    'private_segment_fixed_size': 'Scratch',
    'group_segment_fixed_size': 'LDS',
    'vgpr_spill_count': 'VSpill',
    'sgpr_spill_count': 'SSpill',
}


def compute_maximums(resources):
    maxes = {}
    for key in RESOURCE_KEYS:
        maxes[key] = max((r.get(key, 0) for r in resources), default=0)
    return maxes


def parse_dispatcher_kernels(asm_path):
    """Parse .kd and .amdgpu_metadata from the dispatcher assembly."""
    with open(asm_path) as f:
        lines = f.readlines()

    kernels_kd = {}
    kernels_meta = {}

    # Parse .amdhsa_kernel blocks
    cur_kernel = None
    cur_fields = {}
    for line in lines:
        m = re.match(r'\s*\.amdhsa_kernel\s+(\S+)', line)
        if m:
            cur_kernel = m.group(1)
            cur_fields = {}
            continue
        if cur_kernel and '.end_amdhsa_kernel' in line:
            kernels_kd[cur_kernel] = cur_fields
            cur_kernel = None
            continue
        if cur_kernel:
            for field in ['amdhsa_next_free_vgpr', 'amdhsa_next_free_sgpr',
                          'amdhsa_accum_offset',
                          'amdhsa_private_segment_fixed_size',
                          'amdhsa_group_segment_fixed_size']:
                m = re.match(rf'\s*\.{field}\s+(.*)', line)
                if m:
                    val = m.group(1).strip()
                    try:
                        cur_fields[field] = int(val)
                    except ValueError:
                        cur_fields[field] = val

    # Parse .amdgpu_metadata YAML
    in_metadata = False
    cur_name = None
    cur_meta = {}
    for line in lines:
        stripped = line.strip()
        if stripped == '.amdgpu_metadata':
            in_metadata = True
            continue
        if stripped == '.end_amdgpu_metadata':
            if cur_name:
                kernels_meta[cur_name] = cur_meta
            in_metadata = False
            continue
        if not in_metadata:
            continue

        m = re.match(r'\s+\.name:\s+(\S+)', line)
        if m:
            if cur_name:
                kernels_meta[cur_name] = cur_meta
            cur_name = m.group(1)
            cur_meta = {}
            continue

        for field, key in [('.vgpr_count', 'vgpr_count'),
                           ('.agpr_count', 'agpr_count'),
                           ('.sgpr_count', 'sgpr_count'),
                           ('.private_segment_fixed_size', 'private_segment_fixed_size'),
                           ('.group_segment_fixed_size', 'group_segment_fixed_size')]:
            m = re.match(rf'\s+{re.escape(field)}:\s+(\d+)', line)
            if m:
                cur_meta[key] = int(m.group(1))

    if cur_name:
        kernels_meta[cur_name] = cur_meta

    return kernels_kd, kernels_meta


def parse_gpr_maximums(asm_path):
    """Parse .set amdgpu.max_num_* from the assembly."""
    maxes = {}
    with open(asm_path) as f:
        for line in f:
            m = re.match(r'\s*\.set\s+amdgpu\.max_num_vgpr,\s*(\d+)', line)
            if m:
                maxes['max_num_vgpr'] = int(m.group(1))
            m = re.match(r'\s*\.set\s+amdgpu\.max_num_agpr,\s*(\d+)', line)
            if m:
                maxes['max_num_agpr'] = int(m.group(1))
            m = re.match(r'\s*\.set\s+amdgpu\.max_num_sgpr,\s*(\d+)', line)
            if m:
                maxes['max_num_sgpr'] = int(m.group(1))
    return maxes


def demangle_short(name):
    """Extract a short name from the mangled function name."""
    m = re.search(r'ncclDevFunc_(\w+)', name)
    return m.group(0) if m else name


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <resources_dir> [dispatcher.s]")
        sys.exit(1)

    resources_dir = sys.argv[1]
    dispatcher_path = sys.argv[2] if len(sys.argv) > 2 else None

    resources = load_resources(resources_dir)
    if not resources:
        print("No resource files found.")
        sys.exit(1)

    maxes = compute_maximums(resources)

    # --- Per-function detail, grouped by unique profile ---
    profiles = collections.defaultdict(list)
    for r in resources:
        key = tuple(r.get(k, 0) for k in RESOURCE_KEYS)
        profiles[key].append(demangle_short(r.get('function_name', '?')))

    print(f"{'='*78}")
    print(f"  Resource Summary: {len(resources)} device functions, "
          f"{len(profiles)} unique profiles")
    print(f"{'='*78}")
    print()

    hdr = (f"  {'VGPR':>5} {'AGPR':>5} {'SGPR':>5} "
           f"{'Scratch':>8} {'LDS':>8} {'VSpill':>6} {'SSpill':>6}  "
           f"{'#':>4}  Functions")
    print(hdr)
    print(f"  {'-'*74}")

    for key in sorted(profiles.keys(), key=lambda k: (-k[0], -k[3])):
        funcs = profiles[key]
        v, a, s, scratch, lds, vs, ss = key
        if len(funcs) <= 3:
            func_str = ', '.join(funcs)
        else:
            func_str = f"{funcs[0]}, {funcs[1]}, ... +{len(funcs)-2} more"
        print(f"  {v:5d} {a:5d} {s:5d} {scratch:8d} {lds:8d} {vs:6d} {ss:6d}  "
              f"{len(funcs):4d}  {func_str}")

    # --- Maximums ---
    print()
    print(f"  {'='*74}")
    print(f"  Required maximums (across all {len(resources)} functions):")
    print(f"  {'='*74}")
    for key in RESOURCE_KEYS:
        print(f"    {SHORT_NAMES[key]:>8}: {maxes[key]}")

    # --- Dispatcher comparison ---
    if dispatcher_path and os.path.exists(dispatcher_path):
        print()
        print(f"  {'='*74}")
        print(f"  Dispatcher kernel descriptors ({os.path.basename(dispatcher_path)}):")
        print(f"  {'='*74}")

        kd, meta = parse_dispatcher_kernels(dispatcher_path)
        gpr_max = parse_gpr_maximums(dispatcher_path)

        if gpr_max:
            print()
            print("  .AMDGPU.gpr_maximums:")
            for k, v in sorted(gpr_max.items()):
                required = {'max_num_vgpr': maxes['vgpr_count'],
                            'max_num_agpr': maxes['agpr_count'],
                            'max_num_sgpr': maxes['sgpr_count']}.get(k, '?')
                status = "OK" if v >= required else f"NEEDS UPDATE (have {v}, need {required})"
                print(f"    {k}: {v}  [{status}]")

        for name in sorted(kd.keys()):
            fields = kd[name]
            m = meta.get(name, {})
            short = name.split('ncclDevKernel')[-1].split('24nccl')[0] if 'ncclDevKernel' in name else name

            print()
            print(f"  Kernel: ncclDevKernel{short}")
            print(f"    .kd fields:")
            for field, val in sorted(fields.items()):
                print(f"      {field}: {val}")

            if m:
                print(f"    .amdgpu_metadata:")
                for field in ['vgpr_count', 'agpr_count', 'sgpr_count',
                              'private_segment_fixed_size', 'group_segment_fixed_size']:
                    if field in m:
                        required = maxes.get(field, 0)
                        current = m[field]
                        if field in ('private_segment_fixed_size',):
                            status = "OK" if current >= required else f"NEEDS UPDATE ({current} -> {required})"
                        else:
                            status = ""
                        print(f"        {field}: {current}  {status}")

    # --- Warnings ---
    print()
    print(f"  {'='*74}")
    print("  Consistency checks:")
    print(f"  {'='*74}")

    spill_funcs = [(r['function_name'], r.get('vgpr_spill_count', 0), r.get('sgpr_spill_count', 0))
                   for r in resources
                   if r.get('vgpr_spill_count', 0) > 0 or r.get('sgpr_spill_count', 0) > 0]
    if spill_funcs:
        print(f"  WARNING: {len(spill_funcs)} functions have register spills:")
        for name, vs, ss in spill_funcs[:10]:
            print(f"    {demangle_short(name)}: VGPR_spill={vs}, SGPR_spill={ss}")
        if len(spill_funcs) > 10:
            print(f"    ... and {len(spill_funcs) - 10} more")
    else:
        print("  OK: No register spills detected")

    lds_values = set(r.get('group_segment_fixed_size', 0) for r in resources)
    if dispatcher_path and os.path.exists(dispatcher_path):
        _, meta = parse_dispatcher_kernels(dispatcher_path)
        for name, m in meta.items():
            kd_lds = m.get('group_segment_fixed_size', 0)
            if kd_lds < maxes['group_segment_fixed_size']:
                print(f"  WARNING: Kernel {name} LDS={kd_lds} < max={maxes['group_segment_fixed_size']}")
    if len(lds_values) > 1:
        print(f"  NOTE: Multiple LDS sizes in use: {sorted(lds_values)}")
    else:
        print(f"  OK: All functions use LDS={lds_values.pop()}")

    print()


if __name__ == '__main__':
    main()
