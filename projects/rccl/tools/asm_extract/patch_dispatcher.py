#!/usr/bin/env python3
"""Patch common_device.s with aggregated resource usage from extracted device functions.

Patches all dispatcher kernel descriptors, ABS symbols, and metadata to reflect
the maximum resource requirements of the device functions they dispatch to.

Patched sections:
1. .set ABS symbols per kernel (num_vgpr, num_agpr, numbered_sgpr, etc.)
2. .amdhsa_kernel directives (accum_offset, next_free_vgpr, next_free_sgpr, etc.)
3. .AMDGPU.gpr_maximums (amdgpu.max_num_vgpr/agpr/sgpr)
4. .amdgpu_metadata YAML (vgpr_count, agpr_count, sgpr_count, etc.)

Usage:
    python3 patch_dispatcher.py <common_device.s> <output.s> <max_resources.json> <gpu_target>
"""

import sys
import re
import json


# Maximum addressable (numbered) SGPRs per architecture generation.
# GFX9 has 104 physical SGPRs with s[102:103] reserved for VCC -> 102 addressable.
# GFX10+ has 108 physical SGPRs with s[106:107] reserved for VCC -> 106 addressable.
_MAX_NUMBERED_SGPR = {
    9:  102,
    10: 106,
    11: 106,
    12: 106,
}


def _gfx_generation(gpu_target):
    """Extract the major generation number from a target like 'gfx942' -> 9, 'gfx1200' -> 12."""
    m = re.match(r'gfx(\d+)', gpu_target)
    if not m:
        sys.exit(f"ERROR: cannot parse GPU target '{gpu_target}'")
    gen = int(m.group(1))
    if gen >= 100:
        gen //= 100
    elif gen >= 10:
        gen //= 10
    return gen


def max_numbered_sgpr(gpu_target):
    gen = _gfx_generation(gpu_target)
    if gen not in _MAX_NUMBERED_SGPR:
        sys.exit(f"ERROR: unknown SGPR limit for generation gfx{gen}xx "
                 f"(target '{gpu_target}')")
    return _MAX_NUMBERED_SGPR[gen]


def _has_unified_vgpr_agpr(gpu_target):
    """gfx90a+ (CDNA2+) have a unified VGPR/AGPR register file."""
    return gpu_target in ('gfx90a', 'gfx942', 'gfx950')


def load_max_resources(json_path, gpu_target):
    with open(json_path) as f:
        res = json.load(f)
    # amdhsa_next_free_sgpr is numbered SGPRs only (excludes VCC, flat_scratch, xnack).
    # .sgpr_count in metadata includes system SGPRs. Cap for the .amdhsa_kernel directive.
    res['next_free_sgpr'] = min(res['sgpr_count'], max_numbered_sgpr(gpu_target))

    # On gfx908 (CDNA1) and RDNA, VGPRs and AGPRs are separate banks;
    # .amdhsa_next_free_vgpr counts only VGPRs (max 256 on gfx908).
    if not _has_unified_vgpr_agpr(gpu_target):
        res['next_free_vgpr'] = res['vgpr_count']

    return res


def patch_abs_symbols(lines, max_res):
    """Patch .set ABS symbols for each dispatcher kernel."""
    result = []
    for line in lines:
        m = re.match(r'(\s*\.set\s+\S+)\.(num_vgpr),\s*(.*)', line)
        if m:
            result.append(f'{m.group(1)}.num_vgpr, {max_res["next_free_vgpr"]}\n')
            continue

        m = re.match(r'(\s*\.set\s+\S+)\.(num_agpr),\s*(.*)', line)
        if m:
            result.append(f'{m.group(1)}.num_agpr, {max_res["agpr_count"]}\n')
            continue

        m = re.match(r'(\s*\.set\s+\S+)\.(numbered_sgpr),\s*(.*)', line)
        if m:
            result.append(f'{m.group(1)}.numbered_sgpr, {max_res["next_free_sgpr"]}\n')
            continue

        m = re.match(r'(\s*\.set\s+\S+)\.(num_named_barrier),\s*(.*)', line)
        if m:
            result.append(f'{m.group(1)}.num_named_barrier, 0\n')
            continue

        m = re.match(r'(\s*\.set\s+\S+)\.(private_seg_size),\s*(.*)', line)
        if m:
            result.append(f'{m.group(1)}.private_seg_size, {max_res["private_segment_fixed_size"]}\n')
            continue

        m = re.match(r'(\s*\.set\s+\S+)\.(uses_flat_scratch),\s*(.*)', line)
        if m:
            result.append(f'{m.group(1)}.uses_flat_scratch, 1\n')
            continue

        m = re.match(r'(\s*\.set\s+\S+)\.(has_dyn_sized_stack),\s*(.*)', line)
        if m:
            result.append(f'{m.group(1)}.has_dyn_sized_stack, 1\n')
            continue

        m = re.match(r'(\s*\.set\s+\S+)\.(has_recursion),\s*(.*)', line)
        if m:
            result.append(f'{m.group(1)}.has_recursion, 1\n')
            continue

        m = re.match(r'(\s*\.set\s+\S+)\.(has_indirect_call),\s*(.*)', line)
        if m:
            result.append(f'{m.group(1)}.has_indirect_call, 1\n')
            continue

        result.append(line)
    return result


def patch_amdhsa_kernel_directives(lines, max_res):
    """Patch .amdhsa_kernel block directives with literal values."""
    result = []
    in_kernel = False
    for line in lines:
        if re.match(r'\s*\.amdhsa_kernel\s+', line):
            in_kernel = True
            result.append(line)
            continue
        if in_kernel and re.match(r'\s*\.end_amdhsa_kernel', line):
            in_kernel = False
            result.append(line)
            continue

        if not in_kernel:
            result.append(line)
            continue

        m = re.match(r'(\s*)(\.amdhsa_accum_offset)\s+(.*)', line)
        if m:
            result.append(f'{m.group(1)}.amdhsa_accum_offset {max_res["accum_offset"]}\n')
            continue

        m = re.match(r'(\s*)(\.amdhsa_next_free_vgpr)\s+(.*)', line)
        if m:
            result.append(f'{m.group(1)}.amdhsa_next_free_vgpr {max_res["next_free_vgpr"]}\n')
            continue

        m = re.match(r'(\s*)(\.amdhsa_next_free_sgpr)\s+(.*)', line)
        if m:
            result.append(f'{m.group(1)}.amdhsa_next_free_sgpr {max_res["next_free_sgpr"]}\n')
            continue

        m = re.match(r'(\s*)(\.amdhsa_private_segment_fixed_size)\s+(.*)', line)
        if m:
            result.append(f'{m.group(1)}.amdhsa_private_segment_fixed_size {max_res["private_segment_fixed_size"]}\n')
            continue

        m = re.match(r'(\s*)(\.amdhsa_uses_dynamic_stack)\s+(.*)', line)
        if m:
            result.append(f'{m.group(1)}.amdhsa_uses_dynamic_stack 1\n')
            continue

        m = re.match(r'(\s*)(\.amdhsa_enable_private_segment)\s+(.*)', line)
        if m:
            result.append(f'{m.group(1)}.amdhsa_enable_private_segment 1\n')
            continue

        result.append(line)
    return result


def patch_gpr_maximums(lines, max_res):
    """Patch .set amdgpu.max_num_* in .AMDGPU.gpr_maximums section."""
    result = []
    for line in lines:
        m = re.match(r'(\s*\.set\s+amdgpu\.max_num_vgpr,\s*)\d+', line)
        if m:
            result.append(f'{m.group(1)}{max_res["next_free_vgpr"]}\n')
            continue

        m = re.match(r'(\s*\.set\s+amdgpu\.max_num_agpr,\s*)\d+', line)
        if m:
            result.append(f'{m.group(1)}{max_res["agpr_count"]}\n')
            continue

        m = re.match(r'(\s*\.set\s+amdgpu\.max_num_sgpr,\s*)\d+', line)
        if m:
            result.append(f'{m.group(1)}{max_res["next_free_sgpr"]}\n')
            continue

        result.append(line)
    return result


def patch_amdgpu_metadata(lines, max_res):
    """Patch .amdgpu_metadata YAML values for each kernel."""
    result = []
    in_metadata = False
    for line in lines:
        stripped = line.strip()
        if stripped == '.amdgpu_metadata':
            in_metadata = True
            result.append(line)
            continue
        if stripped == '.end_amdgpu_metadata':
            in_metadata = False
            result.append(line)
            continue

        if not in_metadata:
            result.append(line)
            continue

        m = re.match(r'(\s+\.vgpr_count:\s+)\d+', line)
        if m:
            result.append(f'{m.group(1)}{max_res["next_free_vgpr"]}\n')
            continue

        m = re.match(r'(\s+(?:-\s+)?\.agpr_count:\s+)\d+', line)
        if m:
            result.append(f'{m.group(1)}{max_res["agpr_count"]}\n')
            continue

        m = re.match(r'(\s+\.sgpr_count:\s+)\d+', line)
        if m:
            result.append(f'{m.group(1)}{max_res["sgpr_count"]}\n')
            continue

        m = re.match(r'(\s+\.private_segment_fixed_size:\s+)\d+', line)
        if m:
            result.append(f'{m.group(1)}{max_res["private_segment_fixed_size"]}\n')
            continue

        m = re.match(r'(\s+\.uses_dynamic_stack:\s+)(true|false)', line)
        if m:
            result.append(f'{m.group(1)}true\n')
            continue

        result.append(line)
    return result


def normalize_file_directives(lines):
    """Strip MD5 checksums from .file directives to avoid
    'inconsistent use of MD5 checksums' assembler warnings."""
    return [re.sub(r'(\s+\.file\s+\d+\s+"[^"]*"\s+"[^"]*")\s+md5\s+0x[0-9a-fA-F]+', r'\1', line)
            for line in lines]


def patch(asm_path, output_path, max_resources_path, gpu_target):
    max_res = load_max_resources(max_resources_path, gpu_target)

    print(f"Max resources: VGPR={max_res['vgpr_count']}, AGPR={max_res['agpr_count']}, "
          f"SGPR={max_res['sgpr_count']}, scratch={max_res['private_segment_fixed_size']}, "
          f"accum_offset={max_res['accum_offset']}, next_free_vgpr={max_res['next_free_vgpr']}")

    with open(asm_path) as f:
        lines = f.readlines()

    lines = patch_abs_symbols(lines, max_res)
    lines = patch_amdhsa_kernel_directives(lines, max_res)
    lines = patch_gpr_maximums(lines, max_res)
    lines = patch_amdgpu_metadata(lines, max_res)
    lines = normalize_file_directives(lines)

    with open(output_path, 'w') as f:
        f.writelines(lines)

    print(f"Patched: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <common_device.s> <output.s> <max_resources.json> <gpu_target>")
        sys.exit(1)

    patch(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
