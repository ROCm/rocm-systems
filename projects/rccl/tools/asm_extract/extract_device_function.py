#!/usr/bin/env python3
"""Extract ncclDevFunc_* device functions from specialized kernel assembly.

Given a .s file from compiling a specialized kernel, this script:
1. Keeps all code, debug sections, .rodata, CFI, etc. intact
2. Removes .globl for the kernel wrapper so it becomes a local symbol
3. Strips the .amdhsa_kernel block (no .kd generated for the wrapper)
4. Strips the .amdgpu_metadata block (runtime won't see this kernel)
5. Adds .globl for the ncclDevFunc_* so the linker can resolve it
6. Parses .amdgpu_metadata YAML for resource usage (before stripping)

Usage:
    python3 extract_device_function.py <input.s> <output.s> [output.json]
"""

import sys
import re
import json


def parse_metadata_resources(lines):
    """Parse .amdgpu_metadata YAML to extract resource usage."""
    metadata_start = None
    metadata_end = None
    for i, line in enumerate(lines):
        if line.strip() == '.amdgpu_metadata':
            metadata_start = i + 1
        elif line.strip() == '.end_amdgpu_metadata':
            metadata_end = i

    if metadata_start is None or metadata_end is None:
        return {}

    yaml_text = ''.join(lines[metadata_start:metadata_end])
    resources = {}
    for key, pattern in [
        ('vgpr_count', r'\.vgpr_count:\s+(\d+)'),
        ('agpr_count', r'\.agpr_count:\s+(\d+)'),
        ('sgpr_count', r'\.sgpr_count:\s+(\d+)'),
        ('group_segment_fixed_size', r'\.group_segment_fixed_size:\s+(\d+)'),
        ('private_segment_fixed_size', r'\.private_segment_fixed_size:\s+(\d+)'),
        ('sgpr_spill_count', r'\.sgpr_spill_count:\s+(\d+)'),
        ('vgpr_spill_count', r'\.vgpr_spill_count:\s+(\d+)'),
    ]:
        m = re.search(pattern, yaml_text)
        if m:
            resources[key] = int(m.group(1))

    return resources


def find_devfunc_symbol(lines):
    """Find the mangled ncclDevFunc_* symbol name."""
    for line in lines:
        m = re.match(r'\s+\.type\s+(.*ncclDevFunc[^,]+),\s*@function', line)
        if m:
            mangled = m.group(1)
            dm = re.search(r'ncclDevFunc_\w+', mangled)
            unmangled = dm.group(0) if dm else mangled
            return mangled, unmangled
    return None, None


def find_kernel_symbol(lines):
    """Find the mangled ncclDevKernel_* symbol name."""
    for line in lines:
        m = re.match(r'\s+\.globl\s+(.*ncclDevKernel\S+)', line)
        if m:
            return m.group(1).strip()
    return None


def find_amdhsa_kernel_range(lines):
    """Find the .section .rodata block containing .amdhsa_kernel through
    the .text directive that follows .end_amdhsa_kernel.
    Returns (start, end+1) line indices to skip, or None."""
    amdhsa_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('.amdhsa_kernel'):
            # Walk back to find the .section .rodata and .p2align before it
            start = i
            for j in range(i - 1, max(i - 5, -1), -1):
                s = lines[j].strip()
                if s.startswith('.section') and '.rodata' in s:
                    start = j
                    break
                if s.startswith('.p2align') or s == '':
                    start = j
                else:
                    break
            amdhsa_start = start
        if line.strip() == '.end_amdhsa_kernel':
            # Include the .text directive that typically follows
            end = i + 1
            if end < len(lines) and lines[end].strip() == '.text':
                end += 1
            return (amdhsa_start, end)
    return None


def find_metadata_range(lines):
    """Find .amdgpu_metadata ... .end_amdgpu_metadata block.
    Returns (start, end+1) line indices to skip, or None."""
    start = None
    for i, line in enumerate(lines):
        if line.strip() == '.amdgpu_metadata':
            start = i
        elif line.strip() == '.end_amdgpu_metadata':
            if start is not None:
                return (start, i + 1)
    return None


def extract(input_path, output_asm_path, output_json_path=None):
    with open(input_path) as f:
        lines = f.readlines()

    mangled, unmangled = find_devfunc_symbol(lines)
    if not mangled:
        print(f"ERROR: No ncclDevFunc found in {input_path}", file=sys.stderr)
        return 1

    kernel_sym = find_kernel_symbol(lines)

    # Parse resources BEFORE stripping metadata
    resources = parse_metadata_resources(lines)
    resources['function_name'] = unmangled
    resources['mangled_name'] = mangled

    # Identify ranges to skip
    skip_ranges = []
    amdhsa_range = find_amdhsa_kernel_range(lines)
    if amdhsa_range:
        skip_ranges.append(amdhsa_range)
    metadata_range = find_metadata_range(lines)
    if metadata_range:
        skip_ranges.append(metadata_range)

    def in_skip_range(idx):
        return any(s <= idx < e for s, e in skip_ranges)

    out_lines = []
    globl_inserted = False
    for i, line in enumerate(lines):
        if in_skip_range(i):
            continue

        stripped = line.strip()

        if '__hip_cuid' in stripped:
            continue

        # Remove .globl for the kernel wrapper
        if kernel_sym and re.match(r'\s*\.globl\s+' + re.escape(kernel_sym), stripped):
            continue

        # Add .globl before the .type directive for the device function
        if not globl_inserted and mangled in line and '.type' in line and '@function' in line:
            out_lines.append(f'\t.globl\t{mangled}\n')
            globl_inserted = True

        out_lines.append(line)

    with open(output_asm_path, 'w') as f:
        f.writelines(out_lines)

    if output_json_path:
        with open(output_json_path, 'w') as f:
            json.dump(resources, f, indent=2)

    print(f"Extracted: {unmangled} -> {output_asm_path}")
    print(f"  Resources: VGPR={resources.get('vgpr_count', '?')}, "
          f"SGPR={resources.get('sgpr_count', '?')}, "
          f"scratch={resources.get('private_segment_fixed_size', '?')}, "
          f"LDS={resources.get('group_segment_fixed_size', '?')}")

    return 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.s> <output.s> [output.json]")
        sys.exit(1)

    rc = extract(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    sys.exit(rc)
