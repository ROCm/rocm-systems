#!/usr/bin/env python3
"""Strip kernel entries from AMDGPU assembly while preserving device functions.

Each specialized kernel .s file contains:
  - ncclDevFunc_*  (device function, callable via function pointer)  -- KEEP
  - ncclDevKernel_* (kernel, wrapping the device function)           -- STRIP

This script removes the kernel and its metadata so the .o contains only
the device function, linkable by lld.

Usage:
    strip_kernel.py <input.s> <output.s> [--meta <output.meta>] [--check-lds <expected_lds>]
"""

import argparse
import re
import sys


def fatal(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def find_kernel_name(lines):
    """Find the mangled kernel name from .amdhsa_kernel directive."""
    names = []
    for line in lines:
        m = re.match(r'\s*\.amdhsa_kernel\s+(\S+)', line)
        if m:
            names.append(m.group(1))
    if len(names) == 0:
        fatal("No .amdhsa_kernel directive found in assembly")
    if len(names) > 1:
        fatal(f"Multiple kernels found ({len(names)}): {names}")
    return names[0]


def find_devfunc_name(lines):
    """Find the ncclDevFunc mangled name from .globl directives."""
    names = []
    for line in lines:
        m = re.match(r'\s*\.globl\s+(_Z\S*ncclDevFunc\S+)', line)
        if m:
            names.append(m.group(1))
    return names


def find_function_block(lines, mangled_name):
    """Find start/end line indices of a function block.

    Returns (start, end) where:
      - start is the first line of the function's preamble
        (the line with '; -- Begin function')
      - end is the line with '; -- End function'
    """
    begin_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if '; -- Begin function' in line and mangled_name in line:
            begin_idx = i
        elif begin_idx is not None and end_idx is None:
            if '; -- End function' in line:
                end_idx = i
                break

    if begin_idx is None:
        fatal(f"Could not find '; -- Begin function' for {mangled_name[:60]}...")
    if end_idx is None:
        fatal(f"Could not find '; -- End function' after line {begin_idx}")

    # Walk backward from begin_idx to include preceding context lines
    # (comments like "; TotalNumVgprs:" and the ".text" directive)
    start = begin_idx
    while start > 0:
        prev = lines[start - 1].strip()
        if prev.startswith(';') or prev == '.text' or prev == '':
            start -= 1
        else:
            break

    return (start, end_idx)


def find_csdata_block(lines, start_search, is_kernel):
    """Find the .AMDGPU.csdata comment block for a function/kernel.

    Returns (start, end) of the block, or None if not found.
    The block starts with '.section .AMDGPU.csdata' and the next line
    contains either '; Kernel info:' or '; Function info:'.
    """
    marker = '; Kernel info:' if is_kernel else '; Function info:'
    for i in range(start_search, min(start_search + 50, len(lines))):
        if '.AMDGPU.csdata' in lines[i]:
            if i + 1 < len(lines) and marker in lines[i + 1]:
                end = i + 2
                while end < len(lines) and lines[end].startswith(';'):
                    end += 1
                return (i, end - 1)
    return None


def strip_kernel_yaml(lines):
    """Remove the entire .amdgpu_metadata section from stripped assemblies.

    After kernel stripping, the metadata is empty (no kernels remain).
    Removing it entirely prevents lld from concatenating hundreds of empty
    .note entries that confuse the HSA runtime code object loader.
    """
    result = []
    in_metadata = False

    for line in lines:
        if '.amdgpu_metadata' in line and '.end_amdgpu_metadata' not in line:
            in_metadata = True
            continue
        if '.end_amdgpu_metadata' in line:
            in_metadata = False
            continue
        if in_metadata:
            continue
        result.append(line)

    return result


def strip_debug_sections(lines):
    """Remove compiler-emitted .debug_* sections from the assembly.

    After kernel stripping, these sections contain dangling label references
    (address ranges, frame info) to the removed kernel code.  The assembler
    will regenerate correct .debug_line from the inline .loc directives that
    remain in the kept device function code.
    """
    result = []
    in_debug = False

    for line in lines:
        if re.match(r'\s*\.section\s+\.debug_', line):
            in_debug = True
            continue
        if in_debug:
            if re.match(r'\s*\.section\s+', line) and '.debug_' not in line:
                in_debug = False
                result.append(line)
            continue
        result.append(line)

    return result


def extract_meta(lines):
    """Extract resource requirements from the .amdhsa_kernel KD block.

    Reads the actual hardware-facing values (next_free_vgpr, accum_offset,
    next_free_sgpr, private_segment_fixed_size) rather than the .set
    amdgpu.max_num_* hints, which undercount for kernels with AGPRs or
    complex register allocation.
    """
    kd = {}
    in_kd = False
    for line in lines:
        if re.match(r'\s*\.amdhsa_kernel\s+', line):
            in_kd = True
            continue
        if re.match(r'\s*\.end_amdhsa_kernel', line):
            in_kd = False
            continue
        if not in_kd:
            continue
        for field in ('next_free_vgpr', 'next_free_sgpr', 'accum_offset',
                      'private_segment_fixed_size'):
            m = re.match(rf'\s*\.amdhsa_{field}\s+(\d+)', line)
            if m:
                kd[field] = int(m.group(1))

    meta = []
    if 'next_free_vgpr' in kd:
        meta.append(f'\t.set kd.next_free_vgpr, {kd["next_free_vgpr"]}\n')
    if 'accum_offset' in kd:
        meta.append(f'\t.set kd.accum_offset, {kd["accum_offset"]}\n')
    if 'next_free_sgpr' in kd:
        meta.append(f'\t.set kd.next_free_sgpr, {kd["next_free_sgpr"]}\n')
    if 'private_segment_fixed_size' in kd:
        meta.append(f'\t.set kd.private_segment_fixed_size, {kd["private_segment_fixed_size"]}\n')
    return meta


def extract_lds_from_yaml(lines):
    """Extract .group_segment_fixed_size from the kernel YAML metadata."""
    for line in lines:
        m = re.search(r'\.group_segment_fixed_size:\s*(\d+)', line)
        if m:
            return int(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='Input .s assembly file')
    parser.add_argument('output', help='Output stripped .s file')
    parser.add_argument('--meta', help='Write metadata sidecar (.meta) file')
    parser.add_argument('--check-lds', type=int, default=None,
                        help='Assert group_segment_fixed_size equals this value')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        lines = f.readlines()

    kernel_name = find_kernel_name(lines)
    devfunc_names = find_devfunc_name(lines)

    if not devfunc_names:
        fatal("No ncclDevFunc_* symbol found in assembly")

    #print(f"  kernel:  {kernel_name[:80]}...")
    #print(f"  devfunc: {devfunc_names[0][:80]}...")

    # Find the kernel function block (code + .amdhsa_kernel + everything)
    kern_start, kern_end = find_function_block(lines, kernel_name)
    #print(f"  kernel function: lines {kern_start+1}-{kern_end+1} "
    #      f"({kern_end - kern_start + 1} lines)")

    # Collect line indices to remove
    remove = set(range(kern_start, kern_end + 1))

    # Remove per-kernel .set directives after the function block
    for i in range(kern_end + 1, min(kern_end + 50, len(lines))):
        line = lines[i]
        if line.strip().startswith('.set') and kernel_name in line:
            remove.add(i)

    # Remove the kernel's .AMDGPU.csdata block
    csdata = find_csdata_block(lines, kern_end + 1, is_kernel=True)
    if csdata:
        for i in range(csdata[0], csdata[1] + 1):
            remove.add(i)

    # Extract metadata sidecar before stripping
    meta_lines = extract_meta(lines)

    # Check LDS if requested
    lds = extract_lds_from_yaml(lines)
    if lds is not None:
        #print(f"  LDS (group_segment_fixed_size): {lds}")
        if args.check_lds is not None and lds != args.check_lds:
            fatal(f"LDS mismatch: expected {args.check_lds}, got {lds}")

    # Build output: remove kernel lines but preserve .file directives
    # (needed by DWARF .loc references in the kept device function code)
    kept = [lines[i] for i in range(len(lines))
            if i not in remove or re.match(r'\s*\.file\s+\d+', lines[i])]
    kept = strip_kernel_yaml(kept)
    kept = strip_debug_sections(kept)

    removed_count = len(remove)
    kept_count = len(kept)
    #print(f"  removed {removed_count} lines, kept {kept_count} lines")

    with open(args.output, 'w') as f:
        f.writelines(kept)

    if args.meta:
        with open(args.meta, 'w') as f:
            f.writelines(meta_lines)
        #print(f"  wrote metadata sidecar: {args.meta}")


if __name__ == '__main__':
    main()
