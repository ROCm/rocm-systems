#!/usr/bin/env python3
"""
Patch kernel descriptors in a device ELF to reflect the actual VGPR and
scratch requirements of indirect callees.

In the split-device pipeline, device functions are compiled separately from
the kernel.  The kernel dispatches through a function pointer table (indirect
calls).  Neither the compiler nor the linker propagates callee resource
requirements to the kernel descriptor for indirect calls, leaving:

  - granulated_workitem_vgpr_count too low
  - private_segment_fixed_size at 0 despite callees spilling to scratch

This script analyses every device-function object in the build directory,
determines the maximum VGPR/AGPR/scratch usage, and patches the kernel
object's .kd binary descriptors so that the hardware allocates enough
resources at dispatch time.
"""

import argparse
import os
import re
import struct
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# AMDHSA Kernel Descriptor offsets (64 bytes total)
# Reference: llvm/include/llvm/Support/AMDHSAKernelDescriptor.h
KD_PRIVATE_SEGMENT_FIXED_SIZE_OFF = 4   # uint32 LE
KD_COMPUTE_PGM_RSRC3_OFF = 44          # uint32 LE (GFX90A+)
KD_COMPUTE_PGM_RSRC1_OFF = 48          # uint32 LE
KD_KERNEL_CODE_PROPERTIES_OFF = 56     # uint16 LE

# COMPUTE_PGM_RSRC1 bit fields
RSRC1_VGPR_MASK = 0x3F                  # bits [5:0]

# COMPUTE_PGM_RSRC3 bit fields (GFX90A+)
RSRC3_ACCUM_OFFSET_MASK = 0x3F          # bits [5:0]

# kernel_code_properties bit fields
KCP_USES_DYNAMIC_STACK_BIT = 11

# Scratch instruction access sizes in bytes
_SCRATCH_ACCESS_SIZE = {
    "byte": 1,
    "ubyte": 1,
    "short": 2,
    "ushort": 2,
    "dword": 4,
    "dwordx2": 8,
    "dwordx3": 12,
    "dwordx4": 16,
}

# Architectures with unified VGPR/ACCVGPR register file (granularity = 8)
_UNIFIED_VGPR_ARCHS = frozenset((
    "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950",
))


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

def _find_tool(name: str, rocm_path: str) -> str:
    for d in ("llvm/bin", "bin"):
        p = os.path.join(rocm_path, d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError(f"Cannot find {name} under {rocm_path}")


def _align_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def get_vgpr_granularity(gpu_arch: str) -> int:
    if gpu_arch in _UNIFIED_VGPR_ARCHS:
        return 8
    return 4


# ---------------------------------------------------------------------------
# Analyse device-function objects
# ---------------------------------------------------------------------------

_REG_RE = re.compile(
    r"(?:^|[,\s\[\(])"
    r"([sva])"
    r"(?:\[(\d+):(\d+)\]|(\d+))"
    r"(?=[,\s\]\)\n]|$)"
)

_SCRATCH_RE = re.compile(
    r"\bscratch_(?:load|store)_(\w+)"
)

_SCRATCH_OFFSET_RE = re.compile(
    r"\boffset:(\d+)"
)

_SKIP_PREFIXES = ("__cxa_", "__ockl_", "__assert")


def _scratch_access_size(suffix: str) -> int:
    """Return access size in bytes for a scratch instruction suffix like
    'dwordx4', 'byte', 'ubyte', etc."""
    for key, size in _SCRATCH_ACCESS_SIZE.items():
        if suffix == key:
            return size
    return 4  # default to dword


def analyze_device_objects(dev_obj_dir: str, gpu_arch: str,
                           rocm_path: str, kernel_obj_stem: str) -> dict:
    """Scan all device-function objects and return aggregate resource usage.

    Takes independent maximums for VGPRs and AGPRs across all functions.
    This is correct because all callees share the same kernel descriptor:
    ACCUM_OFFSET is driven by the max-VGPR function, and the total
    allocation must also leave room above that boundary for the max-AGPR
    function — even though those may be different functions.

    Returns dict with keys: max_vgpr, max_agpr, max_scratch.
    """
    objdump = _find_tool("llvm-objdump", rocm_path)

    glob_pattern = f"*.{gpu_arch}.o"
    objs = sorted(Path(dev_obj_dir).glob(glob_pattern))
    if not objs:
        print(f"  WARNING: no objects matching {glob_pattern} in {dev_obj_dir}",
              file=sys.stderr)
        return {"max_vgpr": 0, "max_agpr": 0, "max_scratch": 0}

    global_max_v = -1
    global_max_a = -1
    global_max_scratch = 0

    for obj in objs:
        stem = obj.name.split(f".{gpu_arch}")[0]
        if stem == kernel_obj_stem:
            continue

        r = subprocess.run(
            [objdump, "-d", "--no-show-raw-insn", str(obj)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            continue

        cur_func = None
        func_max_v: dict[str, int] = {}
        func_max_a: dict[str, int] = {}
        func_scratch: dict[str, int] = {}

        for line in r.stdout.splitlines():
            lbl = re.match(r"^[0-9a-f]+\s+<(.+?)>:", line)
            if lbl:
                cur_func = lbl.group(1)
                if any(cur_func.startswith(p) for p in _SKIP_PREFIXES):
                    cur_func = None
                    continue
                func_max_v.setdefault(cur_func, -1)
                func_max_a.setdefault(cur_func, -1)
                func_scratch.setdefault(cur_func, 0)
                continue

            if cur_func is None:
                continue

            s = line.strip()
            if not s or s.startswith(";"):
                continue
            cp = s.find("//")
            if cp > 0:
                s = s[:cp]

            for m in _REG_RE.finditer(s):
                fc = m.group(1)
                if m.group(2) is not None:
                    rn = int(m.group(3))
                elif m.group(4) is not None:
                    rn = int(m.group(4))
                else:
                    continue
                if fc == "v":
                    func_max_v[cur_func] = max(func_max_v[cur_func], rn)
                elif fc == "a":
                    func_max_a[cur_func] = max(func_max_a[cur_func], rn)

            sm = _SCRATCH_RE.search(s)
            if sm:
                suffix = sm.group(1)
                access_sz = _scratch_access_size(suffix)
                om = _SCRATCH_OFFSET_RE.search(s)
                offset = int(om.group(1)) if om else 0
                frame = offset + access_sz
                func_scratch[cur_func] = max(func_scratch[cur_func], frame)

        for v in func_max_v.values():
            global_max_v = max(global_max_v, v)
        for a in func_max_a.values():
            global_max_a = max(global_max_a, a)
        for scr in func_scratch.values():
            global_max_scratch = max(global_max_scratch, scr)

    return {
        "max_vgpr": global_max_v + 1 if global_max_v >= 0 else 0,
        "max_agpr": global_max_a + 1 if global_max_a >= 0 else 0,
        "max_scratch": global_max_scratch,
    }


# ---------------------------------------------------------------------------
# Compute required kernel descriptor values
# ---------------------------------------------------------------------------

def compute_granulated_vgprs(max_vgpr: int, max_agpr: int,
                             granularity: int) -> int:
    """Compute the granulated_workitem_vgpr_count for COMPUTE_PGM_RSRC1."""
    if granularity == 8:
        total = _align_to(max_vgpr, 4) + max_agpr
    else:
        total = max_vgpr
    if total <= 0:
        return 0
    return (_align_to(total, granularity) // granularity) - 1


def compute_accum_offset(max_vgpr: int) -> int:
    """Compute the ACCUM_OFFSET for COMPUTE_PGM_RSRC3 (GFX90A+).

    ACCUM_OFFSET = ceil(num_non_accumulator_vgprs / 4) - 1
    This sets the boundary between regular VGPRs and ACCVGPRs in the
    unified register file.
    """
    if max_vgpr <= 0:
        return 0
    return _align_to(max_vgpr, 4) // 4 - 1


# ---------------------------------------------------------------------------
# Patch kernel descriptors in the kernel device ELF
# ---------------------------------------------------------------------------

def _find_kd_symbols(kernel_obj: str, rocm_path: str) -> list:
    """Return list of (virtual_address, symbol_name) for all .kd symbols."""
    objdump = _find_tool("llvm-objdump", rocm_path)
    r = subprocess.run([objdump, "--syms", kernel_obj],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"llvm-objdump --syms failed: {r.stderr}")

    kd_syms = []
    for line in r.stdout.splitlines():
        if ".kd" not in line:
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        addr = int(parts[0], 16)
        name = parts[-1]
        kd_syms.append((addr, name))
    return kd_syms


def _find_rodata_section(kernel_obj: str, rocm_path: str) -> tuple:
    """Return (virtual_address, file_offset) for .rodata."""
    readelf = _find_tool("llvm-readelf", rocm_path)
    r = subprocess.run([readelf, "-S", kernel_obj],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"llvm-readelf -S failed: {r.stderr}")

    for line in r.stdout.splitlines():
        if ".rodata" in line and "PROGBITS" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "PROGBITS":
                    return int(parts[i + 1], 16), int(parts[i + 2], 16)
    raise RuntimeError(f"No .rodata section found in {kernel_obj}")


def patch_kernel_object(kernel_obj: str, required_gran_vgprs: int,
                        required_accum_offset: int,
                        required_scratch: int, gpu_arch: str,
                        rocm_path: str) -> int:
    """Patch all .kd descriptors with uses_dynamic_stack=true in kernel_obj.

    Returns the number of descriptors patched.
    """
    is_unified = gpu_arch in _UNIFIED_VGPR_ARCHS
    kd_syms = _find_kd_symbols(kernel_obj, rocm_path)
    if not kd_syms:
        print("  WARNING: no .kd symbols found", file=sys.stderr)
        return 0

    rodata_va, rodata_off = _find_rodata_section(kernel_obj, rocm_path)

    with open(kernel_obj, "r+b") as f:
        data = bytearray(f.read())

        patched = 0
        for sym_va, sym_name in kd_syms:
            kd_off = sym_va - rodata_va + rodata_off

            kcp = struct.unpack_from("<H", data, kd_off + KD_KERNEL_CODE_PROPERTIES_OFF)[0]
            uses_dyn_stack = bool(kcp & (1 << KCP_USES_DYNAMIC_STACK_BIT))

            if not uses_dyn_stack:
                continue

            rsrc1 = struct.unpack_from("<I", data, kd_off + KD_COMPUTE_PGM_RSRC1_OFF)[0]
            cur_gran = rsrc1 & RSRC1_VGPR_MASK
            cur_scratch = struct.unpack_from("<I", data, kd_off + KD_PRIVATE_SEGMENT_FIXED_SIZE_OFF)[0]

            new_gran = max(cur_gran, required_gran_vgprs)
            new_scratch = max(cur_scratch, required_scratch)

            # On CDNA (unified VGPR/AGPR), also patch ACCUM_OFFSET in RSRC3
            new_accum = None
            cur_accum = None
            if is_unified:
                rsrc3 = struct.unpack_from("<I", data, kd_off + KD_COMPUTE_PGM_RSRC3_OFF)[0]
                cur_accum = rsrc3 & RSRC3_ACCUM_OFFSET_MASK
                new_accum = max(cur_accum, required_accum_offset)

            needs_patch = (new_gran != cur_gran or
                           new_scratch != cur_scratch or
                           (new_accum is not None and new_accum != cur_accum))

            if needs_patch:
                new_rsrc1 = (rsrc1 & ~RSRC1_VGPR_MASK) | (new_gran & RSRC1_VGPR_MASK)
                struct.pack_into("<I", data, kd_off + KD_COMPUTE_PGM_RSRC1_OFF, new_rsrc1)
                struct.pack_into("<I", data, kd_off + KD_PRIVATE_SEGMENT_FIXED_SIZE_OFF, new_scratch)

                if is_unified and new_accum != cur_accum:
                    new_rsrc3 = (rsrc3 & ~RSRC3_ACCUM_OFFSET_MASK) | (new_accum & RSRC3_ACCUM_OFFSET_MASK)
                    struct.pack_into("<I", data, kd_off + KD_COMPUTE_PGM_RSRC3_OFF, new_rsrc3)

                patched += 1
                short_name = sym_name.split(".kd")[0]
                print(f"  Patched {short_name}:", file=sys.stderr)
                print(f"    VGPRs: gran {cur_gran} -> {new_gran}", file=sys.stderr)
                if is_unified:
                    print(f"    ACCUM_OFFSET: {cur_accum} -> {new_accum} "
                          f"(regular VGPRs: {(cur_accum+1)*4} -> {(new_accum+1)*4})",
                          file=sys.stderr)
                print(f"    Scratch: {cur_scratch} -> {new_scratch} bytes", file=sys.stderr)

        if patched:
            f.seek(0)
            f.write(data)
            f.truncate()

    return patched


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Patch kernel descriptors to reflect indirect callee "
                    "VGPR/scratch requirements.",
    )
    p.add_argument("--kernel-obj", required=True,
                   help="Path to the kernel device ELF (e.g. common.gfx942.o)")
    p.add_argument("--dev-obj-dir", required=True,
                   help="Directory containing all device-function ELF objects")
    p.add_argument("--gpu-arch", required=True,
                   help="GPU architecture (e.g. gfx942)")
    p.add_argument("--rocm-path", default="/opt/rocm",
                   help="ROCm install path")
    args = p.parse_args()

    kernel_stem = Path(args.kernel_obj).name.split(f".{args.gpu_arch}")[0]
    granularity = get_vgpr_granularity(args.gpu_arch)

    print(f"Analyzing device objects in {args.dev_obj_dir} "
          f"(arch={args.gpu_arch}, vgpr_granularity={granularity}) ...",
          file=sys.stderr)

    usage = analyze_device_objects(
        args.dev_obj_dir, args.gpu_arch, args.rocm_path, kernel_stem,
    )
    print(f"  Max VGPR: {usage['max_vgpr']}  "
          f"Max AGPR: {usage['max_agpr']}  "
          f"Max scratch: {usage['max_scratch']} bytes",
          file=sys.stderr)

    if usage["max_vgpr"] == 0 and usage["max_scratch"] == 0:
        print("  No device functions found or all have zero usage — "
              "skipping patch.", file=sys.stderr)
        return

    required_gran = compute_granulated_vgprs(
        usage["max_vgpr"], usage["max_agpr"], granularity,
    )
    required_accum = compute_accum_offset(usage["max_vgpr"])
    alloc_vgprs = (required_gran + 1) * granularity

    print(f"  Required: granulated_vgpr={required_gran} "
          f"(allocates {alloc_vgprs} VGPRs), "
          f"accum_offset={required_accum} "
          f"(regular VGPRs: {(required_accum + 1) * 4}), "
          f"scratch={usage['max_scratch']} bytes",
          file=sys.stderr)

    n = patch_kernel_object(
        args.kernel_obj, required_gran, required_accum,
        usage["max_scratch"], args.gpu_arch, args.rocm_path,
    )
    print(f"  Patched {n} kernel descriptor(s) in {args.kernel_obj}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
