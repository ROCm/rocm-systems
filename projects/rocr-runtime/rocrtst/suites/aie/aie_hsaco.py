#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Inject an aie2/aie2p section (versioned header + kernel table + blob pool)
into an hsaco. See docs/superpowers/specs/2026-07-12-aie-hsaco-loading-design.md."""
import argparse
import struct
import sys

MAGIC = 0x4B454941
VERSION_MAJOR = 1
VERSION_MINOR = 0
ARCHES = ("aie2", "aie2p")

_HDR = "<IHHIIIIII" + "IIII"          # header + reserved[4]
_HDR_SIZE = struct.calcsize(_HDR)
_ENTRY = "<IIIIIII" + "IIII"          # 7 fields + reserved[4]
_ENTRY_SIZE = struct.calcsize(_ENTRY)


def build_section(arch, kernels):
    if arch not in ARCHES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHES}")

    string_table = bytearray()
    name_offsets = []
    for k in kernels:
        name_offsets.append(len(string_table))
        string_table += k["name"].encode() + b"\x00"

    # Blob pool with dedup keyed on raw bytes.
    pool = bytearray()
    blob_off = {}

    def place(blob):
        if not blob:
            return (0, 0)
        key = bytes(blob)
        if key not in blob_off:
            blob_off[key] = len(pool)
            pool.extend(key)
        return (blob_off[key], len(key))

    # Layout: [header][kernel table][string table][blob pool]
    header_size = _HDR_SIZE
    table_size = _ENTRY_SIZE * len(kernels)
    string_table_offset = header_size + table_size
    blob_pool_offset = string_table_offset + len(string_table)

    entries = []
    for k, name_off in zip(kernels, name_offsets):
        insts_off, insts_size = place(k["insts"])
        if insts_size == 0:
            raise ValueError(f"kernel {k['name']!r}: insts must be non-empty")
        pdi_off, pdi_size = place(k.get("pdi"))
        entries.append((name_off,
                        blob_pool_offset + insts_off, insts_size,
                        (blob_pool_offset + pdi_off) if pdi_size else 0, pdi_size,
                        int(k.get("kernarg_size", 0)), int(k.get("num_cols", 1)),
                        0, 0, 0, 0))

    out = bytearray()
    out += struct.pack(_HDR, MAGIC, VERSION_MAJOR, VERSION_MINOR, header_size,
                       len(kernels), _ENTRY_SIZE, string_table_offset,
                       len(string_table), blob_pool_offset, 0, 0, 0, 0)
    for e in entries:
        out += struct.pack(_ENTRY, *e)
    out += string_table
    out += pool
    return bytes(out)


def _inject(hsaco_path, arch, section_bytes):
    import os
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(section_bytes)
        sec_file = f.name
    try:
        # Remove an existing same-named section first (ignore error if absent).
        subprocess.run(["llvm-objcopy", "--remove-section", arch, hsaco_path,
                        hsaco_path], check=False)
        subprocess.run(["llvm-objcopy", f"--add-section={arch}={sec_file}",
                        f"--set-section-flags={arch}=noload,readonly",
                        hsaco_path, hsaco_path], check=True)
    finally:
        os.unlink(sec_file)


def _parse_kernel_arg(s):
    # NAME:INSTS[:PDI]:KERNARG_SIZE:NUM_COLS  (PDI optional)
    parts = s.split(":")
    if len(parts) == 4:
        name, insts_path, kernarg_size, num_cols = parts
        pdi_path = None
    elif len(parts) == 5:
        name, insts_path, pdi_path, kernarg_size, num_cols = parts
    else:
        raise argparse.ArgumentTypeError(f"bad --kernel spec {s!r}")
    with open(insts_path, "rb") as f:
        insts = f.read()
    pdi = None
    if pdi_path:
        with open(pdi_path, "rb") as f:
            pdi = f.read()
    return dict(name=name, insts=insts, pdi=pdi,
                kernarg_size=int(kernarg_size), num_cols=int(num_cols))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsaco", required=True)
    ap.add_argument("--arch", required=True, choices=ARCHES)
    ap.add_argument("--kernel", required=True, action="append", type=_parse_kernel_arg)
    args = ap.parse_args(argv)
    section = build_section(args.arch, args.kernel)
    _inject(args.hsaco, args.arch, section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
