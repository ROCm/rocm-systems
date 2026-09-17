#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Inject an aie2/aie2p section (versioned header + kernel table + blob pool)
into an hsaco. See docs/superpowers/specs/2026-07-12-aie-hsaco-loading-design.md."""
import argparse
import os
import struct
import subprocess
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
        kind = int(k.get("kind", 0))
        if kind not in (0, 1):
            raise ValueError(f"kernel {k['name']!r}: unknown kind {kind}")
        if kind == 1 and pdi_size:
            raise ValueError(f"kernel {k['name']!r}: FullElf entries carry no separate PDI")
        entries.append((name_off,
                        blob_pool_offset + insts_off, insts_size,
                        (blob_pool_offset + pdi_off) if pdi_size else 0, pdi_size,
                        int(k.get("kernarg_size", 0)), int(k.get("num_cols", 1)),
                        kind, 0, 0, 0))

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


def _kernel_name_from_symbol(symbol):
    """'_Z4mainPcPcPc' -> 'main'. Anything else is returned unchanged."""
    if not symbol.startswith("_Z"):
        return symbol
    i, length = 2, 0
    while i < len(symbol) and symbol[i].isdigit():
        length = length * 10 + int(symbol[i])
        i += 1
    if length == 0 or i + length > len(symbol):
        return symbol
    return symbol[i:i + length]


def kernels_from_full_elf(path, kernarg_size=0, num_cols=1):
    """One dict per kernel in a full ELF; every entry embeds the whole ELF."""
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection

    with open(path, "rb") as f:
        blob = f.read()
        f.seek(0)
        elf = ELFFile(f)
        symtab = elf.get_section_by_name(".symtab")
        if not isinstance(symtab, SymbolTableSection):
            raise ValueError(f"{path}: missing .symtab")

        names = []
        for sec in elf.iter_sections():
            if sec["sh_type"] != "SHT_GROUP":
                continue
            instance = symtab.get_symbol(sec["sh_info"])
            kernel = symtab.get_symbol(instance["st_shndx"])
            names.append(f"{_kernel_name_from_symbol(kernel.name)}:{instance.name}")

    if not names:
        raise ValueError(f"{path}: no COMDAT groups; not a full ELF")
    return [dict(name=n, insts=blob, pdi=None, kernarg_size=kernarg_size,
                 num_cols=num_cols, kind=1) for n in sorted(names)]


def pdi_from_xclbin(path):
    """Extracts the PDI from an xclbin's AIE_PARTITION section via xclbinutil."""
    import glob
    import shutil
    import tempfile
    if shutil.which("xclbinutil") is None:
        raise RuntimeError(
            "xclbinutil not found on PATH; it is needed to read a PDI out of an xclbin. "
            "Install XRT or pass the PDI directly with the PDI+insts --kernel form.")
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["xclbinutil", "--input", path,
                        "--dump-section", f"AIE_PARTITION:JSON:{d}/aie.json"],
                       check=True, capture_output=True)
        pdis = glob.glob(f"{d}/**/*.pdi", recursive=True)
        if len(pdis) != 1:
            raise ValueError(f"{path}: expected exactly one PDI, found {len(pdis)}")
        with open(pdis[0], "rb") as f:
            return f.read()


# Minimal ELF64 little-endian container: a header plus a NULL section and an
# empty .shstrtab. llvm-objcopy needs a real .shstrtab to write new section
# names into; an ELF with e_shnum=0 has nowhere for --add-section to put one,
# so --add-section silently no-ops on it instead of failing.
def _make_empty_elf64():
    _SHDR = "<IIQQQQIIQQ"
    ehsize = 64
    shentsize = struct.calcsize(_SHDR)
    shstrtab = b"\x00.shstrtab\x00"
    shstrtab_off = ehsize
    shoff = shstrtab_off + len(shstrtab)
    null_sh = struct.pack(_SHDR, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    shstrtab_sh = struct.pack(_SHDR, 1, 3, 0, 0, shstrtab_off, len(shstrtab), 0, 0, 1, 0)
    ehdr = (
        b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8 +   # e_ident
        struct.pack("<HHIQQQIHHHHHH",
                    1,          # e_type = ET_REL
                    224,        # e_machine = EM_AMDGPU
                    1,          # e_version
                    0, 0, shoff,  # e_entry, e_phoff, e_shoff
                    0,          # e_flags
                    ehsize,     # e_ehsize
                    0, 0,       # e_phentsize, e_phnum
                    shentsize, 2, 1)  # e_shentsize, e_shnum, e_shstrndx
    )
    return ehdr + shstrtab + null_sh + shstrtab_sh


_EMPTY_ELF64 = _make_empty_elf64()


def ensure_hsaco(path):
    """Creates a minimal hsaco at `path` if it does not already exist."""
    if os.path.exists(path):
        return
    with open(path, "wb") as f:
        f.write(_EMPTY_ELF64)


def _parse_kernel_arg(s):
    # elf:PATH[:KERNARG_SIZE[:NUM_COLS]]
    # xclbin:NAME:XCLBIN:INSTS:KERNARG_SIZE:NUM_COLS
    # NAME:INSTS[:PDI]:KERNARG_SIZE:NUM_COLS   (original PDI+insts form)
    parts = s.split(":")
    if parts[0] == "elf":
        if not 2 <= len(parts) <= 4:
            raise argparse.ArgumentTypeError(f"bad --kernel spec {s!r}")
        kernarg_size = int(parts[2]) if len(parts) > 2 else 0
        num_cols = int(parts[3]) if len(parts) > 3 else 1
        return kernels_from_full_elf(parts[1], kernarg_size, num_cols)
    if parts[0] == "xclbin":
        if len(parts) != 6:
            raise argparse.ArgumentTypeError(f"bad --kernel spec {s!r}")
        _, name, xclbin_path, insts_path, kernarg_size, num_cols = parts
        with open(insts_path, "rb") as f:
            insts = f.read()
        return [dict(name=name, insts=insts, pdi=pdi_from_xclbin(xclbin_path),
                     kernarg_size=int(kernarg_size), num_cols=int(num_cols), kind=0)]
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
    return [dict(name=name, insts=insts, pdi=pdi, kernarg_size=int(kernarg_size),
                 num_cols=int(num_cols), kind=0)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsaco", required=True)
    ap.add_argument("--arch", required=True, choices=ARCHES)
    ap.add_argument("--kernel", required=True, action="append", type=_parse_kernel_arg)
    args = ap.parse_args(argv)
    ensure_hsaco(args.hsaco)
    kernels = [k for group in args.kernel for k in group]
    section = build_section(args.arch, kernels)
    _inject(args.hsaco, args.arch, section)
    return 0


if __name__ == "__main__":
    sys.exit(main())
