#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Parse and validate an aie2/aie2p hsaco section; print human-readable summary."""
import argparse
import struct
import sys

MAGIC = 0x4B454941
_HDR = "<IHHIIIIII" + "IIII"
_HDR_SIZE = struct.calcsize(_HDR)
_ENTRY = "<IIIIIII" + "IIII"


def parse_section(section):
    if len(section) < _HDR_SIZE:
        raise ValueError("section smaller than header")
    (magic, vmaj, vmin, hdr_size, kcount, kentry, st_off, st_size,
     pool_off, *_res) = struct.unpack_from(_HDR, section, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}")
    if kentry < struct.calcsize(_ENTRY):
        raise ValueError("kernel_entry_size too small")
    if hdr_size + kcount * kentry > len(section):
        raise ValueError("kernel table out of bounds")

    def in_section(off, ln):
        return off <= len(section) if ln == 0 else (off < len(section) and off + ln <= len(section))

    kernels = []
    for i in range(kcount):
        base = hdr_size + i * kentry
        (name_off, insts_off, insts_size, pdi_off, pdi_size,
         kernarg_size, num_cols, *_e) = struct.unpack_from(_ENTRY, section, base)
        if insts_size == 0 or not in_section(insts_off, insts_size):
            raise ValueError(f"kernel {i}: insts out of bounds/overrun")
        if pdi_size and not in_section(pdi_off, pdi_size):
            raise ValueError(f"kernel {i}: pdi out of bounds/overrun")
        name_abs = st_off + name_off
        if name_abs >= len(section):
            raise ValueError(f"kernel {i}: name out of bounds")
        end = section.find(b"\x00", name_abs)
        if end == -1:
            raise ValueError(f"kernel {i}: name not terminated")
        kernels.append(dict(name=section[name_abs:end].decode(),
                            insts_size=insts_size, has_pdi=bool(pdi_size),
                            pdi_size=pdi_size, kernarg_size=kernarg_size,
                            num_cols=num_cols))
    return dict(arch_version=(vmaj, vmin), kernel_count=kcount, kernels=kernels)


def _read_section_from_hsaco(path, arch):
    from elftools.elf.elffile import ELFFile
    with open(path, "rb") as f:
        elf = ELFFile(f)
        for name in ("aie2", "aie2p"):
            sec = elf.get_section_by_name(name)
            if sec is not None:
                return name, sec.data()
    raise ValueError("no aie2/aie2p section found")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsaco", required=True)
    args = ap.parse_args(argv)
    arch, data = _read_section_from_hsaco(args.hsaco, None)
    info = parse_section(data)
    print(f"arch section: {arch}")
    print(f"version: {info['arch_version'][0]}.{info['arch_version'][1]}")
    for k in info["kernels"]:
        print(f"  kernel {k['name']}: insts={k['insts_size']}B "
              f"pdi={'yes' if k['has_pdi'] else 'no'} "
              f"kernarg={k['kernarg_size']} cols={k['num_cols']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
