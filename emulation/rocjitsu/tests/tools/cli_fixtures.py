# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import struct


def write_code_object(path, machine=0x4F):
    # Minimal ELF64 AMDGPU object with executable .text containing CDNA s_nop
    # and s_endpgm. gfx950 machine flag matches amdgpu_elf.h.
    names = b"\0.text\0.shstrtab\0"
    code = struct.pack("<II", 0xBF800000, 0xBF810000)
    sections = 64 + len(code) + len(names)
    sections = (sections + 7) & ~7
    ident = b"\x7fELF\x02\x01\x01\x40" + bytes(8)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        ident,
        3,
        224,
        1,
        0,
        0,
        sections,
        machine,
        64,
        0,
        0,
        64,
        3,
        2,
    )
    text = struct.pack("<IIQQQQIIQQ", 1, 1, 6, 0, 64, len(code), 0, 0, 4, 0)
    strings = struct.pack(
        "<IIQQQQIIQQ", 7, 3, 0, 0, 64 + len(code), len(names), 0, 0, 1, 0
    )
    payload = header + code + names
    path.write_bytes(
        payload + bytes(sections - len(payload)) + bytes(64) + text + strings
    )
