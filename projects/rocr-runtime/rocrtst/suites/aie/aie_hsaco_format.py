#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""On-disk layout of the AIE hsaco section, shared by the packer and the dumper.

This mirrors `core/inc/amd_aie_section.h`. The C++ header is authoritative; when a field is
added or a reserved word repurposed there, change it here once rather than in each tool.
"""
import struct

# Section magic: 'A','I','E','K' little-endian. Must match kAieSectionMagic.
MAGIC = 0x4B454941
VERSION_MAJOR = 1
VERSION_MINOR = 0

ARCHES = ("aie2", "aie2p")

# aie_section_header: magic, version_major, version_minor, header_size, kernel_count,
# kernel_entry_size, string_table_offset, string_table_size, blob_pool_offset, reserved[4].
_HDR = "<IHHIIIIII" + "IIII"
_HDR_SIZE = struct.calcsize(_HDR)

# aie_kernel_entry: name_offset, insts_offset, insts_size, pdi_offset, pdi_size, kernarg_size,
# num_cols, kind, reserved[3].
_ENTRY = "<IIIIIII" + "IIII"
_ENTRY_SIZE = struct.calcsize(_ENTRY)

# AieKernelKind. KIND_COUNT is the validation bound: anything at or above it is rejected, which
# is what keeps a future kind from being mis-read by a tool that predates it. The enum's
# Undecided shares Count's value but is runtime-only and never appears on disk.
KIND_COUNT = 2
KIND_NAMES = {0: "PdiInsts", 1: "FullElf"}


def header_fields(section):
    """Returns (hdr_size, kernel_entry_size) from a section's header."""
    (_magic, _vmaj, _vmin, hdr_size, _kcount, kentry, *_rest) = struct.unpack_from(_HDR, section, 0)
    return hdr_size, kentry
