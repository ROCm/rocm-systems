"""
ELF binary patching utilities for testing edge cases.

This module provides tools to create modified ELF binaries for testing
scenarios that are difficult to reproduce with real compiled binaries,
such as binaries with unusually small .hip_fatbin sections.

Background:
-----------
The kpack artifact splitter processes "fat binaries" - ELF files containing
GPU device code in a .hip_fatbin section. The splitter:
1. Extracts device code to separate .kpack files
2. Adds a .rocm_kpack_ref marker section pointing to the kpack
3. Zero-pages the .hip_fatbin section to reclaim space

The zero-paging optimization works by removing full 4KB pages from the
.hip_fatbin section. This creates edge cases when:
- Section is < 4KB: No full pages exist to zero-page
- Section is slightly > 4KB: Structural overhead (padding, PHDR relocation)
  may exceed the ~4KB savings

These edge cases occur in practice with small test binaries or validation
utilities that have minimal GPU code.

Usage:
------
    from tests.elf_test_utils import patch_hip_fatbin_size

    # Create a binary with a tiny .hip_fatbin (will fail zero-paging)
    small_binary = patch_hip_fatbin_size(
        source_binary,
        output_path,
        new_size=3000  # Less than one 4KB page
    )

    # Create a binary where overhead exceeds savings
    marginal_binary = patch_hip_fatbin_size(
        source_binary,
        output_path,
        new_size=5000  # Just over one page - saves 4KB but adds ~5KB overhead
    )
"""

import struct
from pathlib import Path
from typing import Optional


class ElfPatchError(Exception):
    """Raised when ELF patching fails."""

    pass


def patch_hip_fatbin_size(
    source_path: Path,
    output_path: Path,
    new_size: int,
) -> Path:
    """
    Create a copy of an ELF binary with a modified .hip_fatbin section size.

    This patches the section header to report a smaller size, which is
    sufficient to trigger the edge cases in zero-paging logic. The actual
    section content is not modified (the size field just tells tools how
    much of the section to consider valid).

    Args:
        source_path: Path to source ELF binary with .hip_fatbin section
        output_path: Path for the patched output binary
        new_size: New size in bytes for .hip_fatbin section.
                  Use < 4096 for "too small to zero-page" case.
                  Use ~5000 for "overhead exceeds savings" case.

    Returns:
        Path to the created output file

    Raises:
        ElfPatchError: If source is not a valid ELF or has no .hip_fatbin
        FileNotFoundError: If source file doesn't exist

    Example section sizes and their effects:
        3000 bytes: Cannot zero any pages (< 4KB), zero-paging fails
        5000 bytes: Can zero 1 page (4KB), but padding/relocation adds ~5KB
        9000 bytes: Can zero 2 pages (8KB), usually net savings
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source binary not found: {source_path}")

    data = bytearray(source_path.read_bytes())

    # Verify ELF magic
    if data[:4] != b"\x7fELF":
        raise ElfPatchError(f"Not an ELF file: {source_path}")

    # Parse ELF64 header to find section headers
    # ELF64 header layout:
    #   0x00: e_ident[16] - ELF identification
    #   0x10: e_type[2] - Object file type
    #   0x12: e_machine[2] - Architecture
    #   0x14: e_version[4] - Object file version
    #   0x18: e_entry[8] - Entry point virtual address
    #   0x20: e_phoff[8] - Program header table offset
    #   0x28: e_shoff[8] - Section header table offset
    #   0x30: e_flags[4] - Processor-specific flags
    #   0x34: e_ehsize[2] - ELF header size
    #   0x36: e_phentsize[2] - Program header entry size
    #   0x38: e_phnum[2] - Program header count
    #   0x3a: e_shentsize[2] - Section header entry size
    #   0x3c: e_shnum[2] - Section header count
    #   0x3e: e_shstrndx[2] - Section name string table index

    e_shoff = struct.unpack("<Q", data[0x28:0x30])[0]
    e_shentsize = struct.unpack("<H", data[0x3A:0x3C])[0]
    e_shnum = struct.unpack("<H", data[0x3C:0x3E])[0]
    e_shstrndx = struct.unpack("<H", data[0x3E:0x40])[0]

    if e_shentsize != 64:
        raise ElfPatchError(
            f"Unexpected section header size: {e_shentsize} (expected 64)"
        )

    # Get string table section to read section names
    # Section header layout (Elf64_Shdr, 64 bytes):
    #   0x00: sh_name[4] - Section name (index into string table)
    #   0x04: sh_type[4] - Section type
    #   0x08: sh_flags[8] - Section flags
    #   0x10: sh_addr[8] - Virtual address
    #   0x18: sh_offset[8] - File offset
    #   0x20: sh_size[8] - Section size  <-- This is what we patch
    #   0x28: sh_link[4] - Link to another section
    #   0x2c: sh_info[4] - Additional section info
    #   0x30: sh_addralign[8] - Alignment
    #   0x38: sh_entsize[8] - Entry size if section holds table

    shstrtab_hdr_off = e_shoff + e_shstrndx * e_shentsize
    shstrtab_offset = struct.unpack(
        "<Q", data[shstrtab_hdr_off + 0x18 : shstrtab_hdr_off + 0x20]
    )[0]

    # Find .hip_fatbin section
    hip_fatbin_found = False
    original_size = 0

    for i in range(e_shnum):
        sh_off = e_shoff + i * e_shentsize
        sh_name_idx = struct.unpack("<I", data[sh_off : sh_off + 4])[0]

        # Read null-terminated section name from string table
        name_start = shstrtab_offset + sh_name_idx
        name_end = data.index(0, name_start)
        section_name = data[name_start:name_end].decode("ascii", errors="replace")

        if section_name == ".hip_fatbin":
            # Found it - patch the size field at offset 0x20 in section header
            sh_size_off = sh_off + 0x20
            original_size = struct.unpack("<Q", data[sh_size_off : sh_size_off + 8])[0]

            if new_size > original_size:
                raise ElfPatchError(
                    f"Cannot increase section size: {new_size} > {original_size}"
                )

            struct.pack_into("<Q", data, sh_size_off, new_size)
            hip_fatbin_found = True
            break

    if not hip_fatbin_found:
        raise ElfPatchError(f"No .hip_fatbin section found in: {source_path}")

    # Write patched binary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)

    return output_path


def get_hip_fatbin_size(binary_path: Path) -> Optional[int]:
    """
    Get the size of the .hip_fatbin section in an ELF binary.

    Args:
        binary_path: Path to ELF binary

    Returns:
        Size in bytes, or None if section not found
    """
    if not binary_path.exists():
        return None

    data = binary_path.read_bytes()

    if data[:4] != b"\x7fELF":
        return None

    e_shoff = struct.unpack("<Q", data[0x28:0x30])[0]
    e_shentsize = struct.unpack("<H", data[0x3A:0x3C])[0]
    e_shnum = struct.unpack("<H", data[0x3C:0x3E])[0]
    e_shstrndx = struct.unpack("<H", data[0x3E:0x40])[0]

    shstrtab_hdr_off = e_shoff + e_shstrndx * e_shentsize
    shstrtab_offset = struct.unpack(
        "<Q", data[shstrtab_hdr_off + 0x18 : shstrtab_hdr_off + 0x20]
    )[0]

    for i in range(e_shnum):
        sh_off = e_shoff + i * e_shentsize
        sh_name_idx = struct.unpack("<I", data[sh_off : sh_off + 4])[0]

        name_start = shstrtab_offset + sh_name_idx
        name_end = data.index(0, name_start)
        section_name = data[name_start:name_end].decode("ascii", errors="replace")

        if section_name == ".hip_fatbin":
            return struct.unpack("<Q", data[sh_off + 0x20 : sh_off + 0x28])[0]

    return None
