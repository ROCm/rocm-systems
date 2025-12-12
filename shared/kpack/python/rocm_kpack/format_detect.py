"""
Binary format detection utilities.

This module provides functions to detect whether a binary file is ELF or PE/COFF
format, enabling the generic API to dispatch to the correct implementation.
"""

from pathlib import Path


# Magic bytes for format detection
ELF_MAGIC = b"\x7fELF"
DOS_MAGIC = b"MZ"
PE_SIGNATURE = b"PE\x00\x00"


class UnsupportedBinaryFormat(ValueError):
    """Raised when a binary is not ELF or PE/COFF format."""

    pass


def detect_binary_format(path: Path) -> str:
    """Detect whether a binary is ELF or PE/COFF format.

    Args:
        path: Path to binary file

    Returns:
        "elf" for ELF binaries, "coff" for PE/COFF binaries

    Raises:
        UnsupportedBinaryFormat: If the binary is neither ELF nor PE/COFF
        FileNotFoundError: If the file doesn't exist
    """
    with open(path, "rb") as f:
        # Read enough for both checks
        header = f.read(64)

    if len(header) < 4:
        raise UnsupportedBinaryFormat(f"File too small to be a valid binary: {path}")

    # Check for ELF magic
    if header[:4] == ELF_MAGIC:
        return "elf"

    # Check for PE/COFF (DOS header with PE signature)
    if header[:2] == DOS_MAGIC:
        # Read PE header offset from DOS header at offset 0x3c
        if len(header) >= 0x40:
            pe_offset = int.from_bytes(header[0x3C:0x40], "little")
            # Validate PE offset is reasonable (within first 1MB, aligned)
            if pe_offset < 0x40 or pe_offset > 0x100000:
                raise UnsupportedBinaryFormat(
                    f"Invalid PE header offset {pe_offset:#x}: {path}"
                )
            # Read PE signature
            with open(path, "rb") as f:
                f.seek(pe_offset)
                pe_sig = f.read(4)
            if pe_sig == PE_SIGNATURE:
                return "coff"

    raise UnsupportedBinaryFormat(f"Binary is neither ELF nor PE/COFF format: {path}")


def is_elf_binary(path: Path) -> bool:
    """Check if a binary is ELF format.

    Args:
        path: Path to binary file

    Returns:
        True if ELF, False otherwise
    """
    try:
        return detect_binary_format(path) == "elf"
    except (UnsupportedBinaryFormat, FileNotFoundError):
        return False


def is_pe_binary(path: Path) -> bool:
    """Check if a binary is PE/COFF format.

    Args:
        path: Path to binary file

    Returns:
        True if PE/COFF, False otherwise
    """
    try:
        return detect_binary_format(path) == "coff"
    except (UnsupportedBinaryFormat, FileNotFoundError):
        return False
