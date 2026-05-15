"""
Generic kpack transformation API.

This module provides format-agnostic versions of the kpack transformation
functions. They automatically detect whether the input binary is ELF or
PE/COFF and dispatch to the appropriate implementation.

Usage:
    from rocm_kpack import kpack_offload_binary, read_kpack_ref_marker

    # Works with both ELF and PE/COFF binaries
    result = kpack_offload_binary(input_path, output_path, ...)
    marker = read_kpack_ref_marker(binary_path)
"""

from pathlib import Path

from .format_detect import detect_binary_format, UnsupportedBinaryFormat


class NotFatBinaryError(ValueError):
    """Raised when kpack_offload_binary is called on a binary without device code.

    This is not necessarily an error condition - callers should catch this
    and skip processing for non-fat binaries.
    """

    pass


def kpack_offload_binary(
    input_path: Path,
    output_path: Path,
    kpack_search_paths: list[str] | None = None,
    kernel_name: str | None = None,
    verbose: bool = False,
) -> dict:
    """Transform a fat binary for kpack use (format auto-detected).

    This function automatically detects whether the input binary is ELF or
    PE/COFF format and delegates to the appropriate implementation.

    The transformation:
    1. Adds kpack marker section (if params provided)
    2. Updates wrapper pointers to point to marker section
    3. Rewrites HIPF->HIPK magic
    4. Zero-pages the fat binary section

    Args:
        input_path: Path to input binary
        output_path: Path for output binary
        kpack_search_paths: List of kpack file paths relative to binary.
            If None, assumes marker section already exists.
        kernel_name: Kernel identifier for TOC lookup.
            Required if kpack_search_paths is provided.
        verbose: Print progress information

    Returns:
        Dictionary with transformation results:
        - format: "elf" or "coff"
        - wrappers_found: Number of wrappers found and rewritten
        - bytes_zeroed: Number of bytes zeroed in fat binary section
        - original_size: Original file size
        - new_size: New file size

    Raises:
        NotFatBinaryError: If binary doesn't contain device code
        UnsupportedBinaryFormat: If binary is neither ELF nor PE/COFF
    """
    fmt = detect_binary_format(input_path)

    if fmt == "elf":
        from .elf.kpack_transform import kpack_offload_binary as elf_impl

        result = elf_impl(
            input_path,
            output_path,
            kpack_search_paths,
            kernel_name,
            verbose,
        )
        result["format"] = "elf"
        return result
    else:
        from .coff.kpack_transform import kpack_offload_binary as coff_impl

        result = coff_impl(
            input_path,
            output_path,
            kpack_search_paths,
            kernel_name,
            verbose,
        )
        result["format"] = "coff"
        return result


def read_kpack_ref_marker(binary_path: Path) -> dict | None:
    """Read kpack marker section from a binary (format auto-detected).

    This function automatically detects whether the binary is ELF or PE/COFF
    format and reads the marker section using the appropriate implementation.

    Args:
        binary_path: Path to binary with marker section

    Returns:
        Marker data dictionary with keys:
        - kpack_search_paths: list[str] of relative paths to kpack files
        - kernel_name: str kernel identifier for TOC lookup
        Returns None if section doesn't exist.

    Raises:
        RuntimeError: If section exists but cannot be read or parsed
        UnsupportedBinaryFormat: If binary is neither ELF nor PE/COFF
    """
    fmt = detect_binary_format(binary_path)

    if fmt == "elf":
        from .elf.kpack_transform import read_kpack_ref_marker as elf_impl

        return elf_impl(binary_path)
    else:
        from .coff.kpack_transform import read_kpack_ref_marker as coff_impl

        return coff_impl(binary_path)


def is_fat_binary(binary_path: Path) -> bool:
    """Check if a binary contains HIP fat binary sections.

    This is a quick check that doesn't require full parsing.

    Args:
        binary_path: Path to binary

    Returns:
        True if binary appears to contain device code, False otherwise
    """
    try:
        fmt = detect_binary_format(binary_path)
    except UnsupportedBinaryFormat:
        return False

    if fmt == "elf":
        from .elf.surgery import ElfSurgery

        surgery = ElfSurgery.load(binary_path)
        return surgery.find_section(".hip_fatbin") is not None
    else:
        from .coff.surgery import CoffSurgery

        surgery = CoffSurgery.load(binary_path)
        return surgery.find_section(".hip_fat") is not None
