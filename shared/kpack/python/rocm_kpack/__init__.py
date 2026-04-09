"""
rocm-kpack: Binary transformation tools for ROCm kernel packaging.

This package provides utilities for transforming HIP fat binaries to work
with externally packaged GPU kernels (kpack files). It supports both ELF
(Linux) and PE/COFF (Windows) binary formats.

The generic API automatically detects binary format and dispatches to the
appropriate implementation:

    from rocm_kpack import kpack_offload_binary, read_kpack_ref_marker

    # Transform any fat binary (ELF or PE/COFF)
    result = kpack_offload_binary(input_path, output_path, ...)

    # Read marker from transformed binary
    marker = read_kpack_ref_marker(binary_path)

For format-specific operations, use the subpackages directly:

    from rocm_kpack.elf import ElfSurgery, kpack_offload_binary
    from rocm_kpack.coff import CoffSurgery, kpack_offload_binary
"""

from .format_detect import (
    detect_binary_format,
    is_elf_binary,
    is_pe_binary,
    UnsupportedBinaryFormat,
)
from .kpack_transform import (
    kpack_offload_binary,
    read_kpack_ref_marker,
    is_fat_binary,
    NotFatBinaryError,
)

__all__ = [
    # Format detection
    "detect_binary_format",
    "is_elf_binary",
    "is_pe_binary",
    "UnsupportedBinaryFormat",
    # Generic kpack transform
    "kpack_offload_binary",
    "read_kpack_ref_marker",
    "is_fat_binary",
    "NotFatBinaryError",
]
