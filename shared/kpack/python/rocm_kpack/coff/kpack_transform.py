"""
Kpack-specific PE/COFF operations.

This module provides high-level operations for transforming fat binaries
for use with kpack'd device code on Windows. It handles:
- Adding .kpackrf marker sections
- Rewriting HIPF->HIPK magic bytes
- Updating wrapper pointers
- Zero-paging .hip_fat
- Full binary transformation pipeline

All operations use CoffSurgery for in-memory manipulation, eliminating
the need for external tools or temporary files.

Key differences from ELF:
- Section names limited to 8 chars (.rocm_kpack_ref -> .kpackrf)
- .hipFatBinSegment -> .hipFatB
- .hip_fatbin -> .hip_fat
- No map_section_to_load needed (PE sections have RVAs directly)
- Base relocations work automatically (just update pointer value)
"""

import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import msgpack

logger = logging.getLogger(__name__)

from ..kpack_transform import NotFatBinaryError
from .surgery import CoffSurgery, SectionInfo, AddSectionResult
from .zero_page import conservative_zero_page
from .types import IMAGE_SCN_MEM_READ, IMAGE_SCN_CNT_INITIALIZED_DATA


# Constants for __CudaFatBinaryWrapper structures
# (identical to ELF - the wrapper format is the same)
# Note: These are 32-bit integers as stored in the binary. When read as
# little-endian, 0x48495046 appears as bytes [0x46, 0x50, 0x49, 0x48].
HIPF_MAGIC = 0x48495046  # Original fat binary wrapper magic
HIPK_MAGIC = 0x4B504948  # Kpack-transformed wrapper magic
WRAPPER_SIZE = 24  # sizeof(__CudaFatBinaryWrapper)

# PE section names (8-char limit)
SECTION_HIP_FATBIN_SEGMENT = ".hipFatB"  # Wrappers
SECTION_HIP_FATBIN = ".hip_fat"  # Fat binary content
SECTION_KPACK_REF = ".kpackrf"  # Kpack marker (was .rocm_kpack_ref)


@dataclass
class ProblematicRelocation:
    """A base relocation that still points into the zeroed region."""

    target_rva: int  # RVA the relocation targets
    reloc_type: int  # Relocation type


def add_kpack_ref_section(
    surgery: CoffSurgery,
    kpack_search_paths: list[str],
    kernel_name: str,
) -> SectionInfo:
    """Add .kpackrf marker section to PE binary.

    The marker section contains a MessagePack structure pointing to kpack files
    and identifying the kernel name for TOC lookup.

    Args:
        surgery: CoffSurgery instance to operate on
        kpack_search_paths: List of kpack file paths relative to binary location
        kernel_name: Kernel identifier for TOC lookup in kpack file

    Returns:
        SectionInfo for the newly added section

    Raises:
        ValueError: If section already exists
    """
    # Create marker structure
    marker_data = {
        "kpack_search_paths": kpack_search_paths,
        "kernel_name": kernel_name,
    }

    # Serialize to MessagePack
    marker_bytes = msgpack.packb(marker_data, use_bin_type=True)

    # Add section using CoffSurgery
    result = surgery.add_section(
        name=SECTION_KPACK_REF,
        content=marker_bytes,
        characteristics=IMAGE_SCN_MEM_READ | IMAGE_SCN_CNT_INITIALIZED_DATA,
    )
    if not result.success:
        raise RuntimeError(f"Failed to add {SECTION_KPACK_REF} section: {result.error}")

    # Return section info
    section = surgery.find_section(SECTION_KPACK_REF)
    if section is None:
        raise RuntimeError(
            f"Internal error: {SECTION_KPACK_REF} section not found after add_section"
        )
    return section


def rewrite_hipfatbin_magic(
    surgery: CoffSurgery,
    verbose: bool = False,
) -> int:
    """Rewrite HIPF->HIPK magic in all __CudaFatBinaryWrapper structures.

    The .hipFatB section contains an array of __CudaFatBinaryWrapper
    structures. Each wrapper is 24 bytes:
    - offset 0: magic (4 bytes) - HIPF or HIPK
    - offset 4: version (4 bytes)
    - offset 8: binary pointer (8 bytes)
    - offset 16: reserved1 (8 bytes) - used for co_index in kpack'd binaries

    This function:
    1. Rewrites magic from HIPF to HIPK for all wrappers
    2. Writes wrapper index to reserved1 for multi-TU support

    For multi-TU binaries (built with -fgpu-rdc), each wrapper corresponds to
    a different translation unit. The index stored in reserved1 tells CLR
    which code object to request from the kpack archive at runtime.

    Args:
        surgery: CoffSurgery instance to operate on
        verbose: If True, print progress information

    Returns:
        Number of wrappers transformed

    Raises:
        RuntimeError: If wrapper has unexpected magic value
    """
    # Find .hipFatB section
    segment = surgery.find_section(SECTION_HIP_FATBIN_SEGMENT)
    if segment is None:
        # No wrappers to transform
        return 0

    segment_rva = segment.rva
    segment_size = segment.virtual_size
    segment_raw_size = segment.raw_size

    # Validate size is multiple of wrapper size
    if segment_size % WRAPPER_SIZE != 0:
        raise RuntimeError(
            f"{SECTION_HIP_FATBIN_SEGMENT} size {segment_size} is not a "
            f"multiple of wrapper size ({WRAPPER_SIZE})"
        )

    # Validate we have enough raw data to read the wrappers
    if segment_raw_size < segment_size:
        raise RuntimeError(
            f"{SECTION_HIP_FATBIN_SEGMENT} has insufficient raw data: "
            f"VirtualSize={segment_size}, SizeOfRawData={segment_raw_size}"
        )

    num_wrappers = segment_size // WRAPPER_SIZE

    # Get file offset for the segment
    segment_offset = surgery.rva_to_file_offset(segment_rva)
    if segment_offset is None:
        raise RuntimeError(
            f"{SECTION_HIP_FATBIN_SEGMENT} at RVA 0x{segment_rva:x} not in file"
        )

    transformed = 0
    for i in range(num_wrappers):
        wrapper_offset = segment_offset + i * WRAPPER_SIZE

        # Read current magic
        current_magic = struct.unpack_from("<I", surgery.data, wrapper_offset)[0]

        if current_magic == HIPK_MAGIC:
            if verbose:
                print(f"    Wrapper {i}: already HIPK (skipped)")
            continue

        if current_magic != HIPF_MAGIC:
            raise RuntimeError(
                f"Unexpected magic 0x{current_magic:08x} at wrapper {i} "
                f"(offset 0x{wrapper_offset:x}). Expected HIPF (0x{HIPF_MAGIC:08x})."
            )

        # Transform: HIPF -> HIPK
        struct.pack_into("<I", surgery.data, wrapper_offset, HIPK_MAGIC)

        # Write wrapper index to reserved1 (offset +16) for multi-TU support
        # CLR uses this as co_index to look up the correct kernel in kpack
        struct.pack_into("<I", surgery.data, wrapper_offset + 16, i)

        transformed += 1
        if verbose:
            print(f"    Wrapper {i}: HIPF -> HIPK at offset 0x{wrapper_offset:x}")

    return transformed


def update_wrapper_pointers(
    surgery: CoffSurgery,
    target_rva: int,
    verbose: bool = False,
) -> int:
    """Update all wrapper pointers to point to the kpack ref section.

    Args:
        surgery: CoffSurgery instance to operate on
        target_rva: RVA of the .kpackrf section
        verbose: If True, print progress information

    Returns:
        Number of wrappers updated

    Raises:
        RuntimeError: If wrappers not found or invalid
    """
    # Find .hipFatB section
    segment = surgery.find_section(SECTION_HIP_FATBIN_SEGMENT)
    if segment is None:
        return 0

    segment_rva = segment.rva
    segment_size = segment.virtual_size
    num_wrappers = segment_size // WRAPPER_SIZE

    # Convert target RVA to VA (what the pointer stores)
    target_va = surgery.rva_to_va(target_rva)

    updated = 0
    for i in range(num_wrappers):
        wrapper_rva = segment_rva + i * WRAPPER_SIZE
        pointer_rva = wrapper_rva + 8  # Pointer at offset +8

        if verbose:
            print(
                f"    Wrapper {i}: pointer at RVA 0x{pointer_rva:x} -> 0x{target_va:x}"
            )

        # Write new pointer value
        surgery.write_pointer_at_rva(
            pointer_rva,
            target_va,
            f"wrapper {i} binary pointer -> kpack ref",
        )
        updated += 1

    return updated


def verify_no_fatbin_relocations(
    surgery: CoffSurgery,
    fatbin_rva: int,
    fatbin_size: int,
) -> list[ProblematicRelocation]:
    """Verify no base relocations point into .hip_fat range.

    After redirecting wrapper pointers to .kpackrf, there should be
    no relocations that still point into the .hip_fat section.

    Args:
        surgery: CoffSurgery instance to operate on
        fatbin_rva: RVA of .hip_fat section
        fatbin_size: Size of .hip_fat section

    Returns:
        List of problematic relocations (empty if all is well)
    """
    problematic: list[ProblematicRelocation] = []

    fatbin_end = fatbin_rva + fatbin_size

    # Check all base relocations
    for reloc in surgery.iter_base_relocations():
        # Read the pointer value at the relocation target
        try:
            ptr_value = surgery.read_pointer_at_rva(reloc.target_rva)
            ptr_rva = surgery.va_to_rva(ptr_value)

            # Check if pointer points into fatbin range
            if fatbin_rva <= ptr_rva < fatbin_end:
                problematic.append(
                    ProblematicRelocation(
                        target_rva=reloc.target_rva,
                        reloc_type=reloc.entry.reloc_type,
                    )
                )
        except (ValueError, struct.error) as e:
            # Log warning for relocations we can't read - may indicate corruption
            logger.debug(
                "Could not read pointer at relocation target RVA 0x%x: %s",
                reloc.target_rva,
                e,
            )

    return problematic


def kpack_offload_binary(
    input_path: Path,
    output_path: Path,
    kpack_search_paths: list[str] | None = None,
    kernel_name: str | None = None,
    verbose: bool = False,
) -> dict:
    """Transform a PE fat binary for kpack use.

    This function transforms a fat binary by:
    1. Adding .kpackrf section (if params provided)
    2. Updating wrapper pointers to point to .kpackrf
    3. Rewriting HIPF->HIPK magic
    4. Zero-paging .hip_fat section

    Args:
        input_path: Path to input binary
        output_path: Path for output binary
        kpack_search_paths: List of kpack file paths relative to binary.
            If None, assumes .kpackrf section already exists.
        kernel_name: Kernel identifier for TOC lookup.
            Required if kpack_search_paths is provided.
        verbose: If True, print detailed progress information

    Returns:
        Dictionary with statistics:
        - removed: Bytes saved (original_size - new_size)
        - original_size: Original file size
        - new_size: Final file size
        - kpack_ref_rva: RVA of .kpackrf section

    Raises:
        RuntimeError: If transformation fails
    """
    original_size = input_path.stat().st_size
    original_mode = input_path.stat().st_mode

    # Load binary
    surgery = CoffSurgery.load(input_path)

    # Check if binary has .hip_fat section BEFORE any modifications
    fatbin = surgery.find_section(SECTION_HIP_FATBIN)
    has_fatbin = fatbin is not None

    # If no .hip_fat and caller wants to add marker, signal this clearly
    if not has_fatbin and kpack_search_paths is not None:
        raise NotFatBinaryError(
            f"Binary {input_path} has no {SECTION_HIP_FATBIN} section. "
            "Cannot add kpack marker to binary without device code."
        )

    # Phase 0: Add .kpackrf section if needed
    if kpack_search_paths is not None:
        if kernel_name is None:
            raise ValueError("kernel_name required when kpack_search_paths provided")

        if verbose:
            print(f"\nPhase 0: Add {SECTION_KPACK_REF} section")

        kpack_section = add_kpack_ref_section(surgery, kpack_search_paths, kernel_name)
        kpack_ref_rva = kpack_section.rva

        if verbose:
            print(f"  Added {SECTION_KPACK_REF} at RVA 0x{kpack_ref_rva:x}")
    else:
        # Find existing .kpackrf section
        kpack_section = surgery.find_section(SECTION_KPACK_REF)
        if kpack_section is None:
            raise RuntimeError(f"{SECTION_KPACK_REF} section not found")
        kpack_ref_rva = kpack_section.rva

    if has_fatbin:
        # Phase 1: Update wrapper pointers
        if verbose:
            print("\nPhase 1: Update wrapper pointers")

        num_updated = update_wrapper_pointers(surgery, kpack_ref_rva, verbose=verbose)
        if verbose:
            print(f"  Updated {num_updated} wrapper pointer(s)")

        # Phase 2: Rewrite HIPF->HIPK magic
        if verbose:
            print("\nPhase 2: Rewrite magic (HIPF -> HIPK)")

        num_transformed = rewrite_hipfatbin_magic(surgery, verbose=verbose)
        if verbose:
            print(f"  Transformed {num_transformed} wrapper(s)")

        # Verify no relocations still point into .hip_fat
        fatbin_rva = fatbin.rva
        fatbin_size = fatbin.virtual_size
        problematic = verify_no_fatbin_relocations(surgery, fatbin_rva, fatbin_size)

        if problematic:
            error_lines = [
                f"ERROR: {len(problematic)} relocation(s) still point "
                f"into {SECTION_HIP_FATBIN} section:",
            ]
            for p in problematic:
                error_lines.append(f"  - RVA 0x{p.target_rva:x} (type={p.reloc_type})")
            error_lines.append(
                f"\nThis indicates a bug: not all __CudaFatBinaryWrapper pointers "
                f"were redirected to {SECTION_KPACK_REF}."
            )
            raise RuntimeError("\n".join(error_lines))

        # Phase 3: Zero-page .hip_fat
        if verbose:
            print(f"\nPhase 3: Zero-page {SECTION_HIP_FATBIN}")

        zero_result = conservative_zero_page(surgery, SECTION_HIP_FATBIN)
        if not zero_result.success:
            raise RuntimeError(f"Zero-page optimization failed: {zero_result.error}")

        if verbose:
            print(f"  Pages zeroed: {zero_result.pages_zeroed}")
            print(f"  Bytes saved: {zero_result.bytes_saved:,}")

    else:
        # No .hip_fat, skip pointer update, magic rewrite, and zero-page
        if verbose:
            print(
                f"\nPhases 1-3: No {SECTION_HIP_FATBIN} section, skipping semantic "
                "transformation and zero-page"
            )

    # Write final output
    surgery.save_preserving_mode(output_path, original_mode)

    final_size = len(surgery.data)
    removed = original_size - final_size

    if verbose:
        print("\nTransformation complete:")
        print(f"  Original size: {original_size:,} bytes")
        print(f"  Final size:    {final_size:,} bytes")
        print(
            f"  Removed:       {removed:,} bytes ({100 * removed / original_size:.1f}%)"
        )

    return {
        "removed": removed,
        "original_size": original_size,
        "new_size": final_size,
        "kpack_ref_rva": kpack_ref_rva,
    }


def read_kpack_ref_marker(
    binary_path: Path,
) -> dict | None:
    """Read .kpackrf marker section from a binary.

    Args:
        binary_path: Path to binary with marker section

    Returns:
        Marker data dictionary with keys:
        - kpack_search_paths: list[str] of relative paths to kpack files
        - kernel_name: str kernel identifier for TOC lookup
        Returns None if section doesn't exist.

    Raises:
        RuntimeError: If section exists but cannot be read or parsed
    """
    surgery = CoffSurgery.load(binary_path)
    section = surgery.find_section(SECTION_KPACK_REF)

    if section is None:
        return None

    try:
        # Read only VirtualSize bytes (actual content), not padded SizeOfRawData
        content = surgery.get_section_content(section)
        actual_size = section.virtual_size
        content = content[:actual_size]

        marker_data = msgpack.unpackb(content, raw=False, strict_map_key=True)
        if not isinstance(marker_data, dict):
            raise RuntimeError(
                f"Invalid {SECTION_KPACK_REF} marker format: expected dict, "
                f"got {type(marker_data).__name__}"
            )
        return marker_data
    except msgpack.exceptions.UnpackException as e:
        raise RuntimeError(
            f"Failed to parse {SECTION_KPACK_REF} marker data from {binary_path}: {e}"
        ) from e
    except ValueError as e:
        raise RuntimeError(
            f"Failed to read {SECTION_KPACK_REF} section from {binary_path}: {e}"
        ) from e
