"""
Kpack-specific ELF operations.

This module provides high-level operations for transforming fat binaries
for use with kpack'd device code. It handles:
- Adding .rocm_kpack_ref marker sections
- Rewriting HIPF→HIPK magic bytes
- Verifying relocations
- Full binary transformation pipeline

All operations use ElfSurgery for in-memory manipulation, eliminating
the need for external tools or temporary files.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import msgpack

from ..kpack_transform import NotFatBinaryError
from .surgery import ElfSurgery, SectionInfo, AddSectionResult
from .operations import map_section_to_load, set_pointer
from .verify import ElfVerifier
from .zero_page import conservative_zero_page
from .types import SHT_PROGBITS


# Constants for __CudaFatBinaryWrapper structures
HIPF_MAGIC = 0x48495046  # "HIPF" in little-endian
HIPK_MAGIC = 0x4B504948  # "HIPK" in little-endian
WRAPPER_SIZE = 24  # sizeof(__CudaFatBinaryWrapper)


@dataclass
class ProblematicRelocation:
    """A relocation that still points into the zeroed region."""

    r_offset: int  # Address where relocation applies
    target: int  # Address the relocation points to
    reloc_type: int  # Relocation type


def add_kpack_ref_section(
    surgery: ElfSurgery,
    kpack_search_paths: list[str],
    kernel_name: str,
) -> SectionInfo:
    """Add .rocm_kpack_ref marker section to ELF binary.

    The marker section contains a MessagePack structure pointing to kpack files
    and identifying the kernel name for TOC lookup.

    This replaces binutils.add_kpack_ref_marker() - no toolchain needed.

    Args:
        surgery: ElfSurgery instance to operate on
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

    # Add section using ElfSurgery
    result = surgery.add_section(
        name=".rocm_kpack_ref",
        content=marker_bytes,
        section_type=SHT_PROGBITS,
        flags=0,  # Non-ALLOC initially
        addralign=1,
    )

    # Return section info
    section = surgery.find_section(".rocm_kpack_ref")
    if section is None:
        raise RuntimeError("Failed to add .rocm_kpack_ref section")
    return section


def rewrite_hipfatbin_magic(
    surgery: ElfSurgery,
    verbose: bool = False,
) -> int:
    """Rewrite HIPF→HIPK magic in all __CudaFatBinaryWrapper structures.

    The .hipFatBinSegment section contains an array of __CudaFatBinaryWrapper
    structures. Each wrapper is 24 bytes:
    - offset 0: magic (4 bytes) - HIPF or HIPK
    - offset 4: version (4 bytes)
    - offset 8: binary pointer (8 bytes)
    - offset 16: reserved1 (8 bytes) - used for co_index in kpack'd binaries

    This function:
    1. Rewrites magic from HIPF to HIPK for all wrappers
    2. Zeros the binary pointer (set later by set_pointer)
    3. Writes wrapper index to reserved1 for multi-TU support

    For multi-TU binaries (built with -fgpu-rdc), each wrapper corresponds to
    a different translation unit. The index stored in reserved1 tells CLR
    which code object to request from the kpack archive at runtime.

    Args:
        surgery: ElfSurgery instance to operate on
        verbose: If True, print progress information

    Returns:
        Number of wrappers transformed

    Raises:
        RuntimeError: If wrapper has unexpected magic value
    """
    # Find .hipFatBinSegment section
    segment = surgery.find_section(".hipFatBinSegment")
    if segment is None:
        # No wrappers to transform
        return 0

    segment_vaddr = segment.header.sh_addr
    segment_size = segment.header.sh_size

    # Validate size is multiple of wrapper size
    if segment_size % WRAPPER_SIZE != 0:
        raise RuntimeError(
            f".hipFatBinSegment size {segment_size} is not a "
            f"multiple of wrapper size ({WRAPPER_SIZE})"
        )

    num_wrappers = segment_size // WRAPPER_SIZE

    # Get file offset for the segment
    segment_offset = surgery.vaddr_to_file_offset(segment_vaddr)
    if segment_offset is None:
        raise RuntimeError(
            f".hipFatBinSegment at vaddr 0x{segment_vaddr:x} not in PT_LOAD"
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

        # Transform: HIPF → HIPK
        struct.pack_into("<I", surgery.data, wrapper_offset, HIPK_MAGIC)

        # Zero the binary pointer (offset +8)
        # The actual pointer will be set by set_pointer()
        struct.pack_into("<Q", surgery.data, wrapper_offset + 8, 0)

        # Write wrapper index to reserved1 (offset +16) for multi-TU support
        # CLR uses this as co_index to look up the correct kernel in kpack
        struct.pack_into("<I", surgery.data, wrapper_offset + 16, i)

        transformed += 1
        if verbose:
            print(f"    Wrapper {i}: HIPF → HIPK at offset 0x{wrapper_offset:x}")

    return transformed


def verify_no_fatbin_relocations(
    surgery: ElfSurgery,
    fatbin_vaddr: int,
    fatbin_size: int,
) -> list[ProblematicRelocation]:
    """Verify no relocations point into .hip_fatbin range.

    After redirecting wrapper pointers to .rocm_kpack_ref, there should be
    no relocations that still point into the .hip_fatbin section. This
    function checks all R_X86_64_RELATIVE relocations.

    Args:
        surgery: ElfSurgery instance to operate on
        fatbin_vaddr: Virtual address of .hip_fatbin section
        fatbin_size: Size of .hip_fatbin section

    Returns:
        List of problematic relocations (empty if all is well)
    """
    problematic: list[ProblematicRelocation] = []

    # Iterate all RELA sections
    for section in surgery.iter_rela_sections():
        for _, rela in surgery.iter_relocations(section):
            in_range = rela.targets_range(fatbin_vaddr, fatbin_size)
            if in_range is True:
                target = rela.get_target_address()
                assert target is not None  # targets_range returned True
                problematic.append(
                    ProblematicRelocation(
                        r_offset=rela.r_offset,
                        target=target,
                        reloc_type=rela.r_type,
                    )
                )
            # in_range is None means unknown relocation type - we ignore those
            # since we only care about relocations we understand

    return problematic


def kpack_offload_binary(
    input_path: Path,
    output_path: Path,
    kpack_search_paths: list[str] | None = None,
    kernel_name: str | None = None,
    verbose: bool = False,
) -> dict:
    """Transform an ELF fat binary for kpack use.

    This function transforms a fat binary by:
    1. Adding .rocm_kpack_ref section (if params provided)
    2. Mapping .rocm_kpack_ref to new PT_LOAD segment
    3. Updating wrapper pointers to point to .rocm_kpack_ref
    4. Rewriting HIPF→HIPK magic
    5. Zero-paging .hip_fatbin section

    Args:
        input_path: Path to input binary
        output_path: Path for output binary
        kpack_search_paths: List of kpack file paths relative to binary.
            If None, assumes .rocm_kpack_ref section already exists.
        kernel_name: Kernel identifier for TOC lookup.
            Required if kpack_search_paths is provided.
        verbose: If True, print detailed progress information

    Returns:
        Dictionary with statistics:
        - removed: Bytes saved (original_size - new_size)
        - original_size: Original file size
        - new_size: Final file size
        - kpack_ref_vaddr: Virtual address of .rocm_kpack_ref section

    Raises:
        RuntimeError: If transformation fails
    """
    original_size = input_path.stat().st_size
    original_mode = input_path.stat().st_mode

    # Load binary
    surgery = ElfSurgery.load(input_path)

    # Check if binary has .hip_fatbin section BEFORE any modifications
    fatbin = surgery.find_section(".hip_fatbin")
    has_fatbin = fatbin is not None

    # If no .hip_fatbin and caller wants to add marker, signal this clearly
    # Callers should catch NotFatBinaryError and skip processing
    if not has_fatbin and kpack_search_paths is not None:
        raise NotFatBinaryError(
            f"Binary {input_path} has no .hip_fatbin section. "
            "Cannot add kpack marker to binary without device code."
        )

    # Phase 0: Add .rocm_kpack_ref section if needed
    if kpack_search_paths is not None:
        if kernel_name is None:
            raise ValueError("kernel_name required when kpack_search_paths provided")

        if verbose:
            print("\nPhase 0: Add .rocm_kpack_ref section")

        add_kpack_ref_section(surgery, kpack_search_paths, kernel_name)

    # Phase 1: Map .rocm_kpack_ref to new PT_LOAD
    if verbose:
        print("\nPhase 1: Map .rocm_kpack_ref to PT_LOAD")

    map_result = map_section_to_load(surgery, ".rocm_kpack_ref")
    if not map_result.success:
        raise RuntimeError(f"Failed to map .rocm_kpack_ref: {map_result.error}")

    kpack_ref_vaddr = map_result.vaddr
    if verbose:
        print(f"  .rocm_kpack_ref mapped to: 0x{kpack_ref_vaddr:x}")

    if has_fatbin:
        # Phase 2: Update wrapper pointers and rewrite magic
        if verbose:
            print("\nPhase 2: Semantic transformation (pointer + magic)")

        # Find .hipFatBinSegment section
        segment = surgery.find_section(".hipFatBinSegment")
        if segment is None:
            raise RuntimeError(".hipFatBinSegment section not found")

        segment_vaddr = segment.header.sh_addr
        segment_size = segment.header.sh_size

        if segment_size % WRAPPER_SIZE != 0:
            raise RuntimeError(
                f".hipFatBinSegment size {segment_size} is not a "
                f"multiple of wrapper size ({WRAPPER_SIZE})"
            )

        num_wrappers = segment_size // WRAPPER_SIZE
        if verbose:
            print(f"  Found {num_wrappers} wrapper(s) in .hipFatBinSegment")

        # Update each wrapper's pointer to point to .rocm_kpack_ref
        for i in range(num_wrappers):
            wrapper_vaddr = segment_vaddr + i * WRAPPER_SIZE
            pointer_vaddr = wrapper_vaddr + 8  # Pointer at offset +8

            if verbose:
                print(f"  Updating wrapper {i} pointer at 0x{pointer_vaddr:x}")

            result = set_pointer(surgery, pointer_vaddr, kpack_ref_vaddr)
            if not result.success:
                raise RuntimeError(
                    f"Failed to set pointer for wrapper {i}: {result.error}"
                )

        # Rewrite magic HIPF→HIPK for all wrappers
        if verbose:
            print(f"  Rewriting magic (HIPF → HIPK) for {num_wrappers} wrapper(s)")

        rewrite_hipfatbin_magic(surgery, verbose=verbose)

        # Verify no relocations still point into .hip_fatbin
        fatbin_vaddr = fatbin.header.sh_addr
        fatbin_size = fatbin.header.sh_size
        problematic = verify_no_fatbin_relocations(surgery, fatbin_vaddr, fatbin_size)

        if problematic:
            error_lines = [
                f"ERROR: {len(problematic)} relocation(s) still point "
                f"into .hip_fatbin section:",
            ]
            for p in problematic:
                type_name = (
                    "R_X86_64_RELATIVE" if p.reloc_type == 8 else f"type={p.reloc_type}"
                )
                error_lines.append(
                    f"  - offset 0x{p.r_offset:x} -> 0x{p.target:x} ({type_name})"
                )
            error_lines.append(
                "\nThis indicates a bug: not all __CudaFatBinaryWrapper pointers "
                "were redirected to .rocm_kpack_ref."
            )
            raise RuntimeError("\n".join(error_lines))

        # Phase 3: Zero-page .hip_fatbin (optimization only)
        if verbose:
            print("\nPhase 3: Zero-page .hip_fatbin (optimization)")

        zero_result = conservative_zero_page(surgery, ".hip_fatbin")
        if not zero_result.success:
            raise RuntimeError(f"Zero-page optimization failed: {zero_result.error}")

        if verbose:
            print(f"  Pages zeroed: {zero_result.pages_zeroed}")
            print(f"  Bytes saved: {zero_result.bytes_saved:,}")

    else:
        # No .hip_fatbin, skip pointer update, magic rewrite, and zero-page
        if verbose:
            print(
                "\nPhases 2-3: No .hip_fatbin section, skipping semantic "
                "transformation and zero-page"
            )

    # Post-transform verification: catch structural corruption at build time
    verify_result = ElfVerifier.verify_data(surgery.data)
    if not verify_result.passed:
        raise RuntimeError(
            f"Post-transform ELF verification failed for {input_path}:\n"
            + "\n".join(f"  - {e}" for e in verify_result.errors)
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
        "kpack_ref_vaddr": kpack_ref_vaddr,
    }


def read_kpack_ref_marker(
    binary_path: Path,
) -> dict | None:
    """Read .rocm_kpack_ref marker section from a binary.

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
    surgery = ElfSurgery.load(binary_path)
    section = surgery.find_section(".rocm_kpack_ref")

    if section is None:
        return None

    try:
        content = surgery.get_section_content(section)
        marker_data = msgpack.unpackb(content, raw=False)
        return marker_data
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse .rocm_kpack_ref marker data from {binary_path}: {e}"
        )
