"""
Zero-page optimization for PE/COFF binaries.

This module handles the removal of content from sections while maintaining
PE validity. The key operation is "zero-paging" - reducing SizeOfRawData
so that VirtualSize > SizeOfRawData, causing the loader to zero-fill.

The PE zero-page algorithm (simpler than ELF!):
1. Find page-aligned region within the section
2. Remove that content from the file
3. Update section's SizeOfRawData to remaining content size
4. Adjust PointerToRawData for all subsequent sections
5. VirtualSize and SizeOfImage stay unchanged (loader zero-fills)

Key advantages over ELF:
- No program header splitting (PE sections define their own virtual layout)
- PE naturally supports VirtualSize > SizeOfRawData
- No NOBITS section type needed - just reduce raw data
"""

from dataclasses import dataclass
from pathlib import Path

from .types import (
    SectionHeader,
    PAGE_SIZE,
    round_up_to_page,
    round_down_to_page,
    round_up_to_alignment,
)
from .surgery import CoffSurgery, SectionInfo


@dataclass
class ZeroPageResult:
    """Result of zero-page optimization."""

    success: bool
    bytes_saved: int
    pages_zeroed: int
    error: str | None = None


def calculate_aligned_range(rva: int, size: int) -> tuple[int, int]:
    """Calculate the page-aligned range within a section.

    Args:
        rva: Section RVA (relative virtual address)
        size: Section size

    Returns:
        Tuple of (aligned_rva, aligned_size) - the range of full pages
        that can be zeroed. Returns (0, 0) if no full pages exist.
    """
    end = rva + size

    # Round up start to next page boundary
    aligned_start = round_up_to_page(rva)

    # Round down end to previous page boundary
    aligned_end = round_down_to_page(end)

    if aligned_start >= aligned_end:
        # No full pages to zero
        return (0, 0)

    return (aligned_start, aligned_end - aligned_start)


def conservative_zero_page(
    surgery: CoffSurgery,
    section_name: str,
) -> ZeroPageResult:
    """Apply conservative zero-page optimization to a section.

    This removes page-aligned content from the file while maintaining
    PE validity. Only full pages are removed; partial pages are preserved.

    Args:
        surgery: CoffSurgery instance to operate on
        section_name: Section to zero-page

    Returns:
        ZeroPageResult with operation outcome
    """
    # Find the target section
    section = surgery.find_section(section_name)
    if section is None:
        return ZeroPageResult(
            success=False,
            bytes_saved=0,
            pages_zeroed=0,
            error=f"Section '{section_name}' not found",
        )

    # Check if already zero-paged (VirtualSize > SizeOfRawData significantly)
    if section.raw_size == 0:
        return ZeroPageResult(
            success=True,
            bytes_saved=0,
            pages_zeroed=0,
            error=f"Section '{section_name}' already has zero raw data",
        )

    section_rva = section.rva
    section_vsize = section.virtual_size
    section_offset = section.file_offset

    # Calculate aligned range
    aligned_rva, aligned_size = calculate_aligned_range(section_rva, section_vsize)

    if aligned_size == 0:
        # Section is too small or misaligned - no optimization possible
        return ZeroPageResult(
            success=True,
            bytes_saved=0,
            pages_zeroed=0,
            error="Section too small or misaligned - no full pages to zero",
        )

    # Calculate file offsets
    offset_within_section = aligned_rva - section_rva
    aligned_file_offset = section_offset + offset_within_section
    pages_to_zero = aligned_size // PAGE_SIZE

    # Calculate prefix and suffix sizes
    prefix_size = offset_within_section  # Bytes before aligned region
    suffix_size = section_vsize - offset_within_section - aligned_size

    # Defensive check: suffix should never be negative given our calculations
    assert suffix_size >= 0, (
        f"Internal error: negative suffix_size={suffix_size} "
        f"(vsize={section_vsize}, offset={offset_within_section}, aligned={aligned_size})"
    )

    # Remove the aligned bytes from file
    old_size = len(surgery.data)
    del surgery.data[aligned_file_offset : aligned_file_offset + aligned_size]
    bytes_removed = old_size - len(surgery.data)

    # Update the target section header
    # New SizeOfRawData = prefix + suffix, rounded to FileAlignment
    new_raw_content_size = prefix_size + suffix_size
    new_raw_size = round_up_to_alignment(new_raw_content_size, surgery.file_alignment)

    # If new_raw_size is 0, we need to handle edge case
    if new_raw_content_size == 0:
        new_raw_size = 0

    old_shdr = section.header
    new_shdr = SectionHeader(
        Name=old_shdr.Name,
        VirtualSize=old_shdr.VirtualSize,  # Unchanged - virtual layout same
        VirtualAddress=old_shdr.VirtualAddress,  # Unchanged
        SizeOfRawData=new_raw_size,  # Reduced
        PointerToRawData=old_shdr.PointerToRawData,  # Same start offset
        PointerToRelocations=old_shdr.PointerToRelocations,
        PointerToLinenumbers=old_shdr.PointerToLinenumbers,
        NumberOfRelocations=old_shdr.NumberOfRelocations,
        NumberOfLinenumbers=old_shdr.NumberOfLinenumbers,
        Characteristics=old_shdr.Characteristics,
    )
    surgery.update_section_header(section.index, new_shdr)

    # Update subsequent sections' PointerToRawData
    for sect in surgery.iter_sections():
        if sect.file_offset > section_offset:
            old_offset = sect.file_offset
            new_offset = old_offset - aligned_size

            old_shdr = sect.header
            new_shdr = SectionHeader(
                Name=old_shdr.Name,
                VirtualSize=old_shdr.VirtualSize,
                VirtualAddress=old_shdr.VirtualAddress,
                SizeOfRawData=old_shdr.SizeOfRawData,
                PointerToRawData=new_offset,  # Shifted down
                PointerToRelocations=old_shdr.PointerToRelocations,
                PointerToLinenumbers=old_shdr.PointerToLinenumbers,
                NumberOfRelocations=old_shdr.NumberOfRelocations,
                NumberOfLinenumbers=old_shdr.NumberOfLinenumbers,
                Characteristics=old_shdr.Characteristics,
            )
            surgery.update_section_header(sect.index, new_shdr)

    # SizeOfImage stays the same - virtual layout unchanged
    # No need to update optional header

    return ZeroPageResult(
        success=True,
        bytes_saved=bytes_removed,
        pages_zeroed=pages_to_zero,
    )


def zero_page_section(
    input_path: Path,
    output_path: Path,
    section_name: str,
) -> ZeroPageResult:
    """Apply zero-page optimization to a file.

    Convenience function that loads, modifies, and saves a PE binary.

    Args:
        input_path: Input PE binary
        output_path: Output PE binary
        section_name: Section to zero-page

    Returns:
        ZeroPageResult with operation outcome
    """
    surgery = CoffSurgery.load(input_path)
    result = conservative_zero_page(surgery, section_name)

    if result.success:
        surgery.save_preserving_mode(output_path, input_path.stat().st_mode)

    return result
