"""
Zero-page optimization for ELF binaries.

This module handles the removal of content from sections while maintaining
ELF validity. The key operation is "zero-paging" - converting file-backed
content to memory-only (NOBITS) so the file can be smaller.

The conservative zero-page algorithm:
1. Find page-aligned region within the section
2. Remove that content from the file
3. Split the containing PT_LOAD into multiple segments
4. Adjust all offsets for the removed bytes
5. Mark the section as NOBITS

Key invariants:
- Only full pages can be removed (partial pages must be preserved)
- mmap alignment must be maintained: (p_offset % PAGE_SIZE) == (p_vaddr % PAGE_SIZE)
- Section headers must reflect the new offsets
"""

from dataclasses import dataclass
from pathlib import Path

from .types import (
    ProgramHeader,
    SectionHeader,
    PAGE_SIZE,
    PT_LOAD,
    SHT_NOBITS,
    round_up_to_page,
    round_down_to_page,
)
from .surgery import ElfSurgery, SectionInfo
from .phdr_manager import ProgramHeaderManager


@dataclass
class ZeroPageResult:
    """Result of zero-page optimization."""

    success: bool
    bytes_saved: int
    phdr_relocated: bool
    pages_zeroed: int
    error: str | None = None


def calculate_aligned_range(vaddr: int, size: int) -> tuple[int, int]:
    """Calculate the page-aligned range within a section.

    Args:
        vaddr: Section virtual address
        size: Section size

    Returns:
        Tuple of (aligned_vaddr, aligned_size) - the range of full pages
        that can be zeroed. Returns (0, 0) if no full pages exist.
    """
    end = vaddr + size

    # Round up start to next page boundary
    aligned_start = round_up_to_page(vaddr)

    # Round down end to previous page boundary
    aligned_end = round_down_to_page(end)

    if aligned_start >= aligned_end:
        # No full pages to zero
        return (0, 0)

    return (aligned_start, aligned_end - aligned_start)


@dataclass
class SegmentPiece:
    """A piece of a split PT_LOAD segment."""

    vaddr: int
    memsz: int
    offset: int
    filesz: int

    @property
    def is_nobits(self) -> bool:
        """Check if this is a NOBITS (file-less) segment."""
        return self.filesz == 0 and self.memsz > 0


def split_load_segment(
    original: ProgramHeader,
    section_vaddr: int,
    section_size: int,
    aligned_vaddr: int,
    aligned_size: int,
    aligned_offset: int,
) -> list[SegmentPiece]:
    """Split a PT_LOAD segment around a zeroed region.

    The original segment is split into up to 4 pieces:
    1. Content before section (if any)
    2. Section prefix (if unaligned start)
    3. NOBITS region covering aligned portion AND any unaligned suffix
       (We extend NOBITS to cover suffix to avoid offset collision)
    4. Content after section (if any)

    Note: We do NOT create a separate suffix segment because that would
    cause a NOBITS collision - both the NOBITS and suffix segments would
    have the same file offset after content removal. Instead, we extend
    the NOBITS memsz to cover the suffix. The suffix content is still
    present in the file (at aligned_offset after removal), but it's not
    mapped by a separate PT_LOAD.

    Args:
        original: The PT_LOAD segment being split
        section_vaddr: Virtual address of the section being zeroed
        section_size: Size of the section
        aligned_vaddr: Virtual address of the aligned (zero-page) region
        aligned_size: Size of the aligned region
        aligned_offset: File offset of the aligned region (before removal)

    Returns:
        List of SegmentPieces representing the split
    """
    pieces: list[SegmentPiece] = []

    section_end_vaddr = section_vaddr + section_size
    section_offset = aligned_offset - (aligned_vaddr - section_vaddr)

    # Piece 1: Content before section (if any)
    if original.p_vaddr < section_vaddr:
        pre_vsize = section_vaddr - original.p_vaddr
        pre_fsize = min(pre_vsize, original.p_filesz)
        pieces.append(
            SegmentPiece(
                vaddr=original.p_vaddr,
                memsz=pre_vsize,
                offset=original.p_offset,
                filesz=pre_fsize,
            )
        )

    # Piece 2: Section prefix (if unaligned start)
    if section_vaddr < aligned_vaddr:
        prefix_size = aligned_vaddr - section_vaddr
        pieces.append(
            SegmentPiece(
                vaddr=section_vaddr,
                memsz=prefix_size,
                offset=section_offset,
                filesz=prefix_size,
            )
        )

    # Piece 3: NOBITS region covering aligned portion AND suffix
    # We extend memsz to cover to section end to avoid creating a separate
    # suffix segment that would have the same file offset (NOBITS collision)
    nobits_vsize = section_end_vaddr - aligned_vaddr
    pieces.append(
        SegmentPiece(
            vaddr=aligned_vaddr,
            memsz=nobits_vsize,  # Extended to cover suffix
            offset=aligned_offset,
            filesz=0,  # NOBITS - no file content
        )
    )

    # Piece 4: Content after section (if any)
    target_end = original.p_vaddr + original.p_memsz
    if section_end_vaddr < target_end:
        post_vaddr = section_end_vaddr
        post_vsize = target_end - post_vaddr
        # Calculate post-section file content
        section_file_end = section_offset + section_size
        post_offset = section_file_end - aligned_size  # Shifted down
        post_fsize = (original.p_offset + original.p_filesz) - section_file_end
        if post_fsize > 0:
            pieces.append(
                SegmentPiece(
                    vaddr=post_vaddr,
                    memsz=post_vsize,
                    offset=post_offset,
                    filesz=post_fsize,
                )
            )

    return pieces


def conservative_zero_page(
    surgery: ElfSurgery,
    section_name: str = ".hip_fatbin",
    spare_slots: int = 16,
) -> ZeroPageResult:
    """Apply conservative zero-page optimization to a section.

    This removes page-aligned content from the file while maintaining
    ELF validity. Only full pages are removed; partial pages are preserved.

    Args:
        surgery: ElfSurgery instance to operate on
        section_name: Section to zero-page (default: .hip_fatbin)
        spare_slots: Number of spare PHDR slots when relocating

    Returns:
        ZeroPageResult with operation outcome
    """
    # Find the target section
    section = surgery.find_section(section_name)
    if section is None:
        return ZeroPageResult(
            success=False,
            bytes_saved=0,
            phdr_relocated=False,
            pages_zeroed=0,
            error=f"Section '{section_name}' not found",
        )

    # Check if already zero-paged (NOBITS)
    if section.header.is_nobits:
        return ZeroPageResult(
            success=True,
            bytes_saved=0,
            phdr_relocated=False,
            pages_zeroed=0,
            error=f"Section '{section_name}' is already NOBITS (already zero-paged)",
        )

    section_vaddr = section.header.sh_addr
    section_size = section.header.sh_size
    section_offset = section.header.sh_offset

    # Calculate aligned range
    aligned_vaddr, aligned_size = calculate_aligned_range(section_vaddr, section_size)

    if aligned_size == 0:
        # Section is too small or misaligned - no optimization possible
        return ZeroPageResult(
            success=True,
            bytes_saved=0,
            phdr_relocated=False,
            pages_zeroed=0,
            error="Section too small or misaligned - no full pages to zero",
        )

    # When the section ends exactly on a page boundary (suffix==0), removing
    # ALL aligned content causes a NOBITS collision: the NOBITS segment and
    # the post-section segment would share the same file offset. Fix: leave
    # one page of content behind so the post-section piece starts one page
    # higher. Cost: one fewer page of savings (4KB), but the ELF structure
    # is completely standard with no synthetic offsets.
    section_end = section_vaddr + section_size
    suffix = section_end - round_down_to_page(section_end)
    if suffix == 0:
        aligned_size -= PAGE_SIZE
        if aligned_size == 0:
            return ZeroPageResult(
                success=True,
                bytes_saved=0,
                phdr_relocated=False,
                pages_zeroed=0,
                error="Section page-aligned end with only one page - skipping to avoid NOBITS collision",
            )

    aligned_offset = section_offset + (aligned_vaddr - section_vaddr)
    pages_to_zero = aligned_size // PAGE_SIZE

    # Find the PT_LOAD containing the section
    target_result = surgery.find_phdr_containing_vaddr(section_vaddr)
    if target_result is None:
        return ZeroPageResult(
            success=False,
            bytes_saved=0,
            phdr_relocated=False,
            pages_zeroed=0,
            error="Could not find PT_LOAD containing section",
        )

    target_idx, target_load = target_result

    if target_load.p_type != PT_LOAD:
        return ZeroPageResult(
            success=False,
            bytes_saved=0,
            phdr_relocated=False,
            pages_zeroed=0,
            error=f"Segment {target_idx} is not PT_LOAD",
        )

    # Remove the aligned bytes from file
    old_size = len(surgery.data)
    del surgery.data[aligned_offset : aligned_offset + aligned_size]
    bytes_removed = old_size - len(surgery.data)

    # Adjust ELF header offsets for removed bytes.
    # This must happen before ProgramHeaderManager.apply() so it sees
    # correct e_phoff when calculating available space for in-place writes.
    ehdr = surgery.ehdr
    if ehdr.e_phoff > aligned_offset:
        surgery._ehdr.e_phoff -= aligned_size
    if ehdr.e_shoff > aligned_offset:
        surgery._ehdr.e_shoff -= aligned_size

    # Split the target PT_LOAD into pieces
    pieces = split_load_segment(
        target_load,
        section_vaddr,
        section_size,
        aligned_vaddr,
        aligned_size,
        aligned_offset,
    )

    # Build new program header list
    new_phdrs: list[ProgramHeader] = []

    for idx, phdr in surgery.iter_program_headers():
        if idx != target_idx:
            # Not the target - adjust offset if after removed bytes
            if phdr.p_offset > aligned_offset:
                new_phdr = ProgramHeader(
                    p_type=phdr.p_type,
                    p_flags=phdr.p_flags,
                    p_offset=phdr.p_offset - aligned_size,
                    p_vaddr=phdr.p_vaddr,
                    p_paddr=phdr.p_paddr,
                    p_filesz=phdr.p_filesz,
                    p_memsz=phdr.p_memsz,
                    p_align=phdr.p_align,
                )
                new_phdrs.append(new_phdr)
            else:
                new_phdrs.append(phdr)
        else:
            # Replace target with split pieces
            for piece in pieces:
                # Create PT_LOAD for this piece
                new_phdr = ProgramHeader(
                    p_type=PT_LOAD,
                    p_flags=target_load.p_flags,
                    p_offset=piece.offset,
                    p_vaddr=piece.vaddr,
                    p_paddr=piece.vaddr,
                    p_filesz=piece.filesz,
                    p_memsz=piece.memsz,
                    p_align=target_load.p_align,
                )
                new_phdrs.append(new_phdr)

    # Use ProgramHeaderManager to write the new headers
    # This handles overflow and relocation automatically
    surgery._phdrs = new_phdrs
    surgery._ehdr.e_phnum = len(new_phdrs)

    # Write all program headers
    manager = ProgramHeaderManager(surgery, spare_slots=spare_slots)
    manager._phdrs = new_phdrs
    phdr_result = manager.apply()

    # Update section headers
    # (e_phoff and e_shoff were already adjusted above, before apply())
    for sect in surgery.iter_sections():
        if sect.index == section.index:
            # Mark as NOBITS
            new_shdr = SectionHeader(
                sh_name=sect.header.sh_name,
                sh_type=SHT_NOBITS,
                sh_flags=sect.header.sh_flags,
                sh_addr=sect.header.sh_addr,
                sh_offset=section_offset,  # Keep original offset (irrelevant for NOBITS)
                sh_size=sect.header.sh_size,
                sh_link=sect.header.sh_link,
                sh_info=sect.header.sh_info,
                sh_addralign=sect.header.sh_addralign,
                sh_entsize=sect.header.sh_entsize,
            )
            surgery.update_section_header(sect.index, new_shdr)
        elif sect.header.sh_offset > aligned_offset:
            # Adjust offset
            new_shdr = SectionHeader(
                sh_name=sect.header.sh_name,
                sh_type=sect.header.sh_type,
                sh_flags=sect.header.sh_flags,
                sh_addr=sect.header.sh_addr,
                sh_offset=sect.header.sh_offset - aligned_size,
                sh_size=sect.header.sh_size,
                sh_link=sect.header.sh_link,
                sh_info=sect.header.sh_info,
                sh_addralign=sect.header.sh_addralign,
                sh_entsize=sect.header.sh_entsize,
            )
            surgery.update_section_header(sect.index, new_shdr)

    # Update ELF header
    surgery.update_elf_header()

    return ZeroPageResult(
        success=True,
        bytes_saved=bytes_removed,
        phdr_relocated=phdr_result.relocated,
        pages_zeroed=pages_to_zero,
    )


def zero_page_section(
    input_path: Path,
    output_path: Path,
    section_name: str = ".hip_fatbin",
    spare_slots: int = 16,
) -> ZeroPageResult:
    """Apply zero-page optimization to a file.

    Convenience function that loads, modifies, and saves an ELF binary.

    Args:
        input_path: Input ELF binary
        output_path: Output ELF binary
        section_name: Section to zero-page
        spare_slots: Number of spare PHDR slots

    Returns:
        ZeroPageResult with operation outcome
    """
    surgery = ElfSurgery.load(input_path)
    result = conservative_zero_page(surgery, section_name, spare_slots)

    if result.success:
        surgery.save_preserving_mode(output_path, input_path.stat().st_mode)

    return result
