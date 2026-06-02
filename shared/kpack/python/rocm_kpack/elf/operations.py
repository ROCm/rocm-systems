"""
High-level ELF surgery operations.

This module provides composable operations for ELF binary manipulation.
Each operation is built on top of the lower-level ElfSurgery and
ProgramHeaderManager primitives.

Operations:
- map_section_to_load: Map a non-ALLOC section to a new PT_LOAD segment
- set_pointer: Write a pointer value with relocation update
- update_relocation_addend: Find and update a relocation's addend
"""

from dataclasses import dataclass
from pathlib import Path

from .types import (
    ProgramHeader,
    SectionHeader,
    RelaEntry,
    PAGE_SIZE,
    PT_LOAD,
    PF_R,
    SHF_ALLOC,
    SHT_NOBITS,
    SHT_NULL,
    R_X86_64_RELATIVE,
    round_up_to_page,
    page_align_offset,
)
from .surgery import ElfSurgery, SectionInfo
from .phdr_manager import ProgramHeaderManager, create_load_segment


def _first_non_alloc_file_offset(surgery: ElfSurgery) -> int | None:
    """Return the lowest file offset occupied by a non-allocatable section.

    This is used to detect whether a binary has non-allocatable sections
    (e.g. .debug_*, .gnu.build.attributes) that follow the allocatable ones
    in file offset space — the layout produced by ASAN / debug-info builds.

    Sections with no file content (SHT_NOBITS, SHT_NULL, sh_offset == 0) are
    excluded because they don't actually occupy bytes on disk.

    Returns None if no non-allocatable sections with real content exist, which
    is the common case for stripped / non-ASAN binaries.
    """
    offsets = [
        shdr.sh_offset
        for shdr in surgery._shdrs
        if not (shdr.sh_flags & SHF_ALLOC)
        and shdr.sh_type not in (SHT_NULL, SHT_NOBITS)
        and shdr.sh_offset > 0
    ]
    return min(offsets) if offsets else None


@dataclass
class MapSectionResult:
    """Result of mapping a section to PT_LOAD."""

    success: bool
    vaddr: int  # Virtual address where section was mapped
    file_offset: int  # File offset of section (may have been relocated)
    phdr_relocated: bool  # Whether PHDR table was relocated
    error: str | None = None


def map_section_to_load(
    surgery: ElfSurgery,
    section_name: str,
    vaddr: int | None = None,
    spare_slots: int = 16,
) -> MapSectionResult:
    """Map a non-ALLOC section to a new PT_LOAD segment.

    This operation:
    1. Finds the specified section
    2. Allocates a virtual address (if not specified)
    3. Relocates section data if needed for mmap alignment
    4. Creates a new PT_LOAD segment
    5. Updates the section header with SHF_ALLOC and new address

    Args:
        surgery: ElfSurgery instance to operate on
        section_name: Name of section to map (e.g., ".rocm_kpack_ref")
        vaddr: Virtual address for new segment (auto-allocate if None)
        spare_slots: Number of spare PHDR slots when relocating

    Returns:
        MapSectionResult with operation outcome
    """
    # Find the section
    section = surgery.find_section(section_name)
    if section is None:
        return MapSectionResult(
            success=False,
            vaddr=0,
            file_offset=0,
            phdr_relocated=False,
            error=f"Section '{section_name}' not found",
        )

    # Check if already allocated
    if section.header.is_alloc:
        return MapSectionResult(
            success=False,
            vaddr=section.header.sh_addr,
            file_offset=section.header.sh_offset,
            phdr_relocated=False,
            error=f"Section '{section_name}' already has SHF_ALLOC",
        )

    # Initialize PHDR manager
    manager = ProgramHeaderManager(surgery, spare_slots=spare_slots)

    # Allocate virtual address if not specified
    if vaddr is None:
        vaddr = manager.allocate_vaddr()

    # Determine where to place the section content in file offset space.
    #
    # The ELF convention (enforced by dwz and dh_dwz) requires that every
    # allocatable section (SHF_ALLOC) has a lower file offset than every
    # non-allocatable section (.debug_*, .gnu.build.attributes, etc.).
    #
    # In normal (stripped / non-ASAN) builds there are no non-alloc sections,
    # so add_section() already placed the content at the end-of-file and that
    # position is fine — we just need the usual mmap alignment fix if required.
    #
    # In ASAN / debug-info builds the binary already has non-alloc sections
    # after the alloc ones.  add_section() appended the .rocm_kpack_ref content
    # past those non-alloc sections, violating the file-offset ordering.  We
    # detect this and use insert_bytes_at() to splice the content before the
    # first non-alloc section instead.  All subsequent sh_offset / p_offset
    # values are shifted automatically by insert_bytes_at().
    section_offset = section.header.sh_offset
    vaddr_mod = vaddr % PAGE_SIZE
    non_alloc_split = _first_non_alloc_file_offset(surgery)

    new_offset = section_offset
    if non_alloc_split is not None and section_offset >= non_alloc_split:
        # ASAN / debug-info build: section content landed after non-alloc data.
        # Compute an insert point that satisfies mmap alignment:
        #   (insert_offset % PAGE_SIZE) == (vaddr % PAGE_SIZE)
        # We choose the smallest offset >= non_alloc_split with this property.
        insert_offset = page_align_offset(non_alloc_split, vaddr)
        if insert_offset < non_alloc_split:
            # page_align_offset can return a value < non_alloc_split when
            # non_alloc_split is already past the desired residue — advance one page.
            insert_offset += PAGE_SIZE
        padding = insert_offset - non_alloc_split
        section_data = surgery.get_section_content(section)
        # Splice: padding bytes + section content inserted at insert_offset.
        # insert_bytes_at() shifts all sh_offset / p_offset / e_shoff values
        # at or after insert_offset, so the rest of the binary stays coherent.
        surgery.insert_bytes_at(
            non_alloc_split,
            b"\x00" * padding + section_data,
            f"insert {section_name} before non-alloc sections (ASAN layout fix)",
        )
        new_offset = insert_offset
    elif (section_offset % PAGE_SIZE) != vaddr_mod:
        # Normal build, alignment mismatch: copy to a correctly-aligned position
        # at end of file (the original behavior, unchanged).
        new_offset = page_align_offset(len(surgery.data), vaddr)
        padding = new_offset - len(surgery.data)
        if padding > 0:
            surgery.append_bytes(b"\x00" * padding, "mmap alignment padding")
            new_offset = len(surgery.data)
        section_data = surgery.get_section_content(section)
        surgery.append_bytes(section_data, f"relocated {section_name} content")
    # else: section is already at a correctly-aligned position, use it as-is.

    # Create PT_LOAD segment for the section
    new_load = create_load_segment(
        vaddr=vaddr,
        file_offset=new_offset,
        size=section.header.sh_size,
        flags=PF_R,  # Read-only
    )
    manager.add_program_header(new_load)

    # Apply PHDR changes
    phdr_result = manager.apply()

    # Update section header: set SHF_ALLOC so the loader maps this section into
    # the process address space (required for the PT_LOAD segment we just added).
    new_shdr = SectionHeader(
        sh_name=section.header.sh_name,
        sh_type=section.header.sh_type,
        sh_flags=section.header.sh_flags | SHF_ALLOC,
        sh_addr=vaddr,
        sh_offset=new_offset,
        sh_size=section.header.sh_size,
        sh_link=section.header.sh_link,
        sh_info=section.header.sh_info,
        sh_addralign=section.header.sh_addralign,
        sh_entsize=section.header.sh_entsize,
    )
    surgery.update_section_header(section.index, new_shdr)

    # Reorder the section header table so the newly-ALLOC section appears at a
    # lower index than all non-allocatable sections.
    #
    # The file-offset ordering (dwz's primary check) is already fixed above by
    # insert_bytes_at() for ASAN builds, or was never violated in normal builds.
    # This call addresses the complementary section-header-index ordering:
    # add_section() appended .rocm_kpack_ref as the last SHT entry, so its
    # index was higher than .debug_* entries.  Some tools also validate that
    # SHF_ALLOC entries precede non-ALLOC entries in index order.
    surgery.reorder_section_headers_alloc_first()

    return MapSectionResult(
        success=True,
        vaddr=vaddr,
        file_offset=new_offset,
        phdr_relocated=phdr_result.relocated,
    )


@dataclass
class SetPointerResult:
    """Result of setting a pointer value."""

    success: bool
    relocation_updated: bool
    error: str | None = None


def set_pointer(
    surgery: ElfSurgery,
    pointer_vaddr: int,
    target_vaddr: int,
    update_relocation: bool = True,
) -> SetPointerResult:
    """Write a pointer value at a virtual address.

    For PIE/shared libraries, this also updates the relocation entry
    to ensure the pointer is correctly adjusted at load time.

    Args:
        surgery: ElfSurgery instance to operate on
        pointer_vaddr: Virtual address where to write the pointer
        target_vaddr: Target address to store
        update_relocation: Whether to update relocation entry

    Returns:
        SetPointerResult with operation outcome
    """
    # Write the pointer value
    try:
        surgery.write_pointer_at_vaddr(
            pointer_vaddr,
            target_vaddr,
            f"set pointer at 0x{pointer_vaddr:x} -> 0x{target_vaddr:x}",
        )
    except ValueError as e:
        return SetPointerResult(
            success=False,
            relocation_updated=False,
            error=str(e),
        )

    # Update relocation if needed (PIE/shared libraries)
    reloc_updated = False
    if update_relocation and surgery.is_pie_or_shared:
        reloc_result = update_relocation_addend(
            surgery,
            pointer_vaddr,
            target_vaddr,
        )
        reloc_updated = reloc_result.success
        if not reloc_updated and reloc_result.error:
            # Not finding a relocation is OK - pointer might be in read-only data
            pass

    return SetPointerResult(
        success=True,
        relocation_updated=reloc_updated,
    )


@dataclass
class UpdateRelocationResult:
    """Result of updating a relocation entry."""

    success: bool
    found: bool
    error: str | None = None


def update_relocation_addend(
    surgery: ElfSurgery,
    target_vaddr: int,
    new_addend: int,
    convert_to_relative: bool = True,
) -> UpdateRelocationResult:
    """Find and update a relocation entry targeting a virtual address.

    Args:
        surgery: ElfSurgery instance to operate on
        target_vaddr: Virtual address the relocation targets (r_offset)
        new_addend: New addend value to set
        convert_to_relative: Convert to R_X86_64_RELATIVE if not already

    Returns:
        UpdateRelocationResult with operation outcome
    """
    # Find the relocation
    reloc_info = surgery.find_relocation_at_vaddr(target_vaddr)
    if reloc_info is None:
        return UpdateRelocationResult(
            success=False,
            found=False,
            error=f"No relocation found at 0x{target_vaddr:x}",
        )

    # Create updated relocation entry
    if convert_to_relative:
        # Convert to R_X86_64_RELATIVE with symbol index 0
        new_info = RelaEntry.make_info(0, R_X86_64_RELATIVE)
    else:
        new_info = reloc_info.entry.r_info

    new_rela = RelaEntry(
        r_offset=reloc_info.entry.r_offset,
        r_info=new_info,
        r_addend=new_addend,
    )

    surgery.update_relocation(
        reloc_info.file_offset,
        new_rela,
        f"update relocation at 0x{target_vaddr:x}: addend={new_addend:#x}",
    )

    return UpdateRelocationResult(
        success=True,
        found=True,
    )


def find_relocation_for_pointer(
    surgery: ElfSurgery,
    pointer_vaddr: int,
) -> tuple[int, RelaEntry] | None:
    """Find the relocation entry for a pointer at a given address.

    Args:
        surgery: ElfSurgery instance
        pointer_vaddr: Virtual address of the pointer

    Returns:
        Tuple of (file_offset, RelaEntry) if found, None otherwise
    """
    reloc_info = surgery.find_relocation_at_vaddr(pointer_vaddr)
    if reloc_info is None:
        return None
    return (reloc_info.file_offset, reloc_info.entry)
