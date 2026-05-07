"""
Program Header Table Management.

This module handles the complexity of managing the ELF program header table,
including resizing and relocating it when space runs out.

The ELF program header table is at a fixed location in the file (e_phoff).
When we need to add more program headers than fit in the available space,
we must relocate the table to the end of the file and create a new PT_LOAD
segment to map it into memory.

Key invariants:
- (p_offset % PAGE_SIZE) == (p_vaddr % PAGE_SIZE) for all PT_LOAD segments
- PT_PHDR (if present) must point to the program header table location
- The program header table must be covered by a PT_LOAD segment
"""

from dataclasses import dataclass
from typing import Iterator

from .types import (
    ProgramHeader,
    ELF64_PHDR_SIZE,
    PAGE_SIZE,
    PT_LOAD,
    PT_PHDR,
    PF_R,
    round_up_to_page,
    page_align_offset,
)
from .surgery import ElfSurgery


@dataclass
class PhdrResizeResult:
    """Result of resizing the program header table."""

    # Whether the table was relocated
    relocated: bool

    # New e_phoff value
    new_phoff: int

    # New program headers (including any added PT_LOAD for PHDR)
    program_headers: list[ProgramHeader]

    # Number of spare slots available
    spare_slots: int


class ProgramHeaderManager:
    """Manages the program header table for an ELF binary.

    This class handles:
    - Adding new program headers
    - Resizing the table when space runs out
    - Relocating the table to end of file when needed
    - Maintaining PT_PHDR consistency

    Usage:
        surgery = ElfSurgery.load(path)
        manager = ProgramHeaderManager(surgery)

        # Add a new PT_LOAD segment
        new_phdr = ProgramHeader(PT_LOAD, 4, ...)
        manager.add_program_header(new_phdr)

        # Apply changes
        result = manager.apply()
    """

    def __init__(
        self,
        surgery: ElfSurgery,
        spare_slots: int = 16,
    ):
        """Initialize with an ElfSurgery instance.

        Args:
            surgery: Parsed ELF binary
            spare_slots: Number of extra slots to allocate when relocating
        """
        self._surgery = surgery
        self._spare_slots = spare_slots

        # Copy current program headers
        self._phdrs: list[ProgramHeader] = []
        for _, phdr in surgery.iter_program_headers():
            self._phdrs.append(phdr)

        # Track if we've made changes
        self._modified = False

    @property
    def program_headers(self) -> list[ProgramHeader]:
        """Current program headers."""
        return self._phdrs

    def add_program_header(self, phdr: ProgramHeader) -> int:
        """Add a new program header.

        Args:
            phdr: Program header to add

        Returns:
            Index of the new program header
        """
        self._phdrs.append(phdr)
        self._modified = True
        return len(self._phdrs) - 1

    def update_program_header(self, index: int, phdr: ProgramHeader) -> None:
        """Update an existing program header.

        Args:
            index: Index of program header to update
            phdr: New program header value
        """
        if index < 0 or index >= len(self._phdrs):
            raise ValueError(f"Invalid program header index: {index}")
        self._phdrs[index] = phdr
        self._modified = True

    def get_max_vaddr(self) -> int:
        """Get maximum virtual address across all PT_LOAD segments."""
        max_vaddr = 0
        for phdr in self._phdrs:
            if phdr.p_type == PT_LOAD:
                max_vaddr = max(max_vaddr, phdr.end_memsz_vaddr)
        return max_vaddr

    def allocate_vaddr(self) -> int:
        """Get next available virtual address after all existing segments.

        Returns:
            Virtual address at page boundary after existing segments
        """
        return round_up_to_page(self.get_max_vaddr())

    def _get_available_space(self) -> int:
        """Get available space for program headers at current location."""
        ehdr = self._surgery.ehdr
        min_offset = self._surgery.get_min_content_offset()
        return min_offset - ehdr.e_phoff

    def _get_current_capacity(self) -> int:
        """Get current capacity in number of program headers."""
        return self._get_available_space() // ELF64_PHDR_SIZE

    def _needs_relocation(self) -> bool:
        """Check if the program header table needs to be relocated."""
        required_size = len(self._phdrs) * ELF64_PHDR_SIZE
        available = self._get_available_space()
        return required_size > available

    def apply(self) -> PhdrResizeResult:
        """Apply changes to the program header table.

        This may:
        - Write in place if sufficient space exists
        - Relocate to end of file if needed

        Returns:
            PhdrResizeResult with details of what was done
        """
        required_size = len(self._phdrs) * ELF64_PHDR_SIZE
        available_space = self._get_available_space()

        if required_size <= available_space:
            # Fits in place
            return self._write_in_place()
        else:
            # Need to relocate
            return self._relocate_to_end()

    def _write_in_place(self) -> PhdrResizeResult:
        """Write program headers in place at current location."""
        ehdr = self._surgery.ehdr

        for i, phdr in enumerate(self._phdrs):
            self._surgery.update_program_header(i, phdr)

        # Update e_phnum in ELF header
        self._surgery._ehdr.e_phnum = len(self._phdrs)
        self._surgery.update_elf_header()

        spare = self._get_current_capacity() - len(self._phdrs)
        return PhdrResizeResult(
            relocated=False,
            new_phoff=ehdr.e_phoff,
            program_headers=list(self._phdrs),
            spare_slots=spare,
        )

    def _relocate_to_end(self) -> PhdrResizeResult:
        """Relocate program header table to end of file."""
        # Calculate new virtual address for PHDR table
        phdr_vaddr = self.allocate_vaddr()

        # Calculate file offset with proper alignment
        current_end = len(self._surgery.data)
        new_phoff = page_align_offset(current_end, phdr_vaddr)

        # Add padding to reach aligned offset
        padding = new_phoff - current_end
        if padding > 0:
            self._surgery.append_bytes(b"\x00" * padding, "padding for PHDR relocation")
            new_phoff = len(self._surgery.data)

        # Calculate capacity with spare slots
        # We need +1 for the PT_LOAD that will cover the PHDR
        entries_needed = len(self._phdrs) + 1
        capacity = (
            (entries_needed + self._spare_slots - 1) // self._spare_slots
        ) * self._spare_slots
        allocated_size = capacity * ELF64_PHDR_SIZE

        # Update PT_PHDR if present
        pt_phdr_index = None
        for i, phdr in enumerate(self._phdrs):
            if phdr.p_type == PT_PHDR:
                pt_phdr_index = i
                self._phdrs[i] = ProgramHeader(
                    p_type=PT_PHDR,
                    p_flags=phdr.p_flags,
                    p_offset=new_phoff,
                    p_vaddr=phdr_vaddr,
                    p_paddr=phdr_vaddr,
                    p_filesz=allocated_size,
                    p_memsz=allocated_size,
                    p_align=phdr.p_align,
                )
                break

        # Create PT_LOAD segment to cover relocated PHDR
        phdr_load = ProgramHeader(
            p_type=PT_LOAD,
            p_flags=PF_R,
            p_offset=new_phoff,
            p_vaddr=phdr_vaddr,
            p_paddr=phdr_vaddr,
            p_filesz=allocated_size,
            p_memsz=allocated_size,
            p_align=PAGE_SIZE,
        )
        self._phdrs.append(phdr_load)

        # Write all program headers to new location
        for phdr in self._phdrs:
            self._surgery.append_bytes(phdr.to_bytes(), "program header")

        # Write zero padding for spare slots
        unused = capacity - len(self._phdrs)
        if unused > 0:
            self._surgery.append_bytes(
                b"\x00" * (unused * ELF64_PHDR_SIZE), "PHDR spare slots"
            )

        # Update ELF header
        self._surgery._ehdr.e_phoff = new_phoff
        self._surgery._ehdr.e_phnum = len(self._phdrs)
        self._surgery.update_elf_header()

        # Re-parse program headers from new location
        self._surgery._phdrs = self._surgery._parse_program_headers()

        return PhdrResizeResult(
            relocated=True,
            new_phoff=new_phoff,
            program_headers=list(self._phdrs),
            spare_slots=unused,
        )


def create_load_segment(
    vaddr: int,
    file_offset: int,
    size: int,
    flags: int = PF_R,
) -> ProgramHeader:
    """Create a PT_LOAD segment.

    Args:
        vaddr: Virtual address
        file_offset: File offset (must be page-aligned with vaddr)
        size: Size of segment
        flags: Segment flags (default: read-only)

    Returns:
        New ProgramHeader
    """
    # Verify alignment constraint
    if (file_offset % PAGE_SIZE) != (vaddr % PAGE_SIZE):
        raise ValueError(
            f"File offset 0x{file_offset:x} not page-aligned with vaddr 0x{vaddr:x}"
        )

    return ProgramHeader(
        p_type=PT_LOAD,
        p_flags=flags,
        p_offset=file_offset,
        p_vaddr=vaddr,
        p_paddr=vaddr,
        p_filesz=size,
        p_memsz=size,
        p_align=PAGE_SIZE,
    )
