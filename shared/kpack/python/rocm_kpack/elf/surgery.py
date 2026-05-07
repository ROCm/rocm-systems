"""
High-level ELF surgery interface.

The ElfSurgery class provides a clean abstraction for ELF binary modifications.
It handles the complexity of maintaining ELF invariants while exposing
simple high-level operations.

Design principles:
- Parse once, modify in memory, write once
- Track all modifications for verification
- Fail fast with clear error messages
- Compose well with other utilities
"""

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .types import (
    ElfHeader,
    ProgramHeader,
    SectionHeader,
    RelaEntry,
    ELF64_EHDR_SIZE,
    ELF64_PHDR_SIZE,
    ELF64_SHDR_SIZE,
    PT_LOAD,
    SHT_PROGBITS,
    SHT_NOBITS,
    SHT_RELA,
    SHT_REL,
    SHF_ALLOC,
    R_X86_64_RELATIVE,
    get_section_name,
)


@dataclass
class SectionInfo:
    """Information about a section, combining header with derived data."""

    index: int
    name: str
    header: SectionHeader

    @property
    def vaddr(self) -> int:
        """Virtual address (only meaningful if SHF_ALLOC)."""
        return self.header.sh_addr

    @property
    def offset(self) -> int:
        """File offset."""
        return self.header.sh_offset

    @property
    def size(self) -> int:
        """Section size in bytes."""
        return self.header.sh_size

    @property
    def is_alloc(self) -> bool:
        """Whether section occupies memory at runtime."""
        return self.header.is_alloc


@dataclass
class Modification:
    """Record of a modification made to the ELF binary."""

    operation: str  # e.g., "write_bytes", "update_phdr"
    file_offset: int
    size: int
    description: str


@dataclass
class RelocationInfo:
    """Information about a found relocation entry."""

    section: "SectionInfo"
    file_offset: int
    entry: RelaEntry


@dataclass
class AddSectionResult:
    """Result of adding a section."""

    index: int  # Section header index
    offset: int  # File offset where content was written
    name_offset: int  # Offset of section name in .shstrtab


class ElfSurgery:
    """High-level interface for ELF binary modifications.

    Usage:
        surgery = ElfSurgery.load(Path("libfoo.so"))

        # Query operations
        section = surgery.find_section(".hip_fatbin")
        phdr = surgery.find_phdr_containing_vaddr(0x1000)

        # Modification operations
        surgery.write_bytes_at_offset(0x1000, b"\\x00" * 100)
        surgery.update_program_header(0, phdr)

        # Save and verify
        surgery.save(Path("libfoo_modified.so"))
    """

    def __init__(
        self,
        data: bytearray,
        path: Path | None = None,
    ):
        """Initialize with binary data.

        Prefer using ElfSurgery.load() for most use cases.

        Args:
            data: Mutable ELF binary data
            path: Original file path (for error messages)
        """
        self._data = data
        self._path = path
        self._modifications: list[Modification] = []

        # Parse structures
        self._ehdr = ElfHeader.from_bytes(data)
        self._phdrs = self._parse_program_headers()
        self._shdrs = self._parse_section_headers()
        self._section_names = self._parse_section_names()

    @classmethod
    def load(cls, path: Path) -> "ElfSurgery":
        """Load an ELF binary from file.

        Args:
            path: Path to ELF binary

        Returns:
            ElfSurgery instance ready for modifications
        """
        data = bytearray(path.read_bytes())
        return cls(data, path)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def data(self) -> bytearray:
        """Access to raw binary data (mutable)."""
        return self._data

    @property
    def ehdr(self) -> ElfHeader:
        """ELF header."""
        return self._ehdr

    @property
    def modifications(self) -> list[Modification]:
        """List of modifications made."""
        return self._modifications

    @property
    def is_pie_or_shared(self) -> bool:
        """Check if binary is PIE or shared library."""
        return self._ehdr.is_pie_or_shared

    # =========================================================================
    # Parsing (internal)
    # =========================================================================

    def _parse_program_headers(self) -> list[ProgramHeader]:
        """Parse all program headers."""
        phdrs = []
        for i in range(self._ehdr.e_phnum):
            offset = self._ehdr.e_phoff + i * self._ehdr.e_phentsize
            phdrs.append(ProgramHeader.from_bytes(self._data, offset))
        return phdrs

    def _parse_section_headers(self) -> list[SectionHeader]:
        """Parse all section headers."""
        shdrs = []
        for i in range(self._ehdr.e_shnum):
            offset = self._ehdr.e_shoff + i * self._ehdr.e_shentsize
            shdrs.append(SectionHeader.from_bytes(self._data, offset))
        return shdrs

    def _parse_section_names(self) -> dict[int, str]:
        """Parse section name string table."""
        names: dict[int, str] = {}
        if self._ehdr.e_shstrndx >= len(self._shdrs):
            return names

        shstrtab = self._shdrs[self._ehdr.e_shstrndx]
        for i, shdr in enumerate(self._shdrs):
            names[i] = get_section_name(self._data, shstrtab.sh_offset, shdr.sh_name)
        return names

    # =========================================================================
    # Query Operations
    # =========================================================================

    def find_section(self, name: str) -> SectionInfo | None:
        """Find a section by name.

        Args:
            name: Section name (e.g., ".hip_fatbin")

        Returns:
            SectionInfo if found, None otherwise
        """
        for idx, shdr in enumerate(self._shdrs):
            if self._section_names.get(idx) == name:
                return SectionInfo(
                    index=idx,
                    name=name,
                    header=shdr,
                )
        return None

    def get_section_by_index(self, index: int) -> SectionInfo | None:
        """Get section by index."""
        if 0 <= index < len(self._shdrs):
            return SectionInfo(
                index=index,
                name=self._section_names.get(index, ""),
                header=self._shdrs[index],
            )
        return None

    def iter_sections(self) -> Iterator[SectionInfo]:
        """Iterate over all sections."""
        for idx, shdr in enumerate(self._shdrs):
            yield SectionInfo(
                index=idx,
                name=self._section_names.get(idx, ""),
                header=shdr,
            )

    def get_section_content(self, section: SectionInfo | str) -> bytes:
        """Get content of a section.

        Args:
            section: SectionInfo or section name

        Returns:
            Section content bytes

        Raises:
            ValueError: If section not found or is NOBITS
        """
        if isinstance(section, str):
            info = self.find_section(section)
            if info is None:
                raise ValueError(f"Section not found: {section}")
            section = info

        if section.header.is_nobits:
            raise ValueError(f"Section {section.name} is NOBITS (no file content)")

        start = section.header.sh_offset
        end = start + section.header.sh_size
        return bytes(self._data[start:end])

    def find_phdr_by_index(self, index: int) -> ProgramHeader | None:
        """Get program header by index."""
        if 0 <= index < len(self._phdrs):
            return self._phdrs[index]
        return None

    def find_phdr_containing_vaddr(
        self, vaddr: int
    ) -> tuple[int, ProgramHeader] | None:
        """Find program header containing a virtual address.

        Args:
            vaddr: Virtual address to find

        Returns:
            Tuple of (index, ProgramHeader) if found, None otherwise
        """
        for idx, phdr in enumerate(self._phdrs):
            if phdr.contains_vaddr(vaddr):
                return (idx, phdr)
        return None

    def find_phdr_containing_offset(
        self, offset: int
    ) -> tuple[int, ProgramHeader] | None:
        """Find PT_LOAD program header containing a file offset.

        Args:
            offset: File offset to find

        Returns:
            Tuple of (index, ProgramHeader) if found, None otherwise
        """
        for idx, phdr in enumerate(self._phdrs):
            if phdr.p_type == PT_LOAD and phdr.contains_offset(offset):
                return (idx, phdr)
        return None

    def iter_program_headers(self) -> Iterator[tuple[int, ProgramHeader]]:
        """Iterate over all program headers with indices."""
        for idx, phdr in enumerate(self._phdrs):
            yield (idx, phdr)

    def iter_load_segments(self) -> Iterator[tuple[int, ProgramHeader]]:
        """Iterate over PT_LOAD segments with indices."""
        for idx, phdr in enumerate(self._phdrs):
            if phdr.p_type == PT_LOAD:
                yield (idx, phdr)

    def file_offset_to_vaddr(self, offset: int) -> int | None:
        """Convert file offset to virtual address.

        Args:
            offset: File offset

        Returns:
            Virtual address if offset is in a PT_LOAD segment, None otherwise
        """
        result = self.find_phdr_containing_offset(offset)
        if result is None:
            return None
        _, phdr = result
        return phdr.p_vaddr + (offset - phdr.p_offset)

    def vaddr_to_file_offset(self, vaddr: int) -> int | None:
        """Convert virtual address to file offset.

        Args:
            vaddr: Virtual address

        Returns:
            File offset if vaddr is in a PT_LOAD segment, None otherwise
        """
        result = self.find_phdr_containing_vaddr(vaddr)
        if result is None:
            return None
        _, phdr = result
        return phdr.p_offset + (vaddr - phdr.p_vaddr)

    def get_max_vaddr(self) -> int:
        """Get maximum virtual address across all PT_LOAD segments."""
        max_vaddr = 0
        for _, phdr in self.iter_load_segments():
            max_vaddr = max(max_vaddr, phdr.end_memsz_vaddr)
        return max_vaddr

    def get_min_content_offset(self) -> int:
        """Get minimum file offset of actual content after the PHDR table.

        This determines how much space is available for expanding the
        program header table in place. Content at or before e_phoff is
        excluded (it doesn't constrain expansion). Content after e_phoff
        — including sections that the linker placed between PHDR entries
        and the first segment — is included because it would be
        overwritten by expansion.
        """
        min_offset = len(self._data)
        e_phoff = self._ehdr.e_phoff

        # Find the first section with content after e_phoff.
        for shdr in self._shdrs:
            if shdr.sh_type != 0 and shdr.sh_offset > e_phoff:
                min_offset = min(min_offset, shdr.sh_offset)

        # Find the first PT_LOAD segment after e_phoff.
        # PT_LOAD at offset 0 (covering the ELF header) is correctly
        # excluded since 0 <= e_phoff for any valid ELF.
        for phdr in self._phdrs:
            if phdr.p_type == PT_LOAD and phdr.p_filesz > 0:
                if phdr.p_offset > e_phoff:
                    min_offset = min(min_offset, phdr.p_offset)

        return min_offset

    # =========================================================================
    # Write Operations (Low-Level)
    # =========================================================================

    def write_bytes_at_offset(
        self, offset: int, data: bytes, description: str = ""
    ) -> None:
        """Write bytes at a file offset.

        Args:
            offset: File offset
            data: Bytes to write
            description: Human-readable description for tracking
        """
        if offset + len(data) > len(self._data):
            raise ValueError(
                f"Write would exceed file bounds: offset={offset}, "
                f"len={len(data)}, file_size={len(self._data)}"
            )

        self._data[offset : offset + len(data)] = data
        self._modifications.append(
            Modification(
                operation="write_bytes",
                file_offset=offset,
                size=len(data),
                description=description or f"write {len(data)} bytes at 0x{offset:x}",
            )
        )

    def write_bytes_at_vaddr(
        self, vaddr: int, data: bytes, description: str = ""
    ) -> None:
        """Write bytes at a virtual address.

        Args:
            vaddr: Virtual address
            data: Bytes to write
            description: Human-readable description for tracking

        Raises:
            ValueError: If vaddr is not in a PT_LOAD segment
        """
        offset = self.vaddr_to_file_offset(vaddr)
        if offset is None:
            raise ValueError(f"Virtual address 0x{vaddr:x} not in any PT_LOAD segment")
        self.write_bytes_at_offset(offset, data, description)

    def zero_range(self, offset: int, size: int, description: str = "") -> None:
        """Zero out a range of bytes.

        Args:
            offset: Start file offset
            size: Number of bytes to zero
            description: Human-readable description
        """
        self.write_bytes_at_offset(
            offset,
            b"\x00" * size,
            description or f"zero {size} bytes at 0x{offset:x}",
        )

    # =========================================================================
    # Header Update Operations
    # =========================================================================

    def update_elf_header(self) -> None:
        """Write current ELF header to binary.

        Call this after modifying self._ehdr fields.
        """
        self._ehdr.write_to(self._data, 0)
        self._modifications.append(
            Modification(
                operation="update_ehdr",
                file_offset=0,
                size=ELF64_EHDR_SIZE,
                description="update ELF header",
            )
        )

    def update_program_header(self, index: int, phdr: ProgramHeader) -> None:
        """Update a program header in the binary.

        Args:
            index: Program header index
            phdr: New program header value
        """
        if index < 0 or index >= self._ehdr.e_phnum:
            raise ValueError(f"Invalid program header index: {index}")

        offset = self._ehdr.e_phoff + index * self._ehdr.e_phentsize
        phdr.write_to(self._data, offset)
        self._phdrs[index] = phdr
        self._modifications.append(
            Modification(
                operation="update_phdr",
                file_offset=offset,
                size=ELF64_PHDR_SIZE,
                description=f"update program header {index}",
            )
        )

    def update_section_header(self, index: int, shdr: SectionHeader) -> None:
        """Update a section header in the binary.

        Args:
            index: Section header index
            shdr: New section header value
        """
        if index < 0 or index >= self._ehdr.e_shnum:
            raise ValueError(f"Invalid section header index: {index}")

        offset = self._ehdr.e_shoff + index * self._ehdr.e_shentsize
        shdr.write_to(self._data, offset)
        self._shdrs[index] = shdr
        self._modifications.append(
            Modification(
                operation="update_shdr",
                file_offset=offset,
                size=ELF64_SHDR_SIZE,
                description=f"update section header {index}",
            )
        )

    # =========================================================================
    # Relocation Operations
    # =========================================================================

    def iter_rela_sections(self) -> Iterator[SectionInfo]:
        """Iterate over RELA sections."""
        for section in self.iter_sections():
            if section.header.sh_type == SHT_RELA:
                yield section

    def iter_relocations(self, section: SectionInfo) -> Iterator[tuple[int, RelaEntry]]:
        """Iterate over relocations in a RELA section.

        Args:
            section: RELA section

        Yields:
            Tuples of (file_offset, RelaEntry)
        """
        if section.header.sh_type != SHT_RELA:
            raise ValueError(f"Section {section.name} is not SHT_RELA")

        offset = section.header.sh_offset
        end = offset + section.header.sh_size
        entry_size = section.header.sh_entsize or RelaEntry.SIZE

        while offset < end:
            yield (offset, RelaEntry.from_bytes(self._data, offset))
            offset += entry_size

    def find_relocation_at_vaddr(self, target_vaddr: int) -> RelocationInfo | None:
        """Find a relocation targeting a specific virtual address.

        Args:
            target_vaddr: Virtual address the relocation targets (r_offset)

        Returns:
            RelocationInfo if found, None otherwise
        """
        for section in self.iter_rela_sections():
            for offset, rela in self.iter_relocations(section):
                if rela.r_offset == target_vaddr:
                    return RelocationInfo(
                        section=section, file_offset=offset, entry=rela
                    )
        return None

    def update_relocation(
        self, file_offset: int, rela: RelaEntry, description: str = ""
    ) -> None:
        """Update a RELA entry at a file offset.

        Args:
            file_offset: File offset of the RELA entry
            rela: New RELA entry value
            description: Human-readable description
        """
        rela.write_to(self._data, file_offset)
        self._modifications.append(
            Modification(
                operation="update_rela",
                file_offset=file_offset,
                size=RelaEntry.SIZE,
                description=description or f"update relocation at 0x{file_offset:x}",
            )
        )

    # =========================================================================
    # File Operations
    # =========================================================================

    def resize(self, new_size: int) -> None:
        """Resize the binary data.

        Args:
            new_size: New size in bytes
        """
        current_size = len(self._data)
        if new_size > current_size:
            self._data.extend(b"\x00" * (new_size - current_size))
        elif new_size < current_size:
            del self._data[new_size:]

    def append_bytes(self, data: bytes, description: str = "") -> int:
        """Append bytes to end of file.

        Args:
            data: Bytes to append
            description: Human-readable description

        Returns:
            File offset where data was appended
        """
        offset = len(self._data)
        self._data.extend(data)
        self._modifications.append(
            Modification(
                operation="append_bytes",
                file_offset=offset,
                size=len(data),
                description=description or f"append {len(data)} bytes",
            )
        )
        return offset

    def save(self, path: Path) -> None:
        """Save modified binary to file.

        Args:
            path: Output file path
        """
        path.write_bytes(self._data)

    def save_preserving_mode(self, path: Path, mode: int | None = None) -> None:
        """Save modified binary, optionally setting file mode.

        Args:
            path: Output file path
            mode: File mode to set. If None, does not set mode (new files get
                default umask, existing files keep their mode).
        """
        path.write_bytes(self._data)
        if mode is not None:
            os.chmod(path, mode)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def read_pointer_at_vaddr(self, vaddr: int) -> int:
        """Read an 8-byte pointer at a virtual address.

        Args:
            vaddr: Virtual address

        Returns:
            Pointer value (little-endian uint64)
        """
        offset = self.vaddr_to_file_offset(vaddr)
        if offset is None:
            raise ValueError(f"Virtual address 0x{vaddr:x} not in any PT_LOAD segment")

        return struct.unpack_from("<Q", self._data, offset)[0]

    def write_pointer_at_vaddr(
        self, vaddr: int, value: int, description: str = ""
    ) -> None:
        """Write an 8-byte pointer at a virtual address.

        Args:
            vaddr: Virtual address
            value: Pointer value to write
            description: Human-readable description
        """
        data = struct.pack("<Q", value)
        self.write_bytes_at_vaddr(
            vaddr,
            data,
            description or f"write pointer 0x{value:x} at vaddr 0x{vaddr:x}",
        )

    # =========================================================================
    # Section Addition
    # =========================================================================

    def add_section(
        self,
        name: str,
        content: bytes,
        section_type: int = SHT_PROGBITS,
        flags: int = 0,
        addralign: int = 1,
    ) -> AddSectionResult:
        """Add a new section to the ELF binary.

        TODO(#3): Optimize this to achieve similar overhead as objcopy.

        This appends the section content, extends .shstrtab with the section
        name, and adds a new section header entry.

        Steps:
        1. Append section content to end of file
        2. Extend .shstrtab with section name
        3. Add new section header entry
        4. Update e_shnum in ELF header
        5. Relocate section header table if needed

        Args:
            name: Section name (e.g., ".rocm_kpack_ref")
            content: Section content bytes
            section_type: SHT_* type (default: SHT_PROGBITS)
            flags: SHF_* flags (default: 0, meaning non-ALLOC)
            addralign: Alignment requirement

        Returns:
            AddSectionResult with section details

        Raises:
            ValueError: If section already exists or .shstrtab not found
        """
        # Check if section already exists
        if self.find_section(name) is not None:
            raise ValueError(f"Section '{name}' already exists")

        # Find .shstrtab section
        shstrtab_idx = self._ehdr.e_shstrndx
        if shstrtab_idx >= len(self._shdrs):
            raise ValueError("Invalid section header string table index")

        shstrtab = self._shdrs[shstrtab_idx]

        # Step 1: Append section content with alignment padding
        content_offset = len(self._data)
        if addralign > 1:
            padding = (addralign - (content_offset % addralign)) % addralign
            if padding > 0:
                self.append_bytes(b"\x00" * padding, f"align padding for {name}")
                content_offset = len(self._data)

        self.append_bytes(content, f"content for {name}")

        # Step 2: Extend .shstrtab with section name
        # The name goes at the end of the current string table content
        name_bytes = name.encode("utf-8") + b"\x00"
        name_offset = shstrtab.sh_size

        # We need to insert the name into the string table
        # Relocate .shstrtab to end of file with the new name appended.
        # This is robust regardless of where .shstrtab currently sits.
        new_shstrtab_offset = len(self._data)
        old_shstrtab_content = bytes(
            self._data[shstrtab.sh_offset : shstrtab.sh_offset + shstrtab.sh_size]
        )
        new_shstrtab_content = old_shstrtab_content + name_bytes
        self.append_bytes(new_shstrtab_content, f"relocated .shstrtab with {name}")

        # Update .shstrtab header
        new_shstrtab = SectionHeader(
            sh_name=shstrtab.sh_name,
            sh_type=shstrtab.sh_type,
            sh_flags=shstrtab.sh_flags,
            sh_addr=shstrtab.sh_addr,
            sh_offset=new_shstrtab_offset,
            sh_size=len(new_shstrtab_content),
            sh_link=shstrtab.sh_link,
            sh_info=shstrtab.sh_info,
            sh_addralign=shstrtab.sh_addralign,
            sh_entsize=shstrtab.sh_entsize,
        )
        self._shdrs[shstrtab_idx] = new_shstrtab

        # Step 3: Create new section header
        new_section_idx = len(self._shdrs)
        new_shdr = SectionHeader(
            sh_name=name_offset,
            sh_type=section_type,
            sh_flags=flags,
            sh_addr=0,  # Not loaded into memory unless SHF_ALLOC
            sh_offset=content_offset,
            sh_size=len(content),
            sh_link=0,
            sh_info=0,
            sh_addralign=addralign,
            sh_entsize=0,
        )
        self._shdrs.append(new_shdr)

        # Step 4 & 5: Update e_shnum and relocate section header table
        # The section header table is typically at the end of the file
        # We need to write all section headers including the new one

        # Calculate new section header table location (at end of file)
        new_shoff = len(self._data)

        # Write all section headers
        for shdr in self._shdrs:
            self.append_bytes(shdr.to_bytes(), "section header")

        # Update ELF header
        self._ehdr.e_shnum = len(self._shdrs)
        self._ehdr.e_shoff = new_shoff
        self.update_elf_header()

        # Re-parse section names to include new section
        self._section_names = self._parse_section_names()

        self._modifications.append(
            Modification(
                operation="add_section",
                file_offset=content_offset,
                size=len(content),
                description=f"add section {name}",
            )
        )

        return AddSectionResult(
            index=new_section_idx,
            offset=content_offset,
            name_offset=name_offset,
        )
