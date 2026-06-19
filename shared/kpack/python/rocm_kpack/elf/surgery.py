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
    SHT_SYMTAB,
    SHT_DYNSYM,
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

    @staticmethod
    def has_fatbin_section(file_path: Path) -> bool:
        """Fast check for .hip_fatbin section without loading the full binary.

        Reads only the ELF header, section header table, and section name
        string table — typically a few KB total even for multi-GB binaries.
        Returns False for non-ELF files (wrong magic, too small, 32-bit).
        Raises on I/O errors or corrupt ELF headers.
        """
        file_size = os.path.getsize(file_path)
        if file_size < ELF64_EHDR_SIZE:
            return False

        with open(file_path, "rb") as f:
            ehdr = f.read(ELF64_EHDR_SIZE)
            if len(ehdr) < ELF64_EHDR_SIZE:
                return False
            if ehdr[:4] != b"\x7fELF":
                return False
            if ehdr[4] != 2:  # ELFCLASS64
                return False

            # All offsets and sizes below follow the System V ABI for ELF64.
            # e_shoff:    offset 40 — file offset to section header table
            # e_shentsize: offset 58 — size of each section header entry
            # e_shnum:    offset 60 — number of section header entries
            # e_shstrndx: offset 62 — index of the .shstrtab section header
            e_shoff = struct.unpack_from("<Q", ehdr, 40)[0]
            e_shentsize = struct.unpack_from("<H", ehdr, 58)[0]
            e_shnum = struct.unpack_from("<H", ehdr, 60)[0]
            e_shstrndx = struct.unpack_from("<H", ehdr, 62)[0]

            if e_shoff == 0 or e_shnum == 0 or e_shstrndx >= e_shnum:
                return False

            # Bounds check section header table against file size
            total_sht_size = e_shentsize * e_shnum
            if (
                e_shoff > file_size
                or total_sht_size > file_size
                or e_shoff + total_sht_size > file_size
            ):
                return False

            # Read section header table
            f.seek(e_shoff)
            shtab = f.read(total_sht_size)
            if len(shtab) < total_sht_size:
                return False

            # Read .shstrtab
            strtab_entry = e_shstrndx * e_shentsize
            sh_offset = struct.unpack_from("<Q", shtab, strtab_entry + 24)[0]
            sh_size = struct.unpack_from("<Q", shtab, strtab_entry + 32)[0]

            # Bounds check .shstrtab region against file size
            if (
                sh_offset > file_size
                or sh_size > file_size
                or sh_offset + sh_size > file_size
            ):
                return False

            f.seek(sh_offset)
            shstrtab = f.read(sh_size)

            # Search for ".hip_fatbin" in section names
            target = b".hip_fatbin\x00"
            if target not in shstrtab:
                return False

            target_offset = shstrtab.index(target)
            for i in range(e_shnum):
                sh_name = struct.unpack_from("<I", shtab, i * e_shentsize)[0]
                if sh_name == target_offset:
                    return True

            return False

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

    def insert_bytes_at(
        self, file_offset: int, data: bytes, description: str = ""
    ) -> None:
        """Insert bytes at a file offset, shifting all subsequent offsets.

        Unlike append_bytes() which always adds to the end of the file, this
        method splices bytes mid-file at ``file_offset`` and then fixes up every
        offset-bearing structure so the binary remains coherent:

          * sh_offset in every section header whose content sits at or past
            ``file_offset`` (SHT_NOBITS sections are skipped — they occupy no
            file space)
          * p_offset in every program header at or past ``file_offset``
          * e_shoff in the ELF header if the section header table was at or
            past ``file_offset``

        Virtual addresses (sh_addr, p_vaddr) are never touched — they are
        load-time concepts unrelated to file layout.

        Primary use case: placing an allocatable section (e.g. .rocm_kpack_ref)
        before the first non-allocatable section in file offset space, which is
        required by tools like ``dwz`` that enforce ELF file-offset ordering
        conventions.  In binaries without non-allocatable sections (the normal
        non-ASAN case) this method is never called — append_bytes() is used
        instead.

        Args:
            file_offset: Position in the file at which to splice the new bytes.
                         All existing bytes at or after this offset are shifted
                         right by len(data).
            data: Bytes to insert.
            description: Human-readable description for the modification log.

        Raises:
            ValueError: If file_offset is beyond the current end of file.
        """
        if file_offset > len(self._data):
            raise ValueError(
                f"insert_bytes_at: offset 0x{file_offset:x} beyond file size 0x{len(self._data):x}"
            )

        n = len(data)

        # --- Splice bytes into the data buffer ------------------------------
        self._data[file_offset:file_offset] = data

        # --- Update section header file offsets -----------------------------
        # sh_offset points to where the section's content lives in the file.
        # SHT_NOBITS sections (like .bss) consume no file space, so their
        # sh_offset is a nominal address and must NOT be shifted.
        for i, shdr in enumerate(self._shdrs):
            if shdr.sh_type == SHT_NOBITS:
                continue
            if shdr.sh_offset >= file_offset:
                updated = SectionHeader(
                    sh_name=shdr.sh_name,
                    sh_type=shdr.sh_type,
                    sh_flags=shdr.sh_flags,
                    sh_addr=shdr.sh_addr,
                    sh_offset=shdr.sh_offset + n,
                    sh_size=shdr.sh_size,
                    sh_link=shdr.sh_link,
                    sh_info=shdr.sh_info,
                    sh_addralign=shdr.sh_addralign,
                    sh_entsize=shdr.sh_entsize,
                )
                self._shdrs[i] = updated

        # --- Update program header file offsets ----------------------------
        # p_offset is the file offset where the segment begins.  p_vaddr and
        # p_paddr are virtual/physical addresses — leave them alone.
        for i, phdr in enumerate(self._phdrs):
            if phdr.p_offset >= file_offset:
                updated = ProgramHeader(
                    p_type=phdr.p_type,
                    p_flags=phdr.p_flags,
                    p_offset=phdr.p_offset + n,
                    p_vaddr=phdr.p_vaddr,
                    p_paddr=phdr.p_paddr,
                    p_filesz=phdr.p_filesz,
                    p_memsz=phdr.p_memsz,
                    p_align=phdr.p_align,
                )
                self._phdrs[i] = updated

        # --- Update e_shoff and e_phoff in the ELF header -------------------
        # e_shoff points to the section header table; e_phoff points to the
        # program header table.  Both are file offsets and must be shifted if
        # they fall at or after the insertion point.  In practice e_phoff is
        # always 64 (immediately after the ELF header) and will never be
        # affected, but we update it for correctness.
        if self._ehdr.e_shoff >= file_offset:
            self._ehdr.e_shoff += n
        if self._ehdr.e_phoff >= file_offset:
            self._ehdr.e_phoff += n

        # --- Flush all updated headers back to the buffer -------------------
        # The buffer has already been spliced, so all offsets above are now
        # correct absolute positions in the new file layout.
        self._ehdr.write_to(self._data, 0)

        phdr_table_offset = self._ehdr.e_phoff
        for i, phdr in enumerate(self._phdrs):
            phdr.write_to(self._data, phdr_table_offset + i * self._ehdr.e_phentsize)

        sht_offset = self._ehdr.e_shoff
        for i, shdr in enumerate(self._shdrs):
            shdr.write_to(self._data, sht_offset + i * self._ehdr.e_shentsize)

        self._modifications.append(
            Modification(
                operation="insert_bytes_at",
                file_offset=file_offset,
                size=n,
                description=description or f"insert {n} bytes at 0x{file_offset:x}",
            )
        )

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

    def reorder_section_headers_alloc_first(self) -> None:
        """Reorder the section header table so all SHF_ALLOC sections precede non-ALLOC.

        Background / what this fixes
        -----------------------------
        This fixes **section header table index ordering** — specifically the
        convention that allocatable sections (SHF_ALLOC) appear at lower
        indices than non-allocatable ones.  When kpack adds .rocm_kpack_ref
        via ``add_section`` and then ``map_section_to_load`` sets SHF_ALLOC
        on it, the new section sits at the last index in the table (after all
        .debug_* entries), violating the index-order convention expected by
        some tools.

        Note on dwz / dh_dwz
        ---------------------
        ``dwz`` (DWARF debuginfo compressor, invoked by ``dh_dwz`` in Debian
        packaging) checks that allocatable sections have a lower **file
        offset** than non-allocatable ones — not that their SHT indices are
        ordered.  The file-offset ordering is fixed separately by
        ``insert_bytes_at()`` in ``map_section_to_load()`` for ASAN /
        debug-info builds.  This method addresses the complementary
        index-order requirement and ensures consistency for other tools that
        do validate SHT index ordering.

        What this method does
        ---------------------
        Only the section header table is reordered; file content is NOT moved.
        The new order is a stable partition:
          1. SHT_NULL entry at index 0 (always — required by ELF spec)
          2. All remaining SHF_ALLOC sections (preserving their relative order)
          3. All remaining non-ALLOC sections (preserving their relative order)

        After reordering, every cross-reference to section indices is updated:
          * sh_link / sh_info in every section header (many point to symtab,
            strtab, or other sections by index)
          * e_shstrndx in the ELF header (index of the section name strtab)
          * st_shndx in every Elf64_Sym entry in .symtab and .dynsym (each
            symbol records which section it is defined in)

        Special symbol index values (SHN_UNDEF=0, SHN_ABS=0xfff1,
        SHN_COMMON=0xfff2, SHN_XINDEX=0xffff) are not section references and
        are left unchanged.

        The updated section header table is written back to the same location
        in the file that ``add_section`` chose (always at the end of the binary
        after its call).  Internal state (self._shdrs, self._section_names,
        self._ehdr.e_shstrndx) is kept consistent.
        """
        n = len(self._shdrs)
        if n == 0:
            return

        # --- Step 1: Build the new ordering ---------------------------------
        # Index 0 must always be SHT_NULL; keep it in place.
        # For the remainder, stable-partition: alloc sections first, then non-alloc.
        null_indices = [0]
        alloc_indices = [i for i in range(1, n) if self._shdrs[i].sh_flags & SHF_ALLOC]
        non_alloc_indices = [
            i for i in range(1, n) if not (self._shdrs[i].sh_flags & SHF_ALLOC)
        ]
        new_order = null_indices + alloc_indices + non_alloc_indices

        # If already correctly ordered, nothing to do.
        if new_order == list(range(n)):
            return

        # --- Step 2: Build old→new index mapping ----------------------------
        # old_to_new[old_index] = new_index
        old_to_new: list[int] = [0] * n
        for new_idx, old_idx in enumerate(new_order):
            old_to_new[old_idx] = new_idx

        # --- Step 3: Reorder self._shdrs in memory --------------------------
        reordered = [self._shdrs[old_idx] for old_idx in new_order]

        # --- Step 4: Update sh_link / sh_info in every section header -------
        # sh_link and sh_info often carry section indices (e.g. .rela.X has
        # sh_link = symtab index, sh_info = target section index; .symtab has
        # sh_link = strtab index).  We remap any value that is a valid section
        # index (0 < value < n) through old_to_new.  Index 0 (SHT_NULL) maps
        # to 0 and is harmless to remap.
        updated_shdrs: list[SectionHeader] = []
        for shdr in reordered:
            new_sh_link = old_to_new[shdr.sh_link] if shdr.sh_link < n else shdr.sh_link
            new_sh_info = old_to_new[shdr.sh_info] if shdr.sh_info < n else shdr.sh_info
            if new_sh_link != shdr.sh_link or new_sh_info != shdr.sh_info:
                shdr = SectionHeader(
                    sh_name=shdr.sh_name,
                    sh_type=shdr.sh_type,
                    sh_flags=shdr.sh_flags,
                    sh_addr=shdr.sh_addr,
                    sh_offset=shdr.sh_offset,
                    sh_size=shdr.sh_size,
                    sh_link=new_sh_link,
                    sh_info=new_sh_info,
                    sh_addralign=shdr.sh_addralign,
                    sh_entsize=shdr.sh_entsize,
                )
            updated_shdrs.append(shdr)
        self._shdrs = updated_shdrs

        # --- Step 5: Update e_shstrndx in the ELF header --------------------
        self._ehdr.e_shstrndx = old_to_new[self._ehdr.e_shstrndx]
        self.update_elf_header()

        # --- Step 6: Update st_shndx in .symtab and .dynsym ----------------
        # Each Elf64_Sym entry is 24 bytes:
        #   offset 0: st_name  (uint32)
        #   offset 4: st_info  (uint8)
        #   offset 5: st_other (uint8)
        #   offset 6: st_shndx (uint16)  <-- section index, needs remapping
        #   offset 8: st_value (uint64)
        #   offset 16: st_size (uint64)
        #
        # Values >= 0xff00 are reserved sentinel indices (SHN_UNDEF, SHN_ABS,
        # SHN_COMMON, SHN_XINDEX, etc.) and must NOT be remapped.
        SHN_LORESERVE = 0xFF00
        ELF64_SYM_SIZE = 24
        ST_SHNDX_OFFSET = 6

        for sym_section in self._shdrs:
            if sym_section.sh_type not in (SHT_SYMTAB, SHT_DYNSYM):
                continue
            entry_size = sym_section.sh_entsize or ELF64_SYM_SIZE
            offset = sym_section.sh_offset
            end = offset + sym_section.sh_size
            while offset + entry_size <= end:
                shndx_off = offset + ST_SHNDX_OFFSET
                st_shndx = struct.unpack_from("<H", self._data, shndx_off)[0]
                if st_shndx < SHN_LORESERVE and st_shndx < n:
                    new_shndx = old_to_new[st_shndx]
                    if new_shndx != st_shndx:
                        struct.pack_into("<H", self._data, shndx_off, new_shndx)
                offset += entry_size

        # --- Step 7: Rewrite the section header table in place --------------
        # add_section() placed the SHT at the end of the file; we overwrite it
        # in the same location with the reordered (and cross-reference-updated)
        # headers.  No change to e_shoff or e_shnum is needed.
        sht_offset = self._ehdr.e_shoff
        for i, shdr in enumerate(self._shdrs):
            offset = sht_offset + i * self._ehdr.e_shentsize
            shdr.write_to(self._data, offset)

        # --- Step 8: Re-parse section names so find_section() stays correct -
        self._section_names = self._parse_section_names()

        self._modifications.append(
            Modification(
                operation="reorder_section_headers",
                file_offset=sht_offset,
                size=n * self._ehdr.e_shentsize,
                description="reorder section headers: alloc sections before non-alloc",
            )
        )
