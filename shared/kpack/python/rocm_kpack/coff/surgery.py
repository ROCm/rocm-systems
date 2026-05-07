"""
High-level PE/COFF surgery interface.

The CoffSurgery class provides a clean abstraction for PE/COFF binary modifications.
It handles the complexity of maintaining PE invariants while exposing
simple high-level operations.

Design principles:
- Parse once, modify in memory, write once
- Track all modifications for verification
- Fail fast with clear error messages
- Compose well with other utilities

Key differences from ELF:
- No program headers (sections define memory layout via characteristics)
- Uses ImageBase + RVA for virtual addresses
- Base relocations instead of RELA sections
- Section names limited to 8 characters
"""

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .types import (
    DosHeader,
    CoffHeader,
    OptionalHeader64,
    SectionHeader,
    DataDirectory,
    BaseRelocationBlock,
    BaseRelocationEntry,
    PE_SIGNATURE,
    DOS_HEADER_SIZE,
    COFF_HEADER_SIZE,
    SECTION_HEADER_SIZE,
    DATA_DIRECTORY_SIZE,
    IMAGE_DIRECTORY_ENTRY_BASERELOC,
    IMAGE_REL_BASED_DIR64,
    IMAGE_REL_BASED_ABSOLUTE,
    IMAGE_SCN_MEM_READ,
    IMAGE_SCN_CNT_INITIALIZED_DATA,
    round_up_to_alignment,
    section_name_to_bytes,
)


@dataclass
class SectionInfo:
    """Information about a section, combining header with derived data."""

    index: int
    name: str
    header: SectionHeader

    @property
    def rva(self) -> int:
        """Relative virtual address."""
        return self.header.VirtualAddress

    @property
    def virtual_size(self) -> int:
        """Size in memory."""
        return self.header.VirtualSize

    @property
    def file_offset(self) -> int:
        """File offset of raw data."""
        return self.header.PointerToRawData

    @property
    def raw_size(self) -> int:
        """Size in file."""
        return self.header.SizeOfRawData


@dataclass
class Modification:
    """Record of a modification made to the PE binary."""

    operation: str  # e.g., "write_bytes", "update_section"
    file_offset: int
    size: int
    description: str


@dataclass
class RelocationInfo:
    """Information about a found base relocation entry."""

    block_rva: int  # Page RVA for the block
    entry_offset: int  # File offset of the entry
    target_rva: int  # Full RVA being relocated
    entry: BaseRelocationEntry


class CoffSurgery:
    """High-level interface for PE/COFF binary modifications.

    Usage:
        surgery = CoffSurgery.load(Path("foo.dll"))

        # Query operations
        section = surgery.find_section(".hip_fat")
        offset = surgery.rva_to_file_offset(0x1000)

        # Modification operations
        surgery.write_bytes_at_offset(0x1000, b"\\x00" * 100)

        # Save and verify
        surgery.save(Path("foo_modified.dll"))
    """

    # Reasonable limits for PE structures to prevent DoS from malformed files
    MAX_NUMBER_OF_SECTIONS = 256
    MAX_NUMBER_OF_DATA_DIRECTORIES = 64

    def __init__(
        self,
        data: bytearray,
        path: Path | None = None,
    ):
        """Initialize with binary data.

        Prefer using CoffSurgery.load() for most use cases.

        Args:
            data: Mutable PE binary data
            path: Original file path (for error messages)
        """
        self._data = data
        self._path = path
        self._modifications: list[Modification] = []

        # Parse structures
        self._dos_hdr = DosHeader.from_bytes(data)
        self._pe_offset = self._dos_hdr.e_lfanew

        # Validate PE offset is within bounds
        if self._pe_offset < 0 or self._pe_offset + 4 > len(data):
            raise ValueError(
                f"Invalid PE header offset {self._pe_offset:#x}: "
                f"must be within file bounds (0 to {len(data) - 4})"
            )

        # Verify PE signature
        pe_sig = bytes(data[self._pe_offset : self._pe_offset + 4])
        if pe_sig != PE_SIGNATURE:
            raise ValueError(f"Invalid PE signature: {pe_sig!r}")

        # Parse COFF header (after PE signature)
        coff_offset = self._pe_offset + 4
        self._coff_hdr = CoffHeader.from_bytes(data, coff_offset)

        # Parse optional header
        opt_offset = coff_offset + COFF_HEADER_SIZE
        self._opt_hdr = OptionalHeader64.from_bytes(data, opt_offset)

        # Parse data directories
        data_dir_offset = opt_offset + self._opt_hdr.SIZE
        self._data_dirs = self._parse_data_directories(data_dir_offset)

        # Parse section headers
        section_offset = opt_offset + self._coff_hdr.SizeOfOptionalHeader
        self._sections = self._parse_sections(section_offset)

    @classmethod
    def load(cls, path: Path) -> "CoffSurgery":
        """Load a PE binary from file.

        Args:
            path: Path to PE binary

        Returns:
            CoffSurgery instance ready for modifications
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
    def dos_header(self) -> DosHeader:
        """DOS header."""
        return self._dos_hdr

    @property
    def coff_header(self) -> CoffHeader:
        """COFF file header."""
        return self._coff_hdr

    @property
    def optional_header(self) -> OptionalHeader64:
        """PE32+ optional header."""
        return self._opt_hdr

    @property
    def modifications(self) -> list[Modification]:
        """List of modifications made."""
        return self._modifications

    @property
    def image_base(self) -> int:
        """Preferred load address."""
        return self._opt_hdr.ImageBase

    @property
    def file_alignment(self) -> int:
        """File alignment for raw data."""
        return self._opt_hdr.FileAlignment

    @property
    def section_alignment(self) -> int:
        """Section alignment in memory."""
        return self._opt_hdr.SectionAlignment

    @property
    def is_dll(self) -> bool:
        """Check if this is a DLL."""
        return self._coff_hdr.is_dll

    @property
    def has_aslr(self) -> bool:
        """Check if ASLR is enabled."""
        return self._opt_hdr.has_aslr

    # =========================================================================
    # Parsing (internal)
    # =========================================================================

    def _parse_data_directories(self, offset: int) -> list[DataDirectory]:
        """Parse all data directories."""
        num_dirs = self._opt_hdr.NumberOfRvaAndSizes
        if num_dirs > self.MAX_NUMBER_OF_DATA_DIRECTORIES:
            raise ValueError(
                f"NumberOfRvaAndSizes ({num_dirs}) exceeds maximum "
                f"({self.MAX_NUMBER_OF_DATA_DIRECTORIES})"
            )
        dirs = []
        for i in range(num_dirs):
            dir_offset = offset + i * DATA_DIRECTORY_SIZE
            dirs.append(DataDirectory.from_bytes(self._data, dir_offset))
        return dirs

    def _parse_sections(self, offset: int) -> list[SectionHeader]:
        """Parse all section headers."""
        num_sections = self._coff_hdr.NumberOfSections
        if num_sections > self.MAX_NUMBER_OF_SECTIONS:
            raise ValueError(
                f"NumberOfSections ({num_sections}) exceeds maximum "
                f"({self.MAX_NUMBER_OF_SECTIONS})"
            )
        sections = []
        for i in range(num_sections):
            sect_offset = offset + i * SECTION_HEADER_SIZE
            sections.append(SectionHeader.from_bytes(self._data, sect_offset))
        return sections

    # =========================================================================
    # Query Operations
    # =========================================================================

    def find_section(self, name: str) -> SectionInfo | None:
        """Find a section by name.

        Args:
            name: Section name (e.g., ".hip_fat", ".hipFatB")
                  Can be the full name or truncated 8-char version.

        Returns:
            SectionInfo if found, None otherwise
        """
        # Normalize name for comparison
        search_name = name[:8] if len(name) > 8 else name

        for idx, shdr in enumerate(self._sections):
            if shdr.name_str == search_name:
                return SectionInfo(
                    index=idx,
                    name=shdr.name_str,
                    header=shdr,
                )
        return None

    def get_section_by_index(self, index: int) -> SectionInfo | None:
        """Get section by index."""
        if 0 <= index < len(self._sections):
            shdr = self._sections[index]
            return SectionInfo(
                index=index,
                name=shdr.name_str,
                header=shdr,
            )
        return None

    def iter_sections(self) -> Iterator[SectionInfo]:
        """Iterate over all sections."""
        for idx, shdr in enumerate(self._sections):
            yield SectionInfo(
                index=idx,
                name=shdr.name_str,
                header=shdr,
            )

    def get_section_content(self, section: SectionInfo | str) -> bytes:
        """Get content of a section.

        Args:
            section: SectionInfo or section name

        Returns:
            Section content bytes

        Raises:
            ValueError: If section not found or has no raw data
        """
        if isinstance(section, str):
            info = self.find_section(section)
            if info is None:
                raise ValueError(f"Section not found: {section}")
            section = info

        if section.header.SizeOfRawData == 0:
            return b""

        start = section.header.PointerToRawData
        end = start + section.header.SizeOfRawData
        return bytes(self._data[start:end])

    def get_data_directory(self, index: int) -> DataDirectory | None:
        """Get a data directory by index."""
        if 0 <= index < len(self._data_dirs):
            return self._data_dirs[index]
        return None

    # =========================================================================
    # Address Conversion
    # =========================================================================

    def rva_to_file_offset(self, rva: int) -> int | None:
        """Convert RVA to file offset using section table.

        Args:
            rva: Relative virtual address

        Returns:
            File offset if RVA is in a section with raw data, None otherwise.
            Returns None for BSS/uninitialized sections (SizeOfRawData == 0).
        """
        for shdr in self._sections:
            if shdr.contains_rva(rva):
                # BSS sections have no file data
                if shdr.SizeOfRawData == 0:
                    return None
                # Calculate offset within section
                section_offset = rva - shdr.VirtualAddress
                # Ensure we don't read beyond the raw data
                if section_offset >= shdr.SizeOfRawData:
                    return None
                return shdr.PointerToRawData + section_offset
        return None

    def file_offset_to_rva(self, offset: int) -> int | None:
        """Convert file offset to RVA.

        Args:
            offset: File offset

        Returns:
            RVA if offset is in a section, None otherwise
        """
        for shdr in self._sections:
            if shdr.contains_file_offset(offset):
                # Calculate offset within section
                section_offset = offset - shdr.PointerToRawData
                return shdr.VirtualAddress + section_offset
        return None

    def rva_to_va(self, rva: int) -> int:
        """Convert RVA to virtual address (ImageBase + RVA)."""
        return self.image_base + rva

    def va_to_rva(self, va: int) -> int:
        """Convert virtual address to RVA (VA - ImageBase)."""
        return va - self.image_base

    def va_to_file_offset(self, va: int) -> int | None:
        """Convert virtual address to file offset."""
        return self.rva_to_file_offset(self.va_to_rva(va))

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

    def write_bytes_at_rva(self, rva: int, data: bytes, description: str = "") -> None:
        """Write bytes at an RVA.

        Args:
            rva: Relative virtual address
            data: Bytes to write
            description: Human-readable description for tracking

        Raises:
            ValueError: If RVA is not in any section
        """
        offset = self.rva_to_file_offset(rva)
        if offset is None:
            raise ValueError(f"RVA 0x{rva:x} not in any section")
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
    # Pointer Operations
    # =========================================================================

    def read_pointer_at_rva(self, rva: int) -> int:
        """Read an 8-byte pointer at an RVA.

        Args:
            rva: Relative virtual address

        Returns:
            Pointer value (little-endian uint64)
        """
        offset = self.rva_to_file_offset(rva)
        if offset is None:
            raise ValueError(f"RVA 0x{rva:x} not in any section")

        return struct.unpack_from("<Q", self._data, offset)[0]

    def write_pointer_at_rva(self, rva: int, value: int, description: str = "") -> None:
        """Write an 8-byte pointer at an RVA.

        Args:
            rva: Relative virtual address
            value: Pointer value to write (as VA, not RVA)
            description: Human-readable description
        """
        data = struct.pack("<Q", value)
        self.write_bytes_at_rva(
            rva,
            data,
            description or f"write pointer 0x{value:x} at RVA 0x{rva:x}",
        )

    def read_u32_at_rva(self, rva: int) -> int:
        """Read a 4-byte unsigned integer at an RVA."""
        offset = self.rva_to_file_offset(rva)
        if offset is None:
            raise ValueError(f"RVA 0x{rva:x} not in any section")
        return struct.unpack_from("<I", self._data, offset)[0]

    def write_u32_at_rva(self, rva: int, value: int, description: str = "") -> None:
        """Write a 4-byte unsigned integer at an RVA."""
        data = struct.pack("<I", value)
        self.write_bytes_at_rva(
            rva,
            data,
            description or f"write u32 0x{value:x} at RVA 0x{rva:x}",
        )

    # =========================================================================
    # Header Update Operations
    # =========================================================================

    def _get_optional_header_offset(self) -> int:
        """Get file offset of optional header."""
        return self._pe_offset + 4 + COFF_HEADER_SIZE

    def _get_section_headers_offset(self) -> int:
        """Get file offset of first section header."""
        return self._get_optional_header_offset() + self._coff_hdr.SizeOfOptionalHeader

    def update_optional_header(self) -> None:
        """Write current optional header to binary."""
        offset = self._get_optional_header_offset()
        self._opt_hdr.write_to(self._data, offset)

        # Also write data directories
        data_dir_offset = offset + self._opt_hdr.SIZE
        for i, dd in enumerate(self._data_dirs):
            dd.write_to(self._data, data_dir_offset + i * DATA_DIRECTORY_SIZE)

        self._modifications.append(
            Modification(
                operation="update_optional_header",
                file_offset=offset,
                size=self._coff_hdr.SizeOfOptionalHeader,
                description="update optional header",
            )
        )

    def update_coff_header(self) -> None:
        """Write current COFF header to binary."""
        offset = self._pe_offset + 4
        self._coff_hdr.write_to(self._data, offset)
        self._modifications.append(
            Modification(
                operation="update_coff_header",
                file_offset=offset,
                size=COFF_HEADER_SIZE,
                description="update COFF header",
            )
        )

    def update_section_header(self, index: int, shdr: SectionHeader) -> None:
        """Update a section header in the binary.

        Args:
            index: Section header index
            shdr: New section header value
        """
        if index < 0 or index >= self._coff_hdr.NumberOfSections:
            raise ValueError(f"Invalid section header index: {index}")

        offset = self._get_section_headers_offset() + index * SECTION_HEADER_SIZE
        shdr.write_to(self._data, offset)
        self._sections[index] = shdr
        self._modifications.append(
            Modification(
                operation="update_section_header",
                file_offset=offset,
                size=SECTION_HEADER_SIZE,
                description=f"update section header {index} ({shdr.name_str})",
            )
        )

    # =========================================================================
    # Base Relocation Operations
    # =========================================================================

    def iter_base_relocations(self) -> Iterator[RelocationInfo]:
        """Iterate over all base relocation entries.

        Yields:
            RelocationInfo for each non-ABSOLUTE entry
        """
        reloc_dir = self.get_data_directory(IMAGE_DIRECTORY_ENTRY_BASERELOC)
        if reloc_dir is None or not reloc_dir.is_present:
            return

        reloc_offset = self.rva_to_file_offset(reloc_dir.VirtualAddress)
        if reloc_offset is None:
            return

        end_offset = reloc_offset + reloc_dir.Size
        current = reloc_offset

        while current < end_offset:
            block = BaseRelocationBlock.from_bytes(self._data, current)
            if block.BlockSize == 0:
                break
            # Validate BlockSize to prevent infinite loop or DoS
            if block.BlockSize < BaseRelocationBlock.SIZE:
                break  # Invalid block, stop iteration

            # Iterate entries in this block
            entry_offset = current + BaseRelocationBlock.SIZE
            for i in range(block.num_entries):
                entry = BaseRelocationEntry.from_bytes(self._data, entry_offset)

                if not entry.is_absolute:  # Skip padding entries
                    target_rva = block.PageRVA + entry.offset
                    yield RelocationInfo(
                        block_rva=block.PageRVA,
                        entry_offset=entry_offset,
                        target_rva=target_rva,
                        entry=entry,
                    )

                entry_offset += BaseRelocationEntry.SIZE

            current += block.BlockSize

    def find_relocation_at_rva(self, target_rva: int) -> RelocationInfo | None:
        """Find a base relocation targeting a specific RVA.

        Args:
            target_rva: RVA the relocation targets

        Returns:
            RelocationInfo if found, None otherwise
        """
        for reloc in self.iter_base_relocations():
            if reloc.target_rva == target_rva:
                return reloc
        return None

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
            mode: File mode to set. If None, does not set mode.
        """
        path.write_bytes(self._data)
        if mode is not None:
            os.chmod(path, mode)

    # =========================================================================
    # Layout Information
    # =========================================================================

    def get_max_section_end_offset(self) -> int:
        """Get the maximum file offset used by any section."""
        max_offset = 0
        for shdr in self._sections:
            if shdr.SizeOfRawData > 0:
                end = shdr.PointerToRawData + shdr.SizeOfRawData
                max_offset = max(max_offset, end)
        return max_offset

    def get_max_section_end_rva(self) -> int:
        """Get the maximum RVA used by any section (for SizeOfImage calculation)."""
        max_rva = 0
        for shdr in self._sections:
            end = shdr.VirtualAddress + shdr.VirtualSize
            max_rva = max(max_rva, end)
        return max_rva

    # =========================================================================
    # Section Addition
    # =========================================================================

    def add_section(
        self,
        name: str,
        content: bytes,
        characteristics: int = IMAGE_SCN_MEM_READ | IMAGE_SCN_CNT_INITIALIZED_DATA,
    ) -> "AddSectionResult":
        """Add a new section to the PE binary.

        This appends the section content to the end of the file and adds
        a new section header. The new section gets the next available RVA.

        Steps:
        1. Check space exists for new section header
        2. Append section content to end of file (aligned to FileAlignment)
        3. Calculate RVA (next available, aligned to SectionAlignment)
        4. Add new section header entry
        5. Update NumberOfSections and SizeOfImage

        Args:
            name: Section name (max 8 chars, will be truncated)
            content: Section content bytes
            characteristics: IMAGE_SCN_* flags (default: readable initialized data)

        Returns:
            AddSectionResult with section details

        Raises:
            ValueError: If section already exists or no space for new header
        """
        # Check if section already exists
        truncated_name = name[:8]
        if self.find_section(truncated_name) is not None:
            raise ValueError(f"Section '{truncated_name}' already exists")

        # Check space for new section header
        headers_end = self._get_section_headers_offset() + (
            self._coff_hdr.NumberOfSections * SECTION_HEADER_SIZE
        )
        new_header_end = headers_end + SECTION_HEADER_SIZE

        if new_header_end > self._opt_hdr.SizeOfHeaders:
            raise ValueError(
                f"No space for new section header. "
                f"Headers end at 0x{new_header_end:x}, "
                f"SizeOfHeaders is 0x{self._opt_hdr.SizeOfHeaders:x}. "
                f"Need {new_header_end - self._opt_hdr.SizeOfHeaders} more bytes."
            )

        # Step 1: Calculate file offset for new section (aligned to FileAlignment)
        current_end = self.get_max_section_end_offset()
        file_offset = round_up_to_alignment(current_end, self.file_alignment)

        # Step 2: Calculate RVA for new section (aligned to SectionAlignment)
        current_max_rva = self.get_max_section_end_rva()
        rva = round_up_to_alignment(current_max_rva, self.section_alignment)

        # Step 3: Prepare content (pad to FileAlignment)
        raw_size = round_up_to_alignment(len(content), self.file_alignment)
        padded_content = content + (b"\x00" * (raw_size - len(content)))

        # Step 4: Extend file with padding + content
        padding_needed = file_offset - len(self._data)
        if padding_needed > 0:
            self._data.extend(b"\x00" * padding_needed)
        self._data.extend(padded_content)

        # Step 5: Create section header
        new_shdr = SectionHeader(
            Name=section_name_to_bytes(truncated_name),
            VirtualSize=len(content),  # Actual size, not padded
            VirtualAddress=rva,
            SizeOfRawData=raw_size,
            PointerToRawData=file_offset,
            PointerToRelocations=0,
            PointerToLinenumbers=0,
            NumberOfRelocations=0,
            NumberOfLinenumbers=0,
            Characteristics=characteristics,
        )

        # Step 6: Write section header to file
        new_shdr.write_to(self._data, headers_end)
        self._sections.append(new_shdr)

        # Step 7: Update COFF header (NumberOfSections)
        self._coff_hdr = CoffHeader(
            Machine=self._coff_hdr.Machine,
            NumberOfSections=self._coff_hdr.NumberOfSections + 1,
            TimeDateStamp=self._coff_hdr.TimeDateStamp,
            PointerToSymbolTable=self._coff_hdr.PointerToSymbolTable,
            NumberOfSymbols=self._coff_hdr.NumberOfSymbols,
            SizeOfOptionalHeader=self._coff_hdr.SizeOfOptionalHeader,
            Characteristics=self._coff_hdr.Characteristics,
        )
        self.update_coff_header()

        # Step 8: Update SizeOfImage in optional header
        # SizeOfImage must include the new section, aligned to SectionAlignment
        new_max_rva = rva + round_up_to_alignment(len(content), self.section_alignment)
        self._opt_hdr.SizeOfImage = new_max_rva
        self.update_optional_header()

        self._modifications.append(
            Modification(
                operation="add_section",
                file_offset=file_offset,
                size=raw_size,
                description=f"add section '{truncated_name}' at RVA 0x{rva:x}",
            )
        )

        return AddSectionResult(
            success=True,
            section_index=len(self._sections) - 1,
            rva=rva,
            file_offset=file_offset,
            size=len(content),
        )


@dataclass
class AddSectionResult:
    """Result of adding a section."""

    success: bool
    section_index: int
    rva: int
    file_offset: int
    size: int
    error: str | None = None
