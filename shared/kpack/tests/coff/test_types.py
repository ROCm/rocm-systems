"""Tests for PE/COFF type definitions and parsing."""

import pytest
import struct

from rocm_kpack.coff.types import (
    DosHeader,
    CoffHeader,
    OptionalHeader64,
    SectionHeader,
    DataDirectory,
    BaseRelocationBlock,
    BaseRelocationEntry,
    DOS_MAGIC,
    PE_SIGNATURE,
    IMAGE_NT_OPTIONAL_HDR64_MAGIC,
    IMAGE_FILE_DLL,
    IMAGE_FILE_EXECUTABLE_IMAGE,
    IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE,
    IMAGE_SCN_CNT_CODE,
    IMAGE_SCN_CNT_INITIALIZED_DATA,
    IMAGE_SCN_CNT_UNINITIALIZED_DATA,
    IMAGE_SCN_MEM_READ,
    IMAGE_SCN_MEM_WRITE,
    IMAGE_SCN_MEM_EXECUTE,
    IMAGE_REL_BASED_ABSOLUTE,
    IMAGE_REL_BASED_DIR64,
    IMAGE_REL_BASED_HIGHLOW,
    PAGE_SIZE,
    round_up_to_alignment,
    round_down_to_alignment,
    round_up_to_page,
    round_down_to_page,
    section_name_to_bytes,
)


class TestDosHeader:
    """Tests for DOS header parsing."""

    def test_parse_valid_header(self):
        """Test parsing a valid DOS header."""
        # Create minimal valid DOS header
        data = bytearray(64)
        struct.pack_into("<H", data, 0, DOS_MAGIC)  # e_magic = "MZ"
        struct.pack_into("<I", data, 60, 0x80)  # e_lfanew = 0x80

        header = DosHeader.from_bytes(data)
        assert header.e_magic == DOS_MAGIC
        assert header.e_lfanew == 0x80

    def test_parse_bad_magic_raises(self):
        """Test that invalid magic raises ValueError."""
        data = bytearray(64)
        struct.pack_into("<H", data, 0, 0x1234)  # Bad magic

        with pytest.raises(ValueError, match="Not a DOS/PE file"):
            DosHeader.from_bytes(data)

    def test_roundtrip_serialization(self):
        """Test that serialization round-trips correctly."""
        data = bytearray(64)
        struct.pack_into("<H", data, 0, DOS_MAGIC)
        struct.pack_into("<I", data, 60, 0x100)

        header = DosHeader.from_bytes(data)
        serialized = header.to_bytes()

        # Parse again and verify
        header2 = DosHeader.from_bytes(serialized)
        assert header2.e_magic == header.e_magic
        assert header2.e_lfanew == header.e_lfanew

    def test_write_to_buffer(self):
        """Test writing header to mutable buffer."""
        data = bytearray(64)
        struct.pack_into("<H", data, 0, DOS_MAGIC)
        struct.pack_into("<I", data, 60, 0x80)

        header = DosHeader.from_bytes(data)

        # Modify and write back
        output = bytearray(128)
        header.write_to(output, 32)

        # Verify at offset
        written = DosHeader.from_bytes(output, 32)
        assert written.e_magic == DOS_MAGIC
        assert written.e_lfanew == 0x80

    def test_data_too_short_raises(self):
        """Test that short data raises ValueError."""
        data = bytearray(32)  # Too short

        with pytest.raises(ValueError, match="Data too short"):
            DosHeader.from_bytes(data)


class TestCoffHeader:
    """Tests for COFF header parsing."""

    def test_parse_valid_header(self):
        """Test parsing a valid COFF header."""
        data = bytearray(20)
        struct.pack_into(
            "<HHIIIHH",
            data,
            0,
            0x8664,  # Machine = AMD64
            5,  # NumberOfSections
            0x12345678,  # TimeDateStamp
            0,  # PointerToSymbolTable
            0,  # NumberOfSymbols
            240,  # SizeOfOptionalHeader
            IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_DLL,  # Characteristics
        )

        header = CoffHeader.from_bytes(data)
        assert header.Machine == 0x8664
        assert header.NumberOfSections == 5
        assert header.SizeOfOptionalHeader == 240

    def test_is_dll_property(self):
        """Test is_dll property."""
        data = bytearray(20)
        struct.pack_into("<HHIIIHH", data, 0, 0x8664, 1, 0, 0, 0, 240, IMAGE_FILE_DLL)

        header = CoffHeader.from_bytes(data)
        assert header.is_dll is True

        # Without DLL flag
        struct.pack_into(
            "<HHIIIHH", data, 0, 0x8664, 1, 0, 0, 0, 240, IMAGE_FILE_EXECUTABLE_IMAGE
        )
        header = CoffHeader.from_bytes(data)
        assert header.is_dll is False

    def test_is_executable_property(self):
        """Test is_executable property."""
        data = bytearray(20)
        struct.pack_into(
            "<HHIIIHH", data, 0, 0x8664, 1, 0, 0, 0, 240, IMAGE_FILE_EXECUTABLE_IMAGE
        )

        header = CoffHeader.from_bytes(data)
        assert header.is_executable is True

    def test_roundtrip_serialization(self):
        """Test that serialization round-trips correctly."""
        data = bytearray(20)
        struct.pack_into("<HHIIIHH", data, 0, 0x8664, 3, 0x11111111, 0, 0, 240, 0x22)

        header = CoffHeader.from_bytes(data)
        serialized = header.to_bytes()

        header2 = CoffHeader.from_bytes(serialized)
        assert header2.Machine == header.Machine
        assert header2.NumberOfSections == header.NumberOfSections
        assert header2.TimeDateStamp == header.TimeDateStamp


class TestOptionalHeader64:
    """Tests for PE32+ optional header parsing."""

    def _make_valid_opt_header(self) -> bytearray:
        """Create a minimal valid PE32+ optional header."""
        data = bytearray(112)
        # Pack the header fields
        struct.pack_into(
            "<HBBIIIIIQIIHHHHHH" "IIIIHHQQQQII",
            data,
            0,
            IMAGE_NT_OPTIONAL_HDR64_MAGIC,  # Magic
            14,  # MajorLinkerVersion
            0,  # MinorLinkerVersion
            0x1000,  # SizeOfCode
            0x2000,  # SizeOfInitializedData
            0,  # SizeOfUninitializedData
            0x1000,  # AddressOfEntryPoint
            0x1000,  # BaseOfCode
            0x140000000,  # ImageBase
            0x1000,  # SectionAlignment
            0x200,  # FileAlignment
            6,  # MajorOperatingSystemVersion
            0,  # MinorOperatingSystemVersion
            0,  # MajorImageVersion
            0,  # MinorImageVersion
            6,  # MajorSubsystemVersion
            0,  # MinorSubsystemVersion
            0,  # Win32VersionValue
            0x10000,  # SizeOfImage
            0x200,  # SizeOfHeaders
            0,  # CheckSum
            3,  # Subsystem (CONSOLE)
            IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE,  # DllCharacteristics (ASLR)
            0x100000,  # SizeOfStackReserve
            0x1000,  # SizeOfStackCommit
            0x100000,  # SizeOfHeapReserve
            0x1000,  # SizeOfHeapCommit
            0,  # LoaderFlags
            16,  # NumberOfRvaAndSizes
        )
        return data

    def test_parse_valid_header(self):
        """Test parsing a valid PE32+ optional header."""
        data = self._make_valid_opt_header()

        header = OptionalHeader64.from_bytes(data)
        assert header.Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC
        assert header.ImageBase == 0x140000000
        assert header.SectionAlignment == 0x1000
        assert header.FileAlignment == 0x200
        assert header.NumberOfRvaAndSizes == 16

    def test_not_pe32plus_raises(self):
        """Test that PE32 (not PE32+) raises ValueError."""
        data = self._make_valid_opt_header()
        struct.pack_into("<H", data, 0, 0x10B)  # PE32 magic instead of PE32+

        with pytest.raises(ValueError, match="Not a PE32\\+ file"):
            OptionalHeader64.from_bytes(data)

    def test_has_aslr_property(self):
        """Test has_aslr property."""
        data = self._make_valid_opt_header()

        header = OptionalHeader64.from_bytes(data)
        assert header.has_aslr is True

        # Without ASLR flag
        struct.pack_into("<H", data, 70, 0)  # Clear DllCharacteristics
        header = OptionalHeader64.from_bytes(data)
        assert header.has_aslr is False

    def test_roundtrip_serialization(self):
        """Test that serialization round-trips correctly."""
        data = self._make_valid_opt_header()

        header = OptionalHeader64.from_bytes(data)
        serialized = header.to_bytes()

        header2 = OptionalHeader64.from_bytes(serialized)
        assert header2.Magic == header.Magic
        assert header2.ImageBase == header.ImageBase
        assert header2.SectionAlignment == header.SectionAlignment


class TestSectionHeader:
    """Tests for PE section header parsing."""

    def _make_section_header(
        self,
        name: bytes = b".text\x00\x00\x00",
        vsize: int = 0x1000,
        vaddr: int = 0x1000,
        raw_size: int = 0x200,
        raw_ptr: int = 0x200,
        characteristics: int = IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE,
    ) -> bytearray:
        """Create a section header."""
        data = bytearray(40)
        struct.pack_into(
            "<8sIIIIIIHHI",
            data,
            0,
            name,
            vsize,
            vaddr,
            raw_size,
            raw_ptr,
            0,  # PointerToRelocations
            0,  # PointerToLinenumbers
            0,  # NumberOfRelocations
            0,  # NumberOfLinenumbers
            characteristics,
        )
        return data

    def test_name_str_property(self):
        """Test name_str property with null-terminated name."""
        data = self._make_section_header(name=b".text\x00\x00\x00")
        header = SectionHeader.from_bytes(data)
        assert header.name_str == ".text"

    def test_name_str_with_null_padding(self):
        """Test name_str with shorter name padded with nulls."""
        data = self._make_section_header(name=b".bss\x00\x00\x00\x00")
        header = SectionHeader.from_bytes(data)
        assert header.name_str == ".bss"

    def test_name_str_full_8_chars(self):
        """Test name_str with exactly 8 characters (no null terminator)."""
        data = self._make_section_header(name=b"12345678")
        header = SectionHeader.from_bytes(data)
        assert header.name_str == "12345678"

    def test_contains_rva(self):
        """Test contains_rva method."""
        data = self._make_section_header(vaddr=0x1000, vsize=0x500)
        header = SectionHeader.from_bytes(data)

        # Inside section
        assert header.contains_rva(0x1000) is True
        assert header.contains_rva(0x1200) is True
        assert header.contains_rva(0x14FF) is True

        # Outside section
        assert header.contains_rva(0x0FFF) is False
        assert header.contains_rva(0x1500) is False

    def test_contains_file_offset(self):
        """Test contains_file_offset method."""
        data = self._make_section_header(raw_ptr=0x200, raw_size=0x400)
        header = SectionHeader.from_bytes(data)

        # Inside section
        assert header.contains_file_offset(0x200) is True
        assert header.contains_file_offset(0x400) is True
        assert header.contains_file_offset(0x5FF) is True

        # Outside section
        assert header.contains_file_offset(0x1FF) is False
        assert header.contains_file_offset(0x600) is False

    def test_contains_file_offset_bss_section(self):
        """Test that BSS sections (raw_size=0) contain no file offsets."""
        data = self._make_section_header(raw_ptr=0, raw_size=0)
        header = SectionHeader.from_bytes(data)

        assert header.contains_file_offset(0) is False
        assert header.contains_file_offset(0x1000) is False

    def test_characteristic_properties(self):
        """Test section characteristic properties."""
        # Code section
        data = self._make_section_header(
            characteristics=IMAGE_SCN_CNT_CODE
            | IMAGE_SCN_MEM_EXECUTE
            | IMAGE_SCN_MEM_READ
        )
        header = SectionHeader.from_bytes(data)
        assert header.is_code is True
        assert header.is_executable is True
        assert header.is_readable is True
        assert header.is_writable is False
        assert header.is_initialized_data is False

        # Data section
        data = self._make_section_header(
            characteristics=IMAGE_SCN_CNT_INITIALIZED_DATA
            | IMAGE_SCN_MEM_READ
            | IMAGE_SCN_MEM_WRITE
        )
        header = SectionHeader.from_bytes(data)
        assert header.is_code is False
        assert header.is_initialized_data is True
        assert header.is_readable is True
        assert header.is_writable is True

        # BSS section
        data = self._make_section_header(
            characteristics=IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ
        )
        header = SectionHeader.from_bytes(data)
        assert header.is_uninitialized_data is True

    def test_end_rva_property(self):
        """Test end_rva property."""
        data = self._make_section_header(vaddr=0x2000, vsize=0x1500)
        header = SectionHeader.from_bytes(data)
        assert header.end_rva == 0x3500

    def test_end_file_offset_property(self):
        """Test end_file_offset property."""
        data = self._make_section_header(raw_ptr=0x400, raw_size=0x600)
        header = SectionHeader.from_bytes(data)
        assert header.end_file_offset == 0xA00

    def test_roundtrip_serialization(self):
        """Test that serialization round-trips correctly."""
        data = self._make_section_header(
            name=b".hip_fat",
            vsize=0x50000,
            vaddr=0x5000,
            raw_size=0x50000,
            raw_ptr=0x1000,
        )

        header = SectionHeader.from_bytes(data)
        serialized = header.to_bytes()

        header2 = SectionHeader.from_bytes(serialized)
        assert header2.name_str == header.name_str
        assert header2.VirtualSize == header.VirtualSize
        assert header2.VirtualAddress == header.VirtualAddress


class TestDataDirectory:
    """Tests for data directory parsing."""

    def test_parse_valid_directory(self):
        """Test parsing a valid data directory."""
        data = bytearray(8)
        struct.pack_into("<II", data, 0, 0x3000, 0x500)

        dd = DataDirectory.from_bytes(data)
        assert dd.VirtualAddress == 0x3000
        assert dd.Size == 0x500

    def test_is_present_property(self):
        """Test is_present property."""
        # Present (non-zero VA)
        data = bytearray(8)
        struct.pack_into("<II", data, 0, 0x1000, 0x100)
        dd = DataDirectory.from_bytes(data)
        assert dd.is_present is True

        # Present (non-zero size, zero VA - unusual but valid)
        struct.pack_into("<II", data, 0, 0, 0x100)
        dd = DataDirectory.from_bytes(data)
        assert dd.is_present is True

        # Not present
        struct.pack_into("<II", data, 0, 0, 0)
        dd = DataDirectory.from_bytes(data)
        assert dd.is_present is False


class TestBaseRelocationBlock:
    """Tests for base relocation block parsing."""

    def test_parse_valid_block(self):
        """Test parsing a valid relocation block."""
        data = bytearray(8)
        struct.pack_into("<II", data, 0, 0x1000, 24)  # PageRVA, BlockSize

        block = BaseRelocationBlock.from_bytes(data)
        assert block.PageRVA == 0x1000
        assert block.BlockSize == 24

    def test_num_entries_property(self):
        """Test num_entries property calculation."""
        data = bytearray(8)
        # BlockSize = 8 (header) + 8*2 (entries) = 24
        struct.pack_into("<II", data, 0, 0x1000, 24)

        block = BaseRelocationBlock.from_bytes(data)
        assert block.num_entries == 8

        # BlockSize = header only
        struct.pack_into("<II", data, 0, 0x1000, 8)
        block = BaseRelocationBlock.from_bytes(data)
        assert block.num_entries == 0


class TestBaseRelocationEntry:
    """Tests for base relocation entry parsing."""

    def test_reloc_type_extraction(self):
        """Test relocation type extraction from high 4 bits."""
        data = bytearray(2)

        # DIR64 type (10 << 12) + offset 0x123
        struct.pack_into("<H", data, 0, (IMAGE_REL_BASED_DIR64 << 12) | 0x123)
        entry = BaseRelocationEntry.from_bytes(data)
        assert entry.reloc_type == IMAGE_REL_BASED_DIR64

        # ABSOLUTE type (0 << 12) + offset 0
        struct.pack_into("<H", data, 0, 0)
        entry = BaseRelocationEntry.from_bytes(data)
        assert entry.reloc_type == IMAGE_REL_BASED_ABSOLUTE

    def test_offset_extraction(self):
        """Test offset extraction from low 12 bits."""
        data = bytearray(2)

        # Type 10 + offset 0xABC
        struct.pack_into("<H", data, 0, (10 << 12) | 0xABC)
        entry = BaseRelocationEntry.from_bytes(data)
        assert entry.offset == 0xABC

        # Max offset 0xFFF
        struct.pack_into("<H", data, 0, (10 << 12) | 0xFFF)
        entry = BaseRelocationEntry.from_bytes(data)
        assert entry.offset == 0xFFF

    def test_is_absolute(self):
        """Test is_absolute property."""
        data = bytearray(2)

        # ABSOLUTE (padding entry)
        struct.pack_into("<H", data, 0, 0)
        entry = BaseRelocationEntry.from_bytes(data)
        assert entry.is_absolute is True

        # DIR64 (not absolute)
        struct.pack_into("<H", data, 0, IMAGE_REL_BASED_DIR64 << 12)
        entry = BaseRelocationEntry.from_bytes(data)
        assert entry.is_absolute is False

    def test_is_dir64(self):
        """Test is_dir64 property."""
        data = bytearray(2)

        struct.pack_into("<H", data, 0, IMAGE_REL_BASED_DIR64 << 12)
        entry = BaseRelocationEntry.from_bytes(data)
        assert entry.is_dir64 is True

        struct.pack_into("<H", data, 0, IMAGE_REL_BASED_HIGHLOW << 12)
        entry = BaseRelocationEntry.from_bytes(data)
        assert entry.is_dir64 is False

    def test_is_highlow(self):
        """Test is_highlow property."""
        data = bytearray(2)

        struct.pack_into("<H", data, 0, IMAGE_REL_BASED_HIGHLOW << 12)
        entry = BaseRelocationEntry.from_bytes(data)
        assert entry.is_highlow is True


class TestAlignmentHelpers:
    """Tests for alignment helper functions."""

    def test_round_up_to_alignment(self):
        """Test round_up_to_alignment function."""
        # Already aligned
        assert round_up_to_alignment(0x1000, 0x1000) == 0x1000

        # Needs rounding up
        assert round_up_to_alignment(0x1001, 0x1000) == 0x2000
        assert round_up_to_alignment(0x1FFF, 0x1000) == 0x2000

        # Zero value
        assert round_up_to_alignment(0, 0x1000) == 0

        # Small alignment
        assert round_up_to_alignment(5, 4) == 8
        assert round_up_to_alignment(8, 4) == 8

    def test_round_down_to_alignment(self):
        """Test round_down_to_alignment function."""
        # Already aligned
        assert round_down_to_alignment(0x1000, 0x1000) == 0x1000

        # Needs rounding down
        assert round_down_to_alignment(0x1001, 0x1000) == 0x1000
        assert round_down_to_alignment(0x1FFF, 0x1000) == 0x1000

        # Zero value
        assert round_down_to_alignment(0, 0x1000) == 0

    def test_alignment_with_zero(self):
        """Test alignment functions with zero alignment (edge case)."""
        # Zero alignment should return value unchanged
        assert round_up_to_alignment(123, 0) == 123
        assert round_down_to_alignment(123, 0) == 123  # Returns unchanged

    def test_round_up_to_page(self):
        """Test round_up_to_page function."""
        assert round_up_to_page(0) == 0
        assert round_up_to_page(1) == PAGE_SIZE
        assert round_up_to_page(PAGE_SIZE) == PAGE_SIZE
        assert round_up_to_page(PAGE_SIZE + 1) == PAGE_SIZE * 2

    def test_round_down_to_page(self):
        """Test round_down_to_page function."""
        assert round_down_to_page(0) == 0
        assert round_down_to_page(PAGE_SIZE - 1) == 0
        assert round_down_to_page(PAGE_SIZE) == PAGE_SIZE
        assert round_down_to_page(PAGE_SIZE + 1) == PAGE_SIZE


class TestSectionNameToBytes:
    """Tests for section_name_to_bytes function."""

    def test_short_name_pads_with_nulls(self):
        """Test that short names are padded with nulls to 8 bytes."""
        result = section_name_to_bytes(".text")
        assert result == b".text\x00\x00\x00"
        assert len(result) == 8

    def test_exact_8_chars(self):
        """Test that exact 8-char names work without padding."""
        result = section_name_to_bytes(".hip_fat")
        assert result == b".hip_fat"
        assert len(result) == 8

    def test_long_name_raises(self):
        """Test that names longer than 8 chars raise ValueError."""
        with pytest.raises(ValueError, match="Section name too long"):
            section_name_to_bytes(".rocm_kpack_ref")

    def test_empty_name(self):
        """Test that empty name is padded to 8 nulls."""
        result = section_name_to_bytes("")
        assert result == b"\x00\x00\x00\x00\x00\x00\x00\x00"
        assert len(result) == 8
