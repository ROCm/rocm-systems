"""
Unit tests for the rocm_kpack.coff.surgery module.

Tests CoffSurgery class for PE/COFF binary manipulation.
"""

import pytest
from pathlib import Path

import struct

from rocm_kpack.coff import (
    CoffSurgery,
    SectionInfo,
    IMAGE_SCN_MEM_READ,
    IMAGE_SCN_CNT_INITIALIZED_DATA,
)
from rocm_kpack.coff.types import DOS_MAGIC


class TestCoffSurgery:
    """Tests for the CoffSurgery class."""

    def test_load_executable(self, test_assets_dir: Path):
        """Test loading a valid PE executable."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        assert surgery.coff_header is not None
        assert surgery.optional_header is not None
        # Should be an executable (not DLL)
        assert not surgery.is_dll

    def test_load_dll(self, test_assets_dir: Path):
        """Test loading a valid DLL."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.dll"
        )
        surgery = CoffSurgery.load(binary)

        assert surgery.coff_header is not None
        assert surgery.optional_header is not None
        # Should be a DLL
        assert surgery.is_dll

    def test_find_section(self, test_assets_dir: Path):
        """Test finding sections by name."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        # Should find .hip_fat (PE uses 8-char names)
        section = surgery.find_section(".hip_fat")
        assert section is not None
        assert section.name == ".hip_fat"
        assert section.raw_size > 0

        # Should not find nonexistent section
        assert surgery.find_section(".nonexistent") is None

    def test_find_hipfatb_segment(self, test_assets_dir: Path):
        """Test finding .hipFatB (wrapper) section."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        # Should find .hipFatB (wrapper section)
        section = surgery.find_section(".hipFatB")
        assert section is not None
        assert section.name == ".hipFatB"

    def test_iter_sections(self, test_assets_dir: Path):
        """Test iterating over all sections."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        sections = list(surgery.iter_sections())
        assert len(sections) > 0

        # Check we have expected sections
        names = [s.name for s in sections]
        assert ".text" in names
        assert ".hip_fat" in names
        assert ".hipFatB" in names

    def test_get_section_content(self, test_assets_dir: Path):
        """Test reading section content."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        section = surgery.find_section(".hip_fat")
        assert section is not None

        content = surgery.get_section_content(section)
        assert len(content) > 0

    def test_rva_to_file_offset(self, test_assets_dir: Path):
        """Test RVA to file offset conversion."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        section = surgery.find_section(".text")
        assert section is not None

        # RVA should map to file offset
        offset = surgery.rva_to_file_offset(section.rva)
        assert offset is not None
        assert offset == section.file_offset

    def test_add_section(self, test_assets_dir: Path, tmp_path: Path):
        """Test adding a new section."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        surgery = CoffSurgery.load(binary)

        # Count original sections
        original_count = len(list(surgery.iter_sections()))

        # Add a new section
        test_data = b"Hello, test section!"
        result = surgery.add_section(
            name=".test",
            content=test_data,
            characteristics=IMAGE_SCN_MEM_READ | IMAGE_SCN_CNT_INITIALIZED_DATA,
        )

        assert result is not None
        assert result.section_index >= 0

        # Verify section exists
        new_section = surgery.find_section(".test")
        assert new_section is not None

        # Verify content can be read back (slice to virtual_size, not padded raw_size)
        content = surgery.get_section_content(new_section)
        content = content[: new_section.virtual_size]
        assert content == test_data

        # Save and reload
        output_path = tmp_path / "with_section.exe"
        surgery.save(output_path)

        # Reload and verify
        surgery2 = CoffSurgery.load(output_path)
        new_count = len(list(surgery2.iter_sections()))
        assert new_count == original_count + 1

        section = surgery2.find_section(".test")
        assert section is not None
        content = surgery2.get_section_content(section)
        content = content[: section.virtual_size]
        assert content == test_data


class TestHostOnlyBinaries:
    """Tests for host-only binaries (negative tests)."""

    def test_host_only_exe_no_hip_fat(self, test_assets_dir: Path):
        """Test that host-only exe has no .hip_fat section."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.exe"
        surgery = CoffSurgery.load(binary)

        # Should NOT have .hip_fat section
        assert surgery.find_section(".hip_fat") is None
        assert surgery.find_section(".hipFatB") is None

    def test_host_only_dll_no_hip_fat(self, test_assets_dir: Path):
        """Test that host-only DLL has no .hip_fat section."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Should NOT have .hip_fat section
        assert surgery.find_section(".hip_fat") is None
        assert surgery.find_section(".hipFatB") is None


class TestCoffSurgeryErrors:
    """Tests for error handling in CoffSurgery."""

    def test_load_truncated_file_raises(self, tmp_path: Path):
        """Test that loading a truncated file raises an error."""
        truncated = tmp_path / "truncated.dll"
        # Write just MZ header, not enough for full PE
        data = bytearray(64)
        struct.pack_into("<H", data, 0, DOS_MAGIC)
        struct.pack_into("<I", data, 60, 0x80)  # PE offset beyond file
        truncated.write_bytes(data)

        with pytest.raises(ValueError, match="Invalid PE header offset|Data too short"):
            CoffSurgery.load(truncated)

    def test_load_non_pe_file_raises(self, tmp_path: Path):
        """Test that loading a non-PE file raises an error."""
        text_file = tmp_path / "text.txt"
        text_file.write_text("This is not a PE file")

        # Small files fail with "Data too short", larger non-PE files fail with
        # "Not a DOS/PE file" - either is acceptable
        with pytest.raises(ValueError, match="Data too short|Not a DOS/PE file"):
            CoffSurgery.load(text_file)

    def test_invalid_pe_offset_raises(self, tmp_path: Path):
        """Test that invalid PE offset raises an error."""
        bad_pe = tmp_path / "bad_pe.dll"
        data = bytearray(256)
        struct.pack_into("<H", data, 0, DOS_MAGIC)
        struct.pack_into("<I", data, 60, 0xFFFFFF)  # PE offset way beyond file
        bad_pe.write_bytes(data)

        with pytest.raises(ValueError, match="Invalid PE header offset"):
            CoffSurgery.load(bad_pe)

    def test_too_many_sections_raises(self, tmp_path: Path, test_assets_dir: Path):
        """Test that too many sections raises an error."""
        # Load a valid DLL and corrupt the NumberOfSections
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        data = bytearray(binary.read_bytes())

        # Find PE offset and corrupt NumberOfSections in COFF header
        pe_offset = struct.unpack_from("<I", data, 60)[0]
        coff_offset = pe_offset + 4

        # Set NumberOfSections to something huge
        struct.pack_into("<H", data, coff_offset + 2, 0x7FFF)

        corrupted = tmp_path / "corrupted.dll"
        corrupted.write_bytes(data)

        with pytest.raises(ValueError, match="NumberOfSections.*exceeds maximum"):
            CoffSurgery.load(corrupted)

    def test_too_many_data_directories_raises(
        self, tmp_path: Path, test_assets_dir: Path
    ):
        """Test that too many data directories raises an error."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        data = bytearray(binary.read_bytes())

        # Find PE offset
        pe_offset = struct.unpack_from("<I", data, 60)[0]
        # Optional header is after COFF header (4 + 20 bytes)
        opt_offset = pe_offset + 4 + 20

        # NumberOfRvaAndSizes is at offset 108 in OptionalHeader64
        # (end of the 112-byte fixed part, minus 4 for the field itself)
        struct.pack_into("<I", data, opt_offset + 108, 0x7FFF)

        corrupted = tmp_path / "corrupted.dll"
        corrupted.write_bytes(data)

        with pytest.raises(ValueError, match="NumberOfRvaAndSizes.*exceeds maximum"):
            CoffSurgery.load(corrupted)


class TestAddressConversion:
    """Tests for address conversion methods."""

    def test_file_offset_to_rva(self, test_assets_dir: Path):
        """Test file offset to RVA conversion."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Get a known section
        section = surgery.find_section(".text")
        if section is None:
            pytest.skip("No .text section")

        # File offset should convert back to RVA
        rva = surgery.file_offset_to_rva(section.file_offset)
        assert rva == section.rva

    def test_rva_outside_section_returns_none(self, test_assets_dir: Path):
        """Test that RVA outside all sections returns None."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Use a very high RVA that's beyond all sections
        offset = surgery.rva_to_file_offset(0xFFFFFFFF)
        assert offset is None

    def test_rva_in_bss_section_returns_none(self, test_assets_dir: Path):
        """Test that RVA in BSS section (no raw data) returns None."""
        # Note: This test may skip if no BSS section exists in test binary
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Look for a section with no raw data
        for section in surgery.iter_sections():
            if section.raw_size == 0 and section.virtual_size > 0:
                # This is a BSS-like section
                offset = surgery.rva_to_file_offset(section.rva)
                assert offset is None
                return

        pytest.skip("No BSS section found in test binary")

    def test_va_conversion_roundtrip(self, test_assets_dir: Path):
        """Test VA to RVA and back conversion."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Get a section RVA
        section = surgery.find_section(".text")
        if section is None:
            pytest.skip("No .text section")

        rva = section.rva
        va = surgery.rva_to_va(rva)
        rva_back = surgery.va_to_rva(va)

        assert rva_back == rva
        assert va == surgery.image_base + rva


class TestWriteOperations:
    """Tests for write operations."""

    def test_write_bytes_at_offset(self, test_assets_dir: Path, tmp_path: Path):
        """Test writing bytes at a file offset."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Find a writable location (end of a section)
        section = surgery.find_section(".text")
        if section is None:
            pytest.skip("No .text section")

        # Write some bytes at a safe offset (within section)
        test_data = b"TEST"
        write_offset = section.file_offset + 10
        surgery.write_bytes_at_offset(write_offset, test_data, "test write")

        # Verify the write
        assert surgery.data[write_offset : write_offset + len(test_data)] == test_data

        # Verify modification was tracked
        assert any(m.operation == "write_bytes" for m in surgery.modifications)

    def test_write_beyond_file_raises(self, test_assets_dir: Path):
        """Test that writing beyond file bounds raises error."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        file_size = len(surgery.data)
        with pytest.raises(ValueError, match="exceed file bounds"):
            surgery.write_bytes_at_offset(file_size, b"OVERFLOW")

    def test_zero_range(self, test_assets_dir: Path):
        """Test zeroing a range of bytes."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Find a section with data
        section = surgery.find_section(".text")
        if section is None:
            pytest.skip("No .text section")

        # Zero some bytes
        zero_offset = section.file_offset + 20
        zero_size = 16
        surgery.zero_range(zero_offset, zero_size)

        # Verify zeros
        assert (
            surgery.data[zero_offset : zero_offset + zero_size] == b"\x00" * zero_size
        )

    def test_write_pointer_at_rva(self, test_assets_dir: Path):
        """Test writing a pointer at an RVA."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Find a section
        section = surgery.find_section(".text")
        if section is None:
            pytest.skip("No .text section")

        # Write a pointer (8 bytes)
        rva = section.rva + 32  # Offset into section
        test_value = 0x140001000
        surgery.write_pointer_at_rva(rva, test_value)

        # Read it back
        read_value = surgery.read_pointer_at_rva(rva)
        assert read_value == test_value


class TestBaseRelocations:
    """Tests for base relocation operations."""

    def test_iter_base_relocations(self, test_assets_dir: Path):
        """Test iterating base relocations."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Iterate relocations
        relocs = list(surgery.iter_base_relocations())

        # DLLs typically have relocations
        # Just verify we can iterate without error
        for reloc in relocs:
            assert reloc.target_rva >= 0
            assert reloc.entry is not None

    def test_find_relocation_at_rva(self, test_assets_dir: Path):
        """Test finding a relocation at a specific RVA."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Get first relocation
        relocs = list(surgery.iter_base_relocations())
        if not relocs:
            pytest.skip("No relocations in test binary")

        first_reloc = relocs[0]

        # Find it
        found = surgery.find_relocation_at_rva(first_reloc.target_rva)
        assert found is not None
        assert found.target_rva == first_reloc.target_rva

    def test_no_relocation_directory(self, tmp_path: Path, test_assets_dir: Path):
        """Test iteration when no relocation directory exists."""
        # Create a PE-like structure without relocation directory
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        data = bytearray(binary.read_bytes())

        # Find PE offset
        pe_offset = struct.unpack_from("<I", data, 60)[0]
        opt_offset = pe_offset + 4 + 20

        # Zero out the base relocation directory (index 5)
        # Data directories start at opt_offset + 112
        reloc_dir_offset = opt_offset + 112 + 5 * 8
        struct.pack_into("<II", data, reloc_dir_offset, 0, 0)

        # Save and load
        no_reloc = tmp_path / "no_reloc.dll"
        no_reloc.write_bytes(data)

        surgery = CoffSurgery.load(no_reloc)

        # Should return empty iterator
        relocs = list(surgery.iter_base_relocations())
        assert relocs == []


class TestAddSectionErrors:
    """Tests for add_section error cases."""

    def test_add_duplicate_section_raises(self, test_assets_dir: Path):
        """Test that adding duplicate section raises error."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # .text already exists
        with pytest.raises(ValueError, match="already exists"):
            surgery.add_section(".text", b"data")

    def test_add_section_name_truncation(self, test_assets_dir: Path, tmp_path: Path):
        """Test that long section names are truncated to 8 chars."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Long name should work - gets silently truncated to 8 chars
        result = surgery.add_section(
            name=".very_long_section_name",
            content=b"\x00" * 16,
            characteristics=0,
        )

        # Name is truncated to first 8 chars: ".very_lo"
        section = surgery.find_section(".very_lo")
        assert section is not None, "Section with truncated name not found"
        assert result.rva == section.rva
