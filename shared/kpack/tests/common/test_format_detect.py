"""Tests for binary format detection edge cases.

These tests focus on edge cases and error handling in format detection,
using synthetic/minimal binary data rather than full test assets.
"""

import struct
from pathlib import Path

import pytest

from rocm_kpack.format_detect import (
    detect_binary_format,
    is_elf_binary,
    is_pe_binary,
    UnsupportedBinaryFormat,
    ELF_MAGIC,
    DOS_MAGIC,
    PE_SIGNATURE,
)


class TestDetectBinaryFormat:
    """Tests for detect_binary_format function."""

    def test_detect_elf_format(self, tmp_path: Path):
        """Test that ELF format is correctly detected."""
        elf_file = tmp_path / "test.so"
        # Minimal ELF header (just magic + padding)
        elf_data = ELF_MAGIC + b"\x02\x01\x01" + (b"\x00" * 57)  # 64 bytes
        elf_file.write_bytes(elf_data)

        assert detect_binary_format(elf_file) == "elf"

    def test_detect_coff_format(self, tmp_path: Path):
        """Test that PE/COFF format is correctly detected."""
        pe_file = tmp_path / "test.dll"
        # Minimal PE: DOS header + PE signature
        data = bytearray(256)
        # DOS magic
        data[0:2] = DOS_MAGIC
        # PE offset at 0x3c points to 0x80
        struct.pack_into("<I", data, 0x3C, 0x80)
        # PE signature at offset 0x80
        data[0x80 : 0x80 + 4] = PE_SIGNATURE
        pe_file.write_bytes(data)

        assert detect_binary_format(pe_file) == "coff"

    def test_empty_file_raises(self, tmp_path: Path):
        """Test that empty file raises UnsupportedBinaryFormat."""
        empty_file = tmp_path / "empty"
        empty_file.write_bytes(b"")

        with pytest.raises(UnsupportedBinaryFormat, match="File too small"):
            detect_binary_format(empty_file)

    def test_small_file_raises(self, tmp_path: Path):
        """Test that file smaller than header raises UnsupportedBinaryFormat."""
        small_file = tmp_path / "small"
        small_file.write_bytes(b"ABC")  # 3 bytes, less than 4 needed

        with pytest.raises(UnsupportedBinaryFormat, match="File too small"):
            detect_binary_format(small_file)

    def test_text_file_raises(self, tmp_path: Path):
        """Test that text file raises UnsupportedBinaryFormat."""
        text_file = tmp_path / "readme.txt"
        text_file.write_text("This is a plain text file, not a binary.")

        with pytest.raises(UnsupportedBinaryFormat, match="neither ELF nor PE/COFF"):
            detect_binary_format(text_file)

    def test_dos_without_pe_raises(self, tmp_path: Path):
        """Test that DOS executable without PE header raises error."""
        dos_file = tmp_path / "dos.com"
        # DOS header magic but no valid PE signature
        data = bytearray(256)
        data[0:2] = DOS_MAGIC
        # PE offset points to garbage
        struct.pack_into("<I", data, 0x3C, 0x80)
        # No PE signature at 0x80 (just zeros)
        dos_file.write_bytes(data)

        with pytest.raises(UnsupportedBinaryFormat, match="neither ELF nor PE/COFF"):
            detect_binary_format(dos_file)

    def test_invalid_pe_offset_raises(self, tmp_path: Path):
        """Test that invalid PE offset raises error."""
        bad_pe = tmp_path / "bad.dll"
        data = bytearray(256)
        data[0:2] = DOS_MAGIC
        # PE offset way too large (beyond reasonable)
        struct.pack_into("<I", data, 0x3C, 0x200000)  # 2MB offset
        bad_pe.write_bytes(data)

        with pytest.raises(UnsupportedBinaryFormat, match="Invalid PE header offset"):
            detect_binary_format(bad_pe)

    def test_pe_offset_too_small_raises(self, tmp_path: Path):
        """Test that PE offset less than 0x40 raises error."""
        bad_pe = tmp_path / "bad2.dll"
        data = bytearray(256)
        data[0:2] = DOS_MAGIC
        # PE offset too small (must be at least 0x40)
        struct.pack_into("<I", data, 0x3C, 0x20)
        bad_pe.write_bytes(data)

        with pytest.raises(UnsupportedBinaryFormat, match="Invalid PE header offset"):
            detect_binary_format(bad_pe)

    def test_nonexistent_file_raises(self, tmp_path: Path):
        """Test that missing file raises FileNotFoundError."""
        missing = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError):
            detect_binary_format(missing)

    def test_mach_o_raises(self, tmp_path: Path):
        """Test that Mach-O binaries raise UnsupportedBinaryFormat."""
        macho_file = tmp_path / "test"
        # Mach-O magic (64-bit little-endian)
        macho_magic = b"\xcf\xfa\xed\xfe"
        macho_file.write_bytes(macho_magic + b"\x00" * 60)

        with pytest.raises(UnsupportedBinaryFormat, match="neither ELF nor PE/COFF"):
            detect_binary_format(macho_file)


class TestIsElfBinary:
    """Tests for is_elf_binary helper function."""

    def test_is_elf_true(self, tmp_path: Path):
        """Test is_elf_binary returns True for ELF."""
        elf_file = tmp_path / "test.so"
        elf_data = ELF_MAGIC + b"\x02\x01\x01" + (b"\x00" * 57)
        elf_file.write_bytes(elf_data)

        assert is_elf_binary(elf_file) is True

    def test_is_elf_false_for_pe(self, tmp_path: Path):
        """Test is_elf_binary returns False for PE."""
        pe_file = tmp_path / "test.dll"
        data = bytearray(256)
        data[0:2] = DOS_MAGIC
        struct.pack_into("<I", data, 0x3C, 0x80)
        data[0x80 : 0x80 + 4] = PE_SIGNATURE
        pe_file.write_bytes(data)

        assert is_elf_binary(pe_file) is False

    def test_is_elf_false_for_missing_file(self, tmp_path: Path):
        """Test is_elf_binary returns False for missing file."""
        missing = tmp_path / "does_not_exist"
        assert is_elf_binary(missing) is False

    def test_is_elf_false_for_text_file(self, tmp_path: Path):
        """Test is_elf_binary returns False for text file."""
        text_file = tmp_path / "readme.txt"
        text_file.write_text("Not a binary")
        assert is_elf_binary(text_file) is False


class TestIsPeBinary:
    """Tests for is_pe_binary helper function."""

    def test_is_pe_true(self, tmp_path: Path):
        """Test is_pe_binary returns True for PE."""
        pe_file = tmp_path / "test.dll"
        data = bytearray(256)
        data[0:2] = DOS_MAGIC
        struct.pack_into("<I", data, 0x3C, 0x80)
        data[0x80 : 0x80 + 4] = PE_SIGNATURE
        pe_file.write_bytes(data)

        assert is_pe_binary(pe_file) is True

    def test_is_pe_false_for_elf(self, tmp_path: Path):
        """Test is_pe_binary returns False for ELF."""
        elf_file = tmp_path / "test.so"
        elf_data = ELF_MAGIC + b"\x02\x01\x01" + (b"\x00" * 57)
        elf_file.write_bytes(elf_data)

        assert is_pe_binary(elf_file) is False

    def test_is_pe_false_for_missing_file(self, tmp_path: Path):
        """Test is_pe_binary returns False for missing file."""
        missing = tmp_path / "does_not_exist"
        assert is_pe_binary(missing) is False

    def test_is_pe_false_for_text_file(self, tmp_path: Path):
        """Test is_pe_binary returns False for text file."""
        text_file = tmp_path / "readme.txt"
        text_file.write_text("Not a binary")
        assert is_pe_binary(text_file) is False


class TestMagicConstants:
    """Tests for magic byte constants."""

    def test_elf_magic_value(self):
        """Test ELF magic bytes are correct."""
        assert ELF_MAGIC == b"\x7fELF"
        assert len(ELF_MAGIC) == 4

    def test_dos_magic_value(self):
        """Test DOS magic bytes are correct."""
        assert DOS_MAGIC == b"MZ"
        assert len(DOS_MAGIC) == 2

    def test_pe_signature_value(self):
        """Test PE signature bytes are correct."""
        assert PE_SIGNATURE == b"PE\x00\x00"
        assert len(PE_SIGNATURE) == 4
