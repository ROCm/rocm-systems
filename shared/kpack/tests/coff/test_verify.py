"""
Unit tests for the rocm_kpack.coff.verify module.

Tests CoffVerifier class for PE binary validation.
"""

import struct

import pytest
from pathlib import Path

from rocm_kpack.coff import (
    CoffSurgery,
    CoffVerifier,
    VerificationResult,
    verify_with_llvm_objdump,
    verify_all,
)
from rocm_kpack.coff.types import SectionHeader


class TestCoffVerifier:
    """Tests for the CoffVerifier class."""

    def test_verify_valid_exe(self, test_assets_dir: Path):
        """Test verification of a valid executable passes."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"

    def test_verify_valid_dll(self, test_assets_dir: Path):
        """Test verification of a valid DLL passes."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.dll"
        )
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"

    def test_verify_host_only_exe(self, test_assets_dir: Path):
        """Test verification of host-only executable passes."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.exe"
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"

    def test_verify_host_only_dll(self, test_assets_dir: Path):
        """Test verification of host-only DLL passes."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"

    def test_verify_multi_arch(self, test_assets_dir: Path):
        """Test verification of multi-arch binary passes."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_multi.exe"
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"

    def test_verify_multi_wrapper(self, test_assets_dir: Path):
        """Test verification of RDC multi-wrapper binary passes."""
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_multi_wrapper.dll"
        )
        result = CoffVerifier.verify(binary)
        assert result.passed, f"Verification failed: {result}"


class TestVerificationResult:
    """Tests for VerificationResult class."""

    def test_empty_result_passes(self):
        """Empty result should pass by default."""
        result = VerificationResult()
        assert result.passed
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_fails(self):
        """Adding an error should fail the result."""
        result = VerificationResult()
        result.add_error("Test error")
        assert not result.passed
        assert len(result.errors) == 1
        assert "Test error" in result.errors[0]

    def test_add_warning_does_not_fail(self):
        """Adding a warning should not fail the result."""
        result = VerificationResult()
        result.add_warning("Test warning")
        assert result.passed
        assert len(result.warnings) == 1
        assert "Test warning" in result.warnings[0]

    def test_merge_results(self):
        """Test merging two results."""
        result1 = VerificationResult()
        result1.add_warning("Warning 1")

        result2 = VerificationResult()
        result2.add_error("Error 1")

        result1.merge(result2)
        assert not result1.passed
        assert len(result1.warnings) == 1
        assert len(result1.errors) == 1


class TestVerifyWithTools:
    """Tests for external tool verification (may skip if tools not available)."""

    def test_verify_with_objdump(self, test_assets_dir: Path, toolchain):
        """Test verification with objdump/llvm-objdump."""
        # Use DLL since .exe assets may not exist
        binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.dll"
        )
        if not binary.exists():
            pytest.skip(f"Test asset not found: {binary}")

        # Try to get objdump from toolchain (falls back to llvm-objdump)
        try:
            objdump_path = toolchain.objdump
        except OSError:
            pytest.skip("objdump/llvm-objdump not available")

        result = verify_with_llvm_objdump(binary, objdump_path)
        # Should pass or give "not found" warning, but not error
        if not result.passed:
            # Check it's just a "not found" warning
            for error in result.errors:
                assert "not found" in error.lower() or "warning" in error.lower()


class TestVerifierDetectsErrors:
    """Tests that verifier correctly detects specific errors in malformed binaries."""

    def test_overlapping_sections_detected(self, test_assets_dir: Path):
        """Test that overlapping sections are detected as an error."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Find two sections with raw data
        sections = [s for s in surgery.iter_sections() if s.raw_size > 0]
        if len(sections) < 2:
            pytest.skip("Need at least 2 sections with raw data")

        # Make second section overlap with first
        first = sections[0]
        second = sections[1]

        # Corrupt: set second section's PointerToRawData to overlap with first
        corrupted_shdr = SectionHeader(
            Name=second.header.Name,
            VirtualSize=second.header.VirtualSize,
            VirtualAddress=second.header.VirtualAddress,
            SizeOfRawData=second.header.SizeOfRawData,
            PointerToRawData=first.file_offset + 0x10,  # Overlap!
            PointerToRelocations=second.header.PointerToRelocations,
            PointerToLinenumbers=second.header.PointerToLinenumbers,
            NumberOfRelocations=second.header.NumberOfRelocations,
            NumberOfLinenumbers=second.header.NumberOfLinenumbers,
            Characteristics=second.header.Characteristics,
        )
        surgery.update_section_header(second.index, corrupted_shdr)

        # Verify with in-memory data
        result = CoffVerifier.verify_data(surgery.data)

        assert not result.passed, "Overlapping sections should fail verification"
        assert any(
            "overlapping" in e.lower() for e in result.errors
        ), f"Expected 'overlapping' error, got: {result.errors}"

    def test_misaligned_section_detected(self, test_assets_dir: Path):
        """Test that misaligned section PointerToRawData is detected."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Find a section with raw data
        section = None
        for s in surgery.iter_sections():
            if s.raw_size > 0:
                section = s
                break

        if section is None:
            pytest.skip("No section with raw data found")

        # Corrupt: set PointerToRawData to misaligned value
        corrupted_shdr = SectionHeader(
            Name=section.header.Name,
            VirtualSize=section.header.VirtualSize,
            VirtualAddress=section.header.VirtualAddress,
            SizeOfRawData=section.header.SizeOfRawData,
            PointerToRawData=section.file_offset + 1,  # Misaligned!
            PointerToRelocations=section.header.PointerToRelocations,
            PointerToLinenumbers=section.header.PointerToLinenumbers,
            NumberOfRelocations=section.header.NumberOfRelocations,
            NumberOfLinenumbers=section.header.NumberOfLinenumbers,
            Characteristics=section.header.Characteristics,
        )
        surgery.update_section_header(section.index, corrupted_shdr)

        result = CoffVerifier.verify_data(surgery.data)

        assert not result.passed, "Misaligned section should fail verification"
        assert any(
            "not aligned" in e.lower() for e in result.errors
        ), f"Expected 'not aligned' error, got: {result.errors}"

    def test_section_beyond_file_detected(self, test_assets_dir: Path):
        """Test that section extending beyond file is detected."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Find a section with raw data
        section = None
        for s in surgery.iter_sections():
            if s.raw_size > 0:
                section = s
                break

        if section is None:
            pytest.skip("No section with raw data found")

        file_size = len(surgery.data)

        # Corrupt: set SizeOfRawData to extend beyond file
        corrupted_shdr = SectionHeader(
            Name=section.header.Name,
            VirtualSize=section.header.VirtualSize,
            VirtualAddress=section.header.VirtualAddress,
            SizeOfRawData=file_size * 2,  # Way beyond file!
            PointerToRawData=section.header.PointerToRawData,
            PointerToRelocations=section.header.PointerToRelocations,
            PointerToLinenumbers=section.header.PointerToLinenumbers,
            NumberOfRelocations=section.header.NumberOfRelocations,
            NumberOfLinenumbers=section.header.NumberOfLinenumbers,
            Characteristics=section.header.Characteristics,
        )
        surgery.update_section_header(section.index, corrupted_shdr)

        result = CoffVerifier.verify_data(surgery.data)

        assert not result.passed, "Section beyond file should fail verification"
        assert any(
            "extends beyond file" in e.lower() for e in result.errors
        ), f"Expected 'extends beyond file' error, got: {result.errors}"

    def test_invalid_size_of_image_detected(self, test_assets_dir: Path):
        """Test that invalid SizeOfImage is detected."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Corrupt: set SizeOfImage to very small value
        # We need to modify the optional header
        opt_hdr_offset = surgery._get_optional_header_offset()
        # SizeOfImage is at offset 56 in OptionalHeader64
        struct.pack_into("<I", surgery.data, opt_hdr_offset + 56, 0x1000)

        result = CoffVerifier.verify_data(surgery.data)

        assert not result.passed, "Invalid SizeOfImage should fail verification"
        assert any(
            "sizeofimage" in e.lower() for e in result.errors
        ), f"Expected 'SizeOfImage' error, got: {result.errors}"

    def test_invalid_file_alignment_detected(self, test_assets_dir: Path):
        """Test that invalid FileAlignment (not power of 2) is detected."""
        binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        surgery = CoffSurgery.load(binary)

        # Corrupt: set FileAlignment to non-power-of-2
        opt_hdr_offset = surgery._get_optional_header_offset()
        # FileAlignment is at offset 36 in OptionalHeader64
        struct.pack_into(
            "<I", surgery.data, opt_hdr_offset + 36, 0x333
        )  # Not power of 2

        result = CoffVerifier.verify_data(surgery.data)

        assert not result.passed, "Invalid FileAlignment should fail verification"
        assert any(
            "filealignment" in e.lower() and "power of 2" in e.lower()
            for e in result.errors
        ), f"Expected FileAlignment power-of-2 error, got: {result.errors}"


class TestVerificationResultStr:
    """Tests for VerificationResult string representation."""

    def test_passed_str(self):
        """Test string output for passed result."""
        result = VerificationResult()
        s = str(result)
        assert "PASSED" in s

    def test_failed_str(self):
        """Test string output for failed result."""
        result = VerificationResult()
        result.add_error("Test error message")
        s = str(result)
        assert "FAILED" in s
        assert "Test error message" in s

    def test_warnings_in_str(self):
        """Test that warnings appear in string output."""
        result = VerificationResult()
        result.add_warning("Test warning message")
        s = str(result)
        assert "PASSED" in s  # Warnings don't fail
        assert "Test warning message" in s
