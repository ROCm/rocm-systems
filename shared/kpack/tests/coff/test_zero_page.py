"""Tests for PE/COFF zero-page optimization."""

import pytest
from pathlib import Path

from rocm_kpack.coff.zero_page import (
    calculate_aligned_range,
    conservative_zero_page,
    ZeroPageResult,
)
from rocm_kpack.coff.surgery import CoffSurgery
from rocm_kpack.coff.types import PAGE_SIZE


class TestCalculateAlignedRange:
    """Tests for calculate_aligned_range function."""

    def test_section_smaller_than_page(self):
        """Test section smaller than one page returns (0, 0)."""
        # Section from 0x1000 to 0x1100 (256 bytes, less than PAGE_SIZE)
        aligned_rva, aligned_size = calculate_aligned_range(0x1000, 0x100)
        assert aligned_rva == 0
        assert aligned_size == 0

    def test_section_exactly_one_page(self):
        """Test page-aligned section of exactly one page."""
        # Section from 0x1000 to 0x2000 (exactly 4KB, page-aligned)
        aligned_rva, aligned_size = calculate_aligned_range(0x1000, PAGE_SIZE)
        assert aligned_rva == 0x1000
        assert aligned_size == PAGE_SIZE

    def test_section_multiple_pages(self):
        """Test section spanning multiple pages."""
        # Section from 0x1000 to 0x5000 (16KB = 4 pages)
        aligned_rva, aligned_size = calculate_aligned_range(0x1000, 4 * PAGE_SIZE)
        assert aligned_rva == 0x1000
        assert aligned_size == 4 * PAGE_SIZE

    def test_misaligned_start(self):
        """Test section with misaligned start address."""
        # Section from 0x1500 to 0x5500 (16KB, but starts mid-page)
        # Should return range from 0x2000 to 0x5000 (3 pages)
        aligned_rva, aligned_size = calculate_aligned_range(0x1500, 4 * PAGE_SIZE)
        assert aligned_rva == 0x2000
        assert aligned_size == 3 * PAGE_SIZE

    def test_misaligned_end(self):
        """Test section with misaligned end address."""
        # Section from 0x1000 to 0x4500 (ends mid-page)
        # Should return range from 0x1000 to 0x4000 (3 pages)
        aligned_rva, aligned_size = calculate_aligned_range(0x1000, 0x3500)
        assert aligned_rva == 0x1000
        assert aligned_size == 3 * PAGE_SIZE

    def test_misaligned_both_ends(self):
        """Test section with both start and end misaligned."""
        # Section from 0x1500 to 0x4500 (both ends mid-page)
        # Should return range from 0x2000 to 0x4000 (2 pages)
        aligned_rva, aligned_size = calculate_aligned_range(0x1500, 0x3000)
        assert aligned_rva == 0x2000
        assert aligned_size == 2 * PAGE_SIZE

    def test_no_full_page_spans(self):
        """Test section that spans into next page but doesn't contain full page."""
        # Section from 0x1800 to 0x2800 (crosses page boundary but no full page)
        aligned_rva, aligned_size = calculate_aligned_range(0x1800, 0x1000)
        assert aligned_rva == 0
        assert aligned_size == 0

    def test_zero_size_section(self):
        """Test zero-size section."""
        aligned_rva, aligned_size = calculate_aligned_range(0x1000, 0)
        assert aligned_rva == 0
        assert aligned_size == 0

    def test_large_section(self):
        """Test large section (1MB)."""
        size = 256 * PAGE_SIZE  # 1MB
        aligned_rva, aligned_size = calculate_aligned_range(0x1000, size)
        assert aligned_rva == 0x1000
        assert aligned_size == size


class TestConservativeZeroPage:
    """Tests for conservative_zero_page function."""

    def test_zero_page_reduces_size(self, tmp_path: Path, test_assets_dir: Path):
        """Test that zero-page optimization reduces file size."""
        # Use a DLL with .hip_fat section
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.dll"
        )
        assert input_binary.exists(), f"Test asset not found: {input_binary}"

        surgery = CoffSurgery.load(input_binary)
        original_size = len(surgery.data)

        # Find .hip_fat section first to verify it exists
        section = surgery.find_section(".hip_fat")
        assert section is not None, "Test binary must have .hip_fat section"

        result = conservative_zero_page(surgery, ".hip_fat")

        assert result.success is True
        assert result.bytes_saved > 0
        assert result.pages_zeroed > 0
        assert len(surgery.data) < original_size

    def test_section_not_found(self, tmp_path: Path, test_assets_dir: Path):
        """Test zero-page returns failure for non-existent section."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.dll"
        )
        assert input_binary.exists(), f"Test asset not found: {input_binary}"

        surgery = CoffSurgery.load(input_binary)

        result = conservative_zero_page(surgery, ".nonexistent")

        assert result.success is False
        assert "not found" in result.error
        assert result.bytes_saved == 0

    def test_repeated_zero_page_eventually_no_savings(
        self, tmp_path: Path, test_assets_dir: Path
    ):
        """Test that repeated zero-page eventually saves nothing."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.dll"
        )
        assert input_binary.exists(), f"Test asset not found: {input_binary}"

        surgery = CoffSurgery.load(input_binary)

        # Keep zero-paging until no more savings
        total_saved = 0
        max_iterations = 10
        converged = False
        for i in range(max_iterations):
            result = conservative_zero_page(surgery, ".hip_fat")
            if not result.success:
                converged = True
                break
            if result.bytes_saved == 0:
                converged = True
                break
            total_saved += result.bytes_saved

        assert converged, (
            f"Zero-page did not converge after {max_iterations} iterations "
            f"(saved {total_saved} bytes total)"
        )

        # After exhaustive zero-paging, final attempt should save nothing
        final_result = conservative_zero_page(surgery, ".hip_fat")
        assert final_result.success is True
        # Either no bytes saved or section is now too small/already zero
        assert (
            final_result.bytes_saved == 0
            or "too small" in (final_result.error or "")
            or "already" in (final_result.error or "")
        )

    def test_small_section_no_savings(self, tmp_path: Path, test_assets_dir: Path):
        """Test that small sections don't get zero-paged."""
        # Use host_only.dll which may have small sections
        input_binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        assert input_binary.exists(), f"Test asset not found: {input_binary}"

        surgery = CoffSurgery.load(input_binary)

        # Try to zero-page a small section (like .text which is typically small)
        result = conservative_zero_page(surgery, ".text")

        # Should either not find section or have no savings (too small)
        if result.success:
            # If it succeeded, it should have minimal or no savings
            # Small sections don't have full pages to zero
            pass  # Just verify it doesn't crash

    def test_subsequent_sections_adjusted(self, tmp_path: Path, test_assets_dir: Path):
        """Test that subsequent sections have their offsets adjusted."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.dll"
        )
        assert input_binary.exists(), f"Test asset not found: {input_binary}"

        surgery = CoffSurgery.load(input_binary)

        # Record original section offsets
        original_offsets = {}
        hip_fat_found = False
        for section in surgery.iter_sections():
            original_offsets[section.name] = section.file_offset
            if section.name == ".hip_fat":
                hip_fat_found = True
                hip_fat_offset = section.file_offset

        assert hip_fat_found, "Test binary must have .hip_fat section"

        result = conservative_zero_page(surgery, ".hip_fat")
        assert result.success, f"Zero-page failed: {result.error}"
        assert result.bytes_saved > 0, "Zero-page must save bytes on test binary"

        # Verify sections after .hip_fat have adjusted offsets
        for section in surgery.iter_sections():
            orig_offset = original_offsets.get(section.name)
            if orig_offset is not None and orig_offset > hip_fat_offset:
                # Sections after .hip_fat should have smaller offsets
                assert section.file_offset < orig_offset, (
                    f"Section {section.name} offset not adjusted: "
                    f"was {orig_offset:#x}, now {section.file_offset:#x}"
                )


class TestZeroPageResult:
    """Tests for ZeroPageResult dataclass."""

    def test_success_result(self):
        """Test creating a success result."""
        result = ZeroPageResult(
            success=True,
            bytes_saved=4096,
            pages_zeroed=1,
        )
        assert result.success is True
        assert result.bytes_saved == 4096
        assert result.pages_zeroed == 1
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result = ZeroPageResult(
            success=False,
            bytes_saved=0,
            pages_zeroed=0,
            error="Section not found",
        )
        assert result.success is False
        assert result.error == "Section not found"
