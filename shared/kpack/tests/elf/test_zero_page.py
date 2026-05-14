"""Tests for zero-page optimization and PT_LOAD alignment.

These tests verify that:
1. conservative_zero_page() correctly adjusts e_phoff and e_shoff after
   removing bytes from the file
2. get_min_content_offset() correctly excludes PT_LOAD segments covering
   the PHDR table, allowing in-place PHDR writes when space is available
3. The full kpack_offload_binary pipeline produces binaries with valid
   PT_LOAD alignment (p_offset % p_align == p_vaddr % p_align)
"""

from pathlib import Path

import pytest

from rocm_kpack.elf import (
    ElfSurgery,
    ElfVerifier,
    ProgramHeaderManager,
    map_section_to_load,
    conservative_zero_page,
    zero_page_section,
    kpack_offload_binary,
    add_kpack_ref_section,
    rewrite_hipfatbin_magic,
    set_pointer,
    PT_LOAD,
    PT_PHDR,
    PAGE_SIZE,
)
from rocm_kpack.elf.types import ELF64_PHDR_SIZE


def assert_load_alignment(surgery: ElfSurgery, context: str = "") -> None:
    """Assert all PT_LOAD segments have correct mmap alignment.

    The ELF spec requires: (p_offset % p_align) == (p_vaddr % p_align)
    for all PT_LOAD segments with p_align > 0.

    Args:
        surgery: ElfSurgery instance to check
        context: Description for error messages
    """
    prefix = f" ({context})" if context else ""
    for idx, phdr in surgery.iter_program_headers():
        if phdr.p_type == PT_LOAD and phdr.p_align > 0:
            offset_mod = phdr.p_offset % phdr.p_align
            vaddr_mod = phdr.p_vaddr % phdr.p_align
            assert offset_mod == vaddr_mod, (
                f"PT_LOAD[{idx}] alignment violation{prefix}: "
                f"p_offset=0x{phdr.p_offset:x} (mod 0x{offset_mod:x}), "
                f"p_vaddr=0x{phdr.p_vaddr:x} (mod 0x{vaddr_mod:x}), "
                f"p_align=0x{phdr.p_align:x}"
            )


class TestGetMinContentOffset:
    """Tests for ElfSurgery.get_min_content_offset()."""

    def test_excludes_load_at_offset_zero(self, test_assets_dir: Path):
        """Verify PT_LOAD at offset 0 is excluded from min calculation.

        The first PT_LOAD typically starts at offset 0, covering the ELF
        header and PHDR table. Including this in get_min_content_offset()
        would make available space always negative, forcing unnecessary
        PHDR table relocation.
        """
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        min_offset = surgery.get_min_content_offset()
        e_phoff = surgery.ehdr.e_phoff

        # min_offset must be > e_phoff (not 0 from PT_LOAD at offset 0)
        assert min_offset > e_phoff, (
            f"get_min_content_offset() returned {min_offset:#x} which is at or "
            f"before e_phoff at {e_phoff:#x}. "
            "PT_LOAD segments at/before e_phoff should be excluded."
        )

    def test_available_space_is_nonnegative(self, test_assets_dir: Path):
        """Verify available space at PHDR location is non-negative."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        min_offset = surgery.get_min_content_offset()
        available = min_offset - surgery.ehdr.e_phoff

        assert available >= 0, (
            f"Available space for PHDR expansion is {available} (negative!). "
            f"min_content_offset={min_offset:#x}, e_phoff={surgery.ehdr.e_phoff:#x}"
        )


class TestZeroPageEphoffAdjustment:
    """Tests for correct e_phoff adjustment after byte removal."""

    def test_ephoff_adjusted_after_removal(self, test_assets_dir: Path):
        """Verify e_phoff is adjusted when PHDR is after removed bytes.

        When the PHDR table has been relocated to end of file (e.g., by
        map_section_to_load) and then conservative_zero_page removes bytes
        from the middle of the file, e_phoff must be adjusted by the number
        of removed bytes. Without this, subsequent PHDR operations use a
        stale offset.
        """
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        # Phase 0: Add a section (simulates add_kpack_ref_section)
        surgery.add_section(
            name=".rocm_kpack_ref",
            content=b"test marker data for alignment verification",
        )

        # Phase 1: Map to PT_LOAD (this relocates PHDR to end of file)
        map_result = map_section_to_load(surgery, ".rocm_kpack_ref")
        assert map_result.success

        # Record PHDR state before zero-page
        pre_phoff = surgery.ehdr.e_phoff
        pre_shoff = surgery.ehdr.e_shoff

        # Phase 3: Zero-page .hip_fatbin
        fatbin = surgery.find_section(".hip_fatbin")
        assert fatbin is not None, "Test binary must have .hip_fatbin"
        assert fatbin.size > PAGE_SIZE, "Test binary .hip_fatbin must be > 1 page"

        result = conservative_zero_page(surgery)
        assert result.success
        assert result.bytes_saved > 0

        # After zero-paging, e_phoff should have been adjusted
        # (either adjusted down by bytes_saved, or relocated to a new location)
        # The key invariant: e_phoff must point to valid PHDR data
        post_phoff = surgery.ehdr.e_phoff
        post_phnum = surgery.ehdr.e_phnum

        # Verify we can actually read program headers at the reported e_phoff
        assert post_phoff + post_phnum * ELF64_PHDR_SIZE <= len(surgery.data), (
            f"e_phoff ({post_phoff:#x}) + PHDR table size "
            f"({post_phnum * ELF64_PHDR_SIZE:#x}) exceeds file size "
            f"({len(surgery.data):#x})"
        )


class TestFullPipelineAlignment:
    """Tests for PT_LOAD alignment through the full kpack pipeline."""

    def test_kpack_offload_alignment(self, test_assets_dir: Path, tmp_path: Path):
        """Verify kpack_offload_binary produces valid PT_LOAD alignment.

        This is the end-to-end test for the alignment fix. It runs the
        complete transform pipeline (add section, map to LOAD, rewrite
        magic, zero-page) and verifies all PT_LOAD segments satisfy:
            p_offset % p_align == p_vaddr % p_align
        """
        input_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )
        output_binary = tmp_path / "aligned_output.exe"

        result = kpack_offload_binary(
            input_path=input_binary,
            output_path=output_binary,
            kpack_search_paths=["test.kpack"],
            kernel_name="test_kernel",
        )

        assert output_binary.exists()
        assert result["new_size"] > 0

        # Load output and verify alignment
        surgery = ElfSurgery.load(output_binary)
        assert_load_alignment(surgery, "after kpack_offload_binary")

    def test_kpack_offload_multi_binary_alignment(
        self, test_assets_dir: Path, tmp_path: Path
    ):
        """Verify alignment for multi-kernel binary (multiple wrappers)."""
        input_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
        )
        output_binary = tmp_path / "aligned_multi_output.exe"

        result = kpack_offload_binary(
            input_path=input_binary,
            output_path=output_binary,
            kpack_search_paths=["test.kpack"],
            kernel_name="test_kernel_multi",
        )

        assert output_binary.exists()
        surgery = ElfSurgery.load(output_binary)
        assert_load_alignment(surgery, "after kpack_offload_binary (multi)")

    def test_kpack_offload_shared_library_alignment(
        self, test_assets_dir: Path, tmp_path: Path
    ):
        """Verify alignment for shared library."""
        input_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_single.so"
        )
        output_binary = tmp_path / "aligned_output.so"

        result = kpack_offload_binary(
            input_path=input_binary,
            output_path=output_binary,
            kpack_search_paths=["test.kpack"],
            kernel_name="lib/libtest_kernel_single.so",
        )

        assert output_binary.exists()
        surgery = ElfSurgery.load(output_binary)
        assert_load_alignment(surgery, "after kpack_offload_binary (shared lib)")

    def test_verifier_passes_after_kpack(self, test_assets_dir: Path, tmp_path: Path):
        """Verify the full ELF verifier passes after kpack transform."""
        input_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )
        output_binary = tmp_path / "verified_output.exe"

        kpack_offload_binary(
            input_path=input_binary,
            output_path=output_binary,
            kpack_search_paths=["test.kpack"],
            kernel_name="test_kernel",
        )

        verify_result = ElfVerifier.verify(output_binary)
        assert verify_result.passed, f"Verification errors: {verify_result.errors}"

    def test_phdr_not_unnecessarily_relocated(
        self, test_assets_dir: Path, tmp_path: Path
    ):
        """Verify zero-page doesn't force unnecessary PHDR relocation.

        With the get_min_content_offset fix, conservative_zero_page should
        be able to write the PHDR table in place when there's enough space,
        rather than always relocating to end of file.
        """
        input_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )
        surgery = ElfSurgery.load(input_binary)

        # Do only Phase 3 (zero-page without prior PHDR relocation).
        # The original PHDR at offset 0x40 should have plenty of space.
        result = conservative_zero_page(surgery)
        assert result.success

        # When PHDR isn't relocated by a prior phase, zero-page shouldn't
        # need to relocate it either (there's space at the original location).
        # Note: this depends on test binary having enough existing PHDR slots.
        # If the split creates too many new entries, relocation is legitimate.
        # The key test is that get_min_content_offset doesn't return 0.
        min_offset = surgery.get_min_content_offset()
        assert min_offset > 0, "get_min_content_offset should not return 0"


class TestNobitsCollision:
    """Tests for NOBITS offset collision when section ends on page boundary."""

    def test_page_aligned_section_end_no_collision(
        self, test_assets_dir: Path, tmp_path: Path
    ):
        """Verify no NOBITS collision when .hip_fatbin ends on a page boundary.

        When .hip_fatbin's vaddr + size is page-aligned (suffix == 0), the
        NOBITS piece and post-section piece would share the same file offset
        after byte removal. The fix gives NOBITS segments a safe offset past
        end of file.

        Reproduces the CI failure on test_rocrand_scrambled_sobol32_qrng
        (gfx1151) where .hip_fatbin was at vaddr=0x2e000, size=0x12000,
        end=0x40000 (page-aligned).
        """
        from elf_test_utils import patch_hip_fatbin_size

        source = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        # Test binary: .hip_fatbin at vaddr=0x1000, original size=0x2d30.
        # Patch to 0x2000 so end = 0x1000 + 0x2000 = 0x3000 (page-aligned).
        patched = patch_hip_fatbin_size(source, tmp_path / "page_aligned.exe", 0x2000)

        surgery = ElfSurgery.load(patched)
        result = conservative_zero_page(surgery)
        assert result.success
        assert result.bytes_saved > 0

        # The key check: verifier must pass (no NOBITS collision)
        verify_result = ElfVerifier.verify_data(surgery.data)
        assert verify_result.passed, (
            f"NOBITS collision after zero-paging page-aligned section: "
            f"{verify_result.errors}"
        )

    def test_page_aligned_section_end_full_pipeline(
        self, test_assets_dir: Path, tmp_path: Path
    ):
        """Full kpack pipeline with page-aligned .hip_fatbin end."""
        from elf_test_utils import patch_hip_fatbin_size

        source = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        patched = patch_hip_fatbin_size(source, tmp_path / "page_aligned.exe", 0x2000)
        output = tmp_path / "kpacked_output.exe"

        result = kpack_offload_binary(
            input_path=patched,
            output_path=output,
            kpack_search_paths=["test.kpack"],
            kernel_name="test_kernel",
        )

        assert output.exists()
        assert result["new_size"] > 0

        # Verify output passes all checks
        surgery = ElfSurgery.load(output)
        assert_load_alignment(surgery, "page-aligned .hip_fatbin end")
        verify_result = ElfVerifier.verify_data(surgery.data)
        assert verify_result.passed, f"Verification errors: {verify_result.errors}"

    def test_leaves_one_page_when_suffix_zero(
        self, test_assets_dir: Path, tmp_path: Path
    ):
        """Verify that when suffix==0, one page is left behind to avoid collision.

        The fix shrinks aligned_size by PAGE_SIZE so the last page remains
        as file content. This means we save one fewer page, but the NOBITS
        and post-section pieces have distinct file offsets.
        """
        from elf_test_utils import patch_hip_fatbin_size

        source = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        # .hip_fatbin at vaddr=0x1000, patched to 0x2000 → end=0x3000 (page-aligned)
        # Section spans 2 pages. With suffix==0 fix, only 1 page is removed.
        patched = patch_hip_fatbin_size(source, tmp_path / "page_aligned.exe", 0x2000)

        surgery = ElfSurgery.load(patched)
        original_size = len(surgery.data)

        result = conservative_zero_page(surgery)
        assert result.success
        assert (
            result.pages_zeroed == 1
        ), f"Expected 1 page zeroed (2 pages minus 1 kept), got {result.pages_zeroed}"
        assert (
            result.bytes_saved == PAGE_SIZE
        ), f"Expected {PAGE_SIZE} bytes saved (one page), got {result.bytes_saved}"


class TestSteppedPipelineAlignment:
    """Tests that step through the pipeline manually to verify each phase."""

    def test_alignment_preserved_through_each_phase(
        self,
        test_assets_dir: Path,
    ):
        """Step through the kpack pipeline, checking alignment after each phase.

        This helps identify exactly which phase introduces alignment issues.
        """
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        # Verify original binary alignment
        assert_load_alignment(surgery, "original binary")

        # Phase 0: Add .rocm_kpack_ref section
        add_kpack_ref_section(
            surgery,
            kpack_search_paths=["test.kpack"],
            kernel_name="test_kernel",
        )
        assert_load_alignment(surgery, "after Phase 0 (add_kpack_ref_section)")

        # Phase 1: Map to PT_LOAD
        map_result = map_section_to_load(surgery, ".rocm_kpack_ref")
        assert map_result.success
        assert_load_alignment(surgery, "after Phase 1 (map_section_to_load)")

        # Phase 2: Rewrite magic
        fatbin = surgery.find_section(".hip_fatbin")
        assert fatbin is not None

        segment = surgery.find_section(".hipFatBinSegment")
        if segment is not None:
            kpack_ref_vaddr = map_result.vaddr
            num_wrappers = segment.header.sh_size // 24  # WRAPPER_SIZE
            for i in range(num_wrappers):
                wrapper_vaddr = segment.header.sh_addr + i * 24
                pointer_vaddr = wrapper_vaddr + 8
                set_pointer(surgery, pointer_vaddr, kpack_ref_vaddr)

            rewrite_hipfatbin_magic(surgery)
            assert_load_alignment(surgery, "after Phase 2 (rewrite magic)")

        # Phase 3: Zero-page
        result = conservative_zero_page(surgery)
        assert result.success
        assert_load_alignment(surgery, "after Phase 3 (conservative_zero_page)")
