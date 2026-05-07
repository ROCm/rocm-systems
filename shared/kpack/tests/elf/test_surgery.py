"""
Unit tests for the rocm_kpack.elf package.

Tests the new ELF surgery infrastructure against the test_assets binaries.
"""
import pytest
import subprocess
from pathlib import Path

from rocm_kpack.elf import (
    ElfSurgery,
    ElfVerifier,
    ProgramHeaderManager,
    map_section_to_load,
    set_pointer,
    conservative_zero_page,
    zero_page_section,
    verify_all,
    verify_with_readelf,
    create_load_segment,
    PT_LOAD,
    SHF_ALLOC,
)
from rocm_kpack.elf.types import ProgramHeader, page_align_offset


class TestElfSurgery:
    """Tests for the ElfSurgery class."""

    def test_load_binary(self, test_assets_dir: Path):
        """Test loading a valid ELF binary."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        assert surgery.ehdr is not None
        assert surgery.is_pie_or_shared  # PIE executable

    def test_find_section(self, test_assets_dir: Path):
        """Test finding sections by name."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        # Should find .hip_fatbin
        section = surgery.find_section(".hip_fatbin")
        assert section is not None
        assert section.name == ".hip_fatbin"
        assert section.size > 0

        # Should not find nonexistent section
        assert surgery.find_section(".nonexistent") is None

    def test_find_phdr_containing_vaddr(self, test_assets_dir: Path):
        """Test finding program header by virtual address."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        # Find the entry point's segment
        entry = surgery.ehdr.e_entry
        result = surgery.find_phdr_containing_vaddr(entry)
        assert result is not None
        idx, phdr = result
        assert phdr.p_type == PT_LOAD

    def test_iter_sections(self, test_assets_dir: Path):
        """Test iterating over all sections."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        sections = list(surgery.iter_sections())
        assert len(sections) > 0

        # Check we have common sections
        names = [s.name for s in sections]
        assert ".text" in names
        assert ".hip_fatbin" in names


class TestElfVerifier:
    """Tests for the ElfVerifier class."""

    def test_verify_valid_binary(self, test_assets_dir: Path):
        """Test verification of a valid binary passes."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        result = ElfVerifier.verify(binary)
        assert result.passed

    def test_verify_detects_nobits_collision(self, test_assets_dir: Path):
        """Test that verification detects NOBITS offset collision."""
        # Load a valid binary and manually inject a collision
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        # Find an existing PT_LOAD segment
        original_phdr = None
        for idx, phdr in surgery.iter_program_headers():
            if phdr.p_type == PT_LOAD and phdr.p_filesz > 0:
                original_phdr = phdr
                break

        assert original_phdr is not None, "Need a PT_LOAD segment for this test"

        # Create a NOBITS segment (filesz=0) at the same offset - this is invalid
        collision_phdr = ProgramHeader(
            p_type=PT_LOAD,
            p_flags=original_phdr.p_flags,
            p_offset=original_phdr.p_offset,  # Same offset!
            p_vaddr=original_phdr.p_vaddr + 0x100000,  # Different vaddr
            p_paddr=original_phdr.p_paddr + 0x100000,
            p_filesz=0,  # NOBITS - no file content
            p_memsz=0x1000,  # But has memory size
            p_align=original_phdr.p_align,
        )

        # Inject the collision
        surgery._phdrs.append(collision_phdr)
        surgery._ehdr.e_phnum += 1

        # Now verify should detect the collision
        verifier = ElfVerifier(surgery)
        result = verifier.check_no_overlapping_load_segments()
        assert not result.passed, "Should detect NOBITS collision"
        assert any("NOBITS collision" in e for e in result.errors)


class TestMapSectionToLoad:
    """Tests for the map_section_to_load operation."""

    def test_map_non_alloc_section(self, test_assets_dir: Path, tmp_path: Path):
        """Test mapping a non-ALLOC section to PT_LOAD."""
        # Use host_only binary which doesn't have .hip_fatbin
        binary = test_assets_dir / "bundled_binaries/linux/cov5/libhost_only.so"
        surgery = ElfSurgery.load(binary)

        # Find a non-ALLOC section to test with
        sections = list(surgery.iter_sections())
        non_alloc = [s for s in sections if not s.is_alloc and s.size > 0]

        assert non_alloc, (
            "Test binary must have at least one non-ALLOC section with content. "
            "Available sections: " + ", ".join(s.name for s in sections)
        )

        section = non_alloc[0]
        result = map_section_to_load(surgery, section.name)

        assert result.success, f"map_section_to_load failed: {result.error}"
        assert result.vaddr > 0

        # Verify section now has ALLOC flag
        updated = surgery.find_section(section.name)
        assert updated is not None
        assert updated.is_alloc


class TestZeroPage:
    """Tests for zero-page optimization."""

    def test_zero_page_hip_fatbin(self, test_assets_dir: Path, tmp_path: Path):
        """Test zero-paging .hip_fatbin section."""
        input_path = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )
        output_path = tmp_path / "zero_paged.exe"

        result = zero_page_section(input_path, output_path)

        assert result.success
        assert result.bytes_saved > 0
        assert result.pages_zeroed > 0
        assert output_path.stat().st_size < input_path.stat().st_size

    def test_zero_page_no_nobits_collision(self, test_assets_dir: Path, tmp_path: Path):
        """Test that zero-page doesn't create NOBITS offset collision."""
        input_path = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )
        output_path = tmp_path / "zero_paged.exe"

        result = zero_page_section(input_path, output_path)
        assert result.success

        # Verify with our internal verifier
        verify_result = ElfVerifier.verify(output_path)
        assert verify_result.passed, f"Errors: {verify_result.errors}"

    def test_zero_page_readelf_valid(self, test_assets_dir: Path, tmp_path: Path):
        """Test that zero-paged binary is valid according to readelf."""
        input_path = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )
        output_path = tmp_path / "zero_paged.exe"

        result = zero_page_section(input_path, output_path)
        assert result.success

        # Run readelf and check for errors
        verify_result = verify_with_readelf(output_path)
        assert verify_result.passed, f"Errors: {verify_result.errors}"

    def test_zero_page_small_section(self, test_assets_dir: Path, tmp_path: Path):
        """Test that small sections are handled gracefully."""
        # Create a patched binary with small .hip_fatbin
        from elf_test_utils import patch_hip_fatbin_size

        input_path = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )
        patched_path = tmp_path / "small_fatbin.exe"
        output_path = tmp_path / "zero_paged.exe"

        # Patch to 3000 bytes (less than one page)
        patch_hip_fatbin_size(input_path, patched_path, new_size=3000)

        # Should succeed but not save bytes (no full pages)
        result = zero_page_section(patched_path, output_path)
        assert result.success
        # No pages to zero for section < PAGE_SIZE
        assert result.pages_zeroed == 0 or result.bytes_saved == 0

    def test_zero_page_already_nobits(self, test_assets_dir: Path, tmp_path: Path):
        """Test that zero-paging an already zero-paged binary is safe."""
        input_path = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )
        first_output = tmp_path / "first_zero_page.exe"
        second_output = tmp_path / "second_zero_page.exe"

        # First zero-page should succeed
        result1 = zero_page_section(input_path, first_output)
        assert result1.success
        assert result1.bytes_saved > 0

        # Second zero-page should succeed but do nothing
        result2 = zero_page_section(first_output, second_output)
        assert result2.success
        assert result2.bytes_saved == 0
        assert result2.pages_zeroed == 0
        assert "already NOBITS" in (result2.error or "")

        # Output should not be created (or if created, should match input)
        # Actually, zero_page_section only saves on success with changes
        # Let's verify the first output is still valid
        verify_result = ElfVerifier.verify(first_output)
        assert verify_result.passed, f"Errors: {verify_result.errors}"


class TestProgramHeaderManager:
    """Tests for ProgramHeaderManager."""

    def test_add_program_header(self, test_assets_dir: Path):
        """Test adding a program header."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        original_count = surgery.ehdr.e_phnum
        manager = ProgramHeaderManager(surgery)

        # Allocate vaddr first, then align file offset to match
        vaddr = manager.allocate_vaddr()
        file_offset = page_align_offset(len(surgery.data), vaddr)

        # Add a dummy PT_LOAD
        new_phdr = create_load_segment(
            vaddr=vaddr,
            file_offset=file_offset,
            size=0x1000,
        )
        manager.add_program_header(new_phdr)

        assert len(manager.program_headers) == original_count + 1

    def test_allocate_vaddr(self, test_assets_dir: Path):
        """Test virtual address allocation."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        surgery = ElfSurgery.load(binary)

        manager = ProgramHeaderManager(surgery)
        vaddr = manager.allocate_vaddr()

        # Should be page-aligned and after all existing segments
        assert vaddr % 0x1000 == 0
        assert vaddr >= manager.get_max_vaddr()


class TestVerifyAll:
    """Tests for the verify_all function."""

    def test_verify_all_valid_binary(self, test_assets_dir: Path, tmp_path: Path):
        """Test that verify_all passes on valid binary."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"

        result = verify_all(binary, tmp_path)
        assert result.passed, f"Errors: {result.errors}"


class TestAddSection:
    """Tests for ElfSurgery.add_section()."""

    def test_add_section_basic(self, tmp_path: Path, test_assets_dir: Path):
        """Test adding a new section to an ELF binary."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        output = tmp_path / "with_section.exe"

        surgery = ElfSurgery.load(binary)

        # Add a new section
        content = b"test section content"
        result = surgery.add_section(
            name=".test_section",
            content=content,
        )

        # Verify result
        assert result.index > 0
        assert result.offset > 0

        # Save and reload
        surgery.save(output)
        surgery2 = ElfSurgery.load(output)

        # Find the section
        section = surgery2.find_section(".test_section")
        assert section is not None
        assert surgery2.get_section_content(section) == content

    def test_add_section_preserves_existing(
        self, tmp_path: Path, test_assets_dir: Path
    ):
        """Test that adding a section preserves existing sections."""
        binary = test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
        output = tmp_path / "with_section.exe"

        surgery = ElfSurgery.load(binary)

        # Get original section count
        original_section_count = surgery.ehdr.e_shnum

        # Verify .hip_fatbin exists
        fatbin = surgery.find_section(".hip_fatbin")
        assert fatbin is not None
        original_fatbin_size = fatbin.header.sh_size

        # Add a new section
        surgery.add_section(name=".my_section", content=b"hello world")
        surgery.save(output)

        # Reload and verify
        surgery2 = ElfSurgery.load(output)
        assert surgery2.ehdr.e_shnum == original_section_count + 1

        # Verify .hip_fatbin preserved
        fatbin2 = surgery2.find_section(".hip_fatbin")
        assert fatbin2 is not None
        assert fatbin2.header.sh_size == original_fatbin_size

        # Verify new section exists
        new_section = surgery2.find_section(".my_section")
        assert new_section is not None
        assert surgery2.get_section_content(new_section) == b"hello world"

    def test_add_section_with_flags(self, tmp_path: Path, test_assets_dir: Path):
        """Test adding a section with specific flags."""
        from rocm_kpack.elf import SHT_NOTE, SHF_ALLOC

        binary = test_assets_dir / "bundled_binaries/linux/cov5/host_only.exe"
        output = tmp_path / "with_note.exe"

        surgery = ElfSurgery.load(binary)
        surgery.add_section(
            name=".my_note",
            content=b"note content here",
            section_type=SHT_NOTE,
            flags=SHF_ALLOC,
            addralign=4,
        )
        surgery.save(output)

        surgery2 = ElfSurgery.load(output)
        section = surgery2.find_section(".my_note")
        assert section is not None
        assert section.header.sh_type == SHT_NOTE
        assert section.header.sh_flags & SHF_ALLOC
        assert section.header.sh_addralign == 4
