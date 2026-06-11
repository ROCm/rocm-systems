"""Tests for kpack_transform ELF operations."""

import struct
import subprocess
from pathlib import Path

import pytest

from rocm_kpack.elf import (
    ElfSurgery,
    kpack_offload_binary,
    add_kpack_ref_section,
    read_kpack_ref_marker,
    rewrite_hipfatbin_magic,
    verify_no_fatbin_relocations,
    NotFatBinaryError,
    HIPF_MAGIC,
    HIPK_MAGIC,
    WRAPPER_SIZE,
)


class TestKpackOffloadBinary:
    """Tests for kpack_offload_binary."""

    def test_kpack_fat_binary(self, tmp_path: Path, test_assets_dir: Path):
        """Test kpacking a fat binary with .hip_fatbin section."""
        input_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
        )
        output_binary = tmp_path / "test_kernel_multi_kpacked.exe"

        # Get original size
        original_size = input_binary.stat().st_size

        # Kpack with marker section in single pass
        result = kpack_offload_binary(
            input_path=input_binary,
            output_path=output_binary,
            kpack_search_paths=["test.kpack"],
            kernel_name="test_kernel",
            verbose=True,
        )

        # Verify output exists
        assert output_binary.exists()

        # Verify size reduction
        new_size = output_binary.stat().st_size
        assert new_size < original_size
        assert result["removed"] > 0
        assert result["original_size"] == original_size
        assert result["new_size"] == new_size

        # Verify output is a valid ELF file
        readelf_result = subprocess.run(
            ["readelf", "-h", output_binary], capture_output=True, text=True
        )
        assert readelf_result.returncode == 0
        assert "ELF" in readelf_result.stdout

        # Verify .hip_fatbin section is marked as NOBITS
        sections_result = subprocess.run(
            ["readelf", "-S", output_binary], capture_output=True, text=True
        )
        assert sections_result.returncode == 0
        # The section should now be NOBITS type
        if ".hip_fatbin" in sections_result.stdout:
            assert "NOBITS" in sections_result.stdout

    def test_kpack_shared_library(self, tmp_path: Path, test_assets_dir: Path):
        """Test kpacking a shared library with .hip_fatbin section."""
        input_library = (
            test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_single.so"
        )
        output_library = tmp_path / "libtest_kernel_single_kpacked.so"

        original_size = input_library.stat().st_size

        result = kpack_offload_binary(
            input_path=input_library,
            output_path=output_library,
            kpack_search_paths=["../kpacks/test.kpack"],
            kernel_name="lib/libtest_kernel_single.so",
        )

        assert output_library.exists()
        # Note: Small .hip_fatbin sections may not reduce size due to overhead
        # The important thing is the operation completes and produces valid output
        assert result["new_size"] > 0

        # Verify it's a valid shared library
        readelf_result = subprocess.run(
            ["readelf", "-h", output_library], capture_output=True, text=True
        )
        assert readelf_result.returncode == 0
        assert "DYN" in readelf_result.stdout  # Shared library

    def test_kpack_ref_marker_in_output(self, tmp_path: Path, test_assets_dir: Path):
        """Verify .rocm_kpack_ref section is present in output."""
        input_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )
        output_binary = tmp_path / "output.exe"

        kpack_offload_binary(
            input_path=input_binary,
            output_path=output_binary,
            kpack_search_paths=["test.kpack", "fallback.kpack"],
            kernel_name="my_kernel",
        )

        # Read back and verify marker section
        surgery = ElfSurgery.load(output_binary)
        section = surgery.find_section(".rocm_kpack_ref")
        assert section is not None

        # Verify marker has correct content
        import msgpack

        content = surgery.get_section_content(section)
        marker = msgpack.unpackb(content, raw=False)
        assert marker["kpack_search_paths"] == ["test.kpack", "fallback.kpack"]
        assert marker["kernel_name"] == "my_kernel"


class TestAddKpackRefSection:
    """Tests for add_kpack_ref_section."""

    def test_add_section(self, tmp_path: Path, test_assets_dir: Path):
        """Test adding .rocm_kpack_ref section."""
        input_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )

        surgery = ElfSurgery.load(input_binary)

        # Add section
        section = add_kpack_ref_section(
            surgery,
            kpack_search_paths=["kernels.kpack"],
            kernel_name="test_kernel",
        )

        # Verify section was added
        assert section.name == ".rocm_kpack_ref"

        # Read back marker
        import msgpack

        content = surgery.get_section_content(section)
        marker = msgpack.unpackb(content, raw=False)
        assert marker["kpack_search_paths"] == ["kernels.kpack"]
        assert marker["kernel_name"] == "test_kernel"


class TestRewriteHipfatbinMagic:
    """Tests for rewrite_hipfatbin_magic."""

    def test_rewrite_magic(self, tmp_path: Path, test_assets_dir: Path):
        """Test HIPF→HIPK magic rewrite."""
        input_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )

        surgery = ElfSurgery.load(input_binary)

        # Check if .hipFatBinSegment exists
        segment = surgery.find_section(".hipFatBinSegment")
        if segment is None:
            pytest.skip("No .hipFatBinSegment in test binary")

        # Read original magic
        segment_offset = surgery.vaddr_to_file_offset(segment.header.sh_addr)
        import struct

        original_magic = struct.unpack_from("<I", surgery.data, segment_offset)[0]
        assert original_magic == HIPF_MAGIC

        # Rewrite magic
        count = rewrite_hipfatbin_magic(surgery, verbose=True)
        assert count > 0

        # Verify magic was rewritten
        new_magic = struct.unpack_from("<I", surgery.data, segment_offset)[0]
        assert new_magic == HIPK_MAGIC

    def test_wrapper_index_in_reserved1(self, test_assets_dir: Path):
        """Test that wrapper index is written to reserved1 field (offset +16).

        For multi-TU support, CLR reads the bundle index from reserved1 and
        passes it to kpack_load_code_object as co_index. This index is used
        to look up the correct kernel in the kpack archive TOC.
        """
        input_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )

        surgery = ElfSurgery.load(input_binary)

        segment = surgery.find_section(".hipFatBinSegment")
        if segment is None:
            pytest.skip("No .hipFatBinSegment in test binary")

        segment_offset = surgery.vaddr_to_file_offset(segment.header.sh_addr)
        num_wrappers = segment.header.sh_size // WRAPPER_SIZE

        # Rewrite magic (this also writes wrapper indices)
        count = rewrite_hipfatbin_magic(surgery)
        assert count > 0

        # Verify each wrapper has correct index in reserved1 (offset +16)
        for i in range(num_wrappers):
            wrapper_offset = segment_offset + i * WRAPPER_SIZE
            reserved1 = struct.unpack_from("<I", surgery.data, wrapper_offset + 16)[0]
            assert reserved1 == i, (
                f"Wrapper {i} reserved1 field should be {i}, got {reserved1}. "
                "This index is used by CLR as co_index for kpack lookup."
            )


class TestSmallHipFatbinHandling:
    """Tests for handling small .hip_fatbin sections."""

    def test_small_section_does_not_fail(
        self, small_hip_fatbin_binary: Path, tmp_path: Path
    ):
        """Verify small .hip_fatbin sections don't cause failure."""
        output_binary = tmp_path / "output.exe"

        # Should complete without error
        result = kpack_offload_binary(
            input_path=small_hip_fatbin_binary,
            output_path=output_binary,
            kpack_search_paths=["test.kpack"],
            kernel_name="test_kernel",
        )

        assert output_binary.exists()
        # May not reduce size for small sections
        assert result["new_size"] > 0

    def test_marginal_section_completes(
        self, marginal_hip_fatbin_binary: Path, tmp_path: Path
    ):
        """Verify marginal .hip_fatbin sections complete successfully."""
        output_binary = tmp_path / "output.exe"

        result = kpack_offload_binary(
            input_path=marginal_hip_fatbin_binary,
            output_path=output_binary,
            kpack_search_paths=["test.kpack"],
            kernel_name="test_kernel",
        )

        assert output_binary.exists()
        # Size may increase due to overhead, but operation should complete
        assert result["new_size"] > 0


class TestKpackRefMarker:
    """Tests for add_kpack_ref_section and read_kpack_ref_marker."""

    def test_roundtrip(self, tmp_path: Path, test_assets_dir: Path):
        """Test adding and reading back a kpack ref marker."""
        source_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
        )
        marked_binary = tmp_path / "marked_binary.exe"

        kpack_paths = [".kpack/blas-gfx100X.kpack", ".kpack/torch-gfx100X.kpack"]
        kernel_name = "bin/test_kernel_multi.exe"

        surgery = ElfSurgery.load(source_binary)
        add_kpack_ref_section(
            surgery,
            kpack_search_paths=kpack_paths,
            kernel_name=kernel_name,
        )
        surgery.save(marked_binary)

        assert marked_binary.exists()
        assert marked_binary.stat().st_size > 0

        marker_data = read_kpack_ref_marker(marked_binary)
        assert marker_data is not None
        assert marker_data["kpack_search_paths"] == kpack_paths
        assert marker_data["kernel_name"] == kernel_name

    def test_with_host_only_binary(self, tmp_path: Path, test_assets_dir: Path):
        """Test adding marker to host-only binary (no .hip_fatbin section)."""
        source_binary = test_assets_dir / "bundled_binaries/linux/cov5/host_only.exe"
        marked_binary = tmp_path / "marked_host_only.exe"

        surgery = ElfSurgery.load(source_binary)
        add_kpack_ref_section(
            surgery,
            kpack_search_paths=[".kpack/runtime-gfx1100.kpack"],
            kernel_name="bin/host_only.exe",
        )
        surgery.save(marked_binary)

        marker_data = read_kpack_ref_marker(marked_binary)
        assert marker_data is not None
        assert marker_data["kpack_search_paths"] == [".kpack/runtime-gfx1100.kpack"]
        assert marker_data["kernel_name"] == "bin/host_only.exe"

    def test_with_shared_library(self, tmp_path: Path, test_assets_dir: Path):
        """Test adding marker to shared library."""
        source_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/libtest_kernel_single.so"
        )
        marked_binary = tmp_path / "marked_library.so"

        surgery = ElfSurgery.load(source_binary)
        add_kpack_ref_section(
            surgery,
            kpack_search_paths=[".kpack/hipblas-gfx1100.kpack"],
            kernel_name="lib/libtest_kernel_single.so",
        )
        surgery.save(marked_binary)

        marker_data = read_kpack_ref_marker(marked_binary)
        assert marker_data is not None
        assert marker_data["kernel_name"] == "lib/libtest_kernel_single.so"

    def test_multiple_search_paths(self, tmp_path: Path, test_assets_dir: Path):
        """Test marker with multiple kpack search paths."""
        source_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
        )
        marked_binary = tmp_path / "multi_search.exe"

        kpack_paths = [
            ".kpack/blas-gfx1100.kpack",
            ".kpack/blas-gfx100X.kpack",
            ".kpack/blas-gfx90a.kpack",
        ]

        surgery = ElfSurgery.load(source_binary)
        add_kpack_ref_section(
            surgery,
            kpack_search_paths=kpack_paths,
            kernel_name="bin/hipcc",
        )
        surgery.save(marked_binary)

        marker_data = read_kpack_ref_marker(marked_binary)
        assert marker_data is not None
        assert len(marker_data["kpack_search_paths"]) == 3
        assert marker_data["kpack_search_paths"] == kpack_paths

    def test_no_section_returns_none(self, test_assets_dir: Path):
        """Test reading marker from binary without .rocm_kpack_ref section."""
        source_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
        )

        marker_data = read_kpack_ref_marker(source_binary)
        assert marker_data is None

    def test_preserves_existing_sections(self, tmp_path: Path, test_assets_dir: Path):
        """Test that adding marker preserves existing sections like .hip_fatbin."""
        source_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
        )
        marked_binary = tmp_path / "preserve_sections.exe"

        # Verify source has .hip_fatbin section
        result = subprocess.run(
            ["readelf", "-S", str(source_binary)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert ".hip_fatbin" in result.stdout

        surgery = ElfSurgery.load(source_binary)
        add_kpack_ref_section(
            surgery,
            kpack_search_paths=[".kpack/test.kpack"],
            kernel_name="bin/test",
        )
        surgery.save(marked_binary)

        # Verify marked binary still has .hip_fatbin section
        result = subprocess.run(
            ["readelf", "-S", str(marked_binary)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert ".hip_fatbin" in result.stdout
        assert ".rocm_kpack_ref" in result.stdout

    def test_special_characters_in_paths(self, tmp_path: Path, test_assets_dir: Path):
        """Test marker with special characters in paths."""
        source_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_multi.exe"
        )
        marked_binary = tmp_path / "special_chars.exe"

        kpack_paths = [
            ".kpack/my-lib_v1.0.kpack",
            "../shared/.kpack/common-gfx100X.kpack",
        ]

        surgery = ElfSurgery.load(source_binary)
        add_kpack_ref_section(
            surgery,
            kpack_search_paths=kpack_paths,
            kernel_name="bin/my-binary_v2.exe",
        )
        surgery.save(marked_binary)

        marker_data = read_kpack_ref_marker(marked_binary)
        assert marker_data is not None
        assert marker_data["kpack_search_paths"] == kpack_paths
        assert marker_data["kernel_name"] == "bin/my-binary_v2.exe"


class TestDebugInfoSectionOrdering:
    """Regression tests for the ELF section reordering fix.

    libtest_multi_wrapper_with_debug.so is compiled with -g, so the ELF
    contains non-allocatable .debug_* sections alongside the HIP fat binary
    sections.  When kpack adds .rocm_kpack_ref (SHF_ALLOC) it must appear
    before .debug_* in both the section header table and in file offsets.

    Tools like dh_dwz (DWARF debuginfo compressor used in Debian packaging)
    abort with "Allocatable section after non-allocatable ones" if an ALLOC
    section follows a non-ALLOC section in the SHT.  The fix:
      1. Reorders the SHT so all ALLOC sections precede non-ALLOC ones.
      2. Splices the .rocm_kpack_ref content before the first non-ALLOC
         file offset so the physical layout matches the SHT order.
    """

    _BINARY = "bundled_binaries/linux/cov5/libtest_multi_wrapper_with_debug.so"

    @pytest.fixture
    def debug_binary(self, test_assets_dir: Path) -> Path:
        path = test_assets_dir / self._BINARY
        if not path.exists():
            pytest.skip(
                f"{self._BINARY} not found — run "
                "test_generation/build_test_bundles.py with ROCm on Linux"
            )
        return path

    def test_binary_has_debug_sections(self, debug_binary: Path):
        """Confirm the test asset actually contains .debug_* sections."""
        result = subprocess.run(
            ["readelf", "-S", str(debug_binary)], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert ".debug_info" in result.stdout, (
            "Expected .debug_info in debug binary — was it compiled without -g?"
        )

    def test_alloc_sections_precede_non_alloc_after_transform(
        self, debug_binary: Path, tmp_path: Path
    ):
        """After kpack transform, all SHF_ALLOC sections must precede non-ALLOC.

        Verifies the section header reordering fix: .rocm_kpack_ref (ALLOC)
        must appear before .debug_* (non-ALLOC) in the SHT.  If not, dh_dwz
        aborts during Debian package builds.
        """
        output = tmp_path / "libtest_multi_wrapper_with_debug_kpacked.so"

        kpack_offload_binary(
            input_path=debug_binary,
            output_path=output,
            kpack_search_paths=["../.kpack/blas_lib_@GFXARCH@.kpack"],
            kernel_name="lib/libtest_multi_wrapper_with_debug.so",
        )

        assert output.exists()

        # Precise check via ElfSurgery: walk sections in SHT order and verify
        # no ALLOC section appears after a non-ALLOC section with file content.
        surgery = ElfSurgery.load(output)
        SHT_NULL = 0
        SHT_NOBITS = 8

        seen_non_alloc_with_content = False
        for section in surgery.iter_sections():
            shdr = section.header
            if shdr.sh_type == SHT_NULL:
                continue
            has_content = shdr.sh_type != SHT_NOBITS and shdr.sh_size > 0
            if not shdr.is_alloc and has_content:
                seen_non_alloc_with_content = True
            if shdr.is_alloc and seen_non_alloc_with_content:
                pytest.fail(
                    f"SHF_ALLOC section '{section.name}' appears after non-ALLOC "
                    "sections in the section header table. dh_dwz will abort with "
                    "'Allocatable section after non-allocatable ones'."
                )

    def test_rocm_kpack_ref_file_offset_before_debug(
        self, debug_binary: Path, tmp_path: Path
    ):
        """The .rocm_kpack_ref content must be at a lower file offset than .debug_info.

        The ELF section-reordering fix also splices the .rocm_kpack_ref content
        before the first non-ALLOC section in the file.  This ensures both the
        SHT order and physical file layout are correct.
        """
        output = tmp_path / "libtest_debug_offset_check.so"

        kpack_offload_binary(
            input_path=debug_binary,
            output_path=output,
            kpack_search_paths=["../.kpack/blas_lib_@GFXARCH@.kpack"],
            kernel_name="lib/libtest_multi_wrapper_with_debug.so",
        )

        surgery = ElfSurgery.load(output)
        SHT_NOBITS = 8

        kpack_ref = surgery.find_section(".rocm_kpack_ref")
        assert kpack_ref is not None, ".rocm_kpack_ref section must exist after transform"

        debug_info = surgery.find_section(".debug_info")
        assert debug_info is not None, ".debug_info must exist in debug binary"

        assert kpack_ref.header.sh_offset < debug_info.header.sh_offset, (
            f".rocm_kpack_ref file offset (0x{kpack_ref.header.sh_offset:x}) "
            f"must be less than .debug_info offset (0x{debug_info.header.sh_offset:x}). "
            "The splice fix must insert the ALLOC section before non-ALLOC content."
        )


class TestNotFatBinaryError:
    """Tests for NotFatBinaryError handling."""

    def test_raises_on_host_only_binary(self, tmp_path: Path, test_assets_dir: Path):
        """Test that kpack_offload_binary raises NotFatBinaryError for host-only binaries."""
        # host_only.exe has no .hip_fatbin section
        host_only = test_assets_dir / "bundled_binaries/linux/cov5/host_only.exe"
        output = tmp_path / "output.exe"

        with pytest.raises(NotFatBinaryError) as exc_info:
            kpack_offload_binary(
                input_path=host_only,
                output_path=output,
                kpack_search_paths=["test.kpack"],
                kernel_name="test",
            )

        # Verify exception message is informative
        assert "no .hip_fatbin section" in str(exc_info.value)
        assert "host_only.exe" in str(exc_info.value)

        # Verify output was NOT created (no modifications made)
        assert not output.exists()

    def test_raises_on_shared_library_without_fatbin(
        self, tmp_path: Path, test_assets_dir: Path
    ):
        """Test that kpack_offload_binary raises NotFatBinaryError for host-only libraries."""
        host_only_lib = test_assets_dir / "bundled_binaries/linux/cov5/libhost_only.so"
        output = tmp_path / "output.so"

        with pytest.raises(NotFatBinaryError):
            kpack_offload_binary(
                input_path=host_only_lib,
                output_path=output,
                kpack_search_paths=["test.kpack"],
                kernel_name="test",
            )

        # Verify output was NOT created
        assert not output.exists()

    def test_no_error_without_kpack_search_paths(
        self, tmp_path: Path, test_assets_dir: Path
    ):
        """Test that NotFatBinaryError is NOT raised when kpack_search_paths is None.

        When kpack_search_paths is None, the function assumes the binary already
        has a .rocm_kpack_ref section. The check is only for new marker additions.
        """
        # First add a marker to a host-only binary
        host_only = test_assets_dir / "bundled_binaries/linux/cov5/host_only.exe"
        marked = tmp_path / "marked.exe"

        surgery = ElfSurgery.load(host_only)
        add_kpack_ref_section(
            surgery,
            kpack_search_paths=["test.kpack"],
            kernel_name="test",
        )
        surgery.save(marked)

        # Now try to process it without kpack_search_paths
        # This would typically be done to re-process an already-marked binary
        output = tmp_path / "output.exe"

        # Should NOT raise NotFatBinaryError - it processes binaries with existing markers
        # Note: This will fail at a later stage (no .hip_fatbin to zero-page) but
        # importantly it doesn't raise NotFatBinaryError
        kpack_offload_binary(
            input_path=marked,
            output_path=output,
            kpack_search_paths=None,
            kernel_name=None,
        )
