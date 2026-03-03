"""Tests for kpack_transform PE/COFF operations."""

import struct
from pathlib import Path

import pytest

from rocm_kpack.coff import (
    CoffSurgery,
    CoffVerifier,
    kpack_offload_binary,
    add_kpack_ref_section,
    read_kpack_ref_marker,
    rewrite_hipfatbin_magic,
    verify_no_fatbin_relocations,
    NotFatBinaryError,
    HIPF_MAGIC,
    HIPK_MAGIC,
    WRAPPER_SIZE,
    SECTION_HIP_FATBIN,
    SECTION_HIP_FATBIN_SEGMENT,
    SECTION_KPACK_REF,
)


class TestKpackOffloadBinary:
    """Tests for kpack_offload_binary."""

    def test_kpack_fat_binary(self, tmp_path: Path, test_assets_dir: Path):
        """Test kpacking a fat binary with .hip_fat section."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_multi.exe"
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

        # Verify size reduction (PE binaries should reduce)
        new_size = output_binary.stat().st_size
        assert new_size < original_size
        assert result["removed"] > 0
        assert result["original_size"] == original_size
        assert result["new_size"] == new_size

        # Verify output is a valid PE file
        verification = CoffVerifier.verify(output_binary)
        assert verification.passed, f"Output verification failed: {verification}"

    def test_kpack_dll(self, tmp_path: Path, test_assets_dir: Path):
        """Test kpacking a DLL with .hip_fat section."""
        input_library = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.dll"
        )
        output_library = tmp_path / "test_kernel_single_kpacked.dll"

        original_size = input_library.stat().st_size

        result = kpack_offload_binary(
            input_path=input_library,
            output_path=output_library,
            kpack_search_paths=["../kpacks/test.kpack"],
            kernel_name="lib/test_kernel_single.dll",
        )

        assert output_library.exists()
        assert result["new_size"] > 0

        # Verify it's a valid DLL
        surgery = CoffSurgery.load(output_library)
        assert surgery.is_dll

        # Verify structural integrity
        verification = CoffVerifier.verify(output_library)
        assert verification.passed, f"Output verification failed: {verification}"

    def test_kpack_ref_marker_in_output(self, tmp_path: Path, test_assets_dir: Path):
        """Verify .kpackrf section is present in output."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        output_binary = tmp_path / "output.exe"

        kpack_offload_binary(
            input_path=input_binary,
            output_path=output_binary,
            kpack_search_paths=["test.kpack", "fallback.kpack"],
            kernel_name="my_kernel",
        )

        # Read back and verify marker section
        surgery = CoffSurgery.load(output_binary)
        section = surgery.find_section(SECTION_KPACK_REF)
        assert section is not None

        # Verify marker has correct content
        import msgpack

        content = surgery.get_section_content(section)
        # Slice to virtual_size (actual content), not raw_size (padded)
        content = content[: section.virtual_size]
        marker = msgpack.unpackb(content, raw=False)
        assert marker["kpack_search_paths"] == ["test.kpack", "fallback.kpack"]
        assert marker["kernel_name"] == "my_kernel"

    def test_kpack_multi_wrapper_dll(self, tmp_path: Path, test_assets_dir: Path):
        """Test kpacking RDC multi-wrapper DLL."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_multi_wrapper.dll"
        )
        output_binary = tmp_path / "test_multi_wrapper_kpacked.dll"

        result = kpack_offload_binary(
            input_path=input_binary,
            output_path=output_binary,
            kpack_search_paths=["kernels.kpack"],
            kernel_name="multi_wrapper",
            verbose=True,
        )

        assert output_binary.exists()

        # Verify output is valid
        verification = CoffVerifier.verify(output_binary)
        assert verification.passed, f"Output verification failed: {verification}"


class TestAddKpackRefSection:
    """Tests for add_kpack_ref_section."""

    def test_add_section(self, test_assets_dir: Path):
        """Test adding .kpackrf section."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )

        surgery = CoffSurgery.load(input_binary)

        # Add section
        section = add_kpack_ref_section(
            surgery,
            kpack_search_paths=["kernels.kpack"],
            kernel_name="test_kernel",
        )

        # Verify section was added
        assert section.name == SECTION_KPACK_REF

        # Read back marker
        import msgpack

        content = surgery.get_section_content(section)
        # Slice to virtual_size (actual content), not raw_size (padded)
        content = content[: section.virtual_size]
        marker = msgpack.unpackb(content, raw=False)
        assert marker["kpack_search_paths"] == ["kernels.kpack"]
        assert marker["kernel_name"] == "test_kernel"

    def test_add_section_raises_if_exists(self, test_assets_dir: Path, tmp_path: Path):
        """Test that adding duplicate section raises error."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        output_binary = tmp_path / "output.exe"

        # First, create a binary with the section already present
        surgery = CoffSurgery.load(input_binary)
        add_kpack_ref_section(
            surgery,
            kpack_search_paths=["first.kpack"],
            kernel_name="first",
        )
        surgery.save(output_binary)

        # Now try to add again
        surgery2 = CoffSurgery.load(output_binary)
        with pytest.raises(ValueError, match="already exists"):
            add_kpack_ref_section(
                surgery2,
                kpack_search_paths=["second.kpack"],
                kernel_name="second",
            )


class TestReadKpackRefMarker:
    """Tests for read_kpack_ref_marker."""

    def test_read_marker_from_transformed(self, tmp_path: Path, test_assets_dir: Path):
        """Test reading marker from transformed binary."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )
        output_binary = tmp_path / "output.exe"

        kpack_offload_binary(
            input_path=input_binary,
            output_path=output_binary,
            kpack_search_paths=["path/to/kernels.kpack"],
            kernel_name="my_kernel_id",
        )

        marker = read_kpack_ref_marker(output_binary)
        assert marker is not None
        assert marker["kpack_search_paths"] == ["path/to/kernels.kpack"]
        assert marker["kernel_name"] == "my_kernel_id"

    def test_read_marker_returns_none_if_missing(self, test_assets_dir: Path):
        """Test reading marker from untransformed binary returns None."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )

        marker = read_kpack_ref_marker(input_binary)
        assert marker is None


class TestRewriteHipfatbinMagic:
    """Tests for rewrite_hipfatbin_magic."""

    def test_rewrite_magic(self, test_assets_dir: Path):
        """Test that HIPF magic is rewritten to HIPK."""
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )

        surgery = CoffSurgery.load(input_binary)

        # Find .hipFatB section and check it has HIPF magic
        segment = surgery.find_section(SECTION_HIP_FATBIN_SEGMENT)
        assert segment is not None

        content_before = surgery.get_section_content(segment)
        # First 4 bytes should be HIPF magic
        magic_before = int.from_bytes(content_before[:4], "little")
        assert magic_before == HIPF_MAGIC

        # Rewrite magic
        count = rewrite_hipfatbin_magic(surgery)
        assert count >= 1

        # Check magic was changed
        content_after = surgery.get_section_content(segment)
        magic_after = int.from_bytes(content_after[:4], "little")
        assert magic_after == HIPK_MAGIC

    def test_wrapper_index_in_reserved1(self, test_assets_dir: Path):
        """Test that wrapper index is written to reserved1 field (offset +16).

        For multi-TU support, CLR reads the bundle index from reserved1 and
        passes it to kpack_load_code_object as co_index. This index is used
        to look up the correct kernel in the kpack archive TOC.
        """
        input_binary = (
            test_assets_dir / "bundled_binaries/windows/cov5/test_kernel_single.exe"
        )

        surgery = CoffSurgery.load(input_binary)

        segment = surgery.find_section(SECTION_HIP_FATBIN_SEGMENT)
        assert segment is not None

        segment_offset = surgery.rva_to_file_offset(segment.rva)
        num_wrappers = segment.virtual_size // WRAPPER_SIZE

        # Rewrite magic (this also writes wrapper indices)
        count = rewrite_hipfatbin_magic(surgery)
        assert count >= 1

        # Verify each wrapper has correct index in reserved1 (offset +16)
        for i in range(num_wrappers):
            wrapper_offset = segment_offset + i * WRAPPER_SIZE
            reserved1 = struct.unpack_from("<I", surgery.data, wrapper_offset + 16)[0]
            assert reserved1 == i, (
                f"Wrapper {i} reserved1 field should be {i}, got {reserved1}. "
                "This index is used by CLR as co_index for kpack lookup."
            )


class TestNotFatBinary:
    """Tests for NotFatBinaryError."""

    def test_host_only_exe_raises(self, tmp_path: Path, test_assets_dir: Path):
        """Test that host-only executable raises NotFatBinaryError."""
        input_binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.exe"
        output_binary = tmp_path / "output.exe"

        with pytest.raises(NotFatBinaryError):
            kpack_offload_binary(
                input_path=input_binary,
                output_path=output_binary,
                kpack_search_paths=["test.kpack"],
                kernel_name="test",
            )

    def test_host_only_dll_raises(self, tmp_path: Path, test_assets_dir: Path):
        """Test that host-only DLL raises NotFatBinaryError."""
        input_binary = test_assets_dir / "bundled_binaries/windows/cov5/host_only.dll"
        output_binary = tmp_path / "output.dll"

        with pytest.raises(NotFatBinaryError):
            kpack_offload_binary(
                input_path=input_binary,
                output_path=output_binary,
                kpack_search_paths=["test.kpack"],
                kernel_name="test",
            )
