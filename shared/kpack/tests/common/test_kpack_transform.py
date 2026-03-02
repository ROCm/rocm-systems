"""
Common tests for kpack transformation using the generic API.

These tests use the format-agnostic rocm_kpack API and are parameterized
to run against both ELF (Linux) and PE/COFF (Windows) binaries.
"""

from pathlib import Path

import pytest

from rocm_kpack import (
    kpack_offload_binary,
    read_kpack_ref_marker,
    is_fat_binary,
    detect_binary_format,
    NotFatBinaryError,
)


class TestFormatDetection:
    """Tests for binary format detection."""

    def test_detect_format(self, single_arch_exe: Path, asset_config: tuple[str, str]):
        """Test that format is correctly detected."""
        platform, _ = asset_config
        expected_format = "coff" if platform == "windows" else "elf"
        assert detect_binary_format(single_arch_exe) == expected_format

    def test_is_fat_binary_true(self, single_arch_exe: Path):
        """Test that fat binaries are detected as such."""
        assert is_fat_binary(single_arch_exe) is True

    def test_is_fat_binary_false(self, host_only_exe: Path):
        """Test that host-only binaries are not fat binaries."""
        assert is_fat_binary(host_only_exe) is False


class TestKpackOffloadBinary:
    """Tests for generic kpack_offload_binary."""

    def test_transform_single_arch_exe(
        self, single_arch_exe: Path, tmp_path: Path, asset_config: tuple[str, str]
    ):
        """Test transforming a single-arch executable."""
        platform, _ = asset_config
        output = tmp_path / "output.exe"

        result = kpack_offload_binary(
            input_path=single_arch_exe,
            output_path=output,
            kpack_search_paths=["kernels.kpack"],
            kernel_name="test_kernel",
        )

        # Verify output exists
        assert output.exists()

        # Verify format is preserved
        expected_format = "coff" if platform == "windows" else "elf"
        assert result["format"] == expected_format

        # Verify result has expected keys
        assert "removed" in result
        assert "original_size" in result
        assert "new_size" in result

    def test_transform_multi_arch_exe(
        self, multi_arch_exe: Path, tmp_path: Path, asset_config: tuple[str, str]
    ):
        """Test transforming a multi-arch executable."""
        platform, _ = asset_config
        output = tmp_path / "output.exe"

        original_size = multi_arch_exe.stat().st_size

        result = kpack_offload_binary(
            input_path=multi_arch_exe,
            output_path=output,
            kpack_search_paths=["kernels.kpack"],
            kernel_name="test_kernel",
        )

        assert output.exists()
        assert result["original_size"] == original_size
        assert result["new_size"] == output.stat().st_size

    def test_transform_shared_lib(
        self, single_arch_lib: Path, tmp_path: Path, asset_config: tuple[str, str]
    ):
        """Test transforming a shared library."""
        platform, _ = asset_config
        ext = ".dll" if platform == "windows" else ".so"
        output = tmp_path / f"output{ext}"

        result = kpack_offload_binary(
            input_path=single_arch_lib,
            output_path=output,
            kpack_search_paths=["libs/kernels.kpack"],
            kernel_name="lib_kernel",
        )

        assert output.exists()
        assert result["new_size"] > 0


class TestKpackRefMarker:
    """Tests for marker section read/write."""

    def test_marker_roundtrip(self, single_arch_exe: Path, tmp_path: Path):
        """Test that marker can be written and read back."""
        output = tmp_path / "output.exe"

        kpack_paths = ["path/to/kernels.kpack", "fallback.kpack"]
        kernel_name = "my_test_kernel"

        kpack_offload_binary(
            input_path=single_arch_exe,
            output_path=output,
            kpack_search_paths=kpack_paths,
            kernel_name=kernel_name,
        )

        # Read marker back
        marker = read_kpack_ref_marker(output)

        assert marker is not None
        assert marker["kpack_search_paths"] == kpack_paths
        assert marker["kernel_name"] == kernel_name

    def test_marker_missing_in_original(self, single_arch_exe: Path):
        """Test that untransformed binaries have no marker."""
        marker = read_kpack_ref_marker(single_arch_exe)
        assert marker is None


class TestNotFatBinary:
    """Tests for NotFatBinaryError."""

    def test_host_only_exe_raises(self, host_only_exe: Path, tmp_path: Path):
        """Test that host-only executable raises NotFatBinaryError."""
        output = tmp_path / "output.exe"

        with pytest.raises(NotFatBinaryError):
            kpack_offload_binary(
                input_path=host_only_exe,
                output_path=output,
                kpack_search_paths=["test.kpack"],
                kernel_name="test",
            )

    def test_host_only_lib_raises(self, host_only_lib: Path, tmp_path: Path):
        """Test that host-only library raises NotFatBinaryError."""
        output = tmp_path / "output"

        with pytest.raises(NotFatBinaryError):
            kpack_offload_binary(
                input_path=host_only_lib,
                output_path=output,
                kpack_search_paths=["test.kpack"],
                kernel_name="test",
            )
