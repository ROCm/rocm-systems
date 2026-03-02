"""
Unit tests for artifact_utils module.

Tests the common utilities for artifact manipulation including manifest
handling, directory traversal, and file classification.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from rocm_kpack.artifact_utils import (
    read_artifact_manifest,
    write_artifact_manifest,
    scan_directory,
    is_fat_binary,
    extract_architecture_from_target,
)
from rocm_kpack.binutils import Toolchain


class TestArtifactManifest:
    """Tests for artifact manifest reading and writing."""

    def test_read_artifact_manifest_simple(self, tmp_path):
        """Test reading a simple artifact manifest."""
        manifest_path = tmp_path / "artifact_manifest.txt"
        manifest_path.write_text("prefix1/stage\nprefix2/stage\n")

        prefixes = read_artifact_manifest(tmp_path)
        assert prefixes == ["prefix1/stage", "prefix2/stage"]

    def test_read_artifact_manifest_with_empty_lines(self, tmp_path):
        """Test reading manifest with empty lines (should ignore them)."""
        manifest_path = tmp_path / "artifact_manifest.txt"
        manifest_path.write_text("prefix1/stage\n\nprefix2/stage\n\n")

        prefixes = read_artifact_manifest(tmp_path)
        assert prefixes == ["prefix1/stage", "prefix2/stage"]

    def test_read_artifact_manifest_missing_file(self, tmp_path):
        """Test reading from directory without manifest file."""
        with pytest.raises(FileNotFoundError, match="artifact_manifest.txt not found"):
            read_artifact_manifest(tmp_path)

    def test_write_artifact_manifest(self, tmp_path):
        """Test writing an artifact manifest."""
        prefixes = ["math-libs/BLAS/rocBLAS/stage", "kpack/stage"]
        write_artifact_manifest(tmp_path, prefixes)

        manifest_path = tmp_path / "artifact_manifest.txt"
        assert manifest_path.exists()

        content = manifest_path.read_text()
        assert content == "math-libs/BLAS/rocBLAS/stage\nkpack/stage\n"

    def test_write_empty_manifest(self, tmp_path):
        """Test writing an empty manifest."""
        write_artifact_manifest(tmp_path, [])

        manifest_path = tmp_path / "artifact_manifest.txt"
        assert manifest_path.exists()
        assert manifest_path.read_text() == ""

    def test_roundtrip_manifest(self, tmp_path):
        """Test reading and writing manifest preserves data."""
        original_prefixes = [
            "math-libs/BLAS/rocBLAS/stage",
            "math-libs/BLAS/hipBLASLt/stage",
            "kpack/stage",
        ]

        write_artifact_manifest(tmp_path, original_prefixes)
        read_prefixes = read_artifact_manifest(tmp_path)

        assert read_prefixes == original_prefixes


class TestScanDirectory:
    """Tests for robust directory scanning."""

    def test_scan_simple_directory(self, tmp_path):
        """Test scanning a simple directory structure."""
        # Create test structure
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "file1.txt").write_text("content1")
        (tmp_path / "dir2").mkdir()
        (tmp_path / "dir2" / "file2.txt").write_text("content2")
        (tmp_path / "file3.txt").write_text("content3")

        # Scan and collect results
        results = list(scan_directory(tmp_path))

        # Check we found all files and directories
        # Use as_posix() for consistent forward-slash paths on all platforms
        paths = [p.relative_to(tmp_path).as_posix() for p, _ in results]
        assert "dir1" in paths
        assert "dir2" in paths
        assert "file3.txt" in paths
        assert "dir1/file1.txt" in paths
        assert "dir2/file2.txt" in paths

    def test_scan_with_symlinks(self, tmp_path):
        """Test that scanning doesn't follow symlinks."""
        # Create a directory with a file
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        (real_dir / "file.txt").write_text("content")

        # Create a symlink to the directory
        symlink = tmp_path / "symlink_dir"
        symlink.symlink_to(real_dir)

        # Create another directory that contains a symlink
        container = tmp_path / "container"
        container.mkdir()
        (container / "link_to_real").symlink_to(real_dir)

        # Scan the container directory
        results = list(scan_directory(container))

        # Should find the symlink itself but not traverse into it
        paths = [str(p.relative_to(container)) for p, _ in results]
        assert "link_to_real" in paths
        assert "link_to_real/file.txt" not in paths

    def test_scan_with_predicate(self, tmp_path):
        """Test scanning with a filter predicate."""
        # Create mixed files
        (tmp_path / "file1.txt").write_text("text")
        (tmp_path / "file2.py").write_text("python")
        (tmp_path / "file3.txt").write_text("more text")
        (tmp_path / "file4.cpp").write_text("c++")

        # Define predicate for .txt files only
        def txt_only(path, entry):
            return path.suffix == ".txt" or entry.is_dir()

        # Scan with predicate
        results = list(scan_directory(tmp_path, predicate=txt_only))

        # Check we only got .txt files
        file_paths = [p for p, e in results if e.is_file()]
        assert len(file_paths) == 2
        assert all(p.suffix == ".txt" for p in file_paths)

    def test_scan_handles_permission_errors(self, tmp_path):
        """Test that scanning fails fast on permission errors."""
        # Create accessible directory
        accessible = tmp_path / "accessible"
        accessible.mkdir()
        (accessible / "file.txt").write_text("content")

        # Mock os.scandir to raise PermissionError for specific path
        original_scandir = os.scandir

        def mock_scandir(path):
            if "forbidden" in str(path):
                raise PermissionError("Access denied")
            return original_scandir(path)

        with patch("os.scandir", side_effect=mock_scandir):
            # Create a "forbidden" directory (won't actually restrict, mock will)
            forbidden = tmp_path / "forbidden"
            forbidden.mkdir()

            # Scan should fail fast on permission error
            with pytest.raises(PermissionError, match="Access denied"):
                list(scan_directory(tmp_path))

    def test_scan_empty_directory(self, tmp_path):
        """Test scanning an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        results = list(scan_directory(empty_dir))

        # Should return empty list for empty directory
        assert results == []


class TestIsFatBinary:
    """Tests for fat binary detection.

    These tests verify that is_fat_binary() correctly identifies:
    - Fat binaries (ELF files with .hip_fatbin section) -> True
    - Host-only binaries (ELF files without .hip_fatbin) -> False
    - Non-ELF files -> False
    - Empty files -> False
    - Missing files -> raises FileNotFoundError
    """

    def test_is_fat_binary_empty_file(self, tmp_path, toolchain):
        """Test that empty files are correctly identified as not fat binaries.

        Empty files can occur in CI/build artifacts due to:
        - Incomplete downloads
        - Build failures that leave empty placeholder files
        - Test artifacts that weren't properly generated
        """
        empty_file = tmp_path / "empty.so"
        empty_file.write_bytes(b"")  # Create empty file

        # Should return False immediately without calling readelf
        result = is_fat_binary(empty_file, toolchain)
        assert result is False

    def test_is_fat_binary_missing_file(self, tmp_path, toolchain):
        """Test that missing files raise FileNotFoundError."""
        missing_file = tmp_path / "nonexistent.so"

        with pytest.raises(FileNotFoundError):
            is_fat_binary(missing_file, toolchain)

    def test_is_fat_binary_host_only_exe(self, test_assets_dir, toolchain):
        """Test that host-only ELF executables (no .hip_fatbin) return False.

        Uses the pre-generated host_only.exe from test_assets which is a
        valid ELF executable without any GPU device code.
        """
        host_only = test_assets_dir / "bundled_binaries/linux/cov5/host_only.exe"
        assert host_only.exists(), f"Test asset not found: {host_only}"

        result = is_fat_binary(host_only, toolchain)
        assert result is False

    def test_is_fat_binary_host_only_so(self, test_assets_dir, toolchain):
        """Test that host-only shared libraries (no .hip_fatbin) return False.

        Uses the pre-generated libhost_only.so from test_assets which is a
        valid ELF shared library without any GPU device code.
        """
        host_only = test_assets_dir / "bundled_binaries/linux/cov5/libhost_only.so"
        assert host_only.exists(), f"Test asset not found: {host_only}"

        result = is_fat_binary(host_only, toolchain)
        assert result is False

    def test_is_fat_binary_with_hip_fatbin_real(self, test_assets_dir, toolchain):
        """Test that fat binaries (with .hip_fatbin) return True.

        Uses the pre-generated test_kernel_single.exe from test_assets which
        is a valid ELF executable with GPU device code.
        """
        fat_binary = (
            test_assets_dir / "bundled_binaries/linux/cov5/test_kernel_single.exe"
        )
        assert fat_binary.exists(), f"Test asset not found: {fat_binary}"

        result = is_fat_binary(fat_binary, toolchain)
        assert result is True

    def test_is_fat_binary_with_hip_fatbin(self, tmp_path):
        """Test detecting a binary with .hip_fatbin section."""
        # Create a real file with ELF magic bytes
        elf_file = tmp_path / "binary.so"
        elf_file.write_bytes(b"\x7fELF" + b"\x00" * 100)  # ELF magic + padding

        mock_toolchain = Mock(spec=Toolchain)
        mock_toolchain.readelf = "/usr/bin/readelf"

        # Mock subprocess to return output with .hip_fatbin
        with patch("subprocess.check_output") as mock_check:
            mock_check.return_value = """
                Section Headers:
                  [Nr] Name              Type
                  [ 0]                   NULL
                  [ 1] .text             PROGBITS
                  [ 2] .hip_fatbin       PROGBITS
                  [ 3] .data             PROGBITS
            """

            result = is_fat_binary(elf_file, mock_toolchain)
            assert result is True

    def test_is_fat_binary_without_hip_fatbin(self, tmp_path):
        """Test detecting a regular binary without .hip_fatbin."""
        # Create a real file with ELF magic bytes
        elf_file = tmp_path / "binary.so"
        elf_file.write_bytes(b"\x7fELF" + b"\x00" * 100)  # ELF magic + padding

        mock_toolchain = Mock(spec=Toolchain)
        mock_toolchain.readelf = "/usr/bin/readelf"

        with patch("subprocess.check_output") as mock_check:
            mock_check.return_value = """
                Section Headers:
                  [Nr] Name              Type
                  [ 0]                   NULL
                  [ 1] .text             PROGBITS
                  [ 2] .data             PROGBITS
            """

            result = is_fat_binary(elf_file, mock_toolchain)
            assert result is False

    def test_is_fat_binary_not_elf(self, tmp_path):
        """Test handling non-ELF files."""
        # Create a real non-ELF file (text file)
        text_file = tmp_path / "text.txt"
        text_file.write_text("This is not an ELF file")

        mock_toolchain = Mock(spec=Toolchain)
        mock_toolchain.readelf = "/usr/bin/readelf"

        # Should return False immediately after checking magic bytes, without calling readelf
        result = is_fat_binary(text_file, mock_toolchain)
        assert result is False

    def test_is_fat_binary_readelf_not_found(self, tmp_path):
        """Test handling when readelf is not available."""
        # Create a real file with ELF magic bytes
        elf_file = tmp_path / "binary.so"
        elf_file.write_bytes(b"\x7fELF" + b"\x00" * 100)  # ELF magic + padding

        mock_toolchain = Mock(spec=Toolchain)
        mock_toolchain.readelf = "/nonexistent/readelf"

        with patch("subprocess.check_output") as mock_check:
            mock_check.side_effect = FileNotFoundError("No such file")

            # Should raise RuntimeError when readelf is not found
            with pytest.raises(RuntimeError, match="readelf not found"):
                is_fat_binary(elf_file, mock_toolchain)


class TestExtractArchitectureFromTarget:
    """Tests for extracting architecture from target strings."""

    def test_extract_simple_architecture(self):
        """Test extracting simple architecture."""
        target = "hipv4-amdgcn-amd-amdhsa--gfx906"
        arch = extract_architecture_from_target(target)
        assert arch == "gfx906"

    def test_extract_cooked_architecture(self):
        """Test extracting 'cooked' architecture with features."""
        target = "hipv4-amdgcn-amd-amdhsa--gfx942:xnack+"
        arch = extract_architecture_from_target(target)
        assert arch == "gfx942:xnack+"

    def test_extract_with_single_dash(self):
        """Test target with single dash separator."""
        target = "hip-amdgcn-amd-amdhsa-gfx1100"
        arch = extract_architecture_from_target(target)
        # Should not find architecture with single dash
        assert arch is None

    def test_extract_architecture_various_formats(self):
        """Test various target format patterns."""
        test_cases = [
            ("hipv4-amdgcn-amd-amdhsa--gfx1100", "gfx1100"),
            ("openmp-amdgcn-amd-amdhsa--gfx90a", "gfx90a"),
            ("hipv4-amdgcn-amd-amdhsa--gfx1030", "gfx1030"),
            ("hip-amdgcn-amd-amdhsa--gfx908:xnack-", "gfx908:xnack-"),
            ("cuda-nvptx64-nvidia-cuda--sm_70", "sm_70"),  # Non-AMD format
        ]

        for target, expected in test_cases:
            arch = extract_architecture_from_target(target)
            assert arch == expected, f"Failed for target: {target}"

    def test_extract_architecture_empty_string(self):
        """Test handling empty target string."""
        assert extract_architecture_from_target("") is None

    def test_extract_architecture_none(self):
        """Test handling None input."""
        assert extract_architecture_from_target(None) is None

    def test_extract_architecture_no_double_dash(self):
        """Test target without double dash separator."""
        target = "some-random-string-without-proper-format"
        arch = extract_architecture_from_target(target)
        assert arch is None
