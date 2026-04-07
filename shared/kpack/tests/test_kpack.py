"""Tests for packed kernel archive format and compression."""

import tempfile
from pathlib import Path

import pytest

from rocm_kpack.compression import (
    Compressor,
    CompressionInput,
    NoOpCompressor,
    ZstdCompressor,
)
from rocm_kpack.kpack import PackedKernelArchive


# Fixtures for parameterized tests
@pytest.fixture(
    params=[
        pytest.param(NoOpCompressor(), id="noop"),
        pytest.param(ZstdCompressor(compression_level=3), id="zstd"),
    ]
)
def compressor(request):
    """Parameterized fixture providing different compressor implementations."""
    return request.param


# ============================================================================
# Compressor Unit Tests
# ============================================================================


class TestNoOpCompressor:
    """Test NoOpCompressor-specific behavior."""

    def test_prepare_kernel(self):
        """Test map phase returns uncompressed data."""
        compressor = NoOpCompressor()
        data = b"hello world"
        result = compressor.prepare_kernel(data, "test_kernel")

        assert isinstance(result, CompressionInput)
        assert hasattr(result, "data")
        assert result.data == data

    def test_finalize_returns_tuple(self):
        """Test finalize returns (blob, toc_metadata) tuple."""
        compressor = NoOpCompressor()

        # Prepare multiple kernels
        kernels = {
            "kernel1": b"first kernel data",
            "kernel2": b"second kernel data with more bytes",
            "kernel3": b"third",
        }

        inputs = []
        for kernel_id, data in kernels.items():
            comp_input = compressor.prepare_kernel(data, kernel_id)
            inputs.append((kernel_id, comp_input))

        # Finalize (reduce phase)
        blob_data, toc_metadata = compressor.finalize(inputs)

        assert isinstance(blob_data, bytes)
        assert isinstance(toc_metadata, dict)
        assert "blobs" in toc_metadata
        assert len(toc_metadata["blobs"]) == 3

    def test_scheme_name(self):
        """Test scheme name is correct."""
        assert NoOpCompressor.SCHEME_NAME == "none"


class TestZstdCompressor:
    """Test ZstdCompressor-specific behavior."""

    def test_prepare_kernel_compresses(self):
        """Test map phase compresses data."""
        compressor = ZstdCompressor(compression_level=3)

        # Create compressible data (lots of repetition)
        data = b"A" * 10000
        result = compressor.prepare_kernel(data, "test_kernel")

        assert isinstance(result, CompressionInput)
        assert hasattr(result, "compressed_frame")
        assert hasattr(result, "original_size")
        assert result.original_size == len(data)
        # Compressed should be much smaller for highly repetitive data
        assert len(result.compressed_frame) < len(data) / 10

    def test_finalize_returns_tuple(self):
        """Test finalize returns (blob, toc_metadata) tuple."""
        compressor = ZstdCompressor()
        kernel_id = "my_kernel"
        original_data = b"x" * 5000

        # Prepare
        comp_input = compressor.prepare_kernel(original_data, kernel_id)

        # Finalize
        blob_data, toc_metadata = compressor.finalize([(kernel_id, comp_input)])

        assert isinstance(blob_data, bytes)
        assert isinstance(toc_metadata, dict)
        assert "zstd_offset" in toc_metadata
        assert "zstd_size" in toc_metadata

    def test_compression_ratio(self):
        """Test that compression actually reduces size for repetitive data."""
        compressor = ZstdCompressor(compression_level=3)

        # Highly compressible data
        data = b"ROCM_KERNEL_" * 1000
        comp_input = compressor.prepare_kernel(data, "test")
        blob_data, toc_metadata = compressor.finalize([("test", comp_input)])

        # Compressed blob should be much smaller
        # Expect at least 10x compression for this repetitive data
        assert len(blob_data) < len(data) / 5

    def test_scheme_name(self):
        """Test scheme name is correct."""
        assert ZstdCompressor.SCHEME_NAME == "zstd-per-kernel"

    def test_different_compression_levels(self, tmp_path):
        """Test that different compression levels work correctly."""
        data = b"compress me! " * 1000

        # Level 1 (fast)
        archive1 = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=ZstdCompressor(compression_level=1),
        )
        archive1.add_kernel(archive1.prepare_kernel("bin/test", "gfx1100", data))
        archive1.finalize_archive()
        path1 = tmp_path / "level1.kpack"
        archive1.write(path1)

        # Level 19 (max compression)
        archive19 = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=ZstdCompressor(compression_level=19),
        )
        archive19.add_kernel(archive19.prepare_kernel("bin/test", "gfx1100", data))
        archive19.finalize_archive()
        path19 = tmp_path / "level19.kpack"
        archive19.write(path19)

        # Higher level should produce smaller (or equal) output
        assert path19.stat().st_size <= path1.stat().st_size

        # Both should decompress to same data
        read1 = PackedKernelArchive.read(path1)
        read19 = PackedKernelArchive.read(path19)
        assert read1.get_kernel("bin/test", "gfx1100") == data
        assert read19.get_kernel("bin/test", "gfx1100") == data

    def test_compressed_smaller_than_uncompressed(self, tmp_path):
        """Verify compressed archives are actually smaller."""
        # Create highly compressible data
        kernel_data = b"REPETITIVE_DATA_" * 5000

        # Uncompressed archive
        archive_plain = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=NoOpCompressor(),
        )
        archive_plain.add_kernel(
            archive_plain.prepare_kernel("bin/test", "gfx1100", kernel_data)
        )
        archive_plain.finalize_archive()
        path_plain = tmp_path / "plain.kpack"
        archive_plain.write(path_plain)

        # Compressed archive
        archive_compressed = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=ZstdCompressor(compression_level=3),
        )
        archive_compressed.add_kernel(
            archive_compressed.prepare_kernel("bin/test", "gfx1100", kernel_data)
        )
        archive_compressed.finalize_archive()
        path_compressed = tmp_path / "compressed.kpack"
        archive_compressed.write(path_compressed)

        # Compressed should be much smaller
        size_plain = path_plain.stat().st_size
        size_compressed = path_compressed.stat().st_size

        assert size_compressed < size_plain / 5  # Expect >5x compression


# ============================================================================
# PackedKernelArchive Tests (Parameterized across compressors)
# ============================================================================


class TestPackedKernelArchive:
    """Test PackedKernelArchive with any compressor."""

    def test_roundtrip(self, compressor, tmp_path):
        """Test creating, writing, and reading back archive."""
        # Create archive
        archive = PackedKernelArchive(
            group_name="blas",
            gfx_arch_family="gfx100X",
            gfx_arches=["gfx1030", "gfx1001"],
            compressor=compressor,
        )

        # Add some kernels
        archive.add_kernel(
            archive.prepare_kernel("bin/hipcc", "gfx1030", b"kernel_data_gfx1030")
        )
        archive.add_kernel(
            archive.prepare_kernel("bin/hipcc", "gfx1001", b"kernel_data_gfx1001")
        )
        archive.add_kernel(
            archive.prepare_kernel(
                "lib/libhipblas.so",
                "gfx1030",
                b"hipblas_kernel_gfx1030",
                metadata={"version": "1.0"},
            )
        )

        # Finalize and write to file
        archive.finalize_archive()
        pack_file = tmp_path / "blas-gfx100X.kpack"
        archive.write(pack_file)

        # Verify file exists
        assert pack_file.exists()
        assert pack_file.stat().st_size > 0

        # Read back
        loaded = PackedKernelArchive.read(pack_file)

        # Verify metadata
        assert loaded.group_name == "blas"
        assert loaded.gfx_arch_family == "gfx100X"
        assert loaded.gfx_arches == ["gfx1030", "gfx1001"]

        # Verify kernels
        assert loaded.get_kernel("bin/hipcc", "gfx1030") == b"kernel_data_gfx1030"
        assert loaded.get_kernel("bin/hipcc", "gfx1001") == b"kernel_data_gfx1001"
        assert (
            loaded.get_kernel("lib/libhipblas.so", "gfx1030")
            == b"hipblas_kernel_gfx1030"
        )

        # Verify metadata preserved
        assert loaded.toc["lib/libhipblas.so"]["gfx1030"]["metadata"] == {
            "version": "1.0"
        }

        # Verify non-existent kernels
        assert loaded.get_kernel("nonexistent/binary", "gfx1030") is None
        assert loaded.get_kernel("bin/hipcc", "gfx1100") is None

    def test_duplicate_kernel_error(self, compressor):
        """Test that adding duplicate kernel raises error."""
        archive = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=compressor,
        )

        archive.add_kernel(archive.prepare_kernel("bin/test", "gfx1100", b"data"))

        with pytest.raises(ValueError, match="Kernel already exists"):
            archive.add_kernel(
                archive.prepare_kernel("bin/test", "gfx1100", b"different_data")
            )

    def test_path_normalization(self, compressor, tmp_path):
        """Test that paths are normalized consistently."""
        archive = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=compressor,
        )

        # Add with backslashes (Windows-style)
        archive.add_kernel(archive.prepare_kernel("bin\\hipcc", "gfx1100", b"data"))

        archive.finalize_archive()
        pack_file = tmp_path / "test.kpack"
        archive.write(pack_file)
        loaded = PackedKernelArchive.read(pack_file)

        # Retrieve with forward slashes (should work)
        assert loaded.get_kernel("bin/hipcc", "gfx1100") == b"data"

    def test_ordinals(self, compressor, tmp_path):
        """Test that ordinals are assigned sequentially."""
        archive = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=compressor,
        )

        archive.add_kernel(archive.prepare_kernel("bin/a", "gfx1100", b"data_0"))
        archive.add_kernel(archive.prepare_kernel("bin/b", "gfx1100", b"data_1"))
        archive.add_kernel(archive.prepare_kernel("bin/c", "gfx1100", b"data_2"))

        # Check ordinals in TOC
        assert archive.toc["bin/a"]["gfx1100"]["ordinal"] == 0
        assert archive.toc["bin/b"]["gfx1100"]["ordinal"] == 1
        assert archive.toc["bin/c"]["gfx1100"]["ordinal"] == 2

        # Verify data by writing and reading back
        archive.finalize_archive()
        pack_file = tmp_path / "test.kpack"
        archive.write(pack_file)
        loaded = PackedKernelArchive.read(pack_file)

        assert loaded.get_kernel("bin/a", "gfx1100") == b"data_0"
        assert loaded.get_kernel("bin/b", "gfx1100") == b"data_1"
        assert loaded.get_kernel("bin/c", "gfx1100") == b"data_2"

    def test_write_without_finalize_raises(self, compressor):
        """Test that write() requires finalize_archive() to be called first."""
        archive = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=compressor,
        )
        archive.add_kernel(archive.prepare_kernel("bin/test", "gfx1100", b"data"))

        with tempfile.NamedTemporaryFile(suffix=".kpack", delete=False) as f:
            output_path = Path(f.name)

        try:
            with pytest.raises(RuntimeError, match="not finalized"):
                archive.write(output_path)
        finally:
            output_path.unlink(missing_ok=True)

    def test_mixed_architectures(self, compressor, tmp_path):
        """Test archive with multiple architectures."""
        archive = PackedKernelArchive(
            group_name="mixed",
            gfx_arch_family="gfx100X",
            gfx_arches=["gfx1030", "gfx1031", "gfx1032"],
            compressor=compressor,
        )

        # Add kernels for different architectures
        for arch in ["gfx1030", "gfx1031", "gfx1032"]:
            archive.add_kernel(
                archive.prepare_kernel(
                    "bin/app", arch, f"kernel_for_{arch}_".encode() * 100
                )
            )

        archive.finalize_archive()

        output_path = tmp_path / "mixed.kpack"
        archive.write(output_path)

        # Read and verify
        archive_read = PackedKernelArchive.read(output_path)

        for arch in ["gfx1030", "gfx1031", "gfx1032"]:
            expected = f"kernel_for_{arch}_".encode() * 100
            actual = archive_read.get_kernel("bin/app", arch)
            assert actual == expected

    def test_toc_contains_compression_scheme(self, compressor, tmp_path):
        """Test that TOC contains compression_scheme field."""
        # Write archive
        archive = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=compressor,
        )
        archive.add_kernel(archive.prepare_kernel("bin/test", "gfx1100", b"data"))
        archive.finalize_archive()
        output_path = tmp_path / "test.kpack"
        archive.write(output_path)

        # Read and check TOC structure
        import msgpack
        import struct

        with output_path.open("rb") as f:
            # Skip header, seek to TOC
            header = f.read(16)
            toc_offset = struct.unpack("<Q", header[8:16])[0]
            f.seek(toc_offset)
            toc_data = msgpack.unpack(f, raw=False)

        assert "compression_scheme" in toc_data
        assert toc_data["compression_scheme"] == compressor.SCHEME_NAME

    def test_ordinals_in_toc(self, compressor, tmp_path):
        """Test that per-kernel TOC entries use ordinals."""
        archive = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=compressor,
        )
        archive.add_kernel(archive.prepare_kernel("bin/app1", "gfx1100", b"data1"))
        archive.add_kernel(archive.prepare_kernel("bin/app2", "gfx1100", b"data2"))
        archive.finalize_archive()
        output_path = tmp_path / "test.kpack"
        archive.write(output_path)

        # Read and check TOC
        import msgpack
        import struct

        with output_path.open("rb") as f:
            header = f.read(16)
            toc_offset = struct.unpack("<Q", header[8:16])[0]
            f.seek(toc_offset)
            toc_data = msgpack.unpack(f, raw=False)

        # Check that entries have ordinals
        assert "bin/app1" in toc_data["toc"]
        assert "gfx1100" in toc_data["toc"]["bin/app1"]
        entry1 = toc_data["toc"]["bin/app1"]["gfx1100"]
        assert "ordinal" in entry1
        assert entry1["ordinal"] == 0

        assert "bin/app2" in toc_data["toc"]
        entry2 = toc_data["toc"]["bin/app2"]["gfx1100"]
        assert "ordinal" in entry2
        assert entry2["ordinal"] == 1


# ============================================================================
# Non-parameterized Tests (Compressor-independent functionality)
# ============================================================================


def test_compute_pack_filename():
    """Test pack filename computation."""
    assert (
        PackedKernelArchive.compute_pack_filename("blas", "gfx1100")
        == "blas-gfx1100.kpack"
    )
    assert (
        PackedKernelArchive.compute_pack_filename("torch", "gfx100X")
        == "torch-gfx100X.kpack"
    )


def test_pack_archive_repr():
    """Test string representation."""
    archive = PackedKernelArchive(
        group_name="blas",
        gfx_arch_family="gfx100X",
        gfx_arches=["gfx1030", "gfx1001"],
    )
    archive.add_kernel(archive.prepare_kernel("bin/a", "gfx1030", b"data"))
    archive.add_kernel(archive.prepare_kernel("bin/a", "gfx1001", b"data"))
    archive.add_kernel(archive.prepare_kernel("bin/b", "gfx1030", b"data"))

    repr_str = repr(archive)
    assert "blas" in repr_str
    assert "gfx100X" in repr_str
    assert "binaries=2" in repr_str
    assert "kernels=3" in repr_str


def test_streaming_mode_raises(tmp_path):
    """Test that streaming write mode is not yet supported."""
    pack_file = tmp_path / "streaming-test.kpack"

    # Attempting to create archive in streaming mode should raise error
    with pytest.raises(ValueError, match="Streaming write mode not yet supported"):
        PackedKernelArchive(
            group_name="blas",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            output_path=pack_file,
        )
