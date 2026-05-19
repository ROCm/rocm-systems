"""Unit tests for parallel kernel preparation utilities."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from rocm_kpack.compression import ZstdCompressor
from rocm_kpack.kpack import PackedKernelArchive
from rocm_kpack.parallel import KernelInput, get_worker_count, parallel_prepare_kernels


class TestGetWorkerCount:
    """Tests for worker count detection."""

    def test_auto_detect(self):
        """Test auto-detection returns at least 1."""
        count = get_worker_count(None)
        assert count >= 1

    def test_explicit_count(self):
        """Test explicit worker count is respected."""
        assert get_worker_count(4) == 4
        assert get_worker_count(1) == 1
        assert get_worker_count(16) == 16

    def test_minimum_one_worker(self):
        """Test minimum of 1 worker even with invalid input."""
        assert get_worker_count(0) == 1
        assert get_worker_count(-1) == 1


class TestParallelPrepareKernels:
    """Tests for parallel_prepare_kernels function."""

    def test_empty_kernel_list(self):
        """Test handling of empty kernel list."""
        archive = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
        )

        result = parallel_prepare_kernels(archive, [], executor=None)
        assert result == []

    def test_sequential_mode_with_executor_none(self):
        """Test sequential processing when executor is None."""
        archive = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
        )

        kernels = [
            KernelInput("bin/test1", "gfx1100", b"kernel_data_1", None),
            KernelInput("bin/test2", "gfx1100", b"kernel_data_2", None),
        ]
        result = parallel_prepare_kernels(archive, kernels, executor=None)

        assert len(result) == 2
        assert result[0].relative_path == "bin/test1"
        assert result[1].relative_path == "bin/test2"

    def test_parallel_mode_with_executor(self):
        """Test parallel processing with ThreadPoolExecutor."""
        archive = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100", "gfx1030"],
        )

        kernels = [
            KernelInput("bin/test1", "gfx1100", b"kernel_data_1", None),
            KernelInput("bin/test2", "gfx1030", b"kernel_data_2", None),
            KernelInput("bin/test3", "gfx1100", b"kernel_data_3", None),
            KernelInput("bin/test4", "gfx1030", b"kernel_data_4", None),
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            result = parallel_prepare_kernels(archive, kernels, executor=executor)

            assert len(result) == 4
            # Verify order is preserved
            assert result[0].relative_path == "bin/test1"
            assert result[0].gfx_arch == "gfx1100"
            assert result[1].relative_path == "bin/test2"
            assert result[1].gfx_arch == "gfx1030"
            assert result[2].relative_path == "bin/test3"
            assert result[2].gfx_arch == "gfx1100"
            assert result[3].relative_path == "bin/test4"
            assert result[3].gfx_arch == "gfx1030"

    def test_parallel_with_compression(self):
        """Test parallel preparation with actual compression."""
        archive = PackedKernelArchive(
            group_name="test",
            gfx_arch_family="gfx1100",
            gfx_arches=["gfx1100"],
            compressor=ZstdCompressor(compression_level=3),
        )

        # Create larger kernels to actually benefit from compression
        large_data = b"A" * 10000
        kernels = [
            KernelInput("bin/test1", "gfx1100", large_data, None),
            KernelInput("bin/test2", "gfx1100", large_data, None),
            KernelInput("bin/test3", "gfx1100", large_data, None),
            KernelInput("bin/test4", "gfx1100", large_data, None),
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            result = parallel_prepare_kernels(archive, kernels, executor=executor)

            assert len(result) == 4
            for prepared in result:
                assert prepared.original_size == len(large_data)
                # ZstdCompressionInput should have compressed data
                assert hasattr(prepared.compression_input, "compressed_frame")
