"""Parallel processing utilities for kernel preparation.

This module provides utilities for parallelizing CPU-intensive kernel preparation
(compression, preprocessing) while keeping metadata manipulation (TOC updates) sequential.
"""

import os
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from typing import NamedTuple

from rocm_kpack.kpack import PackedKernelArchive, PreparedKernel


class KernelInput(NamedTuple):
    """Input data for preparing a kernel for packing.

    Attributes:
        relative_path: Path relative to archive root (e.g., "kernels/my_kernel")
        gfx_arch: GPU architecture (e.g., "gfx1100")
        hsaco_data: Raw HSACO binary data
        metadata: Optional metadata dict to store in TOC
    """

    relative_path: str
    gfx_arch: str
    hsaco_data: bytes
    metadata: dict[str, object] | None


def get_worker_count(max_workers: int | None = None) -> int:
    """Determine the number of worker threads to use.

    Args:
        max_workers: Explicit worker count, or None for auto-detection

    Returns:
        Number of worker threads (minimum 1)
    """
    if max_workers is not None:
        return max(1, max_workers)

    # Auto-detect: use all available cores
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, cpu_count)


def parallel_prepare_kernels(
    archive: PackedKernelArchive,
    kernels: list[KernelInput],
    executor: Executor | None = None,
) -> list[PreparedKernel]:
    """Prepare multiple kernels in parallel using provided executor.

    This is the map phase of map/reduce compression. Each kernel is prepared
    (compressed/preprocessed) independently in parallel, then the results can
    be added to the archive sequentially.

    Args:
        archive: PackedKernelArchive instance
        kernels: List of KernelInput objects containing kernel data and metadata
        executor: Executor for parallel execution. If None, runs sequentially.

    Returns:
        List of PreparedKernel objects in the same order as input
    """
    if not kernels:
        return []

    # Sequential path when no executor provided
    if executor is None:
        return [
            archive.prepare_kernel(
                k.relative_path, k.gfx_arch, k.hsaco_data, k.metadata
            )
            for k in kernels
        ]

    # Parallel preparation using provided executor
    # Submit all tasks
    future_to_index = {}
    for i, k in enumerate(kernels):
        future = executor.submit(
            archive.prepare_kernel,
            k.relative_path,
            k.gfx_arch,
            k.hsaco_data,
            k.metadata,
        )
        future_to_index[future] = i

    # Collect results in original order
    results = [None] * len(kernels)
    for future in as_completed(future_to_index):
        index = future_to_index[future]
        results[index] = future.result()

    return results
