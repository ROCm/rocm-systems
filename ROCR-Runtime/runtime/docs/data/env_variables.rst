.. meta::
    :description: ROCR-Runtime environment variables
    :keywords: AMD, ROCR, environment variables, environment

.. _rocr-env:
.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - Environment variable
      - Default value
      - Value

    * - | ``ROCR_VISIBLE_DEVICES``
        | Specifies a list of device indices or UUIDs to be exposed to the applications.
      - None
      - ``0,GPU-DEADBEEFDEADBEEF``

    * - | ``HSA_NO_SCRATCH_RECLAIM``
        | Controls whether scratch memory allocations are permanently assigned to queues or can be reclaimed based on usage thresholds.
      - ``0``
      - | 0: Disable.
        | When dispatches need scratch memory that are lower than the threshold, the memory will be permanently assigned to the queue. For dispatches that exceed the threshold, a scratch-use-once mechanism will be used, resulting in the memory to be unassigned after the dispatch.
        | 1: Enable.
        | If a kernel dispatch needs scratch memory, runtime will allocate and permanently assign device memory to the queue handling the dispatch, even if the amount of scratch memory exceeds the default threshold. This memory will not be available to other queues or processes until this process exits.

    * - | ``HSA_SCRATCH_SINGLE_LIMIT``
        | Specifies the threshold for the amount of scratch memory allocated and reclaimed in kernel dispatches.
        | Enabling ``HSA_NO_SCRATCH_RECLAIM`` circumvents ``HSA_SCRATCH_SINGLE_LIMIT``, and treats ``HSA_SCRATCH_SINGLE_LIMIT`` as the maximum value.
      - ``146800640``
      - 0 to 4GB per XCC

    * - | ``HSA_SCRATCH_SINGLE_LIMIT_ASYNC``
        | On GPUs that support asynchronous scratch reclaim, this variable is used instead of ``HSA_SCRATCH_SINGLE_LIMIT`` to specify the threshold for scratch memory allocation.
      - ``3221225472`` (3GB)
      - 0 to 4GB per XCC

    * - | ``HSA_ENABLE_SCRATCH_ASYNC_RECLAIM``
        | Controls asynchronous scratch memory reclamation on supported GPUs.
        | When enabled, if a device memory allocation fails, ROCr will attempt to reclaim scratch memory assigned to all queues and retry the allocation.
      - ``1``
      - | 0: Disable asynchronous scratch reclaim.
        | 1: Enable asynchronous scratch reclaim on supported GPUs.

    * - | ``HSA_XNACK``
        | Enables XNACK.
      - None
      - 1: Enable

    * - | ``HSA_CU_MASK``
        | Sets the mask on a lower level of queue creation in the driver.
        | This mask is also applied to the queues being profiled.
      - None
      - ``1:0-8``

    * - | ``HSA_ENABLE_SDMA``
        | Enables the use of direct memory access (DMA) engines in all copy directions (Host-to-Device, Device-to-Host, Device-to-Device), when using any of the following APIs:
        | ``hsa_memory_copy``,
        | ``hsa_amd_memory_fill``,
        | ``hsa_amd_memory_async_copy``,
        | ``hsa_amd_memory_async_copy_on_engine``.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``HSA_ENABLE_PEER_SDMA``
        | **Note**: This environment variable is ignored if ``HSA_ENABLE_SDMA`` is set to 0.
        | Enables the use of DMA engines for Device-to-Device copies, when using any of the following APIs:
        | ``hsa_memory_copy``,
        | ``hsa_amd_memory_async_copy``,
        | ``hsa_amd_memory_async_copy_on_engine``.
      - ``1``
      - | 0: Disable
        | 1: Enable

    * - | ``HSA_ENABLE_MWAITX``
        | When mwaitx is enabled, on AMD CPUs, runtime will hint to the CPU to go into lower power-states when doing busy loops by using the mwaitx instruction.
      - ``0``
      - | 0: Disable
        | 1: Enable

    * - | ``HSA_OVERRIDE_CPU_AFFINITY_DEBUG``
        | Controls whether ROCm helper threads inherit the parent process's CPU affinity mask.
      - ``1``
      - | 0: Enable inheritance. Helper threads use the parent process's core affinity mask, which should be set with enough cores for all threads.
        | 1: Disable inheritance. Helper threads spawn on all available cores, ignoring the parent's affinity settings, which may affect performance in certain environments.

    * - | ``HSA_ENABLE_DEBUG``
        | Enables additional debug information and validation in the runtime.
      - ``0``
      - | 0: Disable debug mode.
        | 1: Enable debug mode with additional validation and logging.


Hardware Debugging Environment Variables
----------------------------------------

The following environment variables are intended for experienced users who are debugging hardware-specific issues.
These settings may impact performance and stability and should only be used when troubleshooting specific hardware problems.

.. _rocr-debug-env:
.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - Environment variable
      - Default value
      - Value

    * - | ``HSA_DISABLE_FRAGMENT_ALLOCATOR``
        | Disables internal memory fragment caching to help debug memory faults.
      - ``0``
      - | 0: Fragment allocator enabled (normal operation).
        | 1: Fragment allocator disabled. Helps debug tools identify memory faults at their origin by preventing cached memory blocks from masking out-of-bounds writes.

    * - | ``HSAKMT_DEBUG_LEVEL``
        | Controls the verbosity level of debug messages from the ``libhsakmt.so`` driver layer.
      - ``3``
      - | 3: Only error messages (``pr_err``) are printed.
        | 4: Error and warning messages (``pr_err``, ``pr_warn``) are printed.
        | 5: Same as level 4 (notice level not implemented).
        | 6: Error, warning, and info messages (``pr_err``, ``pr_warn``, ``pr_info``) are printed.
        | 7: All debug messages including ``pr_debug`` are printed.

    * - | ``HSA_ENABLE_INTERRUPT``
        | Controls how completion signals are detected, useful for diagnosing interrupt storm issues.
      - ``1``
      - | 0: Disable hardware interrupts. Uses memory-based polling for completion signals instead of interrupts.
        | 1: Enable hardware interrupts (normal operation).

    * - | ``HSA_SVM_GUARD_PAGES``
        | Controls the use of guard pages in Shared Virtual Memory (SVM) allocations.
      - ``1``
      - | 0: Disable SVM guard pages (for debugging memory access patterns).
        | 1: Enable SVM guard pages (normal operation).

    * - | ``HSA_DISABLE_CACHE``
        | Controls GPU L2 cache utilization for all memory regions.
      - ``0``
      - | 0: Normal caching behavior (L2 cache enabled).
        | 1: Disables L2 cache entirely. Sets all memory regions as uncacheable (MTYPE=UC) in the GPU, bypassing the L2 cache. Useful for diagnosing cache-related performance or correctness issues.
