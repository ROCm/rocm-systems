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
        |
        | **NOTE:** In the 7.0 release the developer can use the HIP enumerator ``hipExtLimitScratchCurrent`` to programmatically change the default scratch memory allocation size. For more information, see `Global enums and defines <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/global_defines_enums_structs_files.html>`_.
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
      - ``1``
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

    * - | ``HSA_ENABLE_DXG_DETECTION``
        | Controls detection of the DXG driver (/dev/dxg) on WSL2.
      - ``1``
      - | 0: Disable DXG detection.
        | 1: Enable DXG detection, allowing ROCr to detect that it is running in WSL2.

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


Logging Environment Variables
-----------------------------

ROCR provides a comprehensive logging system for debugging and production monitoring.
Logging is disabled by default with zero performance overhead.

**CLR/HIP Integration:** For HIP applications, use ``AMD_LOG_LEVEL >= 6`` for unified CLR and ROCR logging.
When ``AMD_LOG_LEVEL >= 6``, the ``HSA_LOG_*`` variables below are ignored and CLR controls ROCR logging
via the ``hsa_amd_enable_logging()`` API. Both CLR and ROCR logs go to the same output.

For standalone HSA applications (not using HIP/CLR), use the ``HSA_LOG_*`` variables directly.

Set ``HSA_LOG_LEVEL=help`` to print usage information at runtime.

.. _rocr-log-env:
.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - Environment variable
      - Default value
      - Value

    * - | ``HSA_LOG_LEVEL``
        | Controls log verbosity. Set to ``help`` to print usage information.
      - ``0``
      - | 0: Disabled (default, no overhead).
        | 1: ERROR - Critical errors only.
        | 2: WARNING - Warnings and errors.
        | 3: INFO - General operational information.
        | 4: DEBUG - Detailed debug information.
        | 5: TRACE - Very detailed tracing.
        | 6: VERBOSE - Function entry/exit with arguments.
        | ``help``: Print usage information and exit.

    * - | ``HSA_LOG_MASK``
        | Hex bitmask to filter log categories. Combine values with OR.
      - ``0x7FFFFFFF``
      - | 0x1: INIT - Runtime init/shutdown.
        | 0x2: QUEUE - Queue create/destroy.
        | 0x4: MEM - Memory alloc/free.
        | 0x8: SIGNAL - Signal operations.
        | 0x10: IPC - IPC create/attach/detach.
        | 0x20: AGENT - Agent/topology discovery.
        | 0x40: AQL - AQL packet logging.
        | 0x80: SDMA - SDMA operations.
        | 0x100: COPY - Copy operations.
        | 0x200: BLIT - BlitKernel operations.
        | 0x400: SCRATCH - Scratch allocation.
        | 0x800: POOL - Memory/signal pools.
        | 0x2000: FAULT - VM faults.
        | 0x8000: EXCEPT - Exception handlers.
        | 0x10000: WAIT - Signal waits.

    * - | ``HSA_LOG_FILE``
        | Output file path. Process ID is automatically appended to the filename.
      - None (stderr)
      - | Path to log file. Example: ``/tmp/rocr`` creates ``/tmp/rocr.<PID>.log``.

    * - | ``HSA_LOG_SIZE``
        | Maximum log file size in megabytes before automatic truncation.
      - ``2048``
      - Integer value in MB.

    * - | ``HSA_LOG_ASYNC``
        | Enables asynchronous logging with a lock-free ring buffer and background flush thread.
        | Reduces logging overhead for high-frequency events.
      - ``0``
      - | 0: Synchronous logging (default).
        | 1: Asynchronous logging with crash-safe signal handler.

    * - | ``HSA_LOG_FORMAT``
        | Controls log output format.
      - None (text)
      - | ``json``: Structured JSON output for log aggregation tools.

**Logging Examples:**

- Unified CLR+ROCR logging (HIP apps): ``AMD_LOG_LEVEL=6 ./app``
- Error logging only: ``HSA_LOG_LEVEL=1 ./app``
- Debug memory issues: ``HSA_LOG_LEVEL=4 HSA_LOG_MASK=0x4 ./app``
- Debug IPC issues: ``HSA_LOG_LEVEL=4 HSA_LOG_MASK=0x10 ./app``
- Full trace to file: ``HSA_LOG_LEVEL=5 HSA_LOG_FILE=/tmp/rocr ./app``
- JSON output: ``HSA_LOG_LEVEL=3 HSA_LOG_FORMAT=json ./app``

**Production Health Warnings:**

At WARNING level (``HSA_LOG_LEVEL=2``), the logging system automatically reports:

- Memory pressure warnings when utilization exceeds 90% of peak.
- IPC leak detection when active attachments exceed 100.
- Queue saturation warnings when queue depth exceeds 75%.
