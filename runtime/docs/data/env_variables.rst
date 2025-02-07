.. meta::
    :description: ROCR-Runtime environment variables
    :keywords: AMD, ROCR, environment variables, environment

.. _rocr-env:
.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

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
