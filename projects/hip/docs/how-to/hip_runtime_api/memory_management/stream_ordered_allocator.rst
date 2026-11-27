.. meta::
  :description: Use the HIP stream-ordered memory allocator (SOMA) to allocate
                and free memory in stream order and to manage memory pools.
  :keywords: stream, memory allocation, SOMA, stream-ordered memory allocator

.. _stream_ordered_memory_allocator_how-to:

*******************************************************************************
Stream-ordered memory allocator
*******************************************************************************

The stream-ordered memory allocator (SOMA) is part of the HIP runtime API. SOMA provides an asynchronous memory allocation mechanism with stream-ordering semantics. You can use SOMA to allocate and free memory in stream order, which ensures that all asynchronous accesses occur between the stream executions of allocation and deallocation. Compliance with stream order prevents use-before-allocation or use-after-free errors, which avoids undefined behavior.

Advantages of SOMA:

- Efficient reuse: Enables efficient memory reuse across streams, which reduces unnecessary allocation overhead.
- Fine-grained control: Allows you to set attributes and control caching behavior for memory pools.
- Inter-process sharing: Enables secure sharing of allocations between processes.
- Optimizations: Allows the driver to optimize based on its awareness of SOMA and other stream management APIs.

Disadvantages of SOMA:

- Temporal constraints: Requires you to adhere strictly to stream order to avoid errors.
- Complexity: Involves memory management in stream order, which can be intricate.
- Learning curve: Requires additional effort to understand and use SOMA effectively.

Using SOMA
=====================================

Use ``hipMallocAsync()`` to allocate memory with stream-ordered semantics. Access
the memory only between the allocation and the deallocation in stream order.
Use ``hipFreeAsync()`` to return the memory to the pool with stream-ordered
semantics.

The following examples compare stream-ordered memory allocation with ordinary
allocation:

.. tab-set::
  .. tab-item:: Stream-ordered memory allocation

    .. literalinclude:: ../../../tools/example_codes/stream_ordered_memory_allocation.hip
        :start-after: // [sphinx-start]
        :end-before: // [sphinx-end]
        :language: cpp

  .. tab-item:: Ordinary allocation

    .. literalinclude:: ../../../tools/example_codes/ordinary_memory_allocation.hip
        :start-after: // [sphinx-start]
        :end-before: // [sphinx-end]
        :language: cpp

For more details, see :ref:`stream_ordered_memory_allocator_reference`.

Memory pools
============

Memory pools manage memory with stream-ordered behavior while ensuring proper
synchronization and avoiding memory access errors. Dividing a single memory
system into separate pools lets you query the access path properties of each
partition. A memory pool lives on a device and holds either device memory
(``hipMemAllocationTypePinned``) or managed memory
(``hipMemAllocationTypeManaged``).

Default pools
-------------

Each device has a default memory pool. ``hipMallocAsync()`` allocates from the
current memory pool of the stream's device, which is the default pool unless
you set another pool with ``hipDeviceSetMemPool()``. The stream's device is the
device the stream was created on, not the calling thread's current device. For
the null stream, it's the calling thread's current device.

Use :cpp:func:`hipMemGetDefaultMemPool` to get the default pool for a memory
location and allocation type. Set ``location.type`` to
``hipMemLocationTypeDevice`` and ``location.id`` to the device ID. The
allocation type selects the pool:

* ``hipMemAllocationTypePinned``: the default device memory pool, the same pool
  that ``hipDeviceGetDefaultMemPool()`` returns.
* ``hipMemAllocationTypeManaged``: the default managed memory pool.

Other location types and allocation types return ``hipErrorInvalidValue``.

Default pools
-------------

Each device has a default memory pool. ``hipMallocAsync()`` allocates from the
current memory pool of the stream's device, which is the default pool unless
you set another pool with ``hipDeviceSetMemPool()``.

Use :cpp:func:`hipMemGetDefaultMemPool` to get the default pool for a memory
location and allocation type. Set ``location.type`` to
``hipMemLocationTypeDevice`` and ``location.id`` to the device ID. The
allocation type selects the pool:

* ``hipMemAllocationTypePinned``: the default device memory pool, the same pool
  that ``hipDeviceGetDefaultMemPool()`` returns.
* ``hipMemAllocationTypeManaged``: the default managed memory pool.

Other location types and allocation types return ``hipErrorInvalidValue``.

Set pools
---------

``hipMallocAsync()`` allocates from the current memory pool. To use a different
pool, create it with ``hipMemPoolCreate()`` and allocate from it with
``hipMallocFromPoolAsync()``.

Unlike NVIDIA CUDA, where stream-ordered memory allocation can be implicit,
HIP on AMD requires explicit stream management for each allocation, ensuring
precise control over memory usage and synchronization.

.. literalinclude:: ../../../tools/example_codes/memory_pool.hip
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

Trim pools
----------

To control memory usage, set the ``hipMemPoolAttrReleaseThreshold`` attribute.
The threshold is the amount of reserved memory, in bytes, that the pool keeps.

.. literalinclude:: ../../../tools/example_codes/memory_pool_threshold.hip
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

When the amount of memory held in the memory pool exceeds the threshold, the allocator tries to release memory back to the operating system during the next call to stream, event, or context synchronization.

Use ``hipMemPoolTrimTo()`` to release unused memory from a pool back to the
system and reduce your application's memory footprint.

.. literalinclude:: ../../../tools/example_codes/memory_pool_trim.cpp
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

Resource usage statistics
-------------------------

To monitor memory usage, query the following pool attributes:

- ``hipMemPoolAttrReservedMemCurrent``: Returns the total physical GPU memory currently held in the pool.
- ``hipMemPoolAttrUsedMemCurrent``: Returns the total size of all the memory allocated from the pool.
- ``hipMemPoolAttrReservedMemHigh``: Returns the highest amount of physical GPU memory the pool has held since the last reset.
- ``hipMemPoolAttrUsedMemHigh``: Returns the highest amount of memory allocated from the pool since the last reset.

To reset the high watermarks, set ``hipMemPoolAttrReservedMemHigh`` or
``hipMemPoolAttrUsedMemHigh`` to zero with ``hipMemPoolSetAttribute()``.

.. literalinclude:: ../../../tools/example_codes/memory_pool_resource_usage_statistics.cpp
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

Memory reuse policies
---------------------

The allocator can reuse memory as long as compliant memory accesses don't overlap
in time. To tune memory reuse, enable or disable the following memory pool
attributes:

- ``hipMemPoolReuseFollowEventDependencies``: Checks event dependencies before allocating additional GPU memory.
- ``hipMemPoolReuseAllowOpportunistic``: Checks freed allocations to determine if the stream-order semantics indicated by the free operation has been met.
- ``hipMemPoolReuseAllowInternalDependencies``: Manages reuse based on internal dependencies in the runtime. If the driver fails to allocate and map additional physical memory, it searches for memory waiting for another stream's progress and reuses it.

Device accessibility for multi-GPU support
------------------------------------------

Allocations are initially accessible from the device where they reside. To make a
pool's allocations accessible from other devices, use ``hipMemPoolSetAccess()``.

Interprocess memory handling
=============================

.. note::

    Interprocess communication (IPC) API calls are only supported on systems with
    an active ``amdgpu-dkms`` driver. See the
    `AMDGPU documentation <https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/index.html>`__
    to install ``amdgpu-dkms``.

IPC-capable memory pools share GPU memory between processes.

To achieve interprocess memory sharing, you can use either a :ref:`device pointer <device-pointer>` or a :ref:`shareable handle <shareable-handle>`. Both provide allocator (export) and consumer (import) interfaces. Use a device pointer to share a specific allocation. Use a shareable handle when the importing process needs to allocate from the shared pool independently.

.. _device-pointer:

Device pointer
--------------

To export a memory pool allocation for access by another process, use
``hipMemPoolExportPointer()``. It produces a serializable ``hipMemPoolPtrExportData``
struct. Write it to any IPC channel, such as a named pipe (FIFO), and read it in
the importing process.

Here is the exporter:

.. literalinclude:: ../../../tools/example_codes/ipc_memory_pool_device_pointer.hip
    :start-after: // [sphinx-exporter-start]
    :end-before: // [sphinx-exporter-end]
    :language: cpp

To import a memory pool pointer from another process, use ``hipMemPoolImportPointer()``.

Here is the importer:

.. literalinclude:: ../../../tools/example_codes/ipc_memory_pool_device_pointer.hip
    :start-after: // [sphinx-importer-start]
    :end-before: // [sphinx-importer-end]
    :language: cpp

.. _shareable-handle:

Shareable handle
----------------

To export a memory pool as a shareable handle, use ``hipMemPoolExportToShareableHandle()``.
The handle is a POSIX file descriptor, which is process-local and cannot be transferred
by writing its integer value to a pipe. Instead, use ``SCM_RIGHTS`` over a Unix domain
socket to duplicate the file description into the receiving process.

Here is the exporter:

.. literalinclude:: ../../../tools/example_codes/ipc_memory_pool_shareable_handle.hip
    :start-after: // [sphinx-exporter-start]
    :end-before: // [sphinx-exporter-end]
    :language: cpp

To import a memory pool from a shareable handle received through ``SCM_RIGHTS``, use
``hipMemPoolImportFromShareableHandle()``. The importing process can then allocate
GPU memory from the imported pool independently.

Here is the importer:

.. literalinclude:: ../../../tools/example_codes/ipc_memory_pool_shareable_handle.hip
    :start-after: // [sphinx-importer-start]
    :end-before: // [sphinx-importer-end]
    :language: cpp
