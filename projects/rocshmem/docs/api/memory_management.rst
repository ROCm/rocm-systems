.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-memory-management:


---------------------------
Memory management routines
---------------------------

ROCSHMEM_MALLOC
---------------

.. cpp:function:: __host__ void *rocshmem_malloc(size_t size)

  :param size: Memory allocation size in bytes.
  :returns: A pointer to the allocated memory on the symmetric heap.
            If a valid allocation cannot be made, it returns ``NULL``.

**Description:**
This routine allocates memory of ``size`` bytes from the symmetric heap.
This is a collective operation and must be called by all PEs.

ROCSHMEM_ALIGN
--------------

.. cpp:function:: __host__ void *rocshmem_align(size_t alignment, size_t size)

  :param alignment: Required pointer alignment in bytes. Must be a power of
                    two **and** a multiple of ``sizeof(void *)``.
  :param size:      Memory allocation size in bytes.
  :returns: A pointer to the allocated memory on the symmetric heap, aligned
            to ``alignment`` bytes. Returns ``NULL`` if ``alignment`` is
            invalid (not a power of two, or not a multiple of
            ``sizeof(void *)``) or if the allocation cannot be satisfied.

**Description:**
This routine allocates ``size`` bytes from the symmetric heap and returns a
pointer that is aligned to ``alignment`` bytes. It is a collective operation
and must be called by all PEs with identical ``alignment`` and ``size``
arguments.

Constraints on ``alignment`` (mirroring OpenSHMEM ``shmem_align`` and POSIX
``posix_memalign`` semantics):

* ``alignment`` must be a power of two.
* ``alignment`` must be a multiple of ``sizeof(void *)``.

  Any power of two that is at least ``sizeof(void *)`` (8 bytes on a 64-bit
  build) automatically satisfies both rules. The smallest accepted value is
  therefore ``sizeof(void *)``; accepted values are
  ``sizeof(void *), 2*sizeof(void *), 4*sizeof(void *), ...``.

  Invalid values (``0``, ``1``, ``2``, ``4`` on 64-bit, or any non
  power-of-two) cause the routine to return ``NULL`` and emit a warning. The
  collective barrier is still entered so other PEs do not deadlock.

Notes:

* ``alignment`` values less than or equal to the default symmetric-heap
  alignment (128 bytes) yield a 128-byte aligned pointer, identical to
  :ref:`rocshmem_malloc`.

ROCSHMEM_FREE
-------------

.. cpp:function:: __host__ void rocshmem_free(void *ptr)

  :param ptr: A pointer to previously allocated memory on the symmetric heap.
  :returns: None.

**Description:**
This routine frees a memory allocation from the symmetric heap.
It is a collective operation and must be called by all PEs.

ROCSHMEM_BUFFER_REGISTER
------------------------

.. cpp:function:: __host__ int rocshmem_buffer_register(void *addr, size_t length);

  :param addr:   Pointer to previously allocated memory
  :param length: Length of addr
  :returns: ROCSHMEM_SUCCESS or an error.

**Description:**
Registers a user-allocated buffer. This buffer can be used as a local
buffer to most rocSHMEM communication calls. It is erroneous to use it for a
remote buffer.
Currently, this call is only implemented for IPC and RO, GDA is
not supported.

ROCSHMEM_BUFFER_UNREGISTER
--------------------------

.. cpp:function:: __host__ int rocshmem_buffer_unregister(void *addr);

  :param addr:   Pointer to previously registered memory
  :returns: ROCSHMEM_SUCCESS or an error.

**Description:**
Deregisters a previously registered buffer that was registered using
`rocshmem_buffer_register`.
Currently, this call is only implemented for IPC and RO, GDA is
not supported.
