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
