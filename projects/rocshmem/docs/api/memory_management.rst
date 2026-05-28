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
