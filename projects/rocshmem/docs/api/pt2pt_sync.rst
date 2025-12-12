.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-pt2pt-sync:

-----------------------------------------
Point-to-point synchronization routines
-----------------------------------------

ROCSHMEM_WAIT_UNTIL
-------------------

.. cpp:function:: __device__ void rocshmem_TYPENAME_wait_until(TYPE *ivars, int cmp, TYPE val)

  :param ivars: Pointer to memory on the symmetric heap to wait for.
  :param cmp:   Operation for the comparison.
  :param val:   Value to compare the memory at ``ivars`` to.
  :returns:     None.

**Description:**
This routine blocks the caller until the condition ``(*ivars cmp val)`` is true.

Valid ``cmp`` values are listed in :ref:`CMP_VALUES`.

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`STANDARD_AMO_TYPES`.

ROCSHMEM_WAIT_UNTIL_ALL
-----------------------

.. cpp:function:: __device__ void rocshmem_TYPENAME_wait_until_all(TYPE *ivars, size_t nelems, const int* status, int cmp, TYPE val)

  :param ivars:  Pointer to memory on the symmetric heap to wait for.
  :param nelems: Number of elements in the ``ivars`` array.
  :param status: Array of length ``nelems`` to exclude elements from the wait.
  :param cmp:    Operation for the comparison.
  :param val:    Value to compare.
  :returns:      None.

**Description:**
This routine blocks the caller until the condition ``(ivars[i] cmp val)`` is true for all ``ivars``.

Valid ``cmp`` values are listed in :ref:`CMP_VALUES`.

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`STANDARD_AMO_TYPES`.

ROCSHMEM_WAIT_UNTIL_ANY
-----------------------
.. cpp:function:: __device__ size_t rocshmem_TYPENAME_wait_until_any(TYPE *ivars, size_t nelems, const int* status, int cmp, TYPE val)

  :param ivars:  Pointer to memory on the symmetric heap to wait for.
  :param nelems: Number of elements in the ``ivars`` array.
  :param status: Array of length ``nelems`` to exclude elements from the wait.
  :param cmp:    Operation for the comparison.
  :param val:    Value to compare.
  :returns:      The index of an element in the ``ivars`` array that satisfies the wait condition. If the wait set is empty, this routine returns ``SIZE_MAX``.

**Description:**
This routine blocks the caller until any of the condition ``(ivars[i] cmp val)`` is true.

Valid ``cmp`` values are listed in :ref:`CMP_VALUES`.

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`STANDARD_AMO_TYPES`.

ROCSHMEM_WAIT_UNTIL_SOME
------------------------

.. cpp:function:: __device__ size_t rocshmem_TYPENAME_wait_until_some(TYPE *ivars, size_t nelems, size_t* indices, const int* status, int cmp, TYPE val)

  :param ivars:    Pointer to memory on the symmetric heap to wait for.
  :param nelems:   Number of elements in the ``ivars`` array.
  :param indices:  List of indices with a length of at least ``nelems``.
  :param status:   Array of length ``nelems``  to exclude elements from the wait.
  :param cmp:      Operation for the comparison.
  :param val:      Value to compare.
  :returns:        The number of indices returned in the indices array. If the wait set is empty, this routine returns ``0``.

**Description:**
This routine blocks the caller until any of the conditions ``(ivars[i] cmp val)`` is true.

Valid ``cmp`` values are listed in :ref:`CMP_VALUES`.

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`STANDARD_AMO_TYPES`.

ROCSHMEM_TEST
-------------

.. cpp:function:: __device__ int rocshmem_TYPENAME_test(TYPE *ivars, int cmp, TYPE val)
 
  :param ivars: Pointer to memory on the symmetric heap to wait for.
  :param cmp:   Operation for the comparison.
  :param val:   Value to compare the memory at ``ivars`` to.
 
  :returns:     ``1`` if the evaluation is true. ``0`` otherwise.

**Description:**
This routine tests if the condition ``(*ivars cmp val)`` is true.

ROCSHMEM_SIGNAL_WAIT_UNTIL_ON_STREAM
-------------------------------------

.. cpp:function:: __host__ void rocshmem_signal_wait_until_on_stream(uint64_t *sig_addr, int cmp, uint64_t cmp_value, hipStream_t stream)

  :param sig_addr:   Address of the signal variable on the symmetric heap.
  :param cmp:        Comparison operator (e.g., ROCSHMEM_CMP_EQ, ROCSHMEM_CMP_GE, etc.).
  :param cmp_value:  Value to compare against.
  :param stream:     HIP stream on which to enqueue the operation.
  :returns:          None.

**Description:**
This routine enqueues a wait operation on a HIP stream. The function blocks the calling thread
until the signal variable at ``sig_addr`` satisfies the comparison condition ``(*sig_addr cmp cmp_value)``.
The wait operation is executed asynchronously on the specified stream. The caller must synchronize
the stream (e.g., using ``hipStreamSynchronize``) to ensure the wait condition has been satisfied.

Valid ``cmp`` values are listed in :ref:`CMP_VALUES`.

.. _CMP_VALUES:

Supported comparisons
---------------------

The following table lists the point-to-point comparison constants:

.. list-table:: Point-to-Point Comparison Constants
    :widths: 20 20
    :header-rows: 1

    * - Constant
      - Description
    * - ROCSHMEM_CMP_EQ
      - Equal
    * - ROCSHMEM_CMP_NE
      - Not equal
    * - ROCSHMEM_CMP_GT
      - Greater than
    * - ROCSHMEM_CMP_GE
      - Greater than or equal to
    * - ROCSHMEM_CMP_LT
      - Less than
    * - ROCSHMEM_CMP_LE
      - Less than or equal to
