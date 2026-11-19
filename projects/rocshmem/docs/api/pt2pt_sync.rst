.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-pt2pt-sync:

-----------------------------------------
Point-to-point synchronization routines
-----------------------------------------

.. note::

   Every ``rocshmem_TYPENAME_wait_until*`` and ``rocshmem_TYPENAME_test``
   routine on this page is declared with both a ``__device__`` and a
   ``__host__`` overload with identical signatures. Only the ``__device__``
   form is shown below.

   The ``wait_until`` routines block: the host form blocks the calling host
   thread, and the device form blocks the calling device thread — these are
   per-thread operations, not wave collectives. The ``test`` routines do not
   block; both forms return immediately with the result of the comparison.

.. WARNING::

   The GDA host overloads require MPI-backed initialization. Under MPI-free
   initialization the host interface has no MPI window and aborts the process,
   so a host wait terminates the job rather than waiting.

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

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`WAIT_TEST_TYPES`.

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

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`WAIT_TEST_TYPES`.

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

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`WAIT_TEST_TYPES`.

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

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`WAIT_TEST_TYPES`.

ROCSHMEM_WAIT_UNTIL_ALL_VECTOR
------------------------------

.. cpp:function:: __device__ void rocshmem_TYPENAME_wait_until_all_vector(TYPE *ivars, size_t nelems, const int* status, int cmp, TYPE* vals)

  :param ivars:  Pointer to memory on the symmetric heap to wait for.
  :param nelems: Number of elements in the ``ivars`` array.
  :param status: Pass ``nullptr``. Non-null status arrays are not handled
                 correctly by the device vector waits: an all-zero array (every
                 element included) returns without waiting, and an all-excluded
                 array makes the ``any`` and ``some`` variants spin forever.
                 The restriction applies to all three ``*_vector`` variants.
  :param cmp:    Operation for the comparison.
  :param vals:   Array of length ``nelems``. Element ``i`` of ``ivars`` is
                 compared against ``vals[i]``.
  :returns:      None.

**Description:**
This routine blocks the caller until the condition ``(ivars[i] cmp vals[i])``
is true for all elements in the wait set. It differs from
``rocshmem_TYPENAME_wait_until_all`` in that each element is compared against
its own value rather than against a single scalar.

Valid ``cmp`` values are listed in :ref:`CMP_VALUES`.

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`WAIT_TEST_TYPES`.

ROCSHMEM_WAIT_UNTIL_ANY_VECTOR
------------------------------

.. cpp:function:: __device__ size_t rocshmem_TYPENAME_wait_until_any_vector(TYPE *ivars, size_t nelems, const int* status, int cmp, TYPE* vals)

  :param ivars:  Pointer to memory on the symmetric heap to wait for.
  :param nelems: Number of elements in the ``ivars`` array.
  :param status: Pass ``nullptr``. Non-null status arrays are not handled
                 correctly by the device vector waits: an all-zero array (every
                 element included) returns without waiting, and an all-excluded
                 array makes the ``any`` and ``some`` variants spin forever.
                 The restriction applies to all three ``*_vector`` variants.
  :param cmp:    Operation for the comparison.
  :param vals:   Array of length ``nelems``. Element ``i`` of ``ivars`` is
                 compared against ``vals[i]``.
  :returns:      The index of an element in the ``ivars`` array that satisfies
                 the wait condition. If the wait set is empty, this routine
                 returns ``SIZE_MAX``.

**Description:**
This routine blocks the caller until the condition ``(ivars[i] cmp vals[i])``
is true for at least one element in the wait set.

Valid ``cmp`` values are listed in :ref:`CMP_VALUES`.

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`WAIT_TEST_TYPES`.

ROCSHMEM_WAIT_UNTIL_SOME_VECTOR
-------------------------------

.. cpp:function:: __device__ size_t rocshmem_TYPENAME_wait_until_some_vector(TYPE *ivars, size_t nelems, size_t* indices, const int* status, int cmp, TYPE* vals)

  :param ivars:   Pointer to memory on the symmetric heap to wait for.
  :param nelems:  Number of elements in the ``ivars`` array.
  :param indices: List of indices with a length of at least ``nelems``.
  :param status:  Pass ``nullptr``. Non-null status arrays are not handled
                  correctly by the device vector waits; see the note on
                  ``wait_until_all_vector`` above.
  :param cmp:     Operation for the comparison.
  :param vals:    Array of length ``nelems``. Element ``i`` of ``ivars`` is
                  compared against ``vals[i]``.
  :returns:       The number of indices returned in the ``indices`` array. If
                  the wait set is empty, this routine returns ``0``.

**Description:**
This routine blocks the caller until the condition ``(ivars[i] cmp vals[i])``
is true for at least one element in the wait set, then returns the indices of
every element that currently satisfies it.

Valid ``cmp`` values are listed in :ref:`CMP_VALUES`.

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`WAIT_TEST_TYPES`.

ROCSHMEM_TEST
-------------

.. cpp:function:: __device__ int rocshmem_TYPENAME_test(TYPE *ivars, int cmp, TYPE val)

  :param ivars: Pointer to memory on the symmetric heap to wait for.
  :param cmp:   Operation for the comparison.
  :param val:   Value to compare the memory at ``ivars`` to.

  :returns:     ``1`` if the evaluation is true. ``0`` otherwise.

**Description:**
This routine tests if the condition ``(*ivars cmp val)`` is true.

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`WAIT_TEST_TYPES`.

ROCSHMEM_SIGNAL_WAIT_UNTIL_ON_STREAM
-------------------------------------

.. cpp:function:: __host__ void rocshmem_signal_wait_until_on_stream(uint64_t *sig_addr, int cmp, uint64_t cmp_value, hipStream_t stream)

  :param sig_addr:   Address of the signal variable on the symmetric heap.
  :param cmp:        Comparison operator (e.g., ROCSHMEM_CMP_EQ, ROCSHMEM_CMP_GE, etc.).
  :param cmp_value:  Value to compare against.
  :param stream:     HIP stream on which to enqueue the operation.
  :returns:          None.

**Description:**
This routine enqueues a wait operation on a HIP stream without blocking the calling thread. Work
enqueued on ``stream`` after this call does not start until the signal variable at ``sig_addr``
satisfies the comparison condition ``(*sig_addr cmp cmp_value)``. The caller must synchronize
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

.. _WAIT_TEST_TYPES:

Supported wait/test data types
-------------------------------

.. list-table:: Wait/Test Data Types
    :widths: 20 20
    :header-rows: 1

    * - TYPE
      - TYPENAME
    * - float
      - float
    * - double
      - double
    * - char
      - char
    * - signed char
      - schar
    * - short
      - short
    * - int
      - int
    * - long
      - long
    * - long long
      - longlong
    * - unsigned char
      - uchar
    * - unsigned short
      - ushort
    * - unsigned int
      - uint
    * - unsigned long
      - ulong
    * - unsigned long long
      - ulonglong
    * - __half
      - half
    * - __hip_bfloat16
      - bfloat16
    * - int8_t
      - int8
    * - int16_t
      - int16
    * - int32_t
      - int32
    * - int64_t
      - int64
    * - uint8_t
      - uint8
    * - uint16_t
      - uint16
    * - uint32_t
      - uint32
    * - uint64_t
      - uint64
    * - size_t
      - size
    * - ptrdiff_t
      - ptrdiff
