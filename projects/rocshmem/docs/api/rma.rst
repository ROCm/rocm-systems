.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-rma:

-----------------------------------------
Remote memory access routines
-----------------------------------------

- Routines with the ``_wave`` and ``_wg`` suffixes require all threads in a wavefront and workgroup, respectively,
  to call the routine with the same parameters.
- Routines with the ``_nbi`` substring will return as soon as the request is posted.
- Routines without the ``_nbi`` substring will block until the operation completes locally.
- Valid ``TYPENAME`` and ``TYPE`` values can be found in RMA_TYPES_.

ROCSHMEM_PUT
------------

.. cpp:function:: __device__ void rocshmem_TYPENAME_put(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_put_wave(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_put_wg(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_put_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_put_nbi_wave(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_put_nbi_wg(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_wave(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_wg(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_nbi(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_nbi_wave(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_nbi_wg(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)

  :param ctx:    Context with which to perform this operation.
  :param dest:   Destination address. Must be an address on the symmetric heap.
  :param source: Source address. Must be an address on the symmetric heap.
  :param nelems: The number of elements to transfer.
  :param pe:     PE of the remote process.
  :returns:      None.

**Description:**
This routine writes contiguous data of ``nelems`` elements from source on the calling PE to ``dest`` at ``pe``.

ROCSHMEM_PUTMEM
---------------

.. cpp:function:: __device__ void rocshmem_putmem(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_putmem_wave(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_putmem_wg(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_putmem_nbi_wave(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_putmem_nbi_wg(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_wave(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_wg(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_nbi(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_nbi_wave(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_nbi_wg(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)

    :param ctx:    Context with which to perform this operation.
    :param dest:   Destination address. Must be an address on the symmetric heap.
    :param source: Source address. Must be an address on the symmetric heap.
    :param nelems: Size of the transfer in bytes.
    :param pe:     PE of the remote process.

    :returns:      None.

**Description:**
This routine writes contiguous data of ``nelems`` bytes from source on the calling PE to ``dest`` at ``pe``.

ROCSHMEM_PUTMEM_ON_STREAM
--------------------------

.. cpp:function:: __host__ void rocshmem_putmem_on_stream(void *dest, const void *source, size_t nelems, int pe, hipStream_t stream)

    :param dest:    Destination address. Must be an address on the symmetric heap.
    :param source:  Source address. Must be an address on the symmetric heap.
    :param nelems:  Size of the transfer in bytes.
    :param pe:      PE of the remote process.
    :param stream:  HIP stream on which to enqueue the operation.

    :returns:       None.

**Description:**
This routine enqueues a putmem RMA operation on a HIP stream. The function writes contiguous
data of ``nelems`` bytes from source on the calling PE to ``dest`` at ``pe``. The operation
is enqueued on the specified stream and will execute asynchronously. The caller must
synchronize the stream (e.g., using ``hipStreamSynchronize``) to ensure completion.

ROCSHMEM_P
----------

.. cpp:function::  __device__ void rocshmem_TYPENAME_p(TYPE *dest, TYPE value, int pe)
.. cpp:function::  __device__ void rocshmem_ctx_TYPENAME_p(rocshmem_ctx_t ctx, TYPE *dest, TYPE value, int pe)

    :param ctx:    Context with which to perform this operation.
    :param dest:   Destination address. Must be an address on the symmetric heap.
    :param value:  Value to write to ``dest`` at ``pe``.
    :param pe:     PE of the remote process.

    :returns:      None.

**Description:**
This routine writes a single value to to ``dest`` at ``pe``.

ROCSHMEM_GET
------------

.. cpp:function:: __device__ void rocshmem_TYPENAME_get(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_get_wave(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_get_wg(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_get_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_get_nbi_wave(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_get_nbi_wg(TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_get(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_get_wave(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_get_wg(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_get_nbi(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_get_nbi_wave(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_get_nbi_wg(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, int pe)

    :param ctx:     Context with which to perform this operation.
    :param dest:    Destination address; Must be an address on the symmetric heap.
    :param source:  Source address. Must be an address on the symmetric heap.
    :param nelems:  The number of elements to transfer.
    :param pe:      PE of the remote process.

    :returns:       None.

**Description:**
This routine reads contiguous data of ``nelems`` elements from source on ``pe`` to ``dest`` on the calling PE.

ROCSHMEM_GETMEM
---------------

.. cpp:function:: __device__ void rocshmem_getmem(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_getmem_wave(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_getmem_wg(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_getmem_nbi_wave(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_getmem_nbi_wg(void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_getmem(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_getmem_wave(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_getmem_wg(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_getmem_nbi(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_getmem_nbi_wave(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_getmem_nbi_wg(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)

    :param ctx:     Context with which to perform this operation.
    :param dest:    Destination address. Must be an address on the symmetric heap.
    :param source:  Source address. Must be an address on the symmetric heap.
    :param nelems:  Size of the transfer in bytes.
    :param pe:      PE of the remote process.

    :returns:       None.

**Description:**
This routine reads contiguous data of ``nelems`` bytes from source on ``pe`` to ``dest`` on the calling PE.

ROCSHMEM_GETMEM_ON_STREAM
--------------------------

.. cpp:function:: __host__ void rocshmem_getmem_on_stream(void *dest, const void *source, size_t nelems, int pe, hipStream_t stream)

    :param dest:    Destination address. Must be an address on the symmetric heap.
    :param source:  Source address. Must be an address on the symmetric heap.
    :param nelems:  Size of the transfer in bytes.
    :param pe:      PE of the remote process.
    :param stream:  HIP stream on which to enqueue the operation.

    :returns:       None.

**Description:**
This routine enqueues a getmem RMA operation on a HIP stream. The function reads contiguous
data of ``nelems`` bytes from source on ``pe`` to ``dest`` on the calling PE. The operation
is enqueued on the specified stream and will execute asynchronously. The caller must
synchronize the stream (e.g., using ``hipStreamSynchronize``) to ensure completion.

ROCSHMEM_G
----------
.. cpp:function:: __device__ float rocshmem_ctx_float_g(rocshmem_ctx_t ctx, const float *source, int pe)
.. cpp:function:: __device__ float rocshmem_float_g(const float *source, int pe)

    :param ctx:     Context with which to perform this operation.
    :param source:  Source address. Must be an address on the symmetric heap.
    :param pe:      PE of the remote process.

    :returns:       The value read from source at ``pe``.

**Description:**
This routine reads and returns single value from source at ``pe``.

Supported RMA data types
------------------------

The following table lists the supported RMA data types:

.. _RMA_TYPES:

.. list-table:: RMA Data Types
    :widths: 10 20 20
    :header-rows: 1

    * - TYPE
      - TYPENAME
      - Supported
    * - float
      - float
      - Yes
    * - double
      - double
      - Yes
    * - long double
      - longdouble
      - No
    * - char
      - char
      - Yes
    * - signed char
      - schar
      - Yes
    * - short
      - short
      - Yes
    * - int
      - int
      - Yes
    * - long
      - long
      - Yes
    * - long long
      - longlong
      - Yes
    * - unsigned char
      - uchar
      - Yes
    * - unsigned short
      - ushort
      - Yes
    * - unsigned int
      - uint
      - Yes
    * - unsigned long
      - ulong
      - Yes
    * - unsigned long long
      - ulonglong
      - Yes
    * - int8_t
      - int8
      - No
    * - int16_t
      - int16
      - No
    * - int32_t
      - int32
      - No
    * - int64_t
      - int64
      - Yes
    * - uint8_t
      - uint8
      - No
    * - uint16_t
      - uint16
      - No
    * - uint32_t
      - uint32
      - No
    * - uint64_t
      - uint64
      - No
    * - size_t
      - size
      - No
    * - ptrdiff_t
      - ptrdiff
      - No

