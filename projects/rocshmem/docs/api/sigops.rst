.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-sigops:

---------------------
Signaling operations
---------------------

ROCSHMEM_PUTMEM_SIGNAL
----------------------

.. cpp:function:: __device__ void rocshmem_putmem_signal(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_putmem_signal_wave(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_putmem_signal_wg(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_putmem_signal_nbi(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_putmem_signal_nbi_wave(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_putmem_signal_nbi_wg(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_signal(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_signal_wave(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_signal_wg(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_signal_nbi(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_signal_nbi_wave(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_putmem_signal_nbi_wg(rocshmem_ctx_t ctx, void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)

  :param ctx:      Context with which to perform this operation.
  :param dest:     Destination address. Must be an address on the symmetric heap.
  :param source:   Source address. Must be an address on the symmetric heap.
  :param nelems:   The number of bytes to transfer.
  :param sig_addr: Signal address. Must be an address on the symmetric heap.
  :param signal:   Signal value.
  :param sig_op:   Atomic operation to apply the signal value.
  :param pe:       PE of the remote process.
  :returns:        None.

**Description:**
This function writes contiguous data of ``nelems`` bytes from source on the calling PE to ``dest`` at ``pe``, 
then applies ``sig_op`` at ``sig_addr`` with the signal value. 
Valid ``sig_op values`` are listed in SIGNAL_OPERATORS_.

ROCSHMEM_PUT_SIGNAL
-------------------

.. cpp:function:: __device__ void rocshmem_TYPENAME_put_signal(TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_put_signal_wave(TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_put_signal_wg(TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_put_signal_nbi(TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_put_signal_nbi_wave(TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_TYPENAME_put_signal_nbi_wg(TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_signal(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_signal_wave(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_signal_wg(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_signal_nbi(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_signal_nbi_wave(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_put_signal_nbi_wg(rocshmem_ctx_t ctx, TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)

  :param ctx:      Context with which to perform this operation.
  :param dest:     Destination address. Must be an address on the symmetric heap.
  :param source:   Source address. Must be an address on the symmetric heap.
  :param nelems:   The number of elements of size ``TYPE`` to transfer.
  :param sig_addr: Signal address. Must be an address on the symmetric heap.
  :param signal:   Signal value.
  :param sig_op:   Atomic operation to apply the signal value.
  :param pe:       PE of the remote process.
  :returns:        None.

**Description:**
This function writes contiguous data of ``nelems`` elements of ``TYPE`` from source on the calling PE to ``dest`` at ``pe``, 
then applies ``sig_op`` at ``sig_addr`` with the signal value.
Valid ``sig_op values`` are listed in SIGNAL_OPERATORS_.
Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`RMA_TYPES`.

ROCSHMEM_PUTMEM_SIGNAL_ON_STREAM
---------------------------------

.. cpp:function:: __host__ void rocshmem_putmem_signal_on_stream(void *dest, const void *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe, hipStream_t stream)

  :param dest:     Destination address on the remote PE. Must be an address on the symmetric heap.
  :param source:   Source address on the local PE. Must be an address on the symmetric heap.
  :param nelems:   Size of the transfer in bytes.
  :param sig_addr: Address of signal variable on the remote PE. Must be an address on the symmetric heap.
  :param signal:   Signal value to be written.
  :param sig_op:   Signal operation (ROCSHMEM_SIGNAL_SET or ROCSHMEM_SIGNAL_ADD).
  :param pe:       PE number of the remote PE.
  :param stream:   HIP stream on which to enqueue the operation.
  :returns:        None.

**Description:**
This routine enqueues a put-with-signal operation on a HIP stream. The function writes contiguous
data of ``nelems`` bytes from source on the calling PE to ``dest`` at ``pe``, then applies ``sig_op``
at ``sig_addr`` with the signal value. The operation is enqueued on the specified stream and will
execute asynchronously. The caller must synchronize the stream (e.g., using ``hipStreamSynchronize``)
to ensure completion.

Valid ``sig_op`` values are listed in SIGNAL_OPERATORS_.

ROCSHMEM_SIGNAL_FETCH
---------------------

.. cpp:function:: __device__ uint64_t rocshmem_signal_fetch(const uint64_t *sig_addr)
.. cpp:function:: __device__ uint64_t rocshmem_signal_fetch_wg(const uint64_t *sig_addr)
.. cpp:function:: __device__ uint64_t rocshmem_signal_fetch_wave(const uint64_t *sig_addr)

  :param sig_addr: Signal address. Must be an address on the symmetric heap.
  :returns:        Value at ``sig_addr``.

**Description:**
This function atomically fetches the value stored at ``sig_addr``.

Signal operators
----------------
.. _SIGNAL_OPERATORS:

.. list-table:: Signal Operators 
    :widths: 20 40
    :header-rows: 1

    * - Value 
      - Description 
    * - ROCSHMEM_SIGNAL_SET
      - The signaling operation routines will atomically set the signal value at ``sig_addr``.
    * - ROCSHMEM_SIGNAL_ADD
      - The signaling operation routines will atomically add the signal value at ``sig_addr``.

