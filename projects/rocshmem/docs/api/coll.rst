.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-coll:

---------------------------
Collective routines
---------------------------

ROCSHMEM_BARRIER_ALL
--------------------

.. cpp:function:: __device__ void rocshmem_barrier_all()
.. cpp:function:: __device__ void rocshmem_barrier_all_wave()
.. cpp:function:: __device__ void rocshmem_barrier_all_wg()

  :returns:   None.

**Description:**
This routine performs a collective barrier across all PEs in the system.
The caller is blocked until the barrier is resolved and all updates local and remote are completed.
These APIs should be called from only one thread/wavefront/workgroup within the grid to avoid undefined behavior.

ROCSHMEM_BARRIER_ALL_ON_STREAM
-------------------------------

.. cpp:function:: __host__ void rocshmem_barrier_all_on_stream(hipStream_t stream)

  :param stream: HIP stream on which to enqueue the operation.
  :returns:      None.

**Description:**
This routine enqueues a collective barrier operation on a HIP stream. The barrier is performed
across all PEs in the system. The operation is enqueued on the specified stream and will execute
asynchronously. The caller must synchronize the stream (e.g., using ``hipStreamSynchronize``)
to ensure completion.

ROCSHMEM_BARRIER
----------------

.. cpp:function:: __device__ void rocshmem_ctx_barrier(rocshmem_ctx_t ctx, rocshmem_team_t team)
.. cpp:function:: __device__ void rocshmem_ctx_barrier_wave(rocshmem_ctx_t ctx, rocshmem_team_t team)
.. cpp:function:: __device__ void rocshmem_ctx_barrier_wg(rocshmem_ctx_t ctx, rocshmem_team_t team)

  :param ctx: Context with which to perform this operation.
  :returns:   None.

**Description:**
This routine performs a collective barrier between all PEs in the system.
The caller is blocked until the barrier is resolved.

ROCSHMEM_TEAM_SYNC
------------------

.. cpp:function:: __device__ void rocshmem_ctx_sync(rocshmem_ctx_t ctx, rocshmem_team_t team)
.. cpp:function:: __device__ void rocshmem_ctx_sync_wave(rocshmem_ctx_t ctx, rocshmem_team_t team)
.. cpp:function:: __device__ void rocshmem_ctx_sync_wg(rocshmem_ctx_t ctx, rocshmem_team_t team)

  :param ctx:  Context with which to perform this operation.
  :param team: Team with which to perform this operation.
  :returns:    None.

**Description:**
This routine registers the arrival of a PE at a barrier.
The caller is blocked until the synchronization is resolved.

Unlike the ``shmem_barrier_all`` routine, ``shmem_team_sync`` only ensures the
completion and visibility of previously issued memory stores, but does not
ensure the completion of remote memory updates issued via OpenSHMEM routines.

ROCSHMEM_SYNC_ALL
-----------------

.. cpp:function:: __device__ void rocshmem_sync_all()
.. cpp:function:: __device__ void rocshmem_sync_all_wave()
.. cpp:function:: __device__ void rocshmem_sync_all_wg()

  :returns:    None.

**Description:**
These routines behaves the same way as ``rocshmem_team_sync_*`` when called on the world team.
These APIs should be called from only one thread/wavefront/workgroup within the grid to avoid undefined behavior.

ROSHMEM_ALLTOALL
----------------

.. cpp:function:: __device__ void rocshmem_TYPENAME_alltoall_wg(rocshmem_team_t team, TYPE *dest, const TYPE *source, int nelems)
.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_alltoall_wg(rocshmem_ctx_t ctx, rocshmem_team_t team, TYPE *dest, const TYPE *source, int nelems)

  :param team:   The team participating in the collective.
  :param dest:   Destination address. Must be an address on the
                 symmetric heap.
  :param source: Source address. Must be an address on the symmetric
                 heap.
  :param nelems: Number of data blocks transferred per pair of PEs.
  :returns:      None.

**Description:**
This routine exchanges a fixed amount of contiguous data blocks between all pairs
of PEs participating in the collective routine.
This function must be called as a work-group collective.

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`RMA_TYPES`.

ROCSHMEM_ALLTOALLMEM_ON_STREAM
-------------------------------

.. cpp:function:: __host__ void rocshmem_alltoallmem_on_stream(rocshmem_team_t team, void *dest, const void *source, size_t size, hipStream_t stream)

  :param team:   The team participating in the collective.
  :param dest:   Destination address. Must be an address on the symmetric heap.
  :param source: Source address. Must be an address on the symmetric heap.
  :param size:   Number of bytes to transfer per pair of PEs.
  :param stream: HIP stream on which to enqueue the operation.
  :returns:      None.

**Description:**
This routine enqueues an alltoall collective operation on a HIP stream. The function
exchanges a fixed amount of contiguous data blocks between all pairs of PEs participating
in the collective routine. The operation is enqueued on the specified stream and will
execute asynchronously. The caller must synchronize the stream (e.g., using
``hipStreamSynchronize``) to ensure completion.

This function creates a separate context for each workgroup to avoid contention on the
default context, allowing parallel execution across multiple streams.

ROCSHMEM_BROADCAST
------------------

.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_broadcast_wg(rocshmem_ctx_t ctx, rocshmem_team_t team, TYPE *dest, const TYPE *source, int nelems, int pe_root)

  :param ctx:    Context with which to perform this collective.
  :param team:   The team participating in the collective.
  :param dest:   Destination address. Must be an address on the
                 symmetric heap.
  :param source: Source address. Must be an address on the symmetric
                 heap.
  :param nelems: Number of data blocks transferred per pair of PEs.
  :returns:      None.

**Description:**
This routine performs a broadcast across PEs in the team.
The caller is blocked until the broadcast completes.

Valid ``TYPENAME`` and ``TYPE`` values are listed in :ref:`RMA_TYPES`.

ROCSHMEM_BROADCASTMEM_ON_STREAM
--------------------------------

.. cpp:function:: __host__ void rocshmem_broadcastmem_on_stream(rocshmem_team_t team, void *dest, const void *source, size_t nelems, int pe_root, hipStream_t stream)

  :param team:    The team participating in the collective.
  :param dest:    Destination address. Must be an address on the symmetric heap.
  :param source:  Source address. Must be an address on the symmetric heap.
  :param nelems:  Number of bytes to broadcast.
  :param pe_root: Root PE (relative to team) from which to broadcast.
  :param stream:  HIP stream on which to enqueue the operation.
  :returns:       None.

**Description:**
This routine enqueues a broadcast collective operation on a HIP stream. The function broadcasts
data from the root PE to all other PEs participating in the collective routine. The operation
is enqueued on the specified stream and will execute asynchronously. The caller must synchronize
the stream (e.g., using ``hipStreamSynchronize``) to ensure completion.

This function creates a separate context for each workgroup to avoid contention on the
default context, allowing parallel execution across multiple streams.

ROCSHMEM_FCOLLECT
-----------------

.. cpp:function:: __device__ void rocshmem_ctx_TYPENAME_fcollect_wg(rocshmem_ctx_t ctx, rocshmem_team_t team, TYPE *dest, const TYPE *source, int nelems)

  :param ctx:    Context with which to perform this collective.
  :param team:   The team participating in the collective.
  :param dest:   Destination address. Must be an address on the
                 symmetric heap.
  :param source: Source address. Must be an address on the symmetric
                 heap.
  :param nelems: Number of data blocks transferred per pair of PEs.
  :returns:      None.

**Description:**
This routine concatenates blocks of data from multiple PEs to an array in every
PE participating in the collective routine.

ROCSHMEM_REDUCTION
------------------
.. cpp:function:: __device__ int rocshmem_ctx_TYPENAME_OPNAME_reduce_wg(rocshmem_ctx_t ctx, rocshmem_team_t team, TYPE *dest, const TYPE *source, int nreduce)

  :param ctx:     Context with which to perform this collective.
  :param team:    The team participating in the collective.
  :param dest:    Destination address. Must be an address on the
                  symmetric heap.
  :param source:  Source address. Must be an address on the symmetric
                  heap.
  :param nreduce: Number of data blocks transferred per pair of PEs.
  :returns:       Zero on successful local completion. Nonzero otherwise.


**Description:**
This routine  performs an allreduce operation across PEs in the team.

Valid ``TYPENAME``, ``TYPE``, and ``OPNAME`` values are listed in :ref:`REDUCE_TYPES`.

Supported reduction types and operations
----------------------------------------

.. _REDUCE_TYPES:

.. list-table:: Reduction Types, Names and Operations
    :widths: 20 20 20 20
    :header-rows: 1

    * - TYPE
      - TYPENAME
      - OPNAME
      - Supported
    * - char
      - char
      - max, min, sum, prod
      - No
    * - signed char
      - schar
      - max, min, sum, prod
      - No
    * - short
      - short
      - max, min, sum, prod
      - Yes
    * - int
      - int
      - max, min, sum, prod
      - Yes
    * - long
      - long
      - max, min, sum, prod
      - Yes
    * - long long
      - longlong
      - max, min, sum, prod
      - Yes
    * - ptrdiff_t
      - ptrdiff
      - max, min, sum, prod
      - No
    * - unsigned char
      - uchar
      - and, or, xor, max, min, sum, prod
      - No
    * - unsigned short
      - ushort
      - and, or, xor, max, min, sum, prod
      - No
    * - unsigned int
      - uint
      - and, or, xor, max, min, sum, prod
      - No
    * - unsigned long
      - ulong
      - and, or, xor, max, min, sum, prod
      - No
    * - unsigned long long
      - ulonglong
      - and, or, xor, max, min, sum, prod
      - No
    * - int8_t
      - int8
      - and, or, xor, max, min, sum, prod
      - No
    * - int16_t
      - int16
      - and, or, xor, max, min, sum, prod
      - No
    * - int32_t
      - int32
      - and, or, xor, max, min, sum, prod
      - No
    * - int64_t
      - int64
      - and, or, xor, max, min, sum, prod
      - No
    * - uint8_t
      - uint8
      - and, or, xor, max, min, sum, prod
      - No
    * - uint16_t
      - uint16
      - and, or, xor, max, min, sum, prod
      - No
    * - uint32_t
      - uint32
      - and, or, xor, max, min, sum, prod
      - No
    * - uint64_t
      - uint64
      - and, or, xor, max, min, sum, prod
      - No
    * - size_t
      - size
      - and, or, xor, max, min, sum, prod
      - No
    * - float
      - float
      - max, min, sum, prod
      - Yes
    * - double
      - double
      - max, min, sum, prod
      - Yes
    * - long double
      - longdouble
      - max, min, sum, prod
      - No
    * - double _Complex
      - complexd
      - sum, prod
      - No
    * - float _Complex
      - complexf
      - sum, prod
      - No
