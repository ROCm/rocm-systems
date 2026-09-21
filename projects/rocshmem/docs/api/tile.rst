.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-tile:

---------------------------
Tile routines
---------------------------

.. note::

   The tile API is currently only supported with the **IPC backend**. Attempting to
   use these routines with other backends results in an error.

The tile API provides tensor-aware data movement and collective operations that
operate on multi-dimensional sub-regions (tiles) of symmetric-heap tensors.
Each function is available in three execution granularities:

- No suffix: thread-granular — each thread operates independently.
- ``_wave``: wave-granular — all threads in a wavefront collectively participate.
- ``_wg``: workgroup-granular — all threads in the workgroup collectively participate.

Context variants (``rocshmem_ctx_tile_*``) accept an explicit ``rocshmem_ctx_t``;
non-context variants use the default context. All collective operations — context
and non-context alike — take an explicit ``rocshmem_team_t`` parameter.

All functions are templated on the tensor and coordinate types:

- ``dst_tensor_t`` / ``src_tensor_t``: tensor descriptor types carrying element type,
  shape, and strides.
- ``tuple_t``: coordinate tuple type (e.g. a struct of integers matching the tensor rank).

-------------------------------
Remote memory access routines
-------------------------------

ROCSHMEM_TILE_PUT
-----------------

.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_put(dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_put_wave(dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_put_wg(dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_put(rocshmem_ctx_t ctx, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_put_wave(rocshmem_ctx_t ctx, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_put_wg(rocshmem_ctx_t ctx, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)

  :param ctx:         Context with which to perform this operation.
  :param dst:         Destination tensor on the remote PE. Must reside on the symmetric heap.
  :param src:         Source tensor on the local PE. Must reside on the symmetric heap.
  :param start_coord: Starting coordinates of the tile.
  :param boundary:    Boundary coordinates of the tile (exclusive upper bound per dimension).
  :param pe:          PE of the remote process.
  :param flags:       Operation flags (reserved, pass 0).
  :returns:           Zero on success, nonzero on failure.

**Description:**
Writes the tile region ``[start_coord, boundary)`` of the local ``src`` tensor to
the corresponding region of the remote ``dst`` tensor on PE ``pe``.

ROCSHMEM_TILE_GET
-----------------

.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_get(dst_tensor_t dst, src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_get_wave(dst_tensor_t dst, src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_get_wg(dst_tensor_t dst, src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_get(rocshmem_ctx_t ctx, dst_tensor_t dst, src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_get_wave(rocshmem_ctx_t ctx, dst_tensor_t dst, src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_get_wg(rocshmem_ctx_t ctx, dst_tensor_t dst, src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe, uint64_t flags)

  :param ctx:         Context with which to perform this operation.
  :param dst:         Destination tensor on the local PE. Must reside on the symmetric heap.
  :param src:         Source tensor on the remote PE. Must reside on the symmetric heap.
  :param start_coord: Starting coordinates of the tile.
  :param boundary:    Boundary coordinates of the tile (exclusive upper bound per dimension).
  :param pe:          PE of the remote process.
  :param flags:       Operation flags (reserved, pass 0).
  :returns:           Zero on success, nonzero on failure.

**Description:**
Reads the tile region ``[start_coord, boundary)`` of the remote ``src`` tensor on
PE ``pe`` into the corresponding region of the local ``dst`` tensor.

-------------------------------
Collective routines
-------------------------------

ROCSHMEM_TILE_ALLGATHER
-----------------------

.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_allgather(rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_allgather_wave(rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_allgather_wg(rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_allgather(rocshmem_ctx_t ctx, rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_allgather_wave(rocshmem_ctx_t ctx, rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_allgather_wg(rocshmem_ctx_t ctx, rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, uint64_t flags)

  :param ctx:         Context with which to perform this operation.
  :param team:        The team participating in the collective.
  :param dst:         Destination tensor. Must reside on the symmetric heap.
  :param src:         Source tensor. Must reside on the symmetric heap.
  :param start_coord: Starting coordinates of the tile.
  :param boundary:    Boundary coordinates of the tile (exclusive upper bound per dimension).
  :param flags:       Operation flags (reserved, pass 0).
  :returns:           Zero on success, nonzero on failure.

**Description:**
Gathers the tile region ``[start_coord, boundary)`` from all PEs in the team into
every PE's ``dst`` tensor.

ROCSHMEM_TILE_BROADCAST
-----------------------

.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_broadcast(rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe_root, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_broadcast_wave(rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe_root, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_tile_broadcast_wg(rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe_root, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_broadcast(rocshmem_ctx_t ctx, rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe_root, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_broadcast_wave(rocshmem_ctx_t ctx, rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe_root, uint64_t flags)
.. cpp:function:: template<typename dst_tensor_t, typename src_tensor_t, typename tuple_t> __device__ int rocshmem_ctx_tile_broadcast_wg(rocshmem_ctx_t ctx, rocshmem_team_t team, dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord, tuple_t boundary, int pe_root, uint64_t flags)

  :param ctx:         Context with which to perform this operation.
  :param team:        The team participating in the collective.
  :param dst:         Destination tensor. Must reside on the symmetric heap.
  :param src:         Source tensor. Must reside on the symmetric heap.
  :param start_coord: Starting coordinates of the tile.
  :param boundary:    Boundary coordinates of the tile (exclusive upper bound per dimension).
  :param pe_root:     Root PE (relative to team) from which to broadcast.
  :param flags:       Operation flags (reserved, pass 0).
  :returns:           Zero on success, nonzero on failure.

**Description:**
Broadcasts the tile region ``[start_coord, boundary)`` from the root PE's ``src``
tensor to all other PEs' ``dst`` tensors in the team.


