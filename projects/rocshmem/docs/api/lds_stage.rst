.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, LDS, TDM, Tensor Data Mover, staging

.. _rocshmem-api-lds-stage:

------------------------------------------
LDS-staged copy routines
------------------------------------------

On gfx1250, rocSHMEM can route large work-group and wave ``put``/``get``
transfers through a *staging buffer in LDS* (Local Data Share) instead of
copying global-to-global through the register file. The staged path moves data
``global -> LDS -> global`` using asynchronous copies, double-buffered so that
the next tile is fetched while the current tile is written out.

Two backends implement the staged copy, selected at build time:

* ``-DUSE_LDS_STAGE=ON`` - the portable asynchronous global-to-LDS backend.
* ``-DUSE_TDM=ON`` - the gfx1250 **TDM** (Tensor Data Mover) DMA engine, which
  transfers directly between global memory and LDS without consuming VGPRs.

Both are off by default. If rocSHMEM was built without either option, the
routines on this page are still callable but inert:
``rocshmem_query_lds_stage_bytes()`` returns ``0``,
``rocshmem_set_lds_stage()`` is a no-op, and every transfer uses the ordinary
copy path. Application code therefore does not need to be guarded by
``#ifdef``.

.. note::

   The LDS-staged path applies only to the **work-group** (``_wg``) and
   **wave** (``_wave``) variants of ``put``/``get``. Thread-granular and
   host-side transfers are unaffected.

Usage model
-----------

The staging buffer is supplied by the application as dynamic LDS, so that
rocSHMEM never reserves LDS from kernels that do not use the fast path. There
are three steps:

#. On the host, ask the runtime how many bytes of dynamic LDS the fast path
   wants, and pass that as the dynamic-shared-memory argument of the kernel
   launch.
#. In the kernel, declare the ``extern __shared__`` buffer and register it with
   ``rocshmem_set_lds_stage()``.
#. Issue ``_wg`` / ``_wave`` ``put``/``get`` calls as usual.

.. code-block:: cpp

  __global__ void my_kernel(char *dest, const char *source, size_t nbytes,
                            int pe) {
    __shared__ rocshmem_ctx_t ctx;
    rocshmem_wg_ctx_create(ROCSHMEM_CTX_WG_PRIVATE, &ctx);
    if (ctx == ROCSHMEM_CTX_INVALID) {
      return;
    }

    // Register after creating the context: rocshmem_wg_ctx_create() clears
    // any previous registration. Opt into the LDS-staged copy path. Returns 0 (and is a no-op) on
    // builds without USE_LDS_STAGE/USE_TDM.
    extern __shared__ uint8_t lds_stage[];
    rocshmem_set_lds_stage(lds_stage, rocshmem_query_lds_stage_bytes());

    rocshmem_ctx_putmem_wg(ctx, dest, source, nbytes, pe);

    rocshmem_wg_ctx_destroy(&ctx);
  }

  // Host side
  size_t shared_bytes = rocshmem_query_lds_stage_bytes();
  hipLaunchKernelGGL(my_kernel, grid, block, shared_bytes, stream,
                     dest, source, nbytes, pe);

.. note::

  Dynamic LDS reduces the number of work-groups that can be co-resident on a
  compute unit. Use ``hipOccupancyMaxActiveBlocksPerMultiprocessor()`` with
  ``rocshmem_query_lds_stage_bytes()`` as the dynamic-shared-memory argument to
  check how many work-groups the device can sustain before sizing the grid.

Tuning
------

Three environment variables control the staged path at runtime. See
:ref:`rocshmem-api-env-variables` for the full description:

* ``ROCSHMEM_LDS_STAGE_TILE_BYTES`` - tile size. The buffer that
  ``rocshmem_query_lds_stage_bytes()`` reports is this value **doubled**, for
  double buffering.
* ``ROCSHMEM_LDS_STAGE_PUT_THRESHOLD`` - minimum ``put`` size routed through
  the staged path (default ``16384`` bytes).
* ``ROCSHMEM_LDS_STAGE_GET_THRESHOLD`` - minimum ``get`` size routed through
  the staged path (default ``1024`` bytes).

Transfers below the relevant threshold take the ordinary copy path, because the
staged path's issue latency is not amortized at small sizes.

.. note::

  For ``_wave`` transfers the registered buffer is split evenly across the
  waves of the block, so each wave's effective tile is approximately
  ``rocshmem_query_lds_stage_bytes() / (2 * waves_per_block)`` (rounded down to
  a whole element). If a wave's slot ends up smaller than 16 bytes there is no
  room for a double-buffered tile and that transfer falls back to the ordinary
  copy path.

ROCSHMEM_QUERY_LDS_STAGE_BYTES
------------------------------

.. cpp:function:: __host__ __device__ size_t rocshmem_query_lds_stage_bytes()

   :returns: Number of bytes of dynamic LDS the LDS-staged copy path requires,
             or ``0`` if rocSHMEM was not built with ``USE_LDS_STAGE`` or
             ``USE_TDM``.

**Description:**
Returns the dynamic LDS allocation that the staged copy fast path wants for
work-group and wave ``put``/``get``. The value reflects
``ROCSHMEM_LDS_STAGE_TILE_BYTES``, doubled for double buffering. Use the host
variant to size the kernel launch and the device variant to size the buffer
passed to ``rocshmem_set_lds_stage()``; both return the same value.

ROCSHMEM_SET_LDS_STAGE
----------------------

.. cpp:function:: __device__ void rocshmem_set_lds_stage(void *lds, size_t lds_bytes)

  :param lds:       Pointer to a work-group-local LDS buffer of at least
                    ``lds_bytes`` bytes, aligned to at least 8 bytes (256 bytes
                    recommended), typically the kernel's ``extern __shared__``
                    array.
  :param lds_bytes: Size of the buffer in bytes. Pass
                    ``rocshmem_query_lds_stage_bytes()``. A value of ``0``
                    disables the staged path for this work-group.
  :returns:         None.

**Description:**
All threads in the work-group must call this routine collectively; it executes
a work-group barrier when staging is enabled, so calling it from a single
thread or a single wave can hang the kernel.

It registers an LDS staging buffer for subsequent ``_wg`` and ``_wave``
``put``/``get`` calls. The registration is per work-group and lasts for the
remainder of the kernel, but ``rocshmem_wg_ctx_create()`` clears it — register
the buffer *after* each context creation, or staging is silently disabled.

.. important::

   On builds with ``USE_LDS_STAGE`` or ``USE_TDM``, initialize the registration
   before issuing any ``_wg`` or ``_wave`` transfer: either register a buffer,
   or call ``rocshmem_set_lds_stage(nullptr, 0)`` to disable staging explicitly.
   A work-group that uses only the default context and never calls either leaves
   the registration uninitialized, so the ordinary copy path is not a guaranteed
   fallback.

On builds without ``USE_LDS_STAGE`` or ``USE_TDM`` this routine does nothing.

Deprecated aliases
------------------

The routines below are the original TDM-specific names, kept so that existing
code continues to compile.

.. cpp:function:: __host__ __device__ size_t rocshmem_query_tdm_lds_bytes()

   .. deprecated:: 3.9.0
      Use :cpp:func:`rocshmem_query_lds_stage_bytes` instead. This routine
      forwards to it unchanged and will be removed in a future release.

.. cpp:function:: __device__ void rocshmem_set_tdm_lds(void *lds, size_t lds_bytes)

   .. deprecated:: 3.9.0
      Use :cpp:func:`rocshmem_set_lds_stage` instead. This routine forwards to
      it unchanged and will be removed in a future release.
