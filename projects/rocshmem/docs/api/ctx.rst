.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-ctx:

-----------------------------------
Context management routines
-----------------------------------

ROCSHMEM_CTX_CREATE
-------------------

.. cpp:function:: __device__ int rocshmem_wg_ctx_create(int64_t options, rocshmem_ctx_t *ctx)
.. cpp:function:: __device__ int rocshmem_wg_team_create_ctx(rocshmem_team_t team, long options, rocshmem_ctx_t *ctx)

  :param team:    Team handle to derive the context from.
  :param options: Options for context creation. Ignored in current design; use the value ``0``.
  :param ctx:     A handle to the newly created context.

  :returns:       All threads returns ``0`` if the context was created successfully.
                  If any thread returns non-zero value, the operation fails, ctx is set to ``ROCSHMEM_CTX_INVALID`` and a
                  higher number of ``ROCSHMEM_MAX_NUM_CONTEXTS`` is required.

**Description:**
This routine creates an rocSHMEM context. By design, the context is private to the calling work-group.
It must be called collectively by all threads in the work-group. If the context was created successfully, a value
of zero is returned and the context handle pointed to by ctx specifies a valid context; otherwise, a nonzero value
is returned and ctx is set to ``ROCSHMEM_CTX_INVALID``. An unsuccessful context creation call is not treated as an
error and the rocSHMEM library remains in a correct state. The creation call can be reattempted after additional
resources become available.

ROCSHMEM_CTX_DESTROY
--------------------

.. cpp:function:: __device__ void rocshmem_wg_ctx_destroy(rocshmem_ctx_t *ctx)

  :param ctx:     Context handle.

  :returns:       None.

**Description:**
This routine destroys an rocSHMEM context. It must be called collectively by all threads in the work-group.
If ctx has the value ``ROCSHMEM_CTX_INVALID``, no operation is performed.

ROCSHMEM_CTX_CREATE
-------------------

.. cpp:function:: __host__ int rocshmem_ctx_create(int64_t options, rocshmem_ctx_t *ctx)

  :param options: Options for context creation. Ignored in the current design;
                  use the value ``0``.
  :param ctx:     A handle to the newly created context.
  :returns:       Always zero. Pool exhaustion is **not** reported through the
                  return value.

**Description:**
Creates a host-side rocSHMEM context, for use with the host and stream-ordered
APIs.

.. WARNING::

   The context pool is bounded by ``ROCSHMEM_MAX_NUM_HOST_CONTEXTS`` and the
   default context already occupies one slot, so at least two slots are needed
   for one additional context. With the default value of ``1``, the first call
   to this routine exhausts the pool: it trips an assertion in a debug build,
   and indexes out of bounds when assertions are compiled out. See
   :ref:`rocshmem-api-env-variables`.

ROCSHMEM_CTX_DESTROY
--------------------

.. cpp:function:: __host__ void rocshmem_ctx_destroy(rocshmem_ctx_t ctx)

  :param ctx: Context handle.
  :returns:   None.

**Description:**
Destroys a host-side rocSHMEM context previously created with
``rocshmem_ctx_create``.

Context options
---------------

The ``options`` argument of the context creation routines accepts a bitwise OR
of the following constants:

.. list-table:: Context creation options
    :widths: 30 10 60
    :header-rows: 1

    * - Constant
      - Value
      - Description
    * - ``ROCSHMEM_CTX_ZERO``
      - ``0``
      - No options requested. The portable default.
    * - ``ROCSHMEM_CTX_NOSTORE``
      - ``1``
      - The context is not used for stores.
    * - ``ROCSHMEM_CTX_SERIALIZED``
      - ``2``
      - The context is not accessed concurrently.
    * - ``ROCSHMEM_CTX_WG_PRIVATE``
      - ``4``
      - The context is private to the creating work-group.
    * - ``ROCSHMEM_CTX_SHARED``
      - ``8``
      - Reserved for shared contexts. Context creation currently ignores this
        option on every backend, and it does not make a work-group context safe
        to share between work-groups.

ROCSHMEM_GET_DEVICE_CTX
-----------------------

.. WARNING::
   Routine ``rocshmem_get_device_ctx`` has been deprecated. Use
   ``rocshmem_hipmodule_init`` instead, which also works under HIP graph
   capture. See :ref:`rocshmem-api-init`.

.. cpp:function:: [[deprecated]] __host__ void * rocshmem_get_device_ctx()

  :param:    None.

  :returns: Returns ``ROCSHMEM_CTX_DEFAULT`` device pointer that users.
            can query from one instance of rocSHMEM host library and
            use later for dynamic module initialization in
            kernel bitcode device library in the same application.

**Description:**
This routine queries rocSHMEM default device context from host API.
