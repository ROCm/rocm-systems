.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-init:

---------------------------------------
Library setup, exit, and query routines
---------------------------------------

ROCSHMEM_INIT
-------------

.. cpp:function:: __host__ void rocshmem_init(void)

  :Parameters: None.
  :returns: None.

**Description:**
This routine initializes the rocSHMEM library and underlying transport layer.
Before ``rocshmem_init`` is called,
you must select the device that this PE is associated to by calling
`hipSetDevice
<https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/group___device.html#ga43c1e7f15925eeb762195ccb5e063eae>`_.

.. WARNING::
   Routine `rocshmem_wg_init` has been deprecated.

.. cpp:function:: [[deprecated]] __device__ void rocshmem_wg_init(void)

  :Parameters: None.
  :returns: None.

**Description:**
This routine has been deprecated, please do not use.
This routine initializes device-side rocSHMEM resources.
It must be called before any threads in this work-group invoke other rocSHMEM functions.
It must be called collectively by all threads in the work-group.

ROCSHMEM_FINALIZE
-----------------
.. cpp:function:: __host__ void rocshmem_finalize(void)

  :Parameters: None.
  :returns: None.

**Description:**
This routine finalizes the rocSHMEM library.

.. WARNING::
   Routine `rocshmem_wg_finalize` has been deprecated.

.. cpp:function:: [[deprecated]] __device__ void rocshmem_wg_finalize(void)

  :Parameters: None.
  :returns: None.

**Description:**
This routine has been deprecated, please do not use.
This routine finalizes device-side rocSHMEM resources.
It must be called before work-group completion if the work-group also called ``rocshmem_wg_init``.
It must be called collectively by all threads in the work-group.

ROCSHMEM_INIT_ATTR
------------------
.. cpp:function:: __host__ int rocshmem_init_attr(unsigned int flags, rocshmem_init_attr_t *attr)

  :param flags: The initialization method to be used.
  :param attr:  Attribute structure specifying input characteristics.

  :returns int: Returns ``0`` on success; otherwise, returns a nonzero value.

**Description:**
This routine initializes the rocSHMEM runtime and underlying transport layer using
the provided mode and attributes.
The parameter ``flags`` can be either
``ROCSHMEM_INIT_WITH_UNIQUEID`` or ``ROCSHMEM_INIT_WITH_MPI_COMM``.

ROCSHMEM_GET_UNIQUEID
---------------------
.. cpp:function:: __host__ int rocshmem_get_uniqueid(rocshmem_uniqueid_t *uid)

  :param uid: Pointer to a unique ID handle.
  :returns:   Returns ``0`` on success; otherwise, returns a nonzero value.

**Description:**
This routine returns a unique ID.

ROCSHMEM_SET_ATTR_UNIQUEID_ARGS
-------------------------------
.. cpp:function:: __host__ int rocshmem_set_attr_uniqueid_args(int rank, int nranks, rocshmem_uniqueid_t *uid, rocshmem_init_attr_t *attr)

  :param rank:   Rank of the calling process.
  :param nranks: Number of PEs.
  :param uid:    Unique ID used to identify the group processes.
  :param attr:   Attribute structure to be passed to ``rocshmem_init_attr_t``.

  :returns:      Returns ``0`` on success; otherwise, returns a nonzero value.

**Description:**
This routine initializes the ``rocshmem_init_attr_t`` struct.

ROCSHMEM_N_PES
--------------

.. cpp:function:: __host__ int rocshmem_n_pes(void)

  :Parameters: None.
  :returns: Total number of PEs.

**Description:**
This routine queries the total number of PEs.
It can be called before ``rocshmem_init``.

.. cpp:function:: __device__ int rocshmem_n_pes(void)
.. cpp:function:: __device__ int rocshmem_ctx_n_pes(rocshmem_ctx_t ctx)

  :param ctx: GPU side context handle.
  :returns: Total number of PEs.

**Description:**
This routine queries the total number of PEs for a given context.
It can be called per thread with no performance penalty.

ROCSHMEM_MY_PE
--------------

.. cpp:function:: __host__ int rocshmem_my_pe(void)

  :Parameters: None.
  :returns: PE ID of the caller.

**Description:**
This routine queries the PE ID of the caller.
It can be called before ``rocshmem_init``.

.. cpp:function:: __device__ int rocshmem_my_pe(void)
.. cpp:function:: __device__ int rocshmem_ctx_my_pe(rocshmem_ctx_t ctx)

  :param ctx: GPU side context handle.
  :returns: PE ID of the caller.

**Description:**
This routine queries the PE ID of the caller.
It can be called per thread with no performance penalty.

ROCSHMEM_PTR
--------------

.. cpp:function:: __host__ void* rocshmem_ptr(const void *dest, int pe);
.. cpp:function:: __device__ void* rocshmem_ptr(const void *dest, int pe);

  :param dest: Local symmetric heap allocation pointer for current PE.
  :param pe:   Remote PE.
  :returns:    Returns remote symmetric heap device pointer from host-side API.
               ``NULL`` is returned if a valid device pointer cannot be provided.
               This pointer can be used to issue load/store from custom kernels
               instead of using rocshmem device side get/put APIs for RMA operations.

**Description:**
This routine queries rocSHMEM remote symmetric heap pointer.
