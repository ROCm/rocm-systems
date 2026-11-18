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

ROCSHMEM_HIPMODULE_INIT
-----------------------

.. cpp:function:: __host__ int rocshmem_hipmodule_init(hipModule_t module, hipStream_t stream = nullptr)

  :param module: HIP module containing rocSHMEM device code.
  :param stream: HIP stream on which to enqueue the initialization. If
                 ``nullptr``, ``hipStreamPerThread`` is used.
  :returns:      ``0`` on success, non-zero on failure.

**Description:**
Initializes the rocSHMEM device-side global state for a specific HIP module.

.. note::

  This routine does not synchronize ``stream``. The caller is responsible for
  synchronization if the context must be visible before the next host-side
  step.

This routine replaces the deprecated ``rocshmem_get_device_ctx``; see
:ref:`rocshmem-api-ctx`.

ROCSHMEM_QUERY_THREAD
---------------------

.. cpp:function:: __host__ __device__ void rocshmem_query_thread(int *provided)

   :param provided: Receives the thread mode the runtime is operating in, one
                    of the ``rocshmem_thread_ops`` values listed in
                    :ref:`rocshmem-api-library-constants`.
   :returns:        None.

**Description:**
Queries the thread support level that the runtime selected. The level the
runtime provides may be lower than the level an application requested.

.. WARNING::
   Routines ``rocshmem_init_thread`` and ``rocshmem_wg_init_thread`` have been
   deprecated. Use ``rocshmem_init``/``rocshmem_init_attr`` to initialize and
   ``rocshmem_query_thread`` to query the resulting thread mode.

.. cpp:function:: [[deprecated]] __host__ int rocshmem_init_thread(int requested, int *provided, MPI_Comm comm)
.. cpp:function:: [[deprecated]] __device__ void rocshmem_wg_init_thread(int requested, int *provided)

  :param requested: Requested thread mode from ``rocshmem_thread_ops``.
  :param provided:  Thread mode selected by the runtime.
  :param comm:      MPI communicator for rocSHMEM to use. If
                    ``MPI_COMM_NULL``, ``MPI_COMM_WORLD`` is used. The host
                    overload is only declared in builds with external MPI.
  :returns:         ``0`` on success, non-zero otherwise (host overload).

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

.. note::

  When using the VMM POSIX memory allocator (``ROCSHMEM_HEAP_ALLOCATOR_TYPE=vmm_posix``),
  you must use ``ROCSHMEM_INIT_WITH_UNIQUEID`` initialization.
  MPI-based initialization (``ROCSHMEM_INIT_WITH_MPI_COMM``) is not compatible
  with the VMM POSIX allocator and will result in a runtime error.

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

.. cpp:function:: __host__ __device__ int rocshmem_n_pes(void)

  :Parameters: None.
  :returns: Total number of PEs.

**Description:**
This routine queries the total number of PEs.
The host overload can be called before ``rocshmem_init``. The device overload
can be called per thread with no performance penalty.

.. cpp:function:: __device__ int rocshmem_ctx_n_pes(rocshmem_ctx_t ctx)

  :param ctx: GPU side context handle.
  :returns: Total number of PEs.

**Description:**
This routine queries the total number of PEs for a given context.
It can be called per thread with no performance penalty.

ROCSHMEM_MY_PE
--------------

.. cpp:function:: __host__ __device__ int rocshmem_my_pe(void)

  :Parameters: None.
  :returns: PE ID of the caller.

**Description:**
This routine queries the PE ID of the caller.
The host overload can be called before ``rocshmem_init``. The device overload
can be called per thread with no performance penalty.

.. cpp:function:: __device__ int rocshmem_ctx_my_pe(rocshmem_ctx_t ctx)

  :param ctx: GPU side context handle.
  :returns: PE ID of the caller.

**Description:**
This routine queries the PE ID of the caller.
It can be called per thread with no performance penalty.

ROCSHMEM_INFO_GET_VERSION
-------------------------

.. cpp:function:: __host__ void rocshmem_info_get_version(int *major, int *minor)

  :param major: Returns ``ROCSHMEM_MAJOR_VERSION``.
  :param minor: Returns ``ROCSHMEM_MINOR_VERSION``.
  :returns: None.

**Description:**
This routine returns the major and minor version of the OpenSHMEM
specification that this library implements.  Upon return, ``major``
and ``minor`` contain ``ROCSHMEM_MAJOR_VERSION`` and
``ROCSHMEM_MINOR_VERSION``, respectively.

ROCSHMEM_INFO_GET_NAME
----------------------

.. cpp:function:: __host__ void rocshmem_info_get_name(char *name)

  :param name: Buffer of at least ``ROCSHMEM_MAX_NAME_LEN`` bytes.
               Receives ``ROCSHMEM_VENDOR_STRING``.
  :returns: None.

**Description:**
This routine returns the vendor-defined string associated with this
library.  The string is copied into the ``name`` argument, which must
be able to hold at least ``ROCSHMEM_MAX_NAME_LEN`` bytes.  The string
is null-terminated and contains the value of ``ROCSHMEM_VENDOR_STRING``.

ROCSHMEM_VENDOR_GET_VERSION_INFO
--------------------------------

.. cpp:function:: __host__ void rocshmem_vendor_get_version_info(int *major, int *minor, int *patch)

  :param major: Returns ``ROCSHMEM_VENDOR_MAJOR_VERSION``.
  :param minor: Returns ``ROCSHMEM_VENDOR_MINOR_VERSION``.
  :param patch: Returns ``ROCSHMEM_VENDOR_PATCH_VERSION``.
  :returns: None.

**Description:**
This routine returns the vendor-specific version of rocSHMEM.  Upon
return, ``major``, ``minor``, and ``patch`` contain
``ROCSHMEM_VENDOR_MAJOR_VERSION``, ``ROCSHMEM_VENDOR_MINOR_VERSION``,
and ``ROCSHMEM_VENDOR_PATCH_VERSION``, respectively.

ROCSHMEM_PTR
--------------

.. cpp:function:: __host__ __device__ void* rocshmem_ptr(const void *dest, int pe);

   :param dest: Local symmetric heap allocation pointer for current PE.
   :param pe:   Remote PE.
   :returns:    Returns remote symmetric heap device pointer from host-side API.
                ``NULL`` is returned if a valid device pointer cannot be provided.
                This pointer can be used to issue load/store from custom kernels
                instead of using rocshmem device side get/put APIs for RMA operations.

**Description:**
This routine queries rocSHMEM remote symmetric heap pointer.

ROCSHMEM_QUERY_BACKEND_TYPE
---------------------------

.. cpp:function:: BackendType rocshmem_query_backend_type()

  :Parameters: None.
  :returns:    The backend the runtime selected for this job: one of
               ``BackendType::IPC_BACKEND``, ``BackendType::RO_BACKEND``, or
               ``BackendType::GDA_BACKEND``.

**Description:**
Returns the backend that the runtime chose at initialization. Because a single
build can contain several backends, and because selection depends on the job
layout and on environment variables such as ``ROCSHMEM_BACKEND``, this routine
is the authoritative way for an application to discover what it is actually
running on.

ROCSHMEM_GLOBAL_EXIT
--------------------

.. cpp:function:: __host__ void rocshmem_global_exit(int status)

  :param status: Exit status passed to MPI when a backend communicator exists;
                 ignored by the non-MPI fallback.
  :returns:      None.

**Description:**
Allows any PE to force termination of the job. When a backend communicator
exists, this terminates the job through ``MPI_Abort``. Under unique-ID
initialization without MPI, the IPC and GDA backends instead call ``abort()``
on the calling process only and ignore ``status``; terminating the remaining
PEs then depends on the launcher. The routine does not return.

ROCSHMEM_DUMP_STATS
-------------------

.. cpp:function:: __host__ void rocshmem_dump_stats()

  :Parameters: None.
  :returns:    None.

**Description:**
Dumps the runtime's internal statistics counters to ``stdout``. Counters are
only collected when rocSHMEM is built with ``-DPROFILE=ON``; in other builds
this routine produces no useful output. See :ref:`build-options-reference`.

ROCSHMEM_RESET_STATS
--------------------

.. cpp:function:: __host__ void rocshmem_reset_stats()

  :Parameters: None.
  :returns:    None.

**Description:**
Resets the counters supported by the selected backend. Counters on user-created
host contexts are not cleared, and the RO backend resets its profiler rather
than the host and device context counters that ``rocshmem_dump_stats()``
accumulates. A subsequent dump can therefore still include activity from before
this call; do not rely on it to exclude all warm-up activity.
