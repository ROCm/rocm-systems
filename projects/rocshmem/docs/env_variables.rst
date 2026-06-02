.. meta::
  :description: rocSHMEM environment variables reference
  :keywords: rocSHMEM, ROCm, API, environment variables, environment, reference

.. _rocshmem-api-env-variables:

********************************************************************
rocSHMEM environment variables
********************************************************************

This section describes the important environment variables used to
control the behavior of rocSHMEM.

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``ROCSHMEM_DEBUG_LEVEL``
        | Debug output level
      - ``ERROR``
      - | Levels (from least to most verbose):
        | ``NONE``: Suppress all output.
        | ``ERROR``: Print error messages only (default).
        | ``WARN``: Print warnings and errors.
        | ``ENV``: Print modified environment variables at startup.
        | ``VERSION``: Print build/version information at startup.
        | ``INFO``: Print informational messages and above.
        | ``API``: Print API call tracing (requires ``BUILD_DEBUG_TRACE_HOST``/``BUILD_DEBUG_TRACE_DEVICE``).
        | ``TRACE``: Print all messages including internal traces (requires ``BUILD_DEBUG_TRACE_HOST``/``BUILD_DEBUG_TRACE_DEVICE``).
        |
        | Modifiers can be appended with ``:`` to suppress specific categories:
        | ``:noerror``, ``:nowarn``, ``:noenv``, ``:noversion``, ``:noinfo``, ``:noapi``, ``:notrace``
        | ``:full`` or ``:all`` after ``env`` or ``:env`` modifier controls env print detail.
        | ``:color`` (default) or ``:nocolor`` enables/disables ANSI color output.
        |
        | Examples: ``trace:noversion``, ``env:full``, ``api:noenv``, ``trace:nocolor``

    * - | ``ROCSHMEM_HEAP_SIZE``
        | Defines the size of the rocSHMEM symmetric heap in bytes (per PE).
      - ``1073741824`` (1 GB)
      - | Size in bytes (per PE).
        | Note: the heap is on GPU memory.

    * - | ``ROCSHMEM_MAX_NUM_HOST_CONTEXTS``
        | Maximum number of host-side communication contexts
      - ``1``
      - Maximum number of host-side contexts.

    * - | ``ROCSHMEM_MAX_NUM_CONTEXTS``
        | Defines the number of contexts an application can use.
      - ``32``
      - Maximum number of contexts.

    * - | ``ROCSHMEM_MAX_NUM_TEAMS``
        | Defines the number of teams an application can use.
      - ``40``
      - Maximum number of teams.

    * - | ``ROCSHMEM_BACKEND``
        | When rocSHMEM is compiled for all backends, this environment variable
        | selects which backend to execute. The default value is an empty string and rocSHMEM auto-selects the most appropriate backend.
      - `` ``
      - | ``ipc``: IPC Backend
        | ``ro``: Reverse Offload Backend
        | ``gda``: GPU Direct Async Backend

    * - | ``ROCSHMEM_UNIQUEID_WITH_MPI``
        | Defines whether rocSHMEM is expected to use MPI internally when using the uniqueId based initialization.
      - ``0``
      - | ``0``: Do not use MPI.
        | ``1``: Use MPI.

    * - | ``ROCSHMEM_DISABLE_MIXED_IPC``
        | Defines whether to force using the network conduit even when IPC is available.
      - ``0``
      - | ``0``: Use IPC when available.
        | ``1``: Force network conduit.

    * - | ``ROCSHMEM_USE_IB_HCA``
        | Forces the NIC that this PE uses. When this value is set NIC auto-detection and mapping is disabled, the NIC specified in the variable
        | will be selected. The default value is an empty string and rocSHMEM auto-detects the most appropriate NIC.
      - `` ``
      - | Example value: ``bnxt_re0``

    * - | ``ROCSHMEM_HCA_LIST``
        | Comma separated list of NIC names that can be used by rocSHMEM. Unlike ``ROCSHMEM_USE_IB_HCA``, when this variable is set,
        | NIC auto-detection and mapping still executes, but NICs that are not in the list are discarded before auto-detection runs.
        | Prefixing the list with ``^`` turns the list in an *exclude* list, NICs that are in the list are discarded before auto-detection runs.
        | The default value is an empty string and rocSHMEM auto-detects the most appropriate NIC.
      - `` ``
      - | Example value: ``bnxt_re1,bnxt_re11``, ``^mlx5_0,mlx5_3``

    * - | ``ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME``
        | Chooses the interface to bootstrap rocSHMEM with.
        | Only valid when not using MPI.
        | The default value is an empty string and rocSHMEM auto-detects the most appropriate interface.
      - `` ``
      - | Example value: ``eno8303``

    * - | ``ROCSHMEM_GDA_PROVIDER``
        | When rocSHMEM is compiled with support for multiple NIC vendors,
        | the environment variable selects the desired provider.
        | The default value is an empty string and rocSHMEM auto-detects the most appropriate NIC.
      - `` ``
      - | ``bnxt``: Broadcom Thor 2
        | ``pensando``: AMD Pensando Pollara
        | ``ionic``: AMD Pensando Pollara (alias)
        | ``mlx5``: Mellanox ConnectX-7

    * - | ``ROCSHMEM_GDA_ALTERNATE_QP_PORTS``
        | Enables or disables alternating QP mappings across rocSHMEM contexts.
      - ``1``
      - | ``0``: Disabled.
        | ``1``: Enabled. This helps saturate bandwidth on multiport bonded interfaces.

    * - | ``ROCSHMEM_GDA_TRAFFIC_CLASS``
        | When using an NIC with an Ethernet link layer, this sets the traffic class for the QPs.
      - ``0``
      - The traffic class number.

    * - | ``ROCSHMEM_GDA_PCIE_RELAXED_ORDERING``
        | Enables PCIe Relaxed Ordering when registering the symmetric heap with the RDMA NICs.
      - ``0``
      - | ``0``: Disabled.
        | ``1``: Enabled.

    * - | ``ROCSHMEM_GDA_ENABLE_DMABUF``
        | Enable dmabuf support for memory registration.
      - ``0``
      - | ``0``: Disabled.
        | ``1``: Enabled.

    * - | ``ROCSHMEM_GDA_ALLTOALLV_WG_ALGO``
        | Selects between two algorithms to use for GDA based alltoallv.
        | The GET algorithm uses an initial round of alltoallv
        | communication to distribute displacements then a second round to
        | get transfer data. This algorithm has a higher latency but
        | has better performance for large messages.
        | The COPY algorithm does an alltoallv communication
        | pattern into a staging buffer then does a copy into the destination
        | buffers. This reduces latency but requires more memory, this
        | algorithm only works for small messages.
      - ``GET``
      - | ``GET``: GET-based alltoallv algorithm
        | ``COPY``: Copy alltoallv algorithm

    * - | ``ROCSHMEM_GDA_OVERRIDE_NIC_FIRMWARE_CHECK``
        | This environment variable should be used with caution.
        | It overrides the NIC firmware check if
        | a user wants to use an unsupported NIC firmware.
        | If the firmware check is disabled rocSHMEM is not guaranteed to work.
      - ``0``
      - | ``0``: Disabled.
        | ``1``: Enabled.

    * - | ``ROCSHMEM_GDA_SQ_SIZE``
        | This environment variable sets the length of the SQ for GDA.
      - ``1024``
      - | Maximum number of Work Queue Entries (WQEs) posted on the Send Queue (SQ)

    * - | ``ROCSHMEM_GDA_NUM_QPS_PER_PE_DEFAULT_CTX``
        | Sets the number of Queue Pairs (QPs) to create per PE for the default context.
      - ``1``
      - Number of QPs per PE for the default context.

    * - | ``ROCSHMEM_GDA_NUM_QPS_PER_PE_USR_CTX``
        | Sets the number of Queue Pairs (QPs) to create per PE for each user context.
      - ``1``
      - Number of QPs per PE for each user context.


    * - | ``ROCSHMEM_GDA_NUM_USER_BUFFERS``
        | GDA supports ``rocshmem_buffer_register`` and ``rocshmem_buffer_unregister``
        | for user buffers. This variable sets the number of user buffers an
        | application may register when using the GDA backend.
        | If the application uses more user buffers than what is defined with
        | this variable, then the behavior is undefined.
      - ``4``
      - Maximum number of user buffer registrations for GDA

    * - | ``ROCSHMEM_MAX_WF_BUFFERS``
        | Maximum number of wavefront buffer arrays in default context (determines size of status, return, and atomic return buffers)
      - ``1024``
      -

    * - | ``ROCSHMEM_BOOTSTRAP_TIMEOUT``
        | Bootstrap initialization timeout in seconds
      - ``5``
      -

    * - | ``ROCSHMEM_BOOTSTRAP_HOSTID``
        | Override host identifier for bootstrap. Empty string uses hostname.
      - `` ``
      -

    * - | ``ROCSHMEM_BOOTSTRAP_SOCKET_FAMILY``
        | Socket family for bootstrap (AF_UNSPEC, AF_INET, AF_INET6)
      - ``types::socket_family::UNSPEC``
      -

    * - | ``ROCSHMEM_SDMA_ENABLED``
        | Enable or disable the SDMA transport at runtime (requires ``USE_SDMA`` build option).
      - ``1``
      - | ``0``: Disabled. All transfers use GPU load/store (IPC path).
        | ``1``: Enabled. Transfers at or above ``ROCSHMEM_SDMA_THRESHOLD`` use the SDMA engine.

    * - | ``ROCSHMEM_SDMA_THRESHOLD``
        | Minimum transfer size in bytes to route through the SDMA engine.
        | Transfers smaller than this threshold use GPU load/store instead.
      - ``256``
      - Size in bytes.

    * - | ``ROCSHMEM_SDMA_NUM_CHANNELS``
        | Number of SDMA channels (ring buffers) allocated per destination PE.
        | More channels reduce CAS contention when many wavefronts submit concurrently,
        | at the cost of additional queue memory and SDMA engine resources.
      - ``1``
      - Number of channels per destination PE.

    * - | ``ROCSHMEM_SDMA_SPREAD_CHANNELS``
        | When enabled, each wavefront within a workgroup selects its SDMA channel
        | using an offset based on its wavefront index:
        | ``effective_channel = (ctx_channel + wf_id) % num_channels``.
        | This reduces CAS contention when multiple wavefronts in the same workgroup
        | target the same destination PE on a shared context.
        | By default, spreading is automatically enabled only for the default context
        | (``ctx_id=0``), which is shared by all workgroups when contexts are not
        | created per-workgroup. Per-workgroup contexts (created via
        | ``rocshmem_wg_ctx_create``) already distribute workgroups across channels
        | via their context ID; enabling spreading for them reshuffles contention
        | without reducing it.
      - ``0``
      - | ``0``: Apply wf_id spreading only for the default (shared) context.
        | ``1``: Apply wf_id spreading for all contexts.
