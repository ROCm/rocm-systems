.. meta::
   :description: Reference for all RCCL environment variables, grouped by function: configuration, logging, algorithm control, network, and testing.
   :keywords: RCCL, ROCm, environment variables, NCCL_DEBUG, NCCL_ALGO, NCCL_PROTO, NCCL_IB_HCA, NCCL_CUMEM_ENABLE, NCCL_LAUNCH_ORDER_IMPLICIT, RCCL_IB_SPLIT_DATA_THRESHOLD, configuration, tuning

.. _env-variables:

**************************
RCCL environment variables
**************************

This section describes the most important RCCL environment variables,
which are grouped by functionality.

Configuration and setup
=======================

The configuration and setup environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 38 22 40

    * - **Environment variable**
      - **Default value**
      - **Values**

    * - | ``NCCL_CONF_FILE``
        | Specifies the path to the RCCL configuration file.
      - | Unset
      - | String path to a configuration file.
        | Default: ``~/.rccl.conf`` if present, otherwise ``/etc/rccl.conf``.

    * - | ``NCCL_IBVERBS_LIB``
        | Specifies the libibverbs shared object that RCCL loads at runtime for
          the InfiniBand/RoCE (IB verbs) transport. Use it when rdma-core is
          installed in a non-default prefix, such as inside a container or an
          HPC software module, where the loader cannot find the library by its
          default name. When the override is unset or fails to load, RCCL falls
          back to ``libibverbs.so`` and then ``libibverbs.so.1``. ``NCCL_LIBIBVERBS_SO``
          is accepted as an alias and is used when ``NCCL_IBVERBS_LIB`` is unset.
      - | String path or soname of the libibverbs shared object
        | Default: unset (loads ``libibverbs.so`` or ``libibverbs.so.1``)

    * - | ``NCCL_HOSTID``
        | Sets the host identifier for multi-node communication.
      - | Unset
      - | String value for host identification.
        | Used for host hash generation.

    * - | ``NCCL_BOOTSTRAP_BIDIR_ALLGATHER``
        | Enables the bidirectional ring AllGather (N/2 steps) on the socket OOB path
          during bootstrap. The unidirectional ring (N-1 steps) is kept as a fallback.
          Has no effect when net OOB is in use.
      - | ``1``
      - | ``0``: Force unidirectional ring.
        | ``1``: Force bidirectional ring.

    * - | ``NCCL_CUMEM_ENABLE``
        | Enables cuMem virtual memory management (VMM) for RCCL allocations,
          which is required for ``ncclCommSuspend`` and ``ncclCommResume`` to
          release the physical GPU memory of a suspended communicator. See
          :ref:`suspend-resume` for the full prerequisites.
      - | ``0``: Disabled.
        | ``1``: Enabled on any architecture.
        | ``-2``: Auto-detect (default); enable when the platform supports VMM.
          Auto-detect is limited to gfx1250, the only architecture where the VMM
          path is validated. Use ``1`` to force it on elsewhere.

    * - | ``NCCL_MIN_CTAS``
        | Minimum number of CTAs (channels) used for a collective. Overrides
          the ``minCTAs`` field of ``ncclConfig_t``.
      - | Unset
      - | Positive integer. Values ``<= 0`` are ignored.

    * - | ``NCCL_MAX_CTAS``
        | Maximum number of CTAs (channels) used for a collective. Overrides
          the ``maxCTAs`` field of ``ncclConfig_t``.
      - | Unset
      - | Positive integer. Values ``<= 0`` are ignored.

Logging and debugging
=====================

The logging and debugging environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 38 22 40

    * - **Environment variable**
      - **Default value**
      - **Values**

    * - | ``NCCL_DEBUG``
        | Controls debug logging in RCCL for troubleshooting and monitoring collective communication operations.
      - | ``ERROR``
      - | Each logging level includes all output from levels below it.
        |
        | ``NONE``: No logging is printed.
        | ``ERROR``: Fatal conditions that prevent the operation from continuing.
        | ``VERSION``: Prints ``librccl`` version info during initialization.
        | ``WARN``: Unusual conditions that could lead to unexpected results.
        | ``INFO``: Standard status and operation messages.
        | ``ABORT``: Unused.
        | ``TRACE``: Trace-level logging of function calls and parameters.
          Only active when ``librccl`` is built with ``ENABLE_TRACE``.

    * - | ``NCCL_DEBUG_SUBSYS``
        | Controls which subsystems generate debug output.
      - | ``INIT,BOOTSTRAP,ENV``
      - | Comma-separated list of subsystem names. Prefix a name with ``^`` to
          exclude it.
        |
        | ``INIT``: Initialization phase.
        | ``COLL``: Collective execution.
        | ``P2P``: Peer-to-peer setup or communication.
        | ``SHM``: Shared memory.
        | ``NET``: Network setup or communication.
        | ``GRAPH``: Topology parsing.
        | ``TUNING``: Tuner plugin.
        | ``ENV``: Environment variables.
        | ``ALLOC``: Memory allocation.
        | ``CALL``: Function calls (``TRACE`` only).
        | ``PROXY``: Proxy thread.
        | ``NVLS``: Not valid for AMD or RCCL.
        | ``BOOTSTRAP``: Bootstrapping phase of initialization.
        | ``REG``: Registration and deregistration of transport initialization.
        | ``PROFILE``: Profiling and timing info.
        | ``RAS``: RAS-related logs.
        | ``VERBS``: InfiniBand Verbs.
        | ``DESTROY``: Communicator and plugin teardown.
        | ``ALL``: All logging subsystems.

    * - | ``NCCL_WARN_ENABLE_DEBUG_INFO``
        | Converts all ``WARN`` level logs to ``INFO`` level logs.
      - | ``0``
      - | ``0``: Disabled.
        | ``1``: Enabled.

    * - | ``NCCL_DEBUG_TIMESTAMP_LEVELS``
        | The debug levels that include a timestamp prefix in their log output.
      - | ``WARN``
      - | Comma-separated list of ``NCCL_DEBUG`` level names. Prefix a name
          with ``^`` to exclude it.

    * - | ``NCCL_DEBUG_TIMESTAMP_FORMAT``
        | The timestamp format for ``NCCL_DEBUG`` output.
      - | ``"[%F %T] "``
      - | Format string in ``printf`` style.

    * - | ``NCCL_DEBUG_FILE``
        | Write logs to a file rather than ``stdout``.
      - | Unset
      - | Filename string. Supports format specifiers:
          ``%h`` for hostname, ``%p`` for PID, ``%%`` to escape ``%``.
        | Use ``%p`` to write separate files per process and avoid mixed output.
        | Example: ``NCCL_DEBUG_FILE=debugfile.%h.%p``

Algorithm and protocol control
==============================

The algorithm and protocol control environment variables for RCCL are
collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 38 22 40

    * - **Environment variable**
      - **Default value**
      - **Values**

    * - | ``NCCL_ALGO``
        | Restricts automatic algorithm selection to the specified set.
          Re-runs the cost model over the enabled algorithms.
      - | Unset
      - | Algorithm name string: ``Ring``, ``Tree``, ``CollNet``.
        | Multiple values accepted as a comma-separated list.

    * - | ``NCCL_PROTO``
        | Restricts automatic protocol selection to the specified set.
          Re-runs the cost model over the enabled protocols.
      - | Unset
      - | Protocol name string: ``LL``, ``LL128``, ``Simple``.
        | Multiple values accepted as a comma-separated list.

Network and topology
====================

The network and topology environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 38 22 40

    * - **Environment variable**
      - **Default value**
      - **Values**

    * - | ``NCCL_IB_HCA``
        | Specifies which InfiniBand device:port to use.
      - | Unset
      - | Device specification string.
        | Prefix with ``^`` for exclusion, ``=`` for exact match.

    * - | ``NCCL_IB_GID_INDEX``
        | Defines the Global ID index used in RoCE mode.
      - | ``-1``
      - | Integer value. See the InfiniBand ``show_gids`` command for valid values.

    * - | ``NCCL_PXN_C2C``
        | Allows PXN routing through a C2C link to reach a NIC attached to a
          peer GPU. The C2C path is NVIDIA-specific and is not currently
          applicable on AMD hardware.
      - | ``0``
      - | ``0``: Disabled.
        | ``1``: Enabled.

    * - | ``NCCL_SOCKET_IFNAME``
        | Specifies which IP interfaces to use for communication.
      - | Unset
      - | Interface prefix string or comma-separated list.
        | Prefix with ``^`` for exclusion, ``=`` for exact match.
        | Example: ``eth`` (all eth interfaces), ``=eth0`` (exact match).

    * - | ``NCCL_SOCKET_FAMILY``
        | Forces IPv4 or IPv6 interface selection.
      - | Unset
      - | ``AF_INET``: Force IPv4.
        | ``AF_INET6``: Force IPv6.
        | Unset: Use first available.

    * - | ``NCCL_IGNORE_NET_MISMATCH``
        | Controls what happens when ranks report a different number of local
          network (NET) devices during communicator initialization. RCCL gathers
          each rank's local NET device count and compares the minimum and maximum
          across the communicator. A mismatch usually means the job was launched
          with an inconsistent NIC selection (for example, an uneven
          ``NCCL_SOCKET_IFNAME`` or ``NCCL_IB_HCA`` per rank, or nodes with
          different NIC counts), which otherwise surfaces later as obscure
          transport failures. See :ref:`heterogeneous-nic-counts`.
      - | ``1``
      - | ``1``: Detect and continue, logging the mismatch at ``INFO`` level.
        | ``0``: Fail initialization with ``ncclSystemError`` and a warning on
          the mismatch.

    * - | ``NCCL_IGNORE_COLLNET_MISMATCH``
        | Same as ``NCCL_IGNORE_NET_MISMATCH`` but for the number of local CollNet
          devices reported by each rank.
      - | ``0``
      - | ``0``: Fail initialization with ``ncclSystemError`` and a warning on
          the mismatch.
        | ``1``: Detect and continue, logging the mismatch at ``INFO`` level.

    * - | ``NCCL_IB_MERGE_NICS``
        | Enables RCCL to combine several physical IB NICs that are close to the
          same GPU into a single logical network device (NIC Fusion). This allows
          RCCL to aggregate the bandwidth of those NICs. Use
          ``NCCL_NET_MERGE_LEVEL`` and ``NCCL_NET_FORCE_MERGE`` to control which
          NICs are combined.
      - | ``1``: Enabled (default).
        | ``0``: Disabled.
        | On AINIC with the ``IB-CAST`` transport, merging is off unless this
          variable is explicitly set to ``1``.

    * - | ``NCCL_NET_MERGE_LEVEL``
        | Sets the maximum topological distance between two NICs that can be
          merged into a single logical device. NICs farther apart than this level
          are left separate.
      - | ``LOC``: Same device only, which disables merging.
        | ``PORT``: Two ports of the same NIC (default).
        | ``PIX``: Under the same PCIe switch.
        | ``PXB``: Multiple PCIe bridges, without crossing the PCIe host bridge.
        | ``P2C``, ``PXN``: Accepted, with the same effect as ``PXB`` for NIC pairs.
        | ``PHB``: Under the same CPU socket.
        | ``SYS``: Anywhere in the node, including across NUMA nodes.
        | The value is a string, so ``PATH_PORT`` is not valid. An unrecognized
          value falls back to ``LOC`` and disables merging.

    * - | ``NCCL_NET_FORCE_MERGE``
        | Merges the listed groups of NICs regardless of
          ``NCCL_NET_MERGE_LEVEL``. NICs that are not listed are then merged
          automatically.
      - | Semicolon-separated list of groups, each a comma-separated list of
          device names in ``NCCL_IB_HCA`` notation.
        | Default: unset.

    * - | ``NCCL_NETDEVS_POLICY``
        | Controls how many of a GPU's locally reachable NICs are used on the
          network path for ``send``, ``recv``, and ``all-to-all``. The policy
          governs per-channel NIC selection (``ncclTopoGetLocalNet``); the
          per-peer network channel count is still bounded by available NIC
          bandwidth. Any unset, malformed, or out-of-range value falls back to
          ``AUTO``.
      - | ``AUTO``
      - | ``AUTO``: Use ``ceil(localNetCount / localGpuCount)`` NICs, dividing
          the local NICs across the GPUs that share them.
        | ``ALL``: Use every locally reachable NIC.
        | ``MAX:N``: Use at most ``N`` NICs (clamped to the number reachable);
          ``N`` must be a positive integer.

    * - | ``RCCL_IB_SPLIT_DATA_THRESHOLD``
        | Minimum message size (in bytes) before the payload is split across
          multiple NICs or QPs. Smaller messages use one QP for data to reduce
          latency. This variable can be used when NIC Fusion
          (``NCCL_NET_MERGE_LEVEL``) or data splitting on QPs
          (``NCCL_IB_SPLIT_DATA_ON_QPS``) is enabled.
      - | ``128``
      - | Integer value in bytes.
        | ``N``: Split only when message size >= N bytes.

    * - | ``NCCL_NCHANNELS_PER_NET_PEER``
        | Sets the number of channels used per network (remote) peer.
        | This overrides the value of the ``nChannelsPerNetPeer`` field in
        | ``ncclConfig_t``. When neither this variable nor the config field is
        | set, RCCL auto-tunes the per-peer channel count based on the
        | available NIC bandwidth and rank count.
      - | Integer value, ``1`` to ``MAXCHANNELS`` (default: unset/auto-tuned)
        | Values ``<= 0`` are ignored and a warning is logged.
        | Values ``> MAXCHANNELS`` set through ``ncclConfig_t`` are rejected
        | with ``ncclInvalidArgument`` at communicator initialization.

    * - | ``NCCL_RINGS``
        | Defines custom ring topology, overriding automatic topology detection.
      - | Unset
      - | Ring topology specification string.

    * - | ``RCCL_TREES``
        | Defines custom tree topology.
      - | Unset
      - | Tree topology specification string.

    * - | ``NCCL_RINGS_REMAP``
        | Controls ring remapping for specific topologies.
      - | Unset
      - | Remapping specification string. Used with Rome 4P2H topology.

Development and testing (advanced)
==================================

The development and testing environment variables for RCCL are
collected in the following table. These variables are primarily
intended for debugging and development purposes.

.. list-table::
    :header-rows: 1
    :widths: 38 22 40

    * - **Environment variable**
      - **Default value**
      - **Values**

    * - | ``CUDA_LAUNCH_BLOCKING``
        | Controls CUDA kernel launch blocking behavior.
      - | ``0``
      - | ``0``: Non-blocking launches.
        | ``1`` or any non-zero value: Blocking launches.

    * - | ``NCCL_COMM_ID``
        | Enables multi-process mode in test applications.
      - | Unset
      - | Any non-empty value enables multi-process mode.
        | Used with test executables for distributed testing.

    * - | ``NCCL_DISABLE_MEM_MANAGER``
        | Disables the internal RCCL memory manager. This is an internal
          parameter intended for testing and debugging only. When the memory
          manager is disabled, ``ncclCommSuspend``, ``ncclCommResume``, and
          ``ncclCommMemStats`` return ``ncclInvalidUsage``.
      - | ``0``
      - | ``0``: Memory manager enabled.
        | ``1``: Memory manager disabled.

    * - | ``NCCL_NO_CACHE``
        | Disables caching for selected RCCL environment parameters so their
          values are re-read from the environment on each access. By default,
          RCCL caches parameter values after the first read for performance.
          This variable is intended for testing and debugging when parameters
          need to be changed without restarting the process. The value is
          parsed once on first use, so it must be set before RCCL reads any
          parameters. ``NCCL_NO_CACHE`` itself is always cached and cannot
          be listed.
      - | Unset
      - | Unset: All parameters are cached after first read.
        | Comma-separated list of parameter names (for example,
          ``NCCL_DEBUG,NCCL_ALGO``): Disable caching for those keys only.
        | ``ALL``: Disable caching for every parameter except ``NCCL_NO_CACHE``.

Multi-communicator ordering
===========================

When an application uses multiple RCCL communicators on the same device,
collective operations may execute in an unpredictable order unless the
application adds explicit synchronization between streams.

.. list-table::
    :header-rows: 1
    :widths: 38 22 40

    * - **Environment variable**
      - **Default value**
      - **Values**

    * - | ``NCCL_LAUNCH_ORDER_IMPLICIT``
        | Serializes RCCL operations across different communicators on the
          same device according to their host-side launch sequence. This
          provides deterministic execution order for multi-communicator
          workloads such as chained collectives where one operation's
          output feeds into the next.
      - | ``0``
      - | ``0``: Disabled.
        | ``1``: Enabled. Operations execute in host launch order.
