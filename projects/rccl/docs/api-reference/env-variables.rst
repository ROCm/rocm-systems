.. meta::
   :description: RCCL is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs
   :keywords: RCCL, ROCm, library, API, reference, environment variable, environment

.. _env-variables:

********************************************************************
RCCL environment variables
********************************************************************

This section describes the most important RCCL environment variables,
which are grouped by functionality.

Configuration and setup
========================

The configuration and setup environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_CONF_FILE``
        | Specifies the path to the RCCL configuration file.
      - | String path to configuration file
        | Default: ``~/.rccl.conf`` or ``/etc/rccl.conf``

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
      - | String value for host identification
        | Used for host hash generation

    * - | ``NCCL_BOOTSTRAP_BIDIR_ALLGATHER``
        | Enables the bidirectional ring AllGather (N/2 steps) on the socket OOB path
          during bootstrap. The unidirectional ring (N-1 steps) is kept as a fallback.
          Has no effect when net OOB is in use.
      - | ``0``: Force unidirectional ring.
        | ``1``: Force bidirectional ring (default).

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
      - | Positive integer (values ``<= 0`` are ignored).
        | Default: unset (uses the RCCL default).

    * - | ``NCCL_MAX_CTAS``
        | Maximum number of CTAs (channels) used for a collective. Overrides
          the ``maxCTAs`` field of ``ncclConfig_t``.
      - | Positive integer (values ``<= 0`` are ignored).
        | Default: unset (uses the RCCL default).

    * - | ``NCCL_ENV_PLUGIN``
        | Loads an external environment plugin that intercepts all RCCL parameter
          lookups. See :ref:`using-rccl-env-plugin` for full details.
      - | Path to a plugin ``.so`` file, a bare name expanded to
          ``librccl-env-<name>.so``, or ``none`` to disable.
        | Default: unset (tries ``librccl-env.so``, then reads from the process
          environment).

    * - | ``NCCL_ENV_JSON_FILE``
        | Path to a JSON configuration file used by ``librccl-env-json.so``.
          Has no effect unless ``NCCL_ENV_PLUGIN`` points to that plugin.
      - | Path to a flat JSON file mapping variable names to string values.
          Relative paths are resolved from the application's working directory.
        | Default: unset (falls back to ``getenv()`` for all lookups).

    * - | ``NCCL_ALLGATHERV_ENABLE``
        | Fuses grouped multi-root ``ncclBroadcast`` calls into a single AllGatherV
          ring kernel when two or more distinct roots appear in a group.
      - | ``0``: Disabled (default).
        | ``1``: Enabled.

Logging and debugging
=====================

The logging and debugging environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 35,65

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_DEBUG``
        | Controls debug logging in RCCL for troubleshooting and monitoring collective communication operations. 
      - | These are the logging levels in RCCL set via ``NCCL_DEBUG``. Each logging level contains all logging for levels below it. The default logging level is ``ERROR``.
        |
        | ``NONE``: No logging is printed.
        | ``ERROR``: These messages report when a fatal condition has occurred in RCCL and the operation can't continue.
        | ``VERSION``: ``librccl`` version info is printed during the initialization phase.
        | ``WARN``: Prints warnings about unusual conditions that could lead to unexpected results.
        | ``INFO``: Prints standard logging messages about status and operations performed.
        | ``ABORT``: Unused.
        | ``TRACE``: Prints trace-level logging of function calls and parameters. Only active when ``librccl`` is built using ``ENABLE_TRACE``.

    * - | ``NCCL_DEBUG_SUBSYS``
        | Controls which subsystems generate debug output.
      - | These are the logging subsystems set via ``NCCL_DEBUG_SUBSYS``. These can be set as a comma-separated list, and can be inverted using the ``^`` prefix. The default subsystem set is ``INIT``, ``BOOTSTRAP``, and ``ENV``.
        |
        | ``INIT``: Prints during the initialization phase.
        | ``COLL``: Prints during execution of collectives.
        | ``P2P``: Prints logs related to peer-to-peer setup or communication.
        | ``SHM``: Prints logs related to shared memory.
        | ``NET``: Prints logs related to network setup or communication.
        | ``GRAPH``: Prints logs related to parsing the topology of the network.
        | ``TUNING``: Prints logs related to the tuner plugin.
        | ``ENV``: Prints logs related to environment variables.
        | ``ALLOC``: Prints logs related to memory allocation.
        | ``CALL``: Prints logs for function calls (``TRACE`` only).
        | ``PROXY``: Prints logs related to the proxy thread.
        | ``NVLS``: Not valid for AMD/RCCL.
        | ``BOOTSTRAP``: Prints logs related to the bootstrapping phase of initialization.
        | ``REG``: Prints logs related to registration and deregistration of transport initialization.
        | ``PROFILE``: Prints logs related to the profiling/timing info.
        | ``RAS``: Prints logs related to RAS.
        | ``VERBS``: Prints logs related to IB/Verbs.
        | ``DESTROY``: Prints logs related to communicator/plugin teardown (destroy, abort, revoke, plugin unload).
        | ``ALL``: Activates all logging subsystems.

    * - | ``NCCL_WARN_ENABLE_DEBUG_INFO``
        | Converts all ``WARN`` level logs to ``INFO`` level logs.
      - | ``0``: Default value. Variable is not enabled.
        | ``1``: Enable the variable.

    * - | ``NCCL_DEBUG_TIMESTAMP_LEVELS``
        | The timestamp levels for ``NCCL_DEBUG``.
      - | A set of ``NCCL_DEBUG`` levels can have a timestamp prepended set as a comma-separated list which can be inverted using the ``^`` prefix. The default set is ``WARN``.

    * - | ``NCCL_DEBUG_TIMESTAMP_FORMAT``
        | The timestamp format for ``NCCL_DEBUG``.
      - | Set the format of the timestamp in ``printf`` style. The default format is ``"[%F %T] "``.

    * - | ``NCCL_DEBUG_FILE``
        | Write logs to a file rather than ``stdout``.
      - | The filename can be formatted using ``%h`` for hostname, ``%p`` for pid, and ``%%`` to escape the ``%`` character. It is recommended to use ``%p`` to output to individual files per pid to avoid mixing or potentially overwriting the output. Example usage: ``NCCL_DEBUG_FILE=debugfile.%h.%p``

    * - | ``NCCL_CHECK_MODE``
        | Selects how thoroughly RCCL validates the arguments of every
          collective call. Checking costs latency, so it is disabled by default
          and intended for development and bring-up. See
          :ref:`check-mode` for what each mode detects.
      - | ``DEFAULT``: No argument validation (default).
        | ``DEBUG_LOCAL``: Validate the buffer pointers locally on each rank.
          Replaces the deprecated ``NCCL_CHECK_POINTERS``.
        | ``DEBUG_GLOBAL``: Also validate arguments across ranks, including
          symmetric buffer registration.
        | Values other than ``DEBUG_LOCAL`` and ``DEBUG_GLOBAL`` leave the mode
          unchanged, so writing ``DEFAULT`` does not switch checking off again.

    * - | ``NCCL_CHECK_POINTERS``
        | Deprecated. Enables local validation of the buffer pointers passed to
          each collective.
      - | ``0``: Disabled (default).
        | ``1``: Enabled, equivalent to ``NCCL_CHECK_MODE=DEBUG_LOCAL``.
        | Use ``NCCL_CHECK_MODE`` instead. When both are set, ``DEBUG_LOCAL`` or
          ``DEBUG_GLOBAL`` wins; any other ``NCCL_CHECK_MODE`` value keeps the
          mode selected by ``NCCL_CHECK_POINTERS=1``.

.. _check-mode:

Validating collective arguments
-------------------------------

``NCCL_CHECK_MODE=DEBUG_LOCAL`` inspects only what a rank can see by itself: it
verifies that the ``sendbuff`` and ``recvbuff`` arguments are valid device
pointers that belong to the device the communicator was created on. Passing a
host pointer or a pointer from another device makes the collective return
``ncclInvalidArgument`` instead of faulting inside the kernel.

``NCCL_CHECK_MODE=DEBUG_GLOBAL`` adds cross-rank validation of symmetric buffer
registration. The symmetric kernels require every rank to describe its buffers
identically, because a rank addresses a peer's buffer by applying its own offsets
to the peer's symmetric window. RCCL cannot verify that from a single rank, so at
group launch the ranks exchange the identity of the windows backing their buffers
and compare against rank 0. A collective is rejected with
``ncclInvalidArgument`` when:

* Some ranks pass buffers registered with ``NCCL_WIN_COLL_SYMMETRIC`` while
  others pass unregistered buffers.
* The ranks pass buffers from windows registered at different positions in the
  symmetric address space.
* The ranks pass buffers at different offsets inside their windows.

Each rejection is reported by rank 0 with a ``WARN`` message naming the
collective, the message size, and the first rank that disagrees, so set
``NCCL_DEBUG=WARN`` when using this mode. Setting ``NCCL_DEBUG=INFO`` with
``NCCL_DEBUG_SUBSYS=COLL`` additionally prints a ``SymCheck`` line per rank with
the window and user offsets that were compared.

Without this mode such a mismatch is not diagnosed: RCCL silently falls back to
the general kernels for calls it cannot serve symmetrically, so the collective
still produces correct results but loses the performance of the symmetric path.
Enable ``DEBUG_GLOBAL`` when a workload registers symmetric windows yet does not
reach the expected symmetric performance.

.. note::

   ``DEBUG_GLOBAL`` adds a bootstrap all-gather to every group launch, which is
   far more expensive than the collective itself for small messages. Use it to
   diagnose a configuration, not in production.

Algorithm and protocol control
==============================

The algorithm and protocol control environment variables for RCCL are
collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_ALGO``
        | Forces specific algorithm selection for collectives.
      - | Algorithm name string
        | Used to override automatic algorithm selection

    * - | ``NCCL_PROTO``
        | Forces specific protocol selection for communication.
      - | Protocol name string
        | Used to override automatic protocol selection

Network and topology
====================

The network and topology environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_IB_HCA``
        | Specifies InfiniBand device:port to use.
      - | Device specification string
        | Prefix with ``^`` for exclusion, ``=`` for exact match

    * - | ``NCCL_IB_GID_INDEX``
        | Defines the Global ID index used in RoCE mode.
      - | Integer value (default: ``-1``)
        | See InfiniBand ``show_gids`` command for valid values

    * - | ``NCCL_PXN_C2C``
        | Allows PXN routing through a C2C link to reach a NIC attached to a
          peer GPU. The C2C path is NVIDIA-specific and is not currently
          applicable on AMD hardware.
      - | ``0``: Disabled (default).
        | ``1``: Enabled.

    * - | ``NCCL_SOCKET_IFNAME``
        | Specifies which IP interfaces to use for communication.
        | When unset, RCCL auto-selects an interface in this order:
        | ``ib*`` first; if none is found and ``NCCL_COMM_ID`` is set, an
        | interface on the same subnet as that address; then any interface
        | other than ``docker*``, ``lo`` and ``virbr*``; then ``docker*``;
        | then ``lo``; and finally ``virbr*``. Libvirt bridge interfaces
        | (``virbr*``) are considered last because they serve host-to-VM
        | (virtual machine) traffic and cannot reach a remote node.
      - | Interface prefix string or list
        | Multiple prefixes separated by ``,``
        | Prefix with ``^`` for exclusion, ``=`` for exact match
        | Example: ``eth`` (all eth interfaces), ``=eth0`` (exact match)

    * - | ``NCCL_SOCKET_FAMILY``
        | Forces IPv4/IPv6 interface selection.
      - | ``AF_INET``: Force IPv4
        | ``AF_INET6``: Force IPv6
        | Unset: Use first available

    * - | ``NCCL_IGNORE_NET_MISMATCH``
        | Controls what happens when ranks report a different number of local
          network (NET) devices during communicator initialization. RCCL gathers
          each rank's local NET device count and compares the minimum and maximum
          across the communicator. A mismatch usually means the job was launched
          with an inconsistent NIC selection (for example, an uneven
          ``NCCL_SOCKET_IFNAME``/``NCCL_IB_HCA`` per rank, or nodes with different
          NIC counts), which otherwise surfaces later as obscure transport
          failures. See :ref:`heterogeneous-nic-counts`.
      - | ``1``: Detect and continue, logging the mismatch at ``INFO`` level (default).
        | ``0``: Fail initialization with ``ncclSystemError`` and a warning on the mismatch.

    * - | ``NCCL_IGNORE_COLLNET_MISMATCH``
        | Same as ``NCCL_IGNORE_NET_MISMATCH`` but for the number of local CollNet
          devices reported by each rank.
      - | ``0``: Fail initialization with ``ncclSystemError`` and a warning on the mismatch (default).
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
        | network path for ``send``, ``recv``, and ``all-to-all``. The policy
        | governs per-channel NIC selection (``ncclTopoGetLocalNet``); the
        | per-peer network channel count is still bounded by available NIC
        | bandwidth.
        | Any unset, malformed, or out-of-range value falls back to ``AUTO``.
      - | ``AUTO`` (default): use ``ceil(localNetCount / localGpuCount)`` NICs,
        | dividing the local NICs across the GPUs that share them.
        | ``ALL``: use every locally reachable NIC.
        | ``MAX:N``: use at most ``N`` NICs (clamped to the number reachable);
        | ``N`` must be a positive integer.

    * - | ``RCCL_IB_SPLIT_DATA_THRESHOLD``
        | Minimum message size (in bytes) before the payload is split across
        | multiple NICs/QPs.
        | Smaller messages use one QP for data to reduce latency.
        | This variable can be leveraged when NIC Fusion (``NCCL_NET_MERGE_LEVEL``) and/or data splitting on QPs (``NCCL_IB_SPLIT_DATA_ON_QPS``) is enabled.
      - | Integer value in bytes (default: ``128``)
        | ``N``: Split only when message size >= N bytes

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
        | Defines custom ring topology.
      - | Ring topology specification string
        | Overrides automatic topology detection

    * - | ``RCCL_TREES``
        | Defines custom tree topology.
      - | Tree topology specification string
        | Alternative to ring topology

    * - | ``NCCL_RINGS_REMAP``
        | Controls ring remapping for specific topologies.
      - | Remapping specification string
        | Used with Rome 4P2H topology

Development and testing (advanced)
==================================

The development and testing environment variables for RCCL are
collected in the following table. These variables are primarily
intended for debugging and development purposes.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``CUDA_LAUNCH_BLOCKING``
        | Controls CUDA kernel launch blocking behavior.
      - | ``0``: Non-blocking launches
        | ``1`` or non-zero: Blocking launches

    * - | ``NCCL_COMM_ID``
        | Enables multi-process mode in test applications.
      - | Any non-empty value enables multi-process mode
        | Used with test executables for distributed testing

    * - | ``NCCL_DISABLE_MEM_MANAGER``
        | Disables the internal RCCL memory manager. This is an internal
          parameter intended for testing and debugging only. When the memory
          manager is disabled, ``ncclCommSuspend``, ``ncclCommResume``, and
          ``ncclCommMemStats`` return ``ncclInvalidUsage``.
      - | ``0``: Memory manager enabled (default).
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
      - | Unset (default): all parameters are cached after first read.
        | Comma-separated list of parameter names (for example,
          ``NCCL_DEBUG,NCCL_ALGO``): disable caching for those keys only.
        | ``ALL``: disable caching for every parameter except
          ``NCCL_NO_CACHE``.

Multi-communicator ordering
===========================

When an application uses multiple RCCL communicators on the same device,
collective operations may execute in an unpredictable order unless the
application adds explicit synchronization between streams.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_LAUNCH_ORDER_IMPLICIT``
        | Serializes RCCL operations across different communicators on the
        | same device according to their host-side launch sequence. This
        | provides deterministic execution order for multi-communicator
        | workloads such as chained collectives where one operation's
        | output feeds into the next.
      - | ``0``: Disabled (default).
        | ``1``: Enabled. Operations execute in host launch order.

Inspector profiling
===================

The NCCL Inspector is a profiler plugin that emits per-communicator,
per-operation performance data (collectives and point-to-point) as JSON or
Prometheus textfile metrics. For a full walkthrough, see
:doc:`../how-to/using-rccl-inspector-plugin`. The Inspector environment
variables are collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_INSPECTOR_ENABLE``
        | Enables the Inspector profiler plugin. The plugin must also be
        | loaded through ``NCCL_PROFILER_PLUGIN``.
      - | ``0``: Disabled (default).
        | ``1``: Enabled.

    * - | ``NCCL_INSPECTOR_ENABLE_P2P``
        | Enables tracking of point-to-point (``Send``/``Recv``) operations in
        | addition to collectives. Required for the ``nccl_p2p_*`` Prometheus
        | metrics and the P2P panels of the Grafana dashboard.
      - | ``0``: Disabled.
        | ``1``: Enabled (default).

    * - | ``NCCL_INSPECTOR_PROM_DUMP``
        | Selects the Prometheus node-exporter textfile output format
        | (``nccl_inspector_metrics_<uuid>.prom``) instead of the default JSON.
      - | ``0``: JSON output (default).
        | ``1``: Prometheus textfile output.

    * - | ``NCCL_INSPECTOR_DUMP_THREAD_ENABLE``
        | Enables the internal dump thread. When disabled, output is only
        | written at communicator teardown, regardless of the configured
        | dump interval.
      - | ``0``: Disabled.
        | ``1``: Enabled (default).

    * - | ``NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS``
        | Interval, in microseconds, at which the internal dump thread writes
        | output. Output is always written at communicator teardown.
      - | ``-1``: Dump only at teardown (default).
        | ``0``: Dump continuously.
        | ``N``: Dump every ``N`` microseconds. In Prometheus mode a minimum of
        | ``30000000`` (30 s) is enforced to match node-exporter polling.

    * - | ``NCCL_INSPECTOR_DUMP_DIR``
        | Output directory for Inspector logs/metrics. For Prometheus mode,
        | point this at the node-exporter textfile collector directory.
      - | String path.
        | Default: ``nccl-inspector-<slurm_job_id>`` or
        | ``nccl-inspector-unknown-jobid``.

    * - | ``NCCL_INSPECTOR_DUMP_VERBOSE``
        | Includes per-event trace information (sequence numbers and
        | timestamps) in the JSON output.
      - | ``0``: Disabled (default).
        | ``1``: Enabled.

    * - | ``NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES``
        | Minimum message size (in bytes) tracked by the Inspector.
      - | Integer value in bytes (default: ``8192``).

    * - | ``NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING``
        | Requires GPU-based kernel timing for an event to be recorded. When
        | enabled, events that fall back to CPU-measured timing are discarded.
      - | ``0``: Record events regardless of timing source.
        | ``1``: Record only GPU-timed events (default).

    * - | ``NCCL_INSPECTOR_DUMP_COLL_RING_SIZE``
        | Per-communicator capacity of the ring buffer holding completed
        | collectives waiting to be dumped.
      - | Integer number of entries (default: ``1024``).

    * - | ``NCCL_INSPECTOR_DUMP_P2P_RING_SIZE``
        | Per-communicator capacity of the ring buffer holding completed
        | point-to-point operations waiting to be dumped.
      - | Integer number of entries (default: ``1024``).

    * - | ``NCCL_INSPECTOR_COLL_POOL_SIZE``
        | Initial size, and growth stride, of the collective event pool.
      - | Integer number of entries (default: ``256``).

    * - | ``NCCL_INSPECTOR_P2P_POOL_SIZE``
        | Initial size, and growth stride, of the point-to-point event pool.
      - | Integer number of entries (default: ``256``).

    * - | ``NCCL_INSPECTOR_COMM_POOL_SIZE``
        | Initial size, and growth stride, of the communicator event pool.
      - | Integer number of entries (default: ``256``).

    * - | ``NCCL_INSPECTOR_POOL_GROW``
        | Allows the event pools above to grow beyond their initial size. When
        | disabled, events are dropped once a pool is exhausted.
      - | ``0``: Fixed-size pools.
        | ``1``: Pools grow on demand (default).
