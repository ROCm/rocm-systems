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
          the ``minCTAs`` field of ``ncclConfig_t``. Use ``NCCL_MIN_NCHANNELS``
          instead.
      - | Positive integer (values ``<= 0`` are ignored).
        | Default: unset (uses the RCCL default).

    * - | ``NCCL_MAX_CTAS``
        | Maximum number of CTAs (channels) used for a collective. Overrides
          the ``maxCTAs`` field of ``ncclConfig_t``. Use ``NCCL_MAX_NCHANNELS``
          instead.
      - | Positive integer (values ``<= 0`` are ignored).
        | Default: unset (uses the RCCL default).

    * - | ``NCCL_ALLGATHERV_ENABLE``
        | Fuses grouped multi-root ``ncclBroadcast`` calls into a single AllGatherV
          ring kernel when two or more distinct roots appear in a group.
      - | ``0``: Disabled (default).
        | ``1``: Enabled.

    * - | ``NCCL_IGNORE_CPU_AFFINITY``
        | Controls whether RCCL honors the job's supplied CPU affinity when
          deriving the CPU affinity for a rank. When disabled (default), RCCL
          intersects the current process affinity with the CPU affinity near
          the selected GPU. When enabled, RCCL ignores the process affinity
          and uses only the GPU-local CPU affinity.
      - | ``0`` (default): Keep the job-supplied CPU affinity in effect by
          intersecting it with the GPU-local CPU affinity.
        | ``1``: Ignore the job-supplied CPU affinity and use only the
          GPU-local CPU affinity.

    * - | ``NCCL_MULTI_RANK_GPU_ENABLE``
        | When set to ``0`` (default), RCCL rejects communicator configurations
          where two or more ranks on the same node map to the same physical GPU
          UUID or device index. Set to ``1`` to permit multi-rank-per-GPU
          configurations.
      - | ``0`` (default): Multiple ranks sharing the same GPU is an error.
        | ``1``: Allow multiple ranks to share the same physical GPU or GPU
          partition on the same node.

    * - | ``NCCL_RUNTIME_CONNECT``
        | Enables deferred ("lazy") transport connection setup. When disabled
          (default), connections for each algorithm (Ring, Tree, etc.) are
          established on demand the first time a collective using that algorithm
          is enqueued. When enabled, channels are initialized immediately at
          comm creation. PAT algorithm connections are always lazy regardless of
          this setting. Requires cuMem (VMM) support on the device.
      - | ``0`` (default): Force init-stage connection setup.
        | ``1``: Enable runtime connection setup (only active when cuMem support
          is available).

    * - | ``NCCL_UID_STAGGER_RATE``
        | Controls the rate at which ranks stagger their bootstrap connections to
          the root rank during communicator initialization. When the number of
          ranks connecting to a single root exceeds ``NCCL_UID_STAGGER_THRESHOLD``,
          each rank sleeps for a delay proportional to its local rank index divided
          by this rate. Lower values spread connections further apart; higher values
          pack them closer together. Commonly tuned down to ~5000 on large IB
          clusters.
      - | Any strictly positive integer representing message rate
          (messages/second).
        | Default: ``7000`` (rank with local index N sleeps N/7000 microseconds
          before connecting to the root).

    * - | ``NCCL_UID_STAGGER_THRESHOLD``
        | Minimum number of ranks connecting to a single bootstrap root before
          staggering is applied. When the rank count per root is at or below this
          threshold, all ranks connect simultaneously. Above it, each rank sleeps
          for a delay calculated from its local rank index and
          ``NCCL_UID_STAGGER_RATE``. Only affects init time.
      - | Strictly positive integer.
        | Default: ``256``.

    * - | ``NCCL_CTA_POLICY``
        | Sets the resource usage policy for the RCCL communicator.
      - | ``DEFAULT`` or ``0``: Use ``NCCL_CTA_POLICY_DEFAULT`` (default).
          This mode maximizes performance.
        | ``EFFICIENCY`` or ``1``: Use ``NCCL_CTA_POLICY_EFFICIENCY``.
          Efficiency mode is not yet supported.
        | ``ZERO`` or ``2``: Use ``NCCL_CTA_POLICY_ZERO``.

    * - | ``NCCL_MAX_NCHANNELS``
        | Sets the maximum number of CTAs (workgroups/channels) RCCL may use for
          a communicator. Overrides the ``maxCTAs`` field of ``ncclConfig_t``.
          Values ``<= 0`` are ignored. Values above RCCL's channel cap are clamped.
          Max channels allowed are 256 for gfx1250 and 64 for pre-gfx1250
          architectures such as gfx950 and gfx942. Max CTAs is also capped by
          the number of CUs available.
      - | Positive integer.
        | Default: unset (uses communicator configuration or RCCL's automatic
          default channel cap).

    * - | ``NCCL_MIN_NCHANNELS``
        | Controls the minimum number of channels RCCL should use. RCCL preserves
          the ``NCCL_MIN_NRINGS`` alias. RCCL's graph path starts from a higher
          default minimum and adds AMD-specific channel selection logic in both
          graph construction and runtime tuning paths.
      - | Any integer greater than or equal to ``1``.
        | Default: ``1``. Negative values are clamped to ``0``. Values above
          ``MAXCHANNELS`` are capped. RCCL may further reduce or ignore large
          requested values in specific topologies.

    * - | ``NCCL_NTHREADS``
        | Controls the number of GPU threads per block/workgroup that RCCL uses for
          tuned communication kernels. If unset or non-positive, RCCL uses its
          built-in default of 256 threads. Explicit values are range-checked and
          must be divisible by the runtime GPU warp size. On AMD gfx9-class 64-wide
          wavefront GPUs, only 256 is valid. On 32-wide warp GPUs, values of 128,
          160, 192, 224, or 256 are accepted.
      - | Multiples of the runtime warp size in the inclusive range
          [4×WarpSize, 256].
        | Default: ``256``.

    * - | ``NCCL_BUFFSIZE``
        | Controls the per-GPU-pair communication buffer size used by RCCL for the
          simple protocol path. Use this variable to reduce memory usage or
          experiment with a different transfer buffer size.
      - | Integer byte values.
        | Default: ``4194304`` bytes (4 MiB). Powers of 2 are recommended.

    * - | ``NCCL_TUNER_PLUGIN``
        | Selects a tuner plugin by suffix or library name. RCCL first tries the
          exact value as a library name; for suffix-style values it then tries
          ``libnccl-tuner-<suffix>.so``. If no standalone tuner plugin is found,
          RCCL next checks the net plugin for tuner symbols, then falls back to the
          built-in CSV tuner when available.
      - | Plugin suffix, plugin library name/path, or ``none``.
        | Default: unset (tries ``libnccl-tuner.so``).
        | ``none``: Disables external tuner loading.

    * - | ``NCCL_MEM_SYNC_DOMAIN``
        | Controls the CUDA Memory Sync Domain launch attribute for NCCL kernels on
          NVIDIA CUDA 12.0+ sm90/Hopper-class GPUs. RCCL preserves this environment
          variable for upstream parity, but AMD ROCm launches do not apply memory
          sync domains; on AMD hardware the value is effectively ignored.
      - | ``0`` or ``1``.

    * - | ``NCCL_CUMEM_HOST_ENABLE``
        | Determines the usage of HIP Virtual Memory functions to allocate host
          memory in RCCL. Requires ROCm 7.12 or higher.
      - | ``-1`` (default): Auto-select. RCCL probes at runtime whether the cuMem
          host allocation succeeds.
        | ``0``: Disabled.
        | ``1``: Enabled.

    * - | ``NCCL_IPC_USE_ABSTRACT_SOCKET``
        | Determines whether the Linux Abstract Socket mechanism is used when
          creating Unix Domain Sockets (UDS) for intra-node IPC handle exchange.
      - | ``1`` (default): Use abstract sockets.
        | ``0``: Do not use abstract sockets.

    * - | ``NCCL_WIN_ENABLE``
        | Enables memory window registration in RCCL.
      - | ``1`` (default): Enable window memory registration.
        | ``0``: Disable window memory registration.

    * - | ``NCCL_ALLOC_P2P_NET_LL_BUFFERS``
        | Controls whether communicators allocate dedicated LL/LL128 buffers for all
          P2P network connections. When enabled, the applicable threshold
          (``NCCL_P2P_LL128_THRESHOLD`` on gfx942/MI300X and gfx950/MI350X-MI355X
          with LL128 enabled, otherwise ``NCCL_P2P_LL_THRESHOLD``) determines
          whether the latency protocol or SIMPLE is used. Intranode P2P already
          allocates dedicated LL buffers. If running all-to-all workloads with high
          numbers of ranks, this will result in high scaling memory overhead.
      - | ``0`` (default): Disable allocation for LL/LL128 buffers.
        | ``1``: Enable allocations for LL/LL128 buffers.

    * - | ``NCCL_COMM_BLOCKING``
        | Controls whether RCCL communicator calls are allowed to block. This covers
          initialization/finalization and communication calls, including cases where
          send/receive operations may block during lazy connection setup. If set, the
          environment variable overrides the communicator blocking configuration.
      - | ``1``: Select blocking communicators.
        | ``0``: Select nonblocking communicators.
        | Default: unset (effective communicator default is ``1`` when config is
          not provided).

    * - | ``NCCL_GRAPH_REGISTER``
        | Controls whether RCCL registers user buffers during HIP graph capture.
      - | ``1`` (default on ROCm ≥ 7.12): Enable registration of user buffers.
        | ``0`` (default on older ROCm versions): Disable registration of user
          buffers.

    * - | ``NCCL_LAUNCH_MODE``
        | Controls how RCCL launches HIP kernels.
      - | ``PARALLEL`` (default): Use parallel launch mode.
        | ``GROUP``: Select grouped launch path for multi-GPU process management.

    * - | ``NCCL_MNNVL_ENABLE``
        | Controls the enabling of multi-node support over UALoE (Ultra Accelerator
          Link over Ethernet). Relies on AMD SMI APIs to query fabric-related
          information and Virtual Memory Management (VMM) APIs. Applies to
          gfx1250/MI450X systems. ``CU_ENABLE_CUMEM`` must be set to ``1`` to
          enable UALoE.
      - | ``0``: Disabled.
        | ``1``: Enabled.
        | ``2`` (default): Enabled if multi-node or gfx1250/MI450X and P2P not
          disabled.

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

    * - | ``NCCL_CROSS_NIC``
        | Controls whether RCCL allows ring and tree topology searches to use
          different NICs across nodes.
      - | ``0``: Always use the same NIC for the same ring/tree.
        | ``1``: Allow different NICs for the same ring/tree.
        | ``2`` (default): Prefer the same NIC but allow different NICs if that
          yields better performance.

    * - | ``NCCL_IB_ADAPTIVE_ROUTING``
        | Enables Adaptive Routing capable data transfers for the IB/RoCE net_ib
          transport. Adaptive Routing defaults to enabled on InfiniBand links and
          disabled on RoCE links.
      - | ``0``: Disabled.
        | ``1``: Enabled.
        | Default: Automatic link-layer behavior (InfiniBand=1, RoCE=0).

    * - | ``NCCL_IB_ADDR_FAMILY``
        | Selects which IP address family RCCL uses when dynamically choosing an
          InfiniBand/RoCE GID and ``NCCL_IB_GID_INDEX`` is unset.
      - | ``AF_INET`` (default): Use IPv4.
        | ``AF_INET6``: Use IPv6.

    * - | ``NCCL_IB_ADDR_RANGE``
        | Filters dynamically selected InfiniBand/RoCE GIDs by IP subnet when
          ``NCCL_IB_GID_INDEX`` is unset.
      - | IPv4 or IPv6 CIDR notation of the form ``<address>/<prefix-length>``.
        | Default: unset (disables filtering).

    * - | ``NCCL_IB_AR_THRESHOLD``
        | Threshold above which RCCL sends InfiniBand/RoCE net_ib data in a
          separate message that can leverage adaptive routing. Setting it above
          ``NCCL_BUFFSIZE`` disables adaptive routing completely.
      - | Integer byte size.
        | Default: ``8192``.

    * - | ``NCCL_IB_DISABLE``
        | Disables the IB/RoCE verbs transport in RCCL's net_ib transport. When
          enabled, RCCL skips IB transport initialization and falls back to another
          network method such as IP sockets.
      - | ``0`` (default): Keep IB/RoCE transport enabled.
        | ``1``: Disable IB/RoCE verbs transport and force fallback to another
          transport (for example, IP sockets).

    * - | ``NCCL_IB_ECE_ENABLE``
        | Enables Enhanced Connection Establishment (ECE) for RCCL's IB/RoCE verbs
          transport. When enabled, RCCL queries ECE capability during QP setup,
          exchanges the ECE metadata with the peer, and applies the negotiated
          settings before moving QPs to RTR/RTS.
      - | ``1`` (default): Enabled.
        | ``0``: Disabled.

    * - | ``NCCL_IB_FIFO_TC``
        | Sets the traffic class used for net_ib control-message QPs in RCCL. When
          unset, RCCL falls back to the connection traffic class from ``NCCL_IB_TC``,
          or ``0`` if that is also unset.
      - | Integer traffic class value.
        | Default: inherits ``NCCL_IB_TC`` (or ``0`` if ``NCCL_IB_TC`` is unset).

    * - | ``NCCL_IB_OOO_RQ``
        | Enables out-of-order work requests on the IB receive side.
      - | ``0`` (default): Disable out-of-order receive WRs.
        | ``1``: Enable out-of-order receive WRs.

    * - | ``NCCL_IB_PCI_RELAXED_ORDERING``
        | Controls whether RCCL enables PCI Relaxed Ordering for the net_ib
          InfiniBand/RoCE verbs transport.
      - | ``0``: Disable Relaxed Ordering.
        | ``1``: Require Relaxed Ordering and fail if support is unavailable.
        | ``2`` (default): Auto-enable if available.

    * - | ``NCCL_IB_QPS_PER_CONNECTION``
        | Sets the number of InfiniBand/RoCE queue pairs (QPs) to create for each
          connection between two ranks. Increasing the QP count can improve routing
          entropy on multi-level fabrics.
      - | ``1``–``128``.
        | Default: ``1``.

    * - | ``NCCL_IB_RETRY_CNT``
        | Controls the InfiniBand retry count used when RCCL transitions IB queue
          pairs to RTS. Total time spent retrying depends on the combination of
          ``NCCL_IB_RETRY_CNT`` and ``NCCL_IB_TIMEOUT``.
      - | ``0``–``7``.
        | Default: ``7``.

    * - | ``NCCL_IB_RETURN_ASYNC_EVENTS``
        | Controls whether the net_ib transport escalates fatal IB asynchronous
          events into communication termination. IB events are still surfaced as
          warnings, but when enabled RCCL also checks the fatal-error counter and
          stops IB communication after fatal asynchronous events are detected.
      - | ``1`` (default): Stop IB communications on fatal async events.
        | ``0``: Disable stopping IB communications on fatal async events.

    * - | ``NCCL_IB_ROCE_VERSION_NUM``
        | Selects which RoCE version RCCL prefers when dynamically choosing an
          InfiniBand/RoCE GID and ``NCCL_IB_GID_INDEX`` is unset.
      - | ``1`` or ``2``.
        | Default: ``2``.

    * - | ``NCCL_IB_SL``
        | Defines the InfiniBand Service Level used by RCCL for net_ib connections.
      - | Integer Service Level value.
        | Default: ``0``.

    * - | ``NCCL_IB_SPLIT_DATA_ON_QPS``
        | Controls how InfiniBand/RoCE queue pairs are used when
          ``NCCL_IB_QPS_PER_CONNECTION`` creates more than one QP for a connection.
          Set to ``1`` to split each message evenly across all QPs. Set to ``0`` to
          use QPs in round-robin mode on a per-message basis.
      - | ``0`` (default): Round-robin per-message QP usage.
        | ``1``: Split each message evenly across all QPs.

    * - | ``NCCL_IB_TC``
        | Sets the InfiniBand/RoCE traffic class used by RCCL net_ib connections.
      - | Integer traffic class value.
        | Default: ``0``.

    * - | ``NCCL_IB_TIMEOUT``
        | Controls the InfiniBand Verbs local-ack timeout used when RCCL programs
          IB queue pairs. The effective wait before an error also depends on
          ``NCCL_IB_RETRY_CNT``.
      - | ``0``–``31``.
        | Default: ``20``. A value of ``0`` or ``>= 32`` results in an infinite
          timeout.

    * - | ``NCCL_NET``
        | Forces RCCL to use a specific network module. RCCL supports the built-in
          names ``IB`` and ``Socket`` plus plugin-defined names. ``NCCL_NET=ROCM-IB``
          is translated to ``IB-CAST``.
      - | Case-insensitive network name.
        | Built-in names: ``IB``, ``Socket``, ``IB-CAST`` (AMD internal).
        | ``ROCM-IB``: Compatibility alias for ``IB-CAST``.
        | Default: unset (automatic network selection).

    * - | ``NCCL_DMABUF_ENABLE``
        | Enables ROCm dmabuf-based GPU memory registration for RCCL network
          transports. When enabled, RCCL exports AMD GPU buffers through the ROCr
          HSA dma-buf export path and registers the resulting dma-buf FD with
          regMrDmaBuf-capable network plugins so RDMA-capable NICs can access GPU
          memory without host staging. Only effective when ROCr runtime reports
          dmabuf support and the selected network plugin supports ``regMrDmaBuf``.
      - | ``1`` (default): Enabled (subject to runtime capability checks).
        | ``0``: Disabled.

    * - | ``NCCL_NET_PLUGIN``
        | Selects among RCCL net plugins by suffix or library name. RCCL first tries
          the exact value as a library name; if that fails and the value is a suffix,
          RCCL tries ``librccl-net-<value>.so``. If no external plugin is found,
          RCCL uses its internal network transport(s).
      - | Plugin suffix, plugin file name/path, or ``none``.
        | Default: unset (tries ``librccl-net.so``).

    * - | ``NCCL_NET_SHARED_BUFFERS``
        | Allows shared buffers for inter-node point-to-point communication, using a
          single large pool for all remote peers so buffer memory does not scale
          linearly with the number of peers.
      - | ``1`` (default): Enabled.
        | ``0``: Disabled.

    * - | ``NCCL_NET_SHARED_COMMS``
        | Reuses the same network connections in the context of PXN so messages can
          be aggregated across shared network communication paths.
      - | ``1`` (default): Enabled.
        | ``0``: Disabled.

    * - | ``NCCL_NVB_DISABLE``
        | Disables intra-node communication through XGMI via an intermediate
          GPU/device hop.
      - | ``0`` (default): Intra-node XGMI intermediate-hop enabled.
        | ``1``: Disabled.

    * - | ``NCCL_NSOCKS_PERTHREAD``
        | Specifies the number of sockets opened by each helper thread of the socket
          transport. Increasing it above ``1`` can improve performance when per-socket
          bandwidth is limited. The product of ``NCCL_SOCKET_NTHREADS`` and
          ``NCCL_NSOCKS_PERTHREAD`` cannot exceed ``64``.
      - | Positive integer.
        | Default: ``8`` on AWS, ``1`` otherwise.

    * - | ``NCCL_SOCKET_NTHREADS``
        | Specifies how many socket helper threads RCCL uses per connection for the
          socket transport. Increasing this value may increase socket transport
          performance at the cost of higher CPU usage. The product of
          ``NCCL_SOCKET_NTHREADS`` and ``NCCL_NSOCKS_PERTHREAD`` cannot exceed ``64``.
      - | ``1``–``16``.
        | Default: ``2`` on AWS; ``4`` on Google Cloud gVNIC instances (since
          2.5.6); ``1`` otherwise.
        | For generic 100G networks, ``4`` is recommended.

    * - | ``NCCL_SOCKET_POLL_TIMEOUT_MSEC``
        | Sets a timeout in milliseconds for socket polling during bootstrap to
          reduce CPU usage. Non-zero values cause RCCL to poll for the specified
          number of milliseconds before attempting to progress the operation again.
      - | ``0`` (default): No polling; RCCL continuously tries to progress socket
          operations without pausing.
        | ``N``: Poll for N milliseconds between retry attempts.

    * - | ``NCCL_SOCKET_RETRY_CNT``
        | Specifies how many times RCCL retries establishing a socket connection
          after ``ETIMEDOUT``, ``ECONNREFUSED``, or ``EHOSTUNREACH`` before failing
          the connection attempt.
      - | Any positive integer.
        | Default: ``34``.

    * - | ``NCCL_SOCKET_RETRY_SLEEP_MSEC``
        | Specifies the number of milliseconds RCCL waits before retrying to
          establish a socket connection after the first ``ETIMEDOUT``,
          ``ECONNREFUSED``, or ``EHOSTUNREACH`` error. For subsequent errors, the
          wait scales linearly with the retry count.
      - | Any positive integer value in milliseconds.
        | Default: ``100``.

    * - | ``NCCL_OOB_NET_ENABLE``
        | Controls whether RCCL uses the net plugin transport for bootstrap
          out-of-band communication during communicator initialization instead of
          the socket bootstrap path.
      - | ``-1``: Auto-select the net OOB bootstrap path when nranks is greater
          than or equal to ``NCCL_BOOTSTRAP_BIDIR_THRESHOLD``.
        | ``0`` (default): Disable net OOB bootstrap and use the socket bootstrap
          path.
        | ``1``: Force-enable net OOB bootstrap regardless of communicator size.

    * - | ``NCCL_OOB_NET_IFNAME``
        | Specifies which RCCL net interfaces to use for out-of-band bootstrap
          communication when ``NCCL_OOB_NET_ENABLE`` enables the net-based OOB
          path. The list is comma-separated; use ``:`` to specify ports; prefix the
          list with ``^`` to exclude matching interfaces; use ``=`` to require exact
          interface-name matches (otherwise RCCL treats each token as a prefix
          match). If unset or empty, RCCL uses OOB net device 0.
          Check available interfaces with ``ibv_devinfo -l``.
      - | Comma-separated list of interface names or prefixes.
        | Default: unset (uses OOB net device 0).

    * - | ``NCCL_P2P_PXN_LEVEL``
        | Controls when RCCL uses PXN-style relay routing for send/receive
          operations.
      - | ``0``: Disable PXN relay routing for send/receive.
        | ``1``: Enable only when the destination-preferred NIC is not reachable
          through PCI switches.
        | ``2`` (default): Always use PXN, even if the NIC is connected through
          PCI switches, storing data from all GPUs within the node on an
          intermediate GPU to maximize aggregation.

    * - | ``NCCL_PXN_DISABLE``
        | Disables inter-node communication through a non-local NIC by relaying
          traffic via an intermediate GPU on the same node. The default behavior
          differs from NCCL: RCCL's base param default is ``1``, but when the
          variable is unset, RCCL auto-enables PXN at larger scales (≥64 ranks on
          gfx942/MI300X, ≥32 ranks on gfx950/MI350X-MI355X). Setting this variable
          to ``0`` force-enables PXN regardless of scale.
      - | ``0``: Enable the PXN relay path.
        | ``1``: Disable it.
        | Default: RCCL auto-enables PXN based on rank count and GPU architecture
          when unset.

    * - | ``NCCL_P2P_DIRECT_DISABLE``
        | Controls direct access to user buffers through P2P between GPUs of the
          same process.
      - | ``0`` (default): Allow direct same-process user-buffer P2P access.
        | ``1``: Disable direct same-process user-buffer P2P access.

    * - | ``NCCL_P2P_DISABLE``
        | Disables RCCL peer-to-peer (P2P) transport for direct GPU-to-GPU
          communication over XGMI/Infinity Fabric or PCIe.
      - | Unset (default): Enable GPU-to-GPU P2P communication.
        | ``1``: Disable GPU-to-GPU P2P communication.

    * - | ``NCCL_P2P_LL_THRESHOLD``
        | Sets the maximum per-channel message size, in bytes, for which RCCL keeps
          the LL protocol for point-to-point operations before switching to SIMPLE.
      - | Decimal integer in bytes.
        | Default: ``4096``.

    * - | ``NCCL_P2P_NET_CHUNKSIZE``
        | Controls the chunk size, in bytes, used for network point-to-point
          ``ncclSend``/``ncclRecv`` operations. The effective default may be raised
          automatically on supported AMD GPUs when the variable is unset:
          524288 bytes on gfx942/MI300X with ≥64 ranks; 262144 bytes on
          gfx950/MI350X-MI355X with 16–31 ranks; 524288 bytes on gfx950/MI300X
          with ≥32 ranks. Setting the variable skips arch/rank auto-tuning.
      - | Integer value in bytes. Powers of two are recommended.
        | Default: ``131072`` bytes (auto-tuned to higher values on supported
          architectures and rank counts when unset).

    * - | ``NCCL_SHM_DISABLE``
        | Disables RCCL shared-memory (SHM) transport for intra-node communication.
          SHM is used when peer-to-peer cannot be used. If SHM is disabled, RCCL
          falls back to the network transport.
      - | ``0`` (default): Keep SHM transport enabled.
        | ``1``: Disable communication through shared memory.

    * - | ``RCCL_THREADS_PER_BLOCK``
        | Overrides the per-CTA thread count for collectives, replacing the
          per-algorithm and per-protocol defaults. The requested value is rounded up
          to a multiple of the GPU warp size, capped at 256, and raised to 3×warp
          size if below the tree kernel's minimum. On warp-size-64 GPUs (gfx908,
          gfx90a, gfx942, gfx950) the only effective values are 192 and 256.
          On warp-size-32 GPUs (gfx1200, gfx1201, gfx1250) valid values are
          96, 128, 160, 192, 224, and 256. Out-of-range values are silently clamped
          and logged at INFO level.
      - | Integer thread count.
        | Default: ``-1`` (RCCL picks per algorithm and protocol).

    * - | ``RCCL_UNROLL_FACTOR``
        | Selects the device-kernel unroll factor as a base-2 exponent. Invalid
          values and values whose device function table was not generated for the
          current architecture return ``ncclInvalidArgument``, failing communicator
          creation.
      - | ``0``–``5``, mapping to unroll 1, 2, 4, 8, 16, 32.
        | Default: ``-1`` (per-arch auto): gfx950 uses 1 single-node and 2
          multi-node; gfx908 and gfx942 above 80 CUs use 2; gfx1250 uses 32;
          everything else uses 4.

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
