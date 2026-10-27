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
        | Specifies the path to the RCCL static configuration file.
      - | A relative or absolute filesystem path to a configuration file;
          ``~`` is not accepted literally.
        | If unset or empty, RCCL falls back to ``~/.rccl.conf`` when
          ``HOME`` is set, then ``/etc/rccl.conf``.

    * - | ``NCCL_IBVERBS_LIB``
        | Overrides the first ``libibverbs`` shared library name or path
          that RCCL tries to open for the dynamically loaded IB verbs backend.
          When unset, RCCL tries the default ``libibverbs`` sonames; when set,
          RCCL tries the specified library first and then falls back to
          ``libibverbs.so`` and ``libibverbs.so.1`` if the custom open fails.
          ``NCCL_LIBIBVERBS_SO`` is accepted as an alias when
          ``NCCL_IBVERBS_LIB`` is unset.
      - | String path or soname of the ``libibverbs`` shared object.
        | Default: unset (loads ``libibverbs.so`` or ``libibverbs.so.1``).

    * - | ``NCCL_HOSTID``
        | Overrides the string used to generate the per-host hash for host
          identification in multi-node communication.
      - | Unset: RCCL uses the hostname plus the platform-specific host
          identifier (Linux ``boot_id`` or Windows ``MachineGuid``).
        | Set: any string value; RCCL copies the string into the host-hash
          buffer, truncating to fit if necessary, and hashes that value instead.

    * - | ``NCCL_BOOTSTRAP_BIDIR_ALLGATHER``
        | Controls the bootstrap ``AllGather`` algorithm used on RCCL's socket
          out-of-band path. When enabled, RCCL uses a bidirectional ring
          ``AllGather`` that completes in N/2 steps instead of the classic
          unidirectional N-1-step ring, while keeping the unidirectional
          implementation available as a fallback. Has no effect when net OOB
          is in use.
      - | ``0``: Force the classic unidirectional socket-ring ``AllGather``.
        | ``1``: Force the bidirectional socket-ring ``AllGather`` (default).

    * - | ``NCCL_CUMEM_ENABLE``
        | Enables cuMem virtual memory management (VMM) for RCCL allocations,
          which is required for ``ncclCommSuspend`` and ``ncclCommResume`` to
          release the physical GPU memory of a suspended communicator. See
          :ref:`suspend-resume` for the full prerequisites.
      - | ``0``: Disabled.
        | ``1``: Enabled on any architecture.
        | ``-2``: Auto-detect (default); enable when the platform supports VMM.
          Auto-detect is limited to ``gfx1250``, the only architecture where the
          VMM path is validated. Use ``1`` to force it on elsewhere.

    * - | ``NCCL_CUMEM_HOST_ENABLE``
        | Determines whether RCCL uses HIP virtual memory functions to allocate
          host memory. Requires ROCm 7.12 or later.
      - | ``-1``: Auto-select; RCCL probes at runtime whether the cuMem host
          allocation succeeds (default).
        | ``0``: Disabled.
        | ``1``: Enabled.

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

    * - | ``NCCL_MAX_NCHANNELS``
        | Sets the maximum number of CTAs (workgroups/channels) RCCL may use
          for a communicator. Overrides the ``maxCTAs`` field of
          ``ncclConfig_t``; values above RCCL's channel cap are clamped.
      - | Positive integer. Values ``<= 0`` are ignored.
        | Maximum: 256 for ``gfx1250``; 64 for other architectures.
        | Default: unset.

    * - | ``NCCL_MIN_NCHANNELS``
        | Controls the minimum number of channels RCCL should use. RCCL also
          accepts the legacy alias ``NCCL_MIN_NRINGS``. Negative values are
          clamped to ``0`` and values above ``MAXCHANNELS`` are capped.
      - | Integer >= ``1``. Default: ``1``.

    * - | ``NCCL_NTHREADS``
        | Controls the number of GPU threads per block/workgroup that RCCL
          uses for tuned communication kernels.
      - | Positive multiples of the GPU warp size, in the range
          ``[4 * WarpSize, 256]``. Default: ``256``.
        | On AMD ``gfx9``-class GPUs (64-wide wavefront), the only effective
          value is ``256``.
        | On 32-wide warp GPUs: ``128``, ``160``, ``192``, ``224``, or ``256``.

    * - | ``NCCL_BUFFSIZE``
        | Controls the per-GPU-pair communication buffer size used by RCCL
          for the Simple protocol path.
      - | Integer byte value. Default: ``4194304`` (4 MiB).
        | Powers of 2 are recommended.

    * - | ``NCCL_CTA_POLICY``
        | Sets the resource usage policy for the RCCL communicator.
      - | ``DEFAULT`` (or ``0``): Maximize performance (default).
        | ``EFFICIENCY`` (or ``1``): Efficiency mode. Not yet supported.
        | ``ZERO`` (or ``2``): Zero-resource policy.

    * - | ``NCCL_COMM_BLOCKING``
        | Controls whether RCCL communicator calls are allowed to block. If
          set, this variable overrides the communicator blocking configuration;
          if unset, RCCL uses the communicator config, defaulting to blocking
          when config is not provided.
      - | ``1``: Blocking communicators.
        | ``0``: Nonblocking communicators.
        | Default: unset (effective default ``1``).

    * - | ``NCCL_RUNTIME_CONNECT``
        | Enables deferred (lazy) transport connection setup. When disabled,
          connections for each algorithm are established on demand the first
          time a collective using that algorithm is enqueued. PAT algorithm
          connections are always lazy regardless of this setting. Requires
          cuMem (VMM) support on the device.
      - | ``0``: Init-stage connection setup (default).
        | ``1``: Runtime (lazy) connection setup.

    * - | ``NCCL_LAUNCH_MODE``
        | Controls how RCCL launches HIP kernels.
      - | ``PARALLEL``: Default parallel launch mode.
        | ``GROUP``: Grouped launch path for multi-GPU process management.

    * - | ``NCCL_LAUNCH_RACE_FATAL``
        | Attempts to detect host threads racing to launch RCCL work onto the
          same device while launch-order mixing support is active. When
          triggered, RCCL reports the race and returns ``ncclInvalidUsage``.
      - | ``0``: Disable the race check.
        | ``1``: Enable the race check (default).

    * - | ``NCCL_MULTI_RANK_GPU_ENABLE``
        | Controls whether multiple ranks on the same node may map to the same
          physical GPU. By default, RCCL rejects such configurations during
          comm init peer-info validation.
      - | ``0``: Multiple ranks sharing the same GPU is an error (default).
        | ``1``: Allow multiple ranks to share the same physical GPU or GPU
          partition on the same node.

    * - | ``NCCL_UID_STAGGER_RATE``
        | Controls the rate at which ranks stagger their bootstrap connections
          to the root rank. When the rank count exceeds
          ``NCCL_UID_STAGGER_THRESHOLD``, each rank sleeps for a delay
          proportional to its local rank index divided by this rate (converted
          to microseconds).
      - | Strictly positive integer (messages per second). Default: ``7000``.
        | Rank N sleeps ``N / rate`` microseconds before connecting to the root.

    * - | ``NCCL_UID_STAGGER_THRESHOLD``
        | Minimum number of ranks connecting to a single bootstrap root before
          staggering is applied. At or below this threshold, all ranks connect
          simultaneously.
      - | Strictly positive integer. Default: ``256``.

    * - | ``NCCL_GRAPH_REGISTER``
        | Controls whether RCCL registers user buffers during HIP graph capture.
      - | ``0``: Disable user buffer registration.
        | ``1``: Enable user buffer registration (default on ROCm >= 7.12).

    * - | ``NCCL_GRAPH_MIXING_SUPPORT``
        | Enables or disables support for multiple outstanding RCCL calls from
          parallel HIP graphs, or from a HIP graph mixed with non-captured RCCL
          calls on the same communicator.
      - | ``0``: Mixing disabled (default).
        | ``1``: Mixing enabled.

    * - | ``NCCL_LEGACY_CUDA_REGISTER``
        | Controls legacy buffer registration. Registering ``hipMalloc``
          buffers can cause implicit synchronization and potential hangs.
          Prefer HIP virtual memory management allocators when registration
          is required.
      - | ``0``: Disabled (default, recommended).
        | ``1``: Enable legacy buffer registration.

    * - | ``NCCL_MNNVL_ENABLE``
        | Controls multi-node support over UALoE (Ultra Accelerator Link over
          Ethernet). Relies on AMD SMI and VMM APIs. Applies to
          ``gfx1250``/MI450X systems.
      - | ``0``: Disabled.
        | ``1``: Enabled.
        | ``2``: Enabled if multi-node or ``gfx1250``/MI450X and P2P is not
          disabled. ``NCCL_CUMEM_ENABLE=1`` required (default).

    * - | ``NCCL_ALLGATHERV_ENABLE``
        | Fuses grouped multi-root ``ncclBroadcast`` calls into a single
          AllGatherV ring kernel when two or more distinct roots appear in a
          group.
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

    * - | ``NCCL_SET_THREAD_NAME``
        | Gives more meaningful names to RCCL CPU threads to ease debugging and
          analysis.
      - | ``0``: Leave thread names unchanged (default).
        | ``1``: Name internal CPU helper, proxy, bootstrap, RAS, GIN, socket,
          and InfiniBand-related threads.

    * - | ``NCCL_TOPO_DUMP_FILE``
        | Path to a file where RCCL writes the detected topology XML after
          topology detection.
      - | A filesystem path to a file that will be created or overwritten.
        | Default: unset.

    * - | ``NCCL_TOPO_FILE``
        | Path to an XML topology file to load before RCCL runs topology
          detection. RCCL loads the file first and then continues discovery
          using the resulting XML state.
      - | Path to an accessible XML file.
        | If unset, RCCL tries
          ``/var/run/nvidia-topologyd/virtualTopology.xml`` as the default.

    * - | ``NCCL_PROFILER_PLUGIN``
        | Controls selection of the external profiler plugin.
      - | Plugin library name/path: loaded directly.
        | Plugin suffix: resolves to ``libnccl-profiler-<suffix>.so``.
        | ``STATIC_PLUGIN``: searches for profiler symbols in the program binary.
        | ``none``: disables external profiler plugin loading.
        | Default: unset (tries ``libnccl-profiler.so``).

    * - | ``NCCL_RAS_ENABLE``
        | Enables RCCL's reliability, availability, and serviceability (RAS)
          subsystem during communicator bootstrap and split initialization.
      - | ``0``: Disable RAS.
        | ``1``: Enable RAS (default).

    * - | ``NCCL_RAS_ADDR``
        | Specifies the IP address and port of the socket that the RAS subsystem
          listens on for client connections. Use different addresses or ports for
          independent jobs sharing a node.
      - | Host/IP plus port. Examples: ``localhost:12345``,
          ``192.168.1.10:28028``, ``[::1]:28028``.
        | Default: ``localhost:28028``.

    * - | ``NCCL_RAS_TIMEOUT_FACTOR``
        | Multiplier applied to RCCL RAS subsystem timeouts (keepalive,
          connection retry, stuck, idle, peer-dead, and collective timeout
          constants).
      - | Integer timeout multiplier. Default: ``1``.

    * - | ``RCCL_LATENCY_PROFILER``
        | Enables RCCL's host-managed collective profiler, which brackets
          collective kernel plans with GPU events and reports per-operation
          minimum and maximum latency across local ranks.
      - | ``1``: Enable the latency profiler.
        | Default: unset (disabled).

    * - | ``RCCL_LOG_LEVEL``
        | Sets the verbosity of RCCL structured API logging for the Replayer.
          Active only when ``RCCL_REPLAY_FILE`` is set. Level 1 records
          essential calls; a higher level also records informational calls.
      - | Integer. ``1`` (or ``0``): essential APIs only.
        | ``>1``: also informational APIs.
        | Default: ``1``.

    * - | ``RCCL_LOG_ROCTX``
        | Emits ROCTX ranges around RCCL operations so profilers such as
          ``rocprof`` can show them. The library must be built with
          ``ROCTX_ENABLE``.
      - | ``0``: Disabled (default).
        | Non-zero: Enabled.

    * - | ``RCCL_REPLAY_FILE``
        | Enables RCCL structured logging of API traces and generates output
          files. Each process writes its own log file named
          ``{prefix}.{pid}.{hostname}.{extension}``.
      - | A string literal file name. If the value ends in ``.json``, output
          is JSON; otherwise binary.
        | Default: unset (no output).

    * - | ``RCCL_OUTPUT_TREES``
        | After tree connection, logs each channel's parent/child edges in
          Graphviz dot syntax. Requires ``NCCL_DEBUG=INFO`` (or more verbose)
          and ``NCCL_DEBUG_SUBSYS`` including ``GRAPH``.
      - | ``0``: No output (default).
        | Non-zero: Enable tree edge logging.

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
        | Controls which collective algorithms RCCL may use. On AMD ROCm the
          ``NVLS`` and ``NVLSTree`` options are not available; forcing those
          names can leave no usable algorithm and return an error.
      - | Case-insensitive algorithm names: ``Ring``, ``Tree``,
          ``CollNetDirect``, ``CollNetChain``, ``NVLS``, ``NVLSTree``, ``PAT``.
        | Multiple values accepted as a comma-separated list. Optional
          per-function selections use semicolon/colon syntax.
        | ``NVLS`` and ``NVLSTree`` are parsed for compatibility but are not
          available on AMD ROCm.
        | Default: unset (automatic selection).

    * - | ``NCCL_PROTO``
        | Controls which communication protocol(s) RCCL is allowed to use.
      - | Case-insensitive protocol names: ``LL``, ``LL128``, ``Simple``.
        | Values may be given as a comma-separated list; prefix with ``^`` to
          exclude the listed protocols instead of including them.
        | Default: unset (automatic selection).

    * - | ``NCCL_CROSS_NIC``
        | Controls whether ring and tree topology searches may use different
          NICs between local and remote devices, crossing network rails.
      - | ``0``: Always use the same NIC for the same ring/tree.
        | ``1``: Allow different NICs for the same ring/tree.
        | ``2``: Prefer same NIC but allow different NICs for better performance
          (default).

    * - | ``NCCL_TUNER_PLUGIN``
        | Selects a tuner plugin by suffix or library name. RCCL first tries the
          exact value as a library name; for suffix-style values it tries
          ``libnccl-tuner-<suffix>.so``; ``none`` disables external tuner loading;
          if no standalone tuner plugin is found RCCL checks the net plugin for
          tuner symbols, then falls back to the built-in CSV tuner.
      - | Plugin suffix, plugin library name/path, or ``none``.
        | Default: unset (uses built-in CSV tuner).

    * - | ``RCCL_OVERRIDE_ALGO``
        | Forces one collective algorithm after tuning, instead of only shrinking
          the candidate set as ``NCCL_ALGO`` does. Unsupported names at runtime
          warn and leave the tuner's choice unchanged.
      - | ``Tree``, ``Ring``, ``CollNetDirect``, ``CollNetChain``, ``NVLS``,
          ``NVLSTree``, or ``PAT`` (case-insensitive).
        | Default: unset.

    * - | ``RCCL_OVERRIDE_PROTO``
        | Forces one communication protocol after tuning, instead of only
          shrinking the candidate set as ``NCCL_PROTO`` does. Unsupported
          combinations at runtime warn and leave the tuner's choice unchanged.
      - | ``LL``, ``LL128``, or ``Simple`` (case-insensitive).
        | Default: unset.

    * - | ``RCCL_LL128_FORCE_ENABLE``
        | Forces on the communicator's LL128 capability flag, which is otherwise
          set only by a matching topology model. Required before
          ``NCCL_PROTO=LL128`` has any effect on ``gfx942`` and ``gfx950``;
          ``gfx1250`` enables it by default. The flag is AND-reduced across
          ranks, so it must be set identically everywhere.
      - | ``0``: Use topology and architecture defaults (default).
        | ``1``: Force LL128 available.

Network and topology
====================

The network and topology environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_NET``
        | Forces RCCL to use a specific network module. RCCL supports the
          built-in names ``IB`` and ``Socket`` plus plugin-defined names, and
          also has an AMD internal ``IB-CAST`` network. ``NCCL_NET=ROCM-IB`` is
          translated to ``IB-CAST``.
      - | ``IB``, ``Socket``, ``IB-CAST``, ``ROCM-IB`` (alias for ``IB-CAST``),
          or an external plugin name.
        | Default: unset (automatic selection).

    * - | ``NCCL_NET_PLUGIN``
        | Selects among RCCL net plugins by suffix or library name. RCCL first
          tries the exact value as a library name; for suffix-style values it
          tries ``librccl-net-<value>.so``; if unset, RCCL tries
          ``librccl-net.so``.
      - | Plugin suffix, plugin file name/path, or ``none``.
        | Default: unset (tries ``librccl-net.so``).

    * - | ``NCCL_IB_HCA``
        | Selects which IB/RoCE HCAs and ports RCCL uses for the ``net_ib`` and
          ``net_ib_cast`` transports. RCCL also reuses this variable in an
          AMD-added NIC-detection path.
        | When unset, RCCL auto-selects an interface in this order:
        | ``ib*`` first; if none is found and ``NCCL_COMM_ID`` is set, an
        | interface on the same subnet as that address; then any interface
        | other than ``docker*``, ``lo`` and ``virbr*``; then ``docker*``;
        | then ``lo``; and finally ``virbr*``.
      - | Comma-separated ``<hca>[:<port>[:<rail>[:<plane>]]]`` entries.
          Optional leading ``^`` for exclusion; optional leading ``=`` for
          exact HCA-name matching; omitted ``<port>`` means all ports.

    * - | ``NCCL_IB_GID_INDEX``
        | Defines the Global ID index used in RoCE mode.
      - | Integer value. Default: ``-1``.
        | See the InfiniBand ``show_gids`` command for valid values.

    * - | ``NCCL_IB_ADDR_FAMILY``
        | Selects which IP address family RCCL uses when dynamically choosing an
          InfiniBand/RoCE GID and ``NCCL_IB_GID_INDEX`` is unset.
      - | ``AF_INET`` (default) or ``AF_INET6``.

    * - | ``NCCL_IB_ADDR_RANGE``
        | Filters dynamically selected InfiniBand/RoCE GIDs by IP subnet when
          ``NCCL_IB_GID_INDEX`` is unset.
      - | IPv4 or IPv6 CIDR notation, for example ``192.168.1.0/24``.
        | Default: unset (no filtering).

    * - | ``NCCL_IB_ROCE_VERSION_NUM``
        | Selects the RoCE version RCCL prefers when dynamically choosing a GID
          and ``NCCL_IB_GID_INDEX`` is unset.
      - | ``1`` or ``2``. Default: ``2``.

    * - | ``NCCL_IB_DISABLE``
        | Disables the IB/RoCE verbs transport in RCCL. RCCL skips IB transport
          initialization and falls back to IP sockets. Does not disable
          intra-node transports or an externally supplied network plugin unless
          that plugin independently interprets this variable.
      - | ``0``: Keep IB/RoCE transport enabled (default).
        | ``1``: Disable IB/RoCE verbs transport.

    * - | ``NCCL_IB_MERGE_NICS``
        | Enables RCCL to combine several physical IB NICs that are close to the
          same GPU into a single logical network device (NIC Fusion). This allows
          RCCL to aggregate the bandwidth of those NICs. Use
          ``NCCL_NET_MERGE_LEVEL`` and ``NCCL_NET_FORCE_MERGE`` to control which
          NICs are combined. On AINIC with the ``IB-CAST`` transport, merging is
          off unless this variable is explicitly set to ``1``.
      - | ``1``: Enabled (default).
        | ``0``: Disabled.

    * - | ``NCCL_IB_TC``
        | Sets the InfiniBand/RoCE traffic class used by RCCL ``net_ib``
          connections.
      - | Integer traffic class value. Default: ``0``.

    * - | ``NCCL_IB_FIFO_TC``
        | Sets the traffic class used for ``net_ib`` control-message QPs. When
          unset, RCCL falls back to ``NCCL_IB_TC`` or ``0``.
      - | Integer traffic class value. Default: inherits ``NCCL_IB_TC``.

    * - | ``NCCL_IB_SL``
        | Defines the InfiniBand Service Level used by RCCL for ``net_ib``
          connections.
      - | Integer Service Level value. Default: ``0``.

    * - | ``NCCL_IB_TIMEOUT``
        | Controls the InfiniBand Verbs local-ack timeout used when RCCL
          programs IB queue pairs. The effective wait before an error also
          depends on ``NCCL_IB_RETRY_CNT``.
      - | ``0``–``31``. Default: ``20``.
        | A value of ``0`` or ``>= 32`` results in an infinite timeout.

    * - | ``NCCL_IB_RETRY_CNT``
        | Controls the InfiniBand retry count used when RCCL transitions IB
          queue pairs to RTS.
      - | ``0``–``7``. Default: ``7``.

    * - | ``NCCL_IB_QPS_PER_CONNECTION``
        | Sets the number of InfiniBand/RoCE queue pairs to create for each
          connection between two ranks. Increasing the QP count can improve
          routing entropy on multi-level fabrics.
      - | ``1``–``128``. Default: ``1``.

    * - | ``NCCL_IB_SPLIT_DATA_ON_QPS``
        | Controls how QPs are used when ``NCCL_IB_QPS_PER_CONNECTION`` creates
          more than one QP. When ``1``, each message is split evenly across all
          QPs; when ``0``, QPs are used in round-robin per message.
      - | ``0`` (default) or ``1``.

    * - | ``NCCL_IB_PCI_RELAXED_ORDERING``
        | Controls whether RCCL enables PCI Relaxed Ordering for the ``net_ib``
          transport.
      - | ``0``: Disable Relaxed Ordering.
        | ``1``: Require Relaxed Ordering; fail if unavailable.
        | ``2``: Auto-enable if available (default).

    * - | ``NCCL_IB_ECE_ENABLE``
        | Controls whether the IB/RoCE Verbs transport uses Enhanced Connection
          Establishment (ECE) when establishing queue-pair connections.
      - | ``0``: Disable ECE negotiation.
        | ``1``: Enable ECE negotiation when supported (default).

    * - | ``NCCL_IB_ADAPTIVE_ROUTING``
        | Controls whether the IB Verbs transport may use adaptive-routing-capable
          data transfers. Adaptive routing must also be enabled by the network
          configuration (for example, through ``NCCL_IB_SL``).
          When ``NCCL_IB_OOO_RQ=0``, only messages larger than
          ``NCCL_IB_AR_THRESHOLD`` use adaptive routing.
      - | ``0``: Disable adaptive routing.
        | ``1``: Enable adaptive routing.
        | Default: ``1`` for InfiniBand links; ``0`` for RoCE links.

    * - | ``NCCL_IB_AR_THRESHOLD``
        | Threshold above which RCCL sends IB data in a separate message that
          can leverage adaptive routing. Setting it above ``NCCL_BUFFSIZE``
          disables adaptive routing completely.
      - | Integer byte size. Default: ``8192``.

    * - | ``NCCL_IB_RETURN_ASYNC_EVENTS``
        | Controls whether the ``net_ib`` transport escalates fatal IB
          asynchronous events into communication termination.
      - | ``0``: Do not stop IB communications on fatal async events.
        | ``1``: Stop IB communications after fatal async events (default).

    * - | ``NCCL_IB_PKEY``
        | Sets the InfiniBand P_Key table index used by RCCL's ``net_ib``
          transport when bringing IB queue pairs to INIT.
      - | Integer P_Key table index. Default: ``0``.

    * - | ``NCCL_IB_DEVICE_PCI_ORDER``
        | Controls whether the IB transport sorts discovered IB devices by PCI
          path before assigning stable device and realPort ordering.
      - | ``0``: Disable PCI-order sorting.
        | Non-zero: Enable PCI-order sorting (default: ``1``).

    * - | ``NCCL_IB_MERGE_VFS``
        | Controls whether the ``net_ib`` transport merges virtual functions
          (VFs) of the same physical NIC when normalizing and comparing PCI
          paths.
      - | ``0``: Disable VF merging.
        | ``1``: Enable VF merging (default).

    * - | ``NCCL_IB_DATA_DIRECT``
        | Controls mlx5 Data Direct DMA Interface probing for the ``net_ib``
          transport.
      - | ``0``: Disable Data Direct probing (default).
        | ``1``: Enable probing; when detected, expose only the data-direct NIC
          view.
        | ``2``: Enable probing; keep both the regular and data-direct NIC views.

    * - | ``NCCL_IB_MQP_RETRY_CNT``
        | Controls how many times the InfiniBand ``ibv_modify_qp`` wrapper
          retries after a retryable failure. Total attempts = value + 1.
      - | Integer value. Default: ``34``.

    * - | ``NCCL_IB_MQP_RETRY_SLEEP_MSEC``
        | Controls the sleep interval, in milliseconds, between
          ``ibv_modify_qp`` retry attempts.
      - | Integer milliseconds. Default: ``100``.

    * - | ``NCCL_IB_MQP_RETRY_ALL``
        | Controls how broadly RCCL retries ``ibv_modify_qp`` failures.
      - | ``0``: Retry only ``ETIMEDOUT`` failures (default).
        | ``1``: Retry all nonzero ``ibv_modify_qp`` failures.

    * - | ``NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS``
        | Controls whether the ``net_ib`` receive side pre-posts receive work
          requests or posts them on-demand. Forced to ``1`` when IB resiliency
          or ``NCCL_IB_OOO_RQ`` is enabled.
      - | ``0``: Post receive work requests on-demand (default).
        | ``1``: Pre-post receive work requests.

    * - | ``NCCL_IB_RESILIENCY_PORT_FAILOVER``
        | Controls the RCCL-original ``net_ib`` port-failover resiliency
          subsystem. When enabled, RCCL creates a resiliency context so
          failed-device/QP handling and port recovery logic can continue
          communication after a single IB port failure.
      - | ``0``: Disabled (default).
        | ``1``: Enabled.

    * - | ``NCCL_IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS``
        | Sets the maximum number of IB port-failover probing attempts allowed
          for a failed send request.
      - | Integer value. Default: ``1``.

    * - | ``NCCL_IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY``
        | Controls the delay, in milliseconds, before port-failover logic posts
          a probe after a send request encounters an error.
      - | Integer milliseconds. Default: ``10``.

    * - | ``NCCL_IB_RESILIENCY_PORT_RECOVERY``
        | Controls whether the InfiniBand/RoCE port-recovery subsystem is
          enabled for the resiliency path.
      - | ``0``: Disable port recovery (default).
        | Non-zero: Enable port recovery.

    * - | ``NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_TIMEOUT``
        | Controls the ACK wait timeout, in milliseconds, for the port-recovery
          protocol.
      - | Integer milliseconds. Default: ``5000``.

    * - | ``NCCL_IB_QP_SCHED_ENABLE``
        | Master enable for the ``net_ib_cast`` queue-pair (QP) scheduler.
          By default (``-1``) it is enabled when ``NCCL_NET=IB-CAST`` is set,
          and disabled on AINIC otherwise. Also readable as
          ``RCCL_IB_QP_SCHED_ENABLE``.
      - | ``-1`` or unset: automatic behavior (default).
        | ``0``: Disable the QP scheduler.
        | Any other value: Enable the QP scheduler.

    * - | ``NCCL_IB_QP_SCHED_LOG_INTERVAL``
        | Controls how often the ``net_ib_cast`` QP scheduler writes
          scheduler-state log snapshots. Works only when the scheduler is
          enabled and ``RCCL_IB_QP_SCHED_LOG_PATH`` is set.
          Also readable as ``RCCL_IB_QP_SCHED_LOG_INTERVAL``.
      - | Integer, microseconds, ``1``–``60000000``. Default: ``1000000`` (1 s).

    * - | ``NCCL_IB_QP_SCHED_RESET_INTERVAL``
        | Controls how often the ``net_ib_cast`` QP scheduler resets its
          accumulated per-QP RTT statistics.
          Also readable as ``RCCL_IB_QP_SCHED_RESET_INTERVAL``.
      - | Integer, milliseconds. ``0``: Never reset. Default: ``60000``.

    * - | ``NCCL_IB_QP_SCHED_SPLIT_DATA_MIN``
        | Sets the minimum per-QP payload size for the ``net_ib_cast`` QP
          scheduler to split a transfer across multiple QPs. Only effective
          when the scheduler is enabled and ``NCCL_IB_SPLIT_DATA_ON_QPS=1``.
          Also readable as ``RCCL_IB_QP_SCHED_SPLIT_DATA_MIN``.
      - | Integer > ``0``, in bytes. Default: ``65536``.

    * - | ``NCCL_IB_QP_SCHED_UPDATE_INTERVAL``
        | Controls the minimum interval between recomputations of QP weights
          from RTT samples in the ``net_ib_cast`` QP scheduler.
          Also readable as ``RCCL_IB_QP_SCHED_UPDATE_INTERVAL``.
      - | Integer, microseconds, ``1``–``60000000``. Default: ``50``.

    * - | ``NCCL_IB_QP_SCHED_WRR_ENABLE``
        | Unsupported. Weighted round-robin QP selection is always used when the
          IB-CAST QP scheduler is enabled. Value ``0`` has no effect.
          Also readable as ``RCCL_IB_QP_SCHED_WRR_ENABLE``.
      - | ``0`` or ``1``. Default: ``1``. ``0`` has no effect.

    * - | ``RCCL_IB_SPLIT_DATA_THRESHOLD``
        | Applies to the IB plugin only (``net_ib_cast`` does not read it).
          Active when a connection uses more than one QP — with NIC Fusion or
          when ``NCCL_IB_SPLIT_DATA_ON_QPS=1`` and
          ``NCCL_IB_QPS_PER_CONNECTION > 1``. Messages smaller than this
          threshold are sent on a single QP chosen round-robin; larger messages
          are split across QPs.
      - | Integer >= ``0``, in bytes (default: ``128``).
        | ``N``: Split only when message size >= N bytes.

    * - | ``RCCL_AINIC_ROCE``
        | Controls which NICs are counted during AINIC detection. AINIC mode is
          used when the primary NIC type (by majority) is AINIC.
          ``NCCL_NET=ROCM-IB`` bypasses this variable entirely.
      - | ``-1``: Auto-detect (default).
        | ``0``: Ignore AINIC devices.
        | ``1``: Count only AINIC devices.

    * - | ``RCCL_IB_QPS_PER_P2P``
        | Overrides the number of QPs per device for P2P (send/recv)
          connections; collective connections keep
          ``NCCL_IB_QPS_PER_CONNECTION``. Applies to both IB and IB-CAST.
      - | ``0``: Use ``NCCL_IB_QPS_PER_CONNECTION`` (default).
        | ``N > 0``: N QPs per device.

    * - | ``RCCL_IB_P2P_DISABLE_CTS``
        | IB-CAST on AINIC only. When CTS offload is active, controls whether
          P2P connections use it. When ``1``, P2P connections do not use CTS
          offload.
      - | ``0`` or ``1``. Default: ``1``.

    * - | ``NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME``
        | Selects how the ``net_ib`` receiver matches incoming completions to
          receive requests. Forced to ``BY_ID`` when ``NCCL_IB_OOO_RQ`` is
          enabled or ``NCCL_IB_RESILIENCY_PORT_FAILOVER=1``.
      - | ``0`` (``BY_INDEX``): match by request index (default).
        | ``1`` (``BY_ID``): match by request ID.

    * - | ``NCCL_PXN_C2C``
        | Allows PXN routing through a C2C link to reach a NIC attached to a
          peer GPU. The C2C path is NVIDIA-specific and is not currently
          applicable on AMD hardware.
      - | ``0``: Disabled (default).
        | ``1``: Parsed but has no effect on AMD.

    * - | ``NCCL_SOCKET_IFNAME``
        | Specifies which IP interfaces to use for communication.
      - | Interface prefix string or comma-separated list.
          Prefix with ``^`` for exclusion, ``=`` for exact match.
        | Example: ``eth`` (all eth interfaces), ``=eth0`` (exact match).

    * - | ``NCCL_SOCKET_FAMILY``
        | Forces IPv4/IPv6 interface selection.
      - | ``AF_INET``: Force IPv4.
        | ``AF_INET6``: Force IPv6.
        | Unset: Use first available.

    * - | ``NCCL_SOCKET_NTHREADS``
        | Specifies how many socket helper threads RCCL uses per connection.
          The product of ``NCCL_SOCKET_NTHREADS`` and ``NCCL_NSOCKS_PERTHREAD``
          cannot exceed 64.
      - | ``1``–``16``. Default: ``1`` (generic), ``2`` (AWS),
          ``4`` (Google gVNIC).

    * - | ``NCCL_NSOCKS_PERTHREAD``
        | Specifies the number of sockets opened by each socket helper thread.
          The product of ``NCCL_SOCKET_NTHREADS`` and ``NCCL_NSOCKS_PERTHREAD``
          cannot exceed 64.
      - | Positive integer. Default: ``1`` (generic), ``8`` (AWS).

    * - | ``NCCL_SOCKET_POLL_TIMEOUT_MSEC``
        | Sets a timeout in milliseconds for socket polling during bootstrap.
          When non-zero, RCCL polls for the specified number of milliseconds
          before retrying.
      - | ``0``: No polling, continuous retry (default).
        | Non-zero: Poll for N milliseconds between retries.

    * - | ``NCCL_SOCKET_RETRY_CNT``
        | Specifies how many times RCCL retries establishing a socket connection
          after ``ETIMEDOUT``, ``ECONNREFUSED``, or ``EHOSTUNREACH``.
      - | Positive integer. Default: ``34``.

    * - | ``NCCL_SOCKET_RETRY_SLEEP_MSEC``
        | Specifies the number of milliseconds RCCL waits before retrying after
          the first socket connection error. For subsequent errors, the wait
          scales linearly with the retry count.
      - | Positive integer, in milliseconds. Default: ``100``.

    * - | ``RCCL_SOCKET_LINGER``
        | Specifies the socket linger time (in seconds) for sockets created via
          ``ncclSocketInit``. See the UNIX ``SO_LINGER`` socket option for
          details.
      - | ``-1`` (default) or a non-negative integer (seconds).

    * - | ``RCCL_SOCKET_REUSEADDR``
        | Controls reuse of local addresses on sockets created via
          ``ncclSocketInit``. See the UNIX ``SO_REUSEADDR`` socket option for
          details.
      - | ``0``: Disable address reuse (default).
        | ``1``: Enable address reuse.

    * - | ``NCCL_OOB_NET_ENABLE``
        | Controls whether RCCL uses the net plugin transport for bootstrap
          out-of-band communication during communicator initialization, instead
          of the socket bootstrap path.
      - | ``-1``: Auto-select when ``nranks >= NCCL_BOOTSTRAP_BIDIR_THRESHOLD``.
        | ``0``: Use socket bootstrap (default).
        | ``1``: Force net OOB bootstrap.

    * - | ``NCCL_OOB_NET_IFNAME``
        | Specifies which RCCL net interfaces to use for out-of-band bootstrap
          communication when ``NCCL_OOB_NET_ENABLE`` is active. See
          ``ibv_devinfo -l`` for available interfaces.
      - | Comma-separated interface list. Use ``:`` to specify ports.
          Prefix with ``^`` to exclude; use ``=`` for exact matches.
        | Default: unset (uses device 0).

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

    * - | ``NCCL_NET_MERGE_LEVEL``
        | Sets the maximum topological distance between two NICs that can be
          merged into a single logical device. NICs farther apart than this
          level are left separate.
      - | ``LOC``: Same device only, which disables merging.
        | ``PORT``: Two ports of the same NIC (default).
        | ``PIX``: Under the same PCIe switch.
        | ``PXB``: Multiple PCIe bridges, without crossing the PCIe host bridge.
        | ``P2C``, ``PXN``: Accepted, with the same effect as ``PXB`` for NIC
          pairs.
        | ``PHB``: Under the same CPU socket.
        | ``SYS``: Anywhere in the node, including across NUMA nodes.

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

    * - | ``NCCL_NET_SHARED_BUFFERS``
        | Allows shared buffers for inter-node point-to-point communication,
          using a single large pool for all remote peers so buffer memory does
          not scale linearly with the number of peers.
      - | ``0``: Disable shared buffers.
        | ``1``: Enable shared buffers (default).

    * - | ``NCCL_NET_SHARED_COMMS``
        | Reuses the same network connections in the context of PXN so messages
          can be aggregated across shared network communication paths.
      - | ``0``: Disable shared network communicators.
        | ``1``: Enable shared network communicators (default).

    * - | ``NCCL_NVB_DISABLE``
        | Disables intra-node communication through XGMI via an intermediate
          GPU/device hop.
      - | ``0``: Enable XGMI intermediate-hop (default).
        | ``1``: Disable XGMI intermediate-hop.

    * - | ``NCCL_P2P_PXN_LEVEL``
        | Controls when RCCL uses PXN-style relay routing for send/receive
          operations.
      - | ``0``: Disable PXN relay routing for send/receive.
        | ``1``: Enable only when the destination-preferred NIC is not
          reachable through PCI switches.
        | ``2``: Always use PXN (default).

    * - | ``NCCL_PXN_DISABLE``
        | Disables inter-node communication through a non-local NIC by relaying
          traffic via an intermediate GPU on the same node. When unset, RCCL
          auto-enables PXN at larger scales (>= 64 ranks on ``gfx942``/MI300X;
          >= 32 ranks on ``gfx950``/MI350X–MI355X).
      - | ``0``: Enable the PXN relay path.
        | ``1``: Disable the PXN relay path.
        | Default: auto (scale-dependent).

    * - | ``NCCL_NCHANNELS_PER_NET_PEER``
        | Sets the number of channels used per network (remote) peer.
        | This overrides the value of the ``nChannelsPerNetPeer`` field in
        | ``ncclConfig_t``. When neither this variable nor the config field is
        | set, RCCL auto-tunes the per-peer channel count based on the
        | available NIC bandwidth and rank count.
      - | Integer value, ``1`` to ``MAXCHANNELS`` (default: unset/auto-tuned).
        | Values ``<= 0`` are ignored and a warning is logged.

    * - | ``NCCL_RINGS``
        | Defines a user-specified custom ring topology for RCCL. When set,
          RCCL bypasses automatic ring-topology search and parses one or more
          explicit ring channels from the string.
      - | Whitespace-separated list of 0-based GPU indices for the current
          communicator; rings/channels are separated by ``|``.
        | Optional ``N<number>`` tokens specify network interfaces at ring
          endpoints.
        | A ring must contain exactly the communicator's GPU count with no
          duplicate or out-of-range GPU indices; invalid rings are ignored.
        | Default: unset (automatic topology detection).

    * - | ``RCCL_TREES``
        | Defines custom tree topology.
      - | Tree topology specification string.
        | Default: unset.

    * - | ``NCCL_RINGS_REMAP``
        | Provides an RCCL-only ring remapping/topology specification string
          for Rome 4P2H topology matching.
      - | Ring topology/remapping specification string in the ``ringBase``
          format consumed by ``parseGraph`` via ``parseRome4P2H``.
        | Default: unset (uses built-in ``romeTopoModels[i].ringBase``).

    * - | ``RCCL_MODEL_MATCHING_DISABLE``
        | Skips all preset topology matching so RCCL uses its generic graph
          search instead of hand-tuned ring or tree topologies.
      - | ``0``: Match presets (default).
        | Non-zero: Skip matching.

    * - | ``RCCL_MODEL_REVERSAL_DISABLE``
        | Disables per-node reversal of ring channels from a predefined
          topology model or ``NCCL_RINGS``, so every node uses the ring as
          written.
      - | ``0``: Odd nodes reverse (default).
        | Non-zero: No reversal.

    * - | ``RCCL_DISABLE_RAIL_TREES``
        | Skips rail-optimized tree strings (``treeRail``) defined by preset
          topology models during BALANCED_TREE graph search, so RCCL falls
          back to ``treeBase`` and then to trees derived from ``ringBase``.
      - | ``0``: Use rail trees when defined (default).
        | Non-zero: Disable.

    * - | ``RCCL_DUMP_ROME_MODEL_FILE``
        | Writes the detected topology as a C ``rcclRomeModel`` initializer
          for adding or debugging preset models.
      - | Writable file path.
        | Default: unset (no dump).

    * - | ``RCCL_DISABLE_NET_OVERRIDE``
        | Troubleshooting option for 8-GPU, 4-NIC systems that match a
          built-in RCCL topology model. Set to ``1`` if the assumed GPU-NIC
          proximity gives wrong NIC placement or poor performance on your
          system layout.
      - | ``0``: Enable the PCIe proximity assumption (default).
        | ``1``: Disable it.

GPU memory and RDMA
===================

The GPU memory and RDMA environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_DMABUF_ENABLE``
        | Enables ROCm DMA-buf-based GPU-memory registration for RCCL network
          transports. Only takes effect when the ROCr runtime reports DMA-buf
          support, the selected network plugin supports ``regMrDmaBuf``, and
          kernel DMA-BUF/P2PDMA checks pass. The DMA-buf path is only taken if
          ``peermem`` is not available.
      - | ``0``: Disable DMA-buf registration.
        | ``1``: Enable DMA-buf registration (default).

    * - | ``NCCL_NET_GDR_READ``
        | Controls whether RCCL uses ROCm GPUDirect RDMA reads for network
          send operations when the GPU-NIC distance is within
          ``NCCL_NET_GDR_LEVEL``.
      - | ``0``: Disable network GDR reads for sends.
        | ``1``: Force network GDR reads when topology and device support allow.
        | Default: enabled when GDR is supported.

    * - | ``RCCL_FORCE_ENABLE_GDRDMA``
        | Debug only. When set to ``1``, RCCL assumes the GPU-NIC RDMA module
          is loaded and skips the ``peermem`` module scan, the Hyper-V
          blocklist, and the test memory-registration probe. The DMA-buf check
          still runs.
      - | ``-1``: Auto-detect (default).
        | ``1``: Force on.

    * - | ``RCCL_FORCE_ENABLE_DMABUF``
        | Forces RCCL to select the DMA-buf GPU Direct RDMA path instead of
          ``peermem``. When set, RCCL registers GPU buffers through
          ``regMrDmaBuf`` even if ``peermem`` is available. If DMA-buf is not
          actually available, GPU Direct RDMA is disabled for that device.
      - | ``0``: Normal GDR selection (``peermem`` preferred, default).
        | ``1``: Force DMA-buf; skip ``peermem``.

    * - | ``RCCL_NET_CONTIGUOUS_MEM``
        | Allocates GDR (GPU-side) network buffers with
          ``hipDeviceMallocContiguous`` instead of ``hipDeviceMallocUncached``.
          Has no effect when cuMem is enabled.
      - | ``0`` (default) or ``1``.

    * - | ``RCCL_NET_HDP_FLUSH``
        | Debug option. When a GDR receive needs a flush, writes the GPU HDP
          flush register after each received Simple-protocol chunk. Only
          effective if the HDP register was mapped for the connection.
      - | ``0`` (default) or ``1``.

    * - | ``RCCL_GFX9_CHEAP_FENCE_OFF``
        | Selects the memory fence that LL128 and Simple primitives use after
          the sender GPU finishes a batch of writes. ``0`` is
          architecture-tuned auto mode (cheap fence on for ``gfx942``/MI300 and
          ``gfx1250``/MI450; off for ``gfx950``/MI350 and others).
      - | ``0``: Architecture-tuned auto mode (default).
        | ``1``: Force ``__threadfence_system()``.
        | ``2``: Force cheap fence on (including on ``gfx950``).

    * - | ``HSA_FORCE_FINE_GRAIN_PCIE``
        | Controls an AMD ROCm/HSA runtime setting for fine-grain PCIe
          memory/peer-to-peer behavior on PCIe-connected GPU systems. RCCL
          warns during initialization if this is not set to ``1`` on affected
          systems, as missing it can lead to low RCCL performance, system
          instability, or hangs.
      - | ``1``: Recommended value for affected AMD PCIe systems.
        | Any other value: Treated as missing; triggers the init warning.
        | Default: unset.

P2P and shared memory
=====================

The peer-to-peer and shared-memory environment variables for RCCL are
collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_P2P_DISABLE``
        | Disables RCCL peer-to-peer (P2P) transport for direct GPU-to-GPU
          communication over XGMI/Infinity Fabric or PCIe.
      - | ``1``: Disable GPU-to-GPU P2P communication.
        | Default: unset (P2P enabled).

    * - | ``NCCL_P2P_DIRECT_DISABLE``
        | Controls direct access to user buffers through P2P between GPUs of
          the same process.
      - | ``0``: Allow direct same-process user-buffer P2P access (default).
        | ``1``: Disable.

    * - | ``NCCL_P2P_LL_THRESHOLD``
        | Sets the maximum per-channel message size, in bytes, for which RCCL
          uses the LL protocol for point-to-point operations before switching
          to Simple.
      - | Integer byte value. Default: ``4096``.

    * - | ``NCCL_P2P_NET_CHUNKSIZE``
        | Controls the chunk size, in bytes, for network point-to-point
          ``ncclSend``/``ncclRecv`` operations. When unset, RCCL auto-selects
          based on architecture and rank count.
      - | Integer byte value. Default: ``131072`` (auto-tuned on ``gfx942``
          and ``gfx950``).
        | Auto-tuned values: 524288 on ``gfx942``/MI300X with >= 64 ranks;
          262144 on ``gfx950``/MI350X–MI355X with 16–31 ranks; 524288 on
          ``gfx950``/MI300X with >= 32 ranks.

    * - | ``NCCL_SHM_DISABLE``
        | Disables RCCL shared-memory (SHM) transport for intra-node
          communication. If SHM is disabled, RCCL falls back to the network
          transport.
      - | ``0``: Keep SHM transport enabled (default).
        | ``1``: Disable SHM transport.

    * - | ``NCCL_IPC_USE_ABSTRACT_SOCKET``
        | Determines whether the Linux Abstract Socket mechanism is used when
          creating Unix Domain Sockets (UDS) for intra-node IPC handle
          exchange.
      - | ``0``: Do not use abstract sockets.
        | ``1``: Use abstract sockets (default).

    * - | ``NCCL_ALLOC_P2P_NET_LL_BUFFERS``
        | Controls whether communicators allocate dedicated LL/LL128 buffers
          for all P2P network connections. Allocating for all P2P connections
          with high numbers of ranks results in high memory overhead.
      - | ``0``: Disable LL/LL128 buffer allocation for P2P connections (default).
        | ``1``: Enable.

    * - | ``RCCL_ENABLE_INTRANET``
        | Testing and tuning option for single-node communicators. Keeps
          network devices in the topology of a single-node communicator.
          Required for intra-node ring channels over the NIC on fully
          XGMI-connected systems, together with ``RCCL_P2P_NET_DISABLE=0``.
      - | ``1``: Enable.
        | Any other value: Disabled (default).

    * - | ``RCCL_P2P_NET_DISABLE``
        | Controls same-node traffic over the NIC. With ``0``, on Rome 4P2H
          topologies where GDR is available and GPUs are not fully
          XGMI-connected, same-node send/recv larger than
          ``RCCL_P2P_NET_THRESHOLD`` go through the network.
      - | ``0``: Allow same-node NIC traffic.
        | Non-zero: Disable same-node NIC traffic (default: ``1``).

    * - | ``RCCL_P2P_NET_THRESHOLD``
        | Size cutoff for same-node send/recv over the network.
          See ``RCCL_P2P_NET_DISABLE``.
      - | Integer, bytes. Default: ``131072`` (128 KiB).

    * - | ``NCCL_WIN_ENABLE``
        | Enables memory window registration in RCCL.
      - | ``0``: Disable window memory registration.
        | ``1``: Enable window memory registration (default).

RCCL-specific optimizations
============================

The RCCL-specific optimization environment variables are collected in
the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``RCCL_THREADS_PER_BLOCK``
        | Overrides the per-CTA thread count for collectives, replacing the
          per-algorithm and per-protocol defaults. The requested value is
          rounded up to a multiple of the GPU warp size and capped at 256.
      - | Default: ``-1`` (RCCL picks per algorithm and protocol).
        | On 64-wide warp GPUs (``gfx908``, ``gfx90a``, ``gfx942``,
          ``gfx950``): effective values are ``192`` and ``256``.
        | On 32-wide warp GPUs (``gfx1200``, ``gfx1201``, ``gfx1250``):
          ``96``, ``128``, ``160``, ``192``, ``224``, and ``256``.

    * - | ``RCCL_UNROLL_FACTOR``
        | Selects the device-kernel unroll factor as a base-2 exponent.
          Invalid values or values without a generated device function table
          for the current architecture return ``ncclInvalidArgument``.
      - | ``0``–``5``, mapping to unroll 1, 2, 4, 8, 16, 32.
        | Default: ``-1`` (per-arch auto: ``gfx950`` uses 1 single-node and
          2 multi-node; ``gfx908`` and ``gfx942`` above 80 CUs use 2;
          ``gfx1250`` uses 32; others use 4).

    * - | ``RCCL_PIPELINE_ALL_DATA_TYPES``
        | Allows reduce/copy pipelining for data types other than
          ``bfloat16`` (``AllReduce``, ``Reduce``, ``ReduceScatter``) on
          ``gfx942`` only.
      - | ``0``: ``bfloat16`` only (default).
        | Non-zero: All types on ``gfx942``.

    * - | ``RCCL_SYM_MODEL``
        | Selects the symmetric-kernel cost model used to tune ``AllReduce``,
          ``AllGather``, and ``ReduceScatter``.
      - | ``0``: Default model.
        | ``1``: Cost model without ReduceScatter tuning.
        | Values outside 0 and 1 fall back to 0.

    * - | ``RCCL_DIRECT_ALLGATHER_DISABLE``
        | Disables the Direct AllGather path. Also requires ``gfx942`` or
          ``gfx950``, a rank count divisible by 8, and is skipped on AINIC.
      - | ``0``: Allowed (default).
        | Non-zero: Disable Direct AllGather.

    * - | ``RCCL_DIRECT_ALLGATHER_THRESHOLD``
        | Maximum message size in bytes for Direct AllGather. Setting this
          variable suppresses per-architecture auto-tuning. ``-1`` disables
          Direct AllGather.
      - | Integer bytes. Default: ``75497472`` (72 MiB).
        | ``-1``: Disable.

    * - | ``RCCL_DIRECT_REDUCE_SCATTER_DISABLE``
        | Disables Direct ReduceScatter. Requires ``gfx950``, PXN enabled,
          and a specific node count (2, 4, 8, or 16).
      - | ``0``: Allowed (default).
        | Non-zero: Disable.

    * - | ``RCCL_DIRECT_REDUCE_SCATTER_THRESHOLD``
        | Maximum message size, in bytes, for Direct ReduceScatter on
          ``gfx950`` when PXN is enabled. Values above 8 MiB are capped.
      - | ``0``–``8388608`` bytes. Default: ``8388608`` (8 MiB).

    * - | ``RCCL_HIERARCHICAL_ALLGATHER``
        | Enables hierarchical AllGather, which splits the operation across
          intra-node and inter-node sub-communicators. Requires 8 or more
          nodes and compact rank ordering.
      - | ``1``: Enabled when prerequisites are met (default).
        | ``0`` or any other value: Disabled.

    * - | ``RCCL_DISABLE_REDUCE_COPY_PIPELINING``
        | Forces non-pipelined device functions. In practice only changes
          behavior on ``gfx942``.
      - | ``0``: Allow pipelining (default).
        | Non-zero: Disable.

    * - | ``RCCL_DDA_ENABLE``
        | Enables the DDA (Direct Device Access) algorithms for ``AllGather``,
          ``AllReduce``, ``ReduceScatter``, and ``AllToAll``. Supported on
          ``gfx942``, ``gfx950`` (IPC-based, single-node, 8 ranks), and
          ``gfx1250`` (fabric-based, scale-up domain).
      - | ``0``: Disabled.
        | ``1``: Enabled (default).

    * - | ``RCCL_DDA_THRESHOLD``
        | Maximum message size in bytes for DDA on ``gfx950`` and ``gfx1250``
          for ``AllGather``, ``AllReduce``, and ``ReduceScatter``. ``gfx942``
          uses a fixed 8 MiB threshold. ``AllToAll`` uses a fixed 4 MiB
          threshold on all architectures.
      - | Integer bytes. Default: ``134217728`` (128 MiB).
        | ``0``: Disable DDA for collectives that read this variable.

    * - | ``RCCL_ROCSHMEM_ENABLE``
        | Enables rocshmem-based ``AllToAll`` and ``AllToAllv`` when RCCL is
          built with rocshmem support.
      - | ``0``: Disabled (default).
        | ``1``: Enabled.

    * - | ``RCCL_ROCSHMEM_THRESHOLD``
        | Determines the message size threshold for using rocshmem-based
          ``AllToAll`` and ``AllToAllv``.
      - | Integer bytes. Default: ``262144`` (256 KiB).

    * - | ``RCCL_CE_BATCH_ASYNC_ENABLE``
        | Controls use of ``hipMemcpyBatchAsync`` for Copy Engine (CE)
          collective peer copies.
      - | ``-2``: Auto-enable if the driver supports it (default).
        | ``0``: Force off.
        | ``1``: Force on (warns and disables if unsupported).

    * - | ``RCCL_CE_MULTI_STREAMS``
        | Number of parallel HIP streams used for CE peer device-to-device
          copies. Copies are distributed round-robin to improve SDMA
          concurrency.
      - | ``0``: Use the collective's main stream (default).
        | ``1``–``8``: Number of parallel streams.

    * - | ``RCCL_CTS_OFFLOAD_ENABLED``
        | Enables AINIC CTS offload and the inline-data path it depends on.
          Only read inside the IbCastAinicRoce branch of device init. CTS is
          disabled entirely when CAST QP scheduling or IB port
          failover/recovery is active.
      - | ``-1``: Auto-enable on AINIC (default).
        | ``0``: Force off.
        | Non-zero: Same as ``-1``.

    * - | ``RCCL_INJECT_FAULTS``
        | Testing only. Bitmask of faults to inject. Requires RCCL built with
          ``FAULT_INJECTION`` (ON by default); otherwise a warning is printed
          and the value is ignored.
      - | Bitmask. Default: ``0``.
        | ``0x1``: ``RANDOM_DELAY_ON_WARP_START``.

WarpSpeed
=========

WarpSpeed is a high-performance algorithm for ``AllReduce``, ``AllGather``,
and ``ReduceScatter`` on supported AMD GPU configurations. WarpSpeed
environment variables have no effect unless RCCL is built with
``ENABLE_WARP_SPEED``, which is disabled by default and only available on
``gfx950`` targets.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``RCCL_WARP_SPEED_AUTO``
        | Enables WarpSpeed automatically at communicator init and then per
          operation when all prerequisites hold: ``gfx950`` GPU, single-node
          communicator, more than 128 CUs (SPX mode), Ring algorithm for
          ``AllReduce``, ``AllGather``, or ``ReduceScatter``, and message size
          at or above the matching threshold. Set to ``0`` as the documented
          workaround when RCCL warns that the communicator exceeds the
          WarpSpeed channel limit.
      - | ``0``: Disable auto selection.
        | Non-zero: Auto mode enabled (default: ``1``).

    * - | ``RCCL_WARP_SPEED_FORCE_ENABLE``
        | Forces WarpSpeed on unconditionally, bypassing every auto-mode
          precondition and all message-size thresholds. Forces the algorithm to
          RING and the unroll factor to 2 unless ``RCCL_UNROLL_FACTOR`` is set.
          Returns ``ncclInvalidArgument`` when the communicator exceeds the
          supported channel count.
      - | ``0``: Leave selection to auto mode (default).
        | Any value > 0: Force WarpSpeed on.

    * - | ``RCCL_WARP_SPEED_AG_THRESHOLD``
        | Minimum ``AllGather`` message size, in bytes, for WarpSpeed to engage
          in auto mode.
      - | Integer bytes. Default: ``134217728`` (128 MiB).

    * - | ``RCCL_WARP_SPEED_RS_THRESHOLD``
        | Minimum ``ReduceScatter`` message size, in bytes, for WarpSpeed to
          engage in auto mode. The size tested is the total message size, not
          the per-rank share.
      - | Integer bytes. Default: ``2147483648`` (2 GiB).

    * - | ``RCCL_WARP_SPEED_CU_COUNT``
        | Sets how many CUs WarpSpeed occupies; the channel count becomes
          CU count × warps per block. Ignored if ``NCCL_THREAD_THRESHOLDS``,
          ``NCCL_MAX_NCHANNELS``, or ``NCCL_MIN_NCHANNELS`` is set.
      - | Non-negative integer. Default: ``0`` (derive channels from the normal
          tuning path).

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
        | Controls launch-blocking behavior used by RCCL runtime wrappers.
          RCCL on AMD still reads the environment variable named
          ``CUDA_LAUNCH_BLOCKING`` and treats it as a boolean launch-blocking
          switch.
      - | Unset or ``0``: Non-blocking launches.
        | ``1`` or any other non-zero/non-empty value: Blocking launches.

    * - | ``NCCL_COMM_ID``
        | Provides the bootstrap communicator address explicitly. When set,
          RCCL parses ``NCCL_COMM_ID`` as ``<ipv4>:<port>``,
          ``[<ipv6>]:<port>``, or ``<hostname>:<port>``, uses that address for
          bootstrap setup, and rejects ``ncclCommGetUniqueId`` calls made for
          communicator-grow operations while the variable is set.
      - | Unset: use normal bootstrap unique-id generation and automatic
          interface selection.
        | Set: socket address string in one of these forms:
          ``<ipv4>:<port>``, ``[<ipv6>]:<port>``, or ``<hostname>:<port>``.
          Invalid strings return ``ncclInvalidArgument``.

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

    * - | ``NCCL_IGNORE_CPU_AFFINITY``
        | Controls whether RCCL honors the job's supplied CPU affinity when
          deriving the CPU affinity for a rank. When enabled, RCCL ignores the
          process affinity and uses only the GPU-local CPU affinity.
      - | ``0``: Intersect job-supplied affinity with GPU-local CPU affinity
          (default).
        | ``1``: Ignore job-supplied affinity.

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
        | output feeds into the next. On AMD/HIP builds, enabling this causes
        | operations to execute in serial host-launch order.
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
