.. meta::
   :description: RCCL release notes covering NCCL API compatibility, new features, resolved issues, and known issues for each ROCm release version.
   :keywords: RCCL, NCCL compatibility, release notes, ROCm 7, changelog, compatibility matrix, gfx942, gfx950, MI300X, MI350, AllReduce, ncclCommGrow, symmetric memory

.. _rccl-release-notes:

******************
RCCL release notes
******************

RCCL (ROCm Collective Communication Library) is AMD's implementation of the
NCCL (NVIDIA Collective Communications Library) API. Each RCCL release targets
compatibility with one or more NCCL versions. Applications built against NCCL
can run on AMD hardware with no source-level changes.

The table below maps each ROCm release to the corresponding RCCL version and
the NCCL version or versions it is compatible with.

RCCL to NCCL compatibility matrix
===================================

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - ROCm version
     - RCCL version
     - Compatible NCCL versions
   * - 7.14.0
     - 2.30.4
     - 2.30.4, 2.29.7, 2.28.9
   * - 7.13
     - 2.28.3
     - 2.28.3 (implicit)
   * - 7.12
     - 2.28.3
     - 2.28.3 (implicit)
   * - 7.11
     - 2.28.3
     - 2.28.3
   * - 7.2.0
     - 2.27.7
     - 2.27.7 (implicit)
   * - 7.1.1
     - 2.27.7
     - 2.27.7 (implicit)
   * - 7.1.0
     - 2.27.7
     - 2.27.7
   * - 7.0.0
     - 2.26.6
     - 2.23.4, 2.24.3, 2.25.1, 2.26.6
   * - 6.4.2
     - 2.22.3
     - 2.22.3 (implicit)
   * - 6.4.1
     - 2.22.3
     - 2.22.3 (implicit)
   * - 6.4.0
     - 2.22.3
     - 2.22.3
   * - 6.3.1
     - 2.21.5
     - 2.21.5 (implicit)
   * - 6.3.0
     - 2.21.5
     - 2.21.5
   * - 6.2.1
     - 2.20.5
     - 2.20.5 (implicit)
   * - 6.2.0
     - 2.20.5
     - 2.20.5, 2.19.4
   * - 6.1.0
     - 2.18.6
     - 2.18.6
   * - 6.0.0
     - 2.18.3
     - 2.18.3
   * - 5.7.0
     - 2.17.1-1
     - 2.17.1-1
   * - 5.6.0
     - 2.16.2
     - 2.16.2
   * - 5.5.0
     - 2.15.5
     - 2.15.5
   * - 5.4.0
     - 2.13.4
     - 2.13.4
   * - 5.3.0
     - 2.12.10
     - 2.12.10 (implicit)
   * - 5.2.3
     - 2.12.10
     - 2.12.10
   * - 5.2.0
     - 2.11.4
     - 2.11.4 (implicit)
   * - 5.1.0
     - 2.11.4
     - 2.11.4
   * - 5.0.0
     - 2.10.3
     - 2.10.3
   * - 4.5.0
     - 2.9.9
     - 2.9.9
   * - 4.3.0
     - 2.8.4
     - 2.8.4 (implicit)
   * - 4.2.0
     - 2.8.4
     - 2.8.4
   * - 4.1.0
     - 2.7.8
     - —
   * - 3.9.0
     - 2.7.8
     - —
   * - 3.8.0
     - 2.7.6
     - —
   * - 3.7.0
     - 2.7.6
     - —
   * - 3.6.0
     - 2.7.0
     - NCCL 2.6
   * - 3.5.0
     - 2.7.0
     - NCCL 2.6

----

RCCL 2.30.4 for ROCm 7.14.0
============================

This release adds compatibility with NCCL 2.30.4, 2.29.7, and 2.28.9, and
introduces a new Python API, symmetric-memory kernels for MI450, and a proxy
diagnostics framework.

Added
-----

- NCCL compatibility: 2.30.4, 2.29.7, 2.28.9.
- ``proxytrace`` profiler plugin and core proxy-diagnostics hooks
  (``RCCL_PROXYTRACE``).
- ``ncclBarrierSession`` LSA validation for barrier sessions.
- Symmetric-memory ReduceScatter kernel (``RailA2A_LsaLD``) on gfx942 and
  gfx950.
- Bias (accumulation) AllReduce on gfx1250 (MI450).
- Optimized scale-up ReduceScatter, AllGather, and AllToAll kernels.
- ROCProfiler-SDK (rocprofiler-sdk) coverage for ``ncclCommGrow`` and
  ``ncclCommGetUniqueId``.
- P2P batching auto-enabled for gfx950 combined with non-AINIC NICs.
- HIP/ROCm runtime version display in ``NCCL_DEBUG`` output.
- Python API bindings under ``bindings/nccl4py/``: a Cython-based interface to
  RCCL collectives, a ``cuda.core`` HIP shim for ROCm hosts, and RCCL-only
  wrappers for ``ncclAllReduceWithBias`` and ``ncclAllToAllv``.
- RCCL examples added to the repository.
- ``RCCL host API`` pull-in from NCCL 2.30.

Changed
-------

- WarpSpeed auto mode enabled for grow communicators.
- Hierarchical AllGather now enabled by default for multi-node. Algorithm
  selection logic was refactored.
- Legacy ``net_ib`` replaced with the ``net_ib`` implementation from NCCL 2.29.
- Bootstrap AllGather now uses the bidirectional ring (N/2 steps) by default
  on the socket out-of-band (OOB) path. ``NCCL_BOOTSTRAP_BIDIR_ALLGATHER``
  now defaults to ``1``; set it to ``0`` to revert to the unidirectional ring.
- ``NCCL_PXN_C2C`` remains default-off (``0``). Upstream NCCL defaults it to
  ``1`` since 2.28; the C2C proxy-to-proxy cross-node (PXN) routing path is
  NVIDIA-specific and does not apply to AMD hardware.

Removed
-------

- NPKit profiling support (``ENABLE_NPKIT`` build option, related headers,
  device and proxy instrumentation, ``--npkit-enable`` install flag). Use the
  profiler plugin API instead.
- Kernel COLLTRACE support (``COLLTRACE`` build option, device-side collective
  trace buffers, debug kernel variants). The host latency profiler is unchanged.
- Legacy ``ENABLE_PROFILING`` device profiling (``PROFILE`` build option). Use
  the profiler plugin API instead.

Resolved issues
---------------

- Fixed ``ncclCommGrow`` channel-count divergence causing incorrect collective
  routing.
- Fixed ``ncclCommGrow`` hang when growing to an 8-rank single-node
  communicator.
- Fixed symmetric LDS under-reservation in legacy (non-device-linker) builds.
- Fixed LL128 protocol correctness for gfx1250 (MI450).
- Fixed XGMI topology mapping for multi-system (NPS) nodes.
- Fixed gfx950 collective hang from a tuner race condition.
- Fixed ``net_ib_cast``: CTS offload path now gated on per-connection state.
- Fixed ``net_ib``: non-fatal Isend CTS no-match no longer flagged as fatal.
- Fixed acquire-tail polling for gfx950 P2P host staging.
- Fixed LDS overflow in device-linker builds.
- Fixed ``ncclCommFree`` to release symmetric window objects automatically
  (NCCL 2.29.7 defect).
- Fixed static build failing with ``install(EXPORT "rccl-targets" ...)`` when
  ``fmt`` is fetched via ``FetchContent``.
- Fixed proxy channel staging buffers ignoring GDR mode selection on HIP <
  7.12 builds. Workaround for affected builds: ``NCCL_DMABUF_ENABLE=0``.
- Fixed RCCL initialization failing with ``Failed to find ROCm runtime
  library`` on runtime-only ROCm trees that have no unversioned
  ``libhsa-runtime64.so`` symlink. The ``RCCL_ROCR_PATH`` override is no
  longer needed and has been removed.

Known issues
------------

- On gfx90a (MI210, MI250, MI250X) with ROCm 7.13 or later, per-launch
  scratch-memory reclaim in the runtime degrades RCCL performance. Set
  ``HSA_NO_SCRATCH_RECLAIM=1`` to restore performance.
- Elastic-buffer support for GIN compiles on ROCm but is unverified on AMD
  hardware.

----

RCCL 2.28.3 for ROCm 7.13
==========================

This release adds CAST network transport for AMD AINIC hardware, a built-in
CSV tuner, and multi-node hierarchical AllGather for MI350.

Added
-----

- CAST network transport (``ncclNetCast`` / ``net_ib_cast``) for AMD AINIC
  (AMD Integrated Network Interface Controller) hardware.
- Built-in CSV tuner for runtime algorithm, protocol, and channel selection
  without rebuilds.
- Multi-node hierarchical AllGather for MI350, enabled by default for 8 or
  more nodes (64 MB threshold at 8 nodes, 128 MB above). Disable with
  ``RCCL_HIERARCHICAL_ALLGATHER=0``.
- Initial symmetric-memory kernel support on gfx942 and gfx950.
- ``RCCL_IB_SPLIT_DATA_THRESHOLD`` to split payload across multiple QPs/NICs
  in ``ncclIbMultiSend``.
- User Buffer and Graph Registration (``NCCL_NVLS_ENABLE`` / CUMEM) gated on
  Linux kernel version.
- Copy Engine (CE) collectives support.
- gfx1250 (MI450) GPU target support in RCCL and RCCL-Tests.
- Strix-Halo (gfx1151) tuning support.
- ``RCCL_IB_P2P_DISABLE_CTS`` to disable CTS (clear-to-send) offload for P2P
  connections on AINIC (defaults to ``1``, disabled).
- ``RCCL_CTS_INLINE_DATA`` merged into ``RCCL_CTS_OFFLOAD_ENABLED`` as a
  tri-state: ``-1`` (auto-enable on AINIC), ``0`` (force disable), ``1``
  (force enable).

Changed
-------

- Removed MSCCL and MSCCL++ collective integration. The legacy
  ``mscclLoadAlgo``, ``mscclRunAlgo``, and ``mscclUnloadAlgo`` APIs remain as
  no-ops for link compatibility.
- P2P batching disabled by default on MI350.
- AMD-SMI (``amdsmi_init``) disabled by default due to a concurrency issue;
  enable explicitly for ROCm 7.0 and above when the issue is resolved.
- ``RCCL_ENABLE_CONTEXT_TRACKING`` replaced by ``NCCL_LAUNCH_ORDER_IMPLICIT``
  for controlling launch-order tracking.
- Tuning log messages moved from ``NCCL_INIT`` to the ``NCCL_TUNING`` debug
  subsystem.
- 256 threads per block on gfx950 (previously 512).
- Algorithm set to Ring for Navi4x (gfx1100/gfx1101) AllReduce.

Resolved issues
---------------

- Fixed ``netOverride`` being skipped when rail-optimized trees are enabled.
- Fixed RCCL Inspector plugin teardown segfault/hang and collective-count
  correctness.
- Fixed ``ncclGroupSimulateEnd`` planner state leak and resource cleanup.
- Fixed validation errors with ``all_reduce_bias`` kernel on gfx950.
- Fixed ``--generate-sym-kernels`` option when used with ``--device-linker``.
- Fixed ``CUCHECK`` and ``CUCHECKGOTO`` macros to clear the HIP error state
  before returning.
- Fixed CTS-offload corner cases in ``net_ib_rocm`` and ``net_ib_cast``.

----

RCCL 2.28.3 for ROCm 7.12
==========================

This release adds AMD AINIC support, Direct ReduceScatter, and WarpSpeed for
single-node AllGather and ReduceScatter.

Added
-----

- gfx1151 (Strix-Halo) GPU target support.
- AMD AINIC support in the RCCL default internal network plugin.
- ``RCCL_P2P_SHIFT_SIZE`` environment variable for advanced P2P channel and
  part mapping tuning.
- Direct ReduceScatter for improved multi-node performance.
- WarpSpeed support for single-node AllGather and ReduceScatter.
- Virtual device enablement support.
- Navi4 (gfx1100) LL (low-latency) protocol enablement and tuning.
- AMD-SMI wrapper for firmware version queries (switched from rocm-smi to
  amd-smi).

Changed
-------

- GPU Direct RDMA (remote direct memory access) mode selection now prefers
  peermem over DMAbuf by default. ``NCCL_DMABUF_ENABLE`` now defaults to
  ``1``. Set ``RCCL_FORCE_ENABLE_DMABUF=1`` to force DMAbuf exclusively.
- P2P batching node-count cap removed; P2P batching now applies at all node
  counts.
- Default maximum channels set to 48 for gfx950 and MI350 multi-node
  collectives.
- ``NCCL_LAUNCH_ORDER_IMPLICIT`` replaces ``RCCL_ENABLE_CONTEXT_TRACKING``.

Resolved issues
---------------

- Fixed shutdown ordering race condition and use-after-free crash in proxy
  cleanup.
- Fixed DMAbuf support check failure (SWDEV-579889 / ROCM-2855).
- Fixed ``qpIndex`` selection in ``ncclIbIrecv`` for AINIC mode.
- Fixed per-device UD (unreliable datagram) map indexing for NIC fusion
  configurations.
- Fixed bfloat16 reduce kernel bug for ROCm 6.0 and later.
- Fixed memory leak in ``ncclCommInitRankFunc``.
- Fixed memory leaks (ROCM-1721, ROCM-1722).

Known issues
------------

- The upstream one-sided RMA (remote memory access) subsystem (``src/rma``) is
  unverified at scale on ROCm.
- The upstream Copy-Engine collective redesign is not adopted; RCCL retains its
  existing HIP Copy-Engine implementation.
- The Copy-Engine profiler path (``ncclProfiler_v6``) is not enabled; RCCL
  remains on ``ncclProfiler_v5``.
- GIN GDAKI host support is DOCA/Mellanox-specific and is unverified on AMD
  NICs.
- GIN GET and FLUSH (``iget``/``iflush``) are not implemented in the RCCL
  InfiniBand GIN proxy backend.
- Elastic-buffer support for GIN compiles on ROCm but is unverified on AMD
  hardware.

----

RCCL 2.28.3 for ROCm 7.11
==========================

This release adds NCCL 2.28.3 compatibility, a new ``ncclAllReduceWithBias``
API, a collective latency profiler, and WarpSpeed auto mode.

Known issues
------------

- AllToAllv and AllToAll for a single GPU is hanging.
- AllGather regression for small message sizes (less than 1 MB) due to the
  Direct algorithm.
- ROCTx feature needs to be verified.
- Profiler plugin needs to be verified.

Added
-----

- NCCL compatibility: 2.28.3.
- ``ncclAllReduceWithBias`` API for fused AllReduce with elementwise
  accumulation-bias operations.
- Collective latency profiler tool (``--latency-profiler`` in ``install.sh``).
- Dynamic pipelining for reduction collectives via the Simple protocol to
  improve single-node performance.
- LL128 protocol for gfx942 with 4-NIC configurations using a unified tuning
  table.
- Rail-optimized tree topology support for MI3XX nodes with 4 NICs.
- ``ncclCommDump`` API for communicator state inspection.
- GDA (GPU Direct Acceleration) AllToAll integration via rocSHMEM.
- ``RCCL_IB_QPS_PER_P2P`` is now settable separately from
  ``NCCL_IB_QPS_PER_CONNECTION`` (backported from ROCm 7.1.0; noted here for
  completeness).
- WarpSpeed auto mode (``RCCL_WARP_SPEED_AUTO``): reduces CU count by 50% on
  a single node for AllReduce, AllGather from 64 MB, and ReduceScatter from
  256 MB.

Changed
-------

- PIX and PXB treated as equivalent GDR distances for more consistent topology
  detection.
- AllToAll optimized for 64 or more GPUs on gfx942.
- gfx950 thread-block size adjusted to improve LL64 and Simple protocol
  performance for AllReduce, AllGather, and ReduceScatter.
- Multi-node LL/LL128 tuning updated for gfx950 to improve large-message
  bandwidth.
- Graph mode memory registration and user buffer registration disabled as
  unsupported on current hardware.

Resolved issues
---------------

- Fixed missing memory fence in the LL protocol for gfx950 causing collective
  hangs.
- Fixed segmentation fault in the external profiler plugin on communicator
  teardown.
- Fixed LL128 protocol selection ignoring the user's explicit protocol
  override.
- Fixed ``rcclNetP2pPolicy`` returning incorrect policy for multi-NIC
  configurations.
- Fixed P2P batching hang when using batch operations.
- Fixed WarpSpeed auto mode selection bug.

----

RCCL 2.27.7 for ROCm 7.2.0
===========================

Minor behavioral change and performance fix for gfx950.

Changed
-------

- RCCL fatal error messages are now printed by default. Suppress with
  ``NCCL_DEBUG=NONE``.
- Disabled ``reduceCopyPacks`` pipelining for gfx950.

Known issues
------------

- AllToAllv/AllToAll for single GPU is hanging.

----

RCCL 2.27.7 for ROCm 7.1.1
===========================

Resolved issues
---------------

- Fixed crash when using the librccl-profiler plugin with AllToAll after the
  2.27 update.

Changed
-------

- P2P batching with ``RCCL_P2P_BATCH_ENABLE=1`` is limited to 32 nodes or
  fewer.

----

RCCL 2.27.7 for ROCm 7.1.0
===========================

This release adds NCCL 2.27.7 compatibility and several new environment
variables for P2P and IB tuning.

Added
-----

- NCCL compatibility: 2.27.7.
- ``RCCL_IB_QPS_PER_P2P``: sets the number of queue pairs (QPs) per
  connection for P2P operations independently from
  ``NCCL_IB_QPS_PER_CONNECTION``.
- ``RCCL_FORCE_ENABLE_DMABUF``: forces DMAbuf and bypasses system/kernel
  checks.
- ``RCCL_P2P_BATCH_THRESHOLD``: sets the message size limit for P2P batching.
- ``RCCL_P2P_BATCH_ENABLE``: enables P2P batching for messages up to 4 MB to
  improve small-message AllToAll performance at scale.
- ``RCCL_CHANNEL_TUNING_ENABLE``: enables channel tuning that overrides
  RCCL's internal ``threadThreshold``-based adjustments.

Changed
-------

- MSCCL++ is now disabled by default. Use ``--enable-mscclpp`` in
  ``rccl/install.sh`` to enable it (replaces ``--disable-mscclpp``).

Optimized
---------

- Batched P2P operations for improved small-message performance in AllToAll
  and AllGather.
- Channel count selection for small to medium message sizes in ReduceScatter.
- Code inlining for improved latency in AllReduce, AllGather, and
  ReduceScatter.

Known issues
------------

- Symmetric memory kernels are disabled pending CUMEM enablement work.
- ROCm versions earlier than 6.4.0 require ``HSA_NO_SCRATCH_RECLAIM=1``.

----

RCCL 2.26.6 for ROCm 7.0.0
===========================

This release adds NCCL compatibility up to 2.26.6, gfx950 support, and
multiple new tuning APIs.

Added
-----

- NCCL compatibility: 2.23.4, 2.24.3, 2.25.1, 2.26.6.
- New GPU target: gfx950.
- LL128 protocol on gfx950.
- Direct AllGather algorithm, enabled by default for multi-node with 16 or
  fewer nodes (4 MB message threshold).
- ``rcclGetAlgoInfo`` and ``rcclFuncMaxSendRecvCount`` APIs (requires RCCL
  built with ``RCCL_EXPOSE_STATIC``).
- ``RCCL_OVERRIDE_PROTO`` and ``RCCL_OVERRIDE_ALGO``: enforce specific
  protocol/algorithm choices, unlike ``NCCL_PROTO``/``NCCL_ALGO`` which
  re-run the model.
- MSCCL AllGather for multi-node gfx942/gfx950 (enable with
  ``RCCL_MSCCL_FORCE_ENABLE=1``).
- Double-buffering in ``reduceCopyPacks`` for pipelining; tunable via
  ``rcclSetPipelining``.

Changed
-------

- Default 112 channels for single-node with 8× gfx950.
- LL/LL128 usage ranges for AllReduce, AllGather, and ReduceScatter are now
  part of architecture-specific tuning models.

Resolved issues
---------------

- Resolved issue when using more than 64 channels in the same ``ncclGroup()``
  call with multiple collectives.
- Fixed unit test failures in ``ManagedMem`` and ``ManagedMemGraph`` tests.
- Fixed suboptimal algorithmic switching for AllReduce on MI300X.
- Fixed LL protocol on gfx950 by disabling inlining of ``LLGenericOp``
  kernels.
- Fixed segmentation fault in ``ncclCommSplit`` with MSCCL enabled.

Optimized
---------

- FP8 Sum operation now upcast to FP16 for improved performance.

Known issues
------------

- ROCm versions earlier than 6.4.0 require ``HSA_NO_SCRATCH_RECLAIM=1``.

----

RCCL 2.22.3 for ROCm 6.4.2
===========================

Added
-----

- LL128 protocol support on gfx942.

----

RCCL 2.22.3 for ROCm 6.4.1
===========================

Resolved issues
---------------

- Fixed accuracy issue in MSCCLPP ``allreduce7`` kernel in graph mode.
- Fixed IntraNet performance.
- Fixed rare hang due to proxy thread synchronization issue.

Known issues
------------

- ``ncclCommSplit`` with MSCCL enabled can cause a segmentation fault on some
  GPU configurations. Workaround: ``export RCCL_MSCCL_ENABLE=0``.
- Test failures in ``ManagedMem`` and ``ManagedMemGraph`` suffixed tests do not
  affect the RCCL component itself.

----

RCCL 2.22.3 for ROCm 6.4.0
===========================

This release adds NCCL 2.22.3 compatibility and rail-optimized tree support for
MI300.

Added
-----

- NCCL compatibility: 2.22.3.
- ``RCCL_SOCKET_REUSEADDR`` and ``RCCL_SOCKET_LINGER`` environment parameters.
- ``NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=VERBS`` now generates traces for fifo
  and data ``ibv_post_sends``.
- ``--log-trace`` flag in ``install.sh``.

Changed
-------

- Rail-optimized tree algorithm for MI300 series, requiring all 8 GPUs per
  node. Disable with ``RCCL_DISABLE_RAIL_TREES=1``.
- Additional tree-build debug logging via ``RCCL_OUTPUT_TREES=1`` to the
  ``GRAPH`` subsystem.

----

RCCL 2.21.5 for ROCm 6.3.0
===========================

This release adds NCCL 2.21.5 compatibility, MSCCL++ integration for gfx942,
and CPX mode for MI300X.

Added
-----

- NCCL compatibility: 2.21.5.
- MSCCL++ integration for AllReduce and AllGather on gfx942.
- Tuner Plugin example for MI300.
- Tuning table for large node counts.
- Support for amdclang++.
- ``NCCL_RINGS_REMAP`` environment variable for NIC ID remapping.

Changed
-------

- Increased channel count for MI300X multi-node.
- MSCCL enabled for single-process multi-threaded contexts.
- CPX mode enabled for MI300X.
- GDRDMA enabled for Linux kernel 6.4.0 and later.

----

RCCL 2.20.5 for ROCm 6.2.0
===========================

This release adds NCCL 2.20.5 and 2.19.4 compatibility, FP8 support, and
ROC-TX host-side profiling.

Added
-----

- NCCL compatibility: 2.20.5, 2.19.4.
- Support for FP8 and ``rccl_bfloat8`` data types.
- ROC-TX for host-side profiling.
- Static build support.

Changed
-------

- ``rccl_bfloat16`` replaced with ``hip_bfloat16``.
- NVTX (NVIDIA Tools Extension) code enabled in RCCL.

Known issues
------------

- On systems running Linux kernel 6.8.0 (such as Ubuntu 24.04), DMA
  (direct memory access) transfers between the GPU and NIC are disabled,
  affecting multi-node RCCL performance. Broadcom Thor-2 NICs and RoCE
  networks on Linux 6.8.0 or newer are affected. Older RCCL versions are also
  impacted.

----

Earlier releases
================

The following releases are documented for historical reference. Each maps to the
NCCL version shown in parentheses.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - ROCm version
     - RCCL version
     - Notes
   * - 6.1.0
     - 2.18.6
     - NCCL 2.18.6 compatibility.
   * - 6.0.0
     - 2.18.3
     - NCCL 2.18.3 compatibility.
   * - 5.7.0
     - 2.17.1-1
     - NCCL 2.17.1-1 compatibility; gfx94x support; ``NCCL_NCHANNELS_PER_PEER``
       support.
   * - 5.6.0
     - 2.16.2
     - NCCL 2.16.2 compatibility.
   * - 5.5.0
     - 2.15.5
     - NCCL 2.15.5 compatibility; HW-topology aware binary tree; experimental
       MSCCL support; NPKit integration.
   * - 5.4.0
     - 2.13.4
     - NCCL 2.13.4 compatibility; hipGraph improvements.
   * - 5.3.0
     - 2.12.10
     - Initial hipGraph support (``RCCL_ENABLE_HIPGRAPH``); NPKit integration.
   * - 5.2.3
     - 2.12.10
     - NCCL 2.12.10 compatibility; custom signal handler; IB port reuse
       options.
   * - 5.2.0
     - 2.11.4
     - Unit testing framework rework; minor bug fixes.
   * - 5.1.0
     - 2.11.4
     - NCCL 2.11.4 compatibility.
   * - 5.0.0
     - 2.10.3
     - NCCL 2.10.3 compatibility.
   * - 4.5.0
     - 2.9.9
     - NCCL 2.9.9 compatibility; runtime/development package split.
   * - 4.3.0
     - 2.8.4
     - Clique-based AllReduce channel selection (``RCCL_CLIQUE_ALLREDUCE_NCHANNELS``).
   * - 4.2.0
     - 2.8.4
     - NCCL 2.8.4 compatibility; GPU Direct RDMA read from GPU.
   * - 4.1.0
     - 2.7.8
     - Experimental clique-based kernels (``RCCL_ENABLE_CLIQUE=1``).
   * - 3.9.0
     - 2.7.8
     - AllToAllv kernel; XGMI-topology modifications.
   * - 3.8.0
     - 2.7.6
     - Static library build support.
   * - 3.7.0
     - 2.7.6
     - Gather, scatter, and AllToAll collectives added.
   * - 3.6.0
     - 2.7.0
     - NCCL 2.6 compatibility; network interface API v3.
   * - 3.5.0
     - 2.7.0
     - NCCL 2.6 compatibility; hip-clang as default compiler.
