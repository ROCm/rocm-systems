.. meta::
   :description: Learn how RCCL detects AMD GPU hardware, builds topology-aware ring and tree orderings, selects protocols, and tunes collective performance per architecture.
   :keywords: RCCL, hardware tuning, MI300X, MI350, MI200, gfx942, gfx950, gfx90a, xGMI, ring algorithm, tuning model, Rome models, rail-optimized topology, LL128, Simple protocol, ROCm, topology fingerprinting

.. _hardware-specific-optimizations:

****************************************
Hardware-specific optimizations in RCCL
****************************************

RCCL does not apply a single generic algorithm to all hardware. At
communicator initialization, it detects the GPU architecture, fingerprints the
physical interconnect topology, and loads a set of pre-measured constants that
govern algorithm selection, protocol selection, and channel count for every
subsequent collective call. This page explains how that process works and what
it means in practice for each supported AMD GPU generation.

Why hardware-aware tuning matters
==================================

Collective communication performance is determined by two physical limits: the
latency of a single hop and the bandwidth of the slowest link in the
communication path. These numbers differ substantially between GPU generations,
interconnect types, and node counts. A tuning table calibrated on MI200 hardware
will make wrong decisions on MI300X because the xGMI (Infinity Fabric) link
bandwidth, the number of peer GPUs, and the GPU memory system all changed.

RCCL resolves this by maintaining architecture-specific tuning models —
pre-measured tables of latency and bandwidth constants — and selecting the
right table at initialization time based on the detected GPU architecture and
topology.

Architecture detection and tuning index assignment
===================================================

When ``ncclCommInitRank`` runs, RCCL queries the HIP runtime for each GPU's
architecture string (for example ``gfx942`` or ``gfx950``) using
``src/misc/archinfo.cc`` and maps it to a **tuning index**. That index selects
one entry from the array of hardware-specific tuning models defined in
``src/graph/tuning.cc``.

The current mapping is:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Architecture string
     - GPU family
     - Tuning index
   * - ``gfx906``, ``gfx908``, ``gfx90a``
     - MI50, MI100, MI200
     - 0 (default)
   * - ``gfx942``
     - MI300A, MI300X
     - 5
   * - ``gfx950``
     - MI350
     - 6
   * - ``gfx1200``, ``gfx1201``
     - Radeon RX 9000 series
     - 7

If no match is found, RCCL falls back to tuning index 0. The tuning index
controls the specific latency and bandwidth constants used for algorithm and
protocol selection for the lifetime of that communicator.

What a tuning model contains
=============================

Each entry in the tuning model array holds the following fields, all of which
are empirically measured on real hardware:

- **hwLat** — per-hop latency in microseconds for each transport type (xGMI,
  PCIe, network).
- **bwRatio** — the fraction of raw link bandwidth that each algorithm and
  protocol combination actually achieves in practice (for example, Ring + Simple
  might reach 97% of raw link bandwidth for large messages on MI300X).
- **tree/ringCorrectionFactor** — size-bucketed multipliers that account for
  real-world deviations from the analytic model, such as memory system effects
  and GPU overhead.
- **llProtoRanges** — message-size thresholds that determine when to prefer the
  LL or LL128 protocol over Simple.
- **channelThresholds** — the channel count to use for each message size range.

These constants feed the algorithm selection formula used at every collective
launch:

.. code-block:: text

   time = (latency + nsteps × size ÷ bandwidth) × correctionFactor

The algorithm-and-protocol combination with the lowest estimated time is chosen.

Topology fingerprinting and Rome models
========================================

Architecture detection gives RCCL the per-hop constants, but not the
interconnect wiring between GPUs and NICs. That wiring determines which ring and
tree orderings keep traffic on fast links and avoid bottlenecks.

RCCL solves this through **Rome models** — predefined records in
``src/graph/rome_models.cc`` that encode the physical layout of known AMD server
configurations. Each record contains:

- GPU count, CPU socket count, and NIC count for that system class.
- NUMA affinity and PCIe attachment of each GPU and NIC.
- The xGMI connectivity matrix (which GPUs are directly fabric-connected to
  which).
- Hand-optimized ring orderings (``ringBase``) and tree orderings
  (``treeBase``, ``treeRail``) for that specific wiring.
- An ``options`` string that overrides tuning parameters, for example:
  ``"tuning=5,ll128Enabled=1,treeDefined=1,baseBw=161.4"``.

When a communicator is initialized, RCCL fingerprints the live machine and
searches the Rome model list for an exact match. On a match, the hand-optimized
orderings and the overriding tuning parameters are applied directly, bypassing
the generic graph search. On a non-match, RCCL falls back to its graph search
algorithm to derive ring and tree orderings, using the architecture-based tuning
index for the cost model.

Intra-node interconnects: xGMI and PCIe
=========================================

Intra-node GPU communication uses one of two physical transports:

**xGMI (Infinity Fabric).** xGMI provides direct, cache-coherent,
high-bandwidth links between GPUs on the same node. xGMI is the preferred
transport for intra-node collectives. RCCL uses it automatically when the
topology model confirms that two GPUs share an xGMI link.

**PCIe.** Where xGMI links are absent — for example, between GPUs in different
PCIe root complexes — RCCL falls back to PCIe peer access. PCIe bandwidth and
latency are substantially lower than xGMI, so ring and tree orderings are
constructed to minimize the number of PCIe hops.

MI300X single-node topology (gfx942)
--------------------------------------

A standard MI300X node contains 8 GPUs in a **fully connected mesh**: every GPU
has a direct xGMI link to each of the other 7 GPUs. Each link operates at a raw
bandwidth of 64 GB/s, with approximately 46–48 GB/s of effective bandwidth per
link after protocol overhead.

Because every link pair is direct, RCCL can construct 7 independent,
non-colliding rings that each traverse a distinct set of xGMI links. Running all
7 rings simultaneously means each GPU uses all 7 of its xGMI links, achieving a
combined per-GPU bandwidth of approximately 7 × 46 GB/s = 322 GB/s.

Finding 7 non-colliding rings on a fully connected point-to-point topology
(where every GPU connects to every other GPU directly rather than through a
switch) is a non-trivial combinatorial problem. The Rome models contain the
pre-computed ring orderings for known MI300X platform configurations to avoid
solving this at runtime.

MI350 single-node topology (gfx950)
--------------------------------------

MI350 nodes (gfx950) use the same fully connected xGMI mesh topology as MI300X
but with updated link speeds and GPU-internal architecture changes. The tuning
index for gfx950 reflects updated latency and bandwidth constants measured on
MI350 hardware. Key differences from the MI300X tuning:

- Default maximum channels is 48 for multi-node collectives (versus lower
  values on MI300X).
- Thread-block size is 256 threads (versus 512 on earlier architectures) for
  better resource efficiency.
- P2P batching is disabled by default on MI350 to avoid contention.

MI200 series (gfx90a)
-----------------------

MI200 series GPUs (AMD Instinct™ MI210, MI250, MI250X) use xGMI Gen 2 links with
a lower per-link bandwidth than MI300X. RCCL tuning index 0 applies to this
family with constants calibrated for the MI200 mesh bandwidth. On ROCm 7.13 and
later, a known issue causes per-launch scratch-memory reclaim to degrade RCCL
performance on gfx90a. Setting ``HSA_NO_SCRATCH_RECLAIM=1`` restores expected
performance.

Multi-node topologies and NIC placement
=========================================

Inter-node communication adds the NIC as a third actor alongside the GPU and
CPU. RCCL models the NIC placement in the Rome model records and uses it to
build rings that minimize the number of xGMI hops between a GPU and its
associated NIC. In a naïve configuration, data might travel across several xGMI
hops from one GPU to reach the NIC attached to another GPU on the same node.
A topology-aware ring ordering keeps each GPU's traffic on its directly attached
NIC.

Rail-optimized topologies
---------------------------

In a **rail-optimized network**, NIC 1 from every node connects to Switch 1, NIC
2 from every node connects to Switch 2, and so on. This contrasts with a classic
fabric where all NICs from a given node connect to the same switch.

Rail-optimized wiring has two effects relevant to RCCL:

1. NIC traffic for a given rank can be kept on its own rail (its own switch),
   avoiding spine switches entirely for same-rail communication. This can reduce
   latency and congestion.
2. Traffic between two NICs on the same node that happen to be on different rails
   must traverse the spine, which is a regression versus the classic fabric for
   that specific traffic pattern.

RCCL handles rail-optimized topologies through dedicated Rome model entries and
ring-ordering logic. For even node counts, RCCL uses alternating odd-node and
even-node ring orderings that traverse each NIC forward then backward, so NIC
traffic stays within a single rail. For odd node counts, a specialized three-node
tail pattern is used.

Rail-optimized tree collectives apply a similar principle: subtrees alternate
between NIC rail assignments so that intra-node xGMI traffic and inter-node NIC
traffic do not interfere.

Protocols and their hardware dependencies
==========================================

RCCL's three wire protocols have hardware-specific behavior that affects which
GPUs can use them.

**Simple protocol** sends a large payload followed by a single completion flag.
Because the payload and flag are separate writes, a memory fence must be inserted
between them to prevent compiler and hardware reordering. On AMD GPUs, this is
achieved with the ``__threadfence()`` HIP intrinsic. For multi-node transfers, an
additional cache flush is required to ensure the flag is not observed by the NIC
before the payload. The Simple protocol is the only choice for very large messages
where flag-based overhead is negligible.

**LL (Low Latency) protocol** embeds a 4-byte flag alongside every 4 bytes of
payload in a single 8-byte atomic write. Because the flag and its corresponding
data arrive together in one indivisible transaction, no memory fence is required
and the receiver can process data as it arrives rather than waiting for the whole
transfer. On MI300X (gfx942), the largest indivisible atomic transaction is
64 bytes, so the LL packet is 60 bytes of payload plus a 4-byte flag. LL
achieves the lowest latency at the cost of 50% bandwidth efficiency.

**LL128 protocol** uses 128-byte aligned atomic writes (120 bytes of data plus
an 8-byte flag) to achieve approximately 93.75% bandwidth efficiency while
retaining flag-based synchronization. LL128 requires that the hardware support
128-byte aligned atomic stores. On AMD hardware, LL128 is enabled
architecture-specifically:

- **gfx942 (MI300X):** enabled for 4-NIC configurations with a unified tuning
  table.
- **gfx950 (MI350):** enabled by default.

LL128 is not available on all platforms or NIC configurations. RCCL detects
hardware support and silently falls back to LL or Simple if 128-byte atomics are
not available.

The protocol tuning table for gfx942 and gfx950 is pre-measured. The following
table shows representative behavior from the tuning model (message sizes are
approximate thresholds; exact boundaries vary by collective and node count):

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Message size
     - Algorithm
     - Protocol
     - Typical latency
   * - < ~8 KB
     - Tree
     - LL
     - ~15 µs
   * - ~8 KB – 512 KB
     - Tree or Ring
     - LL128
     - ~45 µs
   * - > ~512 KB
     - Ring
     - Simple
     - Scales with bandwidth

The tuning model defaults are not a runtime auto-tuner — they are a
table-driven analytical cost model that applies pre-measured constants. RCCL
does not profile live traffic to adjust its choices.

The end-to-end tuning flow
===========================

The following sequence summarizes how RCCL arrives at an algorithm, protocol,
and channel count for each collective launch:

**At communicator initialization (once per communicator):**

1. Detect GPU architecture using ``rcclGetTuningIndexForArch`` and select a
   tuning model index.
2. Fingerprint the live hardware and search for a matching Rome model. On a
   match, load the hand-optimized ring/tree orderings and any tuning overrides
   from the model's ``options`` string.
3. Build bandwidth and latency matrices using ``ncclTopoTuneModel``.
4. Load the external tuner plugin, if one is configured using
   ``NCCL_TUNER_PLUGIN``.

**At each collective launch:**

5. Score every valid algorithm-protocol combination using
   ``ncclTopoGetAlgoTime``.
6. Allow the external tuner plugin to edit the cost table
   (``getCollInfo`` callback), if a plugin is loaded.
7. Select the combination with the lowest estimated time
   (``topoGetAlgoInfo``).
8. Apply any hard overrides (``RCCL_OVERRIDE_ALGO``,
   ``RCCL_OVERRIDE_PROTO``) and launch the kernel.

Architecture-specific environment variables
============================================

The following environment variables allow you to override or inspect
hardware-specific tuning decisions. Most are diagnostic or advanced-tuning
tools; use them with care in production.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Effect
   * - ``NCCL_DEBUG=INFO`` + ``NCCL_DEBUG_SUBSYS=TUNING``
     - Dumps the latency and bandwidth matrix selected at init, showing which
       tuning model and Rome model were applied.
   * - ``NCCL_ALGO``
     - Restricts algorithm selection to the specified set
       (``Ring``, ``Tree``, ``CollNet``). Re-runs the cost model over the
       enabled options.
   * - ``RCCL_OVERRIDE_ALGO``
     - Forces a specific algorithm unconditionally after cost selection,
       bypassing the tuning model.
   * - ``NCCL_PROTO``
     - Restricts protocol selection to the specified set
       (``LL``, ``LL128``, ``Simple``). This is a debug variable; forcing a
       protocol for all message sizes will degrade performance.
   * - ``RCCL_OVERRIDE_PROTO``
     - Forces a specific protocol unconditionally after cost selection.
   * - ``NCCL_MIN_NCHANNELS`` / ``NCCL_MAX_NCHANNELS``
     - Bounds the channel count. Setting ``NCCL_MIN_NCHANNELS=112`` can
       improve bandwidth on MI300X for distributed workloads.
   * - ``RCCL_CHANNEL_TUNING_ENABLE``
     - Enables or disables channel count overrides based on thread thresholds
       (default: ``1``, enabled).
   * - ``HSA_NO_SCRATCH_RECLAIM=1``
     - Required on gfx90a (MI200) with ROCm 7.13 or later to prevent
       per-launch scratch reclaim from degrading RCCL performance.
   * - ``RCCL_GFX9_CHEAP_FENCE_OFF``
     - Disables the cheap-fence optimization for gfx942/gfx950, which
       skips ``__threadfence()`` on the sender side of the Simple protocol
       where it is not needed for correctness.
   * - ``NCCL_TUNER_PLUGIN``
     - Path to a shared library that implements the external tuner plugin
       interface. The plugin can inspect the cost table and override individual
       entries before RCCL picks the minimum.

Related topics
===============

- :doc:`Collective operations in RCCL <./collective-operations>` — algorithm and
  protocol concepts explained from first principles
- :doc:`RCCL usage tips <../how-to/rccl-usage-tips>` — environment variables
  and general performance guidance
- :ref:`Environment variables <env-variables>` — complete reference for all
  RCCL runtime knobs
- :doc:`RCCL release notes <../release-notes>` — per-release changes to
  hardware support, tuning models, and default channel counts
