.. meta::
   :description: How the RCCL unified cost model selects an algorithm, protocol, symmetric kernel, or copy engine method
   :keywords: RCCL, ROCm, library, cost model, tuning, algorithm, protocol, symmetric kernel, copy engine, TUNING

.. _rccl-cost-model:

*****************************
The RCCL unified cost model
*****************************

RCCL has to pick one implementation for every collective call: an algorithm and protocol pair
such as ``RING``/``LL128``, a symmetric device-API kernel, or a copy engine method. The unified
cost model is the component that makes that choice. It prices every candidate it is allowed to
consider, takes the cheapest one, and writes the winner onto the task that is about to be enqueued.

The unified cost model is a **selector, not a queryable cost API**. This distinction matters
because the NCCL 2.31.2 release notes describe the feature in terms of "querying the cost estimate",
which suggests an API that hands a cost back to the caller. No such API exists in RCCL:

*  The only cost value a user can read is ``ncclSimInfo_t.estimatedTime``, obtained through
   ``ncclGroupSimulateEnd``. That value is the modeled time of the implementation that was
   *already chosen*, not an independent estimate you can request for an arbitrary algorithm.

On a default build, ``estimatedTime`` reports only the general algorithm and protocol cost.
The legacy selector is the only writer, in ``getAlgoInfo``, and symmetric tasks are split off
into their own queue before that selector runs. A simulated group whose every task was claimed
by the symmetric scheduler therefore leaves ``estimatedTime`` at ``NCCL_UNDEF_FLOAT``
(``-1.0``). The winning symmetric cost is never published. With
``NCCL_ENQUEUE_REARCH_ENABLE=1`` the field means something different: it is the sum of the
valid per-task estimates across every queue, symmetric and copy engine included.

.. code-block:: cpp

   ncclSimInfo_t simInfo = NCCL_SIM_INFO_INITIALIZER;

   ncclGroupStart();
   ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
   ncclGroupSimulateEnd(&simInfo);

   // Modeled time, in microseconds, of the general implementation RCCL selected.
   // NCCL_UNDEF_FLOAT (-1.0) means no general selection ran, not a cost of zero: on a
   // default build that is what a symmetric-only group returns.
   if (simInfo.estimatedTime == NCCL_UNDEF_FLOAT) {
     printf("no general estimate for this group\n");
   } else {
     printf("estimated time: %f us\n", simInfo.estimatedTime);
   }

If you want to know *why* a particular implementation was picked, the supported way is to read the
``TUNING`` debug output described in :ref:`cost-model-reading-the-logs`.

.. _cost-model-what-is-active:

Which decisions the cost model makes today
==========================================

Read this section first. The unified cost model is the selector for two decisions, and *not* for
the one most users come here asking about:

.. list-table::
    :header-rows: 1
    :widths: 30,70

    * - **Decision**
      - **Status**

    * - Symmetric kernel selection
      - **Active.** The symmetric scheduler calls the cost model with a symmetric-only candidate
        mask for every symmetric-capable collective.

    * - Copy engine AllGather, unicast versus multicast
      - **Active.** The copy engine path calls the cost model with a CE-only mask. It is skipped
        entirely when ``NCCL_CE_COLL_AG_MULTICAST_THRESHOLD`` is set to a non-negative value: that
        override fixes the threshold, a plain size comparison decides, and the cost model is never
        consulted.

    * - General algorithm and protocol selection
      - **Not active by default.** ``NCCL_ENQUEUE_REARCH_ENABLE`` defaults to ``0``, and group
        launch routes to the legacy path, where selection runs through ``getAlgoInfo`` and the
        per-architecture tuning model in ``src/graph/tuning.cc``.

Two consequences follow, and they shape the rest of this page:

*  Each default-path entry point restricts the model to a **single family**, so the model is
   never *entered* with a mask that would compare a symmetric kernel against a ring kernel
   against a copy engine method. Choosing *between* those backends happens earlier, outside the
   cost model.
*  The per-decision log records described below are emitted only for the two active decisions. A
   collective that uses neither symmetric memory nor the copy engine produces no such records.

The single-family restriction is on the entry points, not on every call into the model. Two
cases cross family boundaries:

*  ``ncclTuningCompute`` queries itself recursively with a general-only mask to decide whether a
   symmetric LL kernel must fall back, and again when the fallback is taken. So a symmetric
   decision can emit a second ``Input:``/``Best tuning`` pair carrying
   ``NCCL_TUNING_MASK_GENERAL_KERNELS`` (``0x1fffff``). Those extra records come from the
   fallback check, not from a general selection the enqueue path requested.
*  With ``NCCL_ENQUEUE_REARCH_ENABLE=1``, ``fillCollTuningInput`` builds one mask spanning
   general and symmetric candidates, and the model does compare them head to head.

.. note::

   ``NCCL_ENQUEUE_REARCH_ENABLE=1`` is experimental and is not recommended for production.
   RCCL's per-architecture tuning constants are not wired into the unified path yet, so turning
   it on routes general algorithm and protocol selection through a model that has not been tuned
   for AMD Instinct accelerators. Expect selection quality, and therefore performance, to change.

.. _cost-model-id-space:

The candidate space
===================

Every candidate implementation the model can select has a single flat integer ID. The ID space is
partitioned into three contiguous families, defined in ``src/include/tuning.h``:

.. list-table::
    :header-rows: 1
    :widths: 20,25,55

    * - **ID range**
      - **Family**
      - **How the ID is formed**

    * - ``[0, 21)``
      - General kernels
      - ``algo * NCCL_NUM_PROTOCOLS + proto``, covering 7 algorithms
        (``TREE``, ``RING``, ``COLLNET_DIRECT``, ``COLLNET_CHAIN``, ``NVLS``, ``NVLS_TREE``, ``PAT``)
        times 3 protocols (``LL``, ``LL128``, ``SIMPLE``). Not every combination is implemented.

    * - ``[21, 39)``
      - Symmetric device-API kernels
      - ``21 + ncclSymkKernelId_*``, the 18 symmetric AllReduce, AllGather, and ReduceScatter kernels.

    * - ``[39, 41)``
      - Copy engine methods
      - ``39 + ncclCeMethodId_*``: ``AllGather_UC`` (unicast) and ``AllGather_MC`` (multicast).

A caller restricts the model to a subset of these families with a 64-bit candidate mask. Four
mask constants name the useful subsets:

*  ``NCCL_TUNING_MASK_GENERAL_KERNELS``: bits ``[0, 21)``
*  ``NCCL_TUNING_MASK_SYM_KERNELS``: bits ``[21, 39)``
*  ``NCCL_TUNING_MASK_CE``: bits ``[39, 41)``
*  ``NCCL_TUNING_MASK_ALL``: all of the above

The mask is the single most useful thing to read out of the logs, because it tells you which
family the model was even allowed to choose from on a given call.

.. _cost-model-reading-the-logs:

Reading the tuning logs
=======================

The cost model is observable through the ``TUNING`` debug subsystem:

.. code-block:: shell

   mkdir -p "$HOME/rccl-logs"
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=TUNING
   export NCCL_DEBUG_FILE=$HOME/rccl-logs/rccl.%h.%p.log

.. note::

   Always capture RCCL debug output with ``NCCL_DEBUG_FILE`` rather than redirecting with ``2>&1``.
   RCCL writes debug output to standard output, and interleaving it with the application's own
   output through a shell redirect tears individual log lines apart, which makes the tuning records
   unreadable. ``NCCL_DEBUG_FILE`` accepts ``%h`` for the hostname and ``%p`` for the process ID,
   so each rank gets its own intact file.

   Write the log somewhere only the job can read. RCCL opens the file with ``fopen(..., "w")``,
   so it lands at whatever the umask allows, commonly ``0644``, and an ``NCCL_DEBUG=INFO`` log
   carries the hostname, PID, PCI bus IDs, ring and tree maps, and buffer addresses. Under a
   world-writable ``/tmp`` on a shared node, every other local account can read it. RCCL does not
   create the directory and falls back to standard output without a warning if it cannot open the
   file, so create the directory first.

The per-decision records
------------------------

These two records are emitted by the cost model itself, so they appear only for the decisions it
makes: symmetric kernel selection, and the copy engine AllGather unicast/multicast choice. A
collective that uses neither produces no ``Input:`` or ``Best tuning`` line, and its algorithm and
protocol are chosen on the legacy path instead. Enabling ``NCCL_ENQUEUE_REARCH_ENABLE=1`` routes
general selection through the cost model as well, and these records then appear for every
collective. See :ref:`cost-model-what-is-active`.

The first record states what was asked for:

.. code-block:: text

   Input: { .comm = %p, .tuningMask = 0x%lx, .func = %s, .redOp = %d, .devRedop = %d, .dataType = %d, .nBytes = %lu, .numPipesOps = %d, .count = %lu, .countMax = %lu, .nWorks = %d, .winRegType = %d, .regBuff = %d }

The second record states what was chosen:

.. code-block:: text

   Best tuning { .id = %d, .valid = %d, timeUs = %f, .algo = %s, .proto = %s, .symKernelId = %s, .ceMethodId = %d, nChannels = %d, maxChannels = %d, nWarps = %d, forced = %d }

The fields worth reading first:

*  ``.tuningMask`` is the candidate mask from :ref:`cost-model-id-space`, printed in hexadecimal.
   Compare it against all three bounded ranges, not against a single threshold: bits ``[0, 21)``
   are general algorithm and protocol candidates (``0x1fffff``), bits ``[21, 39)`` are symmetric
   kernels (``0x7fffe00000``), and bits 39 and 40 are copy engine methods (``0x18000000000``).
   "At or above bit 21" is not enough to call a decision symmetric, because the copy engine bits
   are above 21 as well.

*  Only the caller narrows ``.tuningMask``. On the symmetric path that is the per-call
   ``algSelection``, intersected with the symmetric range. The process-wide overrides work
   differently: ``NCCL_ALGO``, ``NCCL_PROTO``, and ``NCCL_SYM_KERNEL`` are parsed once at
   communicator init and clear entries in ``tuningContext.enabled[id][func]``, which the model
   checks per candidate. Pinning one symmetric kernel with ``NCCL_SYM_KERNEL`` therefore still
   logs the full symmetric mask ``0x7fffe00000``. A full mask is not evidence that the override
   was ignored. Read ``.forced`` on the "Best tuning" line for that.

*  ``.id`` on the "Best tuning" line is the flat candidate ID of the winner. Map it into the three
   ranges to find out which family won: below 21 it is a general ``algo``/``proto`` pair, from 21
   to 38 a symmetric kernel, and 39 or 40 a copy engine method. The ``.algo``/``.proto``,
   ``.symKernelId``, and ``.ceMethodId`` fields are the expanded form of the same answer; the
   fields belonging to the families that did not win are left at their "none" values.

*  ``timeUs`` is the model's cost estimate for the winner, in microseconds. It is modeled, not
   measured, so treat it as a ranking key rather than a performance prediction. Which parameters
   produced it depends on the winning family: the general latency and bandwidth tables for a
   general row, ``sym_model.cc`` for a symmetric kernel, ``ce_model.cc`` for a copy engine
   method.

*  ``.nChannels``, ``.maxChannels``, and ``.nWarps`` are the launch geometry chosen alongside the
   implementation.

*  ``.forced`` is non-zero when the selection was pinned by an override instead of won on cost.

*  ``.nBytes`` on the "Input" line is the message size the model costed, and ``.regBuff`` and
   ``.winRegType`` report whether the user buffers were registered. Both heavily influence which
   candidates are eligible, so a selection that looks wrong is often a registration problem rather
   than a tuning problem.

The communicator init tables
----------------------------

At communicator initialization, rank 0 additionally prints a latency and bandwidth table, one
block per group of three algorithms. Each cell is ``latency/bandwidth`` for one collective,
algorithm, and protocol, and the block is preceded by the maximum thread count per algorithm and
protocol.

Two such tables are printed, back to back and under identical headers, so read them in order:

#. ``ncclTuningInit`` prints the unified model's general table, from
   ``tuningContext.generalLatencies`` and ``tuningContext.generalBandwidths``.
#. ``ncclTopoTuneModel`` then prints the legacy per-architecture table, from ``comm->latencies``
   and ``comm->bandwidths``. On a default build this second table is the one general algorithm
   and protocol selection actually reads.

Both tables cover general algorithm and protocol costs only, so they do not explain either of the
two decisions the cost model makes by default:

*  Symmetric kernel costs come from ``src/tuning/sym_model.cc`` and the per-kernel models under
   ``src/tuning/sym_model/``, with the parameter set chosen by ``RCCL_SYM_MODEL``.
*  Copy engine costs come from ``src/tuning/ce_model.cc``.

If general selections look wrong across the board, check the legacy table first: a bandwidth
entry that does not match the hardware explains far more than any individual decision record. If
a symmetric or copy engine selection looks wrong, the init tables are the wrong place to look.

The tuner plugin boundary
=========================

If you are writing a tuner plugin, the important thing to understand is how narrow the plugin's
view of the candidate space is. When a ``v6`` plugin is loaded, RCCL flattens the general-kernel
candidates into a ``NCCL_NUM_ALGORITHMS`` by ``NCCL_NUM_PROTOCOLS`` table of modeled costs and
passes that table, and only that table, to ``getCollInfo``. Cells for candidates the model found
ineligible are set to ``NCCL_ALGO_PROTO_IGNORE``, which is ``-1.0``. After ``getCollInfo``
returns, RCCL copies the plugin's values back onto the general candidates and runs the argmin over
the whole list.

**A cell is ineligible when it is negative, not only when it equals the sentinel.** The argmin
drops cells equal to ``NCCL_ALGO_PROTO_IGNORE`` and then accepts only costs ``>= 0.0``, and a
later cost adjustment can scale an already-ineligible cell so that it stays negative without
staying equal to ``-1.0``. The fp8 relegation of deep RING reductions does exactly this: it
multiplies the cell by 1024, turning the ``-1.0`` sentinel into ``-1024.0``. Test ``>= 0.0f``,
never ``!= NCCL_ALGO_PROTO_IGNORE``.

The consequences for a plugin author:

*  A plugin sees the 7x3 general ``algo`` x ``proto`` table and nothing else.
*  A plugin **cannot** price, veto, or select a symmetric kernel. Symmetric candidates are not
   represented in the table, and their modeled costs pass through the plugin untouched.
*  A plugin **cannot** price, veto, or select a copy engine method, for the same reason.
*  Writing ``0.0`` into a cell is how a plugin claims that algorithm and protocol combination:
   zero is below every modeled cost, so the argmin picks it. Write it only into a cell that was
   already ``>= 0.0``, or you claim a candidate RCCL ruled out.
*  A plugin is **not** bound by a per-call ``algSelection``. The narrowing described in
   :ref:`cost-model-per-call-selection` runs before ``getCollInfo``, so writing a cost into a
   blanked cell revives a row the selection excluded. The same ``>= 0.0f`` guard leaves the
   caller's selection intact. A selection whose named rows are all ineligible is the one
   exception: with ``forceAlgSelection`` at its default of ``1`` the call fails before
   ``getCollInfo`` runs, so the plugin never sees that table and cannot rescue the selection.

.. code-block:: cpp

   // Claim RING/LL128 for this call; leave every other cell as RCCL costed it.
   // The guard is >= 0.0f, not != NCCL_ALGO_PROTO_IGNORE: an ineligible cell can be
   // negative without equalling the sentinel, and claiming it selects an unusable kernel.
   float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
   if (table[NCCL_ALGO_RING][NCCL_PROTO_LL128] >= 0.0f) {
     table[NCCL_ALGO_RING][NCCL_PROTO_LL128] = 0.0f;
   }

For the full plugin interface, see :doc:`Using the RCCL Tuner plugin <./using-rccl-tuner-plugin-api>`.

.. _cost-model-per-call-selection:

Restricting the candidate set per call
======================================

Environment variables such as ``NCCL_ALGO`` apply to the whole process. To narrow the candidate
set for one collective call instead, set ``ncclCollConfig_t::algSelection`` to a selection string.
The names come from an algorithm registry that covers both the general rows (``TREE_LL``,
``TREE_LL128``, ``TREE_SIMPLE``, ``RING_LL``, ``RING_LL128``, ``RING_SIMPLE``,
``COLLNET_DIRECT_SIMPLE``, ``COLLNET_CHAIN_SIMPLE``, ``NVLS_SIMPLE``, ``NVLSTREE_SIMPLE``,
``PAT_SIMPLE``) and the symmetric rows (``SYMK_AGxLL_R``, ``SYMK_LLMC``, ``SYMK_TmaST``, and so on).

.. important::

   On a default build, ``algSelection`` narrows **both** symmetric kernel selection and general
   algorithm and protocol selection. ``NCCL_ENQUEUE_REARCH_ENABLE`` defaults to ``0``, so group
   launch takes the legacy path, and that path reaches the filter through ``getAlgoInfo``: every
   general row the selection does not name is marked ``NCCL_ALGO_PROTO_IGNORE`` before the argmin
   runs. A selection that names no general row at all, such as ``SYMK_LL``, leaves the general
   table untouched, so the symmetric scheduler can still decline and fall back to it. The
   process-wide ``NCCL_ALGO``, ``NCCL_PROTO``, and ``NCCL_SYM_KERNEL`` still win over
   ``algSelection`` for any collective they force, because all three mark that collective in
   ``tuningContext.forced``. ``RCCL_OVERRIDE_ALGO`` and ``RCCL_OVERRIDE_PROTO`` do not: they are
   applied after the narrowing, so a selection that excludes the overridden row drops the
   override, with a ``WARN``.

   If the selection names general rows but none of them is eligible on this communicator, for
   example ``NVLS_SIMPLE`` without NVLS, the call fails with ``ncclInvalidArgument``. Set
   ``forceAlgSelection`` to ``0`` to fall back to automatic selection instead.

   Every rank must pass a selection that resolves to the same algorithm and protocol for a
   given call. Selection is local and nothing compares the winner across ranks, so a
   disagreement is undefined behavior, as it is for ``cgaClusterSize``. Eligibility is per
   rank too, so ``forceAlgSelection = 0`` can diverge even when every rank passes the same
   string.

.. note::

   A per-call ``algSelection`` naming a general row is a no-op inside one window, because the
   legacy selector reassigns the algorithm after the argmin has already run. On gfx950, with no
   plugin tuner loaded and ``NCCL_ALGO`` unset in the environment, a single-node AllReduce whose
   per-rank size is between 64 and 262144 bytes inclusive is forced to ``TREE``/``LL``.
   ``RING_SIMPLE`` asked for in that window is silently not honored. Three sibling guards in
   ``getAlgoInfo`` have the same shape, including one that forces ``RING`` for single-node
   AllReduce on gfx1200 and gfx1201.

Matching is a case-insensitive **name prefix**, not an exact comparison. A short name therefore
selects a family:

*  ``SYMK_LL`` selects three rows, not two: the AllGather ``SYMK_LL`` (bit 26) and ``SYMK_LLMC``
   (bit 27), and also the ReduceScatter ``SYMK_LL`` (bit 33). A registry name is unique per
   collective but can repeat across collectives, and the prefix match does not look at the
   collective, so a name shared by two collectives selects both rows.
*  ``SYMK_RailA2A`` selects both ``SYMK_RailA2A_LsaLD`` and ``SYMK_RailA2A_LsaLDMC``.
*  ``RING`` selects ``RING_LL``, ``RING_LL128``, and ``RING_SIMPLE``, but not
   ``SYMK_RailRing_LsaSTMC``. The match is anchored at the start of the name, so a name that
   merely *contains* the tag is not selected.

Whatever the string selects becomes the candidate mask for the families the selection applies to,
and the cost model picks the cheapest member of that set.

.. note::

   Copy engine methods have no rows in the registry and cannot be named through ``algSelection``.
   Process-wide overrides such as ``NCCL_ALGO``, ``NCCL_PROTO``, and ``NCCL_SYM_KERNEL`` take
   precedence over a per-call ``algSelection`` for any collective they force.

Related environment variables
=============================

The variables that affect cost model behavior most directly are
``NCCL_ENQUEUE_REARCH_ENABLE``, ``RCCL_SYM_MODEL``, ``NCCL_SYM_CTAS``, ``NCCL_SYM_KERNEL``,
``NCCL_PAT_ENABLE``, and ``NCCL_LL128_C2C``. They are documented under "Algorithm and protocol
control" in the :ref:`environment variable reference <env-variables>`.

.. note::

   ``NCCL_PAT_ENABLE`` defaults to ``0`` in RCCL, whereas upstream NCCL defaults it to ``2``
   (automatic). PAT is therefore not a cost model candidate in RCCL unless you request it
   explicitly, with one exception: ``ncclPatEnable`` also honors ``comm->forcePatEnable``, which
   RCCL sets at init on the inter-node communicator of hierarchical collectives whenever
   ``NCCL_PAT_ENABLE`` is not explicitly ``0`` and AINIC is not in use. On that communicator PAT
   is a candidate with the variable unset.
