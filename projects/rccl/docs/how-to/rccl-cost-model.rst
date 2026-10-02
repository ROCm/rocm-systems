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

.. code-block:: cpp

   ncclSimInfo_t simInfo = NCCL_SIM_INFO_INITIALIZER;

   ncclGroupStart();
   ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
   ncclGroupSimulateEnd(&simInfo);

   // Modeled time, in microseconds, of the implementation RCCL selected.
   printf("estimated time: %f us\n", simInfo.estimatedTime);

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

*  Each live call restricts the model to a **single family**, so the model is never asked to
   compare a symmetric kernel against a ring kernel against a copy engine method. Choosing
   *between* those backends happens earlier, outside the cost model.
*  The per-decision log records described below are emitted only for the two active decisions. A
   collective that uses neither symmetric memory nor the copy engine produces no such records.

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

*  ``NCCL_TUNING_MASK_GENERAL_KERNELS`` -- bits ``[0, 21)``
*  ``NCCL_TUNING_MASK_SYM_KERNELS`` -- bits ``[21, 39)``
*  ``NCCL_TUNING_MASK_CE`` -- bits ``[39, 41)``
*  ``NCCL_TUNING_MASK_ALL`` -- all of the above

The mask is the single most useful thing to read out of the logs, because it tells you which
family the model was even allowed to choose from on a given call.

.. _cost-model-reading-the-logs:

Reading the tuning logs
=======================

The cost model is observable through the ``TUNING`` debug subsystem:

.. code-block:: shell

   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=TUNING
   export NCCL_DEBUG_FILE=/tmp/rccl.%h.%p.log

.. note::

   Always capture RCCL debug output with ``NCCL_DEBUG_FILE`` rather than redirecting with ``2>&1``.
   RCCL writes debug output to standard output, and interleaving it with the application's own
   output through a shell redirect tears individual log lines apart, which makes the tuning records
   unreadable. ``NCCL_DEBUG_FILE`` accepts ``%h`` for the hostname and ``%p`` for the process ID,
   so each rank gets its own intact file.

The per-decision records
------------------------

These two records are emitted by the cost model itself, so they appear only for the decisions it
makes: symmetric kernel selection, and the copy engine AllGather unicast/multicast choice. A
collective that uses neither produces no ``Input:`` or ``Best tuning`` line, and its algorithm and
protocol are chosen on the legacy path instead. Enabling ``NCCL_ENQUEUE_REARCH_ENABLE=1`` routes
general selection through the cost model as well, and these records then appear for every
collective — see :ref:`cost-model-what-is-active`.

The first record states what was asked for:

.. code-block:: text

   Input: { .comm = %p, .tuningMask = 0x%lx, .func = %s, .redOp = %d, .devRedop = %d, .dataType = %d, .nBytes = %lu, .numPipesOps = %d, .count = %lu, .countMax = %lu, .nWorks = %d, .winRegType = %d, .regBuff = %d }

The second record states what was chosen:

.. code-block:: text

   Best tuning { .id = %d, .valid = %d, timeUs = %f, .algo = %s, .proto = %s, .symKernelId = %s, .ceMethodId = %d, nChannels = %d, maxChannels = %d, nWarps = %d, forced = %d }

The fields worth reading first:

*  ``.tuningMask`` is the candidate mask from :ref:`cost-model-id-space`, printed in hexadecimal.
   Compare it against the three ranges to see which families were in play. A mask whose only set
   bits are at or above bit 21 means the call was a symmetric-kernel decision and no general
   algorithm was ever a candidate; a mask confined to bits ``[0, 21)`` means the opposite. A
   narrow mask with a single bit set means something upstream already forced the choice, through
   ``NCCL_ALGO``, ``NCCL_PROTO``, ``NCCL_SYM_KERNEL``, or a per-call ``algSelection``.

*  ``.id`` on the "Best tuning" line is the flat candidate ID of the winner. Map it into the three
   ranges to find out which family won: below 21 it is a general ``algo``/``proto`` pair, from 21
   to 38 a symmetric kernel, and 39 or 40 a copy engine method. The ``.algo``/``.proto``,
   ``.symKernelId``, and ``.ceMethodId`` fields are the expanded form of the same answer; the
   fields belonging to the families that did not win are left at their "none" values.

*  ``timeUs`` is the model's cost estimate for the winner, in microseconds. It is a modeled number
   from latency and bandwidth tables, not a measurement, so treat it as a ranking key rather than
   a performance prediction.

*  ``.nChannels``, ``.maxChannels``, and ``.nWarps`` are the launch geometry chosen alongside the
   implementation.

*  ``.forced`` is non-zero when the selection was pinned by an override instead of won on cost.

*  ``.nBytes`` on the "Input" line is the message size the model costed, and ``.regBuff`` and
   ``.winRegType`` report whether the user buffers were registered. Both heavily influence which
   candidates are eligible, so a selection that looks wrong is often a registration problem rather
   than a tuning problem.

The communicator init table
---------------------------

At communicator initialization, rank 0 additionally prints the latency and bandwidth table the
model is working from, one block per group of three algorithms. Each cell is
``latency/bandwidth`` for one collective, algorithm, and protocol, and the block is preceded by
the maximum thread count per algorithm and protocol. This table is printed once per communicator
and is the ground truth behind every subsequent ``timeUs`` value. If the selections look wrong
across the board, check this table first: a bandwidth entry that does not match the hardware
explains far more than any individual decision record.

The tuner plugin boundary
=========================

If you are writing a tuner plugin, the important thing to understand is how narrow the plugin's
view of the candidate space is. When a ``v6`` plugin is loaded, RCCL flattens the general-kernel
candidates into a ``NCCL_NUM_ALGORITHMS`` by ``NCCL_NUM_PROTOCOLS`` table of modeled costs and
passes that table, and only that table, to ``getCollInfo``. Cells for candidates the model found
ineligible are left at ``NCCL_ALGO_PROTO_IGNORE``. After ``getCollInfo`` returns, RCCL copies the
plugin's values back onto the general candidates and runs the argmin over the whole list.

The consequences for a plugin author:

*  A plugin sees the 7x3 general ``algo`` x ``proto`` table and nothing else.
*  A plugin **cannot** price, veto, or select a symmetric kernel. Symmetric candidates are not
   represented in the table, and their modeled costs pass through the plugin untouched.
*  A plugin **cannot** price, veto, or select a copy engine method, for the same reason.
*  Writing ``0.0`` into a cell is how a plugin claims that algorithm and protocol combination:
   zero is below every modeled cost, so the argmin picks it.

.. code-block:: cpp

   // Claim RING/LL128 for this call; leave every other cell as RCCL costed it.
   float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
   if (table[NCCL_ALGO_RING][NCCL_PROTO_LL128] != NCCL_ALGO_PROTO_IGNORE) {
     table[NCCL_ALGO_RING][NCCL_PROTO_LL128] = 0.0f;
   }

For the full plugin interface, see :doc:`Using the RCCL Tuner plugin <./using-rccl-tuner-plugin-api>`.

Restricting the candidate set per call
======================================

Environment variables such as ``NCCL_ALGO`` apply to the whole process. To narrow the candidate
set for one collective call instead, set ``ncclCollConfig_t::algSelection`` to a selection string.
The names come from an algorithm registry that covers both the general rows (``TREE_LL``,
``TREE_LL128``, ``TREE_SIMPLE``, ``RING_LL``, ``RING_LL128``, ``RING_SIMPLE``,
``COLLNET_DIRECT_SIMPLE``, ``COLLNET_CHAIN_SIMPLE``, ``NVLS_SIMPLE``, ``NVLSTREE_SIMPLE``,
``PAT_SIMPLE``) and the symmetric rows (``SYMK_AGxLL_R``, ``SYMK_LLMC``, ``SYMK_TmaST``, and so on).

.. important::

   On a default build, ``algSelection`` narrows **symmetric kernel selection only**. A string
   naming general rows is parsed and validated — an unsatisfiable selection is rejected when
   ``forceAlgSelection`` is set, and otherwise logged — but it does not currently constrain which
   algorithm and protocol the legacy selector picks. To pin a general algorithm or protocol, use
   the process-wide ``NCCL_ALGO`` and ``NCCL_PROTO`` instead.

Matching is a case-insensitive **name prefix**, not an exact comparison. A short name therefore
selects a family:

*  ``SYMK_LL`` selects ``SYMK_LL`` and ``SYMK_LLMC``, because ``SYMK_LL`` is a prefix of both.
*  ``SYMK_RailA2A`` selects both ``SYMK_RailA2A_LsaLD`` and ``SYMK_RailA2A_LsaLDMC``.
*  ``RING`` selects ``RING_LL``, ``RING_LL128``, and ``RING_SIMPLE``, but not
   ``SYMK_RailRing_LsaSTMC`` — the match is anchored at the start of the name, so a name that
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
   (automatic). PAT is therefore never a cost model candidate in RCCL unless you request it
   explicitly.
