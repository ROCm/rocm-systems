.. meta::
  :description: Using kernel replay with rocprofv3 to collect multiple counter groups or thread trace targets in a single application run
  :keywords: rocprofv3, kernel replay, replay-mode, kernel-replay-beta-enabled, multi-pass counters, application replay, snapshot restore, thread trace, ATT

.. _using-kernel-replay-rocprofv3:

======================================
Using kernel replay with rocprofv3
======================================

``rocprofv3 --replay-mode kernel --kernel-replay-beta-enabled`` collects several hardware-counter
groups in **one application run**. For each kernel dispatch, the profiler snapshots every tracked
coarse-grained allocation owned by that dispatch's agent -- not only the memory the kernel itself
can reach -- re-executes that dispatch once per ``--pmc`` group, and restores the snapshot between
passes so every group observes identical inputs. Host RAM use and copy time therefore scale with
the agent's whole tracked footprint rather than with the kernel's working set, as
`What is snapshotted`_ describes.

Without ``--replay-mode kernel``, multiple ``--pmc`` groups use *application replay*: the whole
application is re-run from start to finish once per group. Kernel replay is useful when those full
re-runs are expensive or non-deterministic.

This page is the command-line how-to. The SDK callback domain, ``replay_pass_count``, and localized
context control are documented in :ref:`using-kernel-replay`.

.. warning::

   Kernel replay is **experimental**. Use ``--replay-mode kernel`` together with
   ``--kernel-replay-beta-enabled`` to acknowledge that the behaviour may change in a future
   release. The SDK header lives under ``rocprofiler-sdk/experimental/``. Use it when you
   understand the limitations below.

How it compares
===============

Hardware has a limited number of counter registers per block. When the counters you want do not
fit in one hardware pass, ``rocprofv3`` has three ways to collect them:

.. list-table::
   :header-rows: 1
   :widths: 22 28 25 25

   * - Approach
     - Scope
     - Memory handling
     - Cost
   * - Application replay (default for multiple ``--pmc`` groups)
     - Whole application, re-run once per group
     - None; each run is a fresh process
     - ``O(N × application runtime)``
   * - **Kernel replay** (``--replay-mode kernel --kernel-replay-beta-enabled``)
     - One dispatch, re-executed in place
     - Device memory snapshot and restore between passes
     - ``O(N × kernel time + N × snap/restore)``
   * - Counter group rotation (``pmc_groups`` / ``pmc_group_interval``)
     - Amortized across successive dispatches
     - None; different dispatches sample different groups
     - ``O(1 × application runtime)``, but not the same dispatch

Kernel replay is **not** the same as :ref:`using-spm`. SPM streams counter samples over time from
hardware ring buffers; kernel replay re-executes a dispatch so each pass can collect a different
counter group against the same inputs.

Collecting counters with kernel replay
======================================

``--replay-mode kernel`` requires counter collection (``--pmc``, input-file ``pmc``, or
``pmc_groups``) and ``--kernel-replay-beta-enabled``. The number of replay passes is the number of
counter groups (one pass per group). There is no separate pass-count flag. ``rocprofv3`` uses
callback dispatch counting for counter records; ``replay_pass`` is emitted on that path.

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES GRBM_COUNT --pmc GRBM_GUI_ACTIVE --replay-mode kernel --kernel-replay-beta-enabled -- <application_path>

The preceding command collects both counter groups in a **single** run of ``<application_path>``.
Each targeted dispatch is replayed twice: pass 0 collects ``SQ_WAVES`` and ``GRBM_COUNT``; pass 1
collects ``GRBM_GUI_ACTIVE``. Device memory is restored between those passes.

A single ``--pmc`` group still works. In that case the tool asks the SDK for one pass, which does
not snapshot or restore — the dispatch takes the ordinary path.

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES GRBM_COUNT --replay-mode kernel --kernel-replay-beta-enabled -- <application_path>

Input files are unaffected by the flag. An input file's jobs each describe a run of their own --
their own output configuration, kernel filters and ranges -- so ``rocprofv3`` runs them exactly as
it would without ``--replay-mode kernel``, and replay applies within each job to the
counter groups that job asks for.

List counters first with ``rocprofv3 --list-avail`` or ``rocprofv3-avail list --pmc``. Each
individual ``--pmc`` group must still fit in one hardware pass; kernel replay does not split a
group that the hardware cannot collect together. Use ``rocprofv3-avail pmc-check`` to verify a
group before profiling.

Multi-SIMD thread trace with ``--att-remap``
============================================

On gfx1250 each pass traces one SIMD, and the hardware places workgroups differently in every
pass, so the SIMDs of one WGP run different work from pass to pass. ``--att-remap`` hands the
traced dispatch to the remapper (``projects/remapper``), which makes the traced WGP run the same
work in every pass:

.. code-block:: bash

   rocprofv3 --att --att-remap --att-target-cu 0 --kernel-include-regex <kernel> \
       --replay-mode kernel --kernel-replay-beta-enabled -- <application_path>

The remapper rewrites the kernel so each workgroup takes its tile from a placement table keyed by
where it landed, loads the rewritten code object through HSA, and runs the replay loop. It records
placement until several passes agree, builds the table and checks it, then traces SIMD 0 to 3 of
the target WGP on every shader engine: with the table, at the natural placement, and the
application's own kernel. The last pass always runs the application's dispatch, untraced, so the
application keeps its own results. If any stage fails, the loop goes straight to that pass and
the dispatch's ``report.json`` says why.

``--att-target-cu`` picks the WGP; the remapper picks each pass's SIMD and shader engines, so
``--att-simd-select`` and ``--att-shader-engine-mask`` are rejected, as is ``--pmc``.
``--att-remap-options`` passes remapper settings as ``key=value`` pairs (for example
``records=3,checks=4,repetitions=2,launch-model=auto``; ``include/remapper.h`` in the remapper
lists them). ``--att-remap-library`` points at ``libremapper.so`` when it is not installed next to
``rocprofv3``.

Each SIMD is traced through its own device thread trace context, started and stopped around the
passes that trace it, so every traced GPU allocates four thread trace buffers of
``--att-buffer-size``. A ``target-cu`` in ``--att-remap-options`` must name the same WGP as
``--att-target-cu``.

Each remapped dispatch writes ``att_remap/<pid>/dispatch-<id>/`` under the output directory: the
rewritten code object, ``manifest.json``, ``report.json`` and ``passes.json``, which says what
each pass ran and traced. After the application exits, ``rocprofv3`` runs the remapper's merge,
which arranges the per-pass traces and writes merged views and audits to
``dispatch-<id>/analysis/``. ``--att-remap-merge False`` skips it.

The remapper supports gfx1250 wave32 kernels. It does not rewrite kernels compiled for cluster
launches when they are launched without clusters (their workgroup id reads the cluster rank);
declare such kernels ``__attribute__((no_cluster))``.

Pass count
==========

The pass count is **not** a user-supplied integer. ``rocprofv3`` derives it per dispatch from the
number of counter groups collectable on **that dispatch's GPU agent**. Pass ``i`` maps to group
``i``. An agent with fewer collectable groups than the global ``--pmc`` list is replayed only as
many times as it has groups, so pass and group stay aligned. With ``--att-remap``, the remapper
decides how many passes a dispatch takes and what each one runs and traces.

There is no ``--kernel-replay-passes`` flag and no pass-count environment variable. The CLI does
not wire ``replay_continue`` or the localized start/stop context callbacks; those remain SDK
tool APIs (:ref:`using-kernel-replay`).

Output
======

All counter groups from one kernel-replay run are written together (there is no ``pass_n/``
directory per group, unlike application replay).

JSON
----

JSON counter records include a ``replay_pass`` field (0-based). All passes of one logical dispatch
share the same ``dispatch_id``; ``replay_pass`` is what distinguishes them. That identity is
enforced by the SDK: one dispatch id is reserved before the first pass and reused for every pass.

CSV
---

CSV ``counter_collection.csv`` uses the same columns as a non-replay run. There is **no**
``Replay_Pass`` column in this design. Passes of a replayed dispatch share ``Dispatch_Id``, and each
pass contributes the counters from its ``--pmc`` group, so a pass can be identified from
``Counter_Name`` only when the groups request distinct counters -- as in the example above. When a
counter is repeated across groups (a sanity counter such as ``SQ_WAVES`` in every group, which the
kernel-replay integration test does deliberately), the rows for one dispatch share both
``Dispatch_Id`` and ``Counter_Name``, and CSV alone cannot attribute them to a pass.

To attribute those rows, use ``--output-format json`` (or ``json`` together with ``csv``), where
``replay_pass`` is explicit.

Thread trace
------------

With ``--att-remap``, each traced pass is written and decoded separately. The raw files are named
``<agent>_pass_<pass>_shader_engine_<se>_<dispatch>.att``, and the decoded output of each pass goes
to ``ui_output_agent_<agent>_dispatch_<dispatch>_pass_<pass>/``. That directory also holds
``replay_target.json``, which records the pass and the ``target_cu``, ``simd_select`` and
``shader_engine_mask`` it traced.

Default ``rocpd``
-----------------

The default output format is ``rocpd``. Convert with ``rocpd convert`` as described in
:ref:`using-rocpd-output-format`.

What is snapshotted
===================

Snapshot and restore are implemented in the SDK. Between passes, kernel replay restores
coarse-grained device VRAM and module-scope ``__device__`` / ``__constant__`` variables. Unified,
managed, ``hipMallocAsync``, host, fine-grained, kernarg, and executable allocations are not
restored. Capture is a full in-memory copy; cost is ``O(tracked_bytes × passes)``. See
:ref:`kernel-replay-memory-snapshot` and :ref:`using-kernel-replay`.

The last executed pass is **not** restored, so the application sees the memory the kernel actually
produced.

Limitations (CLI)
=================

* **Beta.** The flag, the SDK API, and the output schema may change.
* **Requires** ``--pmc`` **or** ``--att``. The flag is an alternative to application replay for
  counter groups and thread trace targets, not a general "replay my kernel N times" switch.
* **Each** ``--pmc`` **group must fit one hardware pass.**
* **Fixed pass count** equal to the number of collectable groups on that agent, or to the number of
  thread trace targets. No ``replay_continue`` and no per-pass local-context toggles from the CLI.
* **Counters or thread trace, not both.** ``--att`` together with ``--pmc``, PC sampling, and
  ``--spm`` are rejected alongside ``--replay-mode kernel``. Because the CLI has no per-pass
  toggles, any other service would remain enabled for every pass and report each kernel once per
  pass, all under the single dispatch ID that replay reuses. The SDK itself is not restricted this
  way -- a custom tool can enable and disable services per pass through the local-context API (see
  :ref:`using-kernel-replay`) -- so this is a CLI limitation, not a hardware or SDK one.
* **HIP graph launches are not replayed.** A graph seen while replay is active warns once and
  runs un-replayed (not a hard error).
* **Only single-packet, single-dispatch submissions** are replayed.
* **Single process.** There is no MPI or cross-process coordination, and there is no multi-GPU
  end-to-end CLI test. Replaying a kernel that participates in an inter-process collective is
  unsafe. The SDK already selects groups per agent; what is missing is a multi-GPU / MPI test.
* **Async copies are not fenced** (SDK). An ``hsa_amd_memory_async_copy`` on another thread can
  mutate device memory during the replay window.
* **Stuck drains abort the process** (SDK, roughly 60 s).
* **Host RAM duplication** of the tracked device footprint. For large footprints, snapshot plus
  restore can cost more than re-running the application.

When to use kernel replay
=========================

Use it when:

* You need several counter groups on the **same** dispatches.
* You need thread traces of several SIMDs or WGPs of the **same** dispatch, for example every SIMD
  of a WGP on an architecture that traces one SIMD per run.
* Re-running the whole application per group is too slow, or the run is non-deterministic so
  application replay would not compare the same work.
* The kernels of interest write coarse-grained ``hipMalloc`` buffers (and optionally module-scope
  device variables), not unified/managed/async-pool memory.

Prefer application replay or counter-group rotation when:

* The tracked device footprint is huge (snapshot/restore bandwidth can dominate).
* The application uses HIP graphs, unified memory, or ``hipMallocAsync`` for the buffers kernels
  write.
* You only need one counter group, or you can rotate groups across successive dispatches.

See also
========

* :ref:`using-kernel-replay` — SDK callback how-to
* :ref:`kernel-replay-sdk-api` — payload and operations
* :ref:`kernel-replay-callback-api` — API contract
