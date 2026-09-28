.. meta::
  :description: Using kernel replay with rocprofv3 to collect multiple counter groups in a single application run
  :keywords: rocprofv3, kernel replay, replay-mode, kernel-replay-beta-enabled, multi-pass counters, application replay, snapshot restore

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

Thread trace and counters on the same dispatch
==============================================

Adding ``--att`` gives each profiled dispatch one more pass, ahead of its counter passes, that
runs the dispatch thread trace with counter collection switched off. The counter groups then run
on the following passes with the thread trace switched off. Each dispatch is traced once, and its
counters are collected against the same restored inputs but without the trace's memory traffic
and serialization perturbing them.

.. code-block:: bash

   rocprofv3 --att --pmc SQ_WAVES GRBM_COUNT --pmc GRBM_GUI_ACTIVE --replay-mode kernel --kernel-replay-beta-enabled -- <application_path>

Each profiled dispatch runs three times: pass 0 is the thread trace, pass 1 collects
``SQ_WAVES`` and ``GRBM_COUNT``, and pass 2 collects ``GRBM_GUI_ACTIVE``. With a single ``--pmc``
group the dispatch is still replayed once, so the trace and the counters get separate passes.

Only the dispatch-scoped thread trace can be given a pass. ``--att-consecutive-kernels``,
``--att-no-intercept``, ``--selected-regions``, and ``--collection-period`` trace the device
rather than one dispatch, so they would capture every pass, and rocprofv3 rejects them together
with ``--replay-mode kernel`` and ``--att``. Kernel filtering behaves as it does for ``--att``
without replay: with no ``--kernel-iteration-range``, only the first launch of each kernel is
traced and has its counters collected. A dispatch that is not replayed (a HIP graph launch or a
multi-packet submission) collects the first counter group and is not traced.

Pass count
==========

The pass count is **not** a user-supplied integer. ``rocprofv3`` derives it per dispatch from the
number of counter groups collectable on **that dispatch's GPU agent**, plus one for the thread
trace pass with ``--att``. Pass ``i`` maps to group ``i`` (group ``i - 1`` with ``--att``, whose
thread trace takes pass 0). An agent with fewer collectable groups than the global ``--pmc`` list
is replayed only as many times as it has groups, so pass and group stay aligned.

A dispatch that the kernel filters (``--kernel-include-regex``, ``--kernel-exclude-regex``,
``--kernel-iteration-range``) do not select is not replayed at all: it runs once on the ordinary
path, without a snapshot.

There is no ``--kernel-replay-passes`` flag and no pass-count environment variable. The CLI does
not wire ``replay_continue``; it uses the localized start/stop context callbacks only to separate
the thread trace pass from the counter passes. Both remain SDK tool APIs
(:ref:`using-kernel-replay`).

Output
======

All counter groups from one kernel-replay run are written together (there is no ``pass_n/``
directory per group, unlike application replay).

JSON
----

JSON counter records include a ``replay_pass`` field (0-based). All passes of one logical dispatch
share the same ``dispatch_id``; ``replay_pass`` is what distinguishes them. That identity is
enforced by the SDK: one dispatch id is reserved before the first pass and reused for every pass.
``replay_pass`` numbers the counter group a record collected, so it runs ``0..N-1`` with or
without ``--att``; the thread trace pass produces no counter record.

Thread trace output is written as it is without replay, once per profiled dispatch and named by
its ``dispatch_id``, so it can be joined to the counter records of the same dispatch.

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
* **Requires** ``--pmc``. The flag is an alternative to application replay for counter groups, not
  a general "replay my kernel N times" switch.
* **Each** ``--pmc`` **group must fit one hardware pass.**
* **Fixed pass count** equal to the number of collectable groups on that agent, plus one with
  ``--att``. No ``replay_continue`` from the CLI.
* **Counters and dispatch thread trace only.** PC sampling and ``--spm`` are rejected alongside
  ``--replay-mode kernel``. The CLI only switches counter groups and the thread trace per pass,
  so any other service would remain enabled for every pass and report each kernel once per pass,
  all under the single dispatch ID that replay reuses. PC sampling is agent-wide and does not
  honor the SDK's per-pass toggles at all, so even a custom tool cannot confine it to one pass
  (see :ref:`using-kernel-replay`).
* **Thread trace runs on its own pass**, never alongside counter collection, and only in dispatch
  mode. Device-mode thread trace options are rejected with ``--replay-mode kernel`` (see
  `Thread trace and counters on the same dispatch`_). ``--att-perfcounters`` and
  ``--att-activity`` remain rejected together with ``--pmc``, as they are without replay.
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
