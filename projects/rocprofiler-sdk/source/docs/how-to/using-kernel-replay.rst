.. meta::
  :description: Using kernel replay with rocprofv3 to collect multiple counter groups in a single application run
  :keywords: rocprofv3, kernel replay, kernel-replay-beta-enabled, multi-pass counters, application replay, snapshot restore

.. _using-kernel-replay:

===================
Using kernel replay
===================

Kernel replay is a beta ``rocprofv3`` feature that collects several hardware-counter groups in
**one application run**. For each kernel dispatch, the profiler snapshots the device memory the
kernel can mutate, re-executes that dispatch once per ``--pmc`` group, and restores the snapshot
between passes so every group observes identical inputs.

Without the flag, multiple ``--pmc`` groups use *application replay*: the whole application is
re-run from start to finish once per group. Kernel replay is useful when those full re-runs are
expensive or non-deterministic.

.. warning::

   Kernel replay is **experimental**. The ``rocprofv3`` flag is ``--kernel-replay-beta-enabled``
   and the SDK header lives under ``rocprofiler-sdk/experimental/``. Both are expected to change
   before a stable release. Use it when you understand the :ref:`kernel-replay-limitations`
   below.

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
     - ``O(N ×`` application runtime``)``
   * - **Kernel replay** (``--kernel-replay-beta-enabled``)
     - One dispatch, re-executed in place
     - Device memory snapshot and restore between passes
     - ``O(N ×`` kernel time ``+ N ×`` snap/restore``)``
   * - Counter group rotation (``pmc_groups`` / ``pmc_group_interval``)
     - Amortized across successive dispatches
     - None; different dispatches sample different groups
     - ``O(1 ×`` application runtime``)``, but not the same dispatch

Kernel replay is **not** the same as :ref:`using-spm`. SPM streams counter samples over time from
hardware ring buffers; kernel replay re-executes a dispatch so each pass can collect a different
counter group against the same inputs.

Collecting counters with kernel replay
======================================

``--kernel-replay-beta-enabled`` requires ``--pmc``. The number of replay passes is the number of
``--pmc`` groups (one pass per group). There is no separate pass-count flag.

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES GRBM_COUNT --pmc GRBM_GUI_ACTIVE --kernel-replay-beta-enabled -- <application_path>

The preceding command collects both counter groups in a **single** run of ``<application_path>``.
Each targeted dispatch is replayed twice: pass 0 collects ``SQ_WAVES`` and ``GRBM_COUNT``; pass 1
collects ``GRBM_GUI_ACTIVE``. Device memory is restored between those passes.

A single ``--pmc`` group still works. In that case the tool asks the SDK for one pass, which does
not snapshot or restore — the dispatch takes the ordinary path.

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES GRBM_COUNT --kernel-replay-beta-enabled -- <application_path>

List counters first with ``rocprofv3 --list-avail`` or ``rocprofv3-avail list --pmc``. Each
individual ``--pmc`` group must still fit in one hardware pass; kernel replay does not split a
group that the hardware cannot collect together. Use ``rocprofv3-avail pmc-check`` to verify a
group before profiling.

Pass count
==========

The pass count is **not** a user-supplied integer. ``rocprofv3`` derives it per dispatch from the
number of counter groups collectable on **that dispatch's GPU agent**. Pass ``i`` maps to group
``i``. An agent with fewer collectable groups than the global ``--pmc`` list is replayed only as
many times as it has groups, so pass and group stay aligned.

There is no ``--kernel-replay-passes`` flag and no pass-count environment variable.

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
``Replay_Pass`` column in this design (an earlier prototype added one). Passes of a replayed
dispatch share ``Dispatch_Id`` and are distinguished by ``Counter_Name``: each pass contributes
the counters from its ``--pmc`` group.

To inspect pass identity, use ``--output-format json`` (or ``json`` together with ``csv``).

Default ``rocpd``
-----------------

The default output format is ``rocpd``. Convert with ``rocpd convert`` as described in
:ref:`using-rocpd-output-format`.

What is snapshotted
===================

Between passes, kernel replay restores:

* Coarse-grained device (VRAM) allocations from ``hipMalloc`` /
  ``hsa_amd_memory_pool_allocate`` / ``hsa_memory_allocate``.
* Module-scope ``__device__`` and ``__constant__`` variables in loaded executables.

It does **not** restore:

* Unified or managed memory.
* ``hipMallocAsync`` and other pool-backed, stream-ordered allocations (virtual-memory map/unmap
  is not tracked).
* Host, fine-grained, kernarg, or executable allocations (excluded by design).
* HIP graph-managed memory.

Capture is a **full in-memory copy** of every tracked region into host RAM. There is no dirty-page
hashing in this version. For large footprints, snapshot plus restore can cost more than re-running
the application. Host-side and/or device-side hashing of dirty regions is expected in a future
version; see :ref:`kernel-replay-memory-snapshot`.

The last executed pass is **not** restored, so the application sees the memory the kernel actually
produced.

.. _kernel-replay-limitations:

Limitations
===========

* **Beta.** The flag, the SDK API, and the output schema may change.
* **Requires** ``--pmc``. The flag is an alternative to application replay for counter groups, not
  a general "replay my kernel N times" switch in ``rocprofv3``.
* **Each** ``--pmc`` **group must fit one hardware pass.**
* **Coarse-grained device VRAM only**, plus module-scope device/constant variables. Unified,
  managed, ``hipMallocAsync``, host, fine-grained, and kernarg memory are not restored.
* **HIP graph launches are not supported.** A graph seen while replay is active warns once and
  runs un-replayed; a graph dispatch that reaches the replay gate is a hard error.
* **Only single-packet, single-dispatch submissions** are replayed.
* **Single process.** There is no MPI or cross-process coordination. Replaying a kernel that
  participates in an inter-process collective is unsafe.
* **Host RAM duplication.** The tracked device footprint is copied into host memory for the
  duration of the replay.
* **Async copies are not fenced.** An ``hsa_amd_memory_async_copy`` (or HIP async memcpy) on
  another thread can mutate device memory during the replay window.
* **Stuck drains abort the process** rather than hanging or proceeding on questionable state
  (roughly 60 s bounds inside the replay window).

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

Writing a custom tool
=====================

``rocprofv3`` is one client of a callback-tracing domain
(``ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY``). A custom tool can replay dispatches for
counters, timing, PC sampling, or thread trace, and can enable or disable contexts per pass.
See :ref:`kernel-replay-callback-api` and :ref:`kernel-replay-sdk-api`.
