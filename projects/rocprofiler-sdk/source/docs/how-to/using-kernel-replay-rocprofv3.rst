.. meta::
  :description: Using kernel replay with rocprofv3 to collect multiple counter groups in a single application run
  :keywords: rocprofv3, kernel replay, replay-mode, kernel-replay-beta-enabled, multi-pass counters, application replay, snapshot restore

.. _using-kernel-replay-rocprofv3:

======================================
Using kernel replay with rocprofv3
======================================

``rocprofv3 --replay-mode kernel --kernel-replay-beta-enabled`` collects several hardware-counter
groups in **one application run**. For each kernel dispatch, the profiler copies that GPU's tracked
device memory to the host, runs the dispatch once per ``--pmc`` group, and copies the memory back
between passes. Every group sees the same captured inputs.

.. code-block:: text

   one process, one dispatch of kernel K on one GPU

   host RAM                              GPU device memory
   +---------------------------+         +-------------------------------+
   | snapshot (full copy)      | <------ | hipMalloc / coarse-grained    |
   | held until the dispatch   | ------> | __device__ and __constant__   |
   | finishes                  |         +-------------------------------+
                                             restored before pass 1, 2, ...
                                             left as the last pass wrote it

   GPU time -------------------------------------------------------------------->
   pass 0: execute K, counters = group 0
           restore snapshot
   pass 1: execute K, counters = group 1
           restore snapshot
   pass 2: execute K, counters = group 2
   application continues; it observes one completion, and the memory of the last pass

The copy covers every tracked coarse-grained allocation owned by that dispatch's GPU, plus
module-scope ``__device__`` and ``__constant__`` variables. It is the GPU's tracked footprint, which
is wider than the buffers the kernel's source names. Host RAM use and copy time scale with that
footprint. Details of the inventory are in :ref:`kernel-replay-memory-snapshot`.

Without ``--replay-mode kernel``, several ``--pmc`` groups use :ref:`using-application-replay`: the
whole application runs again from the start, once per group.

.. warning::

   Kernel replay is **experimental**. Pass ``--kernel-replay-beta-enabled`` together with
   ``--replay-mode kernel``. The behavior, the SDK header under ``rocprofiler-sdk/experimental/``,
   and the output schema can change.

How it compares
===============

.. list-table::
   :header-rows: 1
   :widths: 24 28 24 24

   * - Approach
     - What runs again
     - Memory
     - Cost
   * - :ref:`Application replay <using-application-replay>` (default)
     - The whole process, once per group
     - None. Each run is a new process
     - ``N`` times the application runtime
   * - **Kernel replay**
     - One dispatch, in place
     - Device snapshot and restore between passes
     - ``N`` times the kernel, plus ``N`` snapshot copies
   * - Counter group rotation (``pmc_groups``)
     - Nothing. Later dispatches sample other groups
     - None
     - One application run. Groups are from different dispatches

Kernel replay collects counters. :ref:`using-spm` streams counter samples over time from hardware
ring buffers, which is a different measurement. :ref:`using-thread-trace` and :ref:`using-pc-sampling`
are rejected on this command line; a custom tool can enable them on selected passes
(:ref:`using-kernel-replay`).

Collecting counters
===================

``--replay-mode kernel`` requires counter collection (``--pmc``, an input-file ``pmc`` line, or
``pmc_groups``) and ``--kernel-replay-beta-enabled``. The number of passes is the number of counter
groups that GPU can collect. There is no pass-count flag.

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES GRBM_COUNT --pmc GRBM_GUI_ACTIVE --replay-mode kernel --kernel-replay-beta-enabled -- <application_path>

Both groups are collected in one run of ``<application_path>``. Each targeted dispatch runs twice.
Pass 0 collects ``SQ_WAVES`` and ``GRBM_COUNT``. The snapshot is restored. Pass 1 collects
``GRBM_GUI_ACTIVE``.

A single ``--pmc`` group still works. The tool asks for one pass, and the dispatch takes the
ordinary path: no snapshot and no restore.

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES GRBM_COUNT --replay-mode kernel --kernel-replay-beta-enabled -- <application_path>

List counters with ``rocprofv3 --list-avail`` or ``rocprofv3-avail``, and check a group with
``rocprofv3-avail pmc-check`` before a long run. See :ref:`using-rocprofv3-avail`. Each ``--pmc``
group must fit in one hardware pass. Kernel replay does not split a group the hardware cannot
collect together.

An input file's jobs each describe a run of their own. ``rocprofv3`` runs those jobs as it would
without ``--replay-mode kernel``, and replay applies inside each job to the groups that job asks
for.

Output
======

All groups from one kernel-replay run are written together. There is no ``pass_n/`` directory per
group. ``pass_n`` is the :ref:`application replay <using-application-replay>` layout.

JSON
----

JSON counter records include ``replay_pass`` (0-based). Every pass of one dispatch shares
``dispatch_id``. ``replay_pass`` is what distinguishes them. The SDK reserves that dispatch id
before the first pass and reuses it.

CSV
---

``counter_collection.csv`` has the same columns as a non-replay run. There is no ``Replay_Pass``
column. Passes share ``Dispatch_Id``, and each pass contributes the counters from its group, so the
counter name identifies the pass when the groups request different counters. When the same counter
is repeated in several groups, those rows share both ``Dispatch_Id`` and ``Counter_Name``. Use
``--output-format json``, or JSON together with CSV, so ``replay_pass`` is explicit.

rocpd
-----

The default output format is ``rocpd``. Convert with ``rocpd convert`` as described in
:ref:`using-rocpd-output-format`.

What memory comes back
======================

Between passes, kernel replay restores:

* Coarse-grained device allocations from ``hipMalloc``, ``hsa_amd_memory_pool_allocate``, and
  ``hsa_memory_allocate``. See
  `HIP memory management <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html>`_.
* Module-scope ``__device__`` and ``__constant__`` variables in loaded executables.

It leaves these as the application last wrote them:

* Unified and managed memory.
* ``hipMallocAsync`` and other stream-ordered pools.
* Host memory, fine-grained memory, kernarg, and executable allocations.

The copy is a full in-memory copy of every tracked region. Cost is proportional to tracked bytes
times passes. The last pass is left in place, so the application continues with the memory that
pass produced. :ref:`kernel-replay-memory-snapshot` is the inventory and the restore path.

When to use kernel replay
=========================

Use it when:

* You need several counter groups on the **same** dispatches.
* Re-running the whole application per group is too slow, or the run is not deterministic, so
  :ref:`application replay <using-application-replay>` would not compare the same work.
* The kernels of interest write coarse-grained ``hipMalloc`` buffers, and optionally module-scope
  device variables.

Prefer application replay or counter-group rotation when:

* The tracked device footprint is large. Snapshot and restore bandwidth can dominate a short kernel.
* The application uses HIP graphs, unified memory, or ``hipMallocAsync`` for the buffers kernels write.
* You only need one counter group, or later dispatches can sample other groups.

.. _kernel-replay-rocprofv3-advanced:

Advanced
========

Pass count
----------

The pass count is not a user-supplied integer. ``rocprofv3`` derives it per dispatch from the number
of counter groups collectable on **that dispatch's GPU**. Pass ``i`` maps to group ``i``. A GPU with
fewer collectable groups than the global ``--pmc`` list is replayed only as many times as it has
groups, so the pass index and the group stay aligned.

Filters
-------

``--kernel-include-regex``, ``--kernel-exclude-regex``, and ``--kernel-iteration-range`` are applied
per dispatch on the first pass. A dispatch they reject is still replayed, without counters.

``--selected-regions``
----------------------

A dispatch launched while ``roctxProfilerPause`` has stopped counter collection gets one pass. It
runs once, with no snapshot, no queue drain, and no counters. Replay resumes on the next dispatch
after ``roctxProfilerResume``. Marker ranges themselves are described in
:ref:`using-rocprofiler-sdk-roctx`.

Other services
--------------

``--att``, PC sampling, and ``--spm`` are rejected together with ``--replay-mode kernel``. The
command line has no per-pass switch, so any other service would stay enabled for every pass and
report each kernel once per pass, all under the one dispatch id replay reuses. Collect those in a
separate run, or from a custom tool that toggles contexts per pass (:ref:`using-kernel-replay`).

There is no ``--kernel-replay-passes`` flag. The CLI does not wire an early-exit callback or the
localized start/stop context callbacks. Those are SDK tool APIs.

Cost
----

For one replayed dispatch the tool pays:

* One drain of that GPU's queues, then one snapshot of its tracked device memory.
* One restore before every pass after the first.
* The kernel itself, once per group.

A phase of ``K`` dispatches pays that snapshot once per dispatch. A short kernel over a large pool
spends more time in the copy than on the GPU. ``ROCPROFILER_LOG_LEVEL=info`` logs the phase times
for a replayed dispatch. The fields and the nightly measurements are in
:ref:`kernel-replay-performance`.

Clocks and caches
-----------------

The tool does not pin GPU clocks and does not flush L1 or L2 between passes. A later pass can hit
in a cache the previous pass filled, and a long replay can run at a higher clock than the first
pass. Pin a performance level when you compare passes. See :ref:`gpu-performance-level`.
:ref:`Application replay <using-application-replay>` starts a new process, so its caches start cold.

Isolation
---------

The replay window holds that GPU's replay lock and drains its queues before the snapshot. Other
kernels on that GPU wait. Other GPUs use other locks.

These cases run once, with a warning, and are not replayed:

* A HIP graph launch.
* A submission that is more than one packet or more than one dispatch.

These cases are outside what the snapshot covers:

* An ``hsa_amd_memory_async_copy`` or a HIP async copy on another thread can write device memory
  during the window. The copy is not fenced.
* A kernel that participates in an inter-process collective. There is no MPI coordination, and
  replaying one rank's kernel leaves the other ranks waiting. Stay on application replay for that
  kernel.
* A drain that does not finish. The process aborts after about 60 seconds rather than continuing
  on a partial window.

Host memory holds a second copy of the tracked device footprint for the dispatch. Under host memory
pressure the snapshot is declined and the dispatch runs once.

See also
========

* :ref:`using-application-replay` — one process per counter group
* :ref:`using-kernel-replay` — the same service from a custom tool
* :ref:`kernel-replay-sdk-api` — payload and operations
* :ref:`kernel-replay-callback-api` — API contract
* :ref:`kernel-replay-memory-snapshot` — what ``snap()`` and ``restore()`` capture
* :ref:`kernel-replay-performance` — phase timings
* :ref:`kernel-counter-collection` — ``--pmc``, input files, and group rotation
* :ref:`gpu-performance-level` — pinning clocks
* :ref:`using-rocprofiler-sdk-roctx` — markers and ``roctxProfilerPause``
* `HIP memory management <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html>`_
