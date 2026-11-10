.. meta::
  :description: Application replay, kernel replay, and range replay in rocprofv3, with memory and GPU-execution diagrams
  :keywords: rocprofv3, application replay, range replay, range-host, replay-mode, roctx, kernel replay

.. _using-range-replay-rocprofv3:

================================================
Using range replay with rocprofv3
================================================

``rocprofv3`` has three ways to collect more hardware-counter groups than fit in one pass.
Application replay runs the process again. Kernel replay runs one dispatch again. Range replay
runs a marked phase again. The first two are documented in full under :ref:`kernel-counter-collection`
and :ref:`using-kernel-replay-rocprofv3`. This page starts from application replay, then covers
range replay.

.. warning::

   ``--replay-mode range`` and ``--replay-mode range-host`` are **experimental**. Pass
   ``--kernel-replay-beta-enabled`` with either one. The flags and the SDK header
   ``rocprofiler-sdk/experimental/range_replay.h`` can change. A phase that packet replay cannot
   reproduce runs once, and the extra counter groups are not collected.

Application replay
==================

Application replay is the default. Each ``--pmc`` flag is one counter group, and the application
runs once per group in a new process. The tool does not copy device memory. The GPU executes every
kernel once per process, with the stream overlap the application asked for.

.. code-block:: text

   pass 1, one process                          pass 2, a new process
   host: startup and the whole run              host: startup and the whole run
   GPU:  A ---- B ---- C                        GPU:  A ---- B ---- C
         group 1                                      group 2
   device memory is this process's              device memory is the new process's
   output: pass_1/                              output: pass_2/

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES --pmc GRBM_COUNT --replay-mode application -- <application_path>

``--replay-mode application`` is optional. Omitting ``--replay-mode`` does the same thing.
Dispatch IDs start over in each ``pass_n`` directory. Rows match across passes only when the
application launches the same kernels, on the same devices, in the same order, every time.
The layout and the input-file rules are in :ref:`kernel-counter-collection`.

Use this when the kernels depend on host progress during the launch, when you want the cache
state earlier kernels left behind, or when a device snapshot would cost more than running the
application again.

Kernel replay, for one dispatch
===============================

:ref:`Kernel replay <using-kernel-replay-rocprofv3>` stays in one process. For each dispatch it
copies that GPU's tracked device memory aside, runs the dispatch once per group, and copies the
memory back between passes.

.. code-block:: text

   one process, one dispatch of K
   GPU:  [pass 0, group 0]  restore  [pass 1, group 1]  restore  [pass 2, group 2]
   device memory of that GPU is copied to host RAM and written back between passes
   the application observes one completion, and the memory of the last pass

That granularity fits a single kernel. It does not fit a phase whose kernels feed each other. The
second kernel of a phase reads what the first one wrote. Re-running the first kernel alone, after
the phase has moved on, measures a different input than the phase produced.

Range replay
============

Range replay repeats the phase. You mark the start and the end. One device-memory snapshot covers
the whole phase, and every ``--pmc`` group is collected inside one run of the application.

Two modes select what "repeat the phase" means.

.. list-table::
   :header-rows: 1
   :widths: 18 28 27 27

   * - ``--replay-mode``
     - What runs again
     - Device memory
     - GPU execution inside a pass
   * - ``application``
     - The whole process
     - Not copied
     - The application's streams, unchanged
   * - ``kernel``
     - One dispatch
     - That GPU, restored between passes
     - That one dispatch
   * - ``range``
     - The recorded dispatches
     - The one GPU the range bound to, restored between passes
     - Those dispatches, one after another
   * - ``range-host``
     - Your code between the markers
     - Every GPU, plus host memory you register
     - Your streams, with the overlap you wrote

Packet replay (``range``)
-------------------------

The application's own run of the phase is pass 0. The SDK records the dispatches. After the phase
closes it restores device memory to the state from before the first recorded dispatch, then submits
the recording again, once per extra group.

.. code-block:: text

   application thread                         GPU (one queue)
   range begin
     kernel A  ---------------------------->  A runs          (pass 0, group 0)
     kernel B  ---------------------------->  B runs, reads A's output
   range end
     snapshot was taken before A ran
     restore that snapshot
     re-submit A, then B  ----------------->  A runs          (pass 1, group 1)
                                              B runs
     restore the memory pass 0 left behind
   application continues with pass 0's results

Recorded dispatches in an extra pass are submitted one at a time, with a barrier between them.
Pass 0 keeps the overlap the application submitted. Later passes do not. A measurement that
depends on two kernels running together belongs on ``range-host`` or on application replay.

Host replay (``range-host``)
----------------------------

Nothing is recorded. ``rocprofiler_range_replay_end`` restores memory and jumps back to the marker,
and your code runs again. Copies, memsets, several streams, HIP graph launches, and host computation
are part of every pass, because they are your code.

.. code-block:: text

   ROCPROFILER_RANGE_REPLAY_HOST_BEGIN  <---------------------------+
     your code: kernels, copies, graphs, host math                  |
     several streams overlap the way you wrote them                |
   rocprofiler_range_replay_end                                     |
     more groups left?                                              |
       restore every GPU's tracked device memory                    |
       restore host regions you registered                          |
       jump back  --------------------------------------------------+
     otherwise return; the last pass's memory is what you continue with

``rocprofv3`` does not insert the jump. The application opens the range with the macro below.
``--replay-mode range-host`` is what tells the tool to set host replay on.

Commands
========

Both modes require ``--pmc`` (or an input-file counter group) and ``--kernel-replay-beta-enabled``.
``--att``, PC sampling (:ref:`using-pc-sampling`), and ``--spm`` (:ref:`using-spm`) are rejected,
for the same reason as kernel replay: this command line cannot turn a service on for one pass and
off for the next.

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES --pmc TCC_HIT --replay-mode range --kernel-replay-beta-enabled -- ./app

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES --pmc TCC_HIT --replay-mode range-host --kernel-replay-beta-enabled -- ./app

The number of passes is the number of ``--pmc`` groups. Pass 0 is the application's own run and
collects the first group. Each later pass collects the next group. One group asks for one pass,
and the phase runs once with no snapshot.

Marking the phase
=================

Packet replay (``--replay-mode range``) treats an outermost ROCTx range as the phase. Nested
ranges stay markers. The marker API is :ref:`using-rocprofiler-sdk-roctx`.

.. code-block:: cpp

   #include <rocprofiler-sdk-roctx/roctx.h>

   roctxRangePush("train_step");
   train_step();                  // the dispatches in this call are the phase
   roctxRangePop();

``roctxRangeStart`` / ``roctxRangeStop`` mark a phase the same way. The range that is replayed is
the outermost one on that thread. An application can also call ``rocprofiler_range_replay_begin``
and ``rocprofiler_range_replay_end`` itself. Ranges are per thread and do not nest. Only dispatches
submitted by the thread that opened the range are part of a packet replay.

Host replay ignores ROCTx ranges. Open it in the function that owns the phase:

.. code-block:: cpp

   #include <rocprofiler-sdk/experimental/range_replay.h>

   rocprofiler_status_t status = ROCPROFILER_STATUS_SUCCESS;
   hipDeviceSynchronize();
   rocprofiler_range_replay_host_track_memory(&step_state, sizeof(step_state));
   ROCPROFILER_RANGE_REPLAY_HOST_BEGIN(1, status);

   for(int step = 0; step < n_steps; ++step)
       time_step(step);

   hipDeviceSynchronize();
   rocprofiler_range_replay_end();   // jumps back for each extra group, then returns

``status`` is the result of the open on the first entry, and success on every re-entry. The rules
for the code between the macro and ``end`` are under :ref:`range-replay-host-contract`.

What memory is restored
=======================

.. code-block:: text

   packet replay, one GPU                    host replay, every GPU
   host RAM                                  host RAM
   +----------------------+                  +----------------------+
   | entry snapshot       |                  | entry snapshot       |
   | exit snapshot        |                  | registered host bytes|
   +----------------------+                  +----------------------+
            |  full copy                              |  full copy
            v                                         v
   GPU device memory                         GPU device memory, each GPU
   hipMalloc / coarse-grained                hipMalloc / coarse-grained
   __device__ and __constant__               __device__ and __constant__

   not restored, either mode:
     unified memory, managed memory, hipMallocAsync
     fine-grained memory, host memory you did not register
     files, sockets, HIP runtime host state, the clock

Packet replay takes two device snapshots of the one GPU the range bound to. The entry snapshot is
what each extra pass rewinds to. The exit snapshot is what pass 0 produced, and it is written back
after the last extra pass so the application continues with its own results.

Host replay snapshots every GPU, because the phase may touch any of them. It also copies back the
host regions registered with ``rocprofiler_range_replay_host_track_memory``. The last pass is your
code as well, so host and device state already match and there is no separate exit restore.
Register a region while no range is open and it is captured when the next range on that thread
opens. Register it during pass 0 and it is captured at the call. Registrations during later passes
are ignored.

`HIP memory management <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html>`_
describes the allocation kinds above. The snapshot inventory is the one kernel replay uses:
:ref:`kernel-replay-memory-snapshot`.

Output
======

One run writes one directory. There is no ``pass_n/`` directory; that layout is application replay.

JSON counter records include ``replay_pass``. Pass 0 is the application's run and the first
``--pmc`` group. Unlike kernel replay, each execution is its own dispatch, so ``dispatch_id`` is
not shared across passes. ``replay_pass`` is the group.

CSV ``counter_collection.csv`` has no ``Replay_Pass`` column. When each group names different
counters, the counter name is the group. When the same counter appears in more than one group, use
``--output-format json``.

The default format is ``rocpd``. See :ref:`using-rocpd-output-format`.

A phase that is declined still produces pass 0. The extra groups are absent. ``rocprofv3`` does not
write the decline reason into the counter file. ``ROCPROFILER_LOG_LEVEL=info`` logs the snapshot and
restore times for a phase that did replay; the fields are in :ref:`range-replay-performance`. A
custom tool reads the reason from the ``CLOSE`` callback (:ref:`using-range-replay`).

When packet replay declines the phase
======================================

Packet replay declines a phase it cannot reproduce, runs pass 0, and skips the extra passes. The
application's results are unchanged.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - You will see this when
     - Status a custom tool reads
   * - Dispatches on two queues (two HIP streams that did not share a hardware queue)
     - ``MULTI_QUEUE``
   * - Dispatches on two GPUs
     - ``MULTI_AGENT``
   * - A HIP graph launch inside the phase
     - ``GRAPH_LAUNCH``
   * - A copy or memset that writes device memory
     - ``MEMORY_COPY_IN_RANGE``
   * - Another thread launches to the same GPU, or copies to it
     - ``CONCURRENT_DISPATCH``
   * - ``hipMalloc`` or ``hipFree`` of device memory inside the phase
     - ``ALLOCATION_CHANGED_IN_RANGE``
   * - A kernel from a module loaded inside the phase, or a module unloaded
     - ``CODE_OBJECT_CHANGED_IN_RANGE``
   * - More than 4096 dispatches
     - ``PROGRAM_TOO_LARGE``
   * - No kernel was launched
     - ``NO_DISPATCH``
   * - Queue interposition is enabled
     - ``UNSUPPORTED_QUEUE_PATH``

The first reason wins. :ref:`range-replay-soundness` is why each row is a decline, and the full
status list.

Host replay accepts several queues, several GPUs, copies and memsets by the phase's own thread,
HIP graph launches, and balanced ``hipMalloc`` / ``hipFree`` pairs. It still declines another
thread's launches or copies, a device allocation that leaks or frees a buffer that existed before
the phase, a module unload, a missing jump buffer (``HOST_JMP_NOT_SET``), and a snapshot that could
not be allocated.

Launch every kernel once before the measured phase. HIP loads a module on the first launch of one
of its kernels, and a library may allocate on first use. Do that work in a warmup step.

Choosing a mode
===============

Use ``range`` when the phase is kernels on one stream of one GPU, those kernels read and write
coarse-grained device memory, and you can accept serialized extra passes.

Use ``range-host`` when the phase has several streams, copies, graphs, or host values the kernels
read, and you can follow :ref:`range-replay-host-contract`.

Use :ref:`kernel replay <using-kernel-replay-rocprofv3>` when the unit you care about is one
dispatch.

Use application replay when the phase talks to another process, when the kernels wait on the host
in a way the jump would not replay, or when you want each group in a fresh process.

.. _range-replay-rocprofv3-advanced:

Advanced
========

.. _range-replay-host-contract:

The host-replay contract
------------------------

``siglongjmp`` restores registers and the stack pointer. It does not run C++ destructors. The code
between ``ROCPROFILER_RANGE_REPLAY_HOST_BEGIN`` and ``rocprofiler_range_replay_end`` has to follow
these rules. :ref:`range-replay-host` is the same contract with the SDK calls named.

#. Same thread, same function. The frame that expanded the macro is still live at ``end``.
#. No C++ object with a destructor is live across the jump. Put those objects in a nested block
   that ends before ``end``, or in a function the phase calls.
#. A local of that function which the phase modifies is ``volatile``, or it is set again inside
   the phase. A counter declared inside the phase is fine.
#. Host memory the phase reads and writes is registered with
   ``rocprofiler_range_replay_host_track_memory``. Unregistered host memory, files, sockets, and
   random-number state are left as they are.
#. Before ``begin``, and before ``end`` on every pass, wait for all GPU work on every GPU the
   phase uses. That includes async copies and non-compute queues. ``hipDeviceSynchronize`` on each
   of those devices is the usual call. The SDK fences compute queues; it does not wait for a copy
   it does not track.
#. No other thread submits GPU work or copies into device memory while the phase is open.
#. ``hipMalloc`` / ``hipFree`` pairs inside the phase are balanced. ``hipMallocAsync`` caches
   across frees and is not a balanced pair.
#. Once-only work stays on pass 0. ``rocprofiler_range_replay_get_host_pass`` returns 0 during the
   application's run and 1, 2, ... during the extra passes.

Pass count on more than one GPU
-------------------------------

A range has one pass count, chosen when it opens, before it has bound to a GPU. ``rocprofv3`` uses
the widest counter-group list any GPU was given. Kernel replay can shorten the count per GPU.
Range replay cannot. Give every GPU in the phase the same groups. A GPU with a shorter list
repeats an earlier group on the extra passes.

``--selected-regions``
----------------------

The pass count is fixed when the range opens. A range opened while ``roctxProfilerPause`` has
stopped counter collection asks for one pass and is not replayed. Kernel replay makes this decision
per dispatch; range replay makes it once, at the marker. See :ref:`using-rocprofiler-sdk-roctx`.

Checking that the passes match
------------------------------

``ROCPROF_RANGE_REPLAY_VERIFY=1`` hashes device memory (and, for host replay, registered host
memory) after pass 0 and after the last pass. A custom tool reads ``divergence_count`` on ``CLOSE``.
A non-zero count means some region differed, so the passes did not see the same inputs.

The check is byte-exact. Kernels that use floating-point atomics, or reductions whose order follows
wave scheduling, can differ with identical inputs. Validate the phase with deterministic kernels,
then turn the variable off. It costs an extra read of the snapshot. :ref:`range-replay-soundness`
covers what a zero count does and does not prove.

Cost
----

Packet replay pays one snapshot of the phase's GPU per range, plus one restore per extra pass.
Kernel replay would have paid that once per dispatch. The saving grows with the number of
dispatches in the phase. A phase of one dispatch is kernel replay plus the recording overhead.

Host replay pays one snapshot of every GPU, held until the range closes, and one restore of that
snapshot plus a copy of each registered host region per extra pass. The passes themselves run at
the application's speed.

Shrink the phase until the snapshot is small next to the phase's runtime. Pools the phase does not
touch are still copied if they are tracked coarse-grained allocations on that GPU.

Clocks and caches
-----------------

The tool does not pin clocks and does not flush L1 or L2 between passes. A later pass can hit in a
cache the previous pass filled. Pin a performance level when you compare groups. See
:ref:`gpu-performance-level`.

Several GPUs and collectives
----------------------------

Packet replay is one GPU and one queue. Host replay restores every GPU's tracked device memory, and
the phase's own thread may launch to all of them, as long as no other thread does.

A collective across processes is not replayed safely. One rank jumping back into the collective
waits for a peer that has moved on. Bracket the compute between the communication, or use
application replay so every rank runs again.

What this command line does not expose
--------------------------------------

``rocprofv3`` does not offer an early-exit callback or per-pass context toggles. A tool that needs
those, or that needs the decline reason in its own log, uses :ref:`using-range-replay`.

See also
========

* :ref:`kernel-counter-collection` — application replay, ``pass_n``, and input files
* :ref:`using-kernel-replay-rocprofv3` — one dispatch in process
* :ref:`using-range-replay` — the same service from a custom tool
* :ref:`range-replay-sdk-api` — payload and operations
* :ref:`range-replay-conceptual` — the mechanism
* :ref:`range-replay-host` — host replay implementation and tests
* :ref:`range-replay-soundness` — decline rules
* :ref:`range-replay-performance` — the cost model and the info-log lines
* :ref:`kernel-replay-memory-snapshot` — which allocations are copied
* :ref:`using-rocprofiler-sdk-roctx` — ``roctxRangePush`` and pause/resume
* :ref:`gpu-performance-level` — pinning clocks
* `HIP memory management <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html>`_
