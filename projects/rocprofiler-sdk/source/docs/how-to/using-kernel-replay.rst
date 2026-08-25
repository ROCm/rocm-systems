.. meta::
  :description: Using the ROCprofiler-SDK kernel replay callback tracing domain from a custom tool
  :keywords: rocprofiler-sdk, kernel replay, callback tracing, KERNEL_REPLAY, pass_count_cb, local context

.. _using-kernel-replay:

===================
Using kernel replay
===================

Kernel replay is an experimental ROCprofiler-SDK **callback tracing domain**. A custom tool
subscribes to ``ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY``, tells the SDK how many times to
re-execute a dispatch, and the SDK snapshots tracked device memory, runs that many passes, and
restores the snapshot between them so every pass observes identical captured inputs.

The domain is not tied to hardware counters. A tool can use it for counters, timing, PC sampling,
thread trace, or any other per-pass work.

This page is the SDK how-to. Conceptual and API detail live under :ref:`kernel-replay-conceptual`
and :ref:`kernel-replay-sdk-api`. Command-line ``rocprofv3`` wiring
(``--kernel-replay-beta-enabled``) is the stacked tool integration PR, not this SDK change.

.. warning::

   Kernel replay is **experimental**. The public header is
   ``<rocprofiler-sdk/experimental/kernel_replay.h>``. The domain, payload, and limitations
   below are expected to change before a stable release.

Configure the domain
====================

There is no dedicated ``rocprofiler_configure_kernel_replay_*`` entry point. Subscribe through
ordinary callback tracing:

.. code-block:: cpp

   rocprofiler_context_id_t ctx{};
   rocprofiler_create_context(&ctx);

   rocprofiler_configure_callback_tracing_service(
       ctx,
       ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
       nullptr,  // all operations: CONFIG and PASS
       0,
       tool_kernel_replay_callback,
       nullptr);

   rocprofiler_start_context(ctx);

Configuring the domain also enables the device-allocation tracker used for snapshot and restore.
A process that never configures the domain does not pay that tracking cost. Only one context may
subscribe; a second configure call returns ``ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED``.

Set ``pass_count_cb``
=====================

``CONFIG`` ``PHASE_ENTER`` is where the tool installs the pass-count callback. The SDK calls
that callback after ``CONFIG`` ``PHASE_ENTER`` returns:

.. code-block:: cpp

   void
   tool_kernel_replay_callback(rocprofiler_callback_tracing_record_t record,
                               rocprofiler_user_data_t* /* user_data */,
                               void* /* callback_args */)
   {
       if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY) return;

       auto* payload =
           static_cast<rocprofiler_callback_tracing_kernel_replay_data_t*>(record.payload);

       if(record.operation == ROCPROFILER_KERNEL_REPLAY_CONFIG &&
          record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
       {
           payload->pass_count_cb = tool_pass_count_callback;
           // optionally: payload->replay_continue_cb = tool_continue_callback;
       }
       else if(record.operation == ROCPROFILER_KERNEL_REPLAY_PASS &&
               record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
       {
           // payload->current_pass is 0-based; payload->total_passes is N (or 0 if indefinite)
       }
   }

   uint64_t
   tool_pass_count_callback(rocprofiler_kernel_dispatch_info_t dispatch_info,
                            rocprofiler_user_data_t /* user_data */)
   {
       // Return 1 to skip replay for this dispatch (ordinary single execution, no snapshot).
       // Return N > 1 for a fixed loop. Return 0 only with replay_continue_cb set.
       return groups_for_agent(dispatch_info.agent_id);
   }

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - ``pass_count_cb``
     - ``replay_continue_cb``
     - Behavior
   * - left ``NULL``
     - ignored
     - Not replayed. Ordinary path, no snapshot.
   * - returns ``1``
     - ignored
     - Not replayed. One pass needs no snapshot.
   * - returns ``N > 1``
     - ``NULL``
     - Exactly ``N`` passes.
   * - returns ``N > 1``
     - provided
     - Up to ``N`` passes; continue callback may break early.
   * - returns ``0``
     - provided
     - Indefinite loop until continue callback returns zero.
   * - returns ``0``
     - ``NULL``
     - Rejected: the SDK warns and the dispatch is not replayed.

One dispatch id is reserved before ``CONFIG`` and reused for every pass. The application observes
exactly one kernel completion regardless of pass count.

Localized context control
=========================

During ``PASS`` ``PHASE_ENTER`` the payload carries ``replay_local_start_context_cb`` /
``replay_local_stop_context_cb``. Use them to enable or disable other contexts for that pass
(for example counters every pass, PC sampling once) without calling global
``rocprofiler_start_context`` / ``rocprofiler_stop_context``, which would leak into non-replayed
dispatches.

Overrides are sticky across passes and scoped to the replay loop. See
:ref:`kernel-replay-callback-api` for the contract.

In-tree examples
================

* ``tests/kernel-replay-concurrency/`` — custom client that replays one kernel and opts another
  out, asserting concurrent non-replayed work is not corrupted by snapshot/restore.
* ``tests/kernel-replay-local-context/`` — custom client that starts/stops per-service contexts
  across passes.
* ``source/lib/rocprofiler-sdk/kernel_replay/tests/`` — unit tests for configure, local context,
  and snap/restore.

What is snapshotted
===================

Between passes, kernel replay restores:

* Coarse-grained device (VRAM) allocations from ``hipMalloc`` /
  ``hsa_amd_memory_pool_allocate`` / ``hsa_memory_allocate``.
* Module-scope ``__device__`` and ``__constant__`` variables in loaded executables.

It does **not** restore:

* Unified or managed memory.
* ``hipMallocAsync`` and other pool-backed, stream-ordered allocations.
* Host, fine-grained, kernarg, or executable allocations (excluded by design).
* HIP graph-managed memory.

Capture is a **full in-memory copy** of every tracked region into host RAM. Cost is
``O(tracked_bytes × passes)``, not kernel time alone. The last executed pass is **not** restored,
so the application sees the memory the kernel actually produced.

.. _kernel-replay-declined:

When a dispatch is not replayed
===============================

Requesting replay does not guarantee it happens. The SDK declines when it cannot replay soundly or
affordably, runs the dispatch once with its original completion signal, and logs one
``[kernel-replay]`` line per dispatch giving the outcome, the reason, the snapshot footprint, the
untracked byte counts, and the snap/restore timing. A tool that needs to know whether a dispatch was
actually replayed should count ``PASS`` callbacks rather than assume the requested pass count was
honored.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Reason
     - What it means and what to do
   * - ``untracked_memory``
     - Live virtual-memory mappings on the agent, whose contents the snapshot cannot cover. This is
       what ``hipMallocAsync`` on the default pool, ``hipMemAddressReserve``/``hipMemMap``, PyTorch
       ``PYTORCH_HIP_ALLOC_CONF=expandable_segments``, and Kokkos built with
       ``KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC`` all produce. Switch the application to an allocator
       whose memory is snapshottable, or set
       ``ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_VMEM=0`` to replay anyway and accept that the results
       may be wrong.
   * - ``footprint_budget``
     - The snapshot would exceed the host budget (by default half of ``MemAvailable``). Narrow the
       replay with kernel filtering, reduce the application's resident footprint, or raise
       ``ROCPROFILER_KERNEL_REPLAY_MAX_SNAPSHOT_BYTES``.
   * - ``snapshot_failed``
     - A host allocation or a device-to-host copy failed during capture. Usually host memory
       pressure.
   * - ``queue_drain_stuck``, ``agent_drain_stuck``, ``pass_drain_stuck``
     - GPU work did not complete within the window's bound (roughly 60 s). Persistent, cooperative
       and peer-dependent kernels behave this way legitimately; exclude them with kernel filtering.

Tuning knobs, all optional:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Variable
     - Effect
   * - ``ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_VMEM``
     - Decline when live virtual-memory mappings are found. Default ``1``.
   * - ``ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_UNTRACKED_POOL``
     - Decline when GPU-resident, non-snapshottable pool allocations are found. Default ``0``,
       because runtime-internal allocations land in the same class.
   * - ``ROCPROFILER_KERNEL_REPLAY_MAX_SNAPSHOT_BYTES``
     - Snapshot budget in bytes. ``0`` means unlimited. Default: half of ``MemAvailable``.
   * - ``ROCPROFILER_KERNEL_REPLAY_WARN_SECONDS``
     - Warn when the projected host-link traffic for the window exceeds this. ``0`` disables.
   * - ``ROCPROFILER_KERNEL_REPLAY_ASSUMED_GBPS``
     - Bandwidth assumed when projecting that time. The snapshot destination is unpinned host
       memory, so the default is well below the link's pinned-transfer rate.

.. _kernel-replay-limitations:

Limitations
===========

* **Beta.** The domain, payload, and snapshot policy may change.
* **Single** ``KERNEL_REPLAY`` **subscriber.**
* **Coarse-grained device VRAM only**, plus module-scope device/constant variables.
* **HIP graph launches are not replayed.** The interceptor warns once and the graph runs once on
  the ordinary path. A graph dispatch does **not** hard-error at the replay gate.
* **Only single-packet, single-dispatch submissions** are replayed. Multi-packet batches warn once
  and run once.
* **Writer lock serializes the agent** for the whole replay window. Different GPUs use different
  locks (multi-GPU concurrent at the SDK), but there is no in-tree multi-GPU test and no MPI
  coordination.
* **Async copies are not fenced.** ``hsa_amd_memory_async_copy`` (or HIP async memcpy) on another
  thread can mutate device memory during the replay window.
* **Stuck drains decline the replay** rather than hanging or aborting (roughly 60 s bounds inside
  the window). See :ref:`kernel-replay-declined`.
* **A failed restore between passes aborts the process.** A partial restore cannot be undone, and
  continuing would hand silently corrupted memory to the application.
* **A tool callback must not launch GPU work on the replaying agent.** It mutates device memory
  inside the snapshot window, so the replayed dispatch's counters are not trustworthy. The SDK lets
  the dispatch through rather than deadlocking, and flags the dispatch with ``reentrancy=1``.
* **Host RAM duplication** of the tracked device footprint for the duration of the replay.

See also
========

* :ref:`kernel-replay-sdk-api` — payload, operations, dispatch counting pattern
* :ref:`kernel-replay-callback-api` — API contract and source map
* :ref:`kernel-replay-concurrency` — isolation model
* :ref:`kernel-replay-memory-snapshot` — what ``snap()`` / ``restore()`` actually do
