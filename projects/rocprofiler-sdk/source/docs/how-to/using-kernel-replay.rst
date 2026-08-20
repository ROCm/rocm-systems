.. meta::
  :description: Using the ROCprofiler-SDK kernel replay callback tracing domain from a custom tool
  :keywords: rocprofiler-sdk, kernel replay, callback tracing, KERNEL_REPLAY, replay_pass_count, local context

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

This page is the SDK how-to: it covers subscribing to the replay domain from a custom tool.
Conceptual and API detail live under :ref:`kernel-replay-conceptual` and
:ref:`kernel-replay-sdk-api`. For the ``rocprofv3`` command-line option
(``--kernel-replay-beta-enabled``), see the rocprofv3 how-to guide (:ref:`using-kernel-replay-rocprofv3`).

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

Set ``replay_pass_count``
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
           payload->replay_pass_count = tool_pass_count_callback;
           // optionally: payload->replay_continue = tool_continue_callback;
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
       // Return N > 1 for a fixed loop. Return 0 only with replay_continue set.
       return groups_for_agent(dispatch_info.agent_id);
   }

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - ``replay_pass_count``
     - ``replay_continue``
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

Carry state with ``user_data``
==============================

The tracing callback receives a ``rocprofiler_user_data_t*`` for the current dispatch. Set it
during ``CONFIG PHASE_ENTER`` to carry state through the complete replay sequence:

* Use ``user_data->value`` for a small scalar such as a pass count or policy identifier.
* Use ``user_data->ptr`` for a state object that remains alive through ``CONFIG PHASE_EXIT``.

The SDK copies the union value after ``CONFIG PHASE_ENTER`` and supplies it to
``replay_pass_count``, ``replay_continue``, and every ``PASS`` callback.

Each pass gets its own copy, so replacing the union value during ``PASS PHASE_ENTER`` is scoped to
that one pass:

.. list-table::
   :header-rows: 1

   * - Observed from
     - Value delivered
   * - ``replay_pass_count``
     - the ``CONFIG PHASE_ENTER`` value
   * - ``PASS PHASE_ENTER``
     - the ``CONFIG PHASE_ENTER`` value, on every pass
   * - ``PASS PHASE_EXIT``
     - the write made during ``PASS PHASE_ENTER`` of the same pass
   * - ``replay_continue``
     - the ``CONFIG PHASE_ENTER`` value
   * - ``CONFIG PHASE_EXIT``
     - the ``CONFIG PHASE_ENTER`` value

``replay_continue`` runs after ``PASS PHASE_EXIT`` but still receives the ``CONFIG`` value, so a
tool cannot signal an early exit by overwriting ``user_data->value`` during a pass. Put anything
that has to influence the continue decision behind ``user_data->ptr`` instead: mutations to the
pointed-to object are visible from every callback, because the pointer itself never changes.

``samples/kernel_replay/basic_client_with_user_data.cpp`` demonstrates the pointer form. It stores
tool state in ``user_data.ptr``, reads the requested maximum from ``replay_pass_count``, lets a
``PASS PHASE_EXIT`` update stop the loop through ``replay_continue``, and asserts each row of the
table above.

Localized context control
=========================

During ``PASS`` ``PHASE_ENTER`` the payload carries ``replay_start_context`` /
``replay_stop_context``. Use them to enable or disable override-aware contexts for that
pass (for example counters on selected passes and thread trace once) without calling global
``rocprofiler_start_context`` / ``rocprofiler_stop_context``. PC sampling is agent-wide and does
not currently honor these callbacks, so it cannot be isolated to one replay pass.

Overrides are sticky across passes and scoped to the replay loop. See
:ref:`kernel-replay-callback-api` for the contract.

In-tree examples
================

* ``samples/kernel_replay/basic_client_with_user_data.cpp`` — threads mutable tool state through
  ``user_data.ptr`` without a dispatch-id side table.
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
* **Stuck drains abort the process** rather than hanging (roughly 60 s bounds inside the window).
* **Host RAM duplication** of the tracked device footprint for the duration of the replay. Under
  host memory pressure the snapshot is declined and the dispatch runs once rather than aborting.

See also
========

* :ref:`kernel-replay-sdk-api` — payload, operations, dispatch counting pattern
* :ref:`kernel-replay-callback-api` — API contract and source map
* :ref:`kernel-replay-concurrency` — isolation model
* :ref:`kernel-replay-memory-snapshot` — what ``snap()`` / ``restore()`` actually do
