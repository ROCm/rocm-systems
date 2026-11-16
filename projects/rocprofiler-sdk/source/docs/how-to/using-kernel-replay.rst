.. meta::
  :description: Using the ROCprofiler-SDK kernel replay callback tracing domain from a custom tool
  :keywords: rocprofiler-sdk, kernel replay, callback tracing, KERNEL_REPLAY, replay_pass_count, local context

.. _using-kernel-replay:

===================
Using kernel replay
===================

Kernel replay is an experimental ROCprofiler-SDK callback tracing domain. A custom tool
subscribes to ``ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY`` and specifies how many times to
re-execute a dispatch. The SDK then snapshots tracked device memory, runs that many passes, and
restores the snapshot between passes so every pass observes identical inputs.

The domain doesn't depend on hardware counters. A tool can use it for counters, timing, PC sampling,
thread trace, or any other per-pass work.

This topic explains how to use kernel replay from the SDK: it covers subscribing to the replay domain from a custom tool.
For conceptual and API details, see :ref:`kernel-replay-conceptual` and
:ref:`kernel-replay-sdk-api`. For the ``rocprofv3`` command-line options for kernel replay
(``--replay-mode kernel --kernel-replay-beta-enabled``), see the rocprofv3 how-to guide (:ref:`using-kernel-replay-rocprofv3`).

.. warning::

   Kernel replay is **experimental**. The public header is
   ``<rocprofiler-sdk/experimental/kernel_replay.h>``. The
   :ref:`domain <kernel-replay-configure-domain>`,
   :ref:`payload <kernel-replay-set-pass-count>`, and
   :ref:`limitations <kernel-replay-limitations>` are expected to change
   before a stable release.

Procedure
=========

Using kernel replay takes two steps: subscribe to the domain, then set how many passes to run.

.. _kernel-replay-configure-domain:

Configure the domain
--------------------

There's no dedicated ``rocprofiler_configure_kernel_replay_*`` entry point. Subscribe through
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
A process that never configures the domain doesn't pay that tracking cost. Only one context might
subscribe; a second configure call returns ``ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED``.

.. _kernel-replay-set-pass-count:

Set replay_pass_count
---------------------

During ``CONFIG`` ``PHASE_ENTER``, the tool installs the pass-count callback. The SDK calls
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

The combination of ``replay_pass_count`` and ``replay_continue`` return values determines the
resulting behavior:

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
     - Rejected: the SDK warns and doesn't replay the dispatch.

The SDK reserves one dispatch ID before ``CONFIG`` and reuses it for every pass. The application observes
exactly one kernel completion regardless of pass count.

Additional techniques
======================

These build on the preceding procedure; neither is required to use kernel replay.

Carry state with ``user_data``
------------------------------

The tracing callback receives a ``rocprofiler_user_data_t*`` for the current dispatch. Set it
during ``CONFIG PHASE_ENTER`` to carry state through the complete replay sequence:

* Use ``user_data->value`` for a small scalar such as a pass count or policy identifier.
* Use ``user_data->ptr`` for a state object that remains alive through ``CONFIG PHASE_EXIT``.

The SDK copies the union value after ``CONFIG PHASE_ENTER`` and supplies it to
``replay_pass_count``, ``replay_continue``, and every ``PASS`` callback.

Each pass gets its own copy, so replacing the union value during ``PASS PHASE_ENTER`` only affects
that one pass. The following table shows which value each callback or phase observes:

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
tool can't signal an early exit by overwriting ``user_data->value`` during a pass. Put anything
influencing the continue decision behind ``user_data->ptr`` instead: mutations to the
pointed-to object are visible from every callback, because the pointer itself never changes.

``samples/kernel_replay/basic_client_with_user_data.cpp`` demonstrates the pointer form. It stores
tool state in ``user_data.ptr``, reads the requested maximum from ``replay_pass_count``, lets a
``PASS PHASE_EXIT`` update stop the loop through ``replay_continue``, and asserts each row of the
table above.

Localized context control
-------------------------

During ``PASS`` ``PHASE_ENTER`` the payload carries ``replay_start_context`` and
``replay_stop_context``. Use ``replay_start_context`` to enable, and ``replay_stop_context`` to
disable, override-aware contexts for that pass — for example, enabling counters only on selected
passes, or thread trace only once — without calling the global ``rocprofiler_start_context`` /
``rocprofiler_stop_context``. PC sampling is agent-wide and
doesn't currently honor these callbacks, so they can't isolate it to one replay pass.

Overrides are sticky across passes and scoped to the replay loop. See
:ref:`kernel-replay-callback-api` for the contract.

In-tree examples
================

The ROCprofiler-SDK source tree includes these kernel replay examples:

* ``samples/kernel_replay/basic_client_with_user_data.cpp`` — threads mutable tool state through
  ``user_data.ptr`` without a dispatch-ID side table.
* ``tests/kernel-replay-concurrency/`` — custom client that replays one kernel and opts another
  out, asserting snapshot/restore doesn't corrupt concurrent non-replayed work.
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

Kernel replay **doesn't** restore:

* Unified or managed memory.
* ``hipMallocAsync`` and other pool-backed, stream-ordered allocations.
* Host, fine-grained, kernarg, or executable allocations (excluded by design).
* HIP graph-managed memory.

Capture takes a **full in-memory copy** of every tracked region into host RAM, so its cost scales
as ``O(tracked_bytes × passes)`` — not just kernel execution time. The one exception is the last
pass: kernel replay **doesn't** restore memory after it, so the application sees the memory the
kernel actually produced.

.. _kernel-replay-limitations:

Limitations
===========

* **Beta.** The domain, payload, and snapshot policy may change.
* **Single** ``KERNEL_REPLAY`` **subscriber.**
* **Coarse-grained device VRAM only**, plus module-scope device/constant variables.
* **Kernel replay doesn't replay HIP graph launches.** The interceptor warns once and the graph
  runs once on the ordinary path. A graph dispatch **doesn't** hard-error at the replay gate.
* **Kernel replay replays only single-packet, single-dispatch submissions.** Multi-packet batches
  warn once and run once.
* **Writer lock serializes the agent** for the whole replay window. Different GPUs use different
  locks (multi-GPU concurrent at the SDK), but there's no in-tree multi-GPU test and no MPI
  coordination.
* **Kernel replay doesn't fence async copies.** ``hsa_amd_memory_async_copy`` (or HIP async memcpy) on another
  thread can mutate device memory during the replay window.
* **Stuck drains abort the process** rather than hanging (bounded to roughly 60 s within the
  window).
* **Host RAM duplication** of the tracked device footprint for the duration of the replay. Under
  host memory pressure, kernel replay declines the snapshot and runs the dispatch once rather
  than aborting.

See also
========

* :ref:`kernel-replay-sdk-api` — payload, operations, dispatch counting pattern
* :ref:`kernel-replay-callback-api` — API contract and source map
* :ref:`kernel-replay-concurrency` — isolation model
* :ref:`kernel-replay-memory-snapshot` — what ``snap()`` / ``restore()`` actually do
