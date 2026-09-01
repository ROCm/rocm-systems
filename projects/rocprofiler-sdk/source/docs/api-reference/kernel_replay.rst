.. meta::
    :description: ROCprofiler-SDK kernel replay callback tracing API for custom tools
    :keywords: ROCprofiler-SDK API reference, kernel replay, callback tracing, experimental

.. _kernel-replay-sdk-api:

ROCprofiler-SDK kernel replay (experimental)
=============================================

Kernel replay re-executes a GPU dispatch several times inside one application run and restores
device memory between those executions so each pass observes identical inputs. In the SDK it is a
**callback tracing domain**, not a dedicated counting service.

.. warning::

   This API is experimental. The public header is
   ``<rocprofiler-sdk/experimental/kernel_replay.h>``. The domain and payload are expected
   to change before a stable release. A failed device-memory restore aborts the process.
   Replay is limited to single-packet, single-dispatch submissions; see
   :ref:`kernel-replay-limitations` and :ref:`kernel-replay-memory-snapshot`. Command-line
   ``rocprofv3`` usage is :ref:`using-kernel-replay-rocprofv3`.

   Known failure behavior:

   * A dispatch that cannot be replayed runs **once**, unreplayed, and warns. This covers a
     multi-packet submission, a HIP graph launch, and a snapshot that could not be completed
     because of host memory pressure or because HSA would not enumerate a loaded executable's
     module-scope variables.
   * A failed device-memory **restore** aborts the process. Once part of the snapshot has been
     written back, continuing would leave the application's memory in a mixed state.
   * A drain that does not complete within roughly 60 seconds aborts the process rather than
     hanging.

For the configure / ``replay_pass_count`` / local-context how-to, see :ref:`using-kernel-replay`. For
pass-count semantics, localized context control, and source maps, see
:ref:`kernel-replay-callback-api`.

This page is the tool-author counterpart of :ref:`rocprofiler_sdk_callback_tracing_services`: how to
subscribe, what the payload contains, and how replay interacts with dispatch counting.

Configure the domain
--------------------

Replay has no dedicated counting-service entry point. It is configured as a callback tracing domain,
like any other, and dispatch counting supplies the counter records.

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
A process that never configures the domain does not pay that tracking cost.

Operations and payload
----------------------

Cast ``record.payload`` to ``rocprofiler_callback_tracing_kernel_replay_data_t*``.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Operation
     - Phase
     - Tool responsibility
   * - ``ROCPROFILER_KERNEL_REPLAY_CONFIG``
     - ``PHASE_ENTER``
     - Set ``replay_pass_count``. Optionally set ``replay_continue``. May stash per-dispatch state
       in ``user_data``.
   * - ``ROCPROFILER_KERNEL_REPLAY_CONFIG``
     - ``PHASE_EXIT``
     - Replay of this dispatch has finished (or was declined).
   * - ``ROCPROFILER_KERNEL_REPLAY_PASS``
     - ``PHASE_ENTER``
     - Read ``current_pass`` / ``total_passes``. Optionally call
       ``replay_start_context`` / ``replay_stop_context``.
   * - ``ROCPROFILER_KERNEL_REPLAY_PASS``
     - ``PHASE_EXIT``
     - Pass complete; ``replay_continue`` (if set) runs after this.

``dispatch_info.dispatch_id`` is the same for CONFIG, every PASS, and every record those passes
produce. Distinguish passes with ``current_pass``.

Pass count
----------

After CONFIG ``PHASE_ENTER`` returns, the SDK calls ``replay_pass_count`` if it is non-null:

* ``NULL`` — dispatch is not replayed (no snapshot).
* returns ``1`` — ordinary single execution (no snapshot).
* returns ``N > 1`` — ``N`` passes; ``replay_continue`` may still stop early (custom tools only;
  ``rocprofv3`` never sets this callback).
* returns ``0`` — indefinite loop; ``replay_continue`` is required (custom tools only). The loop is
  unbounded and the SDK applies no pass cap, so ``replay_continue`` must eventually return zero on
  every path; otherwise the dispatch replays for the life of the process and its completion signal,
  which is deferred until after the loop, is never fired.

``rocprofv3`` returns the number of ``--pmc`` groups collectable on
``dispatch_info.agent_id``. A custom tool can return any of the cases above and may set
``replay_continue`` for early exit or an indefinite loop.

Using replay with dispatch counting
-----------------------------------

Replay does **not** replace dispatch counting. Typical pattern:

1. Configure kernel replay on one context.
2. Configure dispatch counting on another (or the same) context as usual.
3. During PASS ``PHASE_ENTER``, publish ``current_pass`` in thread-local storage (the pass callback
   and the dispatch-counting callback run on the submitting thread).
4. In the dispatch-counting callback, select the counter config for that pass.
5. Clear the thread-local pass index on PASS ``PHASE_EXIT``.

To run SPM or thread trace on only some passes, put those services on their own contexts and stop or
start them with the localized toggles during PASS ``PHASE_ENTER``. Which services honor a toggle
varies:

* Dispatch counter collection and SPM consult the override on every dispatch, so they can be placed
  on specific passes.
* Kernel dispatch tracing and dispatch thread trace observe a local *stop* only: they skip a
  dispatch whose context is forced off, but cannot be added to a context that is not already
  collecting.
* PC sampling is agent-wide and device counting is not dispatch-scoped, so neither consults the
  override. A toggle naming such a context reports success and has no effect.

Because PC sampling ignores the override, it cannot be isolated from dispatch counters by putting
them on separate passes, and the two must not be combined under replay: on MI2xx and MI3xx,
collecting them together hits the documented clock-gating conflict. ``rocprofv3`` does not expose
SPM or PC sampling together with kernel replay — that requires a custom tool. Do not call the global
``rocprofiler_start_context`` / ``rocprofiler_stop_context`` from inside the replay loop: that would
leak into non-replayed dispatches.

Doxygen
-------

The payload is in the ``CALLBACK_TRACING_SERVICE`` group:

* :ref:`callback_tracing_reference`
* Header: ``source/include/rocprofiler-sdk/experimental/kernel_replay.h``

There is no separate ``kernel_replay_service`` Doxygen group.

See also
--------

* :ref:`using-kernel-replay` — configure, ``replay_pass_count``, local context
* :ref:`using-kernel-replay-rocprofv3` — ``rocprofv3 --kernel-replay-beta-enabled``
* :ref:`kernel-replay-callback-api` — API contract
* :ref:`kernel-replay-concurrency` — isolation model
* :ref:`kernel-replay-memory-snapshot` — what ``snap()`` / ``restore()`` actually do
