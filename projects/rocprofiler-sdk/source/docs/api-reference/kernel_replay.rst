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
   to change before a stable release. Command-line ``rocprofv3`` wiring is the stacked tool
   integration PR. A failed device-memory restore aborts the process. Replay is limited to
   single-packet, single-dispatch submissions; see :ref:`kernel-replay-limitations` and
   :ref:`kernel-replay-memory-snapshot`.

For the configure / ``pass_count_cb`` / local-context how-to, see :ref:`using-kernel-replay`. For
pass-count semantics, localized context control, and source maps, see
:ref:`kernel-replay-callback-api`.

This page is the tool-author counterpart of :ref:`rocprofiler_sdk_callback_tracing_services`: how to
subscribe, what the payload contains, and how replay interacts with dispatch counting.

Configure the domain
--------------------

There is no ``rocprofiler_configure_kernel_replay_counting_service()``. That entry point belonged to
an earlier prototype and is not in this API.

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
     - Set ``pass_count_cb``. Optionally set ``replay_continue_cb``. May stash per-dispatch state
       in ``user_data``.
   * - ``ROCPROFILER_KERNEL_REPLAY_CONFIG``
     - ``PHASE_EXIT``
     - Replay of this dispatch has finished (or was declined).
   * - ``ROCPROFILER_KERNEL_REPLAY_PASS``
     - ``PHASE_ENTER``
     - Read ``current_pass`` / ``total_passes``. Optionally call
       ``replay_local_enable_context_cb`` / ``replay_local_disable_context_cb``.
   * - ``ROCPROFILER_KERNEL_REPLAY_PASS``
     - ``PHASE_EXIT``
     - Pass complete; ``replay_continue_cb`` (if set) runs after this.

``dispatch_info.dispatch_id`` is the same for CONFIG, every PASS, and every record those passes
produce. Distinguish passes with ``current_pass``.

Pass count
----------

After CONFIG ``PHASE_ENTER`` returns, the SDK calls ``pass_count_cb`` if it is non-null:

* ``NULL`` — dispatch is not replayed (no snapshot).
* returns ``1`` — ordinary single execution (no snapshot).
* returns ``N > 1`` — ``N`` passes; ``replay_continue_cb`` may still stop early (custom tools only;
  ``rocprofv3`` never sets this callback).
* returns ``0`` — indefinite loop; ``replay_continue_cb`` is required (custom tools only).

``rocprofv3`` (in the stacked tool PR) returns the number of ``--pmc`` groups collectable on
``dispatch_info.agent_id``. A custom tool can return any of the cases above and may set
``replay_continue_cb`` for early exit or an indefinite loop.

Using replay with dispatch counting
-----------------------------------

Replay does **not** replace dispatch counting. Typical pattern:

1. Configure kernel replay on one context.
2. Configure dispatch counting on another (or the same) context as usual.
3. During PASS ``PHASE_ENTER``, publish ``current_pass`` in thread-local storage (the pass callback
   and the dispatch-counting callback run on the submitting thread).
4. In the dispatch-counting callback, select the counter config for that pass.
5. Clear the thread-local pass index on PASS ``PHASE_EXIT``.

To run PC sampling, SPM, or thread trace on only some passes, put those services on their own
contexts and stop or start them with the localized toggles during PASS ``PHASE_ENTER``. Counter
collection and PC sampling cannot safely run on the same replay pass; use separate passes.
``rocprofv3`` does not expose SPM or PC sampling together with kernel replay — that requires a
custom tool. Do not call the global
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

* :ref:`using-kernel-replay` — configure, ``pass_count_cb``, local context
* :ref:`kernel-replay-callback-api` — API contract
* :ref:`kernel-replay-concurrency` — isolation model
* :ref:`kernel-replay-memory-snapshot` — what ``snap()`` / ``restore()`` actually do
* :ref:`range-replay-sdk-api` — the same idea applied to a sequence of dispatches
