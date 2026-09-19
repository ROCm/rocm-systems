.. meta::
    :description: ROCprofiler-SDK range replay callback tracing API for custom tools
    :keywords: ROCprofiler-SDK API reference, range replay, kernel replay, callback tracing, experimental

.. _range-replay-sdk-api:

ROCprofiler-SDK range replay (experimental)
============================================

Range replay re-executes a **recorded sequence of dispatches** inside one application run, restoring
device memory to the sequence's entry state before each re-execution so every pass observes identical
inputs. Where :ref:`kernel-replay-sdk-api` repeats one dispatch, range replay repeats a phase — an
iteration, a layer, a collective step — whose kernels feed each other and therefore cannot be
measured one at a time.

.. warning::

   This API is experimental. The public header is
   ``<rocprofiler-sdk/experimental/range_replay.h>``. The domain, the payload, and the two entry
   points are expected to change before a stable release.

For the model, the carve-outs, and the source map, see :ref:`range-replay-conceptual`. For the full
API contract see :ref:`range-replay-callback-api`; for what makes a range replayable and why one gets
declined see :ref:`range-replay-soundness`.

Configure the domain
--------------------

.. code-block:: cpp

    rocprofiler_context_id_t ctx{};
    rocprofiler_create_context(&ctx);

    rocprofiler_configure_callback_tracing_service(
        ctx,
        ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
        nullptr,  // all operations: CONFIG, PASS, and CLOSE
        0,
        tool_range_replay_callback,
        nullptr);

    rocprofiler_start_context(ctx);

The domain is single-subscriber: a second context configuring it gets
``ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED``, because a range runs one plan. Configuring
it also enables the device-allocation tracker used for snapshot and restore, and enables HSA queue
interception. A process that never configures the domain pays neither cost.

Mark a range
------------

.. code-block:: cpp

    rocprofiler_range_replay_begin(iteration_index);  // CONFIG callback fires here

    // ... the application's dispatches for this phase: recorded, and executed as pass 0 ...

    rocprofiler_range_replay_end();                   // PASS callbacks, then CLOSE

Ranges are thread-scoped and do not nest; only the marking thread's dispatches are recorded.
``rocprofiler_range_replay_end()`` blocks until the extra passes have run.

Operations and payload
----------------------

Cast ``record.payload`` to ``rocprofiler_callback_tracing_range_replay_data_t*``.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Operation
     - Phase
     - Tool responsibility
   * - ``ROCPROFILER_RANGE_REPLAY_CONFIG``
     - ``PHASE_ENTER``
     - Set ``pass_count_cb``. Optionally set ``replay_continue_cb``. May stash per-range state in
       ``user_data``, which every later callback for this range receives.
   * - ``ROCPROFILER_RANGE_REPLAY_CONFIG``
     - ``PHASE_EXIT``
     - Configuration exchange complete. Both phases run inside
       ``rocprofiler_range_replay_begin()``; the range has not executed yet.
   * - ``ROCPROFILER_RANGE_REPLAY_PASS``
     - ``PHASE_ENTER``
     - Read ``current_pass`` / ``total_passes``. Optionally call
       ``replay_local_start_context_cb`` / ``replay_local_stop_context_cb``.
   * - ``ROCPROFILER_RANGE_REPLAY_PASS``
     - ``PHASE_EXIT``
     - Pass complete; ``replay_continue_cb`` (if set) runs after this.
   * - ``ROCPROFILER_RANGE_REPLAY_CLOSE``
     - both
     - Read ``status``, ``dispatch_count``, and ``divergence_count``. Delivered exactly once per
       opened range, including for a range that was declined before any pass ran.

``current_pass`` is 1-based: pass 0 is the application's own execution of the range, which the SDK
observes rather than drives, so it raises no ``PASS`` callback. A tool that wants a particular counter
group collected on the live run must have it active before ``rocprofiler_range_replay_begin()``.

Pass count
----------

Same semantics as kernel replay, with pass 0 reinterpreted as the application's own execution:

* ``NULL`` — the range is observed but never re-executed.
* returns ``1`` — same as ``NULL``.
* returns ``N > 1`` — the recording is re-executed ``N - 1`` times; ``replay_continue_cb`` may still
  stop early.
* returns ``0`` — indefinite; ``replay_continue_cb`` is required.

An opt-out still produces a ``CLOSE`` callback, with status
``ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT``, so a tool can tell its own decision apart from an
SDK decline.

Handling declines
-----------------

``rocprofiler_range_replay_end()`` returns ``ROCPROFILER_STATUS_SUCCESS`` whether or not the range was
replayed. The outcome is the ``CLOSE`` callback's ``status``: ``REPLAYED`` when at least one extra
pass ran, otherwise the reason. A declined range is not an error — the application's own execution is
never altered — but a tool that aggregates per-pass measurements must drop a declined range's data
rather than treat its single pass as a complete set.

The statuses a tool is most likely to see in practice are ``MULTI_QUEUE`` and ``MULTI_AGENT`` (the
phase spans more than one HIP stream or device), ``CONCURRENT_DISPATCH`` (another thread was
submitting to the same device), and ``MEMORY_COPY_IN_RANGE`` (the phase contains an ``hipMemcpy`` to
device memory). See :ref:`range-replay-soundness` for the complete table.

Verifying a range once
----------------------

Setting ``ROCPROF_RANGE_REPLAY_VERIFY`` in the environment makes the SDK hash the snapshot regions
after the application's execution and again after the final pass, reporting how many differ as
``divergence_count`` on ``CLOSE``. A non-zero value means the range reads state the snapshot does not
cover, so its passes describe different inputs even though every eligibility check passed. Use it to
validate a range during tool development and leave it off afterwards.

Doxygen
-------

The payload is in the ``CALLBACK_TRACING_SERVICE`` group:

* :ref:`callback_tracing_reference`
* Header: ``source/include/rocprofiler-sdk/experimental/range_replay.h``

See also
--------

* :ref:`range-replay-conceptual` — the model and its carve-outs
* :ref:`range-replay-callback-api` — API contract
* :ref:`range-replay-soundness` — what makes a range replayable
* :ref:`kernel-replay-sdk-api` — the single-dispatch counterpart
