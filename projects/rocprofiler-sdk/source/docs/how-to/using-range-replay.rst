.. meta::
  :description: Using the ROCprofiler-SDK range replay callback domain from a custom tool
  :keywords: rocprofiler-sdk, range replay, callback tracing, RANGE_REPLAY, host replay, custom tool

.. _using-range-replay:

==================
Using range replay
==================

Range replay is an experimental callback tracing domain. A tool subscribes to
``ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY``, the application marks a phase, and the SDK repeats
that phase with device memory restored so each pass sees the same captured inputs.

The domain is not tied to hardware counters. A tool can collect counters, timing, thread trace, or
anything else it can turn on for a pass. ``rocprofv3`` uses the domain for counter groups; that
command line is :ref:`using-range-replay-rocprofv3`.

The mechanism (recording, the decline table, the jump) is :ref:`range-replay-conceptual`. The
payload fields are :ref:`range-replay-sdk-api`.

.. warning::

   Range replay is **experimental**. The public header is
   ``<rocprofiler-sdk/experimental/range_replay.h>``. The domain, the payload, and the entry
   points can change.

Configure the domain
====================

There is no dedicated configure function. One context may subscribe. A second configure call
returns ``ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED``, because a phase has one pass count.

.. code-block:: cpp

   rocprofiler_context_id_t ctx{};
   rocprofiler_create_context(&ctx);

   rocprofiler_configure_callback_tracing_service(
       ctx,
       ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
       nullptr,  // CONFIG, PASS, and CLOSE
       0,
       tool_range_replay_callback,
       nullptr);

   rocprofiler_start_context(ctx);

Configuring the domain turns on the allocation tracker the snapshot uses, and turns on HSA queue
interception. A process that never configures the domain pays neither cost.

Mark a phase, and choose packet or host replay
==============================================

Packet replay records the marking thread's dispatches and submits that recording again. The
application calls ``rocprofiler_range_replay_begin`` and ``rocprofiler_range_replay_end``. ``end``
blocks until the extra passes finish.

Host replay runs the application's code again. The tool sets ``host_replay`` during ``CONFIG``,
and the application opens the phase with ``ROCPROFILER_RANGE_REPLAY_HOST_BEGIN``. The macro is
what keeps the ``sigsetjmp`` in the caller's frame. The contract for the code inside the phase is
:ref:`range-replay-host-contract`.

.. code-block:: cpp

   void
   tool_range_replay_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t* /* user_data */,
                              void* /* callback_args */)
   {
       if(record.kind != ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY) return;

       auto* data =
           static_cast<rocprofiler_callback_tracing_range_replay_data_t*>(record.payload);

       if(record.operation == ROCPROFILER_RANGE_REPLAY_CONFIG &&
          record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
       {
           data->pass_count_cb = tool_pass_count;     // required for any extra pass
           data->host_replay   = tool_wants_host_replay() ? 1 : 0;
           // optional: data->replay_continue_cb = tool_continue;
       }
       else if(record.operation == ROCPROFILER_RANGE_REPLAY_PASS &&
               record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
       {
           // data->current_pass is 1-based. Pass 0 is the application's run and raises no PASS.
       }
       else if(record.operation == ROCPROFILER_RANGE_REPLAY_CLOSE &&
               record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
       {
           // data->status, data->dispatch_count, data->divergence_count
       }
   }

   uint64_t
   tool_pass_count(uint64_t /* range_id */, rocprofiler_user_data_t /* user_data */)
   {
       return 4;   // pass 0 is the application; passes 1..3 are re-executed
   }

Pass 0 is already running by the time a packet replay could configure it, so a service that must
see pass 0 is enabled before ``begin``. ``PASS`` is where a tool changes configuration for the
re-executed passes.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - ``pass_count_cb``
     - What the SDK does
   * - left ``NULL``, or returns ``1``
     - The phase runs once. ``CLOSE`` reports ``NO_PASS_COUNT``.
   * - returns ``N > 1``
     - The phase is re-executed ``N - 1`` times. ``replay_continue_cb`` may stop early.
   * - returns ``0``, with ``replay_continue_cb`` set
     - Re-executes until the continue callback returns 0.
   * - returns ``0``, continue callback unset
     - The phase runs once.

``user_data`` written during ``CONFIG`` is passed to every later callback for that phase.

``rocprofiler_range_replay_end`` returns success whether or not the extra passes ran. The outcome
is ``status`` on ``CLOSE``. Drop a declined phase's per-pass aggregates. Pass 0 is the
application's real execution either way. The statuses and the reasons are in
:ref:`using-range-replay-rocprofv3` and :ref:`range-replay-soundness`.

In-tree examples
================

``samples/range_replay/`` is a HIP application plus a preloaded tool, built with
``-DROCPROFILER_BUILD_SAMPLES=ON``.

* ``basic_client.cpp`` — four passes, ``CLOSE`` status ``REPLAYED``, and a zero divergence count
  when ``ROCPROF_RANGE_REPLAY_VERIFY=1``.
* The same client with host replay — the extra passes re-enter ``main`` through ``siglongjmp``.
* ``opt_out_client.cpp`` — ``pass_count_cb`` left unset, status ``NO_PASS_COUNT``.
* ``decline_client.cpp`` — two streams, status ``MULTI_QUEUE``, application result unchanged.

Advanced
========

Per-pass services
-----------------

During ``PASS`` ``PHASE_ENTER`` the payload carries ``replay_local_start_context_cb`` and
``replay_local_stop_context_cb``. They enable or disable a context for the rest of the phase.
The same rules as kernel replay apply: legal only inside that callback, sticky across the later
passes, and unable to promote a context that is globally inactive. See
:ref:`kernel-replay-callback-api`.

PC sampling is agent-wide and does not honor these callbacks today.

On host replay the override stays installed for the whole pass, because the dispatches that must
honor it are the application's. On packet replay it covers the re-submitted recording.

Checking the passes
-------------------

Set ``ROCPROF_RANGE_REPLAY_VERIFY=1`` while bringing a tool up. ``divergence_count`` on ``CLOSE``
is the number of snapshot regions that differed between the end of pass 0 and the end of the last
pass. Leave the variable unset after that. The comparison is byte-exact, so a kernel with
floating-point atomics can report a difference with identical inputs. See
:ref:`range-replay-soundness`.

What the application observes
-----------------------------

Packet replay's extra passes finish before ``end`` returns. GPU timestamps the application took
inside the phase do not include them. A host timer that wraps ``end`` does.

Host replay's extra passes are more trips through the application's code. Anything the phase
prints, writes, or allocates happens once per pass, subject to the contract.

Dispatch IDs are new on every pass, including the re-executed ones. Kernel replay reuses one
dispatch id; range replay does not. Carry the pass index yourself from the ``PASS`` callback if
the output has to name the group.

See also
========

* :ref:`using-range-replay-rocprofv3` — command line, diagrams, and the host-replay contract
* :ref:`range-replay-sdk-api` — operations and payload
* :ref:`range-replay-callback-api` — the callback contract and the source map
* :ref:`range-replay-host` — the jump, registered host memory, and the tests
* :ref:`range-replay-soundness` — why a phase is declined
* :ref:`using-kernel-replay` — the single-dispatch domain
* :ref:`kernel-replay-memory-snapshot` — which bytes are copied
