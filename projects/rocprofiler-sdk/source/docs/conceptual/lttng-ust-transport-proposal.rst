.. _lttng-ust-transport-proposal:

LTTng-UST Transport Proposal
============================

This draft POC evaluates LTTng-UST as an out-of-process event delivery
transport for rocprofiler-sdk records. It is motivated by the medium-term
HIP/HSA emit-and-subscribe direction in the CPC tracing research branch:
runtime producers emit typed events without naming rocprofiler-sdk as the
consumer, and tools subscribe from another process.

Decision Criteria
-----------------

Transport recommendations in this proposal family are ranked by the lowest
producer-side time overhead first. Memory footprint is the second criterion.
This ordering is intentional: adding latency to HIP/HSA dispatch paths is the
most user-visible regression risk, while memory footprint matters most when two
approaches are close on time overhead.

Recommendation
--------------

For the common rocprofiler-sdk buffer path, the current recommendation remains
the per-thread or per-producer user-space ring transport because it has the
lowest measured producer-side time overhead. The BPF-style buffer improves
footprint compared with the current buffer, but it still has shared producer
contention. LTTng-UST is recommended only for the out-of-process generic
subscription track, not as the lowest-overhead in-process callback backend.

Why LTTng-UST Is Still Worth Testing
------------------------------------

LTTng-UST satisfies the constraints called out in the CPC tracing research:

* no ``LD_PRELOAD`` requirement after the producer is linked with LTTng-UST,
* no runtime text-segment modification when a consumer attaches,
* late attach through ``lttng-sessiond``,
* typed CTF metadata for event schemas,
* out-of-process consumers.

This PR therefore adds an experimental shadow-emission backend:

* the current in-process ``record_header_buffer`` is retained for compatibility,
* each successful record insertion also emits a typed LTTng-UST record envelope,
* the public buffer callback ABI is unchanged,
* the feature is selected only with
  ``ROCPROFILER_EXPERIMENTAL_LTTNG_BUFFER_TRANSPORT``.

The POC deliberately emits only a generic record envelope:

* ``category``
* ``kind``
* ``hash``
* ``payload_size``
* ``payload_alignment``
* ``sequence``

A production design should use typed tracepoints for high-volume domains such
as HIP API, HSA API, and kernel dispatch records instead of serializing opaque
rocprofiler-sdk payloads.

Expected Tradeoffs
------------------

When no LTTng session is active, the ideal tracepoint model is close to one
enabled flag load plus one unlikely branch per event. The measured POC still
shows producer-side overhead, especially with multiple producers, because every
record takes the normal rocprofiler-sdk buffer path and then calls the LTTng-UST
tracepoint. When a session is active, the producer writes into LTTng's user-space
CTF buffers and the out-of-process consumer drains completed sub-buffers
asynchronously. This is a good match for generic out-of-process subscription,
but it is not free: it adds the ``liblttng-ust`` dependency and does not directly
satisfy the existing in-process callback ABI without the local buffer kept by
this POC.

Compatibility Problem
---------------------

LTTng introduces a packaging and backward-compatibility risk that the SDK does
not control. During active-session benchmarking, the host installation had
``lttng-tools`` 2.13 but ``liblttng-ctl0`` 2.14. That mixed userspace stack made
the client and session daemon disagree on control command IDs: ``lttng create``
failed with ``Session name not found`` even though the session daemon was
spawned.

The benchmark worked only after avoiding the system installation and running a
complete matching LTTng 2.14 stack extracted under ``/tmp/lttng-2.14`` with
isolated ``LTTNG_HOME`` and ``XDG_RUNTIME_DIR`` directories. A production
rocprofiler-sdk LTTng transport would need to treat this as a real compatibility
problem: it should validate the LTTng client, session daemon, consumer daemon,
and libraries as a matched stack, document supported version ranges, and fail
with a clear diagnostic when the installed LTTng control-plane ABI is
inconsistent.

Design Position
---------------

Based on the required ranking:

#. lowest time overhead: per-producer user-space rings,
#. next best footprint/time compromise: BPF-style shared buffer,
#. best generic out-of-process subscription experiment: LTTng-UST.

LTTng-UST should be evaluated as a runtime-to-tool delivery path for
out-of-process consumers. It should not displace the per-producer ring as the
preferred rocprofiler-sdk internal buffer transport unless benchmarks show a
lower producer-side time cost.

Benchmark Snapshot
------------------

The draft benchmark compares the existing buffer, the BPF-style buffer, the
per-producer ring POC, and the LTTng-UST shadow-emission POC with 64-byte
payload records, 131072 records per round, and 15 rounds. Time values are
median nanoseconds per record. Footprint values are MiB. The active LTTng
benchmark required a matched LTTng userspace stack: the test host had
``lttng-tools`` 2.13 loading ``liblttng-ctl`` 2.14, which made the client and
session daemon disagree on control command IDs. The active rows below were run
with a complete LTTng 2.14 stack extracted under ``/tmp`` and an isolated
``LTTNG_HOME``/``XDG_RUNTIME_DIR``.

.. list-table::
   :header-rows: 1

   * - Producers
     - Transport
     - Emit ns/record
     - Drain ns/record
     - Total ns/record
     - Storage MiB
     - Max RSS MiB
   * - 1
     - Current
     - 24.518
     - 2.498
     - 27.016
     - 136.07
     - 141.81
   * - 1
     - BPF-style
     - 17.299
     - 13.874
     - 31.173
     - 12.01
     - 17.83
   * - 1
     - Ring private
     - 4.369
     - 1.594
     - 5.963
     - 12.00
     - 17.67
   * - 1
     - Ring shared mmap
     - 4.335
     - 1.574
     - 5.909
     - 12.00
     - 17.91
   * - 1
     - LTTng-UST inactive
     - 29.520
     - 2.433
     - 31.953
     - 136.07
     - 141.81
   * - 1
     - LTTng-UST active
     - 141.760
     - 2.719
     - 144.479
     - 136.07
     - 157.90
   * - 4
     - Current
     - 243.692
     - 2.558
     - 246.250
     - 136.07
     - 141.96
   * - 4
     - BPF-style
     - 167.720
     - 13.020
     - 180.740
     - 12.01
     - 17.97
   * - 4
     - Ring private
     - 5.944
     - 4.993
     - 10.936
     - 12.02
     - 17.68
   * - 4
     - Ring shared mmap
     - 5.970
     - 3.345
     - 9.315
     - 12.02
     - 17.79
   * - 4
     - LTTng-UST inactive
     - 288.077
     - 2.435
     - 290.512
     - 136.07
     - 141.84
   * - 4
     - LTTng-UST active
     - 501.102
     - 2.192
     - 503.294
     - 136.07
     - 180.27
   * - 16
     - Current
     - 260.940
     - 2.484
     - 263.424
     - 136.07
     - 142.28
   * - 16
     - BPF-style
     - 150.954
     - 16.893
     - 167.847
     - 12.01
     - 18.00
   * - 16
     - Ring private
     - 4.993
     - 12.037
     - 17.029
     - 12.06
     - 18.09
   * - 16
     - Ring shared mmap
     - 5.519
     - 5.655
     - 11.174
     - 12.06
     - 18.04
   * - 16
     - LTTng-UST inactive
     - 294.816
     - 2.665
     - 297.481
     - 136.07
     - 142.29
   * - 16
     - LTTng-UST active
     - 563.533
     - 2.330
     - 565.863
     - 136.07
     - 203.93

With the requested ranking, the benchmark keeps the per-producer user-space
ring as the preferred common transport. It has the lowest producer-side time in
both private-memory and shared-mmap forms and it keeps footprint near the
BPF-style buffer. BPF-style storage remains interesting only when minimizing
footprint is more important than producer time. LTTng-UST remains a separate
out-of-process subscription experiment: the active tracepoint path works and
produced 258.93 MiB of CTF trace data for the three-row active benchmark run,
but its measured producer-side time is higher than the ring and BPF-style
proposals.
