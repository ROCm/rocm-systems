.. meta::
  :description: Proposal for using a BPF-style buffer transport inside ROCprofiler-SDK
  :keywords: ROCprofiler-SDK, buffer, BPF ring buffer, tracing transport

.. _bpf-buffer-transport-proposal:

Experimental BPF-style buffer transport proposal
================================================

This proposal keeps the public ROCprofiler-SDK buffer API unchanged and limits the experiment to
the internal transport used by ``rocprofiler_create_buffer`` and the buffered tracing services. A
tool still receives ``rocprofiler_record_header_t**`` batches through
``rocprofiler_buffer_tracing_cb_t``. The only change is the backing storage used between SDK
producers and the callback thread.

The draft implementation adds ``ROCPROFILER_EXPERIMENTAL_BPF_BUFFER``. When enabled, each
``rocprofiler::buffer::instance`` uses a BPF-style user-space record buffer:

* each producer reserves one variable-size slot with an atomic write cursor,
* the public ``rocprofiler_record_header_t`` and payload are stored in the same mapped region,
* flush builds the callback header pointer array by walking the committed slots, and
* no public ABI, callback signature, or service configuration changes.

This intentionally borrows the reserve/commit data layout from BPF ring buffers without creating a
kernel ``BPF_MAP_TYPE_RINGBUF``. ROCprofiler-SDK buffered tracing producers and tool callbacks are
in-process, so using a kernel BPF map would add file descriptors, syscalls, kernel feature checks,
and dependency or permission questions without removing a process boundary.

Decision criteria
-----------------

Transport recommendations in this proposal family are ranked by lowest
producer-side time overhead first. Memory footprint is the second criterion and
is used to choose between designs that are close on producer time. This ordering
keeps the dispatch/API hot path as the primary constraint while still accounting
for long-running tools with many active buffers.

Current buffer behavior
-----------------------

The current path uses ``record_header_buffer`` as the per-slot storage type. It maps a page-rounded
payload region, keeps record headers in a separate vector, and serializes small parts of the producer
path with locks before a flush task invokes the tool callback. Lossless mode uses two internal
buffers so one can be flushed while producers continue into the other.

The existing public API shape is good for tools. The main costs to target are internal:

* header metadata lives outside the payload allocation,
* each record write performs short critical sections around buffer allocation and header updates,
* the callback thread has to process a separate side vector of headers, and
* the payload buffer and header vector sizes scale separately.

Option A: kernel BPF ring buffer
--------------------------------

This is the most literal BPF buffer interpretation, using ``BPF_MAP_TYPE_RINGBUF`` and libbpf-style
polling.

It is not the recommended default for ROCprofiler-SDK buffered services because SDK producers are
user-space code. The BPF ring buffer is optimized for kernel/BPF producers delivering events to
user-space consumers. For this use case it introduces a kernel object, map lifetime management,
polling file descriptors, kernel-version gating, and likely extra syscall overhead. It is only worth
revisiting if ROCprofiler-SDK needs to merge events that are already produced by kernel BPF programs
or if a future out-of-process collector requires fd-based wakeup and isolation.

Option B: BPF-style user-space ring buffer
------------------------------------------

This is the option prototyped by the draft compile flag. It keeps the BPF reserve/commit shape but
implements it directly in ROCprofiler-SDK's address space.

Expected advantages:

* one mapped allocation carries both public record headers and payloads,
* the producer path is an atomic slot reservation plus placement construction,
* flushing can walk contiguous committed slots in allocation order, and
* tool-facing ABI remains unchanged.

Risks and follow-up work:

* the prototype still takes a shared lock to keep flush and producers disjoint; a production version
  should move to explicit committed flags and drain only completed slots,
* metadata overhead means effective payload capacity differs from the user-requested byte size, and
* benchmarks must compare hot-path API tracing, kernel dispatch tracing, and lossless backpressure
  against the existing double-buffer implementation.

Option C: per-thread SPSC shards plus callback aggregation
----------------------------------------------------------

For the fastest producer path, give each producer thread a small single-producer/single-consumer
record shard and let the callback thread aggregate shards during flush. This minimizes atomics and
shared cache-line traffic in API tracing paths.

Expected advantages:

* best hot-path latency for high-frequency runtime API records,
* no multi-producer reservation contention,
* natural cache locality for thread-local correlation and record construction.

Tradeoffs:

* higher memory footprint when many application threads touch traced APIs,
* more complex flush ordering across threads, and
* more lifecycle work for thread creation, teardown, and late attachment.

This is likely the best long-term high-throughput design if traces are allowed to preserve per-thread
ordering plus timestamps instead of strict global insertion order.

Option D: tune the current double buffer
----------------------------------------

This is the least disruptive implementation path. Keep ``record_header_buffer`` and reduce the
producer critical sections, pre-size header storage more tightly, and make watermark decisions use
bytes available after alignment. It has the smallest review risk but leaves the split metadata/payload
layout intact.

Recommendation
--------------

Using the required ordering, Option C is the strongest long-term direction
because per-thread or per-producer shards remove shared producer contention from
the hot path. Option B is still useful as a narrower BPF-style draft because it
shows how much footprint can be saved by colocating record headers and payloads
without changing the public callback ABI. Option D is the least invasive and
lowest review-risk path, but it is not the recommended performance direction
unless implementation risk outweighs producer overhead.

Use this draft to answer one question first: does the BPF-style in-process slot
layout reduce producer overhead enough to justify replacing the current
header-vector buffer when compared with the faster per-producer ring proposal?
If benchmark data shows that BPF-style storage wins mainly on footprint but not
on producer time, keep it as a fallback or intermediate transport rather than
the primary rocprofiler-sdk buffer direction.

Benchmark note
--------------

The cross-proposal microbenchmark uses one unit for time, nanoseconds per
record, and one unit for footprint, MiB. With 64-byte payload records and 131072
records per round, the BPF-style buffer reduces storage from 136.07 MiB to
12.01 MiB, but it does not beat the per-producer ring on producer time:

.. list-table::
   :header-rows: 1

   * - Producers
     - Transport
     - Emit ns/record
     - Total ns/record
     - Storage MiB
     - Max RSS MiB
   * - 1
     - Current
     - 24.518
     - 27.016
     - 136.07
     - 141.81
   * - 1
     - BPF-style
     - 17.299
     - 31.173
     - 12.01
     - 17.83
   * - 1
     - Ring shared mmap
     - 4.335
     - 5.909
     - 12.00
     - 17.91
   * - 1
     - LTTng-UST inactive
     - 29.520
     - 31.953
     - 136.07
     - 141.81
   * - 4
     - Current
     - 243.692
     - 246.250
     - 136.07
     - 141.96
   * - 4
     - BPF-style
     - 167.720
     - 180.740
     - 12.01
     - 17.97
   * - 4
     - Ring shared mmap
     - 5.970
     - 9.315
     - 12.02
     - 17.79
   * - 4
     - LTTng-UST inactive
     - 288.077
     - 290.512
     - 136.07
     - 141.84
   * - 16
     - Current
     - 260.940
     - 263.424
     - 136.07
     - 142.28
   * - 16
     - BPF-style
     - 150.954
     - 167.847
     - 12.01
     - 18.00
   * - 16
     - Ring shared mmap
     - 5.519
     - 11.174
     - 12.06
     - 18.04
   * - 16
     - LTTng-UST inactive
     - 294.816
     - 297.481
     - 136.07
     - 142.29

The resulting recommendation is ring first for lowest overhead, then BPF-style
when low footprint is the stronger secondary constraint.
