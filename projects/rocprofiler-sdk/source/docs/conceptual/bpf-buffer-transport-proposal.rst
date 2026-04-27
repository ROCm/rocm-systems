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

The lowest-footprint proposal is Option D because it changes only the current implementation. The best
balance for a draft implementation is Option B: it demonstrates the BPF buffer idea while avoiding a
kernel BPF dependency and preserving the SDK ABI. The fastest design for sustained high-rate tracing
is Option C, but it should be treated as a second-stage design because its thread lifecycle and flush
ordering semantics are more invasive than a buffer backend swap.

Use this draft to answer one question first: does the BPF-style in-process slot layout reduce producer
overhead enough to justify replacing the current header-vector buffer? If benchmark data does not show
a meaningful win, the safer path is to tune the current double buffer and keep BPF maps out of the
in-process data path.
