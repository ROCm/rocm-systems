.. meta::
  :description: Proposal for a user-space ring transport for ROCprofiler-SDK buffered records
  :keywords: ROCprofiler-SDK, buffer, tracing, shared memory, mmap, user-space ring, proposal

.. _user-ring-transport-proposal:

====================================================
User-space ring transport proposal for buffered data
====================================================

This proposal describes a low-overhead transport for moving buffered
ROCprofiler-SDK records from internal producers to tools. The proposed direction is
to use per-thread or per-producer user-space rings, backed by process-private
memory for in-process tools and shared ``mmap`` memory for out-of-process tools.

The goal is to reduce producer-side overhead and contention without changing the
initial public buffered-services API. Tools would continue to create buffers,
configure buffered services, receive ``rocprofiler_record_header_t`` batches, and
observe discard or lossless backpressure through existing buffer policy semantics.

This draft PR includes a proof-of-concept implementation behind the
``ROCPROFILER_EXPERIMENTAL_USER_RING_BUFFER`` CMake option. The default
``record_header_buffer`` backend is unchanged when the option is disabled.

Motivation
==========

The current buffer implementation is intentionally simple: a buffer handle owns a
small set of internal record-header buffers, records are emplaced into the active
buffer, and watermark or explicit flushes move completed records to the callback
thread. That model keeps the tool API straightforward, but producer threads still
interact with shared buffer state when many runtime callbacks, dispatch records,
or sampling records target the same buffer.

A ring-per-producer design moves the hot path closer to single-producer,
single-consumer behavior:

* the producer reserves from its own shard;
* the producer writes the record payload directly into that shard;
* the producer publishes the record with a release-store update;
* a callback or collector thread drains shards and forms the existing callback
  batch of ``rocprofiler_record_header_t*``.

For in-process tools, each shard can be normal process memory. For out-of-process
tools, the same ring layout can live in shared memory mapped by both the
instrumented process and the collector process.

Design constraints
==================

The initial design should preserve these constraints:

* Keep ``rocprofiler_create_buffer``, ``rocprofiler_flush_buffer``,
  ``rocprofiler_create_callback_thread``, and ``rocprofiler_assign_callback_thread``
  stable.
* Keep the callback ABI based on ``rocprofiler_record_header_t**``,
  ``num_headers``, callback data, and ``drop_count``.
* Preserve per-producer record order.
* Avoid a global total-ordering point on the producer hot path.
* Preserve discard and lossless behavior, but implement the policy per shard.
* Avoid syscalls on normal record insertion.
* Keep shared-memory descriptors and process-attachment control messages private
  to the runtime or collector transport instead of exposing them in the public
  buffer API.

High-level architecture
=======================

The proposed transport separates the public buffer handle from the storage used
by individual producers:

.. code-block:: text

  rocprofiler_buffer_id_t
          |
          v
  logical buffer state
          |
          +-- producer shard 0: SPSC ring
          +-- producer shard 1: SPSC ring
          +-- producer shard 2: SPSC ring
          +-- ...
          |
          v
  drain thread or collector process
          |
          v
  rocprofiler_buffer_tracing_cb_t(headers, num_headers, drop_count)

The logical buffer owns policy, watermark, callback, context, and drop accounting.
Each producer owns a shard that contains fixed-size ring control fields and
variable-size record slots. The drain side walks registered shards, converts
committed slots into the callback's pointer array, invokes the callback, and then
advances the consumer cursor.

In-process backend
==================

In-process use should be the lowest-overhead mode because no inter-process
coordination is required. The recommended backend is:

* allocate shard memory from normal page-aligned process memory;
* create shards lazily for active producer threads or internal producer objects;
* store the producer's active shard in thread-local or producer-local state;
* use monotonically increasing producer and consumer cursors;
* publish records with a release-store commit marker;
* drain records with acquire loads on the consumer side;
* wake the callback thread only at watermark, explicit flush, finalization, or
  lossless-pressure points.

The producer path should avoid a global lock, a shared multi-producer cursor, and
kernel transitions. A fast-path insertion should only need local cursor math, a
space check, the payload write, and a commit-store. If the shard is full, the
backend applies the buffer policy locally: increment the logical drop count for
discard mode or wait for the shard consumer cursor in lossless mode.

The POC backend uses ``user_ring_record_header_buffer`` as the internal storage
type when ``ROCPROFILER_EXPERIMENTAL_USER_RING_BUFFER=ON``. Each producer thread
lazily creates one shard for each logical buffer object. The shard uses
page-aligned process-private memory by default and stores the slot header,
``rocprofiler_record_header_t``, and payload in one contiguous slot.

Out-of-process backend
======================

For an out-of-process tool, the same shard layout can be backed by shared memory:

* create a shared segment for the logical buffer control block and shard metadata;
* allocate ring storage with ``memfd_create`` plus ``mmap(MAP_SHARED)`` on Linux,
  with ``shm_open`` as a possible portability fallback if needed;
* pass file descriptors over the private attachment or collector control channel;
* map producer shards into the instrumented process and collector process;
* use ``eventfd`` or futex-style wakeups for watermark, flush, shutdown, and
  lossless-pressure notifications;
* keep record insertion syscall-free until a wakeup or backpressure transition is
  required.

The shared control block should include an ABI version, page size, byte order
marker, feature flags, logical buffer id, shard count, per-shard state, producer
identity, generation number, and lifecycle state. The collector must be able to
detect a producer that exits without finalizing and drain all committed records
whose commit markers are visible.

The POC does not add the collector control channel yet. To exercise the storage
mode, setting ``ROCPROFILER_EXPERIMENTAL_USER_RING_BUFFER_SHARED_MMAP=1`` makes
the POC allocate shard storage with anonymous shared ``mmap`` instead of
process-private memory. A production out-of-process backend still needs file
descriptor handoff, collector lifecycle, and crash-recovery protocol work.

Ordering model
==============

The lowest-overhead ordering contract is per-producer ordering, not global
insertion ordering. Each shard preserves the order of records produced by that
thread or producer. Cross-thread order should be reconstructed by timestamps,
correlation ids, and existing record metadata.

Requiring a globally ordered ring would reintroduce a shared multi-producer
atomic cursor and contention at exactly the point this proposal is trying to
remove. A global order can still be produced later by the tool or output writer
when it has all records available.

Record layout
=============

Each slot should carry enough metadata for the consumer to validate and skip
records safely:

* total record size;
* category and kind values;
* payload offset or inline payload start;
* producer id and sequence number;
* flags for committed, padding, wrap marker, dropped-record marker, or
  end-of-stream;
* optional checksum or generation field for shared-memory diagnostics.

The in-process callback path can pass pointers directly into process-private
ring memory while the callback is running. The out-of-process path can pass
pointers into the collector's mapping of the same shared segment, or copy only
when a consumer requires a different lifetime. The first implementation should
prefer zero-copy callback delivery and document that record pointers are valid
only for the duration of the callback, matching the existing buffered-service
usage pattern.

Backpressure and flush behavior
===============================

The current buffer policy maps naturally to sharded rings:

``ROCPROFILER_BUFFER_POLICY_DISCARD``
  If a producer shard has insufficient free space, the producer increments a
  drop counter and skips the record. Drop counters can be per shard and folded
  into the logical buffer's ``drop_count`` during drain.

``ROCPROFILER_BUFFER_POLICY_LOSSLESS``
  If a producer shard has insufficient free space, the producer requests a drain
  and waits until the consumer cursor advances. The wait path can start with a
  bounded spin or yield in-process, then fall back to callback-thread wakeup. In
  shared-memory mode, the wait path can use futex-style waits or ``eventfd``.

``rocprofiler_flush_buffer``
  An explicit flush marks all shards under the logical buffer as drain targets
  and waits until all committed records visible at the flush boundary are
  delivered.

Watermarks should be tracked per shard to keep producer checks local. The
logical buffer can also maintain an approximate aggregate watermark for callback
batch sizing, but producer hot paths should not update a globally contended byte
counter for every record.

Footprint controls
==================

The main cost of per-producer rings is memory footprint. The backend should use
bounded and lazy allocation:

* allocate a shard only when a thread or producer first emits a record;
* cap the number of active shards per logical buffer;
* choose small default shard sizes for API tracing and larger sizes for high-rate
  activity streams;
* reclaim inactive shards after thread exit and final drain;
* allow environment or tool configuration to tune shard size and shard count;
* store shared control metadata in a compact control page or small control
  mapping instead of one mapping per field.

For out-of-process collection, the descriptor count should also be bounded. A
single shared memory object per logical buffer plus offsets to shards is usually
preferable to one file descriptor per producer.

Comparison with other transport choices
=======================================

.. list-table::
   :header-rows: 1

   * - Transport
     - In-process producer overhead
     - Out-of-process producer overhead
     - Main tradeoff
   * - Current double-buffer backend
     - Simple, but shared buffer state can contend
     - Requires an additional export path
     - Lowest implementation risk
   * - Kernel BPF ring buffer
     - Higher than needed because user-space records cross a kernel interface
     - Good wakeup and polling model, but still kernel-mediated
     - Useful as inspiration, not ideal for SDK-internal user-space producers
   * - Single shared MPSC user-space ring
     - One shared producer cursor can contend under high thread counts
     - Works with shared ``mmap``
     - Lower footprint than sharding, higher hot-path contention
   * - Per-thread or per-producer SPSC rings
     - Lowest producer overhead because each producer writes a local shard
     - Lowest practical shared-memory option when paired with mmap and eventfd
     - More shard lifecycle and drain complexity

The recommendation is to use per-thread or per-producer SPSC rings as the common
transport model, with separate storage backends for process-private memory and
shared ``mmap`` memory.

Migration plan
==============

The proposal can be implemented in staged PRs:

1. Add an internal ring-shard abstraction behind
   ``ROCPROFILER_EXPERIMENTAL_USER_RING_BUFFER``.
2. Add an in-process backend for one logical buffer using per-thread shards while
   preserving the existing callback ABI.
3. Add tests for direct shard behavior plus discard, lossless, watermark, explicit flush,
   thread exit, and callback-thread assignment.
4. Add microbenchmarks for hot API tracing, kernel dispatch records, and
   multi-thread producer pressure compared with the current double-buffer path.
5. Add the shared-memory storage backend and collector control block for
   out-of-process collection.
6. Add crash and detach tests for collector exit, producer exit, stale shared
   memory, and version mismatch.

Open questions
==============

The implementation should settle these details before enabling the backend by
default:

* whether shards should be per thread, per internal producer object, or selected
  by service;
* default shard size and maximum shard count;
* exact shared-memory setup path for process attachment and external collectors;
* whether in-process lossless waiting should use spin-yield, condition variable,
  futex, or the existing callback-thread task group;
* whether record slots should be variable-sized directly or use fixed-size
  descriptors pointing into a per-shard byte arena;
* how much global ordering, if any, is required by existing output writers.

Expected result
===============

For both in-process and out-of-process transfer, this design minimizes the work
done by producer threads. The in-process hot path stays entirely in user-space
and writes process-private memory. The out-of-process hot path writes the same
record format into shared ``mmap`` memory and pays a kernel cost only for
wakeups, setup, teardown, or backpressure. This makes the ring-shard design the
lowest-overhead common direction while keeping the current ROCprofiler-SDK
buffer API stable for tools.
