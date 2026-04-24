# Known Issues — MEC Firmware-Assisted Dispatch Tracing

This document lists issues, hacks, and missing pieces in the firmware
dispatch ring path for kernel-dispatch tracing. The code path is gated on
`hsa::firmware_dispatch_ring_available()` and is otherwise inactive.

Each item is referenced by `TODO(KNOWN_ISSUES.md item N)` comments at the
relevant `file:line` site.

> **See also:**
> - `HIGH_LEVEL_DESIGN_SUMMARY.md` — concise high-level summary of
>   the overall design.
> - `FIRMWARE_RING_HYBRID_DESIGN.md` — proposed hybrid that resolves
>   items 1, 7, 8, 9, 10 by combining the firmware ring with a
>   launching-thread doorbell hook (no SDK-allocated completion
>   signals).
> - `KFD_DISPATCH_LOG_DESIGN.md` — proposed KFD/UAPI evolution that
>   moves dispatch-log setup from `UPDATE_QUEUE` into the existing
>   profiler ioctl, plus a self-describing JSON descriptor so the SDK
>   no longer hardcodes record format.
> - `TRACING_DELIVERY_RESEARCH.md` — research on replacing the
>   HIP/HSA → rocprofiler-sdk callback delivery with a generic
>   emit-and-subscribe transport (LTTng-UST primary,
>   `user_events` future). Medium-term parallel track.

## Scope

The MEC firmware writes 16-byte dispatch records into a host-visible ring
buffer per HSA queue. The host SDK polls the ring, pairs START/END
records, resolves kernel objects from the AQL packet ring, and emits
`ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH` /
`ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH` records.

The path bypasses HSA queue interception. The known issues fall into three
buckets:

* **Correctness gaps** — items 1, 2, 9 produce records that are wrong or
  unlinked to the rest of the trace.
* **Robustness/scalability** — items 3, 4, 5, 6 are fine in short-lived
  test runs but will misbehave at scale or under stress.
* **Feature gaps** — items 7, 8, 10 are missing data fields and missing
  call sites.

---

## 1. Correlation IDs are decoupled from API tracing

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`process_dispatch_record()` (`fallback_cid` block).

The drainer constructs a fresh `correlation_id` per record at drain time.
Standard kernel-dispatch tracing assigns the correlation id at enqueue
time inside `hsa::WriteInterceptor`, where the HIP API wrapper's
correlation id is on the per-thread stack and gets propagated into the
dispatch.

Consequences:

* Internal correlation IDs do not link kernel records to HIP/HSA API
  records.
* Ancestor correlation IDs are always zero.
* External correlation IDs pushed by the tool before
  `hipLaunchKernel` are not seen by the dispatch record.

If `correlation_tracing_service::construct()` returns null during
finalization (the service has begun teardown), the drainer hands back a
`thread_local` zero-initialized `correlation_id` so the record still
flows. Records emitted via this fallback have all-zero correlation IDs
that cannot be distinguished from a genuine zero correlation.

**Real fix:** either intercept HIP API entry points to push a correlation
id even in firmware-ring mode, or implement a flush handshake so the
drainer drains before the correlation service tears down.

## 2. Multi-XCC START/END pairing is heuristic

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` — `drain_all()`.

START/END records are paired by selecting, for each END, the most recent
pending START with the smallest positive duration. This works for
serialized kernels and reasonably-spaced concurrent kernels. For
concurrent kernels on different XCCs that overlap in time and have
similar durations, the heuristic will pair START and END records from
different XCCs and produce wrong durations.

**Real fix:** the firmware record needs to include an XCC/pipe id (one of
the reserved scratch fields can hold it). Pairing then becomes per-XCC
and trivially correct.

## 3. Shutdown grace period is a fixed 10ms sleep

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`stop_firmware_dispatch_ring_drainer()`.

`stop_firmware_dispatch_ring_drainer()` sleeps for 10ms before signaling
the drainer to stop, in the hope that the drainer's 1ms polling loop will
catch any records emitted by recently-completed kernels.

**Real fix:** quiesce the firmware (no new dispatches in flight), then
explicitly drain and join.

## 4. Hard-coded 1ms drainer poll cadence

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`drainer_loop()`.

The drainer thread wakes every 1ms regardless of activity. On idle
processes this is 1000 wakeups/sec.

**Real fix:** adaptive backoff (e.g., exponential, capped at some upper
bound), or condition-variable wakeups driven by an HSA signal that the
firmware can pulse.

## 5. `g_emitted_dispatch_idx` and `last_processed_record_count` grow unboundedly

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`g_emitted_dispatch_idx`, `queue_ring_state_t::last_processed_record_count`.

`g_emitted_dispatch_idx` is a process-global `unordered_set<uint32_t>` used
to deduplicate records across drain passes. It is cleared only at
shutdown.

`last_processed_record_count` is a per-queue monotonic counter used to
short-circuit the rescan when no new records have arrived.

Two failure modes for long-running processes:

1. `g_emitted_dispatch_idx` memory grows linearly with dispatch count.
2. After 2^32 dispatches, the firmware's `dispatch_idx` (which is
   `read_dispatch_id[31:0]`) wraps. `g_emitted_dispatch_idx` will then
   suppress legitimate new records.

**Real fix:** replace with a per-queue ring of recently-seen indices
sized to `queue->size` (or a small multiple), and use the underlying
`read_dispatch_id` 64-bit width if exposed.

## 6. `hsa_amd_profiling_set_profiler_enabled(true)` is sticky

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`register_or_refresh_queue()`.

We unconditionally enable profiling on every discovered GPU queue and
never disable it. This is an externally-visible HSA state mutation that
persists past `stop_firmware_dispatch_ring_drainer()`.

**Real fix:** track per-queue enabled state, only enable on first
discovery, and disable on `stop_firmware_dispatch_ring_drainer()` for
queues we enabled.

## 7. Late-attach kernel-symbol resolution falls back to raw `kernel_object`

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`emit_kernel_dispatch_tracing()` (kid fallback).

If `code_object::get_kernel_id()` returns zero — typically because the
code-object load callback ran before the tool attached — the drainer
substitutes the raw `kernel_object` GPU VA as a synthetic `kernel_id`.
Tools doing `kernel_id`-keyed lookups against
`ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT` records will see ids they
never registered.

**Real fix:** at attach time, enumerate all loaded executables and
synthesize the corresponding code-object load callbacks before any
dispatch records can be emitted.

## 8. `workgroup_size`, `grid_size`, `private_segment_size`, `group_segment_size` are zeroed

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`emit_kernel_dispatch_tracing()` (`dispatch_info` setup).

The interception path captures these from the AQL packet at enqueue.
The drainer can read them from the same AQL packet slot it already reads
the `kernel_object` from (`lookup_kernel_object`); we just don't yet.

**Real fix:** extend `lookup_kernel_object` to return the relevant fields
or pass back the whole packet by value.

## 9. `thread_id` is the drainer thread's tid

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`emit_kernel_dispatch_tracing()` (`thr_id`).

The interception path captures the launching thread's tid at enqueue.
The drainer runs on its own thread and has no way to know who launched
the kernel.

**Real fix:** if HIP API entry points are intercepted (see item 1), the
launching tid can be propagated through the AQL packet's user data
field.

## 10. `KERNEL_DISPATCH_ENQUEUE` callbacks never fire

The firmware-ring path bypasses `hsa::WriteInterceptor`, which is the
source of `ROCPROFILER_KERNEL_DISPATCH_ENQUEUE` ENTER/EXIT phase
callbacks. Tools relying on the ENQUEUE phase will silently see nothing.
Only the `COMPLETE` phase fires.

**Real fix:** if/when API interception is added (see item 1), fire the
ENQUEUE callbacks from the API entry point.

---

## Out of scope for this branch

* No tests added — existing kernel-dispatch tests exercise only the
  interception path. A firmware-ring path test would need either a
  firmware-capable target or a faked ROCr that produces synthetic ring
  buffers.
* `cmake/rocprofiler_sdk_options.cmake` does not gate this code on a
  build flag. The path is silently inactive without runtime support.
