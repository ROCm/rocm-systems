# Phase 1 Firmware-Ring Drainer — Testing Gap

**Status:** Open. Acknowledged at the close of Phase 1; deferred until the
build/CI environment is restored.

## 1. Scope of the gap

Phase 1 of firmware-assisted kernel-dispatch tracing introduces a drainer
state machine (`kernel_dispatch/firmware_ring_drainer.cpp:480-562`,
specifically `drain_all` and its supporting helpers) that:

- walks per-queue ring slots from a per-ring `read_cursor`,
- pairs START (record_type==1) and END (record_type==2) records by exact
  `dispatch_idx` match via `pending_starts`,
- updates `last_consumed_dispatch_idx[slot]` to detect already-consumed
  slots on subsequent passes,
- emits paired records through `process_dispatch_record`,
- and is gated globally by the `drainer-start` predicate so it never runs
  concurrently with any context that needs the WriteInterceptor.

**Today this state machine has zero unit-level coverage and zero
end-to-end coverage.** The existing `tests/queue_intercept.cpp` exercises
only the queue-intercept ring (`QueueState`, `virtual_wptr`,
`next_submit_pos`, etc.) and emulates the drainer in only the trivial
"consume from a queue" sense (see comment at line 354). It does not
validate any of the invariants that determine drainer correctness.

The build environment was offline at the time this gap was identified,
so a fake-firmware-ring fixture could not be authored and verified. This
document records the design for that fixture and the invariants it must
test, so the work can be picked up cleanly once builds and CI are
working again.

## 2. Invariants that must be tested

The three invariants below are load-bearing for the drainer and are the
ones most likely to regress silently if refactored. They are quoted from
the Phase 1 spec / KNOWN_ISSUES tracker; each has a corresponding
fixture-level test that should exist.

### I6 — Mutual exclusion against the WriteInterceptor

The drainer MUST NOT start when **any** context needs the
WriteInterceptor (counter collection, ATT, scratch reporting, or PC
sampling). The gate is global across all active contexts, not
per-context.

Test surface:

- Construct a fake context table where one context wants counters and
  another wants tracing-only. Assert the drainer-start gate returns
  false.
- Drop the counter context. Assert the gate returns true.
- Add an ATT context, then a scratch context, then a PCS context, one
  at a time. Each must individually flip the gate to false.
- Cover the regression in commit `c82889ecca` ("drainer-start gate
  evaluated globally across contexts"): a per-context evaluation that
  passes when at least one tracing-only context exists must NOT cause
  the drainer to start while another context still wants the WI.

### I7 — Per-queue lookup, no aql_qs fallback

When the drainer encounters a record with a given (agent, queue_id,
ring slot) tuple, it MUST consume from that source queue's drainer
slots, **not** from any aql_qs fallback table. A regression here would
cause cross-queue cross-talk: a START on queue A paired with an END on
queue B because the drainer fell back to a shared lookup.

Test surface:

- Build two fake `queue_ring_state_t` instances, each with its own
  `read_cursor`, `pending_starts`, and `last_consumed_dispatch_idx`
  array. Inject START on ring A, END on ring B, both with the same
  `dispatch_idx` value. Assert NO dispatch is emitted.
- Inject START + END on ring A (matching dispatch_idx). Assert exactly
  one dispatch is emitted, with the start_ts from ring A's START.
- Verify `corr_lookup_qs == &qs` always (i.e. the drainer never
  rebinds the correlation-lookup pointer to a fallback ring).

### I9 — Per-ring cursor + dispatch_idx pairing (no timestamp heuristics)

Pairing of START and END records MUST use:

1. cursor advancement per slot (the `qs.read_cursor` field), and
2. **exact** `dispatch_idx` match between START and END.

It MUST NOT use timestamp ordering, timestamp windows, or any other
heuristic. Timestamp heuristics break when firmware writes records
out of timestamp order (which is permitted) or when multiple
in-flight dispatches have overlapping intervals.

Test surface:

- Inject a START record at slot 3 with dispatch_idx=42, ts=100. Inject
  an END at slot 4 with dispatch_idx=99, ts=50 (earlier!). Assert no
  pairing happens for dispatch_idx=42.
- Add an END at slot 5 with dispatch_idx=42, ts=200. Assert pairing
  succeeds with start_ts=100, end_ts=200.
- Walk the ring twice without advancing firmware. Assert the second
  drain pass is a no-op (the `last_consumed_dispatch_idx[slot] ==
  r16.dispatch_idx` early-out at line 504 fires).
- Wrap `read_cursor` past `num_slots` and verify modular arithmetic
  (line 511 / 539) works.
- Inject an unknown `record_type` (e.g. 7). Assert the cursor still
  advances and `last_consumed_dispatch_idx[slot]` is updated, so the
  drainer can't get stuck on a bad slot.

## 3. Fake-firmware-ring fixture design

The fixture is the missing piece that blocks all of the tests above. It
must stand in for the firmware-written ring buffer plus the HSA queue
queries the drainer makes against it. Sketch:

```cpp
struct FakeFirmwareRing
{
    std::vector<mec_dispatch_record_16> slots;  // backing storage
    uint64_t                            read_idx;   // simulated READ_INDEX
    uint64_t                            write_idx;  // simulated WRITE_INDEX

    void inject_start(uint32_t slot, uint32_t dispatch_idx, uint64_t ts);
    void inject_end  (uint32_t slot, uint32_t dispatch_idx, uint64_t ts);
    void inject_raw  (uint32_t slot, const mec_dispatch_record_16&);
    void advance_write(uint32_t n);  // simulate firmware producing
};
```

Required shims:

1. **`hsa_amd_queue_get_info` shim** — returns the fixture's `read_idx`
   and `write_idx` instead of going to live HSA. The drainer calls this
   to figure out how many slots to walk; without a shim the drainer
   either crashes (no live runtime) or returns nonsense.
2. **Fake queue/agent objects** — `queue_ring_state_t` holds a pointer
   to a `hsa_queue_t*` and an agent handle. The fixture supplies dummy
   non-null values; the drainer never dereferences them beyond passing
   them back through the correlation path.
3. **Fake timestamp conversion** — `process_dispatch_record` converts
   firmware ticks to host ns through the agent's clock-conversion table.
   The fixture must inject an identity (or scaled) conversion so test
   assertions can be written in raw units.
4. **Capture sink for emissions** — replace
   `process_dispatch_record`'s downstream emit with a test-owned
   collector (vector of `(dispatch_idx, start_ts, end_ts)` tuples) so
   tests can assert exactly what was paired.

The fixture lives in `tests/` next to `queue_intercept.cpp`, in a new
`firmware_ring_drainer.cpp` test translation unit (or split into
`fake_firmware_ring.hpp` + `firmware_ring_drainer.cpp` if the helper
grows useful for other tests).

## 4. Adjacent regression test to add at the same time

Independent of the drainer state-machine tests, add a regression test
for the **INLINE-ctor `_mode` default bug** that was fixed by commit
`0b7ba9e789` ("hsa::Queue: propagate Mode through INLINE ctor"). The
bug class: a `Queue` constructor variant silently defaults `_mode` to
`full_intercept` because it doesn't take a Mode parameter, so an
INLINE-constructed queue intended for tracing-only ends up doing full
WI work.

Test shape:

- For every public `Queue` ctor overload, construct an instance and
  assert `q.mode() == ctor_arg_mode` for every Mode value the caller
  could have passed. A ctor that doesn't take a Mode and silently
  defaults will fail this matrix.
- This is a static-style test that catches the mistake at the API
  surface, not in observable behavior — exactly the level the existing
  test suite cannot reach today.

## 5. When this gap closes

This document should be deleted (or moved to a "completed" archive)
when:

1. A `tests/firmware_ring_drainer.cpp` translation unit exists,
2. it exercises I6, I7, and I9 with the fixture described above,
3. the INLINE-ctor Mode-default regression test is in place,
4. all of the above run green in CI.

Until then this file is the single source of truth for what's missing,
why it's missing, and how to build it.
