# Plan: Hybrid MEC Firmware-Ring + Inline Doorbell Correlation Capture

> **Status:** Design proposal. Not yet implemented.
> Companion to `KNOWN_ISSUES.md` (which catalogs the issues this design addresses).

> **Status update — 2026-04-30.** This SDK-side hybrid (drainer + doorbell
> wrapper + per-queue side table) has been overtaken by an alternative
> implementation that moves the drainer **out of rocprofiler-sdk and into
> the HSA runtime**, then delivers the FW kernel-dispatch records via the
> same LTTng-UST channel that already carries HIP/HSA API events
> (PRs #5475 and #5513 — see `TRACING_DELIVERY_RESEARCH.md` and
> `HIGH_LEVEL_DESIGN_SUMMARY.md`). That work lives on
> `users/bewelton/lttng-kernel-ts` (PR #5519, draft) at HEAD `e6abbfc7fa`,
> with a per-queue drainer thread emitting `rocm_hsa:kernel_dispatch_record`
> events and a measured 169% combined-record capture rate on graphbench
> (~85% per record_type). It bypasses PR 5219 entirely (no doorbell
> wrapper, no per-queue side table, no SDK-allocated state).
>
> **The body of this plan is preserved in full** because (a) it remains
> the reference design for the SDK-side path if we choose to revive it
> (e.g., if LTTng-UST consumption proves unworkable for some downstream
> tool), and (b) several of its mechanisms — particularly the doorbell
> wrapper from PR 5219 — are still relevant to other rocprofiler-sdk work.
> See **Section 13** below for how the rocprofiler-sdk now plans to
> integrate against the HSA-resident drainer (two paths: consume LTTng
> directly, or bypass HSA and query KFD via the same ioctl).

This document describes a planned evolution of the MEC firmware-assisted
dispatch tracing path (currently implemented in
`source/lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.{cpp,hpp}`)
to recover correlation IDs and ENQUEUE callbacks **without allocating any
HSA completion signals from rocprofiler-sdk**.

## Section 1 — Goal and Explicit Non-Goals

**Goal:** Combine the MEC firmware-ring drainer (this branch) with a
stripped-down launching-thread doorbell hook (inspired by PR 5219's inline
queue intercept) so that `rocprofiler-sdk` can produce fully correlated
`KERNEL_DISPATCH` ENQUEUE+COMPLETE records with start/end timestamps and
*zero* SDK-allocated completion signals.

**Non-goals:**

1. **Counter collection (`dispatch_counter_collection`) and ATT
   (`dispatch_thread_trace`).** Both require AQL packet expansion (PM4
   instrumentation packets, barrier-with-completion, serializer
   interactions). Those contexts must continue to use the standard
   `WriteInterceptor` path and continue to allocate signals.
2. **PC sampling integration.** This branch already excludes PC-sampling
   contexts from the firmware-ring path
   (`hsa/queue_controller.cpp:567-573`). Hybrid stays out of PCS.
3. **Replacing the standard interception path on hardware that lacks
   `hsa_amd_profiling_get_dispatch_records` and `hsa_amd_queue_iterate`.**
   Both paths must coexist; gate at runtime on
   `firmware_dispatch_ring_available()`.
4. **Removing `WriteInterceptor`** (`hsa/queue.cpp:298-708`). It stays
   as-is for counters/ATT/scratch/PCS, where its signal allocation and
   packet expansion are still needed.
5. **Fixing every item in `KNOWN_ISSUES.md`.** Items 1, 7, 8, 9, 10 are
   addressed by the hybrid; items 2, 5, 6 are scoped as Phase-3
   follow-ups; items 3, 4 are out of scope (separate work).
6. **Virtualizing the queue write index.** PR 5219 wraps
   `add_write_index_*`, `cas_write_index_*`, `store_write_index_*`,
   `load_write_index_*` to virtualize `wptr` so `WriteInterceptor` can
   rewrite/expand packets. Because the hybrid does **not** call
   `WriteInterceptor` and does **not** rewrite packets, none of these
   wrappers are needed. The application's writes go directly to the real
   AQL ring; the doorbell wrapper is the only interception point.

---

## Section 2 — Architecture

### 2.1 Data flow (steady state)

```
                 LAUNCHING THREAD (e.g., HIP runtime worker)
                 =============================================
HIP API call (hipLaunchKernel)
  └─> HIP wrapper (rocprofiler tracing wrapper around hipLaunchKernel)
        ├─ pushes correlation_id onto TLS stack (the ENTER callback path)
        └─> real hipLaunchKernel
              └─> HIP queues an AQL kernel dispatch packet:
                    1. real hsa_queue_add_write_index_screlease(q, 1)  ─── NOT wrapped
                       (returns prev wdid = D)
                    2. application writes packet to q->base_address[D & mask]
                    3. real hsa_signal_store_screlease(q->doorbell, D)
                       ──────────────── WRAPPED by us ────────────────
                          │
                          ▼
                  wrap_signal_store_screlease(sig, val)
                    ├─ doorbell_map.find(sig.handle)  → queue_state*
                    ├─ snapshot real wdid range (prev_published, val+1)
                    ├─ for each newly visible packet slot in that range:
                    │     get correlation_id = context::get_latest_correlation_id()
                    │     side_table[queue].slot[D & mask] =
                    │            { corr_id, ext_corr_ids snapshot, tid, enqueue_ts }
                    │     corr_id->add_ref_count(); corr_id->add_kern_count();
                    │     fire ROCPROFILER_KERNEL_DISPATCH_ENQUEUE ENTER+EXIT
                    │            (synchronously, on this thread, with this tid)
                    └─ chain through to real hsa_signal_store_screlease(sig, val)
                       (no packet rewriting; no completion signal allocated;
                        no barrier injection; no WriteInterceptor invocation)

                                  ─── GPU executes the kernel ───
                                              │
                                              ▼
                            MEC firmware writes 16-byte records into
                            queue's host-visible profiling ring:
                              { ts_lo, ts_hi, record_type=1 (START), dispatch_idx }
                              { ts_lo, ts_hi, record_type=2 (END),   dispatch_idx }


             DRAINER THREAD (`firmware_dispatch_drainer`, 1ms cadence)
             =========================================================
poll all registered queue rings
  ├─ pair START/END by dispatch_idx (smallest-positive-gap heuristic for now)
  ├─ for each completed dispatch:
  │     entry = side_table[queue].slot[dispatch_idx & mask]
  │     if entry valid:
  │         emit KERNEL_DISPATCH_COMPLETE record with:
  │             tid             = entry.tid           (LAUNCHING tid, not drainer tid)
  │             internal_corr   = entry.corr_id->internal
  │             ancestor_corr   = entry.corr_id->ancestor
  │             external_corr   = entry.external_corr_ids
  │             start/end ts    = converted from MEC ticks
  │         entry.corr_id->sub_kern_count(); entry.corr_id->sub_ref_count();
  │         clear side_table slot
  │     else:                       # late-attach kernel that started before hook installed
  │         emit with a fallback zero correlation (same as today's code)
  │
  └─ NO HSA SIGNAL IS EVER QUERIED OR ALLOCATED
```

### 2.2 What is captured where

| Datum                         | Where captured              | When captured           | Source                                                             |
|-------------------------------|-----------------------------|-------------------------|--------------------------------------------------------------------|
| internal correlation id       | Launching thread            | Doorbell store time     | `context::get_latest_correlation_id()` TLS stack                   |
| ancestor correlation id       | Launching thread            | Doorbell store time     | Same TLS stack (`corr_id->ancestor` field)                         |
| external correlation ids      | Launching thread            | Doorbell store time     | Per-context `external_correlator.get(...)`                         |
| launching thread id           | Launching thread            | Doorbell store time     | `common::get_tid()`                                                |
| enqueue timestamp             | Launching thread            | Doorbell store time     | `common::timestamp_ns()`                                           |
| dispatch_id (sequence number) | Launching thread            | Doorbell store time     | Atomic counter (replaces today's `g_next_dispatch_id`)             |
| kernel object / kernel_id     | Drainer thread              | Drain time              | AQL packet at `queue->base_address[idx & mask]`                    |
| workgroup_size, grid_size, segments | Drainer thread        | Drain time              | Same AQL packet (Phase 3 — KNOWN_ISSUES item 8)                    |
| start / end timestamps        | Drainer thread              | Drain time              | MEC firmware ring (16-byte record), converted via                  |
|                               |                             |                         | `hsa_amd_profiling_convert_tick_to_system_domain`                  |
| **completion signal**         | **NEVER ALLOCATED**         | —                       | —                                                                  |

---

## Section 3 — Component Inventory

| # | File                                                                                         | Status     | Est. lines | Public API surface                                                                                                                                                          |
|---|----------------------------------------------------------------------------------------------|------------|-----------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `lib/rocprofiler-sdk/hsa/firmware_ring_correlation.hpp`                                      | **NEW**    |  ~120      | `install_correlation_hook(CoreApiTable&)`, `shutdown_correlation_hook()`, `register_queue(hsa_queue_t*)`, `unregister_queue(uint64_t qid)`, `lookup_and_consume(qid, idx)`  |
| 2 | `lib/rocprofiler-sdk/hsa/firmware_ring_correlation.cpp`                                      | **NEW**    |  ~350      | (impl)                                                                                                                                                                      |
| 3 | `lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.cpp`                              | MODIFIED   |  ~80       | (no API change — replace `correlation_tracing_service::construct(2)` block with `lookup_and_consume()`)                                                                     |
| 4 | `lib/rocprofiler-sdk/hsa/queue_controller.cpp`                                               | MODIFIED   |  ~30       | (no API change — install/shutdown calls in init/fini)                                                                                                                       |
| 5 | `lib/rocprofiler-sdk/hsa/queue.cpp`                                                          | unchanged  |   0        | —                                                                                                                                                                           |
| 6 | `lib/rocprofiler-sdk/registration.cpp`                                                       | MODIFIED   |  ~10       | (no API change — gate logic added)                                                                                                                                          |
| 7 | `hsa/CMakeLists.txt`                                                                         | MODIFIED   |   2        | Add new source files                                                                                                                                                        |

We deliberately **do not** create a new `queue_intercept`-style
virtualization. PR 5219's `QueueState`/`virtual_wptr`/`real_rdid`
machinery exists to gate packet rewriting; we don't rewrite packets. Our
`CorrEntry` table is much simpler.

We deliberately **do not** modify `queue.cpp:WriteInterceptor`. Any
context that needs counter/ATT/scratch/PCS still goes through it
unchanged.

---

## Section 4 — Detailed Design of `firmware_ring_correlation.{cpp,hpp}`

### 4.1 Per-queue side table

```cpp
namespace rocprofiler::hsa::firmware_ring_correlation {

struct CorrEntry {
    // populated on launching thread at doorbell store time
    context::correlation_id*       corr_id   = nullptr;
    rocprofiler_thread_id_t        tid       = 0;
    uint64_t                       enqueue_ts = 0;
    // tracing::external_correlation_id_map_t snapshot — small map (~contexts).
    tracing::external_correlation_id_map_t external_corr_ids;
    // monotonically incremented sequence_id for ordering / debugging
    uint64_t                       seq       = 0;
    // generation counter — incremented when the slot is first written; lets
    // the drainer detect a stale/never-written slot vs a fresh one.
    std::atomic<uint64_t>          gen       {0};
};

struct QueueCorrState {
    const hsa_queue_t*           hsa_queue = nullptr;
    hsa_signal_t                 doorbell  = {0};
    uint32_t                     ring_mask = 0;        // queue->size - 1
    std::vector<CorrEntry>       slots;                // size == queue->size
    // last published wptr we observed — used to know which slots are new.
    std::atomic<uint64_t>        last_observed_wdid {0};
    // serializes producers writing into slots for the SAME slot index.
    std::mutex                   slot_publish_mu;
};

using QueueCorrStatePtr = std::shared_ptr<QueueCorrState>;

// keyed by hsa_queue_t::id (not pointer — drainer keys by id too)
using queue_table_t   = common::Synchronized<std::unordered_map<uint64_t, QueueCorrStatePtr>>;
// keyed by doorbell signal handle, weak ptr (mirrors PR 5219 lifetime model)
using doorbell_map_t  = common::Synchronized<std::unordered_map<uint64_t, std::weak_ptr<QueueCorrState>>>;

queue_table_t&  get_queue_table();
doorbell_map_t& get_doorbell_map();

}
```

**Sizing.** Each ring is `queue->size` slots (matches the AQL ring's
modular indexing, so `dispatch_idx % size` always uniquely identifies the
live slot at the moment we see it). Memory per queue is
`O(size * sizeof(CorrEntry))`; for a typical 4096-slot queue with ~5
contexts that's about 4096 × ~96 bytes = ~400 KB per queue. Acceptable.
If it bites we can reduce to a smaller ring sized to the GPU's
outstanding-dispatch capacity (typically ≤ 64).

**Lifetime.** `shared_ptr<QueueCorrState>` keeps the table alive past
queue destruction until any in-flight drainer pass that holds a
`shared_ptr` finishes — same pattern as `queue_intercept::QueueState`.

**Thread-safety.**
- The slot itself is single-writer at any moment because the launching
  thread that owns dispatch index `D` is the one that wrote slot
  `D & mask` (HSA enforces serialization of the application's
  `add_write_index → write packet → ring doorbell` sequence per packet).
  The slot is read by the drainer thread when the END record arrives. We
  use a release-store of `gen` at end of writer + acquire-load on reader.
- The `queue_table_t` and `doorbell_map_t` are wrapped in
  `common::Synchronized` like PR 5219's registries.

### 4.2 Doorbell wrapper (only `hsa_signal_store_relaxed` and `hsa_signal_store_screlease`)

We wrap exactly the two signal stores that HIP/HSA use to ring a
doorbell. We **do not** wrap any of the write-index APIs. This is a
deliberate departure from PR 5219.

```cpp
// Saved next-in-chain (could be tracing functor or raw HSA, depending on
// install order — same pattern as queue_intercept::s_next_table).
static CoreApiTable s_next_table = {};
static std::atomic<bool> s_installed = false;

inline bool bypass() {
    return !s_installed.load(std::memory_order_acquire) ||
           registration::get_fini_status() > 0;
}

void wrap_signal_store_screlease(hsa_signal_t sig, hsa_signal_value_t val)
{
    if (bypass()) {
        s_next_table.hsa_signal_store_screlease_fn(sig, val);
        return;
    }

    QueueCorrStatePtr st;
    get_doorbell_map().rlock([&](const auto& m){
        auto it = m.find(sig.handle);
        if (it != m.end()) st = it->second.lock();
    });

    if (!st) {                 // not one of our queues
        s_next_table.hsa_signal_store_screlease_fn(sig, val);
        return;
    }

    process_doorbell(st, val);
    s_next_table.hsa_signal_store_screlease_fn(sig, val);   // <-- ALWAYS chain through
}

void wrap_signal_store_relaxed(hsa_signal_t sig, hsa_signal_value_t val)
{
    /* identical, with _relaxed_fn variant */
}
```

### 4.3 `process_doorbell` — capture correlation per new packet

```cpp
void process_doorbell(const QueueCorrStatePtr& st, hsa_signal_value_t val)
{
    // HIP convention: hsa_signal_store_screlease(doorbell, new_wdid - 1).
    // After the doorbell fires, the queue's real wdid is exactly val + 1.
    const uint64_t new_wdid = static_cast<uint64_t>(val) + 1;

    std::lock_guard<std::mutex> g(st->slot_publish_mu);

    uint64_t prev = st->last_observed_wdid.load(std::memory_order_relaxed);

    // GPU may have processed/queued multiple packets since we last saw a
    // doorbell on this queue — capture every packet in [prev, new_wdid).
    if (new_wdid <= prev) return;   // stale/duplicate doorbell — ignore

    auto* corr_id = context::get_latest_correlation_id();
    bool  popping_corr = false;
    if (!corr_id) {
        corr_id = context::correlation_tracing_service::construct(1);
        popping_corr = true;
    }
    // If still null (finalization race), fall through; drainer will see no entry
    // and emit a zero-corr record (the existing fallback path).
    if (!corr_id) { st->last_observed_wdid.store(new_wdid); return; }

    auto _corr_id_dtor = common::scope_destructor{[&]{
        if (popping_corr) {
            context::pop_latest_correlation_id(corr_id);
            corr_id->sub_ref_count();
        }
    }};

    const auto tid        = corr_id->thread_idx ? corr_id->thread_idx : common::get_tid();
    const auto enqueue_ts = common::timestamp_ns();

    // Build the external-correlation-id snapshot once — it's the same for
    // every packet in this batch (same launching thread, same context state).
    tracing::tracing_data td{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                               ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                               td);
    tracing::populate_external_correlation_ids(
        td.external_correlation_ids, tid,
        ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,
        ROCPROFILER_KERNEL_DISPATCH_ENQUEUE,
        corr_id->internal);

    static std::atomic<uint64_t> s_seq{0};

    for (uint64_t d = prev; d < new_wdid; ++d) {
        const uint32_t slot = static_cast<uint32_t>(d & st->ring_mask);

        // Skip non-kernel-dispatch packets (barrier_and/or, agent_dispatch).
        const auto* pkts =
            static_cast<const hsa_kernel_dispatch_packet_t*>(st->hsa_queue->base_address);
        const uint16_t hdr = __atomic_load_n(&pkts[slot].header, __ATOMIC_ACQUIRE);
        const uint8_t  ptype = (hdr >> HSA_PACKET_HEADER_TYPE) &
                               ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u);
        if (ptype != HSA_PACKET_TYPE_KERNEL_DISPATCH) continue;

        // Per-dispatch ref-count lifecycle (mirrors queue.cpp:449-450)
        corr_id->add_ref_count();
        corr_id->add_kern_count();

        auto& entry = st->slots[slot];
        entry.corr_id           = corr_id;
        entry.tid               = tid;
        entry.enqueue_ts        = enqueue_ts;
        entry.external_corr_ids = td.external_correlation_ids;   // copy
        entry.seq               = s_seq.fetch_add(1, std::memory_order_relaxed);
        entry.gen.fetch_add(1, std::memory_order_release);       // publish

        // Phase 2: fire ROCPROFILER_KERNEL_DISPATCH_ENQUEUE ENTER + EXIT here.
        fire_enqueue_callbacks(td, st->hsa_queue, slot, tid, corr_id, entry.seq);
    }

    st->last_observed_wdid.store(new_wdid, std::memory_order_release);
}
```

### 4.4 How we compute `dispatch_idx` for each new packet

We use the **`signal_value` from the doorbell store**, with the HIP
convention that the value is `new_write_dispatch_id - 1`. So
`dispatch_idx_just_pushed = val`. To handle multi-packet doorbells (e.g.,
HIP graphs that push N packets and ring once with
`val = wdid + N - 1`), we capture the full range
`[last_observed_wdid, val + 1)` — every packet in that gap is "newly
visible to the GPU" and gets a slot entry.

**Justification vs alternatives:**
- *Reading the queue's real wdid directly* (e.g.,
  `__atomic_load_n(&queue->write_dispatch_id, ACQUIRE)`): there is no
  portable `wdid_addr` in `hsa_queue_t`; PR 5219 obtains it via the
  `hsa_amd_queue_intercept_create` callback. We don't want to use
  queue-intercept-create, so we don't have the address. The doorbell
  value is the source of truth and is always up to date.
- *Wrapping `add_write_index_*` to count packets ourselves*: that's
  exactly PR 5219's machinery. We want to avoid it because (a) it pulls
  in the `wait_for_free_slot`/`virtual_wptr` complexity and (b) it
  doesn't actually help for HIP private fast-paths that may bypass the
  wrapped functions but still ring the doorbell.

**Caveat:** The HIP convention `val == new_wdid - 1` is documented and
what `queue_intercept.cpp:139` relies on. If a non-HIP runtime ever rings
the doorbell with a different convention, we'd misidentify dispatches.
Acceptable for Phase 1; document as an open question.

### 4.5 Correlation-ID reference count protocol

**On launching thread (in `process_doorbell`):**
- `corr_id->add_ref_count()` once per dispatch packet (mirrors
  `queue.cpp:450`)
- `corr_id->add_kern_count()` once per dispatch packet (mirrors
  `queue.cpp:451`)

**On drainer thread (in `process_dispatch_record` after emitting record):**
- `entry.corr_id->sub_kern_count()` once per dispatch
- `entry.corr_id->sub_ref_count()` once per dispatch

Net effect: each kernel dispatch contributes +1/+1 on enqueue, -1/-1 on
completion — same accounting as the standard path's `dispatch_complete()`
retirement.

**What if the END record is never seen** (GPU hang, queue destroyed
mid-flight, late attach without a matching START):

1. **Queue destroyed cleanly** — `unregister_queue(q)` walks the queue's
   `slots` and for every slot whose `gen != 0` and whose corresponding
   `(queue_id, slot)` has not been emitted by the drainer, calls
   `sub_kern_count + sub_ref_count` and clears the slot.
2. **Process exits / drainer stops with kernels still in flight** —
   `shutdown_correlation_hook()` walks every queue's slots and decrements
   ref counts on any non-zero `gen` slot that still has a non-null
   `corr_id`.
3. **GPU hang** — same as (2): we don't distinguish.

### 4.6 ENQUEUE callback delivery (Phase 2)

We fire `ROCPROFILER_KERNEL_DISPATCH_ENQUEUE` ENTER and EXIT phases
synchronously from `process_doorbell` for each packet, in launching-thread
context. This recovers `KNOWN_ISSUES.md` item 10. Implementation mirrors
`queue.cpp:511-520` and `:632-637`:

```cpp
void fire_enqueue_callbacks(const tracing::tracing_data& td,
                            const hsa_queue_t* queue,
                            uint32_t slot,
                            rocprofiler_thread_id_t tid,
                            context::correlation_id* corr_id,
                            uint64_t dispatch_id)
{
    if (td.callback_contexts.empty()) return;

    const auto* pkts = static_cast<const hsa_kernel_dispatch_packet_t*>(queue->base_address);
    const auto& pkt = pkts[slot];
    const uint64_t kid = code_object::get_kernel_id(pkt.kernel_object);
    const auto* rocp_agent = /* lookup agent for queue */;

    constexpr auto sz = common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();
    auto info = common::init_public_api_struct(rocprofiler_kernel_dispatch_info_t{});
    info.size                 = sz;
    info.agent_id             = rocp_agent->id;
    info.queue_id             = rocprofiler_queue_id_t{queue->id};
    info.kernel_id            = kid;
    info.dispatch_id          = dispatch_id;
    info.private_segment_size = pkt.private_segment_size;
    info.group_segment_size   = pkt.group_segment_size;
    info.workgroup_size       = {pkt.workgroup_size_x, pkt.workgroup_size_y, pkt.workgroup_size_z};
    info.grid_size            = {pkt.grid_size_x, pkt.grid_size_y, pkt.grid_size_z};

    auto tracer = rocprofiler_callback_tracing_kernel_dispatch_data_t{
        sizeof(tracer), /*start*/0, /*end*/0, info};

    tracing::execute_phase_enter_callbacks(
        td.callback_contexts, tid, corr_id->internal, td.external_correlation_ids,
        corr_id->ancestor, ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
        ROCPROFILER_KERNEL_DISPATCH_ENQUEUE, tracer);

    tracing::update_external_correlation_ids(
        const_cast<tracing::external_correlation_id_map_t&>(td.external_correlation_ids),
        tid, ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH);

    tracing::execute_phase_exit_callbacks(
        td.callback_contexts, td.external_correlation_ids,
        ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
        ROCPROFILER_KERNEL_DISPATCH_ENQUEUE, tracer);
}
```

Phase 1 omits this for simplicity.

### 4.7 Late-attach: retroactively associating already-existing queues

This branch already calls `hsa_amd_queue_iterate(register_or_refresh_queue,
nullptr)` in the drainer loop
(`firmware_ring_drainer.cpp:428-430`). We extend this: when
`register_or_refresh_queue` discovers a queue, it also calls our
`firmware_ring_correlation::register_queue(queue)`, which:

1. Allocates a `QueueCorrState` with `slots.resize(queue->size)`,
   `ring_mask = queue->size - 1`, `last_observed_wdid` = current real
   `wdid` (so we don't try to retroactively label kernels that already
   started).
2. Inserts into `s_queue_corr_table[queue->id]`.
3. Inserts into `s_doorbell_map[queue->doorbell_signal.handle]` with a
   weak_ptr.

Kernels already in flight at attach time will appear on the firmware ring
without a matching slot entry → fall back to zero-corr, same as today.

### 4.8 Doorbell-to-queue lookup

PR 5219's `create_queue_state`
(`latestart/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp:261-280`)
stores `queue->doorbell_signal` and registers it in the `doorbell_map_t`.
`wrap_signal_store_screlease` (`:495-512`) calls
`lookup_queue_state_by_doorbell(sig)` which does an O(1) hash lookup
keyed by `sig.handle`. We use the identical pattern. Hash collisions
across the doorbell-signal-handle space are not possible because HSA
guarantees signal handles are unique per process.

---

## Section 5 — Modifications to `firmware_ring_drainer.cpp`

Anchor line numbers reference the existing file at
`source/lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.cpp`.

### 5.1 `process_dispatch_record` (line 241-262) — replace correlation construction

**Current code (lines 247-258):**
```cpp
constexpr uint32_t init_ref = 2;
auto* cid = context::correlation_tracing_service::construct(init_ref);
static thread_local context::correlation_id fallback_cid{};
if(!cid) cid = &fallback_cid;
```

**Replace with:**
```cpp
// Look up correlation captured by the launching-thread doorbell hook.
auto entry = firmware_ring_correlation::lookup_and_consume(
    st->queue->id, dispatch_idx_for_this_record);

context::correlation_id* cid = nullptr;
rocprofiler_thread_id_t   thr_id = 0;
uint64_t                  enqueue_ts = 0;
tracing::external_correlation_id_map_t ext_ids;

if (entry.has_value()) {
    cid       = entry->corr_id;
    thr_id    = entry->tid;
    enqueue_ts = entry->enqueue_ts;
    ext_ids   = std::move(entry->external_corr_ids);
} else {
    // Late-attach for an in-flight kernel — same fallback as today.
    static thread_local context::correlation_id fallback_cid{};
    cid    = &fallback_cid;
    thr_id = common::get_tid();
}
```

The `dispatch_idx` value comes from the
`mec_dispatch_record_16::dispatch_idx` field already plumbed through
`pending_starts[best_i]` in `drain_all` (line 380-410).

`process_dispatch_record`'s signature changes to take `dispatch_idx`:
```cpp
void process_dispatch_record(queue_ring_state_t* st,
                             uint32_t            dispatch_idx,
                             uint64_t            raw_start_ts,
                             uint64_t            raw_end_ts,
                             uint64_t            kernel_object);
```

And the call site in `drain_all` (line 417) updates accordingly.

### 5.2 `emit_kernel_dispatch_tracing` (lines 130-239) — consume side-table tid/ext_ids

Change signature to take `tid`, `external_corr_ids` from the side-table.
Inside, replace lines 202-205:
```cpp
auto thr_id = common::get_tid();                       // DELETE
auto& extern_corr = td.external_correlation_ids;       // DELETE
```
with the parameters passed in. Keep the existing `populate_contexts(...)`
call but **do not** also call `populate_external_correlation_ids` —
those are already snapshotted at enqueue time (this is the whole point
of capturing on the launching thread).

After the buffered_record_emplace call, drop the side-table ref counts:
```cpp
if (cid != /*fallback*/ nullptr) {
    cid->sub_kern_count();
    cid->sub_ref_count();
}
```

### 5.3 `register_or_refresh_queue` (lines 264-317) — also register correlation table

After the existing `g_queue_rings[queue->id] = ent;` insertion, add:
```cpp
firmware_ring_correlation::register_queue(queue);
```

This is idempotent (returns immediately if already registered).

### 5.4 New cleanup path — queue destruction

The current branch has no per-queue destruction hook (queues are torn
down only at process exit via `g_queue_rings.clear()` at line 489). We
don't need to add one to `firmware_ring_drainer.cpp` itself. Instead:

- In the firmware-ring-only case, queues aren't created via our
  `create_queue` interceptor (because `enable_queue_intercept()` returns
  false when only kernel-dispatch tracing is requested — see
  `queue_controller.cpp:533-544`).
- **Solution:** `register_or_refresh_queue`, on each pass, also reaps
  any registered queue whose `hsa_queue_t*` has been freed. Detection:
  try `hsa_amd_queue_get_info(queue, AGENT, ...)`; if it returns
  `HSA_STATUS_ERROR_INVALID_QUEUE`, the queue is gone, so call
  `unregister_queue` and decrement any in-flight slot ref counts. (Race:
  HSA may have reused the pointer for a new queue by the time we look —
  accept that risk for Phase 1; same risk this branch already accepts
  for `g_queue_rings`.)

### 5.5 `g_emitted_dispatch_idx` interaction

Unchanged behavior. The dedup set still suppresses duplicate emissions
across drain passes. The side table is independent — a successful
`lookup_and_consume` removes the entry the first time it's queried.

---

## Section 6 — Why No Completion Signal Is Allocated

### 6.1 What the standard path does

`WriteInterceptor` (`hsa/queue.cpp:298-708`) does three things that
*require* an SDK-allocated signal:

1. **Line 477-478** — `queue.create_signal(0,
   &kernel_packet.kernel_dispatch.completion_signal, true)` allocates a
   pooled HSA signal and writes it into the kernel dispatch packet's
   `completion_signal` field, *replacing* whatever the application put
   there. The application's original signal (if any) is honored later
   via an injected barrier-AND packet (`:592-599`).
2. **Line 644-655** — `signal_async_handler(last_pooled_signal,
   last_completion_signal, ...)` registers an HSA async-signal callback
   on that pooled signal so that, when the signal fires (kernel
   completes), the handler runs `dispatch_complete()`
   (`tracing.cpp:57`) which calls
   `hsa_amd_profiling_get_dispatch_time(signal)` and emits the COMPLETE
   record.
3. **Line 624** — for the `else` branch (no instrumentation packet
   appended), the SDK reuses the application's signal directly and force-
   stores it to 0; this still goes through
   `hsa_amd_profiling_get_dispatch_time` against that signal to read
   timestamps.

The signal exists for **two** reasons: (a) to know when the kernel
finished (to decide when to fire COMPLETE), and (b) to query timestamps
via `hsa_amd_profiling_get_dispatch_time(signal, ...)`.

### 6.2 What the firmware ring provides

The MEC firmware writes a 16-byte
`{ts_lo, ts_hi, record_type=2 (END), dispatch_idx}` to the queue's
host-visible profiling ring at EOP, exactly when the kernel completes.
The 8-byte timestamp **is** the GPU-clock end timestamp (same domain as
`hsa_amd_profiling_get_dispatch_time` after
`hsa_amd_profiling_convert_tick_to_system_domain`). So the firmware ring
delivers both pieces of information that the signal was used for.

### 6.3 Therefore

In the hybrid path:
- We **do not** call `WriteInterceptor`. The launching thread's wrapper
  just records correlation and chains through.
- We **do not** modify the kernel packet. The application's
  `completion_signal` (which may legitimately be `{0}`) is preserved,
  and HSA/the kernel sees exactly what the application wrote — no SDK
  barrier injection, no swapped signal.
- We **do not** call `hsa_signal_create`. The pool
  (`queue.signal_async_handler`, `queue.create_signal`) is simply not
  exercised.
- We **do not** call `hsa_amd_signal_async_handler`. The drainer's
  polling loop *is* our completion mechanism.
- The 1:1 packet-count invariant (PR 5219's chief correctness concern,
  `queue_intercept.cpp:232-237`) is trivially satisfied because there is
  no packet rewriting at all.

This is the core simplification over PR 5219: PR 5219's doorbell hook
still routes through `queue->invoke_write_interceptor(...)`
(`queue_intercept.cpp:224`), which calls into the same `WriteInterceptor`
and therefore still allocates signals. The hybrid bypasses
`invoke_write_interceptor` entirely.

---

## Section 7 — `hsa/queue.cpp` — Does `WriteInterceptor` Need a No-Signal Mode?

**Answer: No.** We do not invoke `WriteInterceptor` at all in the hybrid
path. The hybrid path has no caller of `WriteInterceptor`, so there is
nothing to gate.

### 7.1 Justification

`WriteInterceptor` is invoked from exactly one place today:
`Queue::invoke_write_interceptor` is the data callback registered with
`hsa_amd_queue_intercept_create`. Our hybrid never invokes
`hsa_amd_queue_intercept_create` for these queues (because
`enable_queue_intercept()` returns false for kernel-tracing-only contexts
when the firmware ring is available — see
`queue_controller.cpp:533-544`). Therefore `WriteInterceptor` is never
reached on these queues.

### 7.2 What about contexts that need both kernel tracing AND counters?

A context that requests both kernel-dispatch tracing and counter
collection (`dispatch_counter_collection != nullptr`) will:
- Force `enable_queue_intercept()` to return `true` (line 537).
- Therefore engage `hsa_amd_queue_intercept_create` and route through
  `WriteInterceptor`.
- And `WriteInterceptor` will allocate signals as it does today.

In other words: if the user wants counters, they pay the signal cost —
the firmware-ring fast path is reserved for kernel-tracing-only
contexts. This matches the existing design intent in this branch.

The hybrid hook **also** doesn't run on those queues: when
`WriteInterceptor` is in the picture, our doorbell wrapper will still see
the doorbell stores (because it wraps the core API table, not the
queue-intercept callback). To prevent double-counting, in Phase 1 the
simpler form is: `firmware_ring_correlation::register_queue(q)` is *only*
called from `register_or_refresh_queue` in the firmware-ring drainer,
which itself is only started for kernel-tracing-only contexts. If
counters are also requested, the drainer never starts, so registration
never happens, so the doorbell hook (even if installed) finds no entry
in the doorbell map and falls through to the chained call. Clean
separation by construction.

---

## Section 8 — `enable_queue_intercept` Gate Logic

This branch already added the firmware-ring carve-out
(`hsa/queue_controller.cpp:530-540`):

```cpp
const bool has_fw_ring = firmware_dispatch_ring_available();
const bool need_intercept_for_dispatch_tracing =
    has_kernel_tracing && !has_fw_ring;

if(itr->dispatch_counter_collection || itr->pc_sampler ||
   need_intercept_for_dispatch_tracing || has_scratch_reporting ||
   itr->device_counter_collection || itr->device_thread_trace ||
   itr->dispatch_thread_trace)
    return true;
```

**No change needed here for the hybrid.** When the hybrid is engaged:
- `has_fw_ring` is true → `need_intercept_for_dispatch_tracing` is false.
- If no other condition (counters, PCS, scratch, ATT) is true,
  `enable_queue_intercept()` returns false.
- `hsa_amd_queue_intercept_create` is bypassed; queues are created via
  the unmodified `hsa_queue_create`. The application's queue is a real
  HSA queue, not an intercept queue. **Crucially, the application's
  writes to its own ring buffer go directly to the GPU AQL ring, exactly
  as the GPU expects them.**
- Our doorbell hook is installed in the **core API table** (not
  queue-intercept callback), so it sees every doorbell store on every
  queue.

### 8.1 Where we install the doorbell hook

Add to `registration.cpp:1240-1259`, alongside (or instead of) the
existing `queue_intercept::install_intercept` block:

```cpp
{
    auto inline_intercept = common::get_env("ROCPROFILER_INLINE_INTERCEPT", true);
    auto firmware_ring   = hsa::firmware_dispatch_ring_available();
    auto counter_att = context::get_registered_contexts([](const auto* c){
        return c->dispatch_counter_collection || c->dispatch_thread_trace;
    });

    if (firmware_ring && counter_att.empty()) {
        // hybrid mode — install lightweight doorbell-only correlation hook
        hsa::firmware_ring_correlation::install_correlation_hook(*hsa_api_table->core_);
    } else if (inline_intercept && counter_att.empty()) {
        // PR 5219 path (if it survives upstream review separately)
        hsa::queue_intercept::install_intercept(*hsa_api_table->core_);
    }
    // else: standard path with hsa_amd_queue_intercept_create (legacy)
}
```

The two intercept paths
(`firmware_ring_correlation::install_correlation_hook` and
`queue_intercept::install_intercept`) are mutually exclusive — both would
wrap `hsa_signal_store_screlease`. We pick one based on
`firmware_dispatch_ring_available()`.

---

## Section 9 — Lifecycle / Shutdown

### 9.1 Initialization order

```
1. dispatch_ring_buffer_resolve_apis()
   (this branch already does this in queue_controller_init)
2. queue_controller::init                                   (sets up _supported_agents)
3. hsa::firmware_ring_correlation::install_correlation_hook (NEW; replaces the
                                                              install_intercept block above)
4. (If applicable) start_firmware_dispatch_ring_drainer()   (this branch already does this)
5. ... rest of registration ...
```

Critical ordering: install the doorbell hook **before** the drainer
starts. Otherwise an early kernel could complete (END record) before any
START's correlation is captured.

### 9.2 Shutdown sequence

```
1. Application calls rocprofiler_stop_context (or process exits → atexit).

2. Set fini_status > 0 — every wrapper checks this and bypasses
   immediately (matches PR 5219's bypass guard).

3. firmware_ring_correlation::shutdown_correlation_hook()
   - s_installed.store(false)
   - Walk every queue's slots:
     for each slot with a non-null corr_id and non-zero gen:
        corr_id->sub_kern_count();
        corr_id->sub_ref_count();
     Clear slots.
   - Clear doorbell map and queue table.

4. stop_firmware_dispatch_ring_drainer()
   - 10ms grace sleep (existing behavior; KNOWN_ISSUES item 3 is a separate cleanup)
   - g_drainer_stop.store(true)
   - Final drain pass with last_processed_record_count reset to 0.
   - On the final drain, lookup_and_consume returns nullopt for everything
     (we just cleared the table) → records emit with fallback zero-corr.

5. The 10ms grace sleep is now strictly safer than today: any record the drainer
   collects during the grace will see an empty side-table and emit zero-corr,
   instead of returning a freshly-constructed correlation_id from a service that
   may already be tearing down (which today is what causes the fallback_cid to
   be needed).
```

**Order rationale:** We tear down the side table *before* the drainer's
final pass so the drainer sees a clean "no entry" condition (acceptable:
zero-corr) instead of a possibly-already-freed `correlation_id*`
(catastrophic: use-after-free). This eliminates the entire class of bugs
that motivated `fallback_cid`.

### 9.3 Eliminating `fallback_cid` (KNOWN_ISSUES item 1)

In the new design:
- During steady state, `lookup_and_consume` returns the captured entry →
  real correlation.
- During shutdown after `shutdown_correlation_hook`, the table is empty
  → `lookup_and_consume` returns nullopt → drainer falls back to
  zero-corr. **This is the only path that produces zero-corr**, and it
  produces it *intentionally* (the drainer documents it as "late drain,
  correlation already retired"). No more
  `correlation_tracing_service::construct(...)` returning nullptr is
  reachable.
- True late-attach (kernel that started before our hook installed) →
  also nullopt → also zero-corr. Same behavior.

So `fallback_cid` collapses from "shutdown race workaround" to
"intentional sentinel for unknown-correlation records" with a clean,
documentable definition.

---

## Section 10 — Edge Cases and Open Issues

### 10.1 HIP graphs (one doorbell, N packets)

`hipGraphLaunch` constructs an internal command stream and rings the
doorbell once with `val = wdid_old + N - 1`. Our `process_doorbell` walks
every slot in `[last_observed_wdid, val + 1)` and assigns each the
**same** `corr_id` (the launching thread's TLS top — the
`hipGraphLaunch` API correlation id). This is identical to how PR 5219
(`queue_intercept.cpp:201-225`) handles it: one batch, one correlation,
N dispatches.

Whether this is the "right" semantic is a tooling question, not a
hybrid-design question — we match PR 5219, which matches the standard
path's behavior when the application enqueues multiple packets in one
batch. Document in user-facing notes.

### 10.2 Multi-thread concurrent enqueue to the same queue

If two threads concurrently:
- Thread A: `add_write_index → write packet at slot D → ring doorbell with val=D`
- Thread B: `add_write_index → write packet at slot D+1 → ring doorbell with val=D+1`

HSA does not serialize this; the two doorbell stores can race. Our
`process_doorbell` takes `slot_publish_mu` per-queue, so the two stores
serialize on our mutex. The thread that wins the lock first observes its
`val` and updates `last_observed_wdid`; the second thread (with a
higher val) walks `[last_observed_wdid_after_first, val_second+1)`,
which includes its own slot. Correlations attach correctly.

**Risk:** If thread A wins the lock first but `val_A > val_B`
(out-of-order doorbells from the same producer never happens, but
cross-producer reorderings do), B's range becomes empty
(`val_B + 1 <= last_observed_wdid`). Slot D+1's packet was *already*
captured by A using A's TLS correlation, not B's. **This is wrong** —
slot D+1 was actually B's packet.

Mitigation: this matches PR 5219's exact same hazard. For Phase 1,
accept and document. A real fix requires either:
- Reading packet headers and back-walking which dispatch_id corresponds
  to which TLS-active thread (impossible without a per-packet thread
  cookie), or
- Having HIP push a per-packet thread cookie into a reserved AQL packet
  field that we cross-reference at drain time.

### 10.3 `hsa_signal_silent_store_*`

If HIP ever uses `hsa_signal_silent_store_relaxed` or
`hsa_signal_silent_store_screlease` to ring doorbells (some drivers do
for reduced PCIe MMIO), we miss them entirely. Investigation needed:
- Grep CLR/HIP source for doorbell store calls; verify which
  `hsa_signal_store_*` variant is used.
- If silent variants are used, add wrappers for them too. The existing
  `wrap_signal_store_*` design generalizes trivially.

PR 5219 does not wrap silent variants either — same shared risk.
Document.

### 10.4 Queue destroyed while dispatches are in-flight

Sequence:
- Application enqueues kernels K1, K2, K3 — slots populated.
- Application calls `hsa_queue_destroy(q)` *before* K1/K2/K3 finish.
- HSA tears down the queue; the firmware-ring buffer is freed.
- The drainer's next pass reads garbage from the (now-freed) ring →
  spurious records or skip.

Mitigation: `register_or_refresh_queue` already guards with
`hsa_amd_queue_get_info` (Section 5.4 reaping). For Phase 1, also wrap
`hsa_queue_destroy` in the core API table so we can call
`firmware_ring_correlation::unregister_queue(q)` synchronously before HSA
frees the queue. (This adds one more wrapped function, but it's
symmetric with the doorbell wrapper.)

In `unregister_queue(q)`:
- Erase from doorbell map.
- Walk slots; for each non-zero gen with valid corr_id,
  sub_kern_count/sub_ref_count; clear.
- Erase from queue table (shared_ptr keeps the QueueCorrState alive
  until any in-flight drainer pass releases its weak_ptr lock).

### 10.5 Late-attach for in-flight kernels

Acceptable. Kernels already running at attach time produce END records on
the firmware ring whose `dispatch_idx` corresponds to slots we never
wrote → `lookup_and_consume` returns nullopt → fallback zero-corr. The
dispatch record still flows out with valid timestamps and kernel_object;
tools can detect it as "pre-attach" by the zero internal_correlation_id.

### 10.6 Multi-XCC pairing heuristic (KNOWN_ISSUES item 2)

Unaffected by the hybrid; still wrong for overlapping concurrent kernels
on different XCCs. Phase 3 follow-up; needs an XCC/pipe id in the
firmware record. (See `KNOWN_ISSUES.md` item 2.)

### 10.7 `g_emitted_dispatch_idx` unbounded growth (KNOWN_ISSUES item 5)

Unaffected by the hybrid; still grows. Phase 3 follow-up: replace with a
per-queue last-seen-dispatch-idx watermark + small recently-seen ring
sized to `queue->size`.

### 10.8 Kernels enqueued by the runtime itself

Some HIP internal operations enqueue dispatch packets without a
corresponding HIP-API correlation id on the TLS stack. In that case
`context::get_latest_correlation_id()` returns nullptr and we fall
through to `correlation_tracing_service::construct(1)`. This creates a
fresh correlation id with no ancestor — same behavior as
`queue.cpp:384-389`. Acceptable.

### 10.9 Doorbell store with `val` smaller than expected

Some queue-implementation patterns store a sentinel value (e.g., `-1`)
into the doorbell to wake the kernel processor. We guard with
`if (new_wdid <= prev) return;` in `process_doorbell` — a doorbell that
doesn't advance the apparent wdid is a no-op. This may mask legitimate
retry/replay scenarios; document and revisit if encountered.

### 10.10 ROCprofiler-register attach mode

When the SDK is loaded via `rocprofiler-register` after HSA is already
initialized, the `attach_table` path is used
(`queue_controller.cpp:601-614`). Our doorbell hook also needs to be
installed in the attach path. Add a parallel install call:

```cpp
void queue_controller_init(RocAttachDispatchTable* attach_table)
{
    // ... existing code ...
    if (firmware_dispatch_ring_available() && /* no counter/ATT contexts */)
        firmware_ring_correlation::install_correlation_hook(/* attached core table */);
}
```

(Plumbing detail: the attach table exposes a `set_signal_store_*` shim —
adapt as needed.)

---

## Section 11 — Testing Strategy

### 11.1 Unit tests

Add `tests/unit/firmware_ring_correlation.cpp` with:

1. **Single-producer single-consumer round-trip.** Manually call
   `register_queue`, simulate doorbell store, verify slot is populated.
   Call `lookup_and_consume`, verify entry returned and slot cleared.

2. **Multi-producer concurrent insert into different slots.** N threads
   each "enqueue" to a different `dispatch_idx`. After all threads
   finish, verify all slots populated correctly with non-overlapping
   correlations.

3. **Reference-count balance.**
   - Construct a real `correlation_id` with initial refcount 1.
   - Simulate enqueue → expect `refcount == 2, kerncount == 1`.
   - Call `lookup_and_consume` → simulate completion path → expect
     `refcount == 1, kerncount == 0`.
   - Final API decrement → `refcount == 0` (retired).

4. **Late-attach simulation.** Call `lookup_and_consume(q, idx)` for an
   idx that was never written → expect nullopt.

5. **Shutdown leaves no dangling refs.** Populate slots; call
   `shutdown_correlation_hook`; verify all `corr_id`s' `kern_count` is 0
   and `ref_count` is back to its pre-enqueue value.

### 11.2 Integration tests (firmware-ring-capable target)

Use the existing `samples/api_buffered_tracing` and
`samples/api_callback_tracing` as bases.

1. **Correlation ID match between API and dispatch records.**
   - Run `samples/api_buffered_tracing` with the hybrid engaged.
   - Filter for `ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API` records
     (operation = `HIP_RUNTIME_API_ID_hipLaunchKernel`) and
     `ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH` records.
   - Group by `correlation_id.internal`. Every kernel-dispatch record's
     internal correlation should match the corresponding
     `hipLaunchKernel`'s.
   - Today's branch has zero such matches; hybrid should have 100%.

2. **External correlation IDs survive.**
   - Sample pushes external corr id via
     `rocprofiler_push_external_correlation_id` before
     `hipLaunchKernel`.
   - Verify the dispatch record carries the pushed external id (not
     zero).

3. **ENQUEUE callbacks fire** (Phase 2 only).
   - Register a callback context for
     `ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH` op
     `ROCPROFILER_KERNEL_DISPATCH_ENQUEUE`.
   - Run kernels; verify ENTER+EXIT phases fire with the launching tid.

4. **No `hsa_signal_create` calls under `ltrace`.**
   - `ltrace -e hsa_signal_create+hsa_signal_create_ex@HSA -o trace.log <hip_app>`.
   - With kernel-tracing-only context engaged via hybrid, expect zero
     matches in `trace.log` attributable to rocprofiler (some baseline
     calls from HSA/HIP runtime are unavoidable; identify with a
     no-rocprofiler control run and diff).

5. **No `hsa_amd_queue_intercept_create` calls.**
   - Same `ltrace` approach for `hsa_amd_queue_intercept_create`. Expect
     zero rocprofiler-attributable calls in the kernel-tracing-only
     case.

6. **Diff against standard-interception baseline.**
   - Run identical workload with the hybrid disabled (forces standard
     path).
   - Compare CSV outputs row-by-row: kernel ids should match,
     correlation ids should match (modulo enqueue ordering), tids should
     match, timestamps should be within ~1us.

7. **HIP graph correlation.**
   - Sample that captures and replays a graph with multiple kernels.
   - Verify all kernels in one `hipGraphLaunch` share the same internal
     correlation id (matches PR 5219 and standard-path behavior).

8. **Stress: concurrent multi-thread enqueue.**
   - Launch N threads each doing M `hipLaunchKernel` calls to the same
     stream.
   - Verify the union of dispatch records' internal correlation ids
     matches the union of API records'. (Won't catch the per-slot
     mis-attribution from 10.2, but will catch gross drops.)

### 11.3 Sanitizer runs

- ASAN: detects use-after-free if `unregister_queue` races with
  `process_doorbell`.
- TSAN: detects data races on slot fields. Should be clean given the
  gen-counter publish/acquire pattern.
- LSAN: detects leaked correlation_ids if shutdown ref-count walking is
  wrong.

---

## Section 12 — Implementation Phasing

### Phase 1 (~200 lines) — Minimal hybrid

Goal: kernel dispatch records carry the correct internal correlation,
ancestor correlation, external correlation, and launching thread id.
ENQUEUE callback semantics deferred.

Touched files:
- New: `lib/rocprofiler-sdk/hsa/firmware_ring_correlation.{cpp,hpp}`
  (~250 lines combined, of which ~200 are the core path; the rest is
  plumbing/headers).
- Modified:
  `lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.cpp`
  (~60 lines added, ~20 removed).
- Modified: `lib/rocprofiler-sdk/registration.cpp` (~10 lines).
- Modified: `lib/rocprofiler-sdk/hsa/queue_controller.cpp` (~5 lines for
  install/shutdown call sites).
- Modified: CMakeLists for new sources.

Out of scope for Phase 1:
- ENQUEUE callbacks (still missing; KNOWN_ISSUES item 10 stays open).
- Workgroup/grid/segment sizes in dispatch_info (still zero;
  KNOWN_ISSUES item 8).
- Any KNOWN_ISSUES items 2-9 not yet addressed.

### Phase 2 (~50 lines) — ENQUEUE callbacks

Add `fire_enqueue_callbacks` (Section 4.6) inside `process_doorbell`.
Closes KNOWN_ISSUES item 10.

### Phase 3 (~100 lines) — Remaining KNOWN_ISSUES cleanup

- Item 8: extend `lookup_kernel_object` to return wgs/grid/segments;
  populate dispatch_info correctly.
- Item 5: replace `g_emitted_dispatch_idx` with per-queue ring of
  recently-seen indices.
- Item 6: track first-discovery-only enable of
  `hsa_amd_profiling_set_profiler_enabled`; disable on shutdown.
- Item 2 left for separate firmware-record-format change (out of scope
  for this plan).

---

## Section 13 — SDK integration with the HSA-resident drainer (the path actually shipped)

> This section was added in the 2026-04-30 status update and describes
> an alternative implementation path that has since been built on
> `users/bewelton/lttng-kernel-ts` (PR #5519). The SDK-side hybrid
> described in Sections 1–12 above remains a valid reference design;
> Sections 14 (Risk Register) and Appendices A–B carry over unchanged.

### 13.1 What the HSA-resident drainer ships

Branch: `users/bewelton/lttng-kernel-ts`. PR: `ROCm/rocm-systems#5519`
(draft). HEAD: `e6abbfc7fa`. Spec:
`~/ai/specs/2026-04-27-hsa-lttng-kernel-dispatch-tracing-design.md`
(8-round adversarial debate, 27 claims accepted).

* MEC writes 16-byte records `{ts_lo, ts_hi, record_type, dispatch_idx}`
  into the per-queue ring exactly as today (no firmware change vs the
  current MI350 build).
* HSA's `core/runtime/dispatch_log.cpp` owns the drainer. One worker
  thread per active queue (4 commits at HEAD `e6abbfc7fa`, including
  two stage-1 + stage-2 debate rounds for thread spawn failure handling
  + destructor self-join hardening). Sentinel-scan design: each slot is
  read with `__atomic_load_n(record_type, ACQUIRE)`, emitted, then
  zeroed; a `record_type==0` slot is the empty marker. No FW write
  pointer is required (the substrate publishes none).
* HSA does not interpret `record_type` — it emits one
  `rocm_hsa:kernel_dispatch_record(queue_id, dispatch_idx, gpu_ts,
  record_type, corr_id, parent_corr_id)` LTTng tracepoint per
  non-zero record. Consumers join records on
  `(queue_id, dispatch_idx)` and choose their own start/end polarity
  for the FW version they target.
* Hooks into `core/runtime/hsa.cpp::on_queue_create / on_queue_destroy`
  drive the per-queue enable/disable lifecycle (spec §4).
* Buffer registration uses today's KFD interface — extended
  `AMDKFD_IOC_UPDATE_QUEUE` with `dispatch_record_buffer_addr` +
  `dispatch_record_buffer_size` trailing fields, KFD MINOR=22.
  See `KFD_DISPATCH_LOG_DESIGN.md` §0 for the current ABI and §2 for
  the proposed migration to the profiler ioctl (MINOR=23).
* Measured at 169.3% combined-record capture rate (~85% per
  record_type) on graphbench, 5.17M HIP submits per run. Variance
  158.7%–184.0% across reps. Two perf experiments tried this session
  (256K ring, batched-translate-with-lock-once) both regressed and
  were reverted; 64K ring + per-queue threads is the local optimum.

### 13.2 What this means for rocprofiler-sdk

The SDK no longer needs to:

* Run its own polling drainer thread (`firmware_ring_drainer.cpp` —
  the entire file becomes obsolete).
* Maintain a per-queue side table for correlation capture (Sections
  4.1–4.3 of this plan — the entire `firmware_ring_correlation.cpp`
  becomes obsolete).
* Wrap `hsa_signal_store_*` for doorbell capture (Section 4.2 — the
  entire wrapper becomes obsolete).
* Allocate completion signals (already true in the SDK-side hybrid;
  remains true here).
* Depend on PR 5219 (`users/bewelton/no-interecept-queue`) for the
  doorbell-wrap infrastructure (Section 12 critical-dependency goes
  away).

What the SDK still needs to do:

* Pair the FW dispatch records with the HIP/HSA API events that
  enqueued them (correlation, parent attribution, external-correlation
  IDs, launching tid).
* Synthesize `ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH` /
  `ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH` records in the
  shapes existing tools expect.
* Drive the `KERNEL_DISPATCH_ENQUEUE` ENTER/EXIT phase callbacks
  (the existing HIP wrappers in PR #5475 already fire these via
  `rocm_hip:hip_aql_kernel_dispatch_submit` — see §13.3).

There are two integration paths to choose from. Both are viable.

### 13.3 SDK integration path A — consume the LTTng CTF stream (recommended)

The HSA drainer emits `rocm_hsa:kernel_dispatch_record` events into the
same CTF channel that already carries HIP API events
(`rocm_hip:hip_api_enter`, `rocm_hip:hip_aql_kernel_dispatch_submit`,
the curated `<api>_args` events, etc. — see PR #5475 and
`TRACING_DELIVERY_RESEARCH.md`). One CTF stream, one consumer, one
join.

**Mechanism.** rocprofiler-sdk subscribes via the LTTng-live protocol
(`libbabeltrace2` + `liblttng-ctl`) for live consumption, or reads the
on-disk CTF directory for offline. It walks events in CTF
event-header timestamp order and joins:

* `rocm_hip:hip_aql_kernel_dispatch_submit(queue_id, write_idx, …)`
  — fired on the launching thread. Carries the correlation_id from
  the HIP API ENTER's TLS stack, the launching tid (from the CTF
  channel context's `vtid`), the external-correlation snapshot, and
  the AQL packet's workgroup/grid/private/group sizes. Schema v3
  already specifies `(queue_id, write_idx)` as the cross-runtime
  join key.
* `rocm_hsa:kernel_dispatch_record(queue_id, dispatch_idx, gpu_ts,
  record_type, …)` — fired by the HSA per-queue drainer thread.
  Carries the GPU-domain timestamp (already translated to system
  clock by the drainer via `GpuAgent::TranslateTime`).

The join is `(queue_id, write_idx) == (queue_id, dispatch_idx)`
(both are `read_dispatch_id[31:0]` masked the same way). The
`hip_aql_kernel_dispatch_submit` event provides every datum the
existing `KERNEL_DISPATCH_COMPLETE` record carries except the GPU
timestamps; the `kernel_dispatch_record` events provide those.

**Pros.**
* Zero new IPC mechanisms: the LTTng-UST transport is already vendored
  (LTTng-UST 2.13.7 + URCU 0.14.0 submodules under
  `projects/{clr,rocr-runtime}/external/`) and validated.
* Single CTF stream means tools other than rocprofiler-sdk
  (babeltrace2 directly, custom CTF readers, perfetto importers) get
  the same records for free.
* Works for late-attach: a tool that creates an LTTng session after
  the workload starts captures everything from session-start onward
  with no producer cooperation.
* Multi-process / MPI safe: the CTF channel context's `(vpid, vtid)`
  is unambiguous across ranks (schema v3 design choice — see
  `HIGH_LEVEL_DESIGN_SUMMARY.md`).
* Works even if rocprofiler-sdk is not loaded — the stream is
  consumable by anyone.

**Cons.**
* The SDK's `KERNEL_DISPATCH_ENQUEUE` callbacks have to fire from a
  CTF event delivery, not from the launching thread directly. For
  buffer-tracing consumers this is fine. For callback-tracing
  consumers that expect their callback to run on the launching
  thread, this is a behavior change — the callback fires on
  whichever thread runs the LTTng-live consumer.
* Adds a libbabeltrace2 dependency to rocprofiler-sdk.
* Requires `lttng-sessiond` to be running for live consumption (or
  pre-recorded CTF for offline).

### 13.4 SDK integration path B — bypass HSA, query KFD directly via the same ioctl

The SDK keeps its own polling drainer (the existing
`firmware_ring_drainer.cpp` in this branch), but registers the buffer
via the same KFD ioctl HSA uses, and reads the same in-memory ring
HSA's drainer reads. HSA's LTTng emission becomes redundant from the
SDK's perspective (the SDK ignores those events).

**Mechanism.** SDK calls `AMDKFD_IOC_UPDATE_QUEUE` with the extended
fields (today: MINOR=22; after `KFD_DISPATCH_LOG_DESIGN.md` lands:
MINOR=23 + `KFD_IOC_PROFILER_DISPATCH_LOG`). SDK does its own
sentinel-scan of the ring, pairs records with the doorbell-wrapper
side table from Sections 1–12 of this document, fires
`KERNEL_DISPATCH_COMPLETE` on the SDK's drainer thread.

**Pros.**
* `KERNEL_DISPATCH_ENQUEUE` callbacks fire on the launching thread
  (preserves current SDK callback semantics).
* No libbabeltrace2 dependency.
* No `lttng-sessiond` dependency at runtime.

**Cons.**
* Two drainer threads now read the same ring (HSA's per-queue worker
  AND the SDK's polling thread). They must coordinate ownership of
  the slot-zero memset, or one of them will repeatedly observe slots
  as "empty" because the other already consumed them. Easiest
  resolution: HSA stops draining when the SDK takes ownership (a new
  HSA API: "release the drainer; someone else will read this ring").
  This re-introduces SDK↔HSA coupling that the LTTng path eliminates.
* Re-introduces all of Sections 4.1–4.3's per-queue side-table
  machinery and the PR 5219 doorbell-wrapper dependency.
* Other tools (babeltrace2 users, perfetto importers, crash dumpers)
  cannot consume the FW records — only rocprofiler-sdk can.
* On `KFD_DISPATCH_LOG_DESIGN.md`'s migration to MINOR=23, the SDK
  has to know whether HSA already owns the buffer (HSA's
  `SetProfiling` registers it via the same ioctl) and either
  coordinate with HSA or fail to register.

### 13.5 Recommendation

Path A (consume LTTng) is recommended for the following reasons:

1. It removes the SDK↔HSA ring-buffer coordination problem entirely.
2. It collapses the SDK's drainer thread, side-table machinery, and
   PR-5219 dependency. ~470 LOC of new SDK code is replaced by
   ~200 LOC of CTF event-pair-and-emit logic.
3. It makes the FW records consumable by tools other than
   rocprofiler-sdk for free.
4. The LTTng-UST infrastructure is already shipping in PRs #5475 +
   #5513 — no new transport to build.

Path B remains viable if downstream tooling has hard constraints
against an `lttng-sessiond` runtime dependency. In that case, the
SDK-side hybrid in Sections 1–12 of this document is the
implementation reference, with two modifications: (a) the SDK
registers the ring via the KFD ioctl directly (today: extended
UPDATE_QUEUE; after the `KFD_DISPATCH_LOG_DESIGN.md` migration:
profiler ioctl), and (b) HSA's per-queue drainer must be opt-out so
the SDK can take ownership.

---

## Section 14 — Risk Register

| # | Risk                                                                                       | Likelihood   | Impact   | Mitigation                                                                                  |
|---|--------------------------------------------------------------------------------------------|--------------|----------|---------------------------------------------------------------------------------------------|
| 1 | HIP private fast paths bypass `hsa_signal_store_*`                                         | Low-medium   | High     | Audit CLR for any direct doorbell write that doesn't go through the API table. If found, fix HIP or fall back to standard path on those queues. |
| 2 | Multi-thread enqueue to same queue mis-attributes correlation (Section 10.2)               | Medium       | Medium   | Document. Phase 4 work to add per-packet thread cookie via reserved AQL packet field.       |
| 3 | Firmware/SDK version skew — record size != 16 bytes                                        | Low          | High     | Already handled by `infer_record_size` (`firmware_ring_drainer.cpp:112-121`); skip queues with unknown record size. |
| 4 | `dispatch_idx` 32-bit wrap on long-running processes (KNOWN_ISSUES item 5)                 | Low          | Medium   | Phase 3 fix.                                                                                |
| 5 | Performance regression vs PR 5219                                                          | Very low     | Low      | Hybrid does strictly less work per dispatch (no packet rewriting, no signal alloc, no async-handler register). Expect strict win. |
| 6 | Race between `unregister_queue` and an in-flight `process_doorbell`                        | Low          | High     | shared_ptr lifetime + slot_publish_mu serialize. Verify under TSAN.                         |
| 7 | `hsa_signal_silent_store_*` used by HIP for doorbell                                       | Unknown      | High     | Section 10.3 — investigate first; add wrappers if needed.                                   |
| 8 | Doorbell hook installed on a queue that *also* gets `hsa_amd_queue_intercept_create`       | Low          | Medium   | Section 7.2 separation-by-construction: drainer doesn't run for counter/ATT contexts, so registration never happens. Belt-and-suspenders: per-queue opt-out flag. |
| 9 | `correlation_id` retired (refcount==0) before drainer sees END record                      | Low          | High     | Per-dispatch `add_ref_count` on launching thread holds the id alive. Matches `queue.cpp:450`. |
| 10| ROCprofiler-register attach mode not covered                                               | Medium       | Medium   | Section 10.10 — parallel install call in attach path.                                       |
| 11| Application's queue is not a real HSA queue (some custom HSA tools)                        | Very low     | Low      | Doorbell-map lookup returns null → bypass; correctness preserved, dispatch tracing degrades to off for that queue. |
| 12| Wrap of `hsa_queue_destroy` adds a new failure surface                                     | Low          | Low      | Standard chain-through pattern. Tested via existing destroy-during-stress tests.            |

---

## Appendix A — Function Signatures Summary (`firmware_ring_correlation.hpp`)

```cpp
namespace rocprofiler::hsa::firmware_ring_correlation {

struct CorrEntry {
    context::correlation_id*               corr_id    = nullptr;
    rocprofiler_thread_id_t                tid        = 0;
    uint64_t                               enqueue_ts = 0;
    tracing::external_correlation_id_map_t external_corr_ids;
    uint64_t                               seq        = 0;
};

// Install in core API table — wraps hsa_signal_store_relaxed and _screlease.
void install_correlation_hook(::CoreApiTable& core_table);

// Reverse install_correlation_hook + drain side table.
void shutdown_correlation_hook();

// Register a queue for correlation tracking. Idempotent. Called from
// firmware_ring_drainer::register_or_refresh_queue.
void register_queue(hsa_queue_t* queue);

// Unregister and decrement any in-flight correlation refcounts.
// Called from a wrapped hsa_queue_destroy and from shutdown.
void unregister_queue(uint64_t queue_id);

// Drainer-side lookup: pop the entry for (queue_id, dispatch_idx) and return it.
// Returns nullopt if no entry was captured (late-attach or shutdown race).
std::optional<CorrEntry> lookup_and_consume(uint64_t queue_id, uint32_t dispatch_idx);

bool is_installed();

}  // namespace rocprofiler::hsa::firmware_ring_correlation
```

---

## Appendix B — Quick Reference: file:line Anchors

For the engineer implementing Phase 1.

### Standard interception path (what we are not doing)

| Concern                                | File:line                                                                                                    |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Standard-path signal allocation        | `lib/rocprofiler-sdk/hsa/queue.cpp:477`                                                                      |
| Standard-path correlation +ref +kern   | `lib/rocprofiler-sdk/hsa/queue.cpp:450-451`                                                                  |
| Standard-path async signal handler     | `lib/rocprofiler-sdk/hsa/queue.cpp:653-655`                                                                  |
| Standard-path COMPLETE emission        | `lib/rocprofiler-sdk/kernel_dispatch/tracing.cpp:57-117`                                                     |

### PR 5219 inline intercept (the model to borrow from, not duplicate)

PR 5219 lives on the `latestart` branch / worktree; paths are within that
worktree.

| Concern                                | File:line                                                                                                    |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------|
| PR 5219 doorbell wrapper               | `lib/rocprofiler-sdk/hsa/queue_intercept.cpp:495-512`                                                        |
| PR 5219 process_doorbell_impl          | `lib/rocprofiler-sdk/hsa/queue_intercept.cpp:182-258`                                                        |
| PR 5219 doorbell-map lookup            | `lib/rocprofiler-sdk/hsa/queue_intercept.cpp:67-79`                                                          |
| PR 5219 install gate                   | `lib/rocprofiler-sdk/registration.cpp:1240-1259`                                                             |

### This branch (what we modify)

| Concern                                | File:line                                                                                                    |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------|
| drainer entry                          | `lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.cpp:432-461`                                      |
| drainer correlation gap                | `lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.cpp:241-262` (the `fallback_cid` block)           |
| emit fn                                | `lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.cpp:130-239`                                      |
| register_or_refresh_queue              | `lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.cpp:264-317`                                      |
| enable_queue_intercept                 | `lib/rocprofiler-sdk/hsa/queue_controller.cpp:510-545`                                                       |
| init/fini drainer plumbing             | `lib/rocprofiler-sdk/hsa/queue_controller.cpp:548-598`                                                       |

---

**End of plan.** This document is sufficient for an engineer with
rocprofiler-sdk familiarity to implement Phase 1 directly. Phases 2 and
3 reuse the same data structures and require no architectural changes.

See `KNOWN_ISSUES.md` for the issue catalog this plan addresses.
