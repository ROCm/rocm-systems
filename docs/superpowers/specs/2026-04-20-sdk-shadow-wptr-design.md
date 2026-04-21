# SDK-Level Shadow Write Pointer Design

**Date:** 2026-04-20
**Author:** bewelton + Claude
**Status:** Draft

---

## Problem Statement

The current queue interception in ROCR-Runtime uses `InterceptQueue`, which swaps `base_address`, `doorbell_signal`, and `core_queue` fields at runtime. Applications and runtimes (particularly HIP) cache these pointers at queue creation time, so any mid-flight swap breaks them.

A shadow write pointer prototype was implemented in the HSA runtime (`users/bewelton/shadow_ptr` branch), but hit fundamental issues: the shared ring buffer between app and interceptor causes slot collisions, requiring K-reservation (8x ring inflation), gap markers, and complex CAS-loop logic. 66 of 691 tests still fail. The architecture is inherently fragile because two independent pointer spaces (`write_dispatch_id` and `shadow_wptr_`) index the same physical ring.

See `/home/bewelton/ai/task_info/SHADOW_WPTR_SESSION.md` for the full HSA implementation report.

## Solution: SDK-Level Write-Index Interposition

Move the shadow write pointer logic out of the HSA runtime and into rocprofiler-sdk. The SDK already wraps all HSA API table entries (including all 10 write-index variants and signal stores) via its `copy_table`/`update_table` mechanism in `hsa.cpp`. Instead of relying on `hsa_amd_queue_intercept_create` + `WriteInterceptor`, the SDK directly intercepts write-index operations and doorbell signal stores to control packet flow.

### Core Mechanism

1. **Deferred WDID:** The SDK intercepts `hsa_queue_add_write_index` and does NOT forward the call to HSA. Instead, it maintains a virtual write pointer internally. The real `write_dispatch_id` in `amd_queue_t` only advances when the SDK is ready to release packets to the GPU.

2. **Write-ahead in real ring:** The app writes packets at virtual slot indices, which map to positions in the real GPU ring buffer. These positions are ahead of where `write_dispatch_id` points, so the GPU cannot see them yet.

3. **Copy to submit region at doorbell time:** When the app rings the doorbell, the SDK intercepts the signal store, scans new packets from the write-ahead zone, copies each app packet to the real WDID position interleaved with instrumentation packets, advances real WDID, and rings the real hardware doorbell.

4. **Dynamic K-factor:** The number of instrumentation slots per app packet depends on active services: 0 for trace-only, 7 for counter collection.

### Why Not in HSA?

The HSA approach has the following problems that the SDK approach avoids:

| Problem | HSA Approach | SDK Approach |
|---------|-------------|-------------|
| Shared ring collisions | App and interceptor write to same ring with independent pointers → slot stomps | SDK controls all writes; app writes to write-ahead zone, SDK copies to submit zone |
| K-reservation CAS race | CAS-loop in `InterceptReserveSlots` can lose to concurrent writers | SDK's virtual_wptr is the sole authority for app slot allocation |
| Gap marker fragility | GPU hits gap markers mid-race → must be benign noops | GPU never sees uninstrumented slots; real WDID only advances after SDK has written everything |
| WDID semantics change | `write_dispatch_id` grows K× faster than app kernel count → breaks HIP yield loop | App sees normal virtual indices; real WDID is internal |
| Ring size coupling | Ring inflated at queue creation for worst-case K | Ring inflated at creation but K is dynamic per-service, can be 0 for trace-only |

---

## Architecture: Flat Functions + QueueState Map

### QueueState

Per-queue state managed by the SDK:

```cpp
struct QueueState {
    // Ring buffer (from hsa_queue_t at creation time)
    void*        ring_buf;       // hsa_queue_t::base_address
    uint32_t     ring_size;      // number of 64-byte slots (power of 2)
    uint32_t     ring_mask;      // ring_size - 1

    // Pointer management
    std::atomic<uint64_t> virtual_wptr{0};   // app-visible write pointer (write-ahead)
    volatile uint64_t*    real_wdid;          // &amd_queue_t.write_dispatch_id
    volatile uint64_t*    real_rdid;          // &amd_queue_t.read_dispatch_id
    uint64_t              next_scan_pos{0};   // next position to scan in write-ahead zone
    uint64_t              next_submit_pos{0}; // next position for SDK to write at real WDID

    // Doorbell
    hsa_signal_t          doorbell_signal;    // queue's doorbell signal
    signal_store_fn_t     original_signal_store; // unwrapped hsa_signal_store_relaxed

    // Instrumentation
    uint64_t              k_factor{0};       // 0=trace-only, 7=counter-collection

    // Metadata queue pairing (nullable)
    hsa_queue_t*          metadata_queue;     // paired metadata HSA queue
    QueueState*           metadata_state;     // paired metadata QueueState

    // Interceptor callbacks (counter collection, tracing, etc.)
    std::vector<intercept_callback_t> interceptors;

    // Synchronization
    std::mutex            gate_lock;          // serializes doorbell processing
};
```

### Registry

```cpp
using queue_registry_t = synchronized<std::unordered_map<hsa_queue_t*, std::unique_ptr<QueueState>>>;
queue_registry_t& get_queue_registry();

// Fast lookup by queue pointer
QueueState* lookup_queue_state(const hsa_queue_t* queue);

// Fast lookup by doorbell signal handle (for signal store interception)
QueueState* lookup_queue_state_by_doorbell(hsa_signal_t signal);
```

The doorbell-to-queue lookup is needed because `hsa_signal_store_relaxed` receives a signal handle, not a queue pointer. A secondary map (`signal_handle → QueueState*`) enables O(1) lookup.

---

## Write-Index Interposition

All 10 write-index variants are intercepted. The SDK manages `virtual_wptr` internally and does NOT forward to the real HSA functions.

### AddWriteIndex (4 variants: relaxed, acq_rel, scacquire, screlease)

```cpp
uint64_t intercepted_add_write_index(const hsa_queue_t* queue, uint64_t value) {
    auto* state = lookup_queue_state(queue);
    if (!state) return original_add_write_index(queue, value);

    // Advance virtual wptr — real WDID untouched
    return state->virtual_wptr.fetch_add(value, memory_order);
}
```

The app gets back a virtual slot index and writes its packet at `ring_buf[returned_index & mask]`. This is a real position in the GPU ring buffer, but ahead of `write_dispatch_id`, so the GPU cannot see it.

### StoreWriteIndex (2 variants)

```cpp
void intercepted_store_write_index(const hsa_queue_t* queue, uint64_t value) {
    auto* state = lookup_queue_state(queue);
    if (!state) { original_store_write_index(queue, value); return; }
    state->virtual_wptr.store(value, memory_order);
}
```

### CasWriteIndex (4 variants)

```cpp
uint64_t intercepted_cas_write_index(const hsa_queue_t* queue,
                                      uint64_t expected, uint64_t value) {
    auto* state = lookup_queue_state(queue);
    if (!state) return original_cas_write_index(queue, expected, value);

    uint64_t prev = expected;
    state->virtual_wptr.compare_exchange_strong(prev, value, memory_order);
    return prev;
}
```

### LoadWriteIndex (2 variants)

```cpp
uint64_t intercepted_load_write_index(const hsa_queue_t* queue) {
    auto* state = lookup_queue_state(queue);
    if (!state) return original_load_write_index(queue);
    return state->virtual_wptr.load(memory_order);
}
```

### LoadReadIndex (2 variants)

Forwarded to the original HSA function unchanged. The GPU's `read_dispatch_id` is ground truth.

```cpp
uint64_t intercepted_load_read_index(const hsa_queue_t* queue) {
    return original_load_read_index(queue);
}
```

---

## Doorbell Interception

The doorbell handler is where the SDK does the core work. When the app rings the doorbell (via `hsa_signal_store_relaxed` on the queue's doorbell signal), the SDK:

1. Scans new packets in the write-ahead zone
2. Copies each app packet to the submit position at `next_submit_pos`
3. Writes instrumentation packets (if K > 0) after each app packet
4. Writes metadata entries (if metadata queue is paired)
5. Advances real `write_dispatch_id`
6. Rings the real hardware doorbell

```cpp
void intercepted_signal_store(hsa_signal_t signal, hsa_signal_value_t value) {
    auto* state = lookup_queue_state_by_doorbell(signal);
    if (!state) {
        original_signal_store(signal, value);
        return;
    }

    std::lock_guard<std::mutex> lock(state->gate_lock);

    uint64_t scan_end = state->virtual_wptr.load(std::memory_order_acquire);
    uint64_t scan_pos = state->next_scan_pos;

    for (uint64_t i = scan_pos; i < scan_end; i++) {
        auto* pkt = &((hsa_kernel_dispatch_packet_t*)state->ring_buf)[i & state->ring_mask];

        // Invoke interceptor callback chain (tracing, counter collection, etc.)
        invoke_interceptors(state, pkt, i);

        // Copy app packet from write-ahead position to submit position
        uint64_t dest = state->next_submit_pos;
        auto* dest_pkt = &((hsa_kernel_dispatch_packet_t*)state->ring_buf)[dest & state->ring_mask];
        if (dest_pkt != pkt) {
            memcpy(dest_pkt, pkt, 64);  // AQL packet is 64 bytes
        }

        // Write instrumentation packets after app packet (if counter collection active)
        if (state->k_factor > 0) {
            write_instrumentation(state, pkt, dest + 1, state->k_factor);
        }

        // Sync metadata queue (if paired)
        if (state->metadata_state) {
            sync_metadata(state, pkt, dest);
        }

        state->next_submit_pos = dest + 1 + state->k_factor;
    }

    state->next_scan_pos = scan_end;

    // Advance real WDID — GPU can now see all submitted packets
    __atomic_store_n(state->real_wdid, state->next_submit_pos, __ATOMIC_RELEASE);

    // Ring the real hardware doorbell
    state->original_signal_store(state->doorbell_signal, value);
}
```

### Doorbell value

The doorbell signal store value from the app is the app's virtual write pointer. The SDK passes this value through to the real doorbell. The GPU firmware uses the doorbell as a wake signal; the actual packet range is determined by `write_dispatch_id` (which the SDK has set to `next_submit_pos`).

### Thread safety

The `gate_lock` mutex serializes doorbell processing per queue. Multiple app threads may call `add_write_index` concurrently (virtual_wptr is atomic), but doorbell processing is serialized. This matches the HSA spec: the doorbell is a single signal, so concurrent stores are inherently serialized by the hardware.

---

## Queue Creation Interception

The SDK intercepts `hsa_queue_create` to inflate the ring size and set up QueueState.

```cpp
hsa_status_t intercepted_queue_create(hsa_agent_t agent, uint32_t size,
                                       hsa_queue_type32_t type, ...,
                                       hsa_queue_t** queue) {
    // Determine K based on active services
    uint64_t k = compute_k_factor();

    // Inflate ring size to accommodate write-ahead + submit regions
    uint32_t inflated_size = size * (1 + k) * 2;  // 2x for write-ahead headroom
    inflated_size = next_power_of_two(inflated_size);

    // Create the real queue with inflated size
    hsa_status_t status = original_queue_create(agent, inflated_size, type, ..., queue);
    if (status != HSA_STATUS_SUCCESS) return status;

    // Build QueueState
    auto state = std::make_unique<QueueState>();
    state->ring_buf       = (void*)(*queue)->base_address;
    state->ring_size      = (*queue)->size;
    state->ring_mask      = state->ring_size - 1;
    state->real_wdid      = &reinterpret_cast<amd_queue_t*>(*queue)->write_dispatch_id;
    state->real_rdid      = &reinterpret_cast<amd_queue_t*>(*queue)->read_dispatch_id;
    state->doorbell_signal = (*queue)->doorbell_signal;
    state->k_factor       = k;
    state->original_signal_store = get_original_signal_store_fn();

    get_queue_registry().wlock([&](auto& map) {
        map[*queue] = std::move(state);
    });

    return HSA_STATUS_SUCCESS;
}
```

### Ring size inflation

The ring must be large enough to hold both:
- **Write-ahead zone:** where the app has written packets but the SDK hasn't processed yet
- **Submit zone:** where the SDK has written final packets (app + instrumentation) for the GPU

Worst case: every app packet expands to (1+K) packets. With app requesting `size` slots and K=7, the submit zone needs `size * 8` slots. Adding headroom for the write-ahead zone, `size * (1+K) * 2` is conservative. This is rounded up to the next power of two (HSA requirement).

### Ring size cap

HSA may cap the ring size at a hardware maximum (typically 131072 or 262144 packets). If the inflated size exceeds the cap, the SDK should either:
- Fall back to a smaller K (degrade gracefully)
- Report a warning and proceed with the capped size

---

## Metadata Queue Pairing

The metadata queue is a separate HSA queue whose entries stay in lock-step with the compute queue's AQL ring. Each compute queue packet has a corresponding 256-byte metadata entry at the same index.

### Discovery

The SDK needs to discover which metadata queue is paired with which compute queue. This requires a new HSA API or convention:

**Option A:** New `hsa_amd_queue_get_info` attribute:
```cpp
hsa_amd_queue_get_info(compute_queue,
                        HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_RING_BUFFER,
                        &metadata_ring_ptr);
```

**Option B:** The metadata queue is created by HIP/runtime and the pairing is communicated to the SDK via a registration API.

The design supports either option — `QueueState::metadata_queue` and `QueueState::metadata_state` are set when the pairing is established.

### Sync at doorbell time

When the compute queue's doorbell handler copies an app packet to the submit position, it also writes the corresponding metadata entry:

```cpp
void sync_metadata(QueueState* compute_state,
                   const hsa_kernel_dispatch_packet_t* pkt,
                   uint64_t dest_pos) {
    auto* meta_state = compute_state->metadata_state;
    auto* meta_ring = (uint8_t*)meta_state->ring_buf;
    constexpr size_t META_SIZE = 256;

    uint64_t meta_dest = meta_state->next_submit_pos;

    // Write metadata for the app packet
    auto* meta_entry = &meta_ring[(meta_dest & meta_state->ring_mask) * META_SIZE];
    populate_metadata(meta_entry, pkt);

    // Write noop metadata for instrumentation packets
    for (uint64_t k = 1; k <= compute_state->k_factor; k++) {
        auto* instr_meta = &meta_ring[((meta_dest + k) & meta_state->ring_mask) * META_SIZE];
        fill_noop_metadata(instr_meta);
    }

    meta_state->next_submit_pos = meta_dest + 1 + compute_state->k_factor;

    // Advance metadata queue's real WDID in lock-step
    __atomic_store_n(meta_state->real_wdid, meta_state->next_submit_pos, __ATOMIC_RELEASE);
}
```

---

## HSA Information Requirements

### Available from existing HSA APIs

| Information | Source | Notes |
|-------------|--------|-------|
| Ring buffer base address | `hsa_queue_t::base_address` | Public field |
| Ring size (slot count) | `hsa_queue_t::size` | Public field |
| Doorbell signal | `hsa_queue_t::doorbell_signal` | Public field |
| `write_dispatch_id` address | `&((amd_queue_t*)queue)->write_dispatch_id` | AMD extension, well-known offset |
| `read_dispatch_id` address | `&((amd_queue_t*)queue)->read_dispatch_id` | AMD extension, well-known offset |
| Original HSA API function pointers | Saved by `copy_table()` in `hsa.cpp` | SDK already does this |

### Potentially needed from HSA (new APIs)

| Information | Proposed API | Purpose |
|-------------|-------------|---------|
| Metadata queue ring buffer | `HSA_AMD_QUEUE_INFO_PREFETCH_METADATA_RING_BUFFER` | Discover metadata ring for a compute queue |
| Metadata queue handle | New pairing API or convention | Associate compute queue with metadata queue |

### Not needed

| Information | Why not needed |
|-------------|---------------|
| Shadow wptr allocation (KFD flags) | SDK manages WDID directly, no separate GPU-polled pointer |
| `CP_HQD_PQ_WPTR_POLL_ADDR` | GPU polls `write_dispatch_id` as normal; SDK controls when it advances |
| `hsa_amd_queue_intercept_create` | Not used — SDK interposes at the API table level |
| `hsa_amd_queue_intercept_register` | Not used — SDK's doorbell handler replaces the WriteInterceptor |

---

## Instrumentation Injection

The `write_instrumentation` function writes K packets after each app packet. The exact packet sequence depends on the active service.

### Counter collection (K=7)

K represents the number of *additional* packets the SDK injects around each app packet. With K=7, the SDK writes a total of 8 packets per app dispatch (1 app + 7 instrumentation):

| Slot Offset | Packet | Purpose |
|-------------|--------|---------|
| 0 | BARRIER_AND | Serializer `ready_signal` gate |
| 1 | BARRIER_AND | Serializer `block_signal` gate |
| 2 | PM4 IB | AQLProfile start (counter configuration) |
| 3 | Kernel dispatch | App packet (copied from write-ahead zone, completion_signal replaced) |
| 4 | PM4 IB | Counter read |
| 5 | PM4 IB | Counter stop (with `interrupt_signal`) |
| 6 | BARRIER_AND | Fires `kernel_completion_signal` handler |
| 7 | (reserved) | Available for future use / alignment |

The app's original kernel dispatch is at offset 3, not offset 0. The doorbell handler copies the app packet from the write-ahead zone to `next_submit_pos + 3`, with barriers and PM4 packets at surrounding offsets. Total ring advancement per app dispatch: `1 + K = 8` slots.

### Tracing only (K=0)

No extra packets. The app packet is copied as-is from write-ahead to submit position. Trace data is recorded during the scan but no ring slots are consumed for instrumentation.

### Dynamic K

`compute_k_factor()` checks which contexts/services are active:
- If any context has counter collection enabled: K=7
- If only tracing/kernel-trace: K=0
- Future modes may use other K values

K is set per-queue at creation time and stored in `QueueState::k_factor`.

---

## Pre-implementation Experiment

Before beginning implementation, empirically verify GPU write-pointer polling behavior:

**Question:** When `write_dispatch_id` is advanced (via `hsa_queue_add_write_index`), does the GPU start consuming packets immediately, or only after the doorbell is rung?

**Expected behavior:** The GPU's CP continuously polls `CP_HQD_PQ_WPTR_POLL_ADDR` (which points at `write_dispatch_id`). If the CP is awake, it may start consuming packets as soon as WDID advances, without waiting for a doorbell. The doorbell is an optimization to wake the CP from sleep.

**Test plan:**
1. Create a queue
2. Write a kernel dispatch packet at slot 0 with a known completion signal
3. Call `hsa_queue_add_write_index(1)` to advance WDID — do NOT ring the doorbell
4. Wait 100ms
5. Check if the completion signal has been decremented (indicating the GPU consumed the packet)
6. If not, ring the doorbell and check again

This determines whether the deferred-WDID approach works correctly: if the GPU only acts on doorbell, we wouldn't strictly need deferred WDID. But if the GPU polls WDID, deferred WDID is essential.

---

## Testing Strategy

All interposition functions operate on `QueueState` and raw memory. They can be tested without HSA, HIP, or a GPU.

### Unit tests (no HSA required)

```cpp
TEST(SlotAllocation, AddWriteIndexAdvancesVirtualWptr) {
    QueueState state;
    alignas(64) char ring[64 * 128];
    state.ring_buf = ring;
    state.ring_size = 128;
    state.ring_mask = 127;
    state.k_factor = 7;

    uint64_t idx0 = intercepted_add_write_index_impl(&state, 1);
    EXPECT_EQ(idx0, 0);
    EXPECT_EQ(state.virtual_wptr.load(), 1);

    uint64_t idx1 = intercepted_add_write_index_impl(&state, 3);
    EXPECT_EQ(idx1, 1);
    EXPECT_EQ(state.virtual_wptr.load(), 4);
}

TEST(DoorbellHandler, CopiesAppPacketToSubmitPosition) {
    QueueState state;
    alignas(64) char ring[64 * 256];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state.ring_buf = ring;
    state.ring_size = 256;
    state.ring_mask = 255;
    state.real_wdid = &real_wdid;
    state.real_rdid = &real_rdid;
    state.k_factor = 7;

    // Simulate: app reserved 1 slot, wrote a packet at virtual position 0
    state.virtual_wptr.store(1);
    auto* pkt = (hsa_kernel_dispatch_packet_t*)ring;
    pkt->header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    pkt->kernel_object = 0xDEADBEEF;

    // Process doorbell
    intercepted_doorbell_impl(&state, 0);

    // Verify: app packet at submit position 0, real WDID = 8 (1 + K)
    auto* submitted = (hsa_kernel_dispatch_packet_t*)&ring[0];
    EXPECT_EQ(submitted->kernel_object, 0xDEADBEEF);
    EXPECT_EQ(real_wdid, 8);
    EXPECT_EQ(state.next_submit_pos, 8);
}

TEST(DoorbellHandler, TraceOnlyNoInflation) {
    QueueState state;
    // ... similar setup with k_factor = 0 ...

    intercepted_doorbell_impl(&state, 0);

    EXPECT_EQ(real_wdid, 1);  // No inflation
    EXPECT_EQ(state.next_submit_pos, 1);
}

TEST(MetadataSync, WritesMetadataInLockStep) {
    QueueState compute_state, meta_state;
    // ... setup both with paired pointers ...

    sync_metadata(&compute_state, pkt, 0);

    EXPECT_EQ(meta_state.next_submit_pos, 8);  // 1 + K metadata entries
}
```

### Integration tests (requires HSA + GPU)

- Run `vectorAdd` with tracing enabled — verify trace output and correct execution
- Run `vectorAdd` with counter collection — verify counters are collected and kernel completes
- Run rocprofiler-sdk test suite (`ctest -j1`)
- Stress test: high-dispatch-count workload with counter collection

---

## File Organization

New files in rocprofiler-sdk:

```
source/lib/rocprofiler-sdk/hsa/
    queue_intercept.hpp       # QueueState struct, function declarations
    queue_intercept.cpp       # Interposition functions, doorbell handler
    queue_intercept_meta.cpp  # Metadata queue sync (separable)
```

Modified files:

```
source/lib/rocprofiler-sdk/hsa/
    hsa.cpp                   # Register intercepted functions in update_table
    queue.cpp                 # Remove or bypass WriteInterceptor path
    queue_controller.cpp      # Use new QueueState instead of Queue wrapper
```

---

## Relationship to Existing Code

### What this replaces

- `hsa_amd_queue_intercept_create` / `hsa_amd_queue_intercept_register` — no longer needed
- `WriteInterceptor` callback in `queue.cpp` — replaced by doorbell handler
- `InterceptQueue` in HSA runtime — already deleted on `users/bewelton/shadow_ptr` branch

### What this preserves

- HSA API table wrapping mechanism (`copy_table` / `update_table`) — reused as-is
- Interceptor callback chain concept — `interceptors` vector in QueueState
- Counter collection packet construction (`AQLPacket`, `CounterPacketConstruct`) — reused
- Tracing data recording (`tracing_data`, `queue_info_session`) — reused
- Signal pool infrastructure — reused

### What this enables

- No `base_address` swap — cached pointers remain valid
- No `doorbell_signal` swap — cached doorbell handles remain valid
- Dynamic K-factor per active service
- Clean metadata queue pairing
- Full testability without HSA/HIP/GPU

---

## Open Questions

1. **HIP backpressure:** HIP's yield loop checks `write_dispatch_id - read_dispatch_id >= queue_size - 1`. With deferred WDID, the app's virtual_wptr advances but real WDID doesn't until doorbell. HIP reads `write_dispatch_id` via `hsa_queue_load_write_index`, which the SDK intercepts to return `virtual_wptr`. But HIP also reads `read_dispatch_id` via `hsa_queue_load_read_index`, which returns the real GPU read pointer. The virtual-vs-real gap means HIP's backpressure calculation may be off. Need to verify HIP's exact backpressure logic and whether the intercepted `load_write_index` returning `virtual_wptr` is sufficient.

2. **Queue destroy:** When `hsa_queue_destroy` is called, the SDK must ensure all pending packets in the write-ahead zone are drained before destroying the QueueState. This may require a final doorbell processing pass.

3. **Multiple concurrent doorbells:** If two threads ring the doorbell concurrently for the same queue, the `gate_lock` serializes them. But the doorbell value from the second thread may be stale (it was computed before the first thread's processing). Need to verify this is handled correctly.

4. **Signal store identity:** The SDK must identify which signal stores are doorbell operations vs. regular signal operations. The doorbell-to-queue lookup map handles this, but we need to ensure no false positives (a non-doorbell signal that happens to match a doorbell handle).

5. **Profiler serialization:** The existing serializer (`profile_serializer` in `queue.cpp`) coordinates across queues using `block_signal` and `ready_signal`. This serializer logic needs to be ported into the new doorbell handler.
