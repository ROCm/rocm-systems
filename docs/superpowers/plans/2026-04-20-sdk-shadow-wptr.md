# SDK-Level Shadow Write Pointer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `hsa_amd_queue_intercept_create`/`WriteInterceptor`-based queue interception with SDK-level write-index interposition that manages a virtual write pointer, defers real `write_dispatch_id` advancement to doorbell time, and copies app packets to a submit region interleaved with instrumentation packets.

**Architecture:** Flat functions installed in the HSA API table + a per-queue `QueueState` looked up via `common::Synchronized` concurrent maps. The SDK intercepts `add_write_index`/`store_write_index`/`cas_write_index` to manage a virtual write pointer without forwarding to HSA. At doorbell time (signal store interception), the SDK scans write-ahead packets, copies them to the submit region with instrumentation, advances real WDID, and rings the real doorbell. No HSA runtime changes needed.

**Tech Stack:** C++17, rocprofiler-sdk HSA API table wrapping (`copy_table`/`update_table`), Google Test, CMake

**Design Spec:** `docs/superpowers/specs/2026-04-20-sdk-shadow-wptr-design.md`

**Key References:**
- HSA API table wrapping: `source/lib/rocprofiler-sdk/hsa/hsa.cpp` (lines 574-687)
- Wrapper functor generation: `source/lib/rocprofiler-sdk/hsa/defines.hpp` (lines 193-272)
- API definitions: `source/lib/rocprofiler-sdk/hsa/hsa.def.cpp` (lines 140-182)
- Existing Queue class: `source/lib/rocprofiler-sdk/hsa/queue.hpp`, `queue.cpp`
- QueueController: `source/lib/rocprofiler-sdk/hsa/queue_controller.hpp`
- Synchronized container: `source/lib/common/synchronized.hpp`
- `amd_queue_t` struct: `projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_queue.h` (line 79)
- Unit test pattern: `source/lib/rocprofiler-sdk/tests/hsa.cpp`, `tests/CMakeLists.txt`

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp` | `QueueState` struct, registry accessors, interposition function declarations |
| `source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp` | Registry implementation, all 14 interposition functions (10 write-index + 2 load-read + 2 signal-store), doorbell handler, queue lifecycle hooks |
| `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp` | Unit tests for QueueState, slot allocation, doorbell handler, metadata sync |

### Modified Files

| File | Changes |
|------|---------|
| `source/lib/rocprofiler-sdk/hsa/CMakeLists.txt` | Add `queue_intercept.cpp` to `ROCPROFILER_LIB_HSA_SOURCES`, `queue_intercept.hpp` to `ROCPROFILER_LIB_HSA_HEADERS` |
| `source/lib/rocprofiler-sdk/tests/CMakeLists.txt` | Add `queue_intercept.cpp` to `rocprofiler_lib_sources` |
| `source/lib/rocprofiler-sdk/hsa/hsa.cpp` | Add conditional interposition registration in `update_table` path |
| `source/lib/rocprofiler-sdk/hsa/queue_controller.cpp` | Wire queue creation/destruction to `QueueState` lifecycle |

---

## Task 0: Pre-implementation HW Experiment

**Purpose:** Verify whether the GPU starts consuming AQL packets when `write_dispatch_id` is advanced without ringing the doorbell. This determines whether deferred WDID is essential.

**Files:**
- Create: `~/ai/experiments/wdid_poll_test.cpp` (standalone HSA program)

- [ ] **Step 1: Write the test program**

```cpp
// wdid_poll_test.cpp — standalone HSA program, compile with:
// hipcc -x hip wdid_poll_test.cpp -o wdid_poll_test -lhsa-runtime64
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>

// Minimal empty kernel — just returns
__global__ void empty_kernel() {}

int main() {
    hsa_init();

    // Find a GPU agent
    hsa_agent_t gpu_agent = {0};
    hsa_iterate_agents([](hsa_agent_t agent, void* data) -> hsa_status_t {
        hsa_device_type_t type;
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (type == HSA_DEVICE_TYPE_GPU) {
            *static_cast<hsa_agent_t*>(data) = agent;
            return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
    }, &gpu_agent);

    if (gpu_agent.handle == 0) {
        printf("FAIL: No GPU agent found\n");
        return 1;
    }

    // Create a queue
    hsa_queue_t* queue = nullptr;
    hsa_queue_create(gpu_agent, 1024, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0, &queue);

    // Create a completion signal
    hsa_signal_t signal;
    hsa_signal_create(1, 0, nullptr, &signal);

    // Write a NOP barrier packet at slot 0 with the completion signal
    uint64_t idx = hsa_queue_add_write_index_relaxed(queue, 1);
    hsa_barrier_and_packet_t* pkt = (hsa_barrier_and_packet_t*)(
        (char*)queue->base_address + (idx % queue->size) * 64);
    memset(pkt, 0, 64);
    pkt->completion_signal = signal;

    // Set header LAST (makes packet valid)
    __atomic_store_n(&pkt->header,
        (HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE),
        __ATOMIC_RELEASE);

    printf("WDID advanced, packet written. NOT ringing doorbell.\n");
    printf("Signal value before wait: %ld\n", hsa_signal_load_relaxed(signal));

    // Wait 500ms WITHOUT ringing doorbell
    usleep(500000);

    int64_t val_after = hsa_signal_load_relaxed(signal);
    printf("Signal value after 500ms (no doorbell): %ld\n", val_after);

    if (val_after == 0) {
        printf("RESULT: GPU consumed packet WITHOUT doorbell — deferred WDID is ESSENTIAL\n");
    } else {
        printf("RESULT: GPU did NOT consume packet without doorbell — ringing now...\n");
        hsa_signal_store_relaxed(queue->doorbell_signal, idx);
        usleep(100000);
        val_after = hsa_signal_load_relaxed(signal);
        printf("Signal value after doorbell: %ld\n", val_after);
        if (val_after == 0) {
            printf("RESULT: GPU consumed packet ONLY after doorbell\n");
        } else {
            printf("RESULT: GPU did not consume packet at all — check setup\n");
        }
    }

    hsa_signal_destroy(signal);
    hsa_queue_destroy(queue);
    hsa_shut_down();
    return 0;
}
```

- [ ] **Step 2: Deploy and run on a GPU host**

Compile and run on a machine with GPU access (e.g., banff container):
```bash
hipcc wdid_poll_test.cpp -o wdid_poll_test -lhsa-runtime64
./wdid_poll_test
```

Expected: One of two outcomes:
- `GPU consumed packet WITHOUT doorbell` → deferred WDID is essential
- `GPU consumed packet ONLY after doorbell` → deferred WDID is still correct (conservative), but we could also just hold back the doorbell

- [ ] **Step 3: Record result in spec**

Update `docs/superpowers/specs/2026-04-20-sdk-shadow-wptr-design.md` "Pre-implementation Experiment" section with the empirical result.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-04-20-sdk-shadow-wptr-design.md
git commit -m "docs: record WDID polling experiment result"
```

---

## Task 1: QueueState Struct and Registry

**Purpose:** Define the core data structure and thread-safe lookup maps. This is the foundation everything else builds on.

**Files:**
- Create: `source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp`
- Create: `source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp`
- Modify: `source/lib/rocprofiler-sdk/hsa/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`:

```cpp
// MIT License
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
// [standard license header]

#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"

#include <gtest/gtest.h>

using namespace rocprofiler::hsa::queue_intercept;

TEST(QueueIntercept, QueueStateDefaultInit)
{
    QueueState state{};
    EXPECT_EQ(state.ring_buf, nullptr);
    EXPECT_EQ(state.ring_size, 0u);
    EXPECT_EQ(state.ring_mask, 0u);
    EXPECT_EQ(state.virtual_wptr.load(), 0u);
    EXPECT_EQ(state.real_wdid, nullptr);
    EXPECT_EQ(state.real_rdid, nullptr);
    EXPECT_EQ(state.next_scan_pos, 0u);
    EXPECT_EQ(state.next_submit_pos, 0u);
    EXPECT_EQ(state.k_factor, 0u);
    EXPECT_EQ(state.metadata_state, nullptr);
}

TEST(QueueIntercept, RegistryInsertAndLookup)
{
    auto& registry = get_queue_registry();

    // Use a fake queue pointer
    hsa_queue_t fake_queue{};
    auto        state     = std::make_unique<QueueState>();
    state->ring_size      = 256;
    state->ring_mask      = 255;
    auto* state_ptr       = state.get();

    registry.wlock([&](auto& map) { map[&fake_queue] = std::move(state); });

    auto* found = lookup_queue_state(&fake_queue);
    EXPECT_EQ(found, state_ptr);
    EXPECT_EQ(found->ring_size, 256u);

    // Clean up
    registry.wlock([&](auto& map) { map.erase(&fake_queue); });
    EXPECT_EQ(lookup_queue_state(&fake_queue), nullptr);
}

TEST(QueueIntercept, DoorbellMapInsertAndLookup)
{
    auto& registry = get_queue_registry();

    hsa_queue_t  fake_queue{};
    hsa_signal_t fake_doorbell{42};

    auto state              = std::make_unique<QueueState>();
    state->doorbell_signal  = fake_doorbell;
    auto* state_ptr         = state.get();

    registry.wlock([&](auto& map) { map[&fake_queue] = std::move(state); });
    register_doorbell(&fake_queue, fake_doorbell);

    auto* found = lookup_queue_state_by_doorbell(fake_doorbell);
    EXPECT_EQ(found, state_ptr);

    unregister_doorbell(fake_doorbell);
    EXPECT_EQ(lookup_queue_state_by_doorbell(fake_doorbell), nullptr);

    registry.wlock([&](auto& map) { map.erase(&fake_queue); });
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd projects/rocprofiler-sdk/build && cmake --build . --target rocprofiler-sdk-lib-tests 2>&1 | tail -5
```

Expected: Compilation failure — `queue_intercept.hpp` does not exist.

- [ ] **Step 3: Write queue_intercept.hpp**

Create `source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp`:

```cpp
// MIT License
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
// [standard license header]

#pragma once

#include "lib/common/synchronized.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
struct QueueState
{
    // Ring buffer (from hsa_queue_t at creation time)
    void*    ring_buf  = nullptr;
    uint32_t ring_size = 0;
    uint32_t ring_mask = 0;

    // Pointer management
    std::atomic<uint64_t> virtual_wptr{0};
    volatile uint64_t*    real_wdid      = nullptr;
    volatile uint64_t*    real_rdid      = nullptr;
    uint64_t              next_scan_pos  = 0;
    uint64_t              next_submit_pos = 0;

    // Doorbell
    hsa_signal_t doorbell_signal = {0};

    // Instrumentation
    uint64_t k_factor = 0;

    // Metadata queue pairing (nullable)
    QueueState* metadata_state = nullptr;

    // Synchronization
    std::mutex gate_lock;
};

using queue_registry_t =
    common::Synchronized<std::unordered_map<const hsa_queue_t*, std::unique_ptr<QueueState>>>;
using doorbell_map_t =
    common::Synchronized<std::unordered_map<uint64_t, QueueState*>>;

queue_registry_t&
get_queue_registry();

doorbell_map_t&
get_doorbell_map();

QueueState*
lookup_queue_state(const hsa_queue_t* queue);

QueueState*
lookup_queue_state_by_doorbell(hsa_signal_t signal);

void
register_doorbell(const hsa_queue_t* queue, hsa_signal_t doorbell);

void
unregister_doorbell(hsa_signal_t doorbell);

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
```

- [ ] **Step 4: Write queue_intercept.cpp**

Create `source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp`:

```cpp
// MIT License
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
// [standard license header]

#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"

#include "lib/common/static_object.hpp"

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
queue_registry_t&
get_queue_registry()
{
    static auto*& _v = common::static_object<queue_registry_t>::construct();
    return *_v;
}

doorbell_map_t&
get_doorbell_map()
{
    static auto*& _v = common::static_object<doorbell_map_t>::construct();
    return *_v;
}

QueueState*
lookup_queue_state(const hsa_queue_t* queue)
{
    QueueState* result = nullptr;
    get_queue_registry().rlock([&](const auto& map) {
        auto it = map.find(queue);
        if(it != map.end()) result = it->second.get();
    });
    return result;
}

QueueState*
lookup_queue_state_by_doorbell(hsa_signal_t signal)
{
    QueueState* result = nullptr;
    get_doorbell_map().rlock([&](const auto& map) {
        auto it = map.find(signal.handle);
        if(it != map.end()) result = it->second;
    });
    return result;
}

void
register_doorbell(const hsa_queue_t* queue, hsa_signal_t doorbell)
{
    auto* state = lookup_queue_state(queue);
    if(!state) return;
    get_doorbell_map().wlock([&](auto& map) { map[doorbell.handle] = state; });
}

void
unregister_doorbell(hsa_signal_t doorbell)
{
    get_doorbell_map().wlock([&](auto& map) { map.erase(doorbell.handle); });
}

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
```

- [ ] **Step 5: Add to CMakeLists**

In `source/lib/rocprofiler-sdk/hsa/CMakeLists.txt`, add `queue_intercept.cpp` to `ROCPROFILER_LIB_HSA_SOURCES` and `queue_intercept.hpp` to `ROCPROFILER_LIB_HSA_HEADERS`:

```cmake
set(ROCPROFILER_LIB_HSA_SOURCES
    abi.cpp
    agent_cache.cpp
    aql_packet.cpp
    async_copy.cpp
    hsa_barrier.cpp
    hsa.cpp
    memory_allocation.cpp
    pc_sampling.hpp
    profile_serializer.cpp
    queue_controller.cpp
    queue.cpp
    queue_intercept.cpp
    scratch_memory.cpp)

set(ROCPROFILER_LIB_HSA_HEADERS
    agent_cache.hpp
    aql_packet.hpp
    async_copy.hpp
    memory_allocation.hpp
    defines.hpp
    hsa_barrier.hpp
    hsa.hpp
    pc_sampling.cpp
    profile_serializer.hpp
    queue_controller.hpp
    queue.hpp
    queue_info_session.hpp
    queue_intercept.hpp
    rocprofiler_packet.hpp
    scratch_memory.hpp
    signal.hpp
    utils.hpp)
```

In `source/lib/rocprofiler-sdk/tests/CMakeLists.txt`, add `queue_intercept.cpp` to `rocprofiler_lib_sources`:

```cmake
set(rocprofiler_lib_sources
    agent.cpp
    buffer.cpp
    contexts.cpp
    enum_string.cpp
    hsa.cpp
    late_start.cpp
    naming.cpp
    timestamp.cpp
    version.cpp
    hsa_barrier.cpp
    queue_intercept.cpp)
```

- [ ] **Step 6: Build and run tests**

```bash
cd projects/rocprofiler-sdk/build && cmake .. && cmake --build . --target rocprofiler-sdk-lib-tests -j$(nproc)
./rocprofiler-sdk-lib-tests --gtest_filter="QueueIntercept.*"
```

Expected: All 3 tests PASS.

- [ ] **Step 7: Format and commit**

```bash
clang-format-11 -i source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
                   source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
                   source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
cmake-format -i source/lib/rocprofiler-sdk/hsa/CMakeLists.txt \
                source/lib/rocprofiler-sdk/tests/CMakeLists.txt
git add source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
        source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
        source/lib/rocprofiler-sdk/hsa/CMakeLists.txt \
        source/lib/rocprofiler-sdk/tests/queue_intercept.cpp \
        source/lib/rocprofiler-sdk/tests/CMakeLists.txt
git commit -m "feat(hsa): add QueueState struct and registry for SDK-level queue interception"
```

---

## Task 2: Write-Index Interposition Functions

**Purpose:** Implement the 10 write-index interposition functions that manage `virtual_wptr` without forwarding to HSA. These are pure functions on `QueueState` — fully testable without a GPU.

**Files:**
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp`
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp`
- Modify: `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`

- [ ] **Step 1: Write failing tests for AddWriteIndex**

Append to `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`:

```cpp
TEST(QueueIntercept, AddWriteIndexAdvancesVirtualWptr)
{
    QueueState state{};
    alignas(64) char ring[64 * 128];
    state.ring_buf  = ring;
    state.ring_size = 128;
    state.ring_mask = 127;

    uint64_t idx0 = add_write_index_impl(&state, 1);
    EXPECT_EQ(idx0, 0u);
    EXPECT_EQ(state.virtual_wptr.load(), 1u);

    uint64_t idx1 = add_write_index_impl(&state, 3);
    EXPECT_EQ(idx1, 1u);
    EXPECT_EQ(state.virtual_wptr.load(), 4u);
}

TEST(QueueIntercept, StoreWriteIndexSetsVirtualWptr)
{
    QueueState state{};
    store_write_index_impl(&state, 42);
    EXPECT_EQ(state.virtual_wptr.load(), 42u);
    store_write_index_impl(&state, 0);
    EXPECT_EQ(state.virtual_wptr.load(), 0u);
}

TEST(QueueIntercept, CasWriteIndexSuccess)
{
    QueueState state{};
    state.virtual_wptr.store(10);

    uint64_t prev = cas_write_index_impl(&state, 10, 20);
    EXPECT_EQ(prev, 10u);
    EXPECT_EQ(state.virtual_wptr.load(), 20u);
}

TEST(QueueIntercept, CasWriteIndexFailure)
{
    QueueState state{};
    state.virtual_wptr.store(10);

    uint64_t prev = cas_write_index_impl(&state, 5, 20);
    EXPECT_EQ(prev, 10u);
    EXPECT_EQ(state.virtual_wptr.load(), 10u);
}

TEST(QueueIntercept, LoadWriteIndexReturnsVirtualWptr)
{
    QueueState state{};
    state.virtual_wptr.store(99);
    EXPECT_EQ(load_write_index_impl(&state), 99u);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build . --target rocprofiler-sdk-lib-tests 2>&1 | tail -5
```

Expected: Compilation failure — `add_write_index_impl` not declared.

- [ ] **Step 3: Add declarations to queue_intercept.hpp**

Add inside the `queue_intercept` namespace, after the registry functions:

```cpp
// Write-index interposition — pure functions on QueueState
uint64_t
add_write_index_impl(QueueState* state, uint64_t value);

void
store_write_index_impl(QueueState* state, uint64_t value);

uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value);

uint64_t
load_write_index_impl(const QueueState* state);
```

- [ ] **Step 4: Implement in queue_intercept.cpp**

Append to `queue_intercept.cpp`:

```cpp
uint64_t
add_write_index_impl(QueueState* state, uint64_t value)
{
    return state->virtual_wptr.fetch_add(value, std::memory_order_relaxed);
}

void
store_write_index_impl(QueueState* state, uint64_t value)
{
    state->virtual_wptr.store(value, std::memory_order_relaxed);
}

uint64_t
cas_write_index_impl(QueueState* state, uint64_t expected, uint64_t value)
{
    uint64_t prev = expected;
    state->virtual_wptr.compare_exchange_strong(prev, value, std::memory_order_relaxed);
    return prev;
}

uint64_t
load_write_index_impl(const QueueState* state)
{
    return state->virtual_wptr.load(std::memory_order_relaxed);
}
```

- [ ] **Step 5: Build and run tests**

```bash
cmake --build . --target rocprofiler-sdk-lib-tests -j$(nproc)
./rocprofiler-sdk-lib-tests --gtest_filter="QueueIntercept.*"
```

Expected: All tests PASS (previous 3 + new 5 = 8 total).

- [ ] **Step 6: Format and commit**

```bash
clang-format-11 -i source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
                   source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
                   source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
git add source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
        source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
        source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
git commit -m "feat(hsa): add write-index interposition functions for virtual wptr management"
```

---

## Task 3: Doorbell Handler (Trace-Only, K=0)

**Purpose:** Implement the doorbell handler for the simplest case: K=0 (trace-only). The handler scans the write-ahead zone, copies app packets to the submit position 1:1, advances real WDID, and "rings" the doorbell (via a callback in tests). No instrumentation injection yet.

**Files:**
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp`
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp`
- Modify: `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`

- [ ] **Step 1: Write failing tests**

Append to `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`:

```cpp
namespace
{
hsa_kernel_dispatch_packet_t*
get_pkt(void* ring, uint64_t idx, uint32_t mask)
{
    return &reinterpret_cast<hsa_kernel_dispatch_packet_t*>(ring)[idx & mask];
}
}  // namespace

TEST(QueueIntercept, DoorbellTraceOnlyCopiesPacket)
{
    QueueState state{};
    alignas(64) char ring[64 * 256];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state.ring_buf       = ring;
    state.ring_size      = 256;
    state.ring_mask      = 255;
    state.real_wdid      = &real_wdid;
    state.real_rdid      = &real_rdid;
    state.k_factor       = 0;

    // Simulate: app reserved 1 slot at virtual pos 0, wrote a dispatch packet
    state.virtual_wptr.store(1);
    auto* pkt    = get_pkt(ring, 0, 255);
    pkt->header  = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    pkt->kernel_object = 0xDEADBEEF;

    bool doorbell_rang = false;
    process_doorbell_impl(&state, 0, [&](hsa_signal_t, hsa_signal_value_t) {
        doorbell_rang = true;
    });

    EXPECT_TRUE(doorbell_rang);
    EXPECT_EQ(real_wdid, 1u);
    EXPECT_EQ(state.next_submit_pos, 1u);
    EXPECT_EQ(state.next_scan_pos, 1u);

    // Packet at submit pos 0 should have the kernel_object
    auto* submitted = get_pkt(ring, 0, 255);
    EXPECT_EQ(submitted->kernel_object, 0xDEADBEEFu);
}

TEST(QueueIntercept, DoorbellMultiplePacketsTraceOnly)
{
    QueueState state{};
    alignas(64) char ring[64 * 256];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state.ring_buf  = ring;
    state.ring_size = 256;
    state.ring_mask = 255;
    state.real_wdid = &real_wdid;
    state.real_rdid = &real_rdid;
    state.k_factor  = 0;

    // App reserved 3 slots
    state.virtual_wptr.store(3);
    for(uint64_t i = 0; i < 3; i++)
    {
        auto* pkt       = get_pkt(ring, i, 255);
        pkt->header     = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
        pkt->kernel_object = static_cast<uint64_t>(0xA000 + i);
    }

    process_doorbell_impl(&state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    EXPECT_EQ(real_wdid, 3u);
    EXPECT_EQ(state.next_submit_pos, 3u);

    for(uint64_t i = 0; i < 3; i++)
    {
        auto* submitted = get_pkt(ring, i, 255);
        EXPECT_EQ(submitted->kernel_object, static_cast<uint64_t>(0xA000 + i));
    }
}

TEST(QueueIntercept, DoorbellNoNewPackets)
{
    QueueState state{};
    alignas(64) char ring[64 * 64];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state.ring_buf  = ring;
    state.ring_size = 64;
    state.ring_mask = 63;
    state.real_wdid = &real_wdid;
    state.real_rdid = &real_rdid;
    state.k_factor  = 0;

    // virtual_wptr == next_scan_pos, nothing to process
    state.virtual_wptr.store(0);
    bool doorbell_rang = false;
    process_doorbell_impl(&state, 0, [&](hsa_signal_t, hsa_signal_value_t) {
        doorbell_rang = true;
    });

    // Doorbell should still ring (forward the signal store)
    EXPECT_TRUE(doorbell_rang);
    EXPECT_EQ(real_wdid, 0u);
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: Compilation failure — `process_doorbell_impl` not declared.

- [ ] **Step 3: Add declaration to queue_intercept.hpp**

```cpp
using doorbell_fn_t = std::function<void(hsa_signal_t, hsa_signal_value_t)>;

void
process_doorbell_impl(QueueState* state, hsa_signal_value_t value, doorbell_fn_t ring_doorbell);
```

Add `#include <functional>` to the header includes.

- [ ] **Step 4: Implement process_doorbell_impl**

Append to `queue_intercept.cpp`:

```cpp
void
process_doorbell_impl(QueueState* state, hsa_signal_value_t value, doorbell_fn_t ring_doorbell)
{
    std::lock_guard<std::mutex> lock(state->gate_lock);

    uint64_t scan_end = state->virtual_wptr.load(std::memory_order_acquire);
    uint64_t scan_pos = state->next_scan_pos;

    for(uint64_t i = scan_pos; i < scan_end; i++)
    {
        auto* src = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(state->ring_buf) +
                    (i & state->ring_mask);

        uint64_t dest_idx = state->next_submit_pos;
        auto*    dst      = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(state->ring_buf) +
                         (dest_idx & state->ring_mask);

        if(dst != src)
        {
            memcpy(dst, src, 64);
        }

        state->next_submit_pos = dest_idx + 1 + state->k_factor;
    }

    state->next_scan_pos = scan_end;

    __atomic_store_n(state->real_wdid, state->next_submit_pos, __ATOMIC_RELEASE);

    ring_doorbell(state->doorbell_signal, value);
}
```

- [ ] **Step 5: Build and run tests**

```bash
cmake --build . --target rocprofiler-sdk-lib-tests -j$(nproc)
./rocprofiler-sdk-lib-tests --gtest_filter="QueueIntercept.*"
```

Expected: All 11 tests PASS.

- [ ] **Step 6: Format and commit**

```bash
clang-format-11 -i source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
                   source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
                   source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
git add source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
        source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
        source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
git commit -m "feat(hsa): add doorbell handler for trace-only mode (K=0)"
```

---

## Task 4: Doorbell Handler with K-Factor (Instrumentation Slots)

**Purpose:** Extend the doorbell handler to support K>0. When `k_factor > 0`, the handler copies each app packet to `next_submit_pos` and advances by `1 + k_factor`, leaving K slots for instrumentation packets. This task does NOT fill the instrumentation slots — just reserves them. Instrumentation filling is a separate task.

**Files:**
- Modify: `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`

- [ ] **Step 1: Write failing tests**

Append to `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`:

```cpp
TEST(QueueIntercept, DoorbellWithKFactor)
{
    QueueState state{};
    alignas(64) char ring[64 * 512];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state.ring_buf  = ring;
    state.ring_size = 512;
    state.ring_mask = 511;
    state.real_wdid = &real_wdid;
    state.real_rdid = &real_rdid;
    state.k_factor  = 7;

    // App reserved 1 slot at virtual pos 0
    state.virtual_wptr.store(1);
    auto* pkt          = get_pkt(ring, 0, 511);
    pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    pkt->kernel_object = 0xCAFE;

    process_doorbell_impl(&state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    // With K=7: 1 app packet + 7 instrumentation slots = 8 total
    EXPECT_EQ(real_wdid, 8u);
    EXPECT_EQ(state.next_submit_pos, 8u);
    EXPECT_EQ(state.next_scan_pos, 1u);

    // App packet should be at submit pos 0
    auto* submitted = get_pkt(ring, 0, 511);
    EXPECT_EQ(submitted->kernel_object, static_cast<uint64_t>(0xCAFE));
}

TEST(QueueIntercept, DoorbellTwoPacketsWithKFactor)
{
    QueueState state{};
    alignas(64) char ring[64 * 512];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state.ring_buf  = ring;
    state.ring_size = 512;
    state.ring_mask = 511;
    state.real_wdid = &real_wdid;
    state.real_rdid = &real_rdid;
    state.k_factor  = 7;

    // App reserved 2 slots at virtual pos 0 and 1
    state.virtual_wptr.store(2);
    get_pkt(ring, 0, 511)->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 0, 511)->kernel_object = 0xAAAA;
    get_pkt(ring, 1, 511)->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 1, 511)->kernel_object = 0xBBBB;

    process_doorbell_impl(&state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    // 2 app packets * (1 + 7) = 16 total
    EXPECT_EQ(real_wdid, 16u);
    EXPECT_EQ(state.next_submit_pos, 16u);

    // First app packet at submit pos 0, second at submit pos 8
    EXPECT_EQ(get_pkt(ring, 0, 511)->kernel_object, static_cast<uint64_t>(0xAAAA));
    EXPECT_EQ(get_pkt(ring, 8, 511)->kernel_object, static_cast<uint64_t>(0xBBBB));
}
```

- [ ] **Step 2: Run tests**

```bash
cmake --build . --target rocprofiler-sdk-lib-tests -j$(nproc)
./rocprofiler-sdk-lib-tests --gtest_filter="QueueIntercept.*"
```

Expected: All 13 tests PASS. The K-factor behavior is already implemented in `process_doorbell_impl` from Task 3 — `next_submit_pos` advances by `1 + state->k_factor` per packet.

- [ ] **Step 3: Format and commit**

```bash
clang-format-11 -i source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
git add source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
git commit -m "test(hsa): add K-factor doorbell handler tests"
```

---

## Task 5: Queue Creation and Destruction Hooks

**Purpose:** Add functions that `QueueController` will call when queues are created/destroyed to set up and tear down `QueueState`. Includes ring size inflation logic.

**Files:**
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp`
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp`
- Modify: `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`

- [ ] **Step 1: Write failing tests**

Append to `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`:

```cpp
TEST(QueueIntercept, ComputeInflatedRingSize)
{
    // K=0: no inflation, just round to power of 2
    EXPECT_EQ(compute_inflated_ring_size(1024, 0), 1024u);

    // K=7: 1024 * (1+7) * 2 = 16384, already power of 2
    EXPECT_EQ(compute_inflated_ring_size(1024, 7), 16384u);

    // K=7 with non-power-of-2 result: 512 * 16 = 8192
    EXPECT_EQ(compute_inflated_ring_size(512, 7), 8192u);

    // K=1: 1024 * 4 = 4096
    EXPECT_EQ(compute_inflated_ring_size(1024, 1), 4096u);
}

TEST(QueueIntercept, CreateAndDestroyQueueState)
{
    hsa_queue_t fake_queue{};
    fake_queue.base_address = reinterpret_cast<uint64_t>(malloc(64 * 256));
    fake_queue.size         = 256;
    fake_queue.doorbell_signal = {.handle = 123};

    uint64_t fake_wdid = 0;
    uint64_t fake_rdid = 0;

    create_queue_state(&fake_queue, &fake_wdid, &fake_rdid, 7);

    auto* state = lookup_queue_state(&fake_queue);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->ring_buf, reinterpret_cast<void*>(fake_queue.base_address));
    EXPECT_EQ(state->ring_size, 256u);
    EXPECT_EQ(state->ring_mask, 255u);
    EXPECT_EQ(state->real_wdid, &fake_wdid);
    EXPECT_EQ(state->real_rdid, &fake_rdid);
    EXPECT_EQ(state->k_factor, 7u);
    EXPECT_EQ(state->doorbell_signal.handle, 123u);

    auto* by_doorbell = lookup_queue_state_by_doorbell({.handle = 123});
    EXPECT_EQ(by_doorbell, state);

    destroy_queue_state(&fake_queue);
    EXPECT_EQ(lookup_queue_state(&fake_queue), nullptr);
    EXPECT_EQ(lookup_queue_state_by_doorbell({.handle = 123}), nullptr);

    free(reinterpret_cast<void*>(fake_queue.base_address));
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: Compilation failure — `compute_inflated_ring_size`, `create_queue_state`, `destroy_queue_state` not declared.

- [ ] **Step 3: Add declarations**

Add to `queue_intercept.hpp`:

```cpp
uint32_t
compute_inflated_ring_size(uint32_t requested_size, uint64_t k_factor);

void
create_queue_state(const hsa_queue_t*   queue,
                   volatile uint64_t*   wdid_addr,
                   volatile uint64_t*   rdid_addr,
                   uint64_t             k_factor);

void
destroy_queue_state(const hsa_queue_t* queue);
```

- [ ] **Step 4: Implement**

Append to `queue_intercept.cpp`:

```cpp
namespace
{
uint32_t
next_power_of_two(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
}  // namespace

uint32_t
compute_inflated_ring_size(uint32_t requested_size, uint64_t k_factor)
{
    if(k_factor == 0) return requested_size;
    uint64_t inflated = static_cast<uint64_t>(requested_size) * (1 + k_factor) * 2;
    if(inflated > 262144) inflated = 262144;
    return next_power_of_two(static_cast<uint32_t>(inflated));
}

void
create_queue_state(const hsa_queue_t*   queue,
                   volatile uint64_t*   wdid_addr,
                   volatile uint64_t*   rdid_addr,
                   uint64_t             k_factor)
{
    auto state           = std::make_unique<QueueState>();
    state->ring_buf      = reinterpret_cast<void*>(queue->base_address);
    state->ring_size     = queue->size;
    state->ring_mask     = queue->size - 1;
    state->real_wdid     = wdid_addr;
    state->real_rdid     = rdid_addr;
    state->doorbell_signal = queue->doorbell_signal;
    state->k_factor      = k_factor;

    auto* raw_ptr = state.get();
    get_queue_registry().wlock([&](auto& map) { map[queue] = std::move(state); });
    get_doorbell_map().wlock(
        [&](auto& map) { map[queue->doorbell_signal.handle] = raw_ptr; });
}

void
destroy_queue_state(const hsa_queue_t* queue)
{
    auto* state = lookup_queue_state(queue);
    if(!state) return;

    unregister_doorbell(state->doorbell_signal);
    get_queue_registry().wlock([&](auto& map) { map.erase(queue); });
}
```

- [ ] **Step 5: Build and run tests**

```bash
cmake --build . --target rocprofiler-sdk-lib-tests -j$(nproc)
./rocprofiler-sdk-lib-tests --gtest_filter="QueueIntercept.*"
```

Expected: All 15 tests PASS.

- [ ] **Step 6: Format and commit**

```bash
clang-format-11 -i source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
                   source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
                   source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
git add source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
        source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
        source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
git commit -m "feat(hsa): add queue creation/destruction hooks and ring size inflation"
```

---

## Task 6: HSA API Table Integration

**Purpose:** Wire the interposition functions into the actual HSA API table so they get called when the app uses HSA queue operations. This is the integration point where the unit-tested `_impl` functions get connected to the real HSA API.

**Files:**
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp`
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp`
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_controller.cpp`

- [ ] **Step 1: Add HSA table integration functions to queue_intercept.hpp**

```cpp
// Called by QueueController to install/remove interposition in the live HSA API table.
// core_table: the HSA CoreApiTable to modify (original function pointers saved internally).
void
install_interposition(hsa_core_table_t* core_table);

void
remove_interposition(hsa_core_table_t* core_table);
```

- [ ] **Step 2: Implement install/remove_interposition**

Append to `queue_intercept.cpp`. This stores the original function pointers and replaces them with wrappers that call the `_impl` functions:

```cpp
namespace
{
// Saved original function pointers
decltype(hsa_core_table_s::hsa_queue_add_write_index_relaxed_fn)    orig_add_relaxed    = nullptr;
decltype(hsa_core_table_s::hsa_queue_add_write_index_scacq_screl_fn) orig_add_acqrel    = nullptr;
decltype(hsa_core_table_s::hsa_queue_add_write_index_scacquire_fn)  orig_add_acquire    = nullptr;
decltype(hsa_core_table_s::hsa_queue_add_write_index_screlease_fn)  orig_add_release    = nullptr;
decltype(hsa_core_table_s::hsa_queue_store_write_index_relaxed_fn)  orig_store_relaxed  = nullptr;
decltype(hsa_core_table_s::hsa_queue_store_write_index_screlease_fn) orig_store_release = nullptr;
decltype(hsa_core_table_s::hsa_queue_cas_write_index_relaxed_fn)    orig_cas_relaxed    = nullptr;
decltype(hsa_core_table_s::hsa_queue_cas_write_index_scacq_screl_fn) orig_cas_acqrel   = nullptr;
decltype(hsa_core_table_s::hsa_queue_cas_write_index_scacquire_fn)  orig_cas_acquire    = nullptr;
decltype(hsa_core_table_s::hsa_queue_cas_write_index_screlease_fn)  orig_cas_release    = nullptr;
decltype(hsa_core_table_s::hsa_queue_load_write_index_relaxed_fn)   orig_load_w_relaxed = nullptr;
decltype(hsa_core_table_s::hsa_queue_load_write_index_scacquire_fn) orig_load_w_acquire = nullptr;
decltype(hsa_core_table_s::hsa_signal_store_relaxed_fn)             orig_sig_relaxed    = nullptr;
decltype(hsa_core_table_s::hsa_signal_store_screlease_fn)           orig_sig_release    = nullptr;

// Wrapper template: delegates to _impl if queue is intercepted, else calls original
uint64_t wrap_add_write_index_relaxed(const hsa_queue_t* q, uint64_t v)
{
    auto* s = lookup_queue_state(q);
    if(s) return add_write_index_impl(s, v);
    return orig_add_relaxed(q, v);
}
// ... repeat for all 10 write-index variants and 2 load-write-index variants ...

void wrap_signal_store_relaxed(hsa_signal_t sig, hsa_signal_value_t val)
{
    auto* s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
            orig_sig_relaxed(db, v);
        });
        return;
    }
    orig_sig_relaxed(sig, val);
}

void wrap_signal_store_screlease(hsa_signal_t sig, hsa_signal_value_t val)
{
    auto* s = lookup_queue_state_by_doorbell(sig);
    if(s)
    {
        process_doorbell_impl(s, val, [](hsa_signal_t db, hsa_signal_value_t v) {
            orig_sig_release(db, v);
        });
        return;
    }
    orig_sig_release(sig, val);
}

}  // namespace

void
install_interposition(hsa_core_table_t* core_table)
{
    // Save originals
    orig_add_relaxed    = core_table->hsa_queue_add_write_index_relaxed_fn;
    orig_add_acqrel     = core_table->hsa_queue_add_write_index_scacq_screl_fn;
    orig_add_acquire    = core_table->hsa_queue_add_write_index_scacquire_fn;
    orig_add_release    = core_table->hsa_queue_add_write_index_screlease_fn;
    orig_store_relaxed  = core_table->hsa_queue_store_write_index_relaxed_fn;
    orig_store_release  = core_table->hsa_queue_store_write_index_screlease_fn;
    orig_cas_relaxed    = core_table->hsa_queue_cas_write_index_relaxed_fn;
    orig_cas_acqrel     = core_table->hsa_queue_cas_write_index_scacq_screl_fn;
    orig_cas_acquire    = core_table->hsa_queue_cas_write_index_scacquire_fn;
    orig_cas_release    = core_table->hsa_queue_cas_write_index_screlease_fn;
    orig_load_w_relaxed = core_table->hsa_queue_load_write_index_relaxed_fn;
    orig_load_w_acquire = core_table->hsa_queue_load_write_index_scacquire_fn;
    orig_sig_relaxed    = core_table->hsa_signal_store_relaxed_fn;
    orig_sig_release    = core_table->hsa_signal_store_screlease_fn;

    // Install wrappers
    core_table->hsa_queue_add_write_index_relaxed_fn    = wrap_add_write_index_relaxed;
    // ... set all 14 wrapper function pointers ...
    core_table->hsa_signal_store_relaxed_fn             = wrap_signal_store_relaxed;
    core_table->hsa_signal_store_screlease_fn           = wrap_signal_store_screlease;
}

void
remove_interposition(hsa_core_table_t* core_table)
{
    if(orig_add_relaxed) core_table->hsa_queue_add_write_index_relaxed_fn = orig_add_relaxed;
    // ... restore all 14 originals ...
    if(orig_sig_release) core_table->hsa_signal_store_screlease_fn = orig_sig_release;
}
```

**Note to implementer:** The comment `// ... repeat for all ...` above means you must write all 10 `wrap_add/store/cas/load_write_index_*` variants following the exact same pattern. Each one: check `lookup_queue_state(q)`, if found call the `_impl` variant, else call `orig_*`. The wrapper function signature must match the HSA API exactly. Check the field names in `hsa_core_table_s` (in the ROCm HSA headers).

- [ ] **Step 3: Wire into QueueController::init**

In `source/lib/rocprofiler-sdk/hsa/queue_controller.cpp`, find `QueueController::init()` and add:

```cpp
#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"

// At the end of QueueController::init(), after existing table setup:
queue_intercept::install_interposition(&_core_table);
```

- [ ] **Step 4: Wire queue creation into QueueController::add_queue**

In `QueueController::add_queue()`, after the existing queue insertion, add:

```cpp
// Get amd_queue_t pointers for WDID/RDID
auto* amd_q = reinterpret_cast<amd_queue_t*>(hsa_queue);
queue_intercept::create_queue_state(
    hsa_queue,
    &amd_q->write_dispatch_id,
    &amd_q->read_dispatch_id,
    /* k_factor from active services */ 0);
```

Include `amd_hsa_queue.h` at the top of queue_controller.cpp.

- [ ] **Step 5: Wire queue destruction into QueueController::destroy_queue**

In `QueueController::destroy_queue()`, before the existing queue removal:

```cpp
queue_intercept::destroy_queue_state(hsa_queue);
```

- [ ] **Step 6: Build**

```bash
cmake --build . -j$(nproc) 2>&1 | tail -20
```

Expected: Compiles successfully. No runtime test at this stage — integration testing comes in Task 8.

- [ ] **Step 7: Format and commit**

```bash
clang-format-11 -i source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
                   source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
                   source/lib/rocprofiler-sdk/hsa/queue_controller.cpp
git add source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
        source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
        source/lib/rocprofiler-sdk/hsa/queue_controller.cpp
git commit -m "feat(hsa): wire queue interposition into HSA API table via QueueController"
```

---

## Task 7: Metadata Queue Sync

**Purpose:** Add metadata queue pairing and sync logic. When a compute queue has a paired metadata queue, the doorbell handler writes corresponding metadata entries in lock-step.

**Files:**
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp`
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp`
- Modify: `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`

- [ ] **Step 1: Write failing tests**

Append to `source/lib/rocprofiler-sdk/tests/queue_intercept.cpp`:

```cpp
TEST(QueueIntercept, MetadataSyncWritesEntries)
{
    QueueState compute_state{};
    QueueState meta_state{};

    alignas(64) char compute_ring[64 * 256];
    alignas(256) char meta_ring[256 * 256];
    memset(compute_ring, 0, sizeof(compute_ring));
    memset(meta_ring, 0, sizeof(meta_ring));

    uint64_t compute_wdid = 0, compute_rdid = 0;
    uint64_t meta_wdid = 0, meta_rdid = 0;

    compute_state.ring_buf  = compute_ring;
    compute_state.ring_size = 256;
    compute_state.ring_mask = 255;
    compute_state.real_wdid = &compute_wdid;
    compute_state.real_rdid = &compute_rdid;
    compute_state.k_factor  = 7;
    compute_state.metadata_state = &meta_state;

    meta_state.ring_buf  = meta_ring;
    meta_state.ring_size = 256;
    meta_state.ring_mask = 255;
    meta_state.real_wdid = &meta_wdid;
    meta_state.real_rdid = &meta_rdid;

    // App wrote 1 dispatch packet
    compute_state.virtual_wptr.store(1);
    auto* pkt       = get_pkt(compute_ring, 0, 255);
    pkt->header     = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    pkt->kernel_object = 0x1234;

    process_doorbell_impl(&compute_state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    // Metadata queue should have advanced by 1 + K = 8
    EXPECT_EQ(meta_wdid, 8u);
    EXPECT_EQ(meta_state.next_submit_pos, 8u);
}

TEST(QueueIntercept, NoMetadataSyncWhenNoPairing)
{
    QueueState compute_state{};
    alignas(64) char ring[64 * 256];
    memset(ring, 0, sizeof(ring));
    uint64_t wdid = 0, rdid = 0;

    compute_state.ring_buf       = ring;
    compute_state.ring_size      = 256;
    compute_state.ring_mask      = 255;
    compute_state.real_wdid      = &wdid;
    compute_state.real_rdid      = &rdid;
    compute_state.k_factor       = 7;
    compute_state.metadata_state = nullptr;

    compute_state.virtual_wptr.store(1);
    get_pkt(ring, 0, 255)->header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);

    // Should not crash — metadata_state is null
    process_doorbell_impl(&compute_state, 0, [](hsa_signal_t, hsa_signal_value_t) {});
    EXPECT_EQ(wdid, 8u);
}
```

- [ ] **Step 2: Run test to verify behavior**

The metadata sync is already called in `process_doorbell_impl` (from Task 3) when `metadata_state != nullptr`, but the actual `sync_metadata` function isn't implemented yet. The test should fail because metadata_state pointers are dereferenced but no sync logic exists.

- [ ] **Step 3: Add sync_metadata to process_doorbell_impl**

In `process_doorbell_impl`, inside the per-packet loop, after the `memcpy`, add:

```cpp
if(state->metadata_state)
{
    sync_metadata_impl(state, src, dest_idx);
}
```

Add to `queue_intercept.hpp`:

```cpp
void
sync_metadata_impl(QueueState* compute_state,
                   const hsa_kernel_dispatch_packet_t* pkt,
                   uint64_t dest_pos);
```

Add to `queue_intercept.cpp`:

```cpp
void
sync_metadata_impl(QueueState* compute_state,
                   const hsa_kernel_dispatch_packet_t* /*pkt*/,
                   uint64_t /*dest_pos*/)
{
    auto* meta = compute_state->metadata_state;
    if(!meta) return;

    uint64_t meta_dest = meta->next_submit_pos;
    meta->next_submit_pos = meta_dest + 1 + compute_state->k_factor;
    __atomic_store_n(meta->real_wdid, meta->next_submit_pos, __ATOMIC_RELEASE);
}
```

- [ ] **Step 4: Build and run tests**

```bash
cmake --build . --target rocprofiler-sdk-lib-tests -j$(nproc)
./rocprofiler-sdk-lib-tests --gtest_filter="QueueIntercept.*"
```

Expected: All tests PASS (previous 15 + new 2 = 17 total).

- [ ] **Step 5: Format and commit**

```bash
clang-format-11 -i source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
                   source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
                   source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
git add source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
        source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
        source/lib/rocprofiler-sdk/tests/queue_intercept.cpp
git commit -m "feat(hsa): add metadata queue sync in doorbell handler"
```

---

## Task 8: Integration Test — Trace-Only Smoke Test

**Purpose:** Verify the SDK-level interposition works end-to-end with a real GPU. Run a simple HIP program with kernel tracing enabled. This validates that the deferred WDID mechanism works, packets are delivered to the GPU, and the trace data is captured.

**Files:**
- No new files — uses existing test infrastructure

- [ ] **Step 1: Build everything**

Build HSA runtime (unmodified develop branch), HIP, and rocprofiler-sdk with the new code:

```bash
cd /path/to/build && cmake --build . -j$(nproc)
```

- [ ] **Step 2: Run vectorAdd with kernel trace**

```bash
rocprofv3 --kernel-trace -- ./tests/bin/vector-ops/vector-ops
```

Expected: Program completes successfully, trace output shows kernel dispatch records with timestamps.

- [ ] **Step 3: Run vectorAdd with HSA API trace**

```bash
rocprofv3 --hsa-trace -- ./tests/bin/vector-ops/vector-ops
```

Expected: Program completes, HSA API trace shows `hsa_queue_add_write_index_relaxed` calls (now going through our interposition).

- [ ] **Step 4: Run a subset of the test suite**

```bash
ctest --test-dir . --output-on-failure -j1 -R "rocprofv3-test-kernel-trace"
```

Expected: kernel-trace tests pass.

- [ ] **Step 5: Document results**

Update `~/ai/task_info/SHADOW_WPTR_SESSION.md` with integration test results.

---

## Task 9: Integration — Counter Collection (K=7)

**Purpose:** Wire the counter collection path into the new interposition. The existing `WriteInterceptor` in `queue.cpp` constructs the 7-packet instrumentation sequence. This task connects that logic to the new doorbell handler so that `process_doorbell_impl` invokes the interceptor callback chain and fills the K instrumentation slots.

**This task requires detailed exploration of the existing `WriteInterceptor` callback chain in `queue.cpp` (lines 296-706) and careful porting of the packet construction logic. The exact implementation depends on which parts of the existing `WriteInterceptor` can be reused vs. must be refactored.**

**Files:**
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp`
- Modify: `source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp`
- Modify: `source/lib/rocprofiler-sdk/hsa/queue.cpp` (bypass old WriteInterceptor path)

- [ ] **Step 1: Add interceptor callback type to QueueState**

Add to `queue_intercept.hpp`:

```cpp
using intercept_callback_t = std::function<void(
    QueueState*                         state,
    const hsa_kernel_dispatch_packet_t* app_pkt,
    uint64_t                            submit_base,
    void*                               ring_buf,
    uint32_t                            ring_mask)>;
```

Add `interceptors` field to `QueueState`:

```cpp
std::vector<intercept_callback_t> interceptors;
```

- [ ] **Step 2: Invoke interceptors in process_doorbell_impl**

After copying the app packet, invoke all registered interceptors:

```cpp
for(auto& cb : state->interceptors)
{
    cb(state, src, dest_idx, state->ring_buf, state->ring_mask);
}
```

- [ ] **Step 3: Port counter collection packet writing**

This is the largest step. The existing `WriteInterceptor` in `queue.cpp` lines 410-706 builds the 7-packet sequence. Port the packet construction into an interceptor callback that writes directly into the K slots:

```cpp
auto counter_interceptor = [](QueueState* state,
                              const hsa_kernel_dispatch_packet_t* app_pkt,
                              uint64_t submit_base,
                              void* ring_buf,
                              uint32_t ring_mask) {
    // Write barriers at submit_base + 0, +1 (serializer gates)
    // Write PM4 IB at submit_base + 2 (counter start)
    // App packet is already at submit_base + 3 (or wherever it was copied)
    // Write PM4 IBs at submit_base + 4, +5 (counter read, stop)
    // Write barrier at submit_base + 6 (completion)
};
```

**Important:** The exact packet layout depends on the existing `CounterAQLPacket` and `profile_serializer` code. The implementer should read:
- `source/lib/rocprofiler-sdk/hsa/queue.cpp:410-706` (WriteInterceptor)
- `source/lib/rocprofiler-sdk/aql/packet_construct.cpp` (CounterPacketConstruct)
- `source/lib/rocprofiler-sdk/hsa/profile_serializer.cpp`

- [ ] **Step 4: Update queue creation to set K based on active contexts**

In `queue_controller.cpp`, when calling `create_queue_state`, determine K:

```cpp
uint64_t k = 0;
// Check if any callback has counter collection enabled
_callback_cache.rlock([&](const auto& map) {
    for(const auto& [id, tuple] : map)
    {
        const auto& [agent, cbs] = tuple;
        if(cbs.write_interceptor) k = 7;
    }
});
queue_intercept::create_queue_state(hsa_queue, &amd_q->write_dispatch_id,
                                     &amd_q->read_dispatch_id, k);
```

- [ ] **Step 5: Test with counter collection**

```bash
rocprofv3 --pmc SQ_WAVES -- ./tests/bin/vector-ops/vector-ops
```

Expected: Counter values collected, program exits 0.

- [ ] **Step 6: Run counter collection tests**

```bash
ctest --test-dir . --output-on-failure -j1 -R "counter-collection"
```

- [ ] **Step 7: Format and commit**

```bash
clang-format-11 -i source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
                   source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
                   source/lib/rocprofiler-sdk/hsa/queue.cpp \
                   source/lib/rocprofiler-sdk/hsa/queue_controller.cpp
git add source/lib/rocprofiler-sdk/hsa/queue_intercept.hpp \
        source/lib/rocprofiler-sdk/hsa/queue_intercept.cpp \
        source/lib/rocprofiler-sdk/hsa/queue.cpp \
        source/lib/rocprofiler-sdk/hsa/queue_controller.cpp
git commit -m "feat(hsa): wire counter collection into SDK-level queue interposition"
```

---

## Task 10: Full Test Suite Validation

**Purpose:** Run the complete rocprofiler-sdk test suite and compare results against the develop branch baseline.

- [ ] **Step 1: Run full suite on new branch**

```bash
cd build && ctest --output-on-failure -j1 2>&1 | tee ~/ai/sdk-shadow-wptr-test-results.txt
```

- [ ] **Step 2: Count pass/fail**

```bash
grep -c "Test #" ~/ai/sdk-shadow-wptr-test-results.txt
grep -c "Passed" ~/ai/sdk-shadow-wptr-test-results.txt
grep -c "Failed" ~/ai/sdk-shadow-wptr-test-results.txt
```

- [ ] **Step 3: Compare against develop baseline**

Build and test develop branch for comparison. Document any regressions introduced by the new code vs. pre-existing failures.

- [ ] **Step 4: Document results**

Write a summary to `~/ai/sdk-shadow-wptr-test-summary.md` with:
- Total tests, pass/fail counts
- List of new regressions (if any)
- List of pre-existing failures
- Comparison to HSA shadow_wptr branch (66 failures)

---

## Dependency Graph

```
Task 0 (HW experiment) — independent, can run in parallel with Tasks 1-5

Task 1 (QueueState + Registry)
  └── Task 2 (Write-Index Functions)
       └── Task 3 (Doorbell K=0)
            └── Task 4 (Doorbell K>0 tests)
            └── Task 7 (Metadata Sync)
       └── Task 5 (Queue Create/Destroy)
            └── Task 6 (HSA Table Integration)
                 └── Task 8 (Integration: Trace)
                      └── Task 9 (Integration: Counters)
                           └── Task 10 (Full Suite)
```
