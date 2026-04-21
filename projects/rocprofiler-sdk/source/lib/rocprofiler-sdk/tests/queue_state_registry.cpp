// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/queue_state_registry.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"

#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
namespace
{
// Test fixture that clears the registry + doorbell map on setup and teardown
// so tests do not observe state leaked from other tests that exercise the
// same globals (most notably tests/queue_intercept.cpp).
class QueueStateRegistryTest : public ::testing::Test
{
protected:
    void SetUp() override { clear_all(); }
    void TearDown() override { clear_all(); }

private:
    static void clear_all()
    {
        get_queue_registry().wlock([](auto& m) { m.clear(); });
        get_doorbell_map().wlock([](auto& m) { m.clear(); });
    }
};

// Build a dummy hsa_queue_t. The registry only stores the pointer, but
// create_queue_state reads base_address/size/doorbell_signal off of it, so
// populate those with plausible values.
struct FakeQueue
{
    alignas(64) char ring_mem[64 * 64]{};
    hsa_queue_t queue{};
    uint64_t    wdid = 0;
    uint64_t    rdid = 0;

    explicit FakeQueue(uint64_t doorbell_handle)
    {
        queue.base_address           = ring_mem;
        queue.size                   = 64;
        queue.doorbell_signal.handle = doorbell_handle;
    }
};

TEST_F(QueueStateRegistryTest, CreateQueueStateRegisters)
{
    FakeQueue fq{0x1001};

    create_queue_state(&fq.queue, &fq.wdid, &fq.rdid, /*k_factor=*/0);

    auto state = lookup_queue_state(&fq.queue);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->hsa_queue, &fq.queue);
    EXPECT_EQ(state->ring_view.buf, static_cast<void*>(fq.ring_mem));
    EXPECT_EQ(state->ring_view.size, 64u);
    EXPECT_EQ(state->ring_view.mask, 63u);
    EXPECT_EQ(state->real_wdid, &fq.wdid);
    EXPECT_EQ(state->real_rdid, &fq.rdid);
    EXPECT_EQ(state->stride, 1u);
    EXPECT_EQ(state->doorbell_signal.handle, 0x1001u);
}

TEST_F(QueueStateRegistryTest, DestroyQueueStateUnregisters)
{
    FakeQueue fq{0x1002};

    create_queue_state(&fq.queue, &fq.wdid, &fq.rdid, /*k_factor=*/0);
    ASSERT_NE(lookup_queue_state(&fq.queue), nullptr);

    destroy_queue_state(&fq.queue);
    EXPECT_EQ(lookup_queue_state(&fq.queue), nullptr);
}

TEST_F(QueueStateRegistryTest, DoorbellLookupFindsState)
{
    FakeQueue fq{0x1003};

    // create_queue_state already registers the doorbell; verify that path.
    create_queue_state(&fq.queue, &fq.wdid, &fq.rdid, /*k_factor=*/0);

    auto by_queue = lookup_queue_state(&fq.queue);
    ASSERT_NE(by_queue, nullptr);

    auto by_doorbell = lookup_queue_state_by_doorbell(fq.queue.doorbell_signal);
    ASSERT_NE(by_doorbell, nullptr);
    EXPECT_EQ(by_doorbell.get(), by_queue.get());
}

TEST_F(QueueStateRegistryTest, UnregisterDoorbellRemovesMapping)
{
    FakeQueue fq{0x1004};

    create_queue_state(&fq.queue, &fq.wdid, &fq.rdid, /*k_factor=*/0);
    ASSERT_NE(lookup_queue_state_by_doorbell(fq.queue.doorbell_signal), nullptr);

    EXPECT_TRUE(unregister_doorbell(fq.queue.doorbell_signal));
    EXPECT_FALSE(unregister_doorbell(fq.queue.doorbell_signal));
    EXPECT_EQ(lookup_queue_state_by_doorbell(fq.queue.doorbell_signal), nullptr);
}

TEST_F(QueueStateRegistryTest, DoorbellMapHoldsWeakRef)
{
    // Register doorbell via the full lifecycle so the doorbell map picks up
    // the weak_ptr. Then destroy the queue-state entry (removing the strong
    // reference held by the registry) WITHOUT clearing the doorbell map,
    // and confirm the weak_ptr expired and lookup_queue_state_by_doorbell
    // returns empty.
    FakeQueue    fq{0x1005};
    hsa_signal_t saved_doorbell = {};

    create_queue_state(&fq.queue, &fq.wdid, &fq.rdid, /*k_factor=*/0);
    saved_doorbell = fq.queue.doorbell_signal;

    // Drop the registry entry directly so we do NOT also erase the doorbell
    // map (destroy_queue_state would). This proves the map's weak_ptr would
    // correctly expire on its own if the strong owner went away.
    get_queue_registry().wlock([&](auto& m) { m.erase(&fq.queue); });

    // Doorbell map still has an entry, but it points to an expired weak_ptr.
    bool has_raw_entry = false;
    get_doorbell_map().rlock([&](const auto& m) {
        has_raw_entry = (m.find(saved_doorbell.handle) != m.end());
    });
    EXPECT_TRUE(has_raw_entry);

    EXPECT_EQ(lookup_queue_state_by_doorbell(saved_doorbell), nullptr);
}

// Step 4' regression test for bug #5: destroy_queue_state must not return
// while a concurrent process_doorbell_impl call is still scanning this queue.
// The drain is implemented by taking+releasing state->gate_lock after erasing
// from the registry/doorbell maps. We simulate an in-flight scan by having a
// worker thread acquire gate_lock directly and hold it for ~150ms while the
// main thread calls destroy_queue_state; destroy must block until the worker
// releases the lock.
TEST_F(QueueStateRegistryTest, DestroyDrainsInFlightScan)
{
    FakeQueue fq{0x1006};

    create_queue_state(&fq.queue, &fq.wdid, &fq.rdid, /*k_factor=*/0);

    auto state = lookup_queue_state(&fq.queue);
    ASSERT_NE(state, nullptr);

    std::atomic<bool> worker_has_lock{false};
    constexpr auto    kHoldDuration = std::chrono::milliseconds(150);

    std::thread worker([&] {
        std::lock_guard<std::mutex> guard{state->gate_lock};
        worker_has_lock.store(true, std::memory_order_release);
        std::this_thread::sleep_for(kHoldDuration);
        // gate_lock released here; main's destroy_queue_state should unblock.
    });

    // Wait for the worker to actually be holding the lock before we attempt
    // to destroy, so we deterministically observe the drain path.
    while(!worker_has_lock.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }

    auto t0 = std::chrono::steady_clock::now();
    destroy_queue_state(&fq.queue);
    auto elapsed = std::chrono::steady_clock::now() - t0;

    worker.join();

    // destroy_queue_state must have blocked for most of the worker's hold
    // duration. Use a 100ms lower bound (leaves ~50ms slack on kHoldDuration
    // to tolerate scheduler jitter).
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    EXPECT_GE(elapsed_ms, 100)
        << "destroy_queue_state returned in " << elapsed_ms
        << "ms; expected to block until the in-flight scan released gate_lock";

    // After destroy completes, the state must be unreachable via both maps.
    EXPECT_EQ(lookup_queue_state(&fq.queue), nullptr);
    EXPECT_EQ(lookup_queue_state_by_doorbell(fq.queue.doorbell_signal), nullptr);
}

}  // namespace
}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
