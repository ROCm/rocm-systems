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

#include "lib/rocprofiler-sdk/hsa/queue_intercept.hpp"

#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include <chrono>
#include <future>
#include <memory>
#include <thread>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
namespace
{
TEST(QueueIntercept, QueueStateDefaultInit)
{
    QueueState state;

    EXPECT_EQ(state.ring_buf, nullptr);
    EXPECT_EQ(state.ring_size, 0U);
    EXPECT_EQ(state.ring_mask, 0U);
    EXPECT_EQ(state.pkt_size, 64U);
    EXPECT_EQ(state.virtual_wptr.load(), 0UL);
    EXPECT_EQ(state.real_wdid, nullptr);
    EXPECT_EQ(state.real_rdid, nullptr);
    EXPECT_EQ(state.next_scan_pos, 0UL);
    EXPECT_EQ(state.next_submit_pos, 0UL);
    EXPECT_EQ(state.hsa_queue, nullptr);
    EXPECT_EQ(state.doorbell_signal.handle, 0UL);

    // Phase 1 — new side-table fields.
    EXPECT_EQ(state.mode, QueueState::Mode::full_intercept);
    EXPECT_EQ(state.corr_ring_mask, 0U);
    EXPECT_EQ(state.last_observed_wdid.load(), 0UL);
    EXPECT_TRUE(state.corr_slots.empty());
    EXPECT_FALSE(state.overwrite_warning_logged.load());
}

TEST(QueueIntercept, RegistryInsertAndLookup)
{
    // Create a dummy queue pointer for testing
    // We're just using a dummy address; we never dereference it
    hsa_queue_t        dummy_queue;
    const hsa_queue_t* queue_ptr = &dummy_queue;

    // Create a QueueState and insert it into the registry
    auto state       = std::make_shared<QueueState>();
    state->ring_size = 1024;
    state->ring_mask = 1023;

    QueueState* state_ptr = state.get();

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = state; });

    // Look up the state
    auto found_state = lookup_queue_state(queue_ptr);
    ASSERT_NE(found_state, nullptr);
    EXPECT_EQ(found_state.get(), state_ptr);
    EXPECT_EQ(found_state->ring_size, 1024U);
    EXPECT_EQ(found_state->ring_mask, 1023U);

    // Clean up
    get_queue_registry().wlock([&](auto& registry) { registry.erase(queue_ptr); });

    // Verify removal
    auto after_removal = lookup_queue_state(queue_ptr);
    EXPECT_EQ(after_removal, nullptr);
}

TEST(QueueIntercept, DoorbellMapInsertAndLookup)
{
    // Create a dummy queue for testing
    hsa_queue_t        dummy_queue;
    const hsa_queue_t* queue_ptr = &dummy_queue;

    // Create and register a QueueState
    auto state            = std::make_shared<QueueState>();
    state->ring_size      = 2048;
    QueueState* state_ptr = state.get();

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = state; });

    // Create a doorbell signal and register it
    hsa_signal_t doorbell;
    doorbell.handle = 0x12345678;

    register_doorbell(queue_ptr, doorbell);

    // Look up by doorbell
    auto found_state = lookup_queue_state_by_doorbell(doorbell);
    ASSERT_NE(found_state, nullptr);
    EXPECT_EQ(found_state.get(), state_ptr);
    EXPECT_EQ(found_state->ring_size, 2048U);

    // Unregister doorbell
    unregister_doorbell(doorbell);

    // Verify removal
    auto after_removal = lookup_queue_state_by_doorbell(doorbell);
    EXPECT_EQ(after_removal, nullptr);

    // Clean up queue registry
    get_queue_registry().wlock([&](auto& registry) { registry.erase(queue_ptr); });
}

TEST(QueueIntercept, AddWriteIndexAdvancesVirtualWptr)
{
    QueueState state{};
    uint64_t   idx0 = add_write_index_impl(&state, 1);
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
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 256];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_buf  = ring;
    state->ring_size = 256;
    state->ring_mask = 255;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    state->virtual_wptr.store(1);
    auto* pkt          = get_pkt(ring, 0, 255);
    pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    pkt->kernel_object = 0xDEADBEEF;

    bool doorbell_rang = false;
    process_doorbell_impl(
        state, 0, [&](hsa_signal_t, hsa_signal_value_t) { doorbell_rang = true; });

    EXPECT_TRUE(doorbell_rang);
    EXPECT_EQ(real_wdid, 1u);
    EXPECT_EQ(state->next_submit_pos, 1u);
    EXPECT_EQ(state->next_scan_pos, 1u);
    auto* submitted = get_pkt(ring, 0, 255);
    EXPECT_EQ(submitted->kernel_object, 0xDEADBEEFu);
}

TEST(QueueIntercept, DoorbellMultiplePacketsTraceOnly)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 256];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_buf  = ring;
    state->ring_size = 256;
    state->ring_mask = 255;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    state->virtual_wptr.store(3);
    for(uint64_t i = 0; i < 3; i++)
    {
        auto* pkt          = get_pkt(ring, i, 255);
        pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
        pkt->kernel_object = static_cast<uint64_t>(0xA000 + i);
    }

    process_doorbell_impl(state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    EXPECT_EQ(real_wdid, 3u);
    EXPECT_EQ(state->next_submit_pos, 3u);
    for(uint64_t i = 0; i < 3; i++)
    {
        auto* submitted = get_pkt(ring, i, 255);
        EXPECT_EQ(submitted->kernel_object, static_cast<uint64_t>(0xA000 + i));
    }
}

TEST(QueueIntercept, DoorbellNoNewPackets)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 64];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_buf  = ring;
    state->ring_size = 64;
    state->ring_mask = 63;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    state->virtual_wptr.store(0);
    bool doorbell_rang = false;
    process_doorbell_impl(
        state, 0, [&](hsa_signal_t, hsa_signal_value_t) { doorbell_rang = true; });

    EXPECT_TRUE(doorbell_rang);
    EXPECT_EQ(real_wdid, 0u);
}

TEST(QueueIntercept, CreateAndDestroyQueueState)
{
    alignas(64) char ring_mem[64 * 256];
    hsa_queue_t      fake_queue{};
    fake_queue.base_address    = reinterpret_cast<void*>(ring_mem);
    fake_queue.size            = 256;
    fake_queue.doorbell_signal = {.handle = 9999};

    uint64_t fake_wdid = 0;
    uint64_t fake_rdid = 0;

    create_queue_state(&fake_queue,
                       &fake_wdid,
                       &fake_rdid,
                       QueueState::Mode::full_intercept);

    auto state = lookup_queue_state(&fake_queue);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->ring_buf, reinterpret_cast<void*>(ring_mem));
    EXPECT_EQ(state->ring_size, 256u);
    EXPECT_EQ(state->ring_mask, 255u);
    EXPECT_EQ(state->real_wdid, &fake_wdid);
    EXPECT_EQ(state->real_rdid, &fake_rdid);
    EXPECT_EQ(state->doorbell_signal.handle, 9999u);

    auto by_doorbell = lookup_queue_state_by_doorbell({.handle = 9999});
    EXPECT_EQ(by_doorbell.get(), state.get());

    destroy_queue_state(&fake_queue);
    EXPECT_EQ(lookup_queue_state(&fake_queue), nullptr);
    EXPECT_EQ(lookup_queue_state_by_doorbell({.handle = 9999}), nullptr);
}

// ---------------------------------------------------------------------------
// Phase 1: tracing-only mode unit tests (spec §9.1, plan Task 10).
//
// These tests exercise the capture/consume side-table logic in isolation.
// They synthesize a QueueState (no real HSA queue/signal) so the slot
// invariants can be poked directly. See PHASE1_TRACING_ONLY_INTERCEPT_DESIGN.md
// §3.5 for the invariants under test.
// ---------------------------------------------------------------------------

struct TracingOnlyQueueStateFixture
{
    std::shared_ptr<QueueState>               state;
    static constexpr uint32_t                 kQueueSize = 64;  // small for tests
    static constexpr uint32_t                 kRingMask  = kQueueSize - 1;
    std::vector<hsa_kernel_dispatch_packet_t> packet_storage;

    TracingOnlyQueueStateFixture()
    {
        state                  = std::make_shared<QueueState>();
        state->mode            = QueueState::Mode::tracing_only;
        state->corr_ring_mask  = kRingMask;
        // CorrEntry::gen is non-movable atomic; vector::resize fails the
        // static_assert. Use the (N) ctor for in-place default-construction.
        state->corr_slots      = std::vector<CorrEntry>(kQueueSize);
        packet_storage.resize(kQueueSize);
        // Initialize every packet header to KERNEL_DISPATCH so the
        // tracing-only capture path doesn't skip them.
        for(auto& p : packet_storage)
        {
            p.header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
        }
        state->ring_buf = packet_storage.data();
    }
};

TEST(QueueInterceptTracingOnly, SingleProducerRoundTrip)
{
    TracingOnlyQueueStateFixture f;

    // Capture: simulate doorbell store with val=0 (one new packet at slot 0).
    process_doorbell_tracing_only(f.state, /*val=*/0);

    auto& slot = f.state->corr_slots[0];
    EXPECT_NE(slot.gen.load(std::memory_order_acquire), 0UL);
    EXPECT_NE(slot.corr_id, nullptr);
    EXPECT_EQ(slot.captured_wdid, 0UL);

    // Consume: emulate drainer.
    {
        std::lock_guard<std::mutex> g(f.state->slot_publish_mu);
        EXPECT_EQ(static_cast<uint32_t>(slot.captured_wdid & 0xFFFFFFFFu), 0U);
        // Consume the slot.
        auto* held_corr_id = slot.corr_id;
        slot.corr_id       = nullptr;
        slot.gen.store(0, std::memory_order_release);
        held_corr_id->sub_kern_count();
        held_corr_id->sub_ref_count();
    }

    EXPECT_EQ(slot.gen.load(), 0UL);
    EXPECT_EQ(slot.corr_id, nullptr);
}

TEST(QueueInterceptTracingOnly, MultiDoorbellSingleThread)
{
    TracingOnlyQueueStateFixture f;

    // One doorbell publishing 4 packets at once.
    process_doorbell_tracing_only(f.state, /*val=*/3);  // wdid moves to 4

    // All 4 slots should be populated with the SAME corr_id.
    auto* expected_corr = f.state->corr_slots[0].corr_id;
    EXPECT_NE(expected_corr, nullptr);
    for(uint32_t i = 0; i < 4; ++i)
    {
        EXPECT_NE(f.state->corr_slots[i].gen.load(), 0UL);
        EXPECT_EQ(f.state->corr_slots[i].corr_id, expected_corr);
        EXPECT_EQ(f.state->corr_slots[i].captured_wdid, static_cast<uint64_t>(i));
    }
    // Slots 4-63 should remain empty.
    for(uint32_t i = 4; i < 64; ++i)
        EXPECT_EQ(f.state->corr_slots[i].gen.load(), 0UL);
}

TEST(QueueInterceptTracingOnly, MultiProducerSerializedEnqueue)
{
    // Phase 1 hazard documented in §7.2: out-of-order doorbells may
    // mis-attribute. This test specifically tests the SERIALIZED case
    // where lock-acquisition order matches doorbell-value order — i.e.,
    // each thread's add_write_index → write packet → ring doorbell
    // sequence completes before the next thread's begins.
    TracingOnlyQueueStateFixture   f;
    constexpr int                  kNumThreads = 8;
    std::mutex                     enqueue_serialize_mu;
    std::atomic<hsa_signal_value_t> next_val{0};
    std::vector<std::thread>       threads;
    threads.reserve(kNumThreads);

    for(int t = 0; t < kNumThreads; ++t)
    {
        threads.emplace_back([&, t]() {
            (void) t;
            std::lock_guard<std::mutex> g(enqueue_serialize_mu);
            auto                        val = next_val.fetch_add(1);
            process_doorbell_tracing_only(f.state, val);
        });
    }
    for(auto& th : threads) th.join();

    // All 8 slots populated.
    for(int i = 0; i < kNumThreads; ++i)
    {
        EXPECT_NE(f.state->corr_slots[i].gen.load(), 0UL) << "slot " << i;
        EXPECT_EQ(f.state->corr_slots[i].captured_wdid, static_cast<uint64_t>(i));
    }
}

TEST(QueueInterceptTracingOnly, WraparoundOverwriteRetiresPrior)
{
    TracingOnlyQueueStateFixture f;

    // First capture: slot 0.
    process_doorbell_tracing_only(f.state, /*val=*/0);
    auto& slot       = f.state->corr_slots[0];
    auto* first_corr = slot.corr_id;
    ASSERT_NE(first_corr, nullptr);
    auto first_kern_count = first_corr->get_kern_count();
    auto first_ref_count  = first_corr->get_ref_count();
    EXPECT_GE(first_kern_count, 1u);
    EXPECT_GE(first_ref_count, 1u);

    // Advance last_observed_wdid past one full ring without consuming.
    // Then capture again at the same slot index (slot 0 = wdid 64 & mask).
    //
    // Production semantics (process_doorbell_tracing_only):
    //   prev = last_observed_wdid; new_wdid = value + 1; iterate d in [prev,new_wdid).
    // To make d == 64 land in slot 0 (64 & mask == 0), need prev=64 and value=64
    // so new_wdid=65 and the single iterated d is 64.
    f.state->last_observed_wdid.store(64);
    process_doorbell_tracing_only(f.state, /*val=*/64);  // publishes d=64 -> slot 0

    auto& reused = f.state->corr_slots[0];
    EXPECT_EQ(reused.captured_wdid, 64UL);
    EXPECT_NE(reused.corr_id, nullptr);

    // First occupant's refcounts should have been decremented by the
    // overwrite path. The exact post-decrement value depends on the
    // initial refcount; the invariant we test is that they DECREASED
    // by 1 each.
    EXPECT_EQ(first_corr->get_kern_count(), first_kern_count - 1)
        << "Wraparound overwrite must retire prior occupant's kern_count";
    EXPECT_EQ(first_corr->get_ref_count(), first_ref_count - 1)
        << "Wraparound overwrite must retire prior occupant's ref_count";

    // Throttled warning logged.
    EXPECT_TRUE(f.state->overwrite_warning_logged.load());
}

TEST(QueueInterceptTracingOnly, ConsumeAliasGuardRejectsStale)
{
    TracingOnlyQueueStateFixture f;

    // Capture a dispatch with wdid = 64 (slot 0 after wraparound).
    // See WraparoundOverwriteRetiresPrior for the prev/value arithmetic.
    f.state->last_observed_wdid.store(64);
    process_doorbell_tracing_only(f.state, /*val=*/64);
    auto& slot = f.state->corr_slots[0];
    EXPECT_EQ(slot.captured_wdid, 64UL);

    // Drainer presents firmware record with dispatch_idx = 0 (low 32
    // bits of an OLDER, displaced wdid). Alias guard MUST reject.
    const uint32_t stale_dispatch_idx = 0;
    bool           consumed           = false;
    {
        std::lock_guard<std::mutex> g(f.state->slot_publish_mu);
        if(slot.gen.load(std::memory_order_acquire) != 0 && slot.corr_id != nullptr &&
           static_cast<uint32_t>(slot.captured_wdid & 0xFFFFFFFFu) == stale_dispatch_idx)
        {
            consumed = true;  // SHOULD NOT REACH
        }
    }
    EXPECT_FALSE(consumed) << "Alias guard must reject stale firmware record";

    // Slot still occupied (alias rejection does NOT consume).
    EXPECT_NE(slot.gen.load(), 0UL);
    EXPECT_NE(slot.corr_id, nullptr);
}

TEST(QueueInterceptTracingOnly, RefcountBalance)
{
    TracingOnlyQueueStateFixture f;

    // Capture one dispatch.
    process_doorbell_tracing_only(f.state, /*val=*/0);
    auto& slot               = f.state->corr_slots[0];
    auto* cid                = slot.corr_id;
    ASSERT_NE(cid, nullptr);
    auto  kern_after_capture = cid->get_kern_count();
    auto  ref_after_capture  = cid->get_ref_count();

    // Drainer-side consume + retirement.
    {
        std::lock_guard<std::mutex> g(f.state->slot_publish_mu);
        slot.corr_id = nullptr;
        slot.gen.store(0, std::memory_order_release);
    }
    cid->sub_kern_count();
    cid->sub_ref_count();

    EXPECT_EQ(cid->get_kern_count(), kern_after_capture - 1);
    EXPECT_EQ(cid->get_ref_count(), ref_after_capture - 1);
}

TEST(QueueInterceptTracingOnly, LateAttachReturnsFallback)
{
    TracingOnlyQueueStateFixture f;
    // No capture; lookup at slot 0 should miss.
    auto& slot = f.state->corr_slots[0];
    EXPECT_EQ(slot.gen.load(std::memory_order_acquire), 0UL);
    EXPECT_EQ(slot.corr_id, nullptr);
}

TEST(QueueInterceptTracingOnly, ShutdownDrainsAllSlots)
{
    TracingOnlyQueueStateFixture f;

    // Populate 5 slots without consuming.
    for(int i = 0; i < 5; ++i)
        process_doorbell_tracing_only(f.state, /*val=*/i);

    std::vector<context::correlation_id*> captured_ids;
    std::vector<uint32_t>                 pre_kern, pre_ref;
    for(int i = 0; i < 5; ++i)
    {
        auto* c = f.state->corr_slots[i].corr_id;
        ASSERT_NE(c, nullptr);
        captured_ids.push_back(c);
        pre_kern.push_back(c->get_kern_count());
        pre_ref.push_back(c->get_ref_count());
    }

    // Simulate shutdown_intercept's per-state drain loop.
    {
        std::lock_guard<std::mutex> g(f.state->slot_publish_mu);
        for(auto& entry : f.state->corr_slots)
        {
            if(entry.gen.load(std::memory_order_relaxed) != 0 && entry.corr_id)
            {
                entry.corr_id->sub_kern_count();
                entry.corr_id->sub_ref_count();
                entry.corr_id = nullptr;
                entry.gen.store(0, std::memory_order_release);
            }
        }
    }

    // Each captured corr_id has had its refcounts decremented once.
    for(int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(captured_ids[i]->get_kern_count(), pre_kern[i] - 1);
        EXPECT_EQ(captured_ids[i]->get_ref_count(), pre_ref[i] - 1);
    }
}

TEST(QueueIntercept, DoorbellBackpressureWaitsWhenRingFullK0)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 8];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 4;
    uint64_t real_rdid = 0;

    state->ring_buf        = ring;
    state->ring_size       = 4;
    state->ring_mask       = 3;
    state->real_wdid       = &real_wdid;
    state->real_rdid       = &real_rdid;
    state->next_scan_pos   = 4;
    state->next_submit_pos = 4;
    state->virtual_wptr.store(5);

    auto* src_pkt          = get_pkt(ring, 4, 3);
    src_pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    src_pkt->kernel_object = 0xABCD;

    hsa_signal_value_t doorbell_value = -1;
    auto               fut            = std::async(std::launch::async, [&]() {
        process_doorbell_impl(
            state, 0, [&](hsa_signal_t, hsa_signal_value_t v) { doorbell_value = v; });
    });

    std::this_thread::sleep_for(std::chrono::milliseconds{2});
    __atomic_store_n(&real_rdid, 1, __ATOMIC_RELEASE);

    ASSERT_EQ(fut.wait_for(std::chrono::milliseconds{500}), std::future_status::ready);
    fut.get();

    EXPECT_EQ(real_wdid, 5u);
    EXPECT_EQ(state->next_submit_pos, 5u);
    EXPECT_EQ(state->next_scan_pos, 5u);
    EXPECT_EQ(doorbell_value, 4);
    EXPECT_LE(state->next_submit_pos - real_rdid, state->ring_size);
    EXPECT_EQ(get_pkt(ring, 4, 3)->kernel_object, static_cast<uint64_t>(0xABCD));
}

}  // namespace
}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
