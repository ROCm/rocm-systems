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
#include "lib/rocprofiler-sdk/hsa/queue_state_registry.hpp"

#include <gtest/gtest.h>
#include <hsa/hsa.h>

#include <memory>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
namespace
{
// Test helper — directly populate the pending-claim map without going through
// add_write_index_impl. Unit tests prefer this form because they want to
// assert on specific claim shapes without reasoning about fetch_add races.
void
record_claim(QueueState& s, uint64_t claim_index, uint32_t packet_count, uint32_t stride)
{
    std::lock_guard<std::mutex> lk{s.pending_lock};
    s.pending.emplace(claim_index, PendingClaim{claim_index, packet_count, stride});
}

TEST(QueueIntercept, QueueStateDefaultInit)
{
    QueueState state;

    EXPECT_EQ(state.ring_view.buf, nullptr);
    EXPECT_EQ(state.ring_view.size, 0U);
    EXPECT_EQ(state.ring_view.mask, 0U);
    EXPECT_EQ(state.ring_view.pkt_size, 64U);
    EXPECT_EQ(state.stride, 1U);
    EXPECT_EQ(state.claim_pos.load(), 0UL);
    EXPECT_EQ(state.real_wdid, nullptr);
    EXPECT_EQ(state.real_rdid, nullptr);
    EXPECT_EQ(state.published_pos, 0UL);
    EXPECT_EQ(state.hsa_queue, nullptr);
    EXPECT_EQ(state.doorbell_signal.handle, 0UL);
    EXPECT_TRUE(state.pending.empty());
}

TEST(QueueIntercept, RegistryInsertAndLookup)
{
    // Create a dummy queue pointer for testing
    // We're just using a dummy address; we never dereference it
    hsa_queue_t        dummy_queue;
    const hsa_queue_t* queue_ptr = &dummy_queue;

    // Create a QueueState and insert it into the registry
    auto state            = std::make_shared<QueueState>();
    state->ring_view.size = 1024;
    state->ring_view.mask = 1023;

    QueueState* state_ptr = state.get();

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = state; });

    // Look up the state
    auto found_state = lookup_queue_state(queue_ptr);
    ASSERT_NE(found_state, nullptr);
    EXPECT_EQ(found_state.get(), state_ptr);
    EXPECT_EQ(found_state->ring_view.size, 1024U);
    EXPECT_EQ(found_state->ring_view.mask, 1023U);

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
    state->ring_view.size = 2048;
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
    EXPECT_EQ(found_state->ring_view.size, 2048U);

    // Unregister doorbell
    unregister_doorbell(doorbell);

    // Verify removal
    auto after_removal = lookup_queue_state_by_doorbell(doorbell);
    EXPECT_EQ(after_removal, nullptr);

    // Clean up queue registry
    get_queue_registry().wlock([&](auto& registry) { registry.erase(queue_ptr); });
}

TEST(QueueIntercept, AddWriteIndexAdvancesClaimPos)
{
    QueueState state{};
    state.stride    = 1;
    uint64_t idx0   = add_write_index_impl(&state, 1, std::memory_order_relaxed);
    EXPECT_EQ(idx0, 0u);
    EXPECT_EQ(state.claim_pos.load(), 1u);

    uint64_t idx1 = add_write_index_impl(&state, 3, std::memory_order_relaxed);
    EXPECT_EQ(idx1, 1u);
    EXPECT_EQ(state.claim_pos.load(), 4u);
}

TEST(QueueIntercept, StoreWriteIndexSetsClaimPos)
{
    QueueState state{};
    store_write_index_impl(&state, 42, std::memory_order_relaxed);
    EXPECT_EQ(state.claim_pos.load(), 42u);
    store_write_index_impl(&state, 0, std::memory_order_relaxed);
    EXPECT_EQ(state.claim_pos.load(), 0u);
}

TEST(QueueIntercept, StoreWriteIndexRewindNoUnderflow)
{
    // Regression test for bug #6: with stride > 1, store_write_index_impl
    // must not rescale the stored value. Prior behaviour computed
    // `prev + (value - prev) * stride`, which underflowed when value < prev.
    QueueState state{};
    state.stride = 3;  // e.g. k_factor = 2 -> stride = 3

    // Advance claim_pos to 9 via add_write_index (3 packets * stride 3 = 9)
    add_write_index_impl(&state, 3, std::memory_order_relaxed);
    EXPECT_EQ(state.claim_pos.load(), 9u);

    // Rewind to 3 via store_write_index_impl. The resulting claim_pos must
    // be exactly 3, not an underflowed value from rescaling.
    store_write_index_impl(&state, 3, std::memory_order_relaxed);
    EXPECT_EQ(state.claim_pos.load(), 3u);
}

TEST(QueueIntercept, CasWriteIndexSuccess)
{
    QueueState state{};
    state.claim_pos.store(10);
    uint64_t prev = cas_write_index_impl(&state, 10, 20, std::memory_order_relaxed);
    EXPECT_EQ(prev, 10u);
    EXPECT_EQ(state.claim_pos.load(), 20u);
}

TEST(QueueIntercept, CasWriteIndexFailure)
{
    QueueState state{};
    state.claim_pos.store(10);
    uint64_t prev = cas_write_index_impl(&state, 5, 20, std::memory_order_relaxed);
    EXPECT_EQ(prev, 10u);
    EXPECT_EQ(state.claim_pos.load(), 10u);
}

TEST(QueueIntercept, CasWriteIndexNoRescaling)
{
    // With stride > 1, cas_write_index_impl must not rescale the new value.
    // The CAS target is in claim_pos units, same as the expected value.
    QueueState state{};
    state.stride = 4;  // k_factor = 3 -> stride = 4
    state.claim_pos.store(12);

    uint64_t prev = cas_write_index_impl(&state, 12, 20, std::memory_order_relaxed);
    EXPECT_EQ(prev, 12u);
    // Value stored verbatim, NOT 12 + (20-12)*4 = 44
    EXPECT_EQ(state.claim_pos.load(), 20u);
}

TEST(QueueIntercept, LoadWriteIndexReturnsClaimPos)
{
    QueueState state{};
    state.claim_pos.store(99);
    EXPECT_EQ(load_write_index_impl(&state, std::memory_order_relaxed), 99u);
}

// -----------------------------------------------------------------------
// ClaimPosMemoryOrder tests (Step 7', bug #15)
//
// These shape tests confirm that each impl accepts a std::memory_order
// parameter and that the atomic operation still advances/observes
// claim_pos as expected. They do not validate memory-order semantics
// on weakly-ordered hardware (that would require TSAN + multithreading).
// -----------------------------------------------------------------------

TEST(ClaimPosMemoryOrder, AddWriteIndexHonorsMemoryOrder)
{
    QueueState state{};
    state.stride  = 1;
    uint64_t prev = add_write_index_impl(&state, 1, std::memory_order_release);
    EXPECT_EQ(prev, 0u);
    EXPECT_EQ(state.claim_pos.load(), 1u);
}

TEST(ClaimPosMemoryOrder, LoadWriteIndexAcquireReturnsClaimPos)
{
    QueueState state{};
    state.claim_pos.store(123);
    EXPECT_EQ(load_write_index_impl(&state, std::memory_order_acquire), 123u);
}

TEST(ClaimPosMemoryOrder, StoreWriteIndexAcceptsReleaseOrder)
{
    QueueState state{};
    store_write_index_impl(&state, 42, std::memory_order_release);
    EXPECT_EQ(state.claim_pos.load(), 42u);
}

TEST(ClaimPosMemoryOrder, CasWriteIndexAcqRelDerivesFailureOrder)
{
    QueueState state{};
    state.claim_pos.store(0);

    // Success: expected (0) matches claim_pos; cas advances to 10.
    uint64_t prev = cas_write_index_impl(&state, 0, 10, std::memory_order_acq_rel);
    EXPECT_EQ(prev, 0u);
    EXPECT_EQ(state.claim_pos.load(), 10u);

    // Failure: expected (5) does not match claim_pos (10); no change,
    // returns observed value. This exercises the derived failure order
    // (acquire, since success is acq_rel).
    uint64_t prev2 = cas_write_index_impl(&state, 5, 99, std::memory_order_acq_rel);
    EXPECT_EQ(prev2, 10u);
    EXPECT_EQ(state.claim_pos.load(), 10u);
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

    state->ring_view = RingView{ring, 256, 255, 64};
    state->stride    = 1;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    state->claim_pos.store(1);
    record_claim(*state, 0, 1, 1);
    auto* pkt          = get_pkt(ring, 0, 255);
    pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    pkt->kernel_object = 0xDEADBEEF;

    bool doorbell_rang = false;
    process_doorbell_impl(
        state, 0, [&](hsa_signal_t, hsa_signal_value_t) { doorbell_rang = true; });

    EXPECT_TRUE(doorbell_rang);
    EXPECT_EQ(real_wdid, 1u);
    EXPECT_EQ(state->published_pos, 1u);
    auto* submitted = get_pkt(ring, 0, 255);
    EXPECT_EQ(submitted->kernel_object, 0xDEADBEEFu);
    EXPECT_TRUE(state->pending.empty());
}

TEST(QueueIntercept, DoorbellMultiplePacketsTraceOnly)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 256];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_view = RingView{ring, 256, 255, 64};
    state->stride    = 1;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    state->claim_pos.store(3);
    // One bulk claim of 3 packets (HIP-graph-shaped producer).
    record_claim(*state, 0, 3, 1);
    for(uint64_t i = 0; i < 3; i++)
    {
        auto* pkt          = get_pkt(ring, i, 255);
        pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
        pkt->kernel_object = static_cast<uint64_t>(0xA000 + i);
    }

    process_doorbell_impl(state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    EXPECT_EQ(real_wdid, 3u);
    EXPECT_EQ(state->published_pos, 3u);
    for(uint64_t i = 0; i < 3; i++)
    {
        auto* submitted = get_pkt(ring, i, 255);
        EXPECT_EQ(submitted->kernel_object, static_cast<uint64_t>(0xA000 + i));
    }
    EXPECT_TRUE(state->pending.empty());
}

TEST(QueueIntercept, DoorbellNoNewPackets)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 64];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_view = RingView{ring, 64, 63, 64};
    state->stride    = 1;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    state->claim_pos.store(0);
    bool doorbell_rang = false;
    process_doorbell_impl(
        state, 0, [&](hsa_signal_t, hsa_signal_value_t) { doorbell_rang = true; });

    EXPECT_TRUE(doorbell_rang);
    EXPECT_EQ(real_wdid, 0u);
}

TEST(QueueIntercept, DoorbellWithKFactor)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 512];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_view = RingView{ring, 512, 511, 64};
    state->stride    = 8;  // k_factor = 7 -> stride = 8
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    // claim_pos uses stride-scaled positions: 1 packet * stride(8) = 8
    state->claim_pos.store(8);
    record_claim(*state, 0, 1, 8);
    auto* pkt          = get_pkt(ring, 0, 511);
    pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    pkt->kernel_object = 0xCAFE;

    process_doorbell_impl(state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    // With stride=8: 1 app packet at submit pos 0, published_pos advances to 0 + 8 = 8
    EXPECT_EQ(real_wdid, 8u);
    EXPECT_EQ(state->published_pos, 8u);

    auto* submitted = get_pkt(ring, 0, 511);
    EXPECT_EQ(submitted->kernel_object, static_cast<uint64_t>(0xCAFE));
    EXPECT_TRUE(state->pending.empty());
}

TEST(QueueIntercept, DoorbellTwoPacketsWithKFactor)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 512];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_view = RingView{ring, 512, 511, 64};
    state->stride    = 8;  // k_factor = 7 -> stride = 8
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    // One bulk claim of 2 packets (HIP-graph-shaped producer).
    // claim_pos advances by 2 * stride = 16.
    state->claim_pos.store(16);
    record_claim(*state, 0, 2, 8);

    get_pkt(ring, 0, 511)->header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 0, 511)->kernel_object = 0xAAAA;
    get_pkt(ring, 1, 511)->header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 1, 511)->kernel_object = 0xBBBB;

    process_doorbell_impl(state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    // One claim of 2 packets at stride 8: published_pos reaches reservation_end 16.
    EXPECT_EQ(real_wdid, 16u);
    EXPECT_EQ(state->published_pos, 16u);

    // User packets stay at slots 0 and 1; slots 2..15 are BARRIER_AND gap markers.
    EXPECT_EQ(get_pkt(ring, 0, 511)->kernel_object, static_cast<uint64_t>(0xAAAA));
    EXPECT_EQ(get_pkt(ring, 1, 511)->kernel_object, static_cast<uint64_t>(0xBBBB));

    for(uint64_t i = 2; i < 16; ++i)
    {
        EXPECT_EQ(get_pkt(ring, i, 511)->header,
                  static_cast<uint16_t>(HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE))
            << "slot " << i;
        EXPECT_EQ(get_pkt(ring, i, 511)->kernel_object, 0u) << "slot " << i;
    }
    EXPECT_TRUE(state->pending.empty());
}

// Step 12' — strided source layout with k_factor > 0. Single-dispatch HIP path
// calls add_write_index(q, 1) per packet, producing two separate claims, one
// per packet. Each claim occupies one stride-block on the ring.
TEST(QueueIntercept, DoorbellTwoPacketsStridedLayout)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 512];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_view = RingView{ring, 512, 511, 64};
    state->stride    = 8;  // k_factor = 7 -> stride = 8
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    // Two single-packet claims at stride 8. claim_pos advances by stride per claim.
    state->claim_pos.store(16);
    record_claim(*state, 0, 1, 8);
    record_claim(*state, 8, 1, 8);

    // Strided layout: packets at slots 0 and 8 only. Slot 1 is left zeroed —
    // it's part of claim #1's reservation and should be gap-padded by the
    // consumer, not treated as a source slot.
    get_pkt(ring, 0, 511)->header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 0, 511)->kernel_object = 0xAAAA;
    get_pkt(ring, 8, 511)->header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 8, 511)->kernel_object = 0xBBBB;

    process_doorbell_impl(state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    EXPECT_EQ(real_wdid, 16u);
    EXPECT_EQ(state->published_pos, 16u);

    // Each claim owns one stride-block. Packet stays at its claim_index slot;
    // the rest of the stride-block becomes gap-padded.
    EXPECT_EQ(get_pkt(ring, 0, 511)->kernel_object, static_cast<uint64_t>(0xAAAA));
    EXPECT_EQ(get_pkt(ring, 0, 511)->header,
              static_cast<uint16_t>(HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE));

    EXPECT_EQ(get_pkt(ring, 8, 511)->kernel_object, static_cast<uint64_t>(0xBBBB));
    EXPECT_EQ(get_pkt(ring, 8, 511)->header,
              static_cast<uint16_t>(HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE));

    // Slots 1..7 are claim #1's gap padding; slots 9..15 are claim #2's.
    for(uint64_t i = 1; i < 8; ++i)
    {
        EXPECT_EQ(get_pkt(ring, i, 511)->header,
                  static_cast<uint16_t>(HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE))
            << "slot " << i;
        EXPECT_EQ(get_pkt(ring, i, 511)->kernel_object, 0u) << "slot " << i;
    }
    for(uint64_t i = 9; i < 16; ++i)
    {
        EXPECT_EQ(get_pkt(ring, i, 511)->header,
                  static_cast<uint16_t>(HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE))
            << "slot " << i;
        EXPECT_EQ(get_pkt(ring, i, 511)->kernel_object, 0u) << "slot " << i;
    }
    EXPECT_TRUE(state->pending.empty());
}

// Bug #9 — HSA_QUEUE_TYPE_SINGLE queues require monotonically non-decreasing
// doorbell values. Verify the early-exit path clamps a smaller incoming value
// up to the last observed high-water mark.
TEST(QueueIntercept, SingleQueueDoorbellClamp)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 64];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_view = RingView{ring, 64, 63, 64};
    state->stride    = 1;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;
    state->is_single = true;
    state->last_doorbell_val.store(100, std::memory_order_relaxed);

    // Empty scan window (claim_pos == published_pos) -> early-exit forwards
    // caller's value, which for SINGLE queues must be clamped to >= 100.
    state->claim_pos.store(0);
    state->published_pos = 0;

    hsa_signal_value_t captured = -1;
    process_doorbell_impl(
        state, 50, [&](hsa_signal_t, hsa_signal_value_t v) { captured = v; });

    EXPECT_EQ(captured, 100);
    EXPECT_EQ(state->last_doorbell_val.load(std::memory_order_relaxed), 100u);
}

// Bug #9 — multi queues never clamp. A forwarded value smaller than
// last_doorbell_val must pass through unchanged.
TEST(QueueIntercept, MultiQueueDoorbellNoClamp)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 64];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_view = RingView{ring, 64, 63, 64};
    state->stride    = 1;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;
    state->is_single = false;
    state->last_doorbell_val.store(100, std::memory_order_relaxed);

    state->claim_pos.store(0);
    state->published_pos = 0;

    hsa_signal_value_t captured = -1;
    process_doorbell_impl(
        state, 50, [&](hsa_signal_t, hsa_signal_value_t v) { captured = v; });

    // Multi queue: no clamp. Forward the user's value verbatim.
    EXPECT_EQ(captured, 50);
}

// Bug #9 — publish_and_ring must also update last_doorbell_val. Drive the
// full scan path on a SINGLE queue with 1 real packet, then issue a second
// empty-scan doorbell with a smaller value and verify it's clamped to the
// high-water mark published by the first call.
TEST(QueueIntercept, SingleQueueDoorbellHighWaterFromPublish)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 64];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_view = RingView{ring, 64, 63, 64};
    state->stride    = 1;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;
    state->is_single = true;
    state->last_doorbell_val.store(0, std::memory_order_relaxed);

    state->claim_pos.store(1);
    record_claim(*state, 0, 1, 1);
    auto* pkt          = get_pkt(ring, 0, 63);
    pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    pkt->kernel_object = 0xC0DE;

    hsa_signal_value_t captured1 = -1;
    process_doorbell_impl(
        state, 0, [&](hsa_signal_t, hsa_signal_value_t v) { captured1 = v; });

    // After the publish, last_doorbell_val is published_pos - 1 == 0.
    EXPECT_EQ(captured1, 0);
    EXPECT_EQ(state->last_doorbell_val.load(std::memory_order_relaxed), 0u);

    // Now nothing new is claimed. The caller forwards a value of 5. last
    // observed is 0; 5 >= 0 so no clamp; it forwards.
    hsa_signal_value_t captured2 = -1;
    process_doorbell_impl(
        state, 5, [&](hsa_signal_t, hsa_signal_value_t v) { captured2 = v; });
    EXPECT_EQ(captured2, 5);
    EXPECT_EQ(state->last_doorbell_val.load(std::memory_order_relaxed), 5u);

    // And forwarding 3 should now clamp to the new high-water mark of 5.
    hsa_signal_value_t captured3 = -1;
    process_doorbell_impl(
        state, 3, [&](hsa_signal_t, hsa_signal_value_t v) { captured3 = v; });
    EXPECT_EQ(captured3, 5);
}

// -----------------------------------------------------------------------
// Step 12' — explicit per-claim tracking tests.
// -----------------------------------------------------------------------

TEST(PendingClaim, RecordedOnAddWriteIndex)
{
    QueueState state{};
    state.stride = 4;  // non-trivial stride

    uint64_t idx0 = add_write_index_impl(&state, 1, std::memory_order_relaxed);
    uint64_t idx1 = add_write_index_impl(&state, 1, std::memory_order_relaxed);
    uint64_t idx2 = add_write_index_impl(&state, 1, std::memory_order_relaxed);

    EXPECT_EQ(idx0, 0u);
    EXPECT_EQ(idx1, 4u);
    EXPECT_EQ(idx2, 8u);

    ASSERT_EQ(state.pending.size(), 3u);
    auto it = state.pending.begin();
    EXPECT_EQ(it->second.claim_index, 0u);
    EXPECT_EQ(it->second.packet_count, 1u);
    EXPECT_EQ(it->second.stride, 4u);
    ++it;
    EXPECT_EQ(it->second.claim_index, 4u);
    EXPECT_EQ(it->second.packet_count, 1u);
    EXPECT_EQ(it->second.stride, 4u);
    ++it;
    EXPECT_EQ(it->second.claim_index, 8u);
    EXPECT_EQ(it->second.packet_count, 1u);
    EXPECT_EQ(it->second.stride, 4u);
}

TEST(PendingClaim, BulkSingleEntry)
{
    QueueState state{};
    state.stride = 1;

    uint64_t idx = add_write_index_impl(&state, 5, std::memory_order_relaxed);
    EXPECT_EQ(idx, 0u);

    ASSERT_EQ(state.pending.size(), 1u);
    const auto& pc = state.pending.begin()->second;
    EXPECT_EQ(pc.claim_index, 0u);
    EXPECT_EQ(pc.packet_count, 5u);
    EXPECT_EQ(pc.stride, 1u);
}

TEST(PendingClaim, DoorbellDefersNonReadyClaim)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 256];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_view = RingView{ring, 256, 255, 64};
    state->stride    = 1;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    // Two single-packet claims. Only the first slot is actually written — the
    // second claim's slot is still INVALID (header == 0 which maps to
    // HSA_PACKET_TYPE_VENDOR_SPECIFIC == 0, but HSA_PACKET_TYPE_INVALID == 1
    // is what we detect — leave it explicitly INVALID).
    state->claim_pos.store(2);
    record_claim(*state, 0, 1, 1);
    record_claim(*state, 1, 1, 1);

    get_pkt(ring, 0, 255)->header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 0, 255)->kernel_object = 0xA000;
    // Slot 1 header is HSA_PACKET_TYPE_INVALID, writer not yet finished.
    get_pkt(ring, 1, 255)->header = (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE);

    process_doorbell_impl(state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    // First claim consumed, second claim still deferred.
    EXPECT_EQ(state->pending.size(), 1u);
    EXPECT_EQ(state->pending.begin()->second.claim_index, 1u);

    // published_pos advanced by claim #1 only (packet_count=1, stride=1).
    EXPECT_EQ(state->published_pos, 1u);
    EXPECT_EQ(real_wdid, 1u);
}

TEST(PendingClaim, DoorbellResumesAfterPreviousDefer)
{
    auto             state = std::make_shared<QueueState>();
    alignas(64) char ring[64 * 256];
    memset(ring, 0, sizeof(ring));
    uint64_t real_wdid = 0;
    uint64_t real_rdid = 0;

    state->ring_view = RingView{ring, 256, 255, 64};
    state->stride    = 1;
    state->real_wdid = &real_wdid;
    state->real_rdid = &real_rdid;

    state->claim_pos.store(2);
    record_claim(*state, 0, 1, 1);
    record_claim(*state, 1, 1, 1);

    get_pkt(ring, 0, 255)->header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 0, 255)->kernel_object = 0xA000;
    get_pkt(ring, 1, 255)->header = (HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE);

    // First doorbell: only claim #1 consumed.
    process_doorbell_impl(state, 0, [](hsa_signal_t, hsa_signal_value_t) {});
    ASSERT_EQ(state->pending.size(), 1u);
    EXPECT_EQ(state->published_pos, 1u);

    // Producer finishes writing the second packet.
    get_pkt(ring, 1, 255)->header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 1, 255)->kernel_object = 0xB000;

    // Second doorbell: claim #2 consumed, pending emptied.
    process_doorbell_impl(state, 0, [](hsa_signal_t, hsa_signal_value_t) {});
    EXPECT_TRUE(state->pending.empty());
    EXPECT_EQ(state->published_pos, 2u);
    EXPECT_EQ(real_wdid, 2u);
}

TEST(PendingClaim, StoreWriteIndexClearsPending)
{
    QueueState state{};
    state.stride = 1;
    add_write_index_impl(&state, 1, std::memory_order_relaxed);
    add_write_index_impl(&state, 1, std::memory_order_relaxed);
    ASSERT_EQ(state.pending.size(), 2u);

    store_write_index_impl(&state, 100, std::memory_order_relaxed);
    EXPECT_TRUE(state.pending.empty());
    EXPECT_EQ(state.claim_pos.load(), 100u);
}

TEST(PendingClaim, CasWriteIndexSuccessClearsPending)
{
    QueueState state{};
    state.stride = 1;
    add_write_index_impl(&state, 1, std::memory_order_relaxed);
    add_write_index_impl(&state, 1, std::memory_order_relaxed);
    ASSERT_EQ(state.pending.size(), 2u);

    const uint64_t cur = state.claim_pos.load();
    uint64_t       prev =
        cas_write_index_impl(&state, cur, 100, std::memory_order_relaxed);
    EXPECT_EQ(prev, cur);
    EXPECT_TRUE(state.pending.empty());
    EXPECT_EQ(state.claim_pos.load(), 100u);
}

TEST(PendingClaim, CasWriteIndexFailureKeepsPending)
{
    QueueState state{};
    state.stride = 1;
    add_write_index_impl(&state, 1, std::memory_order_relaxed);
    add_write_index_impl(&state, 1, std::memory_order_relaxed);
    const size_t before = state.pending.size();
    ASSERT_EQ(before, 2u);

    const uint64_t cur = state.claim_pos.load();
    // Wrong expected value -> CAS fails.
    uint64_t prev =
        cas_write_index_impl(&state, cur + 123, 999, std::memory_order_relaxed);
    EXPECT_EQ(prev, cur);
    EXPECT_EQ(state.pending.size(), before);
    EXPECT_EQ(state.claim_pos.load(), cur);
}

TEST(PendingClaim, MultiProducerClaimsSortedByIndex)
{
    // Simulate the producer race where threads insert into `pending` in an
    // order different from claim_index order. std::map must keep them
    // claim_index-sorted for the consumer.
    QueueState state{};
    record_claim(state, 24, 1, 8);  // inserted first, highest index
    record_claim(state, 8, 1, 8);
    record_claim(state, 16, 1, 8);
    record_claim(state, 0, 1, 8);   // inserted last, lowest index

    ASSERT_EQ(state.pending.size(), 4u);
    auto it = state.pending.begin();
    EXPECT_EQ(it->first, 0u);
    EXPECT_EQ(it->second.claim_index, 0u);
    ++it;
    EXPECT_EQ(it->first, 8u);
    ++it;
    EXPECT_EQ(it->first, 16u);
    ++it;
    EXPECT_EQ(it->first, 24u);
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

    create_queue_state(&fake_queue, &fake_wdid, &fake_rdid, 7);

    auto state = lookup_queue_state(&fake_queue);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->ring_view.buf, reinterpret_cast<void*>(ring_mem));
    EXPECT_EQ(state->ring_view.size, 256u);
    EXPECT_EQ(state->ring_view.mask, 255u);
    EXPECT_EQ(state->real_wdid, &fake_wdid);
    EXPECT_EQ(state->real_rdid, &fake_rdid);
    EXPECT_EQ(state->stride, 8u);
    EXPECT_EQ(state->doorbell_signal.handle, 9999u);

    auto by_doorbell = lookup_queue_state_by_doorbell({.handle = 9999});
    EXPECT_EQ(by_doorbell.get(), state.get());

    destroy_queue_state(&fake_queue);
    EXPECT_EQ(lookup_queue_state(&fake_queue), nullptr);
    EXPECT_EQ(lookup_queue_state_by_doorbell({.handle = 9999}), nullptr);
}

}  // namespace
}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
