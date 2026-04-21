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

#include <memory>

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
    EXPECT_EQ(state.k_factor, 0UL);
    EXPECT_EQ(state.metadata_state, nullptr);
}

TEST(QueueIntercept, RegistryInsertAndLookup)
{
    // Create a dummy queue pointer for testing
    // We're just using a dummy address; we never dereference it
    hsa_queue_t        dummy_queue;
    const hsa_queue_t* queue_ptr = &dummy_queue;

    // Create a QueueState and insert it into the registry
    auto state       = std::make_unique<QueueState>();
    state->ring_size = 1024;
    state->ring_mask = 1023;

    QueueState* state_ptr = state.get();

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = std::move(state); });

    // Look up the state
    QueueState* found_state = lookup_queue_state(queue_ptr);
    ASSERT_NE(found_state, nullptr);
    EXPECT_EQ(found_state, state_ptr);
    EXPECT_EQ(found_state->ring_size, 1024U);
    EXPECT_EQ(found_state->ring_mask, 1023U);

    // Clean up
    get_queue_registry().wlock([&](auto& registry) { registry.erase(queue_ptr); });

    // Verify removal
    QueueState* after_removal = lookup_queue_state(queue_ptr);
    EXPECT_EQ(after_removal, nullptr);
}

TEST(QueueIntercept, DoorbellMapInsertAndLookup)
{
    // Create a dummy queue for testing
    hsa_queue_t        dummy_queue;
    const hsa_queue_t* queue_ptr = &dummy_queue;

    // Create and register a QueueState
    auto state            = std::make_unique<QueueState>();
    state->ring_size      = 2048;
    QueueState* state_ptr = state.get();

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = std::move(state); });

    // Create a doorbell signal and register it
    hsa_signal_t doorbell;
    doorbell.handle = 0x12345678;

    register_doorbell(queue_ptr, doorbell);

    // Look up by doorbell
    QueueState* found_state = lookup_queue_state_by_doorbell(doorbell);
    ASSERT_NE(found_state, nullptr);
    EXPECT_EQ(found_state, state_ptr);
    EXPECT_EQ(found_state->ring_size, 2048U);

    // Unregister doorbell
    unregister_doorbell(doorbell);

    // Verify removal
    QueueState* after_removal = lookup_queue_state_by_doorbell(doorbell);
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
    QueueState       state{};
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

    state.virtual_wptr.store(1);
    auto* pkt          = get_pkt(ring, 0, 255);
    pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    pkt->kernel_object = 0xDEADBEEF;

    bool doorbell_rang = false;
    process_doorbell_impl(
        &state, 0, [&](hsa_signal_t, hsa_signal_value_t) { doorbell_rang = true; });

    EXPECT_TRUE(doorbell_rang);
    EXPECT_EQ(real_wdid, 1u);
    EXPECT_EQ(state.next_submit_pos, 1u);
    EXPECT_EQ(state.next_scan_pos, 1u);
    auto* submitted = get_pkt(ring, 0, 255);
    EXPECT_EQ(submitted->kernel_object, 0xDEADBEEFu);
}

TEST(QueueIntercept, DoorbellMultiplePacketsTraceOnly)
{
    QueueState       state{};
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

    state.virtual_wptr.store(3);
    for(uint64_t i = 0; i < 3; i++)
    {
        auto* pkt          = get_pkt(ring, i, 255);
        pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
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
    QueueState       state{};
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

    state.virtual_wptr.store(0);
    bool doorbell_rang = false;
    process_doorbell_impl(
        &state, 0, [&](hsa_signal_t, hsa_signal_value_t) { doorbell_rang = true; });

    EXPECT_TRUE(doorbell_rang);
    EXPECT_EQ(real_wdid, 0u);
}

TEST(QueueIntercept, DoorbellWithKFactor)
{
    QueueState       state{};
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

    // virtual_wptr uses stride-scaled positions: 1 packet * stride(8) = 8
    state.virtual_wptr.store(8);
    auto* pkt          = get_pkt(ring, 0, 511);
    pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    pkt->kernel_object = 0xCAFE;

    process_doorbell_impl(&state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    // With K=7: 1 app packet at submit pos 0, next_submit_pos advances to 0 + 1 + 7 = 8
    EXPECT_EQ(real_wdid, 8u);
    EXPECT_EQ(state.next_submit_pos, 8u);
    EXPECT_EQ(state.next_scan_pos, 8u);

    auto* submitted = get_pkt(ring, 0, 511);
    EXPECT_EQ(submitted->kernel_object, static_cast<uint64_t>(0xCAFE));
}

TEST(QueueIntercept, DoorbellTwoPacketsWithKFactor)
{
    QueueState       state{};
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

    // 2 packets * stride(8) = 16
    state.virtual_wptr.store(16);
    get_pkt(ring, 0, 511)->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 0, 511)->kernel_object = 0xAAAA;
    get_pkt(ring, 8, 511)->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    get_pkt(ring, 8, 511)->kernel_object = 0xBBBB;

    process_doorbell_impl(&state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    // 2 app packets * (1 + 7) = 16 total
    EXPECT_EQ(real_wdid, 16u);
    EXPECT_EQ(state.next_submit_pos, 16u);

    // First app packet at submit pos 0, second at submit pos 8
    EXPECT_EQ(get_pkt(ring, 0, 511)->kernel_object, static_cast<uint64_t>(0xAAAA));
    EXPECT_EQ(get_pkt(ring, 8, 511)->kernel_object, static_cast<uint64_t>(0xBBBB));
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

    auto* state = lookup_queue_state(&fake_queue);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->ring_buf, reinterpret_cast<void*>(ring_mem));
    EXPECT_EQ(state->ring_size, 256u);
    EXPECT_EQ(state->ring_mask, 255u);
    EXPECT_EQ(state->real_wdid, &fake_wdid);
    EXPECT_EQ(state->real_rdid, &fake_rdid);
    EXPECT_EQ(state->k_factor, 7u);
    EXPECT_EQ(state->doorbell_signal.handle, 9999u);

    auto* by_doorbell = lookup_queue_state_by_doorbell({.handle = 9999});
    EXPECT_EQ(by_doorbell, state);

    destroy_queue_state(&fake_queue);
    EXPECT_EQ(lookup_queue_state(&fake_queue), nullptr);
    EXPECT_EQ(lookup_queue_state_by_doorbell({.handle = 9999}), nullptr);
}

TEST(QueueIntercept, MetadataSyncWritesEntries)
{
    QueueState compute_state{};
    QueueState meta_state{};

    alignas(64) char  compute_ring[64 * 256];
    alignas(256) char meta_ring[256 * 256];
    memset(compute_ring, 0, sizeof(compute_ring));
    memset(meta_ring, 0, sizeof(meta_ring));

    uint64_t compute_wdid = 0, compute_rdid = 0;
    uint64_t meta_wdid = 0, meta_rdid = 0;

    compute_state.ring_buf       = compute_ring;
    compute_state.ring_size      = 256;
    compute_state.ring_mask      = 255;
    compute_state.real_wdid      = &compute_wdid;
    compute_state.real_rdid      = &compute_rdid;
    compute_state.k_factor       = 7;
    compute_state.metadata_state = &meta_state;

    meta_state.ring_buf  = meta_ring;
    meta_state.ring_size = 256;
    meta_state.ring_mask = 255;
    meta_state.real_wdid = &meta_wdid;
    meta_state.real_rdid = &meta_rdid;

    // 1 packet * stride(8) = 8
    compute_state.virtual_wptr.store(8);
    auto* pkt          = get_pkt(compute_ring, 0, 255);
    pkt->header        = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    pkt->kernel_object = 0x1234;

    process_doorbell_impl(&compute_state, 0, [](hsa_signal_t, hsa_signal_value_t) {});

    EXPECT_EQ(meta_wdid, 8u);
    EXPECT_EQ(meta_state.next_submit_pos, 8u);
}

TEST(QueueIntercept, NoMetadataSyncWhenNoPairing)
{
    QueueState       compute_state{};
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

    // 1 packet * stride(8) = 8
    compute_state.virtual_wptr.store(8);
    get_pkt(ring, 0, 255)->header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);

    process_doorbell_impl(&compute_state, 0, [](hsa_signal_t, hsa_signal_value_t) {});
    EXPECT_EQ(wdid, 8u);
}

}  // namespace
}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
