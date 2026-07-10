// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"
#include "lib/common/environment.hpp"

#include <gtest/gtest.h>

#include <hsa/amd_hsa_queue.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>

namespace rocprofiler
{
namespace hsa
{
namespace queue_interposition
{
namespace
{
TEST(queue_interposition, queue_state_default_init)
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
}

TEST(queue_interposition, registry_insert_and_lookup)
{
    // Create a dummy queue pointer for testing
    // We're just using a dummy address; we never dereference it
    auto               dummy_queue = hsa_queue_t{};
    const hsa_queue_t* queue_ptr   = &dummy_queue;

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

TEST(queue_interposition, doorbell_map_insert_and_lookup)
{
    // Create a dummy queue for testing
    auto               dummy_queue = hsa_queue_t{};
    const hsa_queue_t* queue_ptr   = &dummy_queue;

    // Create and register a QueueState
    auto state            = std::make_shared<QueueState>();
    state->ring_size      = 2048;
    QueueState* state_ptr = state.get();

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = state; });

    // Create a doorbell signal and register it. kind must be a doorbell kind: lookup_queue_state_
    // by_doorbell only trusts queue_ptr for doorbell-kind signals (queue_ptr aliases reserved2 in
    // the union for other kinds).
    auto amd_doorbell = amd_signal_t{};
    amd_doorbell.kind = AMD_SIGNAL_KIND_DOORBELL;
    amd_doorbell.queue_ptr =
        const_cast<amd_queue_v2_t*>(reinterpret_cast<const amd_queue_v2_t*>(queue_ptr));
    auto doorbell = hsa_signal_t{.handle = reinterpret_cast<uint64_t>(&amd_doorbell)};

    // Look up by doorbell
    auto found_state = lookup_queue_state_by_doorbell(doorbell);
    ASSERT_NE(found_state, nullptr);
    EXPECT_EQ(found_state.get(), state_ptr);
    EXPECT_EQ(found_state->ring_size, 2048U);

    // Clean up queue registry
    destroy_queue_state(queue_ptr);

    // Verify removal
    auto after_removal = lookup_queue_state_by_doorbell(doorbell);
    EXPECT_EQ(after_removal, nullptr);
}

TEST(queue_interposition, add_write_index_advances_virtual_wptr)
{
    QueueState state{};
    uint64_t   idx0 = add_write_index_impl(&state, 1);
    EXPECT_EQ(idx0, 0u);
    EXPECT_EQ(state.virtual_wptr.load(), 1u);

    uint64_t idx1 = add_write_index_impl(&state, 3);
    EXPECT_EQ(idx1, 1u);
    EXPECT_EQ(state.virtual_wptr.load(), 4u);
}

TEST(queue_interposition, store_write_index_sets_virtual_wptr)
{
    QueueState state{};
    store_write_index_impl(&state, 42);
    EXPECT_EQ(state.virtual_wptr.load(), 42u);
    store_write_index_impl(&state, 0);
    EXPECT_EQ(state.virtual_wptr.load(), 0u);
}

TEST(queue_interposition, cas_write_index_success)
{
    QueueState state{};
    state.virtual_wptr.store(10);
    uint64_t prev = cas_write_index_impl(&state, 10, 20);
    EXPECT_EQ(prev, 10u);
    EXPECT_EQ(state.virtual_wptr.load(), 20u);
}

TEST(queue_interposition, cas_write_index_failure)
{
    QueueState state{};
    state.virtual_wptr.store(10);
    uint64_t prev = cas_write_index_impl(&state, 5, 20);
    EXPECT_EQ(prev, 10u);
    EXPECT_EQ(state.virtual_wptr.load(), 10u);
}

TEST(queue_interposition, load_write_index_returns_virtual_wptr)
{
    QueueState state{};
    state.virtual_wptr.store(99);
    EXPECT_EQ(load_write_index_impl(&state), 99u);
}

TEST(queue_interposition, async_signal_handler_thread_count_uses_gpu_queue_count)
{
    common::env_store env(
        {{"GPU_MAX_HW_QUEUES", "7", 1}, {"ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS", "", 1}});
    ASSERT_TRUE(env.push());

    EXPECT_EQ(get_async_signal_handler_thread_count(), 7u);
}

TEST(queue_interposition, async_signal_handler_thread_count_clamps_zero_gpu_queue_count)
{
    common::env_store env(
        {{"GPU_MAX_HW_QUEUES", "0", 1}, {"ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS", "", 1}});
    ASSERT_TRUE(env.push());

    EXPECT_EQ(get_async_signal_handler_thread_count(), 1u);
}

TEST(queue_interposition, async_signal_handler_thread_count_override_takes_precedence)
{
    common::env_store env(
        {{"GPU_MAX_HW_QUEUES", "0", 1}, {"ROCPROFILER_ASYNC_SIGNAL_HANDLER_THREADS", "3", 1}});
    ASSERT_TRUE(env.push());

    EXPECT_EQ(get_async_signal_handler_thread_count(), 3u);
}

namespace
{
hsa_kernel_dispatch_packet_t*
get_pkt(void* ring, uint64_t idx, uint32_t mask)
{
    return &reinterpret_cast<hsa_kernel_dispatch_packet_t*>(ring)[idx & mask];
}
}  // namespace

TEST(queue_interposition, doorbell_trace_only_copies_packet)
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

    // doorbell value is the index of the last committed packet (here: slot 0)
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

TEST(queue_interposition, doorbell_multiple_packets_trace_only)
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

    // doorbell value is the index of the last committed packet (here: slot 2 of 3)
    process_doorbell_impl(state, 2, [](hsa_signal_t, hsa_signal_value_t) {});

    EXPECT_EQ(real_wdid, 3u);
    EXPECT_EQ(state->next_submit_pos, 3u);
    for(uint64_t i = 0; i < 3; i++)
    {
        auto* submitted = get_pkt(ring, i, 255);
        EXPECT_EQ(submitted->kernel_object, static_cast<uint64_t>(0xA000 + i));
    }
}

TEST(queue_interposition, doorbell_no_new_packets)
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

TEST(queue_interposition, create_and_destroy_queue_state)
{
    // char ring_mem[64 * 256];
    constexpr auto   ring_size = 64UL * 256UL;
    alignas(64) auto ring_mem  = std::array<std::byte, ring_size>{};
    ring_mem.fill(std::byte{0});

    amd_queue_v2_t amd_fake_queue{};
    hsa_queue_t&   hsa_fake_queue    = amd_fake_queue.hsa_queue;
    hsa_fake_queue.base_address      = reinterpret_cast<void*>(ring_mem.data());
    hsa_fake_queue.size              = 256;
    hsa_fake_queue.doorbell_signal   = {.handle = 9999};
    amd_fake_queue.write_dispatch_id = 0;
    amd_fake_queue.read_dispatch_id  = 0;

    // make sure that the hsa_queue field is at offset 0, so that we can safely reinterpret_cast
    // from hsa_queue_t* to amd_queue_v2_t*
    ASSERT_EQ(offsetof(amd_queue_v2_t, hsa_queue), 0u);

    auto* fake_queue = reinterpret_cast<hsa_queue_t*>(&amd_fake_queue);

    create_queue_state(fake_queue);

    auto state = lookup_queue_state(fake_queue);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->ring_buf, reinterpret_cast<void*>(ring_mem.data()));
    EXPECT_EQ(state->ring_size, 256u);
    EXPECT_EQ(state->ring_mask, 255u);
    EXPECT_EQ(*state->real_wdid, amd_fake_queue.write_dispatch_id);
    EXPECT_EQ(*state->real_rdid, amd_fake_queue.read_dispatch_id);
    EXPECT_EQ(state->doorbell_signal.handle, 9999u);

    auto amd_doorbell      = amd_signal_t{};
    amd_doorbell.kind      = AMD_SIGNAL_KIND_DOORBELL;
    amd_doorbell.queue_ptr = &amd_fake_queue;
    auto doorbell          = hsa_signal_t{.handle = reinterpret_cast<uint64_t>(&amd_doorbell)};

    auto by_doorbell = lookup_queue_state_by_doorbell(doorbell);
    EXPECT_EQ(by_doorbell.get(), state.get());

    destroy_queue_state(fake_queue);
    EXPECT_EQ(lookup_queue_state(fake_queue), nullptr);
    EXPECT_EQ(lookup_queue_state_by_doorbell(doorbell), nullptr);
}

TEST(queue_interposition, doorbell_backpressure_waits_when_ring_full_k0)
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

    // doorbell value is the index of the last committed packet (here: virtual slot 4)
    hsa_signal_value_t doorbell_value = -1;
    auto               fut            = std::async(std::launch::async, [&]() {
        process_doorbell_impl(
            state, 4, [&](hsa_signal_t, hsa_signal_value_t v) { doorbell_value = v; });
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

TEST(queue_interposition, doorbell_out_of_order_does_not_hang)
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
    state->virtual_wptr.store(2);

    auto set_header = [&](uint64_t idx, uint16_t type) {
        auto* pkt = get_pkt(ring, idx, 255);
        __atomic_store_n(&pkt->header, type, __ATOMIC_RELEASE);
    };

    set_header(0, static_cast<uint16_t>(HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE));
    get_pkt(ring, 1, 255)->kernel_object = 0xB1;
    set_header(1, static_cast<uint16_t>(HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE));

    std::atomic<bool> done{false};
    std::thread       worker([&]() {
        process_doorbell_impl(state, 1, [](hsa_signal_t, hsa_signal_value_t) {});
        done.store(true, std::memory_order_release);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while(!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});

    const bool hung = !done.load(std::memory_order_acquire);
    if(hung)
    {
        set_header(
            0, static_cast<uint16_t>(HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE));
    }
    worker.join();
    ASSERT_FALSE(hung) << "process_doorbell_impl hung waiting on an uncommitted earlier slot "
                          "after an out-of-order doorbell ring";

    EXPECT_EQ(state->next_scan_pos, 0u);
    EXPECT_EQ(real_wdid, 0u);

    get_pkt(ring, 0, 255)->kernel_object = 0xB0;
    set_header(0, static_cast<uint16_t>(HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE));

    process_doorbell_impl(state, 0, [](hsa_signal_t, hsa_signal_value_t) {});
    EXPECT_EQ(real_wdid, 2u);
    EXPECT_EQ(state->next_scan_pos, 2u);
    EXPECT_EQ(get_pkt(ring, 0, 255)->kernel_object, static_cast<uint64_t>(0xB0));
    EXPECT_EQ(get_pkt(ring, 1, 255)->kernel_object, static_cast<uint64_t>(0xB1));
}
}  // namespace
}  // namespace queue_interposition
}  // namespace hsa
}  // namespace rocprofiler
