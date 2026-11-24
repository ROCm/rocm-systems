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
#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/kfd/correlation_types.hpp"

#include <gtest/gtest.h>

#include <hsa/amd_hsa_queue.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

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

namespace
{
constexpr uint16_t k_dispatch_header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
constexpr uint16_t k_retired_header  = HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;

uint64_t
deadline_in(std::chrono::milliseconds budget)
{
    return kfd::steady_now_ns() + static_cast<uint64_t>(std::chrono::nanoseconds{budget}.count());
}
}  // namespace

// The read index lags a finished burst (gfx1250 defers its write-back while the
// queue idles), but the CP has reset every slot it retired: the drain must not
// spend its budget waiting for a read index that will not move.
TEST(queue_interposition, hw_drain_accepts_retired_slots_behind_stale_read_index)
{
    QueueState       state{};
    alignas(64) char ring[64 * 64];
    memset(ring, 0, sizeof(ring));
    uint64_t real_rdid = 16;

    state.ring_buf  = ring;
    state.ring_size = 64;
    state.ring_mask = 63;
    state.real_rdid = &real_rdid;

    for(uint64_t i = 16; i < 19; ++i)
        get_pkt(ring, i, 63)->header = k_retired_header;

    const auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(wait_queue_hw_drained_locked(state, 19, deadline_in(std::chrono::seconds{10})));
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds{1});
}

TEST(queue_interposition, hw_drain_times_out_while_a_packet_is_in_flight)
{
    QueueState       state{};
    alignas(64) char ring[64 * 64];
    memset(ring, 0, sizeof(ring));
    uint64_t real_rdid = 0;

    state.ring_buf  = ring;
    state.ring_size = 64;
    state.ring_mask = 63;
    state.real_rdid = &real_rdid;

    // Older packets retired, the newest still running; a zero-filled header
    // (type VENDOR_SPECIFIC) is not retired either.
    get_pkt(ring, 0, 63)->header = k_retired_header;
    get_pkt(ring, 1, 63)->header = k_retired_header;
    get_pkt(ring, 2, 63)->header = k_dispatch_header;
    EXPECT_FALSE(
        wait_queue_hw_drained_locked(state, 3, deadline_in(std::chrono::milliseconds{20})));

    get_pkt(ring, 2, 63)->header = k_retired_header;
    get_pkt(ring, 0, 63)->header = 0;
    EXPECT_FALSE(
        wait_queue_hw_drained_locked(state, 3, deadline_in(std::chrono::milliseconds{20})));
}

TEST(queue_interposition, hw_drain_completes_when_the_last_packet_retires)
{
    QueueState       state{};
    alignas(64) char ring[64 * 64];
    memset(ring, 0, sizeof(ring));
    uint64_t real_rdid = 0;

    state.ring_buf  = ring;
    state.ring_size = 64;
    state.ring_mask = 63;
    state.real_rdid = &real_rdid;

    get_pkt(ring, 0, 63)->header = k_retired_header;
    get_pkt(ring, 1, 63)->header = k_dispatch_header;

    auto* last_hdr = &get_pkt(ring, 1, 63)->header;
    auto  retire   = std::async(std::launch::async, [last_hdr]() {
        std::this_thread::sleep_for(std::chrono::milliseconds{5});
        __atomic_store_n(last_hdr, k_retired_header, __ATOMIC_RELEASE);
    });

    const auto start = std::chrono::steady_clock::now();
    EXPECT_TRUE(wait_queue_hw_drained_locked(state, 2, deadline_in(std::chrono::seconds{10})));
    EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds{1});
    retire.get();
}

// A read index that has caught up is sufficient on its own: slots it has passed
// may already hold the producer's next packets.
TEST(queue_interposition, hw_drain_trusts_a_caught_up_read_index)
{
    QueueState       state{};
    alignas(64) char ring[64 * 64];
    memset(ring, 0, sizeof(ring));
    uint64_t real_rdid = 3;

    state.ring_buf  = ring;
    state.ring_size = 64;
    state.ring_mask = 63;
    state.real_rdid = &real_rdid;

    for(uint64_t i = 0; i < 3; ++i)
        get_pkt(ring, i, 63)->header = k_dispatch_header;

    EXPECT_TRUE(wait_queue_hw_drained_locked(state, 3, deadline_in(std::chrono::milliseconds{20})));
}

// The window [read index, P) wraps the ring: only the slots it covers count, so a
// live packet in a slot outside it does not hold the drain.
TEST(queue_interposition, hw_drain_checks_the_window_across_ring_wrap)
{
    QueueState       state{};
    alignas(64) char ring[64 * 4];
    memset(ring, 0, sizeof(ring));
    uint64_t real_rdid = 6;

    state.ring_buf  = ring;
    state.ring_size = 4;
    state.ring_mask = 3;
    state.real_rdid = &real_rdid;

    // indices 6, 7, 8 -> slots 2, 3, 0; slot 1 holds the next, unpublished packet
    get_pkt(ring, 6, 3)->header = k_retired_header;
    get_pkt(ring, 7, 3)->header = k_retired_header;
    get_pkt(ring, 8, 3)->header = k_retired_header;
    get_pkt(ring, 9, 3)->header = k_dispatch_header;
    EXPECT_TRUE(wait_queue_hw_drained_locked(state, 9, deadline_in(std::chrono::milliseconds{20})));

    // a window longer than the ring cannot be decided from headers
    real_rdid                   = 4;
    get_pkt(ring, 9, 3)->header = k_retired_header;
    EXPECT_FALSE(
        wait_queue_hw_drained_locked(state, 9, deadline_in(std::chrono::milliseconds{20})));

    real_rdid                   = 6;
    get_pkt(ring, 8, 3)->header = k_dispatch_header;
    EXPECT_FALSE(
        wait_queue_hw_drained_locked(state, 9, deadline_in(std::chrono::milliseconds{20})));
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

void
fill_hsa_fn_tables()
{
    auto* core = rocprofiler::hsa::get_core_table();
    auto* amd  = rocprofiler::hsa::get_amd_ext_table();
    ASSERT_NE(core, nullptr);
    ASSERT_NE(amd, nullptr);

    core->hsa_signal_store_screlease_fn = hsa_signal_store_screlease;
    core->hsa_signal_wait_relaxed_fn    = hsa_signal_wait_relaxed;
    core->hsa_signal_destroy_fn         = hsa_signal_destroy;
    core->hsa_signal_load_scacquire_fn  = hsa_signal_load_scacquire;

    amd->hsa_amd_signal_create_fn   = hsa_amd_signal_create;
    amd->hsa_amd_signal_wait_any_fn = hsa_amd_signal_wait_any;
}

CoreApiTable
make_core_table()
{
    auto table = CoreApiTable{};
    std::memset(&table, 0, sizeof(table));
    table.hsa_queue_add_write_index_relaxed_fn     = hsa_queue_add_write_index_relaxed;
    table.hsa_queue_add_write_index_scacq_screl_fn = hsa_queue_add_write_index_scacq_screl;
    table.hsa_queue_add_write_index_scacquire_fn   = hsa_queue_add_write_index_scacquire;
    table.hsa_queue_add_write_index_screlease_fn   = hsa_queue_add_write_index_screlease;
    table.hsa_queue_store_write_index_relaxed_fn   = hsa_queue_store_write_index_relaxed;
    table.hsa_queue_store_write_index_screlease_fn = hsa_queue_store_write_index_screlease;
    table.hsa_queue_cas_write_index_relaxed_fn     = hsa_queue_cas_write_index_relaxed;
    table.hsa_queue_cas_write_index_scacq_screl_fn = hsa_queue_cas_write_index_scacq_screl;
    table.hsa_queue_cas_write_index_scacquire_fn   = hsa_queue_cas_write_index_scacquire;
    table.hsa_queue_cas_write_index_screlease_fn   = hsa_queue_cas_write_index_screlease;
    table.hsa_queue_load_write_index_relaxed_fn    = hsa_queue_load_write_index_relaxed;
    table.hsa_queue_load_write_index_scacquire_fn  = hsa_queue_load_write_index_scacquire;
    table.hsa_signal_store_relaxed_fn              = hsa_signal_store_relaxed;
    table.hsa_signal_store_screlease_fn            = hsa_signal_store_screlease;
    table.hsa_signal_silent_store_relaxed_fn       = hsa_signal_silent_store_relaxed;
    table.hsa_signal_silent_store_screlease_fn     = hsa_signal_silent_store_screlease;
    return table;
}

void
make_dispatch_tracing_context(rocprofiler::context::context& ctx)
{
    ctx.callback_tracer = std::make_unique<rocprofiler::context::callback_tracing_service>();
    EXPECT_EQ(rocprofiler::context::add_domain(ctx.callback_tracer->domains,
                                               ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH),
              ROCPROFILER_STATUS_SUCCESS);
}

class QueueInterpositionConsumerTransition : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(hsa_init() != HSA_STATUS_SUCCESS)
        {
            GTEST_SKIP() << "hsa_init failed; consumer-transition tests need HSA";
        }
        fill_hsa_fn_tables();
        core_table_ = make_core_table();
        interposition_init(&core_table_, true);
        ASSERT_TRUE(supports_queue_interposition());
    }

    void TearDown() override
    {
        if(!supports_queue_interposition()) return;
        stop_completion_monitor();
        interposition_fini();
    }

    CoreApiTable core_table_{};
};

TEST_F(QueueInterpositionConsumerTransition, start_resyncs_stale_shadow)
{
    auto               dummy_queue = hsa_queue_t{};
    const hsa_queue_t* queue_ptr   = &dummy_queue;

    auto     state     = std::make_shared<QueueState>();
    uint64_t real_wdid = 17;
    uint64_t real_rdid = 3;
    state->real_wdid   = &real_wdid;
    state->real_rdid   = &real_rdid;
    state->virtual_wptr.store(0);
    state->next_scan_pos   = 0;
    state->next_submit_pos = 0;
    state->hsa_queue       = queue_ptr;

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = state; });

    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);
    notify_queue_interposition_consumer_context_started(&ctx);

    EXPECT_EQ(state->virtual_wptr.load(), 17u);
    EXPECT_EQ(state->next_scan_pos, 17u);
    EXPECT_EQ(state->next_submit_pos, 17u);

    notify_queue_interposition_consumer_context_stopped(&ctx);
    get_queue_registry().wlock([&](auto& registry) { registry.erase(queue_ptr); });
}

TEST_F(QueueInterpositionConsumerTransition, non_tracing_context_does_not_resync)
{
    auto               dummy_queue = hsa_queue_t{};
    const hsa_queue_t* queue_ptr   = &dummy_queue;

    auto     state     = std::make_shared<QueueState>();
    uint64_t real_wdid = 9;
    state->real_wdid   = &real_wdid;
    state->virtual_wptr.store(1);
    state->next_scan_pos   = 1;
    state->next_submit_pos = 1;

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = state; });

    auto ctx = rocprofiler::context::context{};
    notify_queue_interposition_consumer_context_started(&ctx);
    notify_queue_interposition_consumer_context_stopped(&ctx);

    EXPECT_EQ(state->virtual_wptr.load(), 1u);
    EXPECT_EQ(state->next_scan_pos, 1u);
    EXPECT_EQ(state->next_submit_pos, 1u);

    get_queue_registry().wlock([&](auto& registry) { registry.erase(queue_ptr); });
}

TEST_F(QueueInterpositionConsumerTransition, stop_then_start_resyncs_again)
{
    auto               dummy_queue = hsa_queue_t{};
    const hsa_queue_t* queue_ptr   = &dummy_queue;

    auto     state     = std::make_shared<QueueState>();
    uint64_t real_wdid = 4;
    state->real_wdid   = &real_wdid;
    state->virtual_wptr.store(4);
    state->next_scan_pos   = 4;
    state->next_submit_pos = 4;

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = state; });

    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);
    notify_queue_interposition_consumer_context_started(&ctx);
    notify_queue_interposition_consumer_context_stopped(&ctx);

    real_wdid = 12;
    notify_queue_interposition_consumer_context_started(&ctx);

    EXPECT_EQ(state->virtual_wptr.load(), 12u);
    EXPECT_EQ(state->next_scan_pos, 12u);
    EXPECT_EQ(state->next_submit_pos, 12u);

    notify_queue_interposition_consumer_context_stopped(&ctx);
    get_queue_registry().wlock([&](auto& registry) { registry.erase(queue_ptr); });
}

TEST_F(QueueInterpositionConsumerTransition, rapid_start_stop_does_not_hang)
{
    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};

    for(int i = 0; i < 50; ++i)
    {
        notify_queue_interposition_consumer_context_started(&ctx);
        notify_queue_interposition_consumer_context_stopped(&ctx);
        ASSERT_LT(std::chrono::steady_clock::now(), deadline)
            << "hung in consumer 0→1 / 1→0 after " << i << " cycles";
    }
}

// Fake hardware write index. Installed as the next-table write-index ops so a
// bypassing wrapper advances "HW" without a real HSA queue.
uint64_t g_stub_hw_wdid = 0;

// Holds a raw add on one queue inside the next-table call until released.
std::atomic<const hsa_queue_t*> g_stub_add_hold_queue{nullptr};
std::atomic<bool>               g_stub_add_held{false};
std::atomic<bool>               g_stub_add_release{false};

uint64_t
stub_add_write_index(const hsa_queue_t* queue, uint64_t value)
{
    if(queue && queue == g_stub_add_hold_queue.load())
    {
        g_stub_add_held.store(true);
        while(!g_stub_add_release.load())
            std::this_thread::yield();
    }
    return __atomic_fetch_add(&g_stub_hw_wdid, value, __ATOMIC_ACQ_REL);
}

void
stub_store_write_index(const hsa_queue_t*, uint64_t value)
{
    __atomic_store_n(&g_stub_hw_wdid, value, __ATOMIC_RELEASE);
}

uint64_t
stub_cas_write_index(const hsa_queue_t*, uint64_t expected, uint64_t value)
{
    auto cur = expected;
    __atomic_compare_exchange_n(
        &g_stub_hw_wdid, &cur, value, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    return cur;
}

// One-shot callback run from the forwarded raw write-index load.
std::function<void()> g_stub_load_hook = {};

uint64_t
stub_load_write_index(const hsa_queue_t*)
{
    const auto value = __atomic_load_n(&g_stub_hw_wdid, __ATOMIC_ACQUIRE);
    if(auto hook = std::exchange(g_stub_load_hook, nullptr)) hook();
    return value;
}

// Doorbell values forwarded to the next table; read only after the storing thread is joined.
std::vector<hsa_signal_value_t> g_stub_doorbells = {};

// One-shot callback run from the forwarded doorbell, e.g. under process_doorbell_impl's gate_lock.
std::function<void()> g_stub_doorbell_hook = {};

void
stub_signal_store(hsa_signal_t, hsa_signal_value_t value)
{
    g_stub_doorbells.emplace_back(value);
    if(auto hook = std::exchange(g_stub_doorbell_hook, nullptr)) hook();
}

class QueueInterpositionConsumerTransitionConcurrent : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(hsa_init() != HSA_STATUS_SUCCESS)
        {
            GTEST_SKIP() << "hsa_init failed; consumer-transition tests need HSA";
        }
        fill_hsa_fn_tables();
        g_stub_hw_wdid = 0;
        core_table_    = make_core_table();
        // Wire stubs in before init: interposition_init copies this table as the
        // next-in-chain that bypass wrappers call.
        core_table_.hsa_queue_add_write_index_relaxed_fn     = stub_add_write_index;
        core_table_.hsa_queue_add_write_index_scacq_screl_fn = stub_add_write_index;
        core_table_.hsa_queue_add_write_index_scacquire_fn   = stub_add_write_index;
        core_table_.hsa_queue_add_write_index_screlease_fn   = stub_add_write_index;
        core_table_.hsa_queue_store_write_index_relaxed_fn   = stub_store_write_index;
        core_table_.hsa_queue_store_write_index_screlease_fn = stub_store_write_index;
        core_table_.hsa_queue_cas_write_index_relaxed_fn     = stub_cas_write_index;
        core_table_.hsa_queue_cas_write_index_scacq_screl_fn = stub_cas_write_index;
        core_table_.hsa_queue_cas_write_index_scacquire_fn   = stub_cas_write_index;
        core_table_.hsa_queue_cas_write_index_screlease_fn   = stub_cas_write_index;
        core_table_.hsa_queue_load_write_index_relaxed_fn    = stub_load_write_index;
        core_table_.hsa_queue_load_write_index_scacquire_fn  = stub_load_write_index;
        core_table_.hsa_signal_store_relaxed_fn              = stub_signal_store;
        core_table_.hsa_signal_store_screlease_fn            = stub_signal_store;
        core_table_.hsa_signal_silent_store_relaxed_fn       = stub_signal_store;
        core_table_.hsa_signal_silent_store_screlease_fn     = stub_signal_store;
        g_stub_doorbells.clear();
        g_stub_doorbell_hook = nullptr;
        g_stub_load_hook     = nullptr;
        g_stub_add_hold_queue.store(nullptr);
        g_stub_add_held.store(false);
        g_stub_add_release.store(false);
        interposition_init(&core_table_, true);
        ASSERT_TRUE(supports_queue_interposition());
    }

    void TearDown() override
    {
        if(!supports_queue_interposition()) return;
        stop_completion_monitor();
        interposition_fini();
    }

    CoreApiTable core_table_{};
};

// 0→1 with live bypass traffic: several threads call the installed write-index
// wrappers (so inflight counting and should_bypass run) while consumer start
// resyncs shadow. Start must complete within a deadline, and afterward shadow
// must not sit behind the hardware write index (that inversion is the hang).
TEST_F(QueueInterpositionConsumerTransitionConcurrent,
       start_with_concurrent_bypass_producers_keeps_shadow_ahead_of_hw)
{
    auto               dummy_queue = hsa_queue_t{};
    const hsa_queue_t* queue_ptr   = &dummy_queue;

    auto state       = std::make_shared<QueueState>();
    state->real_wdid = &g_stub_hw_wdid;
    state->virtual_wptr.store(0);
    state->next_scan_pos   = 0;
    state->next_submit_pos = 0;
    state->hsa_queue       = queue_ptr;

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = state; });

    std::atomic<bool> run{true};
    std::atomic<bool> pause{false};
    std::atomic<int>  paused{0};
    auto              producer = [&]() {
        while(run.load(std::memory_order_relaxed))
        {
            if(pause.load(std::memory_order_acquire))
            {
                paused.fetch_add(1, std::memory_order_acq_rel);
                while(pause.load(std::memory_order_acquire) && run.load(std::memory_order_relaxed))
                    std::this_thread::sleep_for(std::chrono::microseconds{50});
                paused.fetch_sub(1, std::memory_order_acq_rel);
                continue;
            }
            // Wrapper entry point (not the stub): exercises bypass inflight + gate.
            core_table_.hsa_queue_add_write_index_relaxed_fn(queue_ptr, 1);
        }
    };

    constexpr int            k_producers = 4;
    std::vector<std::thread> threads;
    threads.reserve(k_producers);
    for(int i = 0; i < k_producers; ++i)
        threads.emplace_back(producer);

    // Join producers on every exit (including ASSERT failure) so the vector
    // destructor never sees joinable threads and calls std::terminate.
    auto join_producers = common::scope_destructor{[&]() {
        run.store(false, std::memory_order_relaxed);
        pause.store(false, std::memory_order_release);
        for(auto& t : threads)
        {
            if(t.joinable()) t.join();
        }
    }};

    const auto warm_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while(__atomic_load_n(&g_stub_hw_wdid, __ATOMIC_ACQUIRE) < 256u &&
          std::chrono::steady_clock::now() < warm_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    ASSERT_GE(__atomic_load_n(&g_stub_hw_wdid, __ATOMIC_ACQUIRE), 256u)
        << "producers failed to advance stub HW write index under bypass";

    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);

    auto start_fut = std::async(
        std::launch::async, [&]() { notify_queue_interposition_consumer_context_started(&ctx); });
    ASSERT_EQ(start_fut.wait_for(std::chrono::seconds{10}), std::future_status::ready)
        << "0→1 hung with concurrent bypass producers";
    start_fut.get();

    // Stop further advances so the shadow-vs-HW check is stable.
    pause.store(true, std::memory_order_release);
    const auto pause_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while(paused.load(std::memory_order_acquire) < k_producers &&
          std::chrono::steady_clock::now() < pause_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    ASSERT_EQ(paused.load(std::memory_order_acquire), k_producers);

    const uint64_t hw   = __atomic_load_n(&g_stub_hw_wdid, __ATOMIC_ACQUIRE);
    const uint64_t vwp  = state->virtual_wptr.load();
    const uint64_t scan = state->next_scan_pos;
    // next_scan_pos is only advanced by resync/doorbell publish — not by post-cutover
    // intercept write-index adds — so it exposes HW races that virtual_wptr can mask.
    EXPECT_GE(scan, hw) << "scan/shadow behind HW after 0→1 with concurrent bypass producers"
                        << " (scan=" << scan << " hw=" << hw << " vwp=" << vwp << ")";
    EXPECT_GE(vwp, hw) << "virtual_wptr behind HW after 0→1 with concurrent bypass producers";

    notify_queue_interposition_consumer_context_stopped(&ctx);
    get_queue_registry().wlock([&](auto& registry) { registry.erase(queue_ptr); });
}

// Registered queue whose real write index is g_stub_hw_wdid and whose doorbell signal is a fake
// doorbell-kind amd_signal_t, so the installed wrappers run without a real HSA queue.
struct stub_doorbell_queue
{
    stub_doorbell_queue()
    {
        for(uint64_t i = 0; i < k_ring_size; ++i)
            get_pkt(ring, i, k_ring_size - 1)->header = k_retired_header;
        queue.id = ~uint64_t{0};  // no QueueController queue matches, so doorbells stay untraced

        amd_doorbell.kind      = AMD_SIGNAL_KIND_DOORBELL;
        amd_doorbell.queue_ptr = reinterpret_cast<amd_queue_v2_t*>(&queue);
        doorbell               = hsa_signal_t{.handle = reinterpret_cast<uint64_t>(&amd_doorbell)};

        state->ring_buf        = ring;
        state->ring_size       = k_ring_size;
        state->ring_mask       = k_ring_size - 1;
        state->real_wdid       = &g_stub_hw_wdid;
        state->real_rdid       = &real_rdid;
        state->hsa_queue       = &queue;
        state->doorbell_signal = doorbell;
        get_queue_registry().wlock([this](auto& registry) { registry[&queue] = state; });
    }

    ~stub_doorbell_queue()
    {
        get_queue_registry().wlock([this](auto& registry) { registry.erase(&queue); });
    }

    stub_doorbell_queue(const stub_doorbell_queue&) = delete;

    stub_doorbell_queue& operator=(const stub_doorbell_queue&) = delete;

    void write_packet(uint64_t idx, uint64_t kernel_object)
    {
        auto* pkt          = get_pkt(ring, idx, k_ring_size - 1);
        pkt->kernel_object = kernel_object;
        __atomic_store_n(&pkt->header, k_dispatch_header, __ATOMIC_RELEASE);
    }

    static constexpr uint64_t k_ring_size = 256;

    alignas(64) char ring[64 * k_ring_size]  = {};
    hsa_queue_t                 queue        = {};
    amd_signal_t                amd_doorbell = {};
    hsa_signal_t                doorbell     = {};
    uint64_t                    real_rdid    = 0;
    std::shared_ptr<QueueState> state        = std::make_shared<QueueState>();
};

// A slot claimed while tracing whose doorbell comes after the last consumer stops must still be
// published to the real write index, and a raw claim made before that doorbell must not reuse it.
TEST_F(QueueInterpositionConsumerTransitionConcurrent,
       stop_with_unpublished_shadow_claim_keeps_slots_and_publishes)
{
    auto q   = stub_doorbell_queue{};
    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);

    notify_queue_interposition_consumer_context_started(&ctx);
    const uint64_t old_slot = core_table_.hsa_queue_add_write_index_scacq_screl_fn(&q.queue, 1);
    notify_queue_interposition_consumer_context_stopped(&ctx);

    ASSERT_EQ(old_slot, 0u);
    ASSERT_EQ(q.state->virtual_wptr.load(), 1u) << "tracing did not intercept the old claim";
    ASSERT_EQ(g_stub_hw_wdid, 0u);

    const uint64_t new_slot = core_table_.hsa_queue_add_write_index_scacq_screl_fn(&q.queue, 1);
    EXPECT_EQ(new_slot, 1u) << "raw claim reused the unpublished shadow slot";

    q.write_packet(old_slot, 0xA0);
    q.write_packet(1, 0xA1);
    core_table_.hsa_signal_store_screlease_fn(q.doorbell,
                                              static_cast<hsa_signal_value_t>(old_slot));
    core_table_.hsa_signal_store_screlease_fn(q.doorbell, 1);

    EXPECT_EQ(g_stub_hw_wdid, 2u);
    EXPECT_EQ(q.state->virtual_wptr.load(), 2u);
    ASSERT_FALSE(g_stub_doorbells.empty());
    EXPECT_EQ(g_stub_doorbells.back(), 1);
    for(auto value : g_stub_doorbells)
        EXPECT_LT(static_cast<uint64_t>(value), g_stub_hw_wdid) << "doorbell past published slots";
    for(uint64_t i = 0; i < 2; ++i)
    {
        EXPECT_EQ(get_pkt(q.ring, i, q.k_ring_size - 1)->header, k_dispatch_header);
        EXPECT_EQ(get_pkt(q.ring, i, q.k_ring_size - 1)->kernel_object, 0xA0 + i);
    }

    // Caught up: claims return to the raw write index.
    EXPECT_EQ(core_table_.hsa_queue_add_write_index_scacq_screl_fn(&q.queue, 1), 2u);
    EXPECT_EQ(g_stub_hw_wdid, 3u);
    EXPECT_EQ(q.state->virtual_wptr.load(), 2u);
}

// Same handoff without a later claim: the old doorbell alone must publish the real write index,
// otherwise the next raw claim reuses a slot the GPU already consumed.
TEST_F(QueueInterpositionConsumerTransitionConcurrent,
       stop_with_unpublished_shadow_claim_publishes_on_doorbell)
{
    auto q   = stub_doorbell_queue{};
    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);

    notify_queue_interposition_consumer_context_started(&ctx);
    const uint64_t old_slot = core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1);
    notify_queue_interposition_consumer_context_stopped(&ctx);

    ASSERT_EQ(old_slot, 0u);
    ASSERT_EQ(q.state->virtual_wptr.load(), 1u) << "tracing did not intercept the old claim";

    q.write_packet(old_slot, 0xB0);
    core_table_.hsa_signal_store_relaxed_fn(q.doorbell, static_cast<hsa_signal_value_t>(old_slot));

    EXPECT_EQ(g_stub_hw_wdid, 1u);
    EXPECT_EQ(core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1), 1u);
    EXPECT_EQ(g_stub_hw_wdid, 2u);
}

// A restart before the old doorbell must not resync the unflushed claim away.
TEST_F(QueueInterpositionConsumerTransitionConcurrent,
       restart_with_unpublished_shadow_claim_keeps_claim)
{
    auto q   = stub_doorbell_queue{};
    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);

    notify_queue_interposition_consumer_context_started(&ctx);
    const uint64_t old_slot = core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1);
    notify_queue_interposition_consumer_context_stopped(&ctx);
    ASSERT_EQ(old_slot, 0u);
    ASSERT_EQ(q.state->virtual_wptr.load(), 1u) << "tracing did not intercept the old claim";

    notify_queue_interposition_consumer_context_started(&ctx);
    EXPECT_EQ(q.state->virtual_wptr.load(), 1u);
    q.write_packet(old_slot, 0xC0);
    core_table_.hsa_signal_store_relaxed_fn(q.doorbell, static_cast<hsa_signal_value_t>(old_slot));
    const uint64_t next_slot = core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1);
    notify_queue_interposition_consumer_context_stopped(&ctx);

    EXPECT_EQ(g_stub_hw_wdid, 1u);
    EXPECT_EQ(next_slot, 1u);
}

// A raw add admitted before the start seal must finish before resync: start waits on it, and the
// next-table call runs without claim_mu or gate_lock held.
TEST_F(QueueInterpositionConsumerTransitionConcurrent, start_waits_for_admitted_raw_add)
{
    auto q   = stub_doorbell_queue{};
    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);
    g_stub_add_hold_queue.store(&q.queue);

    uint64_t          raw_slot = ~uint64_t{0};
    std::thread       producer;
    std::future<void> start_fut;
    auto              cleanup = common::scope_destructor{[&]() {
        g_stub_add_release.store(true);
        if(producer.joinable()) producer.join();
        if(start_fut.valid())
        {
            start_fut.get();
            notify_queue_interposition_consumer_context_stopped(&ctx);
        }
    }};

    producer = std::thread{
        [&]() { raw_slot = core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1); }};
    const auto held_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while(!g_stub_add_held.load() && std::chrono::steady_clock::now() < held_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    ASSERT_TRUE(g_stub_add_held.load()) << "bypass add never reached the next table";

    ASSERT_TRUE(q.state->claim_mu.try_lock()) << "claim_mu held across the next-table call";
    q.state->claim_mu.unlock();
    ASSERT_TRUE(q.state->gate_lock.try_lock()) << "gate_lock held across the next-table call";
    q.state->gate_lock.unlock();

    start_fut = std::async(std::launch::async,
                           [&]() { notify_queue_interposition_consumer_context_started(&ctx); });
    EXPECT_EQ(start_fut.wait_for(std::chrono::milliseconds{200}), std::future_status::timeout)
        << "start finished while a root admitted before the seal was still in its raw add";

    g_stub_add_release.store(true);
    producer.join();
    ASSERT_EQ(start_fut.wait_for(std::chrono::seconds{10}), std::future_status::ready);

    EXPECT_EQ(raw_slot, 0u);
    EXPECT_EQ(g_stub_hw_wdid, 1u);
    EXPECT_EQ(q.state->virtual_wptr.load(), 1u);
    EXPECT_EQ(q.state->next_scan_pos, 1u);
    EXPECT_EQ(q.state->next_submit_pos, 1u);
    EXPECT_EQ(core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1), 1u);
    EXPECT_EQ(g_stub_hw_wdid, 1u);
}

// A write-index call nested in a doorbell callback runs with process_doorbell_impl's gate_lock
// held on the same thread; it must claim through claim_mu and still stay on unflushed shadow.
TEST_F(QueueInterpositionConsumerTransitionConcurrent,
       nested_claim_under_held_gate_stays_on_unflushed_shadow)
{
    auto q   = stub_doorbell_queue{};
    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);

    notify_queue_interposition_consumer_context_started(&ctx);
    const uint64_t slot0 = core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1);
    const uint64_t slot1 = core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1);
    notify_queue_interposition_consumer_context_stopped(&ctx);
    ASSERT_EQ(slot0, 0u);
    ASSERT_EQ(slot1, 1u);
    ASSERT_EQ(q.state->virtual_wptr.load(), 2u) << "tracing did not intercept the old claims";

    // Only slot 0 is written, so the doorbell publishes 1 and slot 1 stays unflushed.
    q.write_packet(slot0, 0xD0);
    uint64_t nested_slot = ~uint64_t{0};
    g_stub_doorbell_hook = [&]() {
        nested_slot = core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1);
    };
    core_table_.hsa_signal_store_relaxed_fn(q.doorbell, static_cast<hsa_signal_value_t>(slot0));

    EXPECT_EQ(nested_slot, 2u) << "nested raw claim reused an unflushed shadow slot";
    EXPECT_EQ(g_stub_hw_wdid, 1u);

    q.write_packet(slot1, 0xD1);
    q.write_packet(2, 0xD2);
    core_table_.hsa_signal_store_relaxed_fn(q.doorbell, 2);
    EXPECT_EQ(g_stub_hw_wdid, 3u);
    EXPECT_EQ(q.state->virtual_wptr.load(), 3u);
}

// Bounded poll for a condition another thread establishes; false if it never does.
template <typename PredT>
bool
wait_until(PredT&& pred)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while(!pred())
    {
        if(std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

// A wrapper stuck behind a seal leaves the process unusable for later tests, so a deadlock aborts.
void
require_ready_or_abort(std::future<void>& fut, const char* what)
{
    if(fut.wait_for(std::chrono::seconds{10}) == std::future_status::ready) return;
    std::fprintf(stderr, "queue_interposition test deadlocked: %s\n", what);
    std::abort();
}

// A tracing callback on an ordinary (non-doorbell) signal store may stop the last consumer: the
// store is not an admitted root, so the stop's seal does not wait on the thread running it.
TEST_F(QueueInterpositionConsumerTransitionConcurrent, user_signal_callback_can_stop_last_consumer)
{
    auto q   = stub_doorbell_queue{};
    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);
    notify_queue_interposition_consumer_context_started(&ctx);
    auto cleanup = common::scope_destructor{
        [&]() { notify_queue_interposition_consumer_context_stopped(&ctx); }};

    auto amd_user        = amd_signal_t{};
    amd_user.kind        = AMD_SIGNAL_KIND_USER;
    const auto user_sig  = hsa_signal_t{.handle = reinterpret_cast<uint64_t>(&amd_user)};
    g_stub_doorbell_hook = [&]() { notify_queue_interposition_consumer_context_stopped(&ctx); };

    auto store = std::async(std::launch::async,
                            [&]() { core_table_.hsa_signal_store_screlease_fn(user_sig, 7); });
    require_ready_or_abort(store, "last-consumer stop from a user-signal store callback");
    store.get();

    ASSERT_EQ(g_stub_doorbells.size(), 1u);
    EXPECT_EQ(g_stub_doorbells.front(), 7);
    EXPECT_FALSE(get_root_admission_snapshot().sealed);
    EXPECT_EQ(get_root_admission_snapshot().admitted_roots, 0u);
    // Tracing really stopped: a caught-up queue now claims raw.
    EXPECT_EQ(core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1), 0u);
    EXPECT_EQ(g_stub_hw_wdid, 1u);
    EXPECT_EQ(q.state->virtual_wptr.load(), 0u);
}

// A raw write-index load's callback may start the first consumer: a load is not an admitted root,
// so the start's seal does not wait on the thread running it.
TEST_F(QueueInterpositionConsumerTransitionConcurrent, raw_load_callback_can_start_first_consumer)
{
    auto q   = stub_doorbell_queue{};
    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);

    bool started     = false;
    g_stub_load_hook = [&]() {
        notify_queue_interposition_consumer_context_started(&ctx);
        started = true;
    };
    auto cleanup = common::scope_destructor{[&]() {
        if(started) notify_queue_interposition_consumer_context_stopped(&ctx);
    }};

    uint64_t loaded = ~uint64_t{0};
    auto     load   = std::async(std::launch::async, [&]() {
        loaded = core_table_.hsa_queue_load_write_index_scacquire_fn(&q.queue);
    });
    require_ready_or_abort(load, "first-consumer start from a raw load callback");
    load.get();

    ASSERT_TRUE(started) << "the raw load never reached the next table";
    EXPECT_EQ(loaded, 0u);
    EXPECT_FALSE(get_root_admission_snapshot().sealed);
    EXPECT_EQ(get_root_admission_snapshot().admitted_roots, 0u);
    // Tracing really started: the next claim lands on shadow.
    EXPECT_EQ(core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1), 0u);
    EXPECT_EQ(q.state->virtual_wptr.load(), 1u);
    EXPECT_EQ(g_stub_hw_wdid, 0u);
}

// An add admitted under tracing is parked (on the registry lock) after selecting shadow mode but
// before mutating shadow. The last stop seals and must wait for it, so the claim lands on shadow.
TEST_F(QueueInterpositionConsumerTransitionConcurrent, stop_waits_for_admitted_shadow_add)
{
    auto q   = stub_doorbell_queue{};
    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);
    notify_queue_interposition_consumer_context_started(&ctx);

    std::atomic<bool> registry_held{false};
    std::atomic<bool> release_registry{false};
    uint64_t          paused_slot = ~uint64_t{0};
    std::thread       holder;
    std::thread       producer;
    std::future<void> stop_fut;
    auto              cleanup = common::scope_destructor{[&]() {
        release_registry.store(true);
        if(holder.joinable()) holder.join();
        if(producer.joinable()) producer.join();
        if(stop_fut.valid()) stop_fut.get();
        notify_queue_interposition_consumer_context_stopped(&ctx);
    }};

    holder = std::thread{[&]() {
        get_queue_registry().wlock([&](auto&) {
            registry_held.store(true);
            while(!release_registry.load())
                std::this_thread::yield();
        });
    }};
    ASSERT_TRUE(wait_until([&]() { return registry_held.load(); }));

    producer = std::thread{
        [&]() { paused_slot = core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1); }};
    ASSERT_TRUE(wait_until([]() { return get_root_admission_snapshot().admitted_roots == 1; }))
        << "the add was never admitted";

    stop_fut = std::async(std::launch::async,
                          [&]() { notify_queue_interposition_consumer_context_stopped(&ctx); });
    ASSERT_TRUE(wait_until([]() { return get_root_admission_snapshot().sealed; }))
        << "stop never sealed admission";
    // Stop is parked in the seal (not later in its gate fence): the admitted root is still counted.
    EXPECT_EQ(get_root_admission_snapshot().admitted_roots, 1u);
    EXPECT_EQ(stop_fut.wait_for(std::chrono::milliseconds{100}), std::future_status::timeout)
        << "stop finished while an admitted shadow add had not mutated shadow";
    EXPECT_EQ(q.state->virtual_wptr.load(), 0u);

    release_registry.store(true);
    holder.join();
    producer.join();
    require_ready_or_abort(stop_fut, "last-consumer stop after releasing the admitted add");
    stop_fut.get();

    EXPECT_EQ(paused_slot, 0u);
    EXPECT_EQ(q.state->virtual_wptr.load(), 1u);
    EXPECT_EQ(g_stub_hw_wdid, 0u) << "the admitted add claimed raw instead of shadow";

    // The next claim must not reuse the unflushed slot, and both doorbells publish in order.
    EXPECT_EQ(core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1), 1u);
    q.write_packet(0, 0xE0);
    q.write_packet(1, 0xE1);
    core_table_.hsa_signal_store_relaxed_fn(q.doorbell, 0);
    core_table_.hsa_signal_store_relaxed_fn(q.doorbell, 1);
    EXPECT_EQ(g_stub_hw_wdid, 2u);
    EXPECT_EQ(q.state->virtual_wptr.load(), 2u);
}

// While stop's seal is pending on an admitted doorbell root, a same-thread nested claim from that
// root's callback must not wait on the seal, and it keeps the root's tracing mode.
TEST_F(QueueInterpositionConsumerTransitionConcurrent, nested_claim_while_stop_seal_pending)
{
    auto q   = stub_doorbell_queue{};
    auto ctx = rocprofiler::context::context{};
    make_dispatch_tracing_context(ctx);
    notify_queue_interposition_consumer_context_started(&ctx);

    const uint64_t slot0 = core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1);
    ASSERT_EQ(slot0, 0u);
    q.write_packet(slot0, 0xF0);

    std::atomic<bool> in_hook{false};
    bool              saw_seal    = false;
    uint64_t          nested_slot = ~uint64_t{0};
    g_stub_doorbell_hook          = [&]() {
        in_hook.store(true);
        saw_seal    = wait_until([]() { return get_root_admission_snapshot().sealed; });
        nested_slot = core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1);
    };

    std::future<void> stop_fut;
    auto              doorbell_fut = std::async(std::launch::async, [&]() {
        core_table_.hsa_signal_store_relaxed_fn(q.doorbell, static_cast<hsa_signal_value_t>(slot0));
    });
    auto              cleanup      = common::scope_destructor{[&]() {
        if(doorbell_fut.valid()) doorbell_fut.get();
        if(stop_fut.valid()) stop_fut.get();
        notify_queue_interposition_consumer_context_stopped(&ctx);
    }};

    ASSERT_TRUE(wait_until([&]() { return in_hook.load(); })) << "doorbell callback never ran";
    stop_fut = std::async(std::launch::async,
                          [&]() { notify_queue_interposition_consumer_context_stopped(&ctx); });
    require_ready_or_abort(doorbell_fut, "nested claim while the stop seal is pending");
    doorbell_fut.get();
    require_ready_or_abort(stop_fut, "last-consumer stop after the doorbell root returned");
    stop_fut.get();

    EXPECT_TRUE(saw_seal) << "stop did not seal while the doorbell root was in its callback";
    EXPECT_EQ(nested_slot, 1u);
    EXPECT_EQ(g_stub_hw_wdid, 1u);
    EXPECT_EQ(q.state->virtual_wptr.load(), 2u);

    // Slot 1 is still unflushed after stop, so the next claim stays on shadow.
    EXPECT_EQ(core_table_.hsa_queue_add_write_index_relaxed_fn(&q.queue, 1), 2u);
    q.write_packet(1, 0xF1);
    q.write_packet(2, 0xF2);
    core_table_.hsa_signal_store_relaxed_fn(q.doorbell, 2);
    EXPECT_EQ(g_stub_hw_wdid, 3u);
    EXPECT_EQ(q.state->virtual_wptr.load(), 3u);
}
}  // namespace
}  // namespace queue_interposition
}  // namespace hsa
}  // namespace rocprofiler
