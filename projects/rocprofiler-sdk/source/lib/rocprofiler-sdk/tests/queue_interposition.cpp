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
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <gtest/gtest.h>

#include <hsa/amd_hsa_queue.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <chrono>
#include <cstring>
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
    table.hsa_queue_load_write_index_relaxed_fn   = hsa_queue_load_write_index_relaxed;
    table.hsa_queue_load_write_index_scacquire_fn = hsa_queue_load_write_index_scacquire;
    table.hsa_signal_store_relaxed_fn              = hsa_signal_store_relaxed;
    table.hsa_signal_store_screlease_fn            = hsa_signal_store_screlease;
    table.hsa_signal_silent_store_relaxed_fn       = hsa_signal_silent_store_relaxed;
    table.hsa_signal_silent_store_screlease_fn     = hsa_signal_silent_store_screlease;
    return table;
}

rocprofiler::context::context
make_dispatch_tracing_context()
{
    auto ctx = rocprofiler::context::context{};
    ctx.callback_tracer = std::make_unique<rocprofiler::context::callback_tracing_service>();
    EXPECT_EQ(rocprofiler::context::add_domain(ctx.callback_tracer->domains,
                                               ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH),
              ROCPROFILER_STATUS_SUCCESS);
    return ctx;
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

    auto     state    = std::make_shared<QueueState>();
    uint64_t real_wdid = 17;
    uint64_t real_rdid = 3;
    state->real_wdid       = &real_wdid;
    state->real_rdid       = &real_rdid;
    state->virtual_wptr.store(0);
    state->next_scan_pos   = 0;
    state->next_submit_pos = 0;
    state->hsa_queue       = queue_ptr;

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = state; });

    auto ctx = make_dispatch_tracing_context();
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
    state->real_wdid       = &real_wdid;
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
    state->real_wdid       = &real_wdid;
    state->virtual_wptr.store(4);
    state->next_scan_pos   = 4;
    state->next_submit_pos = 4;

    get_queue_registry().wlock([&](auto& registry) { registry[queue_ptr] = state; });

    auto ctx = make_dispatch_tracing_context();
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
    auto ctx      = make_dispatch_tracing_context();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};

    for(int i = 0; i < 50; ++i)
    {
        notify_queue_interposition_consumer_context_started(&ctx);
        notify_queue_interposition_consumer_context_stopped(&ctx);
        ASSERT_LT(std::chrono::steady_clock::now(), deadline)
            << "hung in consumer 0→1 / 1→0 after " << i << " cycles";
    }
}
}  // namespace
}  // namespace queue_interposition
}  // namespace hsa
}  // namespace rocprofiler
