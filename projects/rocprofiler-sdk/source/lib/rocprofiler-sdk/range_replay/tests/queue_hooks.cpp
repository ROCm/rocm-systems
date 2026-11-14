// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/range_replay/queue_hooks.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/replay_window.hpp"
#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <thread>

namespace range_replay = ::rocprofiler::range_replay;

namespace
{
AmdExtTable
noop_ext_table()
{
    auto ext                                  = AmdExtTable{};
    ext.hsa_amd_agent_iterate_memory_pools_fn = [](hsa_agent_t,
                                                   hsa_status_t (*)(hsa_amd_memory_pool_t, void*),
                                                   void*) { return HSA_STATUS_SUCCESS; };
    return ext;
}

CoreApiTable
noop_core_table()
{
    auto core = CoreApiTable{};
    core.hsa_signal_create_fn =
        [](hsa_signal_value_t, uint32_t, const hsa_agent_t*, hsa_signal_t* signal) {
            signal->handle = 0;
            return HSA_STATUS_SUCCESS;
        };
    return core;
}
}  // namespace

// The early-return tests pass a null queue. A regression that stops the hook returning early aborts
// the test instead of passing quietly.

TEST(range_replay_queue_hooks, inert_when_no_range_is_open)
{
    ASSERT_FALSE(range_replay::any_range_open());

    range_replay::submission_hook(nullptr, nullptr, 0, false, nullptr);

    EXPECT_FALSE(range_replay::any_range_open());
    EXPECT_EQ(range_replay::current_range(), nullptr);
}

TEST(range_replay_queue_hooks, ignores_a_thread_that_is_resubmitting_a_replay)
{
    // any_range_open() is process-wide, so an open range here makes the replaying guard the only
    // thing standing between the hook and its queue.
    ASSERT_TRUE(range_replay::open_range(7));
    auto _take = rocprofiler::common::scope_destructor{[]() {
        auto taken = range_replay::range_context_t{};
        range_replay::take_range(taken);
    }};

    range_replay::set_this_thread_replaying(true);
    range_replay::submission_hook(nullptr, nullptr, 1, false, nullptr);
    range_replay::set_this_thread_replaying(false);

    auto* open = range_replay::current_range();
    ASSERT_NE(open, nullptr);
    EXPECT_TRUE(open->record.eligible());
    EXPECT_EQ(open->record.observed_dispatch_count(), 0U)
        << "a replay's own re-submission must not be seen by a range at all";
}

TEST(range_replay_queue_hooks, own_range_submission_also_notifies_foreign_ranges)
{
    auto agent      = rocprofiler_agent_t{};
    agent.id.handle = 0x2001;
    agent.name      = "test";
    auto ext        = noop_ext_table();
    auto core       = noop_core_table();
    auto cache =
        rocprofiler::hsa::AgentCache{&agent, {.handle = 0x2001}, 0, {.handle = 0}, ext, core};
    auto queue = rocprofiler::hsa::Queue{cache, core};

    ASSERT_TRUE(range_replay::open_range(1));
    range_replay::set_host_mode(*range_replay::current_range());
    auto _take = rocprofiler::common::scope_destructor{[]() {
        auto taken = range_replay::range_context_t{};
        range_replay::take_range(taken);
    }};

    std::thread{[&]() {
        ASSERT_TRUE(range_replay::open_range(2));
        range_replay::set_host_mode(*range_replay::current_range());
        auto packet                   = rocprofiler::hsa::rocprofiler_packet{};
        packet.kernel_dispatch.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
        const auto original           = packet;
        range_replay::submission_hook(&queue, &packet, 1, false, nullptr);
        EXPECT_EQ(std::memcmp(&packet, &original, sizeof(packet)), 0);

        auto taken = range_replay::range_context_t{};
        ASSERT_TRUE(range_replay::take_range(taken));
        EXPECT_TRUE(taken.record.eligible());
        EXPECT_EQ(taken.record.observed_dispatch_count(), 1U);
    }}.join();

    auto* open = range_replay::current_range();
    range_replay::fold_external_decline(*open);
    EXPECT_EQ(open->record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH);
}

// A non-host range is not packet-replayed, so the submit hook must not capture device memory.
// The old hook called ensure_entry_snapshot here. snapshot_taken stays false, and the range is
// not declined as SNAPSHOT_FAILED.
TEST(range_replay_queue_hooks, non_host_submission_does_not_snapshot)
{
    auto agent      = rocprofiler_agent_t{};
    agent.id.handle = 0x2002;
    agent.name      = "test";
    auto ext        = noop_ext_table();
    auto core       = noop_core_table();
    auto cache =
        rocprofiler::hsa::AgentCache{&agent, {.handle = 0x2002}, 0, {.handle = 0}, ext, core};
    auto queue = rocprofiler::hsa::Queue{cache, core};

    ASSERT_TRUE(range_replay::open_range(9));
    auto* open                  = range_replay::current_range();
    open->plan.replay_requested = true;
    auto _take                  = rocprofiler::common::scope_destructor{[]() {
        auto taken = range_replay::range_context_t{};
        range_replay::take_range(taken);
    }};

    range_replay::submission_hook(&queue, nullptr, 0, false, nullptr);

    EXPECT_FALSE(open->host.enabled);
    EXPECT_FALSE(open->snapshot_taken);
    EXPECT_NE(open->record.status(), ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
    EXPECT_TRUE(open->record.eligible());
}

TEST(range_replay_queue_hooks, non_compute_queues_are_not_replay_ring_targets)
{
    auto agent      = rocprofiler_agent_t{};
    agent.id.handle = 0x2001;
    agent.name      = "test";
    auto ext        = noop_ext_table();
    auto core       = noop_core_table();
    auto cache =
        rocprofiler::hsa::AgentCache{&agent, {.handle = 0x2001}, 0, {.handle = 0}, ext, core};
    auto dummy      = hsa_queue_t{};
    auto controller = rocprofiler::hsa::QueueController{};
    controller.add_queue(
        &dummy, std::make_unique<rocprofiler::hsa::Queue>(cache, core), /*is_compute=*/false);

    const auto* queue = controller.get_queue(dummy);
    ASSERT_NE(queue, nullptr);
    EXPECT_EQ(queue->get_engine_type(), rocprofiler::hsa::Queue::engine_type::non_compute);
    auto packet = rocprofiler::hsa::rocprofiler_packet{};
    EXPECT_FALSE(rocprofiler::hsa::replay_ring_submit(*queue, &packet, 1));
}
