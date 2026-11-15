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

#include "lib/rocprofiler-sdk/hsa/replay_window.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include "fake_queue.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

namespace hsa = ::rocprofiler::hsa;
namespace rrt = ::rocprofiler::range_replay::test;

using namespace std::chrono_literals;

namespace
{
constexpr uint64_t agent_handle = 0x2001;
constexpr auto     short_budget = hsa::replay_wait_budget_t{1ms, 3};
}  // namespace

TEST(replay_window, replay_wait_gives_up_after_its_budget)
{
    auto tries = 0;
    EXPECT_FALSE(hsa::replay_wait(
        [&tries]() {
            ++tries;
            return false;
        },
        "a wait that never drains",
        short_budget));
    EXPECT_EQ(tries, short_budget.slices);
}

TEST(replay_window, replay_wait_returns_at_the_first_try_that_drains)
{
    auto tries = 0;
    EXPECT_TRUE(hsa::replay_wait(
        [&tries]() { return ++tries == 2; }, "a wait that drains on its second try", short_budget));
    EXPECT_EQ(tries, 2);
}

TEST(replay_window, agent_drain_gives_up_while_a_handler_is_in_flight)
{
    auto  agent      = rrt::test_agent_t{agent_handle};
    auto& controller = *hsa::get_queue_controller();
    auto  hsa_queue  = hsa_queue_t{};
    hsa_queue.id     = 3;
    auto& queue      = rrt::add_test_queue(controller, hsa_queue, agent);

    const auto hsa_agent = hsa_agent_t{.handle = agent_handle};
    EXPECT_TRUE(hsa::replay_drain_agent(hsa_agent, short_budget));
    queue.async_started();
    EXPECT_FALSE(hsa::replay_drain_agent(hsa_agent, short_budget));
    queue.async_complete();
    EXPECT_TRUE(hsa::replay_drain_agent(hsa_agent, short_budget));

    controller.destroy_queue(&hsa_queue);
}

TEST(replay_window, agent_drain_counts_a_retiring_queue_against_its_budget)
{
    auto  agent      = rrt::test_agent_t{agent_handle};
    auto& controller = *hsa::get_queue_controller();
    auto  hsa_queue  = hsa_queue_t{};
    hsa_queue.id     = 4;
    auto& queue      = rrt::add_test_queue(controller, hsa_queue, agent);

    auto destroy = rrt::in_flight_destroy_t{controller, hsa_queue, queue};
    ASSERT_TRUE(rrt::wait_until_unmapped(controller));

    const auto hsa_agent = hsa_agent_t{.handle = agent_handle};
    EXPECT_FALSE(hsa::replay_drain_agent(hsa_agent, short_budget));

    destroy.finish();
    EXPECT_TRUE(hsa::replay_drain_agent(hsa_agent, short_budget));
}

// destroy_queue erases a queue from the map before it waits for the queue's completion handlers,
// so the retiring set is the only place a drain can still see them.
TEST(replay_window, a_queue_being_destroyed_stays_counted_until_its_sync_ends)
{
    auto agent      = rrt::test_agent_t{agent_handle};
    auto controller = hsa::QueueController{};
    auto hsa_queue  = hsa_queue_t{};
    hsa_queue.id    = 1;
    auto& queue     = rrt::add_test_queue(controller, hsa_queue, agent);

    auto destroy = rrt::in_flight_destroy_t{controller, hsa_queue, queue};
    ASSERT_TRUE(rrt::wait_until_unmapped(controller));

    const auto hsa_agent = hsa_agent_t{.handle = agent_handle};
    EXPECT_EQ(controller.in_flight_async_packets(hsa_agent), 1);

    destroy.finish();
    EXPECT_EQ(controller.in_flight_async_packets(hsa_agent), 0);
}

TEST(replay_window, agent_drain_waits_for_a_queue_whose_destroy_is_still_syncing)
{
    auto  agent      = rrt::test_agent_t{agent_handle};
    auto& controller = *hsa::get_queue_controller();
    auto  hsa_queue  = hsa_queue_t{};
    hsa_queue.id     = 2;
    auto& queue      = rrt::add_test_queue(controller, hsa_queue, agent);

    auto destroy = rrt::in_flight_destroy_t{controller, hsa_queue, queue};
    ASSERT_TRUE(rrt::wait_until_unmapped(controller));

    auto drained = std::atomic<bool>{false};
    auto drain   = std::thread{[&drained]() {
        hsa::replay_drain_agent_or_fatal(hsa_agent_t{.handle = agent_handle});
        drained = true;
    }};

    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(drained) << "the drain returned while a handler on a retiring queue was running";

    destroy.finish();
    drain.join();
    EXPECT_TRUE(drained);
}
