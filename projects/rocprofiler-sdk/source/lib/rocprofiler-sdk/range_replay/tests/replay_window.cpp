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
#include <vector>

namespace hsa = ::rocprofiler::hsa;
namespace rrt = ::rocprofiler::range_replay::test;

using namespace std::chrono_literals;

namespace
{
constexpr uint64_t agent_handle = 0x2001;
}  // namespace

// destroy_queue erases a queue from the map before it waits for the queue's completion handlers,
// so the retiring set is the only place a drain can still see them.
TEST(replay_window, a_queue_being_destroyed_stays_visible_until_its_sync_ends)
{
    auto agent      = rrt::test_agent_t{agent_handle};
    auto controller = hsa::QueueController{};
    auto hsa_queue  = hsa_queue_t{};
    hsa_queue.id    = 1;
    auto& queue     = rrt::add_test_queue(controller, hsa_queue, agent);

    auto destroy = rrt::in_flight_destroy_t{controller, hsa_queue, queue};
    ASSERT_TRUE(rrt::wait_until_unmapped(controller));

    auto retiring  = std::vector<const hsa::Queue*>{};
    auto in_flight = int64_t{0};
    controller.iterate_retiring_queues([&](const hsa::Queue* retired) {
        retiring.emplace_back(retired);
        in_flight += retired->active_async_packets();
    });
    EXPECT_EQ(retiring, (std::vector<const hsa::Queue*>{&queue}));
    EXPECT_EQ(in_flight, 1);

    destroy.finish();

    auto left = 0;
    controller.iterate_retiring_queues([&left](const hsa::Queue*) { ++left; });
    EXPECT_EQ(left, 0) << "a destroyed queue must leave the retiring set before it is freed";
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
