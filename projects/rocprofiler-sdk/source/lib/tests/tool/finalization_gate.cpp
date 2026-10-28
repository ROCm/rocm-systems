// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Regression tests for rocprofv3 losing its output when a process exits while another thread is
// still finalizing. Multi-process servers hit this on Ctrl+C: the signal worker starts finalizing
// (e.g. a long ATT decode), the server then asks the process to exit, and the exit hook returned
// at once because finalization had already *started*, so exit() killed the decode midway.

#include "finalization_gate.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace
{
using rocprofiler::tool::finalization_gate;
using status = finalization_gate::status;
using namespace std::chrono_literals;

// Long enough that a caller that did not block would have returned already.
constexpr auto still_blocked = 200ms;
// Generous bound for blocked callers to return once the finalization finishes.
constexpr auto wake_bound = 10s;

// Plays the signal worker: starts the finalization on its own thread and keeps it running, like a
// long ATT decode, until release().
class slow_finalization
{
public:
    explicit slow_finalization(finalization_gate& gate)
    : m_thread{[this, &gate]() {
        gate.run_or_wait([this]() {
            m_started.set_value();
            m_release_future.wait();
            m_finished = true;
        });
    }}
    {
        m_started.get_future().wait();
    }

    ~slow_finalization()
    {
        release();
        m_thread.join();
    }

    slow_finalization(const slow_finalization&) = delete;
    slow_finalization& operator=(const slow_finalization&) = delete;

    void release()
    {
        if(!m_released.exchange(true)) m_release.set_value();
    }

    bool finished() const { return m_finished; }

private:
    std::promise<void>       m_started        = {};
    std::promise<void>       m_release        = {};
    std::shared_future<void> m_release_future = m_release.get_future().share();
    std::atomic<bool>        m_released       = {false};
    std::atomic<bool>        m_finished       = {false};
    std::thread              m_thread         = {};
};
}  // namespace

TEST(finalization_gate, runs_the_finalization_once)
{
    finalization_gate gate{};
    int               runs = 0;
    EXPECT_EQ(gate.run_or_wait([&runs]() { ++runs; }), status::ran);
    EXPECT_EQ(gate.run_or_wait([&runs]() { ++runs; }), status::completed);
    EXPECT_EQ(runs, 1);
}

TEST(finalization_gate, wait_only_blocks_before_finalization_starts)
{
    finalization_gate gate{};
    auto              waiter_started = std::promise<void>{};
    // Models SDK shutdown running earlier teardown steps before it reaches tool_fini.
    auto waiter = std::async(std::launch::async, [&]() {
        waiter_started.set_value();
        return gate.wait_until_complete_or_reentered();
    });
    waiter_started.get_future().wait();
    EXPECT_EQ(waiter.wait_for(still_blocked), std::future_status::timeout);

    slow_finalization tool_fini{gate};
    EXPECT_EQ(waiter.wait_for(still_blocked), std::future_status::timeout);

    tool_fini.release();
    ASSERT_EQ(waiter.wait_for(wake_bound), std::future_status::ready);
    EXPECT_EQ(waiter.get(), status::completed);
}

TEST(finalization_gate, wait_only_blocks_while_tool_finalization_runs)
{
    finalization_gate gate{};
    slow_finalization tool_fini{gate};

    // Models a signal request arriving while SDK-driven tool_fini is writing output.
    auto exit_path = std::async(std::launch::async, [&]() {
        auto result = gate.wait_until_complete_or_reentered();
        return std::make_pair(result, tool_fini.finished());
    });
    EXPECT_EQ(exit_path.wait_for(still_blocked), std::future_status::timeout)
        << "returned while another thread was still finalizing";

    tool_fini.release();
    ASSERT_EQ(exit_path.wait_for(wake_bound), std::future_status::ready);
    auto [result, finished_before_return] = exit_path.get();
    EXPECT_EQ(result, status::completed);
    EXPECT_TRUE(finished_before_return);
}

TEST(finalization_gate, wakes_every_waiter)
{
    finalization_gate gate{};
    slow_finalization signal_worker{gate};

    auto waiters = std::vector<std::future<status>>{};
    for(int i = 0; i < 4; ++i)
        waiters.emplace_back(
            std::async(std::launch::async, [&gate]() { return gate.run_or_wait([]() {}); }));
    std::this_thread::sleep_for(still_blocked);

    signal_worker.release();
    for(auto& waiter : waiters)
    {
        ASSERT_EQ(waiter.wait_for(wake_bound), std::future_status::ready)
            << "a waiter was not woken when the finalization finished";
        EXPECT_EQ(waiter.get(), status::completed);
    }
}

TEST(finalization_gate, finalizing_thread_does_not_wait_on_itself)
{
    // e.g. the application calls exit() while this thread is finalizing
    finalization_gate gate{};
    auto              inner = status::ran;
    EXPECT_EQ(gate.run_or_wait([&]() { inner = gate.wait_until_complete_or_reentered(); }),
              status::ran);
    EXPECT_EQ(inner, status::reentered);
}

TEST(finalization_gate, serializes_joiners_after_worker_skips_self_join)
{
    finalization_gate join_gate{};
    auto              worker_id         = std::atomic<std::thread::id>{};
    auto              join_ready        = std::atomic<bool>{false};
    auto              cleanup_count     = std::atomic<int>{0};
    auto              self_result       = std::promise<status>{};
    auto              release_worker    = std::promise<void>{};
    auto              join_started      = std::promise<void>{};
    auto              allow_join        = std::promise<void>{};
    auto              release_future    = release_worker.get_future().share();
    auto              allow_join_future = allow_join.get_future().share();
    auto              worker            = std::thread{};
    auto              join_worker       = std::function<status()>{};

    worker = std::thread{[&]() {
        worker_id.store(std::this_thread::get_id(), std::memory_order_release);
        while(!join_ready.load(std::memory_order_acquire))
            std::this_thread::yield();
        self_result.set_value(join_worker());
        release_future.wait();
    }};

    join_worker = [&]() {
        if(worker_id.load(std::memory_order_acquire) == std::this_thread::get_id())
            return status::reentered;
        return join_gate.run_or_wait([&]() {
            ++cleanup_count;
            join_started.set_value();
            allow_join_future.wait();
            release_worker.set_value();
            worker.join();
        });
    };
    join_ready.store(true, std::memory_order_release);

    EXPECT_EQ(self_result.get_future().get(), status::reentered);
    auto first_joiner = std::async(std::launch::async, join_worker);
    join_started.get_future().wait();
    auto second_joiner = std::async(std::launch::async, join_worker);
    EXPECT_EQ(second_joiner.wait_for(still_blocked), std::future_status::timeout);

    allow_join.set_value();
    ASSERT_EQ(first_joiner.wait_for(wake_bound), std::future_status::ready);
    ASSERT_EQ(second_joiner.wait_for(wake_bound), std::future_status::ready);
    EXPECT_EQ(first_joiner.get(), status::ran);
    EXPECT_EQ(second_joiner.get(), status::completed);
    EXPECT_EQ(cleanup_count, 1);
    EXPECT_FALSE(worker.joinable());
}

TEST(finalization_gate, waiters_are_released_when_the_finalization_throws)
{
    finalization_gate gate{};
    EXPECT_THROW(gate.run_or_wait([]() { throw std::runtime_error{"finalization failed"}; }),
                 std::runtime_error);
    EXPECT_EQ(gate.run_or_wait([]() {}), status::completed);
}
