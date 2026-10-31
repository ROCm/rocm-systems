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
// Waiters use a finite timeout so a regression fails the test instead of hanging it.
constexpr int wait_timeout_sec = 30;

// Plays the signal worker: starts the finalization on its own thread and keeps it running, like a
// long ATT decode, until release().
class slow_finalization
{
public:
    explicit slow_finalization(finalization_gate& gate)
    : m_thread{[this, &gate]() {
        gate.run_or_wait(
            [this]() {
                m_started.set_value();
                m_release_future.wait();
                m_finished = true;
            },
            0);
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
    EXPECT_EQ(gate.run_or_wait([&runs]() { ++runs; }, 0), status::ran);
    EXPECT_EQ(gate.run_or_wait([&runs]() { ++runs; }, 0), status::completed);
    EXPECT_EQ(runs, 1);
}

TEST(finalization_gate, exit_waits_for_the_finalization_a_signal_started)
{
    finalization_gate gate{};
    slow_finalization signal_worker{gate};

    // rocprofv3_main after main() returns
    auto exit_path = std::async(std::launch::async, [&]() {
        auto result = gate.run_or_wait([]() {}, wait_timeout_sec);
        return std::make_pair(result, signal_worker.finished());
    });
    EXPECT_EQ(exit_path.wait_for(still_blocked), std::future_status::timeout)
        << "returned while another thread was still finalizing";

    signal_worker.release();
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
        waiters.emplace_back(std::async(
            std::launch::async, [&gate]() { return gate.run_or_wait([]() {}, wait_timeout_sec); }));
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
    EXPECT_EQ(gate.run_or_wait([&]() { inner = gate.run_or_wait([]() {}, 5); }, 0), status::ran);
    EXPECT_EQ(inner, status::reentered);
}

TEST(finalization_gate, wait_gives_up_after_the_timeout)
{
    finalization_gate gate{};
    slow_finalization signal_worker{gate};

    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(gate.run_or_wait([]() {}, 1), status::timed_out);
    EXPECT_GE(std::chrono::steady_clock::now() - start, 950ms);
}

TEST(finalization_gate, waiters_are_released_when_the_finalization_throws)
{
    finalization_gate gate{};
    EXPECT_THROW(gate.run_or_wait([]() { throw std::runtime_error{"finalization failed"}; }, 0),
                 std::runtime_error);
    EXPECT_EQ(gate.run_or_wait([]() {}, 5), status::completed);
}
