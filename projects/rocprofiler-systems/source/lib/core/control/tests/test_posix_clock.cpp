// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/control/clocks/posix.hpp"
#include "core/control/session.hpp"

#include "core/control/triggers/time_window.hpp"
#include <memory>

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
using rocprofsys::control::clock_duration;
using rocprofsys::control::session;
using rocprofsys::control::clocks::posix;
using time_window_t = rocprofsys::control::triggers::time_window<posix>;

// Polls the session's active state until it reaches @p expected or a
// wall-clock timeout elapses. posix uses real time, so events cannot be
// deterministically stepped - poll instead.
bool
wait_until_active(session& s, bool expected)
{
    constexpr auto timeout  = std::chrono::seconds{ 2 };
    const auto     deadline = std::chrono::steady_clock::now() + timeout;
    while(s.is_active() != expected)
    {
        if(std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds{ 1 });
    }
    return true;
}
}  // namespace

TEST(posix_clock_test, now_returns_advancing_time_points)
{
    using namespace std::chrono_literals;
    posix clk{ CLOCK_REALTIME };

    const auto t0 = clk.now();
    std::this_thread::sleep_for(5ms);
    const auto t1 = clk.now();

    EXPECT_GT(t1, t0) << "now() should advance over time";
}

TEST(posix_clock_test, interrupt_makes_sleep_until_return_false)
{
    using namespace std::chrono_literals;
    posix clk{ CLOCK_REALTIME };

    // Schedule a deadline far in the future; interrupt from another thread.
    const auto deadline = clk.now() + clock_duration{ 30'000'000'000LL };  // 30s

    std::atomic<bool> result{ true };
    std::thread       sleeper{ [&] { result.store(clk.sleep_until(deadline)); } };

    std::this_thread::sleep_for(10ms);
    clk.interrupt();
    sleeper.join();

    EXPECT_FALSE(result.load()) << "sleep_until should return false when interrupted";
}

TEST(posix_clock_test, reset_allows_reuse_after_interrupt)
{
    posix clk{ CLOCK_REALTIME };
    clk.interrupt();
    clk.reset();

    // After reset, a deadline already in the past should return true (no interrupt).
    const auto past = clk.now() - clock_duration{ 1 };
    EXPECT_TRUE(clk.sleep_until(past));
}

TEST(posix_clock_test, time_window_with_posix_clock_drives_session)
{
    auto  s_ptr = std::make_shared<session>();
    auto& s     = *s_ptr;

    // 20ms/40ms windows polled against a 2s timeout (100x/50x margin) -
    // deliberately generous so the test doesn't flake under CI scheduler
    // jitter. Don't shrink the timeout to "tighten" this; shrink the
    // windows if faster tests are needed instead.
    posix          clk{ CLOCK_REALTIME };
    constexpr auto delay = clock_duration{ 20'000'000 };  // 20 ms
    constexpr auto dur   = clock_duration{ 40'000'000 };  // 40 ms
    time_window_t  tw{ s_ptr, clk, delay, dur };

    EXPECT_FALSE(s.is_active()) << "initial action should be paused when delay > 0";

    tw.start();
    ASSERT_TRUE(wait_until_active(s, true))
        << "session should become active once the delay elapses";
    ASSERT_TRUE(wait_until_active(s, false))
        << "session should pause again once the duration elapses";
}
