// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/clocks/waitable/posix.hpp"
#include "core/control/session.hpp"

#include "core/clocks/waitable/clock.hpp"
#include "core/control/triggers/time_window.hpp"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

namespace
{
using rocprofsys::clocks::waitable::duration;
using rocprofsys::clocks::waitable::posix;
using rocprofsys::control::session;
using time_window_t = rocprofsys::control::triggers::time_window<posix>;

// Polls the session's active state until it reaches @p expected or a
// wall-clock timeout elapses. posix uses real time, so events cannot be
// deterministically stepped - poll instead.
bool
wait_until_active(session& sess, bool expected)
{
    constexpr auto k_timeout = std::chrono::seconds{ 2 };
    const auto     deadline  = std::chrono::steady_clock::now() + k_timeout;
    while(sess.is_active() != expected)
    {
        if(std::chrono::steady_clock::now() > deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{ 1 });
    }
    return true;
}
}  // namespace

TEST(waitable_posix_test, now_returns_advancing_time_points)
{
    using namespace std::chrono_literals;
    const posix clk{ CLOCK_REALTIME };  // NOLINT

    const auto earlier = clk.now();
    std::this_thread::sleep_for(5ms);  // NOLINT(misc-include-cleaner)
    const auto later = clk.now();

    EXPECT_GT(later, earlier) << "now() should advance over time";
}

TEST(waitable_posix_test, interrupt_makes_sleep_until_return_false)
{
    using namespace std::chrono_literals;
    posix clk{ CLOCK_REALTIME };

    // Schedule a deadline far in the future; interrupt from another thread.
    const auto deadline = clk.now() + duration{ 30'000'000'000LL };  // 30s

    std::atomic<bool> result{ true };
    std::thread       sleeper{ [&] { result.store(clk.sleep_until(deadline)); } };

    std::this_thread::sleep_for(10ms);  // NOLINT(misc-include-cleaner)
    clk.interrupt();
    sleeper.join();

    EXPECT_FALSE(result.load()) << "sleep_until should return false when interrupted";
}

TEST(waitable_posix_test, reset_allows_reuse_after_interrupt)
{
    posix clk{ CLOCK_REALTIME };
    clk.interrupt();
    clk.reset();

    // After reset, a deadline already in the past should return true (no interrupt).
    const auto past = clk.now() - duration{ 1 };
    EXPECT_TRUE(clk.sleep_until(past));
}

TEST(waitable_posix_test, time_window_with_posix_clock_drives_session)
{
    const auto s_ptr = std::make_shared<session>();
    auto&      sess  = *s_ptr;

    // 20ms/40ms windows polled against a 2s timeout (100x/50x margin) -
    // deliberately generous so the test doesn't flake under CI scheduler
    // jitter. Don't shrink the timeout to "tighten" this; shrink the
    // windows if faster tests are needed instead.
    posix          clk{ CLOCK_REALTIME };
    constexpr auto k_delay = duration{ 20'000'000 };  // 20 ms
    constexpr auto k_dur   = duration{ 40'000'000 };  // 40 ms
    time_window_t  window{ s_ptr, clk, { .delay = k_delay, .duration = k_dur } };

    EXPECT_FALSE(sess.is_active()) << "initial action should be paused when delay > 0";

    window.start();
    ASSERT_TRUE(wait_until_active(sess, true))
        << "session should become active once the delay elapses";
    ASSERT_TRUE(wait_until_active(sess, false))
        << "session should pause again once the duration elapses";
}
