// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/control/clocks/steady.hpp"
#include "core/control/session.hpp"

#include "core/control/triggers/time_window.hpp"
#include <memory>

#include <chrono>
#include <gtest/gtest.h>
#include <thread>

namespace
{
using rocprofsys::control::clock_duration;
using rocprofsys::control::session;
using rocprofsys::control::clocks::steady;
using time_window_t = rocprofsys::control::triggers::time_window<steady>;

bool
wait_until_active(session& sess, bool expected)
{
    constexpr auto timeout  = std::chrono::seconds{ 2 };
    const auto     deadline = std::chrono::steady_clock::now() + timeout;
    while(sess.is_active() != expected)
    {
        if(std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds{ 1 });
    }
    return true;
}
}  // namespace

TEST(steady_clock_test, reset_allows_reuse_after_interrupt)
{
    steady clk{};
    clk.interrupt();
    clk.reset();

    EXPECT_TRUE(clk.sleep_until(clk.now() - clock_duration{ 1 }));
}

// A clock outlives the windows driven by it, so stopping one window must not
// make the next one bail out of its first sleep. The two windows share a clock
// sequentially - the first is joined and destroyed before the second exists -
// which is the arrangement production uses across a stop/restart.
TEST(steady_clock_test, window_started_on_a_stopped_clock_still_runs)
{
    auto   s_ptr = std::make_shared<session>();
    auto&  s     = *s_ptr;
    steady clk{};

    constexpr auto long_delay = clock_duration{ 10'000'000'000LL };  // 10 s
    {
        time_window_t first{ s_ptr, clk, time_window_t::config{ long_delay, {} } };
        first.start();
        first.stop();
    }

    // No duration: the active state is terminal, so the poll cannot miss it.
    constexpr auto delay = clock_duration{ 20'000'000 };  // 20 ms
    time_window_t  second{ s_ptr, clk, time_window_t::config{ delay, {} } };

    ASSERT_FALSE(s.is_active()) << "a pending delay should leave the session paused";

    second.start();
    EXPECT_TRUE(wait_until_active(s, true))
        << "session should become active once the second window's delay elapses";
}
