// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/clocks.hpp"
#include "core/collection/scheduler.hpp"
#include "core/collection/tests/wait_helpers.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <future>
#include <latch>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace rocprofsys::collection
{
namespace
{
using namespace std::chrono_literals;
using testing::call_counter;
using testing::failing_thread_creation;
using testing::k_timeout;
using testing::latch_releaser;
using testing::let_other_threads_run;

constexpr auto k_fast_interval = std::chrono::nanoseconds{ 1ms };
constexpr auto k_slow_interval = std::chrono::nanoseconds{ 1h };
constexpr int  k_wanted_ticks  = 5;

/** A scheduler callback that counts its calls and keeps going. */
[[nodiscard]] scheduler::callback
counting(call_counter& counter)
{
    return [&counter](std::uint64_t) {
        counter();
        return true;
    };
}

[[nodiscard]] bool
is_ready(const std::future<void>& future)
{
    return future.wait_for(k_timeout) == std::future_status::ready;
}

void
throw_non_std_exception()
{
    std::rethrow_exception(std::make_exception_ptr(k_wanted_ticks));
}
}  // namespace

TEST(scheduler, first_tick_is_immediate_and_ticks_repeat)
{
    call_counter ticks{ k_wanted_ticks };
    scheduler    sched{ "test" };

    sched.start(k_fast_interval, counting(ticks));

    EXPECT_TRUE(ticks.wait());
}

TEST(scheduler, first_tick_is_immediate_with_a_slow_interval)
{
    call_counter ticks{ 1 };
    scheduler    sched{ "test" };

    sched.start(k_slow_interval, counting(ticks));

    EXPECT_TRUE(ticks.wait());
}

TEST(scheduler, callback_gets_the_timeline_time)
{
    std::promise<std::uint64_t> seen;
    auto                        seen_future = seen.get_future();
    std::atomic<bool>           first{ true };
    scheduler                   sched{ "test" };
    const auto                  before = clocks::timeline::now_ns();

    sched.start(k_slow_interval, [&](std::uint64_t now) {
        if(first.exchange(false))
        {
            seen.set_value(now);
        }
        return true;
    });

    ASSERT_EQ(seen_future.wait_for(k_timeout), std::future_status::ready);
    const auto after = clocks::timeline::now_ns();
    const auto now   = seen_future.get();
    EXPECT_GE(now, before);
    EXPECT_LE(now, after);
}

TEST(scheduler, thread_start_runs_on_the_thread_before_the_first_tick)
{
    std::atomic<bool>  started{ false };
    std::thread::id    start_thread;
    std::promise<bool> first_tick_after_start;
    auto               first_tick_future = first_tick_after_start.get_future();
    std::atomic<bool>  first{ true };
    scheduler          sched{ "test" };

    sched.start(
        k_slow_interval,
        [&](std::uint64_t) {
            if(first.exchange(false))
            {
                first_tick_after_start.set_value(
                    started.load() && start_thread == std::this_thread::get_id());
            }
            return true;
        },
        [&] {
            start_thread = std::this_thread::get_id();
            started      = true;
        });

    ASSERT_EQ(first_tick_future.wait_for(k_timeout), std::future_status::ready);
    EXPECT_TRUE(first_tick_future.get());
}

TEST(scheduler, throwing_thread_start_ends_the_thread_without_a_tick)
{
    std::promise<void> exited;
    const auto         exited_future = exited.get_future();
    std::atomic<int>   ticks{ 0 };
    scheduler          sched{ "test" };

    sched.start(
        k_fast_interval,
        [&](std::uint64_t) {
            ++ticks;
            return true;
        },
        [&] {
            exited.set_value_at_thread_exit();
            throw std::runtime_error("thread start failed");
        });

    EXPECT_TRUE(is_ready(exited_future));
    EXPECT_EQ(ticks.load(), 0);
}

TEST(scheduler, start_that_cannot_create_its_thread_throws_and_can_be_retried)
{
    call_counter ticks{ 1 };
    scheduler    sched{ "test" };
    {
        const failing_thread_creation no_threads;
        EXPECT_THROW(sched.start(k_slow_interval, counting(ticks)), std::system_error);
    }

    sched.start(k_slow_interval, counting(ticks));

    EXPECT_TRUE(ticks.wait());
}

TEST(scheduler, stop_returns_from_a_long_wait)
{
    call_counter ticks{ 1 };
    scheduler    sched{ "test" };
    sched.start(k_slow_interval, counting(ticks));
    ASSERT_TRUE(ticks.wait());

    sched.stop();

    EXPECT_EQ(ticks.count(), 1);
}

TEST(scheduler, stop_without_start_is_a_no_op)
{
    scheduler sched{ "test" };

    sched.stop();
    sched.pause();
    sched.resume();
}

TEST(scheduler, callback_returning_false_ends_the_thread)
{
    std::promise<void> exited;
    const auto         exited_future = exited.get_future();
    std::atomic<int>   calls{ 0 };
    scheduler          sched{ "test" };

    sched.start(k_fast_interval, [&](std::uint64_t) {
        exited.set_value_at_thread_exit();
        ++calls;
        return false;
    });

    EXPECT_TRUE(is_ready(exited_future));
    EXPECT_EQ(calls.load(), 1);
}

TEST(scheduler, std_exception_in_the_callback_ends_the_thread)
{
    std::promise<void> exited;
    const auto         exited_future = exited.get_future();
    std::atomic<int>   calls{ 0 };
    scheduler          sched{ "test" };

    sched.start(k_fast_interval, [&](std::uint64_t) -> bool {
        exited.set_value_at_thread_exit();
        ++calls;
        throw std::runtime_error("poll failed");
    });

    EXPECT_TRUE(is_ready(exited_future));
    EXPECT_EQ(calls.load(), 1);
}

TEST(scheduler, non_std_exception_in_the_callback_ends_the_thread)
{
    std::promise<void> exited;
    const auto         exited_future = exited.get_future();
    std::atomic<int>   calls{ 0 };
    scheduler          sched{ "test" };

    sched.start(k_fast_interval, [&](std::uint64_t) -> bool {
        exited.set_value_at_thread_exit();
        ++calls;
        throw_non_std_exception();
        return true;
    });

    EXPECT_TRUE(is_ready(exited_future));
    EXPECT_EQ(calls.load(), 1);
}

TEST(scheduler, stop_from_the_callback_detaches_and_the_thread_ends)
{
    std::promise<void> exited;
    const auto         exited_future = exited.get_future();
    scheduler          sched{ "test" };

    sched.start(k_fast_interval, [&](std::uint64_t) {
        exited.set_value_at_thread_exit();
        sched.stop();
        return true;
    });

    EXPECT_TRUE(is_ready(exited_future));
}

TEST(scheduler, pause_stops_callbacks_and_resume_restarts_them)
{
    call_counter      before{ k_wanted_ticks };
    call_counter      after{ k_wanted_ticks };
    std::atomic<bool> resumed{ false };
    scheduler         sched{ "test" };
    sched.start(k_fast_interval, [&](std::uint64_t) {
        (resumed.load() ? after : before)();
        return true;
    });
    ASSERT_TRUE(before.wait());

    sched.pause();
    // one callback may already be running when pause() returns
    const int at_pause = before.count() + 1;
    let_other_threads_run();
    EXPECT_LE(before.count(), at_pause);

    resumed = true;
    sched.resume();
    EXPECT_TRUE(after.wait());
}

TEST(scheduler, pause_during_a_callback_without_resume_stops_the_ticks)
{
    call_counter ticks{ 1 };
    std::latch   finish_first{ 1 };
    scheduler    sched{ "test" };
    sched.start(k_fast_interval, [&](std::uint64_t) {
        ticks();
        finish_first.wait();
        return true;
    });
    {
        const latch_releaser release_first{ finish_first };
        ASSERT_TRUE(ticks.wait());
        sched.pause();
    }

    let_other_threads_run();

    EXPECT_EQ(ticks.count(), 1);
}

TEST(scheduler, resume_after_pause_fires_without_waiting_for_the_interval)
{
    call_counter      first{ 1 };
    call_counter      second{ 1 };
    std::atomic<bool> resumed{ false };
    scheduler         sched{ "test" };
    sched.start(k_slow_interval, [&](std::uint64_t) {
        (resumed.load() ? second : first)();
        return true;
    });
    ASSERT_TRUE(first.wait());
    sched.pause();
    // the first callback has returned and the thread waits while paused
    let_other_threads_run();

    resumed = true;
    sched.resume();

    EXPECT_TRUE(second.wait());
}

TEST(scheduler, resume_during_a_running_callback_fires_without_waiting)
{
    call_counter      first{ 1 };
    call_counter      second{ 1 };
    std::latch        finish_first{ 1 };
    std::atomic<bool> resumed{ false };
    scheduler         sched{ "test" };
    sched.start(k_slow_interval, [&](std::uint64_t) {
        if(resumed.load())
        {
            second();
            return true;
        }
        first();
        finish_first.wait();
        return true;
    });
    {
        const latch_releaser release_first{ finish_first };
        ASSERT_TRUE(first.wait());
        sched.pause();
        resumed = true;
        sched.resume();
    }

    EXPECT_TRUE(second.wait());
}

TEST(scheduler, second_start_throws)
{
    scheduler sched{ "test" };
    sched.start(k_slow_interval, [](std::uint64_t) { return true; });

    EXPECT_THROW(sched.start(k_slow_interval, [](std::uint64_t) { return true; }),
                 std::logic_error);
}

TEST(scheduler, non_positive_interval_throws)
{
    scheduler sched{ "test" };

    EXPECT_THROW(
        sched.start(std::chrono::nanoseconds::zero(), [](std::uint64_t) { return true; }),
        std::invalid_argument);
}

TEST(scheduler, a_blocked_callback_does_not_hold_back_another_scheduler)
{
    std::latch   unblock{ 1 };
    call_counter blocked_started{ 1 };
    call_counter other{ k_wanted_ticks };
    scheduler    blocked{ "blocked" };
    scheduler    running{ "running" };
    // destroyed before the schedulers, so the blocked thread can be joined
    const latch_releaser release_at_exit{ unblock };

    blocked.start(k_fast_interval, [&](std::uint64_t) {
        blocked_started();
        unblock.wait();
        return true;
    });
    ASSERT_TRUE(blocked_started.wait());
    running.start(k_fast_interval, counting(other));

    EXPECT_TRUE(other.wait());
}
}  // namespace rocprofsys::collection
