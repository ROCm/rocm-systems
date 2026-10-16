// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/clocks/thread_cputime.hpp"

#include <chrono>
#include <gtest/gtest.h>
#include <thread>

namespace
{
using rocprofsys::clocks::thread_cputime;

constexpr auto k_spin_time = std::chrono::milliseconds{ 20 };

// Keeps the calling thread on the CPU for @p wall_time of wall-clock time.
void
spin_for(std::chrono::nanoseconds wall_time)
{
    const auto deadline = std::chrono::steady_clock::now() + wall_time;
    while(std::chrono::steady_clock::now() < deadline)
    {
    }
}
}  // namespace

TEST(thread_cputime_test, never_decreases)
{
    const auto earlier = thread_cputime();
    const auto later   = thread_cputime();
    EXPECT_LE(earlier, later);
}

TEST(thread_cputime_test, advances_while_the_thread_runs)
{
    const auto before = thread_cputime();
    spin_for(k_spin_time);
    EXPECT_GT(thread_cputime(), before);
}

// The sampler subtracts two reads of one thread, so another thread's work must not
// show up in the reading.
TEST(thread_cputime_test, counts_only_the_calling_thread)
{
    spin_for(k_spin_time);
    const auto spinning_thread_time = thread_cputime();

    auto        new_thread_time = std::chrono::nanoseconds::max();
    std::thread new_thread{ [&new_thread_time] { new_thread_time = thread_cputime(); } };
    new_thread.join();

    EXPECT_LT(new_thread_time, spinning_thread_time);
}
