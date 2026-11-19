// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/clocks.hpp"

#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <ratio>
#include <thread>
#include <time.h>

namespace
{
namespace detail     = rocprofsys::common::clocks::detail;
namespace thread_cpu = rocprofsys::common::clocks::thread_cpu;
namespace timeline   = rocprofsys::common::clocks::timeline;

// The kernel rejects positive clock IDs past its last static clock with EINVAL.
constexpr clockid_t k_invalid_clock_id = 1'000;

constexpr auto k_spin_time = std::chrono::milliseconds{ 20 };

std::uint64_t
read_boottime_ns()
{
    timespec specs{};
    EXPECT_EQ(clock_gettime(CLOCK_BOOTTIME, &specs), 0);
    return static_cast<std::uint64_t>(specs.tv_sec) * std::nano::den +
           static_cast<std::uint64_t>(specs.tv_nsec);
}

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

// The role clocks cannot reach this path: their clocks always exist.
TEST(clocks_read_test, returns_zero_when_the_clock_fails)
{
    EXPECT_EQ(detail::read(k_invalid_clock_id), std::chrono::nanoseconds::zero());
}

TEST(timeline_clock_test, reads_the_timeline_clock)
{
    const auto before_duration = read_boottime_ns();
    const auto now_duration    = timeline::now();
    const auto after_duration  = read_boottime_ns();
    EXPECT_LE(before_duration, static_cast<std::uint64_t>(now_duration.count()));
    EXPECT_LE(static_cast<std::uint64_t>(now_duration.count()), after_duration);

    const auto before_ns = read_boottime_ns();
    const auto now_ns    = timeline::now_ns();
    const auto after_ns  = read_boottime_ns();
    EXPECT_LE(before_ns, now_ns);
    EXPECT_LE(now_ns, after_ns);
}

TEST(thread_cpu_test, never_decreases)
{
    const auto earlier = thread_cpu::now();
    const auto later   = thread_cpu::now();
    EXPECT_LE(earlier, later);
}

TEST(thread_cpu_test, advances_while_the_thread_runs)
{
    const auto before = thread_cpu::now();
    spin_for(k_spin_time);
    EXPECT_GT(thread_cpu::now(), before);
}

// The sampler subtracts two reads of one thread, so another thread's work must not
// show up in the reading.
TEST(thread_cpu_test, counts_only_the_calling_thread)
{
    spin_for(k_spin_time);
    const auto spinning_thread_time = thread_cpu::now();

    auto        new_thread_time = std::chrono::nanoseconds::max();
    std::thread new_thread{ [&new_thread_time] { new_thread_time = thread_cpu::now(); } };
    new_thread.join();

    EXPECT_LT(new_thread_time, spinning_thread_time);
}
