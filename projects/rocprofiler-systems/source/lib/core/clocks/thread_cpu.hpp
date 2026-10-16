// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <ctime>
#include <time.h>

namespace rocprofsys::clocks::thread_cpu
{
/// CPU time used so far by the calling thread (`CLOCK_THREAD_CPUTIME_ID`); zero if
/// the clock fails. Only the difference between two reads on one thread is meaningful.
/// Async-signal-safe: the sampler's signal handler calls it.
[[nodiscard]] inline std::chrono::nanoseconds
now() noexcept
{
    struct timespec cpu_time = {};
    if(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_time) != 0)
    {
        return std::chrono::nanoseconds::zero();
    }
    return std::chrono::seconds{ cpu_time.tv_sec } +
           std::chrono::nanoseconds{ cpu_time.tv_nsec };
}
}  // namespace rocprofsys::clocks::thread_cpu
