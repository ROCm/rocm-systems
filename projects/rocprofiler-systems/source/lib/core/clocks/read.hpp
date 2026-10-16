// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <ctime>
#include <time.h>

namespace rocprofsys::clocks::detail
{
/// Reads a POSIX clock; zero if the clock fails.
/// Async-signal-safe: no logging, no locks, no allocation.
[[nodiscard]] inline std::chrono::nanoseconds
read(clockid_t clock_id) noexcept
{
    struct timespec specs = {};
    if(clock_gettime(clock_id, &specs) != 0)
    {
        return std::chrono::nanoseconds::zero();
    }
    return std::chrono::seconds{ specs.tv_sec } +
           std::chrono::nanoseconds{ specs.tv_nsec };
}
}  // namespace rocprofsys::clocks::detail
