// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <time.h>

namespace rocprofsys::control::clocks
{
/// Absolute timeline clock for CPU and call-stack event stamps.
/// Matches the host domain used by rocprofiler-sdk (`CLOCK_BOOTTIME`).
[[nodiscard]] inline bool
timeline_available() noexcept
{
    struct timespec ts = {};
    return clock_gettime(CLOCK_BOOTTIME, &ts) == 0;
}

[[nodiscard]] inline std::chrono::nanoseconds
timeline_now() noexcept
{
    struct timespec ts = {};
    if(clock_gettime(CLOCK_BOOTTIME, &ts) != 0)
    {
        return std::chrono::nanoseconds::zero();
    }
    return std::chrono::seconds{ ts.tv_sec } + std::chrono::nanoseconds{ ts.tv_nsec };
}

template <typename Tp = std::uint64_t>
[[nodiscard]] inline Tp
timeline_ns() noexcept
{
    return static_cast<Tp>(timeline_now().count());
}
}  // namespace rocprofsys::control::clocks
