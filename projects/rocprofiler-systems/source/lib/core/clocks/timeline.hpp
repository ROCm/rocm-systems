// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <time.h>

namespace rocprofsys::clocks
{
/// Should match rocprofiler-sdk host events stamp clock
inline constexpr clockid_t k_timeline_clock_id = CLOCK_BOOTTIME;

/// Absolute timeline clock for CPU and call-stack event stamps.
[[nodiscard]] inline std::chrono::nanoseconds
timeline_now() noexcept
{
    struct timespec specs = {};
    if(clock_gettime(k_timeline_clock_id, &specs) != 0)
    {
        return std::chrono::nanoseconds::zero();
    }
    return std::chrono::seconds{ specs.tv_sec } +
           std::chrono::nanoseconds{ specs.tv_nsec };
}

template <typename Tp = std::uint64_t>
[[nodiscard]] inline Tp
timeline_ns() noexcept
{
    return static_cast<Tp>(timeline_now().count());
}
}  // namespace rocprofsys::clocks
