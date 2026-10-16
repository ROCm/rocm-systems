// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <time.h>

namespace rocprofsys::clocks::timeline
{
/// Should match rocprofiler-sdk host events stamp clock
inline constexpr clockid_t k_clock_id = CLOCK_BOOTTIME;

/// Absolute timeline clock for CPU and call-stack event stamps.
[[nodiscard]] inline std::chrono::nanoseconds
now() noexcept
{
    struct timespec specs = {};
    if(clock_gettime(k_clock_id, &specs) != 0)
    {
        return std::chrono::nanoseconds::zero();
    }
    return std::chrono::seconds{ specs.tv_sec } +
           std::chrono::nanoseconds{ specs.tv_nsec };
}

template <typename Tp = std::uint64_t>
[[nodiscard]] inline Tp
now_ns() noexcept
{
    return static_cast<Tp>(now().count());
}
}  // namespace rocprofsys::clocks::timeline
