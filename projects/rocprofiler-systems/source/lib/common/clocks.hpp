// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <ctime>
#include <time.h>

namespace rocprofsys::inline common::clocks::detail
{
/// Reads a POSIX clock; zero if the clock fails. Async-signal-safe
[[nodiscard]] inline std::chrono::nanoseconds
read(clockid_t clock_id) noexcept
{
    timespec specs = {};
    if(clock_gettime(clock_id, &specs) != 0)
    {
        return std::chrono::nanoseconds::zero();
    }
    return std::chrono::seconds{ specs.tv_sec } +
           std::chrono::nanoseconds{ specs.tv_nsec };
}
}  // namespace rocprofsys::inline common::clocks::detail

namespace rocprofsys::inline common::clocks::timeline
{
/// Must match the rocprofiler-sdk host clock so that CPU and SDK events line up in the
/// trace; the SDK backend tests static_assert this.
inline constexpr clockid_t k_timeline_clock_id = CLOCK_BOOTTIME;

/// Absolute timeline clock for CPU and call-stack event stamps.
[[nodiscard]] inline std::chrono::nanoseconds
now() noexcept
{
    return detail::read(k_timeline_clock_id);
}

template <typename Tp = std::uint64_t>
[[nodiscard]] inline Tp
now_ns() noexcept
{
    return static_cast<Tp>(now().count());
}
}  // namespace rocprofsys::inline common::clocks::timeline

namespace rocprofsys::inline common::clocks::thread_cpu
{
/// CPU time used so far by the calling thread (`CLOCK_THREAD_CPUTIME_ID`); zero if
/// the clock fails. Only the difference between two reads on one thread is meaningful.
[[nodiscard]] inline std::chrono::nanoseconds
now() noexcept
{
    return detail::read(CLOCK_THREAD_CPUTIME_ID);
}
}  // namespace rocprofsys::inline common::clocks::thread_cpu
