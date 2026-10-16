// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/clocks/read.hpp"

#include <chrono>
#include <cstdint>
#include <time.h>

namespace rocprofsys::clocks::timeline
{
/// Should match rocprofiler-sdk host events stamp clock
inline constexpr clockid_t k_clock_id = CLOCK_BOOTTIME;

/// Absolute timeline clock for CPU and call-stack event stamps.
[[nodiscard]] inline std::chrono::nanoseconds
now() noexcept
{
    return detail::read(k_clock_id);
}

template <typename Tp = std::uint64_t>
[[nodiscard]] inline Tp
now_ns() noexcept
{
    return static_cast<Tp>(now().count());
}
}  // namespace rocprofsys::clocks::timeline
