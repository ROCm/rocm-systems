// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <concepts>

namespace rocprofsys::control
{
using clock_duration = std::chrono::nanoseconds;
using clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock, clock_duration>;

template <typename Tp>
concept clock_policy = requires(Tp clk, clock_time_point deadline) {
    { clk.now() } noexcept -> std::convertible_to<clock_time_point>;
    { clk.sleep_until(deadline) } -> std::same_as<bool>;
    { clk.interrupt() } noexcept;
    { clk.reset() } noexcept;
};
}  // namespace rocprofsys::control
