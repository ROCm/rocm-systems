// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <concepts>

namespace rocprofsys::clocks::waitable
{
using duration   = std::chrono::nanoseconds;
using time_point = std::chrono::time_point<std::chrono::steady_clock, duration>;

template <typename Tp>
concept clock = requires(Tp clk, time_point deadline) {
    { clk.now() } noexcept -> std::convertible_to<time_point>;
    { clk.sleep_until(deadline) } -> std::same_as<bool>;
    { clk.interrupt() } noexcept;
    { clk.reset() } noexcept;
};
}  // namespace rocprofsys::clocks::waitable
