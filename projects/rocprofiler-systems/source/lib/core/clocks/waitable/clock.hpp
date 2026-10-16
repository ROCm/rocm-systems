// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <concepts>

namespace rocprofsys::clocks::waitable
{
using duration   = std::chrono::nanoseconds;
using time_point = std::chrono::time_point<std::chrono::steady_clock, duration>;

// Kept with the clocks rather than in policies/: only core uses it, and it speaks in
// the clock types above, which a policy header could not take from core.
template <typename Tp>
concept clock = requires(Tp clk, time_point deadline) {
    { clk.now() } noexcept -> std::convertible_to<time_point>;
    { clk.sleep_until(deadline) } -> std::same_as<bool>;
    { clk.interrupt() } noexcept;
    { clk.reset() } noexcept;
};
}  // namespace rocprofsys::clocks::waitable
