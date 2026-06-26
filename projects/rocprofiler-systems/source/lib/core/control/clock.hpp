// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <concepts>
#include <cstdint>

namespace rocprofsys::control
{
using clock_duration = std::chrono::nanoseconds;
using clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock, clock_duration>;

/// Satisfied by any type that can drive a time_window trigger.
/// sleep_until returns true when the deadline was reached, false when
/// interrupt() woke it early. interrupt() and reset() are idempotent
/// and thread-safe.
template <typename C>
concept ClockPolicy = requires(C c, clock_time_point tp) {
    { c.now() } noexcept -> std::convertible_to<clock_time_point>;
    { c.sleep_until(tp) } -> std::same_as<bool>;
    { c.interrupt() } noexcept;
    { c.reset() } noexcept;
};
}  // namespace rocprofsys::control
