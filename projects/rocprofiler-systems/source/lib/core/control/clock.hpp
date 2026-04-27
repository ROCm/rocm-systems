// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>

namespace rocprofsys::control
{
/// Shared type aliases for all clock implementations. The clock concept
/// itself is duck-typed (no virtual base): triggers that need a clock are
/// templated on it. A type C satisfies the clock concept when it provides:
///
///   clock_time_point now() const noexcept;
///   bool             sleep_until(clock_time_point deadline);
///   void             interrupt();
///
/// Where sleep_until returns true if the deadline was reached and false if
/// interrupt() woke it early. interrupt() is idempotent and thread-safe.
///
/// See clocks::steady (production) and clocks::manual (test-only) for
/// concrete impls.
using clock_duration = std::chrono::nanoseconds;
using clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock, clock_duration>;
}  // namespace rocprofsys::control
