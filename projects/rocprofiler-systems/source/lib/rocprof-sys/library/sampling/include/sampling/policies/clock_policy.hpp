// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// clock — named requirement (ClockPolicy concept)
// ============================================================================
// Required member functions:
//   uint64_t now_ns() noexcept;
//       Return current CLOCK_REALTIME time in nanoseconds since epoch.
//       Must be async-signal-safe (clock_gettime is signal-safe on Linux).
//   std::chrono::steady_clock::time_point now_steady() noexcept;
//       Return the current steady-clock time point for deadline computations.
//
// Production: rocprofsys::sampling::steady_clock
//             - now_ns: clock_gettime(CLOCK_REALTIME, ...)
//             - now_steady: std::chrono::steady_clock::now()
// Test double: rocprofsys::sampling::test::fake_clock
//             - both methods return a manually-advanced counter
//             - FakeClock::advance_ns(delta) advances the shared counter
//             - FakeClock::reset(start_ns) resets to a known base

namespace rocprofsys::sampling
{
class steady_clock;
}
namespace rocprofsys::sampling::test
{
struct fake_clock;
}
