// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file time_source.h
/// @brief The narrow view of a timing model that the guest clock consumes.
///
/// @details Separate from TimingModel because the clock has different callers
/// and much harder constraints than the observation path: it is read from guest
/// threads inside an ioctl and from an in-kernel s_memtime, on any thread, at
/// any time, possibly while execution callbacks are running elsewhere.
/// Narrowing the type states that, and keeps SimulatedClock independent of the
/// plugin machinery so it can be tested with a two-line stub.

#pragma once

#include <cstdint>

namespace rocjitsu::timing {

class TimeSource {
public:
  virtual ~TimeSource() = default;

  /// @brief The current simulated time, in shader-clock cycles.
  ///
  /// @details Must not take locks and must not move backwards. Both matter
  /// because the value reaches guest-visible timers: a non-monotonic clock
  /// makes a program compute a negative elapsed time, which is worse than a
  /// clock that is merely wrong.
  virtual std::uint64_t current_cycles() const = 0;

  /// @brief The shader clock rate. Fixed for the run.
  virtual double clock_ghz() const = 0;
};

} // namespace rocjitsu::timing
