// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sim_types.h
/// @brief Core type aliases, constants, and enumerations for the simulation framework.

#ifndef SIMDOJO_SIM_TYPES_H_
#define SIMDOJO_SIM_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace simdojo {

/// @brief Cache line size for alignment (64 bytes for x86-64 and most AArch64).
inline constexpr std::size_t CACHE_LINE_SIZE = 64;

/// @brief Simulation tick type.
using Tick = uint64_t;

/// @brief Simulation tick resolution: 1 tick = 1 picosecond (1 THz).
///
/// Fixed, global minimum simulation time quantum. A 1 GHz clock has a
/// period of 1000 ticks (1 ns), a 2 GHz clock has 500 ticks (500 ps), etc.
inline constexpr Tick TICKS_PER_SECOND = 1'000'000'000'000ULL;

/// @brief Maximum tick value (used as a sentinel for "no event").
inline constexpr Tick TICK_MAX = std::numeric_limits<Tick>::max();

/// @brief Add two tick values, saturating at TICK_MAX rather than wrapping.
///
/// @details Simulation time is unsigned, so an overflow does not produce a
/// large wrong answer -- it produces a small one, and a duration that wraps to
/// "immediately" is indistinguishable from work that really is instantaneous.
/// Saturating turns it into "never", which is visible in a result.
/// @param a First addend in ticks.
/// @param b Second addend in ticks.
/// @returns @p a + @p b, or TICK_MAX if that would overflow.
inline constexpr Tick saturating_add(Tick a, Tick b) { return a > TICK_MAX - b ? TICK_MAX : a + b; }

/// @brief Unique identifier for a Node or Component.
using ComponentID = uint32_t;

/// @brief Unique identifier for a Link.
using LinkID = uint32_t;

/// @brief Unique identifier for a Port.
using PortID = uint32_t;

/// @brief Identifier for a simulation partition (thread affinity).
using PartitionID = uint32_t;

/// @brief Sentinel value indicating no partition has been assigned.
inline constexpr PartitionID INVALID_PARTITION_ID = std::numeric_limits<PartitionID>::max();

/// @brief Reason the simulation stopped.
enum class ExitReason : uint8_t {
  COMPLETED,    ///< Normal completion (all children halted or max_ticks reached).
  EXIT_REQUEST, ///< A component called request_exit().
  INTERRUPTED,  ///< External interrupt (ctrl-C, watchdog, debugger).
};

/// @brief Information about why the simulation stopped.
class ExitStatus {
public:
  ExitStatus() = default;
  ExitStatus(ExitReason r, Tick t, std::string msg, int c = 0)
      : reason(r), tick(t), message(std::move(msg)), code(c) {}

  ExitReason reason = ExitReason::COMPLETED;
  Tick tick = 0;       ///< Simulation tick at which the exit occurred.
  std::string message; ///< Human-readable reason.
  int code = 0;        ///< Exit code (0 = success).
};

} // namespace simdojo

#endif // SIMDOJO_SIM_TYPES_H_
