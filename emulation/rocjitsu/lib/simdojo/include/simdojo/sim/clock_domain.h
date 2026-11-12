// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file clock_domain.h
/// @brief Clock domain definition for shared frequency, period, and phase among components.

#ifndef SIMDOJO_SIM_CLOCK_DOMAIN_H_
#define SIMDOJO_SIM_CLOCK_DOMAIN_H_

#include "simdojo/sim/sim_types.h"

#include "util/diagnostic.h"
#include "util/result.h"
#include <cstdint>
#include <string>
#include <utility>

namespace simdojo {

/// @brief A clock domain defines a shared clock source for simulation
/// components.
///
/// @details Components in the same clock domain share identical frequency,
/// period, and phase offset. Derived domains can be created from a parent
/// by using a divider.
class ClockDomain {
public:
  /// @brief Create a clock domain with a positive frequency.
  /// @param name Human-readable domain name.
  /// @param frequency_hz Clock frequency in Hz.
  /// @param phase_offset Phase offset in simulation ticks.
  /// @param emit_error Destination for failure diagnostics. Omitting it discards the message.
  /// @returns A domain, or failure with a diagnostic if the frequency is zero
  /// or exceeds TICKS_PER_SECOND.
  static util::FailureOr<ClockDomain> create(std::string name, uint64_t frequency_hz,
                                             Tick phase_offset = 0,
                                             util::DiagnosticEmitter emit_error = {}) {
    if (frequency_hz == 0) {
      return emit_error.emit() << "Clock frequency must be positive";
    }
    if (frequency_hz > TICKS_PER_SECOND)
      return emit_error.emit() << "Clock frequency must not exceed " << TICKS_PER_SECOND << " Hz";
    return ClockDomain(std::move(name), frequency_hz, phase_offset);
  }

  /// @brief Create a derived domain with a divided frequency.
  /// @param[in] name Name for the derived domain.
  /// @param[in] divisor Frequency divisor (parent freq / divisor).
  /// @param[in] phase_offset Additional phase offset in simulation ticks.
  /// @param emit_error Destination for failure diagnostics. Omitting it discards the message.
  /// @returns A domain, or diagnosed failure for a zero divisor or derived frequency.
  util::FailureOr<ClockDomain> derive(std::string name, uint32_t divisor, Tick phase_offset = 0,
                                      util::DiagnosticEmitter emit_error = {}) const {
    if (divisor == 0) {
      return emit_error.emit() << "Clock divisor must be positive";
    }
    return create(std::move(name), frequency_ / divisor, phase_offset_ + phase_offset, emit_error);
  }

  /// @brief Return the human-readable domain name.
  /// @returns Const reference to the domain name.
  const std::string &name() const { return name_; }

  /// @brief Return the clock frequency in Hz.
  /// @returns Frequency in Hz.
  uint64_t frequency() const { return frequency_; }

  /// @brief Return the clock period in simulation ticks.
  /// @returns Period in ticks.
  Tick period() const { return period_; }

  /// @brief Return the phase offset in simulation ticks.
  /// @returns Phase offset in ticks.
  Tick phase_offset() const { return phase_offset_; }

  /// @brief Return the tick of the first rising edge for this domain.
  /// @returns First edge tick (phase_offset + period).
  Tick first_edge() const { return phase_offset_ + period_; }

private:
  // Only create() calls this constructor, after validating a usable frequency.
  ClockDomain(std::string name, uint64_t frequency_hz, Tick phase_offset)
      : name_(std::move(name)), frequency_(frequency_hz), period_(TICKS_PER_SECOND / frequency_hz),
        phase_offset_(phase_offset) {}

  const std::string name_;
  const uint64_t frequency_;
  const Tick period_;
  const Tick phase_offset_;
};

} // namespace simdojo

#endif // SIMDOJO_SIM_CLOCK_DOMAIN_H_
