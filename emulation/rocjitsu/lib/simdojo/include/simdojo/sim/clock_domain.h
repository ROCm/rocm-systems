// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file clock_domain.h
/// @brief Clock domain definition for shared frequency, period, and phase among components.

#ifndef SIMDOJO_SIM_CLOCK_DOMAIN_H_
#define SIMDOJO_SIM_CLOCK_DOMAIN_H_

#include "simdojo/sim/sim_types.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace simdojo {

/// @brief A clock domain defines a shared clock source for simulation
/// components.
///
/// @details Components in the same clock domain share identical frequency,
/// period, and phase offset. Derived domains can be created from a parent
/// by using a divider.
///
/// A domain's rising edges are the ticks `phase_offset + k * period` for
/// k >= 1. Everything that converts between cycles and ticks, or rounds a tick
/// onto that grid, belongs here: a component that reimplements it gets a
/// second grid that agrees with this one almost everywhere.
class ClockDomain {
public:
  /// @brief Construct a clock domain.
  /// @param name Human-readable domain name.
  /// @param frequency_hz Requested clock frequency in Hz. The period is the
  ///        number of whole ticks closest to it from below; see
  ///        effective_frequency().
  /// @param phase_offset Phase offset in simulation ticks.
  /// @throws std::invalid_argument if the frequency is zero or above the tick
  ///         resolution, or if the phase leaves no representable first edge.
  ClockDomain(std::string name, uint64_t frequency_hz, Tick phase_offset = 0)
      : ClockDomain(std::move(name), frequency_hz, checked_period(frequency_hz), phase_offset) {}

  /// @brief Create a derived domain with a divided frequency.
  ///
  /// @details The derived period is the parent's period times the divisor, not
  /// the period of the divided frequency. Those differ whenever the parent's
  /// own period was rounded -- a 3 GHz parent has a 333-tick period, and a
  /// quarter-rate child would otherwise get 1333 rather than 1332 -- and the
  /// difference is the whole point of deriving: every edge of the child must
  /// also be an edge of the parent, or the two are just two clocks of similar
  /// frequency and no component in one can sample a component in the other.
  /// @param[in] name Name for the derived domain.
  /// @param[in] divisor Frequency divisor (parent freq / divisor).
  /// @param[in] phase_offset Additional phase offset in simulation ticks.
  /// @returns A new ClockDomain with the derived parameters.
  /// @throws std::invalid_argument if the divisor is zero, or if the derived
  ///         period or phase is not representable.
  ClockDomain derive(std::string name, uint32_t divisor, Tick phase_offset = 0) const {
    if (divisor == 0)
      throw std::invalid_argument("Clock divisor must be positive");
    if (period_ > TICK_MAX / divisor)
      throw std::invalid_argument("Derived clock period exceeds simulation time");
    if (phase_offset > TICK_MAX - phase_offset_)
      throw std::invalid_argument("Derived clock phase offset exceeds simulation time");
    return ClockDomain(std::move(name), frequency_ / divisor, period_ * divisor,
                       phase_offset_ + phase_offset);
  }

  /// @brief Return the human-readable domain name.
  /// @returns Const reference to the domain name.
  const std::string &name() const { return name_; }

  /// @brief Return the requested clock frequency in Hz.
  ///
  /// @details What the domain was asked for, which is the number to print. It
  /// is not the frequency the domain runs at unless it divides the tick
  /// resolution exactly; a report that multiplies a cycle count by this and
  /// compares the answer against elapsed ticks wants effective_frequency().
  /// @returns Requested frequency in Hz.
  uint64_t frequency() const { return frequency_; }

  /// @brief Return the frequency the integral period actually realises.
  ///
  /// @details `TICKS_PER_SECOND / period`. Equal to frequency() when the
  /// requested frequency divides the tick resolution, and above it otherwise,
  /// because the period rounds down: 2.1 GHz becomes a 476-tick period and a
  /// real 2.100840 GHz, which is 190 microseconds of disagreement per billion
  /// cycles.
  /// @returns Realised frequency in Hz.
  uint64_t effective_frequency() const { return TICKS_PER_SECOND / period_; }

  /// @brief Return the clock period in simulation ticks.
  /// @returns Period in ticks, always positive.
  Tick period() const { return period_; }

  /// @brief Return the phase offset in simulation ticks.
  /// @returns Phase offset in ticks.
  Tick phase_offset() const { return phase_offset_; }

  /// @brief Return the tick of the first rising edge for this domain.
  /// @returns First edge tick (phase_offset + period), always representable.
  Tick first_edge() const { return phase_offset_ + period_; }

  /// @brief Round @p tick up to this domain's next rising edge.
  ///
  /// @details A tick at or before first_edge() resolves there: the domain has
  /// no edges before it, so there is nothing earlier to round to, and handing
  /// back anything else would name a tick the framework does not treat as an
  /// edge. A tick already on an edge is returned unchanged, which makes this
  /// idempotent and safe to apply to a value aligned once already.
  ///
  /// TICK_MAX is never an edge: it is the framework's "no such tick" value and
  /// the event queue's empty sentinel, so a queued event at it would be
  /// invisible to the termination check. It is what both alignment helpers
  /// answer when the grid has run out, and a caller reaching it must treat it
  /// as "never" rather than scheduling it -- see Clocked's handler, which stops
  /// the clock instead.
  /// @param tick Tick to align.
  /// @returns The first edge at or after @p tick, or TICK_MAX if there is none.
  Tick next_edge(Tick tick) const {
    const Tick first = first_edge();
    if (tick <= first)
      return first;
    const Tick elapsed = (tick - phase_offset_) % period_;
    if (elapsed == 0)
      return tick;
    return saturating_add(tick, period_ - elapsed);
  }

  /// @brief Return the edge that follows @p edge.
  ///
  /// @details For a caller standing on an edge, which next_edge() would hand
  /// straight back. It is a single addition rather than next_edge()'s modulus,
  /// which matters because the caller is the clock handler: it runs once per
  /// edge per clocked component, and a 64-bit division there is paid on the
  /// busiest path the engine has.
  ///
  /// @p edge is assumed to be an edge of this domain. Passing anything else
  /// returns a tick that is not on the grid; use next_edge() when in doubt.
  /// @param edge An edge of this domain.
  /// @returns The following edge, or TICK_MAX if there is none. See next_edge()
  ///          for why TICK_MAX is not itself an edge.
  Tick edge_after(Tick edge) const { return saturating_add(edge, period_); }

  /// @brief Convert @p cycles of this domain into a duration in ticks.
  ///
  /// @details A duration, not a tick. Saturating it at TICK_MAX only makes the
  /// duration unreachable; adding a saturated duration to a base tick with a
  /// bare `+` wraps it back into the past, which is worse than the overflow it
  /// was guarding. Callers computing when something finishes want deadline(),
  /// and callers doing their own arithmetic want saturating_add().
  /// @param cycles Number of clock cycles.
  /// @returns Equivalent duration in ticks, saturating at TICK_MAX.
  Tick cycles_to_ticks(uint64_t cycles) const {
    return cycles > TICK_MAX / period_ ? TICK_MAX : cycles * period_;
  }

  /// @brief Return the tick @p cycles of work starting at @p start completes.
  ///
  /// @details The composed form of cycles_to_ticks(), and the one to reach for:
  /// it saturates the sum rather than the duration, so an unrepresentable
  /// deadline reads as "never" instead of wrapping into the past.
  /// @param start Tick the work begins.
  /// @param cycles Duration of the work in this domain's cycles.
  /// @returns Completion tick, saturating at TICK_MAX.
  Tick deadline(Tick start, uint64_t cycles) const {
    return saturating_add(start, cycles_to_ticks(cycles));
  }

  /// @brief Convert a duration in @p ticks into whole cycles of this domain.
  ///
  /// @details Truncates: a duration that does not fill a cycle is no cycles,
  /// and the remainder is dropped rather than rounded up. Callers costing work
  /// want the opposite rounding and should say so themselves; this is the plain
  /// conversion.
  /// @param ticks Duration in simulation ticks.
  /// @returns Number of complete cycles in @p ticks.
  uint64_t ticks_to_cycles(Tick ticks) const { return ticks / period_; }

  /// @brief Cycles to move @p units of work at @p units_per_cycle, rounding up.
  ///
  /// @details The other rounding direction from ticks_to_cycles(), and the one
  /// a component costing work wants: a cache asked for eight lines at two
  /// lines per cycle is busy for four. Never zero, even for an empty request,
  /// because a component handed work has looked at it and a rate high enough
  /// to round the work away would otherwise make it infinitely fast out of a
  /// rounding mode.
  ///
  /// Static because a rate is not a property of a clock domain; it is here
  /// because this is where cycle arithmetic lives.
  /// @param units Amount of work, in whatever unit the rate is expressed in.
  /// @param units_per_cycle Service rate; a non-positive rate means one unit
  ///        per cycle.
  /// @returns Service duration in cycles, at least one.
  static uint64_t service_cycles(uint64_t units, double units_per_cycle) {
    constexpr uint64_t kMaxCycles = std::numeric_limits<uint64_t>::max();
    // NaN lands here too, deliberately: it is not a rate.
    if (!(units_per_cycle > 0.0))
      return units == 0 ? 1 : units;
    // A whole-number rate is done in integers: static_cast<double>(units) drops
    // the low bits above 2^53, so the double path under-charges a large request
    // and makes a rate of 1.0 disagree with the rate of 0.0 handled above.
    if (units_per_cycle <= 9007199254740992.0 && units_per_cycle == std::floor(units_per_cycle)) {
      if (units == 0)
        return 1;
      const uint64_t rate = static_cast<uint64_t>(units_per_cycle);
      const uint64_t whole = units / rate + (units % rate != 0 ? 1 : 0);
      return whole == 0 ? 1 : whole;
    }
    const double cycles = std::ceil(static_cast<double>(units) / units_per_cycle);
    if (!(cycles > 1.0))
      return 1;
    if (cycles >= static_cast<double>(kMaxCycles))
      return kMaxCycles;
    return static_cast<uint64_t>(cycles);
  }

private:
  /// @brief Construct with an already-computed period.
  /// @param name Human-readable domain name.
  /// @param frequency_hz Requested frequency, recorded but not re-derived.
  /// @param period Period in ticks; must be positive.
  /// @param phase_offset Phase offset in simulation ticks.
  /// @throws std::invalid_argument if the phase leaves no representable first edge.
  ClockDomain(std::string name, uint64_t frequency_hz, Tick period, Tick phase_offset)
      : name_(std::move(name)), frequency_(frequency_hz), period_(period),
        phase_offset_(phase_offset) {
    // first_edge() is the base of every alignment below, and it is a bare
    // addition. Checking it once here is what lets the rest assume the grid
    // starts somewhere representable.
    if (phase_offset_ > TICK_MAX - period_)
      throw std::invalid_argument("Clock phase leaves no representable rising edge");
  }

  /// @brief The period of @p frequency_hz, rejecting one that has none.
  ///
  /// @details A frequency above the one-picosecond tick resolution rounds to a
  /// zero period, and a zero period is not a fast clock: it is a division by
  /// zero in every conversion and alignment above, and a modulus by zero in
  /// next_edge(). Rejecting it here is what makes period() positive by
  /// construction, so none of them has to test for it.
  static Tick checked_period(uint64_t frequency_hz) {
    if (frequency_hz == 0)
      throw std::invalid_argument("Clock frequency must be positive");
    const Tick period = TICKS_PER_SECOND / frequency_hz;
    if (period == 0)
      throw std::invalid_argument("Clock frequency exceeds the simulation tick resolution");
    return period;
  }

  const std::string name_;
  const uint64_t frequency_;
  const Tick period_;
  const Tick phase_offset_;
};

} // namespace simdojo

#endif // SIMDOJO_SIM_CLOCK_DOMAIN_H_
