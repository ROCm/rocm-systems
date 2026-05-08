// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file clocked.h
/// @brief CRTP mixin for clock-driven simulation components.

#ifndef SIMDOJO_SIM_CLOCKED_H_
#define SIMDOJO_SIM_CLOCKED_H_

#include "simdojo/sim/clock_domain.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/event_queue.h"

#include <string>
#include <utility>

namespace simdojo {

/// @brief CRTP mixin for components that operate on a clock.
///
/// @details Derives clock period and phase from a ClockDomain. Subclasses
/// override advance() which is called on each rising edge. Owns a
/// reusable Event that is enqueued into the engine at each cycle.
///
/// @tparam Base A Component-derived type (i.e., Component, CompositeComponent).
template <typename Base> class Clocked : public Base {
public:
  /// @brief Construct a clocked component in the given clock domain.
  /// @param name Human-readable component name.
  /// @param domain Clock domain that provides period and phase.
  Clocked(std::string name, const ClockDomain &domain) : Base(std::move(name)), domain_(domain) {}

  /// @brief Schedule the first clock-edge event when the simulation starts.
  void initialize() override {
    assert(this->engine() && "Clocked component must be added to a topology before initialize()");
    running_ = true;
    this->schedule_event(&clock_event_, domain_.first_edge());
  }

  /// @brief Resume clocking from the next clock edge at or after the given tick.
  /// No-op if already running.
  /// @param after Earliest tick from which to resume clocking.
  void resume_clock(Tick after) {
    if (running_)
      return;
    running_ = true;
    Tick first = domain_.first_edge();
    if (after < first) {
      this->schedule_event(&clock_event_, first);
      return;
    }
    Tick elapsed = (after - domain_.phase_offset()) % period();
    Tick remainder = period() - elapsed;
    Tick next =
        (elapsed == 0) ? after : ((after > TICK_MAX - remainder) ? TICK_MAX : after + remainder);
    this->schedule_event(&clock_event_, next);
  }

  /// @brief Return whether the clock is currently running.
  /// @retval true Clock is active and scheduling events.
  /// @retval false Clock is stopped.
  bool running() const { return running_; }

  /// @brief Return the clock domain this component belongs to.
  /// @returns Const reference to the clock domain.
  const ClockDomain &clock_domain() const { return domain_; }

  /// @brief Return clock period in simulation ticks.
  /// @returns Period in ticks.
  Tick period() const { return domain_.period(); }

  /// @brief Return clock frequency in Hz.
  /// @returns Frequency in Hz.
  uint64_t frequency() const { return domain_.frequency(); }

  /// @brief Execute one quantum of work on the rising clock edge.
  /// @param now The simulation tick of this clock edge.
  /// @retval true Continue clocking.
  /// @retval false Halt this component's clock.
  virtual bool advance(Tick now) = 0;

private:
  const ClockDomain &domain_; ///< Clock source for period/phase.
  /// @brief Reusable clock edge event. Handler re-enqueues on the next edge
  /// if advance() returns true, otherwise stops the clock.
  Event clock_event_{this, EventType::TIMER_CALLBACK, [this](Tick now, Message *) {
                       if (advance(now)) {
                         Tick next = (now > TICK_MAX - period()) ? TICK_MAX : now + period();
                         this->schedule_event(&clock_event_, next);
                       } else {
                         running_ = false;
                       }
                     }};
  bool running_ = false; ///< True while the clock is active.
};

} // namespace simdojo

#endif // SIMDOJO_SIM_CLOCKED_H_
