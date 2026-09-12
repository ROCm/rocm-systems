// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file clocked.h
/// @brief CRTP mixin for clock-driven simulation components.

#ifndef SIMDOJO_SIM_CLOCKED_H_
#define SIMDOJO_SIM_CLOCKED_H_

#include "simdojo/sim/clock_domain.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/event_queue.h"

#include <cassert>
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
  ///
  /// @details The domain is copied, not borrowed. A `ClockDomain` is four
  /// integers and a name, all const for its whole life, so a copy can never
  /// disagree with the original and there is nothing to keep in sync -- while
  /// a reference has to be right about a lifetime that this class cannot see.
  ///
  /// Borrowing was the earlier design, guarded by deleting the rvalue
  /// overload. That guard does not hold: it is a guard on *this* constructor,
  /// and every concrete component reaches it through its own, which takes
  /// `const ClockDomain &` and forwards a named parameter -- an lvalue. So
  ///
  /// ```
  /// Hart("h", ClockDomain("tmp", 1'000'000'000));
  /// ```
  ///
  /// compiled, bound the member to a temporary, and left every later edge
  /// computation reading a destroyed object. Because a domain is nothing but
  /// integers the read usually succeeds, so the symptom was a silently wrong
  /// schedule rather than a crash. Restoring the guard would mean deleting an
  /// rvalue overload on every clocked type ever written, and being wrong
  /// exactly once is enough to reintroduce the bug; owning the value removes
  /// the class of error instead of naming its instances.
  ///
  /// It also makes `parent.derive("child", 2)` -- a temporary, and the natural
  /// way to write a divided clock -- expressible rather than a compile error.
  /// @param name Human-readable component name.
  /// @param domain Clock domain that provides period and phase.
  Clocked(std::string name, ClockDomain domain)
      : Base(std::move(name)), domain_(std::move(domain)) {}

  /// @brief Schedule the first clock-edge event when the simulation starts.
  void startup() override {
    assert(this->engine() && "Clocked component must be added to a topology before startup()");
    // The one clock_event_ is reusable but not re-entrant: arming it twice
    // queues two entries and advances this component twice per edge.
    assert(!running_ && "Clocked component was already clocking before startup()");
    running_ = true;
    this->schedule_event(&clock_event_, domain_.first_edge());
  }

  /// @brief Stop clocking so a later startup() can re-arm the event.
  ///
  /// @details SimulationEngine::create() also rebuilds after shutdown(). Without
  /// this, running_ survives teardown and startup()'s precondition fails for a
  /// component that was simply still clocking when the previous run ended.
  void shutdown() override {
    running_ = false;
    Base::shutdown();
  }

  /// @brief Resume clocking from the next clock edge at or after the given tick.
  ///
  /// @details No-op if already running, and no-op if the domain has no edge at
  /// or after @p after -- the clock stays stopped rather than being armed at
  /// TICK_MAX. @p after is not compared against the engine's current tick: a
  /// caller that passes a tick already past schedules an edge in the past and
  /// moves simulation time backwards.
  /// @param after Earliest tick from which to resume clocking.
  void resume_clock(Tick after) {
    if (running_)
      return;
    const Tick next = domain_.next_edge(after);
    // Same rule as the edge handler below: TICK_MAX must never be scheduled.
    if (next == TICK_MAX)
      return;
    running_ = true;
    this->schedule_event(&clock_event_, next);
  }

  /// @brief Return whether the clock is currently running.
  /// @retval true Clock is active and scheduling events.
  /// @retval false Clock is stopped.
  bool running() const { return running_; }

  /// @brief Return the clock domain this component belongs to.
  /// @returns Const reference to the clock domain.
  const ClockDomain &clock_domain() const { return domain_; }

  /// @brief Execute one quantum of work on the rising clock edge.
  /// @param now The simulation tick of this clock edge.
  /// @retval true Continue clocking.
  /// @retval false Halt this component's clock.
  virtual bool advance(Tick now) = 0;

private:
  const ClockDomain domain_; ///< Clock source for period/phase. Owned; see the constructor.
  /// @brief Reusable clock edge event. Handler re-enqueues on the next edge
  /// if advance() returns true, otherwise stops the clock.
  Event clock_event_{this, EventType::TIMER_CALLBACK, [this](Tick now, Message *) {
                       const Tick next = advance(now) ? domain_.edge_after(now) : TICK_MAX;
                       if (next != TICK_MAX) {
                         this->schedule_event(&clock_event_, next);
                       } else {
                         // Either advance() asked to stop, or the domain has no
                         // edge left. See ClockDomain::next_edge() for why
                         // TICK_MAX must never be scheduled.
                         running_ = false;
                       }
                     }};
  bool running_ = false; ///< True while the clock is active.
};

} // namespace simdojo

#endif // SIMDOJO_SIM_CLOCKED_H_
