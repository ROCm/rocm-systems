// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file functional.h
/// @brief CRTP mixin for functional (untimed) simulation components.

#ifndef SIMDOJO_SIM_FUNCTIONAL_H_
#define SIMDOJO_SIM_FUNCTIONAL_H_

#include "simdojo/sim/component.h"
#include "simdojo/sim/event_queue.h"

#include <string>
#include <utility>

namespace simdojo {

/// @brief CRTP mixin for functional (untimed) simulation components.
///
/// @details Provides an execution model where each call to advance()
/// represents one logical operation (e.g., one instruction, one memory
/// request). A monotonic tick counter is passed to advance() as a synthetic
/// timestamp, but the value has no relation to wall time or clock edges -
/// it simply counts invocations. This allows functional components to
/// participate in the same event-driven loop as timed components, including
/// parallel execution with PDES synchronization when using multiple threads.
///
/// Subclasses override advance(Tick), which is called once per tick. Return
/// true to continue advancing, false to halt this component.
///
/// @tparam Base A Component-derived type (Component or CompositeComponent).
template <typename Base> class Functional : public Base {
public:
  /// @brief Construct a functional component.
  /// @param name Human-readable component name.
  Functional(std::string name) : Base(std::move(name)) {}

  /// @brief Schedule the first advance event when the simulation starts.
  void initialize() override {
    assert(this->engine() != nullptr &&
           "Functional component must be added to a topology before initialize()");
    active_ = true;
    this->schedule_event(&step_event_, 1);
  }

  /// @brief Execute one quantum of work. Return true to continue, false to halt.
  /// @param now Monotonic tick counter (synthetic timestamp, not wall time).
  /// @retval true Continue advancing.
  /// @retval false Halt this component.
  virtual bool advance(Tick now) = 0;

  /// @brief Whether this component is still actively advancing.
  /// @retval true Component is actively advancing.
  /// @retval false Component has halted.
  bool active() const { return active_; }

private:
  Tick tick_counter_ = 0; ///< Monotonic invocation counter passed to advance().
  /// @brief Reusable event that drives advance() each epoch. If advance()
  /// returns true, re-enqueues for the next tick; otherwise marks the
  /// component inactive.
  ///
  /// NOTE: `now + 1` schedules for the next picosecond tick, meaning "advance
  /// as soon as possible", NOT "one logical step per engine epoch". Since the
  /// simulation tick resolution is 1 ps (1 THz), a functional component will
  /// be called once per picosecond. In mixed-mode simulations this means
  /// functional components are invoked ~1000x more often than a 1 GHz clocked
  /// component (which has a 1000 ps period). This is by design: functional
  /// components run at maximum simulation speed with no artificial gating.
  Event step_event_{this, EventType::TIMER_CALLBACK, [this](Tick now, Message *) {
                      if (advance(tick_counter_++)) {
                        this->schedule_event(&step_event_, now + 1);
                      } else {
                        active_ = false;
                      }
                    }};
  bool active_ = false; ///< True while this component is advancing.
};

} // namespace simdojo

#endif // SIMDOJO_SIM_FUNCTIONAL_H_
