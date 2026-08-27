// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/timed_component.h"

#include "rocjitsu/vm/timing/engine.h"

#include <algorithm>
#include <utility>

namespace rocjitsu::timing {

TimedComponent::TimedComponent(std::string name, const simdojo::ClockDomain &domain,
                               TimingEngine &engine)
    : simdojo::Component(std::move(name)), domain_(domain), engine_(engine) {
  engine_.add(this);
}

TimedComponent::~TimedComponent() = default;

std::uint64_t TimedComponent::cycles_to_ticks(std::uint64_t cycles) const {
  const std::uint64_t step = domain_.period();
  if (step == 0)
    return 0;
  // Saturate rather than wrap. An absurd duration should read as "never", which
  // is visible, and not as "immediately", which is not.
  if (cycles > simdojo::TICK_MAX / step)
    return simdojo::TICK_MAX;
  return cycles * step;
}

std::uint64_t TimedComponent::align_to_edge(std::uint64_t tick) const {
  const std::uint64_t step = domain_.period();
  if (step == 0)
    return tick;
  const std::uint64_t phase = domain_.phase_offset();
  if (tick <= phase)
    return phase;
  const std::uint64_t elapsed = (tick - phase) % step;
  if (elapsed == 0)
    return tick;
  const std::uint64_t remainder = step - elapsed;
  return tick > simdojo::TICK_MAX - remainder ? simdojo::TICK_MAX : tick + remainder;
}

std::uint64_t TimedComponent::reserve(std::uint64_t ready, std::uint64_t cycles) {
  const std::uint64_t start = align_to_edge(std::max(ready, busy_until_));
  const std::uint64_t span = cycles_to_ticks(cycles);
  busy_until_ = start > simdojo::TICK_MAX - span ? simdojo::TICK_MAX : start + span;
  return busy_until_;
}

void TimedComponent::deliver(const MemoryRequest &request) {
  inbox_.push_back(request);
  // The component may already be busy; waking it at the later of "now" and
  // "when it is free" is what turns a burst of arrivals into a queue rather
  // than into simultaneous service.
  schedule_wake(std::max(request.issued_tick, busy_until_));
}

void TimedComponent::schedule_wake(std::uint64_t tick) { engine_.wake(this, tick); }

} // namespace rocjitsu::timing
