// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/engine.h"

#include "rocjitsu/vm/timing/timed_component.h"

#include <algorithm>

namespace rocjitsu::timing {

void TimingEngine::add(TimedComponent *component) {
  if (component != nullptr)
    components_.push_back(component);
}

void TimingEngine::wake(TimedComponent *component, std::uint64_t tick) {
  if (component == nullptr)
    return;
  // Never schedule into the past: a component asked to wake at a tick the
  // engine has already passed runs at the current tick instead. This happens
  // legitimately, because a request can be handed to a component that is
  // already behind, and refusing it would strand the request forever.
  tick = std::max(tick, now_);
  if (component->wake_scheduled_) {
    if (tick >= component->wake_at_)
      return;
    // An earlier wake supersedes a later one. The stale heap entry is left in
    // place and skipped when it fires, which is cheaper than finding and
    // removing it and is why entries carry the tick they were made for.
    component->wake_at_ = tick;
  } else {
    component->wake_scheduled_ = true;
    component->wake_at_ = tick;
  }
  heap_.push_back(Pending{tick, sequence_++, component});
  std::push_heap(heap_.begin(), heap_.end(), std::greater<Pending>());
}

std::uint64_t TimingEngine::run_until_idle() {
  static constexpr bool never = false;
  return run_until(never);
}

std::uint64_t TimingEngine::run_until(const bool &done) {
  while (!heap_.empty() && !done) {
    std::pop_heap(heap_.begin(), heap_.end(), std::greater<Pending>());
    const Pending entry = heap_.back();
    heap_.pop_back();
    TimedComponent *component = entry.component;
    // Skip an entry a later wake() superseded. Both the flag and the tick are
    // checked: the flag alone would drop a legitimate re-arm made while this
    // entry was still in the queue.
    if (!component->wake_scheduled_ || component->wake_at_ != entry.tick)
      continue;
    now_ = std::max(now_, entry.tick);
    component->wake_scheduled_ = false;
    ++events_;
    const std::uint64_t next = component->advance(now_);
    if (next != 0)
      wake(component, next);
  }
  return now_;
}

} // namespace rocjitsu::timing
