// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timed_component.h
/// @brief The base every timed component derives from.
///
/// @details A timed component is a simulation component like any other: it
/// lives in the machine's component tree, it belongs to a clock domain, and it
/// makes progress by executing events. What it is not is *clocked*: it does not
/// schedule an event per clock edge. At the one-picosecond tick resolution a
/// 2.1 GHz domain is a 476-tick period, so a clocked component costs one event
/// per cycle, and a part with 256 compute units and 128 memory channels would
/// spend its entire budget on empty edges.
///
/// Instead each component behaves the way a hardware block's model should: it
/// holds a queue of work, it knows the tick it is next free, and it keeps at
/// most one event of its own in the queue at a time. That is the timestamp
/// advance idiom -- the same one a DRAM controller uses when it tracks when each
/// bank may next be activated rather than stepping every bank every cycle -- and
/// it makes the event count proportional to the work rather than to the elapsed
/// time.
///
/// Deriving a timed version of an existing functional component is therefore
/// mechanical: template the component on ExecMode, derive from
/// ExecBase<Mode, Component>, express delays in cycles rather than ticks, and
/// call reserve() for a resource that serialises or schedule_wake() for one
/// that pipelines. docs/timing.md carries the worked recipe.

#pragma once

#include "rocjitsu/vm/timing/request.h"

#include "simdojo/sim/clock_domain.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/event_queue.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::timing {

class TimingEngine;

/// @brief A component that advances on scheduled events rather than on edges.
class TimedComponent : public simdojo::Component {
public:
  TimedComponent(std::string name, const simdojo::ClockDomain &domain, TimingEngine &engine);
  ~TimedComponent() override;

  /// @brief Accept a request. Called by the component upstream.
  ///
  /// @details Never does the work: it queues the request and makes sure this
  /// component has an event scheduled for when it can next act. Doing the work
  /// here instead would collapse the whole hierarchy into the caller's stack
  /// and there would be no modelled time in it.
  void deliver(const MemoryRequest &request);

  /// @brief Do whatever is due at @p now. Returns the tick to wake at next, or
  ///        zero when there is nothing left to do.
  virtual std::uint64_t advance(std::uint64_t now) = 0;

  /// @brief Round @p tick up to this component's next clock edge.
  std::uint64_t align_to_edge(std::uint64_t tick) const;

  /// @brief Occupy this component for @p cycles starting no earlier than
  ///        @p ready, and no earlier than when it is next free.
  /// @returns The tick the work completes.
  std::uint64_t reserve(std::uint64_t ready, std::uint64_t cycles);

  std::uint64_t busy_until() const { return busy_until_; }
  std::uint64_t period() const { return domain_.period(); }
  const simdojo::ClockDomain &clock_domain() const { return domain_; }

  /// @brief Ticks for @p cycles of this component's clock, saturating.
  std::uint64_t cycles_to_ticks(std::uint64_t cycles) const;

  bool is_composite() const override { return false; }

protected:
  /// @brief Ask to be woken at @p tick. Collapses onto the earliest request.
  void schedule_wake(std::uint64_t tick);

  std::vector<MemoryRequest> &inbox() { return inbox_; }
  TimingEngine &engine() { return engine_; }

private:
  const simdojo::ClockDomain &domain_;
  TimingEngine &engine_;
  std::vector<MemoryRequest> inbox_;
  std::uint64_t busy_until_ = 0;
  std::uint64_t wake_at_ = 0;
  bool wake_scheduled_ = false;

  friend class TimingEngine;
};

} // namespace rocjitsu::timing
