// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file engine.h
/// @brief The event loop the timed components run on.
///
/// @details Built from the simulation framework's own event queue, over the
/// same one-picosecond tick axis, with the same priority rules. It is a
/// *separate* queue from the functional engine's, and that is a deliberate
/// decision rather than a shortcut.
///
/// The two axes are not the same axis. The functional engine's tick is
/// bookkeeping: a compute unit reschedules itself one tick per instruction it
/// executed, so a tick there is roughly an instruction and a microsecond of
/// modelled hardware time is two million of them. The timing plane's tick is
/// modelled hardware time. Sharing one queue would put every timing event
/// millions of ticks beyond every functional event, so the functional side
/// would run to completion first and the whole timing plane would drain at the
/// end -- with every pending request live in the queue at once. It would also
/// make the engine's global time mean neither thing.
///
/// Keeping them separate costs nothing in fidelity. Nothing in the timing plane
/// feeds back into execution: the functional simulator has already resolved
/// every branch, every atomic and every memory ordering by the time the plane
/// sees the instruction. What the plane produces is time, and time is read at
/// the points where the guest can observe it -- a dispatch's completion signal,
/// a clock ioctl, an in-kernel counter read -- which is where the plane is run
/// to quiescence and the result published.

#pragma once

#include "simdojo/sim/event_queue.h"
#include "simdojo/sim/sim_types.h"

#include <cstdint>
#include <vector>

namespace rocjitsu::timing {

class TimedComponent;

/// @brief A discrete-event loop over modelled hardware time.
class TimingEngine {
public:
  TimingEngine() = default;

  /// @brief Register a component so the engine can wake it.
  void add(TimedComponent *component);

  /// @brief Ask for @p component to be woken at @p tick.
  ///
  /// @details A component may have at most one outstanding wake. Asking again
  /// for an earlier tick moves it; asking for a later one is ignored. That cap
  /// is what bounds the queue: it holds at most one entry per component,
  /// however much work is in flight.
  void wake(TimedComponent *component, std::uint64_t tick);

  /// @brief Run until no component has work left.
  /// @returns The tick the last event fired at.
  std::uint64_t run_until_idle();

  /// @brief Run until @p done becomes true, or nothing is left to run.
  ///
  /// @details A compute unit asks the hierarchy what an access cost and needs
  /// the answer before it can issue the wavefront's next instruction, so the
  /// plane runs the components forward until that one request has been
  /// answered. Every other request in flight advances with it, which is what
  /// keeps contention between wavefronts real: this is a bounded run of a
  /// shared simulation, not a private one per access.
  std::uint64_t run_until(const bool &done);

  /// @brief The tick the engine has advanced to.
  std::uint64_t now() const { return now_; }

  /// @brief Events executed so far, for the overhead report.
  std::uint64_t events() const { return events_; }

private:
  struct Pending {
    std::uint64_t tick = 0;
    std::uint64_t sequence = 0;
    TimedComponent *component = nullptr;
    /// @brief Min-heap ordering: earliest tick first, then insertion order.
    bool operator>(const Pending &other) const {
      if (tick != other.tick)
        return tick > other.tick;
      return sequence > other.sequence;
    }
  };

  std::vector<Pending> heap_;
  std::vector<TimedComponent *> components_;
  std::uint64_t now_ = 0;
  std::uint64_t sequence_ = 0;
  std::uint64_t events_ = 0;
};

} // namespace rocjitsu::timing
