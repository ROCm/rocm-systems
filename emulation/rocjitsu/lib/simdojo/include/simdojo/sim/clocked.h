// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file clocked.h
/// @brief CRTP mixin for clock-driven simulation components.

#ifndef SIMDOJO_SIM_CLOCKED_H_
#define SIMDOJO_SIM_CLOCKED_H_

#include "simdojo/sim/clock_domain.h"
#include "simdojo/sim/component.h"
#include "simdojo/sim/event_queue.h"

#include <algorithm>
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
  /// @param domain Clock domain that provides period and phase. Held by
  ///        reference; it must outlive this component, which is what the
  ///        topology owning every domain guarantees.
  Clocked(std::string name, const ClockDomain &domain) : Base(std::move(name)), domain_(domain) {}

  /// @brief Refuse a temporary clock domain.
  ///
  /// @details Binding domain_ to a temporary compiles cleanly and leaves every
  /// later edge computation reading freed memory. Because a domain is nothing
  /// but integers the read usually succeeds, so the symptom is a silently
  /// wrong schedule rather than a crash.
  Clocked(std::string name, ClockDomain &&domain) = delete;

  /// @brief Schedule the first clock-edge event when the simulation starts,
  ///        unless this component clocks only on demand.
  ///
  /// @details Also clears any reservation, because a create() generation
  /// starts from tick zero and a completion tick from the last one would leave
  /// this component looking busy until a tick that will never arrive again.
  void startup() override {
    assert(this->engine() && "Clocked component must be added to a topology before startup()");
    busy_until_ = 0;
    if (clock_at_startup())
      arm_edge(domain_.first_edge());
  }

  /// @brief Whether this component should begin clocking at startup.
  ///
  /// @details True for a component that genuinely runs every edge. A component
  /// that instead advances on demand -- one that knows the tick it is next due
  /// and would otherwise spend its whole budget on empty edges -- overrides
  /// this to false and drives itself with wake_at(). At the one-picosecond tick
  /// resolution a 2.1 GHz domain is a 476-tick period, so a model with hundreds
  /// of such components cannot afford an edge each.
  /// @retval true Clock starts at the domain's first edge.
  /// @retval false Nothing is scheduled until wake_at() asks for it.
  virtual bool clock_at_startup() const { return true; }

  /// @brief Ask to advance at @p tick, rounded up to this domain's next edge.
  ///
  /// @details Collapses onto the earliest outstanding request, so a component
  /// keeps at most one live entry in the queue however much work is in flight.
  /// Works while the component is already running, which is how one that
  /// advances on demand schedules its own next visit from inside advance().
  ///
  /// A tick already past is pulled forward to now *before* being aligned. The
  /// engine would otherwise clamp the aligned edge up to the current tick,
  /// which is not an edge of this domain, and every edge after it would
  /// inherit the offset -- a component woken by a request that reached it late
  /// would leave its own clock grid permanently.
  ///
  /// A tick beyond the domain's last representable edge is ignored, because
  /// there is no edge to wake on.
  ///
  /// Asked from inside advance(), the floor is the edge *after* the one being
  /// advanced, not that edge itself. next_edge() is idempotent on an edge, so
  /// a tick at or before now would otherwise align to now, and the entry the
  /// engine pushes at the tick it is already draining is popped again in the
  /// same iteration -- advance() re-entered at the same edge, with simulated
  /// time frozen, for as long as it keeps asking.
  /// @param tick Earliest tick to advance at.
  void wake_at(Tick tick) { arm_edge(domain_.next_edge(std::max(tick, wake_floor()))); }

  /// @brief Occupy this component for @p cycles, starting no earlier than
  ///        @p ready and no earlier than when it is next free.
  ///
  /// @details The timestamp-advance idiom: a resource that serialises tracks
  /// when it is next available rather than stepping every cycle. Link::latency
  /// is fixed propagation and QueuedLink buffers without serialising, so
  /// neither expresses a server that is busy, and being busy is how a queueing
  /// delay gets into a model at all -- a request arriving behind others is
  /// served late because this pushes its start out.
  ///
  /// Independent of the clock: reserving does not schedule anything. A
  /// component that wants to act at the completion tick asks for it with
  /// wake_at().
  /// @param ready Earliest tick the work could start.
  /// @param cycles Duration of the work in this domain's cycles.
  /// @returns The tick the work completes, or TICK_MAX if that is beyond
  ///          representable time.
  Tick reserve(Tick ready, uint64_t cycles) {
    busy_until_ = domain_.deadline(domain_.next_edge(std::max(ready, busy_until_)), cycles);
    return busy_until_;
  }

  /// @brief The tick this component's serialised work completes.
  /// @returns Completion tick of the last reserve(), or 0 if never reserved or
  ///          if the engine has been started since.
  Tick busy_until() const { return busy_until_; }

  /// @brief Resume clocking from the next clock edge at or after the given tick.
  ///
  /// @details Exactly wake_at(), kept because it is the name this mixin has
  /// always had for the operation. It no longer refuses while the component is
  /// running: wakes collapse, so a second ask cannot double-queue, and
  /// refusing would silently drop an ask for an *earlier* edge than the one
  /// already armed.
  /// @param after Earliest tick from which to resume clocking.
  void resume_clock(Tick after) { wake_at(after); }

  /// @brief Return whether the clock is currently running.
  ///
  /// @details Asked of the engine rather than mirrored in a member: the clock
  /// is live exactly when a wake is still queued for it in this create()
  /// generation. A cached flag would go stale wherever the engine changes the
  /// answer without going through this class -- across shutdown(), which
  /// destroys the queue entry, and across an advance() that throws.
  /// @retval true Clock is active and scheduling events.
  /// @retval false Clock is stopped.
  bool running() const { return this->wake_pending(clock_event_); }

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
  /// @brief Arm the clock event for @p edge, which must already be an edge.
  ///
  /// @details Every path that schedules this component goes through here, so
  /// the component holds at most one live queue entry whether it is clocking
  /// every edge or waking on demand. TICK_MAX is the domain's "no edge left"
  /// answer and the queue's empty sentinel, so it is never armed.
  void arm_edge(Tick edge) {
    if (edge == TICK_MAX)
      return;
    this->schedule_wake(&clock_event_, edge);
  }

  /// @brief The earliest edge wake_at() may arm: the partition's current tick,
  ///        or strictly past it while inside advance(), whose edge is served.
  /// @returns Floor tick for the next wake.
  Tick wake_floor() const {
    const Tick now = this->current_tick();
    return advancing_ ? domain_.edge_after(now) : now;
  }

  const ClockDomain &domain_; ///< Clock source for period/phase.
  /// @brief Reusable clock edge event.
  ///
  /// @details The engine consumes the wake before this runs, so running() is
  /// already false for the duration of advance() and an advance() that throws
  /// leaves the component stopped and restartable rather than reporting a
  /// clock it does not have.
  ///
  /// The automatic re-arm is skipped when advance() armed a wake itself.
  /// edge_after(now) is the earliest edge there is, so it would supersede any
  /// later visit advance() asked for and destroy the request outright -- a
  /// component that says "clock me, and specifically at 9000" would be clocked
  /// at the next edge and never at 9000.
  Event clock_event_{this, EventType::TIMER_CALLBACK, [this](Tick now, Message *) {
                       advancing_ = true;
                       bool keep_clocking = false;
                       try {
                         keep_clocking = advance(now);
                       } catch (...) {
                         advancing_ = false;
                         throw;
                       }
                       advancing_ = false;
                       if (keep_clocking && !this->wake_pending(clock_event_))
                         arm_edge(domain_.edge_after(now));
                     }};
  bool advancing_ = false; ///< True while this component is inside advance().
  Tick busy_until_ = 0;    ///< Tick this component's serialised work completes.
};

} // namespace simdojo

#endif // SIMDOJO_SIM_CLOCKED_H_
