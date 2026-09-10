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
#include <cassert>
#include <cstdint>
#include <string>
#include <utility>

namespace simdojo {

/// @brief Where a clocked component is in its clock's lifecycle.
///
/// @details Owned by the component, not the engine: the engine only ever sees
/// one ordinary queued event, and whether that event is there, and why, is
/// this state.
enum class ClockState : uint8_t {
  /// No clock event is queued. The component costs nothing until something
  /// wakes it.
  Quiesced,
  /// A clock event is queued, and advance() runs on it.
  Running,
  /// As Running, but drain() has been asked for: the next time advance()
  /// reports it has nothing left to do, the component quiesces and drained()
  /// is called.
  Draining,
};

/// @brief CRTP mixin for components that operate on a clock.
///
/// @details Derives clock period and phase from a ClockDomain. Subclasses
/// override advance(), which is called on a rising edge. Owns one reusable
/// Event, and keeps at most one entry for it queued.
///
/// A component either clocks every edge from startup, or clocks on demand:
/// it overrides clock_at_startup() to false, quiesces when it runs out of
/// work, and is woken again by a peer (wake()) or by its own knowledge of
/// when it is next due (wake_at()). The quiesce, drain and wake logic lives
/// here and in the subclass hooks; the engine sees only schedule and
/// deschedule of an ordinary event.
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
      arm(domain_.next_edge(this->current_tick()));
  }

  /// @brief Forget the clock event so a later startup() can re-arm it.
  ///
  /// @details SimulationEngine::create() also rebuilds after shutdown(), and
  /// shutdown() throws the queued event away with the queue. Without this the
  /// component would still believe it was armed, and would never be armed
  /// again.
  void shutdown() override {
    state_ = ClockState::Quiesced;
    armed_edge_ = TICK_MAX;
    Base::shutdown();
  }

  /// @brief Whether this component should begin clocking at startup.
  ///
  /// @details True for a component that genuinely runs every edge. A component
  /// that advances on demand overrides this to false and is started by wake()
  /// or wake_at() instead. At the one-picosecond tick resolution a 2.1 GHz
  /// domain is a 476-tick period, so a model with hundreds of mostly idle
  /// components cannot afford an event each per edge.
  /// @retval true Clock starts at the domain's first edge.
  /// @retval false Nothing is scheduled until the component is woken.
  virtual bool clock_at_startup() const { return true; }

  /// @brief Wake this component on its next edge.
  ///
  /// @details What a peer or subcomponent calls when it hands this component
  /// work. A quiesced component starts clocking; a running one is unaffected,
  /// since it is already due no later than its next edge.
  void wake() { wake_at(this->current_tick()); }

  /// @brief Ask to advance at @p tick, rounded up to this domain's grid.
  ///
  /// @details The earliest outstanding ask wins, so the component keeps at
  /// most one queued entry however much work is in flight: asking for a later
  /// edge than the one armed is ignored, and asking for an earlier one moves
  /// the armed entry. Works from inside advance(), which is how an on-demand
  /// component schedules its own next visit.
  ///
  /// The result is always strictly after the current tick. A tick in the past
  /// is served on the next edge rather than on an off-grid "now", and a
  /// component inside advance() asking for the edge it is standing on gets the
  /// following one instead of re-entering itself at the same tick. A tick
  /// beyond the domain's last representable edge is ignored.
  /// @param tick Earliest tick to advance at.
  void wake_at(Tick tick) {
    // next_edge() is strictly after its argument, so step from the tick before
    // the one asked for: an ask for an edge lands on that edge.
    const Tick now = this->current_tick();
    arm(domain_.next_edge(tick > now ? tick - 1 : now));
  }

  /// @brief Quiesce once the work in hand is finished.
  ///
  /// @details The component keeps clocking until advance() next reports it has
  /// nothing left to do, then quiesces and drained() is called. What counts as
  /// finished is the subclass's decision, made in advance().
  /// @retval true Already quiesced; drained() is not called.
  /// @retval false Draining; drained() will be called when it quiesces.
  bool drain() {
    if (state_ == ClockState::Quiesced)
      return true;
    state_ = ClockState::Draining;
    return false;
  }

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
    // The start is the first edge at or after the later of the two, hence the
    // step back: next_edge() is strictly after its argument.
    const Tick from = std::max(ready, busy_until_);
    busy_until_ = domain_.deadline(domain_.next_edge(from > 0 ? from - 1 : 0), cycles);
    return busy_until_;
  }

  /// @brief The tick this component's serialised work completes.
  /// @returns Completion tick of the last reserve(), or 0 if never reserved or
  ///          if the engine has been started since.
  Tick busy_until() const { return busy_until_; }

  /// @brief Resume clocking from the first clock edge after the given tick.
  ///
  /// @details wake_at(after + 1), kept because it is the name this mixin has
  /// always had for the operation. No-op if the domain has no edge after
  /// @p after.
  /// @param after Tick after which to resume clocking.
  void resume_clock(Tick after) { wake_at(saturating_add_ticks(after, 1)); }

  /// @brief Return where this component is in its clock's lifecycle.
  /// @returns The current clock state.
  ClockState clock_state() const { return state_; }

  /// @brief Return whether the clock is currently running.
  /// @retval true A clock event is queued for this component.
  /// @retval false The component is quiesced.
  bool running() const { return state_ != ClockState::Quiesced; }

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
  ///
  /// @details Runs inside an event handler, so it must not throw
  /// (docs/style.md); the clock's bookkeeping is not exception-safe.
  /// @param now The simulation tick of this clock edge.
  /// @retval true Continue clocking: advance again on the next edge, unless
  ///         advance() has already asked for a specific one with wake_at().
  /// @retval false Nothing left to do: quiesce until woken, unless advance()
  ///         has already asked for a visit with wake_at().
  virtual bool advance(Tick now) = 0;

protected:
  /// @brief Called when this component quiesces after drain().
  /// @param now The tick of the edge on which it quiesced.
  virtual void drained(Tick now) { (void)now; }

private:
  /// @brief Arm the clock event for @p edge, unless it is already armed no
  ///        later.
  ///
  /// @details Every path that schedules this component goes through here, so
  /// it holds at most one queued entry. An earlier edge takes the armed entry
  /// back and re-queues it; that is the one place a component deschedules.
  /// TICK_MAX is the domain's "no edge left" answer and the queue's empty
  /// sentinel, so it is never armed.
  void arm(Tick edge) {
    if (edge == TICK_MAX || edge >= armed_edge_)
      return;
    if (armed_edge_ != TICK_MAX)
      this->deschedule_event(&clock_event_, armed_edge_);
    armed_edge_ = edge;
    if (state_ == ClockState::Quiesced)
      state_ = ClockState::Running;
    this->schedule_event(&clock_event_, edge);
  }

  const ClockDomain domain_; ///< Clock source for period/phase. Owned; see the constructor.
  /// @brief Reusable clock edge event.
  ///
  /// @details The entry is consumed before advance() runs, so armed_edge_ is
  /// cleared first and a wake_at() from inside advance() arms a fresh entry.
  ///
  /// @p now is always an edge here, so continuing is one period on; the
  /// handler adds it directly rather than paying next_edge()'s modulus on the
  /// busiest path the engine has.
  Event clock_event_{this, EventType::TIMER_CALLBACK, [this](Tick now, Message *) {
                       armed_edge_ = TICK_MAX;
                       const bool keep_clocking = advance(now);
                       // advance() may have armed a visit itself; that ask
                       // stands, and continuing would only supersede it with
                       // an earlier edge.
                       if (keep_clocking && armed_edge_ == TICK_MAX)
                         arm(saturating_add_ticks(now, domain_.period()));
                       if (armed_edge_ == TICK_MAX) {
                         const bool draining = state_ == ClockState::Draining;
                         state_ = ClockState::Quiesced;
                         if (draining)
                           drained(now);
                       }
                     }};
  ClockState state_ = ClockState::Quiesced; ///< Where the clock is in its lifecycle.
  Tick armed_edge_ = TICK_MAX; ///< Tick the queued clock event fires at; TICK_MAX if none.
  Tick busy_until_ = 0;        ///< Tick this component's serialised work completes.
};

} // namespace simdojo

#endif // SIMDOJO_SIM_CLOCKED_H_
