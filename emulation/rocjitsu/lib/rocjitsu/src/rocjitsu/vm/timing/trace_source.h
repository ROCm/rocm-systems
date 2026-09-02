// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file trace_source.h
/// @brief The timing model's ingress: functional facts become SimDojo messages.

#ifndef ROCJITSU_VM_TIMING_TRACE_SOURCE_H_
#define ROCJITSU_VM_TIMING_TRACE_SOURCE_H_

#include "rocjitsu/vm/timing/trace_event.h"

#include "simdojo/sim/component.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <span>
#include <string>
#include <unordered_set>

namespace rocjitsu::timing {

/// @brief Turns facts observed during functional execution into messages on
///        the timing model's own topology.
///
/// @details The one place the two planes meet. Functional execution hands it
/// facts; it stamps each with a position in the stream and sends it out of an
/// ordinary SimDojo port, so everything downstream is a modelled component
/// receiving a modelled message.
///
/// It has no clock of its own. It schedules nothing, and it does not decide
/// when anything happens: the tick a message departs at is the timing engine's
/// current tick, and the timing engine is advanced by whoever owns it. A
/// source that kept its own idea of time would be a second clock in a design
/// whose whole point is that there is one.
///
/// Sequence numbers are consecutive and gapless from one, so a consumer can
/// tell a dropped event from a reordered one. They are deterministic when
/// functional execution is, which means a single-partition functional engine;
/// with several partitions the facts arrive here in whatever order the threads
/// produce them, and the lock below makes that well-defined rather than
/// reproducible.
///
/// That lock is also what makes sending safe at all. Every submit pushes into
/// the timing engine's event queue, which is owner-thread-only, so submitters
/// are serialised against one another. What the lock cannot do is serialise
/// them against the timing engine being *advanced*: the owner must not run the
/// timing engine while functional execution is still submitting into it.
class TimingTraceSource final : public simdojo::Component {
public:
  explicit TimingTraceSource(std::string name = "timing_trace_source");

  void initialize() override;

  /// @brief The port modelled consumers connect to.
  simdojo::Port *output_port() { return output_; }

  /// @brief Take one live fact, stamp it, and send it.
  ///
  /// @details Rejects a fact whose dispatch has already ended. That is not
  /// defensive: a dispatch's terms are closed when it ends, and an event
  /// arriving afterwards would either be silently dropped by the consumer or
  /// quietly change a number already reported.
  /// It refuses rather than throws when the timing plane cannot take the
  /// event -- no consumer wired up, or the engine not created or already shut
  /// down. This is called from inside functional execution, from a plugin hook
  /// with a lock held, on a path that has no way to handle an exception; a
  /// refusal that shows up in rejected() is the only answer that does not take
  /// the emulator down with it.
  /// @param event The fact; its sequence field is overwritten.
  /// @retval true Accepted, stamped, and sent.
  /// @retval false Its dispatch had already ended, or the timing plane could
  ///         not take it.
  bool submit(TraceEvent event);

  /// @brief Send a recorded stream again, keeping its sequence numbers.
  ///
  /// @details What makes a stream replayable: the same facts in the same
  /// order produce the same messages, so a model can be re-run against a
  /// recording rather than against the emulator.
  /// Only onto a source that has sent nothing. Replaying onto a live one would
  /// reissue sequence numbers it had already used and wind its counter
  /// backwards, which is exactly the gaplessness a consumer relies on to
  /// detect a dropped event.
  /// @param events The recorded stream, in sequence order.
  /// @throws std::logic_error if this source has already sent something.
  /// @throws std::invalid_argument if the stream is not consecutive from one,
  ///         or if it continues a dispatch after that dispatch's end.
  void replay(std::span<const TraceEvent> events);

  /// @brief Forget which dispatches have ended, and restart the stream at one.
  void reset();

  /// @brief Events sent.
  uint64_t accepted() const;
  /// @brief Events refused because their dispatch had already ended.
  uint64_t rejected() const;
  /// @brief Sequence number the next accepted event will carry.
  uint64_t next_sequence() const;

private:
  /// @brief Whether the timing plane can take a message right now.
  ///        Caller holds the lock.
  bool can_send() const;

  /// @brief Send @p event, having decided it is acceptable. Caller holds the lock.
  void send_locked(TraceEvent event);

  /// @brief Note that @p dispatch_id has ended. Caller holds the lock.
  void mark_ended_locked(uint32_t dispatch_id);

  /// @brief How many ended dispatches are remembered.
  ///
  /// @details A window, not a full history. Dispatch ids climb monotonically
  /// within a queue and only wrap after hundreds of millions, so remembering
  /// every one that ended would grow for the life of the run. A late event
  /// arrives just after its dispatch ended, not a thousand dispatches later,
  /// so a window of this size refuses everything the guard exists for and
  /// forgets only what cannot happen.
  static constexpr std::size_t kEndedWindow = 1024;

  simdojo::Port *output_ = nullptr;
  /// @brief Guards the stream state against a functional engine that has more
  /// than one partition. It buys mutual exclusion, not reproducibility; see
  /// the class comment.
  mutable std::mutex mutex_;
  /// @brief Dispatches whose END has been seen, most recent kEndedWindow of
  /// them. An id announced again is removed, because that is a new dispatch
  /// reusing the id rather than a continuation of the one that ended.
  std::unordered_set<uint32_t> ended_;
  /// @brief The order ids entered @ref ended_, so the oldest can be dropped.
  std::deque<uint32_t> ended_order_;
  uint64_t next_sequence_ = 1;
  uint64_t accepted_ = 0;
  uint64_t rejected_ = 0;
};

} // namespace rocjitsu::timing

#endif // ROCJITSU_VM_TIMING_TRACE_SOURCE_H_
