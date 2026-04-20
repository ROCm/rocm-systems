// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file event_queue.h
/// @brief Event descriptors, priority queues, and cross-partition queues for the simulation engine.

#ifndef SIMDOJO_SIM_EVENT_QUEUE_H_
#define SIMDOJO_SIM_EVENT_QUEUE_H_

#include "simdojo/sim/message.h"
#include "simdojo/sim/sim_types.h"

#include "util/spinlock.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace simdojo {

/// @brief Type of simulation event.
///
/// @details Numeric order determines priority at the same tick: smaller value = higher
/// priority. SIM_EXIT is last so that all real events at max_ticks fire first.
enum class EventType : uint8_t {
  TIMER_CALLBACK,  ///< A scheduled timer/tick callback.
  MESSAGE_ARRIVAL, ///< A message arrived at a port.
  SIM_EXIT,        ///< Simulation exit sentinel (lowest priority).
};

class Component;

/// @brief Callback type for event handling.
using EventHandler = std::function<void(Tick, Message *)>;

/// @brief A reusable simulation event descriptor.
///
/// @details Events are long-lived objects owned by Ports, Components,
/// or the SimulationEngine. They hold a target component, event type, and
/// handler callback. Per-firing state (timestamp, message payload) is stored
/// in EventQueueEntry, not in the Event itself. The same Event object can
/// appear in the event queue multiple times at different timestamps.
class Event {
public:
  /// @brief Construct an event.
  /// @param target Component that will process this event.
  /// @param type Category of event (determines processing priority).
  /// @param handler Optional callback invoked when the event executes.
  Event(Component *target, EventType type, EventHandler handler = nullptr)
      : target_(target), type_(type), handler_(std::move(handler)) {}
  virtual ~Event() = default;

  /// @brief Execute the event's handler with the given firing context.
  /// @param timestamp The simulation tick at which this firing occurs.
  /// @param message Optional message payload for this firing.
  virtual void execute(Tick timestamp, Message *message) {
    assert(handler_ && "execute() called on event with no handler");
    handler_(timestamp, message);
  }

  /// @brief Set the event handler callback.
  /// @param h New handler to assign.
  void set_handler(EventHandler h) { handler_ = std::move(h); }

  /// @brief Check whether a handler is registered.
  /// @retval true A handler is assigned.
  /// @retval false No handler is assigned.
  bool has_handler() const { return handler_ != nullptr; }

  /// @brief Return the target component.
  /// @returns Pointer to the component that processes this event.
  Component *target() const { return target_; }

  /// @brief Return the event type.
  /// @returns The EventType category.
  EventType type() const { return type_; }

private:
  Component *const target_; ///< Component that processes this event.
  EventType type_;          ///< Event category / priority class.
  EventHandler handler_;    ///< Callback invoked on execute().
};

/// @brief A single entry in the event priority queue.
///
/// @details Captures the per-firing state: timestamp, message payload,
/// and a pointer to the reusable Event descriptor.
class EventQueueEntry {
public:
  Tick timestamp = 0;               ///< Simulation tick when this entry fires.
  uint64_t sequence = 0;            ///< Tie-breaking sequence number.
  Event *event = nullptr;           ///< Reusable event descriptor.
  std::unique_ptr<Message> message; ///< Optional message payload for this firing.

  /// @brief Greater-than for min-heap ordering: smallest timestamp first,
  /// then event type priority, then sequence number.
  bool operator>(const EventQueueEntry &other) const {
    if (timestamp != other.timestamp)
      return timestamp > other.timestamp;
    if (event->type() != other.event->type())
      return event->type() > other.event->type();
    return sequence > other.sequence;
  }
};

/// @brief Per-partition priority queue of simulation events.
///
/// @details Each partition (and therefore each worker thread) has exactly one
/// EventQueue. Entries are ordered by timestamp via a min-heap backed by
/// a flat vector. The queue stores EventQueueEntry values; Events are externally
/// owned and reusable across multiple firings.
class EventQueue {
public:
  EventQueue() = default;

  /// @brief Enqueue a heap entry. Assigns a sequence number for tie-breaking.
  /// @param entry The entry to enqueue (ownership of message transferred).
  void push(EventQueueEntry entry) {
    entry.sequence = next_sequence_++;
    entries_.push_back(std::move(entry));
    std::push_heap(entries_.begin(), entries_.end(), std::greater<>{});
  }

  /// @brief Dequeue and return the earliest entry.
  /// @returns The EventQueueEntry with the smallest timestamp.
  EventQueueEntry pop() {
    assert(!entries_.empty());
    std::pop_heap(entries_.begin(), entries_.end(), std::greater<>{});
    auto entry = std::move(entries_.back());
    entries_.pop_back();
    return entry;
  }

  /// @brief Peek at the earliest entry's timestamp without removing it.
  /// @returns Timestamp of the next entry, or TICK_MAX if empty.
  Tick next_event_time() const { return entries_.empty() ? TICK_MAX : entries_.front().timestamp; }

  /// @brief Check whether the queue is empty.
  /// @retval true No entries are enqueued.
  /// @retval false At least one entry is enqueued.
  bool empty() const { return entries_.empty(); }

  /// @brief Return the number of enqueued entries.
  /// @returns Current queue size.
  size_t size() const { return entries_.size(); }

  /// @brief Return the last processed tick (set by the engine).
  /// @returns The current tick value.
  Tick current_tick() const { return current_tick_; }

  /// @brief Update the current tick (called by the engine after processing).
  /// @param t New current tick value.
  void set_current_tick(Tick t) { current_tick_ = t; }

private:
  std::vector<EventQueueEntry> entries_; ///< Min-heap of entries by timestamp.
  uint64_t next_sequence_ = 0;           ///< Monotonic counter for deterministic tie-breaking.
  Tick current_tick_ = 0;                ///< Last processed simulation tick.
};

/// @brief Concurrent queue for cross-partition event delivery.
///
/// @details Each destination partition has one CrossPartitionQueue per source
/// partition. Push and drain can be concurrent, so a spinlock protects the
/// shared state. Drain uses a swap pattern to minimize time under the lock.
class CrossPartitionQueue {
public:
  CrossPartitionQueue() = default;

  // Non-copyable, non-movable.
  CrossPartitionQueue(const CrossPartitionQueue &) = delete;
  CrossPartitionQueue &operator=(const CrossPartitionQueue &) = delete;

  /// @brief Push an entry into the queue (thread-safe).
  /// @param entry The entry to push (ownership of message transferred).
  void push(EventQueueEntry entry) {
    std::lock_guard<util::Spinlock> guard(lock_);
    entries_.push_back(std::move(entry));
  }

  /// @brief Drain all entries into a local EventQueue.
  /// @param local_queue The partition's thread-local EventQueue.
  /// @returns Number of entries drained.
  size_t drain_into(EventQueue &local_queue) {
    {
      std::lock_guard<util::Spinlock> guard(lock_);
      drain_buf_.swap(entries_);
    }
    for (auto &e : drain_buf_)
      local_queue.push(std::move(e));
    size_t count = drain_buf_.size();
    drain_buf_.clear(); // Clear but keep capacity for reuse.
    return count;
  }

  /// @brief Check whether the queue is empty (thread-safe).
  /// @retval true No entries are pending.
  /// @retval false At least one entry is pending.
  bool empty() const {
    std::lock_guard<util::Spinlock> guard(lock_);
    return entries_.empty();
  }

  /// @brief Return the number of pending entries (thread-safe).
  /// @returns Current queue size.
  size_t size() const {
    std::lock_guard<util::Spinlock> guard(lock_);
    return entries_.size();
  }

private:
  mutable util::Spinlock lock_;          ///< Protects entries_.
  std::vector<EventQueueEntry> entries_; ///< Buffered entries awaiting drain.
  std::vector<EventQueueEntry>
      drain_buf_; ///< Pre-allocated buffer for drain_into() (avoids per-call allocation).
};

} // namespace simdojo

#endif // SIMDOJO_SIM_EVENT_QUEUE_H_
