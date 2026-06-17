// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file message.h
/// @brief Message types and queues for inter-component communication over links.

#ifndef SIMDOJO_SIM_MESSAGE_H_
#define SIMDOJO_SIM_MESSAGE_H_

#include "simdojo/sim/sim_types.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace simdojo {

/// @brief Operation type carried in a message header.
enum class MessageOp : uint8_t {
  NONE,     ///< Non-memory message (dispatch, status, etc.).
  READ,     ///< Memory read request.
  WRITE,    ///< Memory write request.
  RESPONSE, ///< Memory read/write response.
};

/// @brief Common header for all messages sent over links.
struct MessageHeader {
  Tick timestamp =
      TICK_MAX;     ///< Simulation tick when the message was sent (TICK_MAX = not yet stamped).
  Tick latency = 0; ///< Propagation delay set by the link.
  uint32_t size_bytes = 0;        ///< Payload size in bytes (for bandwidth modeling).
  PortID src_port = 0;            ///< Source port identifier.
  PortID dst_port = 0;            ///< Destination port identifier.
  uint64_t sequence_num = 0;      ///< Sender-assigned sequence number.
  uint64_t addr = 0;              ///< Memory address (unused for non-memory ops).
  MessageOp op = MessageOp::NONE; ///< Operation type.
  uint8_t mtype = 0;              ///< Memory type (Mtype enum, unused for non-memory ops).
};

/// @brief Simulation message sent over links between components.
///
/// @details Carries routing/timing metadata in the header and an optional
/// payload as a `uintptr_t`. The payload can hold either a small integer
/// value or a pointer (via `reinterpret_cast`) to arbitrary data owned
/// elsewhere. This eliminates the need for derived message classes in most
/// cases — components that need richer payloads can store the data in a
/// side structure and pass a pointer via the payload field.
class Message {
public:
  /// @brief Construct a message with default header (unstamped).
  Message() = default;

  /// @brief Construct a message with the given header.
  /// @param hdr The message header (routing and timing metadata).
  explicit Message(MessageHeader hdr) : header_(std::move(hdr)) {}

  /// @brief Construct a message with header and payload.
  /// @param hdr The message header.
  /// @param payload Integer or pointer payload.
  Message(MessageHeader hdr, uintptr_t payload) : header_(std::move(hdr)), payload_(payload) {}

  virtual ~Message() = default;

  /// @brief Return the message header (const).
  const MessageHeader &header() const { return header_; }

  /// @brief Return the message header (mutable, for handlers to update op/addr).
  MessageHeader &header() { return header_; }

  /// @brief Return the payload value.
  uintptr_t payload() const { return payload_; }

  /// @brief Set the payload value.
  void set_payload(uintptr_t val) { payload_ = val; }

  /// @brief Set the source and destination port identifiers.
  void set_ports(PortID src, PortID dst) {
    header_.src_port = src;
    header_.dst_port = dst;
  }

  /// @brief Set the propagation latency (called by Link during send).
  void set_latency(Tick lat) { header_.latency = lat; }

  /// @brief Set the send timestamp (called by Link if not already stamped).
  void set_timestamp(Tick ts) { header_.timestamp = ts; }

  /// @brief Compute the simulation tick at which this message arrives.
  /// @returns timestamp + latency.
  Tick arrival_tick() const { return header_.timestamp + header_.latency; }

  /// @brief Order by arrival tick (for min-heap via std::greater).
  bool operator>(const Message &other) const { return arrival_tick() > other.arrival_tick(); }

protected:
  MessageHeader header_;  ///< Routing and timing metadata.
  uintptr_t payload_ = 0; ///< Optional integer or pointer payload.
};

/// @brief A bounded priority queue of messages ordered by arrival tick
/// (min-heap).
///
/// @details Used by QueuedLink to buffer messages for explicit consumption
/// by the receiving component. Capacity is fixed at construction.
class MessageQueue {
public:
  /// @brief Construct a message queue with the given capacity.
  /// @param capacity Maximum number of messages the queue can hold.
  explicit MessageQueue(size_t capacity) : capacity_(capacity) {}

  /// @brief Try to enqueue a message.
  /// @param msg The message to enqueue (ownership transferred).
  /// @retval true Message was enqueued successfully.
  /// @retval false Queue is full; message was not enqueued.
  bool push(std::unique_ptr<Message> msg) {
    if (entries_.size() >= capacity_)
      return false;
    entries_.push_back(std::move(msg));
    std::push_heap(entries_.begin(), entries_.end(), ptr_greater);
    return true;
  }

  /// @brief Dequeue and return the earliest message.
  /// @returns The message with the smallest arrival tick.
  std::unique_ptr<Message> pop() {
    assert(!entries_.empty());
    std::pop_heap(entries_.begin(), entries_.end(), ptr_greater);
    auto msg = std::move(entries_.back());
    entries_.pop_back();
    return msg;
  }

  /// @brief Peek at the earliest message without removing it.
  /// @returns Pointer to the earliest message, or nullptr if empty.
  const Message *peek() const { return entries_.empty() ? nullptr : entries_.front().get(); }

  /// @brief Check whether the queue is empty.
  /// @retval true No messages are buffered.
  /// @retval false At least one message is buffered.
  bool empty() const { return entries_.empty(); }

  /// @brief Check whether the queue is at capacity.
  /// @retval true Queue has reached its maximum capacity.
  /// @retval false Queue has room for more messages.
  bool full() const { return entries_.size() >= capacity_; }

  /// @brief Return the number of buffered messages.
  /// @returns Current queue size.
  size_t size() const { return entries_.size(); }

  /// @brief Return the maximum number of messages the queue can hold.
  /// @returns Queue capacity.
  size_t capacity() const { return capacity_; }

  /// @brief Return the arrival tick of the earliest message.
  /// @returns Arrival tick, or TICK_MAX if empty.
  Tick next_message_time() const {
    return entries_.empty() ? TICK_MAX : entries_.front()->arrival_tick();
  }

private:
  /// @brief Min-heap comparator: dereferences unique_ptrs and delegates to Message::operator>.
  static constexpr auto ptr_greater = [](const std::unique_ptr<Message> &a,
                                         const std::unique_ptr<Message> &b) { return *a > *b; };

  size_t capacity_;                               ///< Maximum number of buffered messages.
  std::vector<std::unique_ptr<Message>> entries_; ///< Min-heap of messages by arrival tick.
};

} // namespace simdojo

#endif // SIMDOJO_SIM_MESSAGE_H_
