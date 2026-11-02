// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file issue_queue.h
/// @brief In-order queue whose entries leave once their ready cycle arrives.

#pragma once

#include <cassert>
#include <cstdint>
#include <deque>
#include <utility>

namespace simdojo {

/// @brief Entries that wait, in order, until the cycle each becomes ready.
///
/// @details A producer push()es an entry with the first cycle it may leave;
/// the consumer pop()s the oldest entry once ready() says that cycle has
/// arrived. An entry never overtakes an older one. Cycles count the owning
/// component's clock and never decrease.
///
/// @code
///   producer -- push(value, ready_cycle) --> [ oldest ... youngest ] -- pop() --> consumer
///                                                 ^ ready(cycle)
/// @endcode
///
/// A fixed-latency pipeline pushes each entry with the current cycle plus its
/// latency. An entry pushed with a later ready cycle than the current one is
/// not visible in the current cycle, so the stages of one cycle may run in
/// any order.
///
/// @tparam T Entry type. It must be movable.
template <typename T> class IssueQueue {
public:
  /// @brief Add @p value behind every entry already queued; it may leave from @p ready_cycle.
  void push(T value, uint64_t ready_cycle) { entries_.push_back({std::move(value), ready_cycle}); }

  /// @brief Whether the oldest entry may leave in @p cycle.
  bool ready(uint64_t cycle) const {
    return !entries_.empty() && entries_.front().ready_cycle <= cycle;
  }

  /// @brief Remove and return the oldest entry.
  /// @pre The queue is not empty.
  T pop() {
    assert(!entries_.empty() && "issue queue is empty");
    T value = std::move(entries_.front().value);
    entries_.pop_front();
    return value;
  }

  /// @brief Whether no entry is queued.
  bool empty() const { return entries_.empty(); }

private:
  struct Entry {
    T value;
    uint64_t ready_cycle;
  };
  std::deque<Entry> entries_;
};

} // namespace simdojo
