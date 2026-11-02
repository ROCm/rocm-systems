// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pipe_latch.h
/// @brief Pipeline latch between the stages of a clocked component.

#pragma once

#include <cassert>
#include <optional>
#include <utility>

namespace simdojo {

/// @brief A pipeline register between two stages of a clocked component.
///
/// @details The producer stage write()s during a cycle; the owning component
/// calls clock() on the clock edge, after every stage has run; the consumer
/// stage take()s the value in the next cycle.
///
/// @code
///   producer -- write() --> [ next ] -- clock() --> [ current ] -- take() --> consumer
/// @endcode
///
/// Nothing written in a cycle is visible before the clock edge, so the stages
/// of one cycle may run in any order. clock() replaces whatever the consumer
/// did not take.
///
/// @tparam T Value type. It must be movable.
template <typename T> class PipeLatch {
public:
  /// @brief Write the value the latch captures on the next clock edge.
  void write(T value) { next_ = std::move(value); }

  /// @brief Whether a value latched on the last clock edge is waiting.
  bool valid() const { return current_.has_value(); }

  /// @brief Consume and return the latched value.
  /// @pre valid().
  T take() {
    assert(valid() && "pipe latch has no value for the consumer");
    T value = std::move(*current_);
    current_.reset();
    return value;
  }

  /// @brief The clock edge: latch the value written this cycle.
  void clock() {
    current_ = std::move(next_);
    next_.reset();
  }

  /// @brief Whether no value is written or latched.
  bool empty() const { return !current_ && !next_; }

private:
  std::optional<T> next_;
  std::optional<T> current_;
};

} // namespace simdojo
