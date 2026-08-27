// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cache_des.h
/// @brief A timed cache level, and the tag array behind it.

#pragma once

#include "rocjitsu/vm/timing/timed_component.h"
#include "rocjitsu/vm/timing/tuning.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

namespace rocjitsu::timing {

/// @brief A set-associative tag array. Tags only; the data lives in the
///        functional simulator and the timing plane never reads it.
class TagArray {
public:
  void configure(std::uint64_t sets, std::uint64_t ways, std::uint64_t line_bytes);
  /// @brief Probe and allocate, replacing least recently used.
  /// @returns true when the line was already resident.
  bool access(std::uint64_t byte_address, std::uint64_t stamp);
  void invalidate();
  std::uint64_t line_bytes() const { return line_bytes_; }
  std::uint64_t line_shift() const { return line_shift_; }

private:
  std::uint64_t sets_ = 0;
  std::uint64_t ways_ = 0;
  std::uint64_t set_mask_ = 0;
  std::uint64_t line_bytes_ = 64;
  std::uint64_t line_shift_ = 6;
  std::vector<std::uint64_t> tags_;
  std::vector<std::uint64_t> stamps_;
  static constexpr std::uint64_t kEmpty = ~0ULL;
};

/// @brief One instance of a cache level: a first-level cache on one compute
///        unit, a second-level cache on one die, or one channel of the
///        memory-side cache.
///
/// @details Serves the lines it holds and forwards the rest to whatever is
/// downstream. The split is per request rather than per line, so one wave's
/// access that half hits produces one completion and one forwarded request
/// rather than sixty-four events.
class CacheDes final : public TimedComponent {
public:
  /// @brief What a level does with the requests it cannot serve.
  using Downstream = std::function<void(const MemoryRequest &)>;
  /// @brief What it does with the requests it can.
  using Completion = std::function<void(const MemoryRequest &, std::uint64_t tick)>;

  CacheDes(std::string name, const simdojo::ClockDomain &domain, TimingEngine &engine,
           const CacheTuning &tuning, std::vector<std::uint64_t> *line_pool);

  void set_downstream(Downstream downstream) { downstream_ = std::move(downstream); }
  void set_completion(Completion completion) { completion_ = std::move(completion); }
  /// @brief Bypass the tag array entirely; used for a level a request marks
  ///        non-temporal, which the hardware documents as not allocating.
  void set_allocates(bool allocates) { allocates_ = allocates; }

  std::uint64_t advance(std::uint64_t now) override;

  void invalidate();

  std::uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
  std::uint64_t misses() const { return misses_.load(std::memory_order_relaxed); }
  std::uint64_t bytes() const { return bytes_.load(std::memory_order_relaxed); }

private:
  const CacheTuning &tuning_;
  std::vector<std::uint64_t> *line_pool_;
  TagArray tags_;
  Downstream downstream_;
  Completion completion_;
  bool allocates_ = true;
  std::uint64_t stamp_ = 1;
  std::atomic<std::uint64_t> hits_{0};
  std::atomic<std::uint64_t> misses_{0};
  std::atomic<std::uint64_t> bytes_{0};
};

} // namespace rocjitsu::timing
