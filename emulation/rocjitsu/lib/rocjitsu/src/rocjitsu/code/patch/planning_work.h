// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

namespace rocjitsu {

[[nodiscard]] constexpr size_t saturated_add(size_t lhs, size_t rhs) {
  return rhs > std::numeric_limits<size_t>::max() - lhs ? std::numeric_limits<size_t>::max()
                                                        : lhs + rhs;
}

[[nodiscard]] constexpr size_t saturated_multiply(size_t lhs, size_t rhs) {
  return lhs != 0u && rhs > std::numeric_limits<size_t>::max() / lhs
             ? std::numeric_limits<size_t>::max()
             : lhs * rhs;
}

inline void accumulate_saturated(size_t &total, size_t value) {
  total = saturated_add(total, value);
}

/// A typed base-plus-input allowance. Callers choose an input unit that
/// reflects their algorithm (for example coordinates, instruction words, or
/// candidate trials) instead of hiding input scaling in an unlabelled cap.
struct PlanningWorkLimit {
  size_t base = 0u;
  size_t per_input = 0u;

  [[nodiscard]] constexpr size_t for_inputs(size_t input_count) const {
    return saturated_add(base, saturated_multiply(per_input, input_count));
  }
};

inline constexpr PlanningWorkLimit kDefaultSoppRelayPlanningWorkLimit{2'000'000u, 1'024u};
inline constexpr PlanningWorkLimit kDefaultDirectReservoirPlanningWorkLimit{4'096u, 32u};
// A refinement iteration can perform eight ranked single trials, logarithmic
// batch trials, and a tail trial before accepting one reservoir.
inline constexpr PlanningWorkLimit kDefaultLdsRelayLayoutPlanningWorkLimit{16u, 24u};
inline constexpr PlanningWorkLimit kDefaultLdsConvergencePlanningWorkLimit{2u, 1u};

/// A saturating, fail-closed work meter. A zero configured allowance still
/// permits one unit so callers cannot accidentally create a non-runnable
/// planner; consuming beyond the allowance permanently marks it exhausted.
class BoundedPlanningWorkMeter {
public:
  explicit constexpr BoundedPlanningWorkMeter(size_t limit) : limit_(std::max<size_t>(limit, 1u)) {}

  [[nodiscard]] constexpr bool consume(size_t amount = 1u) {
    if (exhausted_)
      return false;
    if (amount > limit_ - consumed_) {
      exhausted_ = true;
      return false;
    }
    consumed_ += amount;
    return true;
  }

  [[nodiscard]] constexpr size_t limit() const { return limit_; }
  [[nodiscard]] constexpr size_t consumed() const { return consumed_; }
  [[nodiscard]] constexpr size_t remaining() const { return exhausted_ ? 0u : limit_ - consumed_; }
  [[nodiscard]] constexpr bool exhausted() const { return exhausted_; }

private:
  size_t limit_ = 1u;
  size_t consumed_ = 0u;
  bool exhausted_ = false;
};

/// A bounded meter connected to one work counter and one exhaustion counter.
/// Each successful charge contributes only the units actually consumed. The
/// first rejected charge records one exhaustion event and latches the meter.
class MeteredPlanningWork {
public:
  explicit MeteredPlanningWork(size_t limit, size_t *work_count = nullptr,
                               size_t *exhaustion_count = nullptr)
      : meter_(limit), work_count_(work_count), exhaustion_count_(exhaustion_count) {}

  [[nodiscard]] bool consume(size_t amount = 1u) {
    const size_t before = meter_.consumed();
    const bool available = meter_.consume(amount);
    if (work_count_ != nullptr)
      accumulate_saturated(*work_count_, meter_.consumed() - before);
    if (!available && !exhaustion_recorded_) {
      if (exhaustion_count_ != nullptr)
        accumulate_saturated(*exhaustion_count_, 1u);
      exhaustion_recorded_ = true;
    }
    return available;
  }

  [[nodiscard]] size_t limit() const { return meter_.limit(); }
  [[nodiscard]] size_t consumed() const { return meter_.consumed(); }
  [[nodiscard]] size_t remaining() const { return meter_.remaining(); }
  [[nodiscard]] bool exhausted() const { return meter_.exhausted(); }

private:
  BoundedPlanningWorkMeter meter_;
  size_t *work_count_ = nullptr;
  size_t *exhaustion_count_ = nullptr;
  bool exhaustion_recorded_ = false;
};

} // namespace rocjitsu
