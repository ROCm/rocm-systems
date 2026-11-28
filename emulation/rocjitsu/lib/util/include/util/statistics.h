// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/diagnostic.h"
#include "util/result.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace stats {

struct CounterSnapshot {
  uint64_t value = 0;
  bool overflowed = false;
};

/// @brief A monotonic counter with saturating, explicitly reported overflow.
/// @details Statistics primitives require one owner or external synchronization,
/// including for snapshots. They do not create a publication or sampling barrier.
class Counter {
public:
  constexpr void increment() noexcept { add(1); }
  constexpr void add(uint64_t amount) noexcept {
    if (amount > std::numeric_limits<uint64_t>::max() - value_) {
      value_ = std::numeric_limits<uint64_t>::max();
      overflowed_ = true;
      return;
    }
    value_ += amount;
  }

  constexpr void merge(const Counter &other) noexcept {
    add(other.value_);
    overflowed_ |= other.overflowed_;
  }

  [[nodiscard]] constexpr uint64_t value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool overflowed() const noexcept { return overflowed_; }
  [[nodiscard]] constexpr CounterSnapshot snapshot() const noexcept {
    return {.value = value_, .overflowed = overflowed_};
  }

private:
  uint64_t value_ = 0;
  bool overflowed_ = false;
};

struct GaugeSnapshot {
  int64_t value = 0;
  bool available = false;
};

/// @brief An explicitly initialized current value with no implicit merge policy.
class Gauge {
public:
  constexpr void set(int64_t value) noexcept {
    value_ = value;
    available_ = true;
  }

  [[nodiscard]] constexpr int64_t value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool available() const noexcept { return available_; }
  [[nodiscard]] constexpr GaugeSnapshot snapshot() const noexcept {
    return {.value = value_, .available = available_};
  }

private:
  int64_t value_ = 0;
  bool available_ = false;
};

struct SummarySnapshot {
  uint64_t count = 0;
  double sum = 0;
  double minimum = 0;
  double maximum = 0;
  double mean = 0;
  double variance = 0;
  bool count_overflowed = false;
  bool invalid = false;
};

/// @brief Mergeable sample moments using Welford's online population variance.
/// @details An empty summary has count zero; its numeric fields are placeholders.
/// Nonfinite input is rejected. Overflow or rejected input makes invalid sticky;
/// consumers must then treat its numeric aggregates as unavailable. Count retains
/// the number of finite observations, saturating on overflow and setting
/// count_overflowed independently of invalid floating-point aggregates. Updates
/// allocate no storage, and snapshots never expose a nonfinite floating-point value.
class Summary {
public:
  void record(double value) noexcept {
    if (!std::isfinite(value)) {
      invalid_ = true;
      return;
    }
    if (count_.value() == std::numeric_limits<uint64_t>::max()) {
      count_.increment();
      invalid_ = true;
      return;
    }
    count_.increment();
    if (count_.value() == 1) {
      sum_ = minimum_ = maximum_ = mean_ = value;
      return;
    }
    minimum_ = std::min(minimum_, value);
    maximum_ = std::max(maximum_, value);
    const double delta = value - mean_;
    const double mean = mean_ + delta / static_cast<double>(count_.value());
    const double squared_deviations = squared_deviations_ + delta * (value - mean);
    const double sum = sum_ + value;
    if (!std::isfinite(sum) || !std::isfinite(mean) || !std::isfinite(squared_deviations)) {
      invalid_ = true;
      return;
    }
    sum_ = sum;
    mean_ = mean;
    squared_deviations_ = squared_deviations;
  }

  void merge(const Summary &other) noexcept {
    const bool invalid = invalid_ || other.invalid_;
    if (other.count_.value() == 0) {
      invalid_ = invalid;
      return;
    }
    if (count_.value() == 0) {
      *this = other;
      invalid_ = invalid;
      return;
    }
    const uint64_t left_count = count_.value();
    const uint64_t right_count = other.count_.value();
    const double delta = other.mean_ - mean_;
    const double total_count = static_cast<double>(left_count) + static_cast<double>(right_count);
    const double mean = mean_ + delta * (static_cast<double>(right_count) / total_count);
    // Weight one factor before multiplying to avoid an unnecessary overflow of delta squared.
    const double squared_deviations =
        squared_deviations_ + other.squared_deviations_ +
        delta * (delta * (static_cast<double>(left_count) / total_count)) *
            static_cast<double>(right_count);
    const double sum = sum_ + other.sum_;
    const double minimum = std::min(minimum_, other.minimum_);
    const double maximum = std::max(maximum_, other.maximum_);
    count_.merge(other.count_);
    invalid_ = invalid || count_.overflowed();
    minimum_ = minimum;
    maximum_ = maximum;
    if (!std::isfinite(sum) || !std::isfinite(mean) || !std::isfinite(squared_deviations)) {
      invalid_ = true;
      return;
    }
    sum_ = sum;
    mean_ = mean;
    squared_deviations_ = squared_deviations;
  }

  [[nodiscard]] uint64_t count() const noexcept { return count_.value(); }
  [[nodiscard]] double sum() const noexcept { return sum_; }
  [[nodiscard]] bool invalid() const noexcept { return invalid_; }
  [[nodiscard]] SummarySnapshot snapshot() const noexcept {
    return {.count = count_.value(),
            .sum = sum_,
            .minimum = minimum_,
            .maximum = maximum_,
            .mean = mean_,
            .variance =
                count_.value() == 0
                    ? 0
                    : std::max(0.0, squared_deviations_ / static_cast<double>(count_.value())),
            .count_overflowed = count_.overflowed(),
            .invalid = invalid_};
  }

private:
  Counter count_;
  double sum_ = 0;
  double minimum_ = 0;
  double maximum_ = 0;
  double mean_ = 0;
  double squared_deviations_ = 0;
  bool invalid_ = false;
};

struct HistogramSnapshot {
  std::vector<double> upper_bounds;
  std::vector<uint64_t> counts;
  SummarySnapshot summary;
  bool overflowed = false;
};

/// @brief A fixed-layout histogram with inclusive upper bounds and an unbounded final bucket.
/// @details Each bucket counts samples in its own interval over the accumulator
/// lifetime. The first bucket includes all values up to its bound; each later
/// bucket excludes the previous bound. The final, unbounded bucket counts values
/// above the last bound. Snapshot overflowed means at least one bucket counter
/// exceeded UINT64_MAX and saturated. Setup, rejected-merge diagnostics, and owned
/// snapshots can allocate; recording and successful merges cannot.
class Histogram {
public:
  static util::FailureOr<Histogram> create(std::span<const double> upper_bounds,
                                           util::DiagnosticEmitter emit_error = {}) {
    for (size_t index = 0; index < upper_bounds.size(); ++index) {
      if (!std::isfinite(upper_bounds[index]) ||
          (index != 0 && upper_bounds[index - 1] >= upper_bounds[index]))
        return emit_error.emit() << "histogram bounds must be finite and strictly increasing";
    }
    return Histogram(upper_bounds);
  }

  void record(double value) noexcept {
    summary_.record(value);
    if (!std::isfinite(value))
      return;
    const size_t bucket =
        static_cast<size_t>(std::lower_bound(upper_bounds_.begin(), upper_bounds_.end(), value) -
                            upper_bounds_.begin());
    counts_[bucket].increment();
  }

  util::Result merge(const Histogram &other, util::DiagnosticEmitter emit_error = {}) {
    if (upper_bounds_ != other.upper_bounds_)
      return emit_error.emit() << "cannot merge histograms with different bounds";
    for (size_t bucket = 0; bucket < counts_.size(); ++bucket)
      counts_[bucket].merge(other.counts_[bucket]);
    summary_.merge(other.summary_);
    return util::Result::success();
  }

  [[nodiscard]] HistogramSnapshot snapshot() const {
    HistogramSnapshot result;
    result.upper_bounds = upper_bounds_;
    result.counts.reserve(counts_.size());
    for (const Counter &counter : counts_) {
      result.counts.push_back(counter.value());
      result.overflowed |= counter.overflowed();
    }
    result.summary = summary_.snapshot();
    return result;
  }

private:
  explicit Histogram(std::span<const double> upper_bounds)
      : upper_bounds_(upper_bounds.begin(), upper_bounds.end()), counts_(upper_bounds.size() + 1) {}

  std::vector<double> upper_bounds_;
  std::vector<Counter> counts_;
  Summary summary_;
};

} // namespace stats
