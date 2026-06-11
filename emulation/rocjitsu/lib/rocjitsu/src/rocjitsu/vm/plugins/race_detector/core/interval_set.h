// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <algorithm>
#include <cassert>
#include <vector>

namespace rocjitsu::plugins::race_detector {

/// A half-open byte range [start, end).
struct Interval {
  int start;
  int end;
};

/// A set of non-overlapping, sorted intervals. Built by appending intervals
/// (with opportunistic back-merging), then finalized via sort + merge.
///
/// Typical usage:
///   IntervalSet s;
///   s.reserve(n);
///   for (...) s.append(lo, hi);   // O(1) each, merges with back if adjacent
///   s.finalize();                  // sort + merge for the general case
///
/// After finalize(), intervals are sorted by start and non-overlapping.
class IntervalSet {
public:
  /// Append [start, end). If adjacent/overlapping with the last interval,
  /// merges in O(1). Otherwise pushes a new entry.
  void append(int start, int end) {
    assert(start < end && "Interval must be non-empty");
    if (!intervals.empty() && start <= intervals.back().end && intervals.back().start <= end) {
      intervals.back().start = std::min(intervals.back().start, start);
      intervals.back().end = std::max(intervals.back().end, end);
    } else {
      intervals.push_back({start, end});
    }
  }

  /// Sort by start and merge overlapping/adjacent intervals. Idempotent.
  void finalize() {
    if (intervals.size() <= 1) {
      return;
    }
    std::sort(intervals.begin(), intervals.end(),
              [](const Interval &a, const Interval &b) { return a.start < b.start; });
    size_t out = 0;
    for (size_t i = 1; i < intervals.size(); ++i) {
      if (intervals[i].start <= intervals[out].end) {
        intervals[out].end = std::max(intervals[out].end, intervals[i].end);
      } else {
        intervals[++out] = std::move(intervals[i]);
      }
    }
    intervals.resize(out + 1);
  }

  void reserve(size_t n) { intervals.reserve(n); }

  bool empty() const { return intervals.empty(); }

  /// Number of disjoint intervals.
  size_t size() const { return intervals.size(); }

  /// Total number of bytes covered by all intervals.
  int getTotalBytes() const {
    int total = 0;
    for (const auto &iv : intervals) {
      total += iv.end - iv.start;
    }
    return total;
  }

  /// Check whether a byte address falls within any interval. The early-exit
  /// optimization assumes intervals are sorted (i.e. finalize() was called).
  bool contains(int byte) const {
    for (const auto &iv : intervals) {
      if (byte < iv.start) {
        return false; // sorted: no later interval can contain it
      }
      if (byte < iv.end) {
        return true;
      }
    }
    return false;
  }

  /// Check whether [qStart, qEnd) overlaps any interval. Uses binary search.
  /// Requires intervals to be sorted (i.e. finalize() was called).
  ///
  /// For sorted non-overlapping intervals, iv[i].end < iv[i+1].start (strict
  /// inequality -- equal endpoints would have been merged). So the only
  /// candidate is the last interval with start < qEnd: if its end > qStart,
  /// it overlaps; otherwise no interval does.
  bool overlapsRange(int qStart, int qEnd) const {
    // Find first interval with start >= qEnd.
    auto it = std::lower_bound(intervals.begin(), intervals.end(), qEnd,
                               [](const Interval &iv, int val) { return iv.start < val; });
    // The candidate is the interval just before it.
    if (it == intervals.begin()) {
      return false;
    }
    --it;
    return it->end > qStart;
  }

  auto begin() const { return intervals.begin(); }
  auto end() const { return intervals.end(); }

private:
  std::vector<Interval> intervals;
};

} // namespace rocjitsu::plugins::race_detector
