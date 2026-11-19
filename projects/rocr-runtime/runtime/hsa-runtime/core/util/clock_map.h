////////////////////////////////////////////////////////////////////////////////
//
//
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Monotone mapping of one clock onto another from correlated samples.
//
// GPU timestamps are translated to system time from pairs of readings taken
// at the same instant from both clocks (HsaClockCounters). The pairs are
// refreshed while translation is going on, and each new pair disagrees a
// little with the line that was in use before it: read latency moves a pair
// by up to a few microseconds and the clocks drift relative to each other.
// Re-anchoring a single line on every refresh re-maps every tick near the
// refresh, so two GPU timestamps that were in order (the end of one dispatch
// and the start of the next) translate out of order when they are translated
// on opposite sides of a refresh. ClockMap removes that failure mode by
// construction:
//   - monotone: tick_a <= tick_b implies Translate(tick_a) <= Translate(tick_b),
//     no matter how many samples are added between the two calls;
//   - stable: once a tick has been translated, translating it again returns
//     exactly the same value for as long as it is inside the retained history;
//   - continuous, and within the sample read uncertainty of the measured
//     clock relationship at all times.
// The map is a piecewise-linear function with two parts. The history is a
// bounded ring of knots covering every tick already handed out; it is never
// revised. The plan covers everything past the history: a join segment that
// slews from the last knot to the line of the newest sample (the baseline
// clock ratio through the sample), reached one horizon past the sample, and
// that line itself beyond the join. Values translated from the plan advance a
// frontier. When a new sample arrives, the plan up to the frontier becomes
// history and a new plan starts from there, so the disagreement between the
// old plan and the new sample is spread over the horizon instead of being
// absorbed over the few microseconds between the frontier and the sample.

#ifndef HSA_RUNTIME_CORE_UTIL_CLOCK_MAP_H_
#define HSA_RUNTIME_CORE_UTIL_CLOCK_MAP_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>

namespace rocr {
namespace core {

// One correlated reading of both clocks.
struct ClockSample {
  uint64_t tick = 0;  // source clock, e.g. the GPU timestamp counter
  uint64_t sys = 0;   // destination clock, e.g. system time in nanoseconds
};

// Capacity is the number of retained history knots. A sample adds at most two
// knots, so the history covers at least Capacity / 2 refresh periods.
// Not thread safe: callers serialize access.
template <size_t Capacity = 2048> class ClockMap {
  static_assert(Capacity >= 2, "ClockMap needs room for the origin and one sample");

 public:
  ClockMap() { Reset(ClockSample()); }
  explicit ClockMap(const ClockSample& origin) { Reset(origin); }

  // Discards all history and starts a new map at origin.
  void Reset(const ClockSample& origin) {
    origin_ = origin;
    newest_ = origin;
    frontier_ = origin;
    target_ = origin;
    last_attempt_tick_ = origin.tick;
    ratio_ = 0.0;
    historical_ratio_ = 0.0;
    head_ = 0;
    size_ = 0;
    Push(Knot{origin.tick, origin.sys, 0.0});
  }

  // True once a sample has fixed the clock ratio. Until then Translate()
  // returns origin().sys for every tick at or after the origin.
  bool HasBaseline() const { return ratio_ > 0.0; }

  const ClockSample& origin() const { return origin_; }

  // The newest accepted sample (the origin until the first sample is accepted).
  const ClockSample& newest() const { return newest_; }

  // Destination units per source tick, measured from the origin to the newest
  // sample. Zero until HasBaseline().
  double ratio() const { return ratio_; }

  // Whether tick lies far enough past the newest sample that a fresh sample
  // should be taken before translating it: further than a quarter of the
  // baseline, which bounds the error contributed by the read uncertainty of
  // the correlated pair, or further than max_extrapolation destination units,
  // which bounds the error contributed by relative frequency drift. A rejected
  // sample counts as an attempt, so a retry is not requested on every
  // translation that follows it.
  bool NeedsSample(uint64_t tick, uint64_t max_extrapolation) const {
    if (!HasBaseline()) return true;
    const uint64_t anchor = std::max(newest_.tick, last_attempt_tick_);
    if (tick <= anchor) return false;
    const uint64_t ahead = tick - anchor;
    if (ahead > ((newest_.tick - origin_.tick) >> 2)) return true;
    return !(ratio_ * static_cast<double>(ahead) < static_cast<double>(max_extrapolation));
  }

  // Incorporates a new correlated sample: the plan becomes a join from the
  // frontier to the line of the baseline ratio through the sample, reached
  // `horizon` destination units past the sample, followed by that line. An
  // error of a few microseconds spread over a horizon of tens of milliseconds
  // distorts the durations of intervals under the join by well under 0.1%.
  // A sample whose line is not ahead of the frontier even one horizon out
  // cannot be joined without revising values already handed out; it is
  // rejected and the map is left unchanged.
  bool AddSample(const ClockSample& sample, uint64_t horizon) {
    last_attempt_tick_ = std::max(last_attempt_tick_, sample.tick);
    if (sample.tick <= origin_.tick || sample.sys <= origin_.sys) return false;
    const double ratio = Slope(origin_, sample);
    const uint64_t horizon_ticks =
        std::max<uint64_t>(1, static_cast<uint64_t>(static_cast<double>(horizon) / ratio));
    const ClockSample target{sample.tick + horizon_ticks,
                             Project(sample.sys, ratio, horizon_ticks)};
    if (target.tick <= frontier_.tick || target.sys <= frontier_.sys) return false;

    // Everything handed out from the old plan becomes history, keeping the
    // slopes it was handed out with.
    if (HasBaseline()) {
      if (frontier_.tick > target_.tick) Push(Knot{target_.tick, target_.sys, ratio_});
      if (frontier_.tick > back().tick) Push(Knot{frontier_.tick, frontier_.sys, 0.0});
    }
    Knot& last = back();
    last.slope = Slope(ClockSample{last.tick, last.sys}, target);
    target_ = target;
    ratio_ = ratio;
    newest_ = sample;
    frontier_ = ClockSample{last.tick, last.sys};
    return true;
  }

  // Translates tick. With commit, a value taken from the plan advances the
  // frontier so that a later sample cannot revise it. Pass false for a
  // speculative query whose tick may not be a real timestamp.
  uint64_t Translate(uint64_t tick, bool commit = true) {
    if (tick < origin_.tick) return TranslateBeforeOrigin(tick);
    if (!HasBaseline()) return origin_.sys;

    const Knot& last = back();
    if (tick >= last.tick) {
      const uint64_t sys = tick < target_.tick
          ? std::min(Project(last.sys, last.slope, tick - last.tick), target_.sys)
          : Project(target_.sys, ratio_, tick - target_.tick);
      if (commit && tick > frontier_.tick) frontier_ = ClockSample{tick, sys};
      return sys;
    }

    const Knot& first = front();
    if (tick < first.tick) {
      // Older than the retained history: extrapolate backwards from the oldest
      // knot with the baseline ratio.
      return ProjectBack(first.sys, ratio_, first.tick - tick);
    }

    // Find the knot that starts the history segment containing tick:
    // at(lo).tick <= tick < at(hi).tick.
    size_t lo = 0;
    size_t hi = size_ - 1;
    while (hi - lo > 1) {
      const size_t mid = lo + (hi - lo) / 2;
      if (at(mid).tick <= tick)
        lo = mid;
      else
        hi = mid;
    }
    const Knot& knot = at(lo);
    // The clamp keeps a segment below its next knot under floating point
    // rounding; the next knot always lies on or above the segment.
    return std::min(Project(knot.sys, knot.slope, tick - knot.tick), at(hi).sys);
  }

 private:
  // Knot i defines the segment [tick, next.tick) as sys + slope * (t - tick).
  // The last knot's segment is the join of the plan and ends at target_.
  struct Knot {
    uint64_t tick;
    uint64_t sys;
    double slope;
  };

  static double Slope(const ClockSample& from, const ClockSample& to) {
    return static_cast<double>(to.sys - from.sys) / static_cast<double>(to.tick - from.tick);
  }

  // slope * delta truncated to an integer, saturating instead of overflowing.
  // Truncation keeps the projection monotone in delta.
  static uint64_t Scale(double slope, uint64_t delta) {
    const double scaled = slope * static_cast<double>(delta);
    if (!(scaled < 18446744073709551616.0)) return UINT64_MAX;
    return static_cast<uint64_t>(scaled);
  }

  static uint64_t Project(uint64_t sys, double slope, uint64_t delta) {
    const uint64_t offset = Scale(slope, delta);
    return offset < UINT64_MAX - sys ? sys + offset : UINT64_MAX;
  }

  static uint64_t ProjectBack(uint64_t sys, double slope, uint64_t delta) {
    const uint64_t offset = Scale(slope, delta);
    return offset < sys ? sys - offset : 0;
  }

  // Ticks that predate the origin, e.g. a signal written before the runtime
  // started, are mapped with the first ratio measured and that ratio is then
  // fixed, so frequency adjustments of the system clock made after startup are
  // not amplified across the long extrapolation.
  uint64_t TranslateBeforeOrigin(uint64_t tick) {
    if (historical_ratio_ == 0.0) historical_ratio_ = ratio_;
    return ProjectBack(origin_.sys, historical_ratio_, origin_.tick - tick);
  }

  Knot& at(size_t i) { return knots_[(head_ + i) % Capacity]; }
  const Knot& at(size_t i) const { return knots_[(head_ + i) % Capacity]; }
  Knot& front() { return at(0); }
  const Knot& front() const { return at(0); }
  Knot& back() { return at(size_ - 1); }
  const Knot& back() const { return at(size_ - 1); }

  void Push(const Knot& knot) {
    if (size_ == Capacity) {
      head_ = (head_ + 1) % Capacity;
      --size_;
    }
    at(size_) = knot;
    ++size_;
  }

  Knot knots_[Capacity];
  size_t head_ = 0;
  size_t size_ = 0;
  ClockSample origin_;
  ClockSample newest_;
  // Where the plan's join reaches the newest sample's line.
  ClockSample target_;
  // Furthest tick handed out from the plan, with its value.
  ClockSample frontier_;
  uint64_t last_attempt_tick_ = 0;
  double ratio_ = 0.0;
  double historical_ratio_ = 0.0;
};

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_UTIL_CLOCK_MAP_H_
