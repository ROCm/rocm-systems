/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Unit tests for the GPU-to-system clock map (core/util/clock_map.h) behind
// GpuAgent::TranslateTime. The map must stay monotone and stable while the
// correlated clock samples it is built from are refreshed, so that GPU
// timestamps which are ordered in the GPU domain never translate out of order
// (the end of one dispatch landing after the start of the next on the same
// queue) because they were translated on opposite sides of a refresh.

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "core/util/clock_map.h"

using rocr::core::ClockMap;
using rocr::core::ClockSample;

namespace {

// 100 MHz GPU counter against a nanosecond system clock, as on current hardware.
constexpr double kNominalRatio = 10.0;
// GpuAgent::TranslateTime bounds extrapolation to 1/16 s of system time and
// blends new samples in over the same horizon.
constexpr uint64_t kHorizonNs = 62'500'000;
constexpr uint64_t kHorizonTicks = 6'250'000;

ClockSample Origin() { return ClockSample{1'000'000, 5'000'000'000ull}; }

// A simulated pair of clocks: the system clock runs at kNominalRatio ns per
// tick plus a small relative drift, and every correlated reading carries the
// read jitter of a clock-counter ioctl.
struct SimulatedClocks {
  explicit SimulatedClocks(uint32_t seed, int64_t jitter_ns = 3'000)
      : rng(seed), jitter(-jitter_ns, jitter_ns) {}
  uint64_t SystemAt(uint64_t tick) const {
    return Origin().sys +
        static_cast<uint64_t>((kNominalRatio + drift) * double(tick - Origin().tick));
  }
  ClockSample ReadAt(uint64_t tick) { return ClockSample{tick, SystemAt(tick) + jitter(rng)}; }
  std::mt19937 rng;
  std::uniform_int_distribution<int64_t> jitter;
  double drift = 1.0e-5;
};

// Exact samples on the nominal line, for tests that reason about exact values.
ClockSample Exact(uint64_t tick) {
  return ClockSample{
      tick, Origin().sys + static_cast<uint64_t>(kNominalRatio * double(tick - Origin().tick))};
}

}  // namespace

// Before the first sample the ratio is unknown: nothing is extrapolated and a
// sample is always requested.
TEST(ClockMapTest, OriginOnlyHasNoBaseline) {
  ClockMap<> map(Origin());
  EXPECT_FALSE(map.HasBaseline());
  EXPECT_EQ(map.ratio(), 0.0);
  EXPECT_TRUE(map.NeedsSample(Origin().tick + 1, kHorizonNs));
  EXPECT_EQ(map.Translate(Origin().tick + 12345), Origin().sys);
  EXPECT_EQ(map.newest().tick, Origin().tick);
}

// Exact samples on one line reproduce that line exactly everywhere.
TEST(ClockMapTest, ExactSamplesReproduceTheLine) {
  ClockMap<> map(Origin());
  for (uint64_t i = 1; i <= 50; ++i) {
    const uint64_t tick = Origin().tick + i * kHorizonTicks;
    (void)map.Translate(tick - 1'000);  // extrapolate a little before every sample
    ASSERT_TRUE(map.AddSample(Exact(tick), kHorizonNs));
    EXPECT_EQ(map.newest().tick, tick);
    EXPECT_DOUBLE_EQ(map.ratio(), kNominalRatio);
  }
  for (uint64_t tick = Origin().tick; tick < Origin().tick + 52 * kHorizonTicks; tick += 7'919)
    ASSERT_EQ(map.Translate(tick), Exact(tick).sys) << "tick " << tick;
}

// The defect: a tick extrapolated before a refresh must translate to the same
// value after it, even though the new sample disagrees with the old line.
TEST(ClockMapTest, TranslationIsStableAcrossRefresh) {
  ClockMap<> map(Origin());
  ASSERT_TRUE(map.AddSample(Exact(Origin().tick + 100'000'000), kHorizonNs));

  const uint64_t boundary = Origin().tick + 112'000'000;  // 120 ms past the newest sample
  const uint64_t before = map.Translate(boundary);
  std::vector<uint64_t> grid_before;
  for (uint64_t t = Origin().tick + 100'000'000; t <= boundary; t += 997)
    grid_before.push_back(map.Translate(t));

  // The next sample lands 9 ns earlier than the old line predicts.
  const uint64_t next_tick = Origin().tick + 112'000'500;
  ClockSample next = Exact(next_tick);
  next.sys -= 9;
  ASSERT_TRUE(map.AddSample(next, kHorizonNs));

  EXPECT_EQ(map.Translate(boundary), before);
  size_t i = 0;
  for (uint64_t t = Origin().tick + 100'000'000; t <= boundary; t += 997, ++i)
    ASSERT_EQ(map.Translate(t), grid_before[i]) << "tick " << t << " was revised by the refresh";
  // From the frontier through the join the map is continuous and monotone, and
  // it is back on the new sample's line one horizon past the sample.
  uint64_t prev = before;
  for (uint64_t t = boundary; t <= next_tick + kHorizonTicks; t += 101) {
    const uint64_t v = map.Translate(t);
    ASSERT_GE(v, prev) << "tick " << t;
    ASSERT_LE(v - prev, 101 * kNominalRatio + 1) << "tick " << t;  // no jump
    prev = v;
  }
  EXPECT_NEAR(double(map.Translate(next_tick + kHorizonTicks)),
              double(next.sys + static_cast<uint64_t>(kNominalRatio * kHorizonTicks)), 1.0);
}

// The ordering guarantee that matters to profilers: the end of a dispatch and
// the start of its successor are the same GPU tick, translated at different
// times across refreshes, and must never invert.
TEST(ClockMapTest, SharedBoundaryNeverInverts) {
  SimulatedClocks clocks(7);
  ClockMap<> map(Origin());
  ASSERT_TRUE(map.AddSample(clocks.ReadAt(Origin().tick + 50'000'000), kHorizonNs));
  uint64_t tick = Origin().tick + 50'100'000;
  uint64_t previous_end = map.Translate(tick);
  for (int dispatch = 0; dispatch < 50'000; ++dispatch) {
    const uint64_t start_tick = tick;
    tick += 7'500 + (dispatch % 13) * 100;  // 75-87 us kernels
    // The host observes completion a few us later and refreshes then.
    if (map.NeedsSample(tick, kHorizonNs)) {
      ASSERT_TRUE(map.AddSample(clocks.ReadAt(tick + 500), kHorizonNs));
    }
    const uint64_t end = map.Translate(tick);
    const uint64_t start = map.Translate(start_tick);
    ASSERT_LE(previous_end, start) << "dispatch " << dispatch;
    ASSERT_LT(start, end) << "dispatch " << dispatch;
    previous_end = end;
  }
}

// Interval durations survive a refresh: the disagreement between the old line
// and the new sample is spread over the horizon, not over the few microseconds
// between the frontier and the sample.
TEST(ClockMapTest, DurationsArePreservedThroughRefresh) {
  SimulatedClocks clocks(21);
  ClockMap<> map(Origin());
  ASSERT_TRUE(map.AddSample(clocks.ReadAt(Origin().tick + 50'000'000), kHorizonNs));
  uint64_t tick = Origin().tick + 50'000'100;
  int refreshes = 0;
  for (int dispatch = 0; dispatch < 50'000; ++dispatch) {
    const uint64_t duration_ticks = 1'000 + (dispatch % 7) * 500;  // 10-40 us
    const uint64_t start_tick = tick;
    tick += duration_ticks;
    if (map.NeedsSample(tick, kHorizonNs)) {
      ASSERT_TRUE(map.AddSample(clocks.ReadAt(tick + 300), kHorizonNs));
      ++refreshes;
    }
    const uint64_t end = map.Translate(tick);
    const uint64_t start = map.Translate(start_tick);
    const double expected = kNominalRatio * double(duration_ticks);
    const double measured = double(end - start);
    ASSERT_NEAR(measured, expected, expected * 1.0e-3 + 2.0) << "dispatch " << dispatch;
  }
  EXPECT_GE(refreshes, 15);
}

// Random interleaving of out-of-order translations and refreshes with jittered,
// drifting clocks: monotone at all times, stable for every committed tick,
// bounded error against the true clock relationship.
TEST(ClockMapTest, MonotoneAndStableUnderRandomUse) {
  SimulatedClocks clocks(1234);
  ClockMap<> map(Origin());
  std::map<uint64_t, uint64_t> committed;  // tick -> value handed out
  int accepted = 0;
  uint64_t now_tick = Origin().tick + 1'000'000;
  std::uniform_int_distribution<uint64_t> step(1, 2'000'000);
  std::uniform_int_distribution<uint64_t> lag(0, 20'000'000);  // translated up to 200 ms late
  for (int iteration = 0; iteration < 200'000; ++iteration) {
    now_tick += step(clocks.rng);
    const uint64_t age = lag(clocks.rng);
    const uint64_t tick = now_tick > age ? now_tick - age : now_tick;
    if (map.NeedsSample(tick, kHorizonNs) && map.AddSample(clocks.ReadAt(now_tick), kHorizonNs))
      ++accepted;
    const uint64_t value = map.Translate(tick);
    auto it = committed.find(tick);
    if (it != committed.end()) {
      ASSERT_EQ(it->second, value) << "tick " << tick << " revised at iteration " << iteration;
    } else {
      committed[tick] = value;
    }
    const uint64_t truth = clocks.SystemAt(tick);
    const uint64_t error = value > truth ? value - truth : truth - value;
    ASSERT_LT(error, 10'000u) << "tick " << tick;  // jitter plus drift over the horizon
  }
  uint64_t prev = 0;
  for (const auto& kv : committed) {
    ASSERT_GE(kv.second, prev) << "tick " << kv.first;
    prev = kv.second;
  }
  EXPECT_GT(accepted, 100);
}

// A sample whose line cannot be reached without revising values already handed
// out is rejected and leaves the map untouched, and the rejection rate-limits
// retries. A sample that merely lies behind the frontier is still joinable: its
// line is reached one horizon out.
TEST(ClockMapTest, RejectsUnjoinableSamplesAndRateLimitsRetries) {
  ClockMap<> map(Origin());
  ASSERT_TRUE(map.AddSample(Exact(Origin().tick + 50'000'000), kHorizonNs));
  const uint64_t far = Origin().tick + 60'000'000;  // 100 ms past the sample
  const uint64_t frontier_value = map.Translate(far);
  EXPECT_TRUE(map.NeedsSample(far, kHorizonNs));

  // A sample whose line stays below the frontier value even one horizon out.
  ClockSample low = Exact(far + 1);
  low.sys = frontier_value - 2 * kHorizonNs;
  EXPECT_FALSE(map.AddSample(low, kHorizonNs));
  EXPECT_EQ(map.newest().tick, Origin().tick + 50'000'000);
  EXPECT_EQ(map.Translate(far), frontier_value);
  // The rejected attempt anchors the retry policy: no new sample is requested
  // for ticks up to a horizon past the attempt.
  EXPECT_FALSE(map.NeedsSample(far + kHorizonTicks - 1, kHorizonNs));
  EXPECT_TRUE(map.NeedsSample(far + kHorizonTicks + 1, kHorizonNs));

  // A sample behind the frontier is accepted; the frontier value is kept and
  // the join lands exactly on the new line one horizon past the sample.
  const uint64_t behind = far - 100;
  EXPECT_TRUE(map.AddSample(Exact(behind), kHorizonNs));
  EXPECT_EQ(map.newest().tick, behind);
  EXPECT_EQ(map.Translate(far), frontier_value);
  EXPECT_EQ(map.Translate(behind + kHorizonTicks), Exact(behind + kHorizonTicks).sys);
}

// A speculative translation (commit=false) does not freeze the map.
TEST(ClockMapTest, UncommittedTranslationDoesNotFreeze) {
  ClockMap<> map(Origin());
  ASSERT_TRUE(map.AddSample(Exact(Origin().tick + 1'000'000), kHorizonNs));
  const uint64_t bogus = Origin().tick + 900'000'000'000ull;
  (void)map.Translate(bogus, false);
  EXPECT_TRUE(map.AddSample(Exact(Origin().tick + 8'000'000), kHorizonNs));
  EXPECT_EQ(map.newest().tick, Origin().tick + 8'000'000);
}

// The retained history is bounded: once it wraps, old ticks are extrapolated
// from the oldest retained knot and the map stays monotone throughout.
TEST(ClockMapTest, BoundedHistoryStaysMonotone) {
  SimulatedClocks clocks(99);
  ClockMap<4> map(Origin());
  uint64_t last_tick = 0;
  for (uint64_t i = 1; i <= 40; ++i) {
    const uint64_t tick = Origin().tick + i * kHorizonTicks;
    (void)map.Translate(tick + 10);  // extrapolate past the knot so each refresh adds two knots
    ASSERT_TRUE(map.AddSample(clocks.ReadAt(tick + 1'000), kHorizonNs));
    last_tick = tick + 1'000;
  }
  uint64_t prev = 0;
  for (uint64_t t = Origin().tick; t <= last_tick + 2 * kHorizonTicks; t += 12'345) {
    const uint64_t v = map.Translate(t);
    ASSERT_GE(v, prev) << "tick " << t;
    prev = v;
  }
}

// Ticks before the origin use the first measured ratio, fixed thereafter.
TEST(ClockMapTest, PreOriginTicksUseFixedRatio) {
  ClockMap<> map(Origin());
  EXPECT_EQ(map.Translate(Origin().tick - 1'000), Origin().sys);  // no ratio yet
  ASSERT_TRUE(map.AddSample(Exact(Origin().tick + 1'000'000), kHorizonNs));
  const uint64_t early = Origin().tick - 1'000;
  EXPECT_EQ(map.Translate(early), Origin().sys - 10'000);
  // A later sample changes ratio() but not the historical mapping.
  ASSERT_TRUE(map.AddSample({Origin().tick + 8'000'000, Origin().sys + 88'000'000}, kHorizonNs));
  EXPECT_NE(map.ratio(), kNominalRatio);
  EXPECT_EQ(map.Translate(early), Origin().sys - 10'000);
  EXPECT_EQ(map.Translate(0), Origin().sys - 10 * Origin().tick);
  // Saturates at zero rather than wrapping.
  ClockMap<> late(ClockSample{2'000'000'000'000ull, 100});
  ASSERT_TRUE(late.AddSample({2'000'001'000'000ull, 10'000'100}, kHorizonNs));
  EXPECT_EQ(late.Translate(0), 0u);
}

// NeedsSample implements both extrapolation limits.
TEST(ClockMapTest, NeedsSamplePolicy) {
  ClockMap<> map(Origin());
  const uint64_t baseline = 400'000'000;  // 4 s of ticks
  ASSERT_TRUE(map.AddSample(Exact(Origin().tick + baseline), kHorizonNs));
  const uint64_t newest = Origin().tick + baseline;
  EXPECT_FALSE(map.NeedsSample(newest, kHorizonNs));
  EXPECT_FALSE(map.NeedsSample(newest - 1, kHorizonNs));
  EXPECT_FALSE(map.NeedsSample(newest + kHorizonTicks - 1, kHorizonNs));
  EXPECT_TRUE(map.NeedsSample(newest + kHorizonTicks, kHorizonNs));
  // A short baseline tightens the limit to a quarter of the baseline.
  ClockMap<> young(Origin());
  ASSERT_TRUE(young.AddSample(Exact(Origin().tick + 1'000'000), kHorizonNs));
  EXPECT_FALSE(young.NeedsSample(Origin().tick + 1'250'000, kHorizonNs));
  EXPECT_TRUE(young.NeedsSample(Origin().tick + 1'250'001, kHorizonNs));
}

// Extreme ticks saturate instead of wrapping and keep the map monotone.
TEST(ClockMapTest, ExtremeTicksSaturate) {
  ClockMap<> map(Origin());
  ASSERT_TRUE(map.AddSample(Exact(Origin().tick + 1'000'000), kHorizonNs));
  const uint64_t huge = map.Translate(UINT64_MAX, false);
  EXPECT_EQ(huge, UINT64_MAX);
  EXPECT_GE(huge, map.Translate(UINT64_MAX / 2, false));
  EXPECT_GE(map.Translate(UINT64_MAX / 2, false), map.Translate(Origin().tick + 2'000'000, false));
}
