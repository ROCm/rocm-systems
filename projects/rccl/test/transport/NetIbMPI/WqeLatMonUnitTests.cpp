/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#ifdef MPI_TESTS_ENABLED
#include "wqe_lat_mon_cast.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

/*
 * Pure-math unit coverage for the P^2 percentile estimator + Welford
 * mean/variance running stats in wqe_lat_mon_cast.cc, independent of any QP,
 * MPI rank, or real completion. The functional MPI tests in WqeLatMonTests.cpp
 * drive this through real ibv_post_send/poll_cq round trips and only check
 * ladder/positivity invariants on whatever latencies the NIC happens to
 * produce; these tests feed known synthetic deltas directly into
 * IbCastWqeLatMonOnComplete/CheckStall and check the computed statistics
 * against hand-computed expected values.
 *
 * These build into rccl-UnitTestsMPI and run under the MPI launcher
 * (main_mpi.cpp initializes MPI) but use no MPI communication, no RDMA, and no
 * hardware: plain CPU, operating directly on a stack struct ncclIbCastWqeLatMon.
 */

namespace {

// Feeds one completed sample of exactly deltaNs through the same tracked/
// pendingBefore=0 path IbCastWqeLatMonStampSend would set up for a single
// signaled send, without needing a real ncclIbQp/ibv_send_wr.
void recordSample(struct ncclIbCastWqeLatMon* m, uint64_t deltaNs) {
  m->tracking = true;
  m->trackedPostNs = 0;
  m->pendingBefore = 0;
  IbCastWqeLatMonOnComplete(m, deltaNs, nullptr, nullptr, nullptr);
}

// ncclIbCastWqeLatThresholdNs is a process-global read once per binary via
// std::call_once (ensureInitialized in wqe_lat_mon_cast.cc), so a test that
// needs a specific threshold must override and restore the extern directly
// rather than relying on env-var timing against other tests sharing this
// process.
struct ScopedThreshold {
  uint64_t saved;
  explicit ScopedThreshold(uint64_t ns) : saved(ncclIbCastWqeLatThresholdNs) {
    ncclIbCastWqeLatThresholdNs = ns;
  }
  ~ScopedThreshold() { ncclIbCastWqeLatThresholdNs = saved; }
};

}  // namespace

// =============================================================================
// Test: WelfordMeanAndStddevMatchHandComputedValues
//
// Feeds the deterministic sequence {100, 200, 300, 400, 500} ns and checks the
// running mean/stddev against the closed-form sample mean/stddev (n-1
// divisor), and that maxNs tracks the largest delta seen.
// =============================================================================
TEST(WqeLatMonUnit, WelfordMeanAndStddevMatchHandComputedValues) {
  struct ncclIbCastWqeLatMon m;
  IbCastWqeLatMonInit(&m);

  const uint64_t samples[] = {100, 200, 300, 400, 500};
  for (uint64_t s : samples) recordSample(&m, s);

  struct ncclIbCastWqeLatStats stats;
  IbCastWqeLatMonSnapshot(&m, &stats);

  EXPECT_EQ(stats.count, 5u);
  EXPECT_NEAR(stats.meanNs, 300.0, 1e-6);
  // Sample variance (n-1 divisor) of {100,200,300,400,500} is 25000, stddev ~=158.1139.
  EXPECT_NEAR(stats.stddevNs, 158.11388300841898, 1e-6);
  EXPECT_EQ(stats.maxNs, 500u);
}

// =============================================================================
// Test: PercentileLadderMonotonicAndCloseToKnownDistribution
//
// Feeds a fixed-seed shuffled permutation of 1..1000 (ns) through the P^2
// estimator and checks: (a) the ladder is non-decreasing end to end, capped by
// the true observed max, and (b) p50 lands close to the true median of a
// uniform 1..1000 distribution -- P^2 is an approximation, not exact, so this
// uses a generous tolerance rather than an exact match.
// =============================================================================
TEST(WqeLatMonUnit, PercentileLadderMonotonicAndCloseToKnownDistribution) {
  struct ncclIbCastWqeLatMon m;
  IbCastWqeLatMonInit(&m);

  std::vector<uint64_t> samples(1000);
  for (uint64_t i = 0; i < 1000; i++) samples[i] = i + 1;
  std::mt19937 rng(12345);
  std::shuffle(samples.begin(), samples.end(), rng);

  for (uint64_t s : samples) recordSample(&m, s);

  struct ncclIbCastWqeLatStats stats;
  IbCastWqeLatMonSnapshot(&m, &stats);

  EXPECT_EQ(stats.count, 1000u);
  EXPECT_EQ(stats.maxNs, 1000u);
  EXPECT_LE(stats.p50Ns, stats.p90Ns);
  EXPECT_LE(stats.p90Ns, stats.p99Ns);
  EXPECT_LE(stats.p99Ns, stats.p999Ns);
  EXPECT_LE(stats.p999Ns, stats.maxNs);
  // True median of 1..1000 is ~500.5; P^2 approximation tolerance is generous.
  EXPECT_NEAR(static_cast<double>(stats.p50Ns), 500.5, 50.0);
}

// =============================================================================
// Test: OnCompleteNoOpWhenNotTracking
//
// A completion that arrives while the monitor isn't tracking a signaled send
// (e.g. an unsignaled WQE's CQE, or a spurious wakeup) must not perturb count/
// mean/max and must report "not slow".
// =============================================================================
TEST(WqeLatMonUnit, OnCompleteNoOpWhenNotTracking) {
  struct ncclIbCastWqeLatMon m;
  IbCastWqeLatMonInit(&m);
  m.tracking = false;

  bool slow = IbCastWqeLatMonOnComplete(&m, 123456789ULL, nullptr, nullptr, nullptr);

  EXPECT_FALSE(slow);
  EXPECT_EQ(m.count, 0u);
  EXPECT_EQ(m.maxNs, 0u);
}

// =============================================================================
// Test: OnCompletePendingBeforeSkipsStaleCompletionsWithoutRecording
//
// pendingBefore counts CQEs for WQEs posted before the currently-tracked one
// started tracking (inflight carried over from an earlier, already-handled
// signaled send). Those completions must decrement pendingBefore and return
// without touching count/mean/max, and tracking must stay armed for the real
// sample that follows. ncclIbCastWqeLatThresholdNs defaults to 0 (env var
// unset), so the real completion's delta (500ns) is "slow" by definition and
// IbCastWqeLatMonOnComplete reports true on it (first over-threshold sample,
// rate limiter not yet armed) -- pin the threshold explicitly so this isn't
// implicitly relying on process-global state left by other tests.
// =============================================================================
TEST(WqeLatMonUnit, OnCompletePendingBeforeSkipsStaleCompletionsWithoutRecording) {
  ScopedThreshold threshold(1000);

  struct ncclIbCastWqeLatMon m;
  IbCastWqeLatMonInit(&m);
  m.tracking = true;
  m.trackedPostNs = 0;
  m.pendingBefore = 2;

  EXPECT_FALSE(IbCastWqeLatMonOnComplete(&m, 999, nullptr, nullptr, nullptr));
  EXPECT_EQ(m.pendingBefore, 1u);
  EXPECT_EQ(m.count, 0u);
  EXPECT_TRUE(m.tracking);

  EXPECT_FALSE(IbCastWqeLatMonOnComplete(&m, 999, nullptr, nullptr, nullptr));
  EXPECT_EQ(m.pendingBefore, 0u);
  EXPECT_EQ(m.count, 0u);
  EXPECT_TRUE(m.tracking);

  // The real tracked completion: still armed, now actually recorded. delta
  // (500ns) is under the 1000ns threshold pinned above, so this is not slow.
  EXPECT_FALSE(IbCastWqeLatMonOnComplete(&m, 500, nullptr, nullptr, nullptr));
  EXPECT_EQ(m.count, 1u);
  EXPECT_EQ(m.maxNs, 500u);
  EXPECT_FALSE(m.tracking);
}

// =============================================================================
// Test: StallDetectionThresholdBoundaryAndStallCountIndependentOfRateLimiter
//
// CheckStall must not trigger at exactly the threshold (age <= threshold), must
// trigger just past it, and stallCount must keep incrementing on every
// over-threshold check even while the once-per-second rate limiter suppresses
// the boolean return (regression coverage for the stallCount introspection
// counter added in this change).
// =============================================================================
TEST(WqeLatMonUnit, StallDetectionThresholdBoundaryAndStallCountIndependentOfRateLimiter) {
  ScopedThreshold threshold(1000);

  struct ncclIbCastWqeLatMon m;
  IbCastWqeLatMonInit(&m);
  m.tracking = true;
  m.trackedPostNs = 0;

  // Exactly at the threshold: no stall.
  uint64_t age = 0;
  uint32_t inflight = 0;
  EXPECT_FALSE(IbCastWqeLatMonCheckStall(&m, 1000, &age, &inflight));
  EXPECT_EQ(age, 1000u);
  EXPECT_EQ(m.stallCount, 0u);

  // Just past the threshold: first detection reports (rate limiter not yet armed).
  EXPECT_TRUE(IbCastWqeLatMonCheckStall(&m, 1001, &age, &inflight));
  EXPECT_EQ(age, 1001u);
  EXPECT_EQ(m.stallCount, 1u);

  // Still over threshold, but within the 1s rate-limit window: stallCount still
  // increments, but the reporting return value is suppressed.
  EXPECT_FALSE(IbCastWqeLatMonCheckStall(&m, 1002, &age, &inflight));
  EXPECT_EQ(m.stallCount, 2u);
}

#endif /* MPI_TESTS_ENABLED */
