/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#ifdef MPI_TESTS_ENABLED
#include "net_ib_wqe_lat_inspect.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

/*
 * CPU-only unit coverage for the net_ib per-QP WQE post-to-poll latency
 * monitor (src/transport/net_ib/wqe_lat_mon.cc): the sampling state machine
 * (ncclIbWqeLatMonStampSend / OnComplete), the Welford mean/stddev and P^2
 * percentile statistics, the slow-completion threshold and the stall check
 * with their 1 s report rate limiters.
 *
 * Synthetic timestamps are fed straight into OnComplete/CheckStall, so the
 * expected values are exact. No MPI communication, RDMA or GPU is used; the
 * tests live in rccl-UnitTestsMPI only because that binary links librccl with
 * default symbol visibility. NetIbWqeLatMonTests.cpp covers the same monitor
 * end to end on live net_ib connections.
 */

namespace {

constexpr uint64_t kOneSecondNs = 1000000000ULL;
// Arbitrary non-zero base so post/poll timestamps never wrap below zero.
constexpr uint64_t kBaseNs = 1000 * kOneSecondNs;

// The monitor's globals are set once per process from the environment via
// std::call_once (ensureInitialized, triggered by ncclIbWqeLatMonInit). Fire
// that first so it can not overwrite the override later, then restore on exit.
struct ScopedWqeLatGlobals {
  bool savedEnabled;
  uint64_t savedThresholdNs;
  bool savedReport;
  explicit ScopedWqeLatGlobals(uint64_t thresholdNs) {
    struct ncclIbWqeLatMon dummy;
    ncclIbWqeLatMonInit(&dummy);
    savedEnabled = ncclIbWqeLatEnabled;
    savedThresholdNs = ncclIbWqeLatThresholdNs;
    savedReport = ncclIbWqeLatReportEnabled;
    ncclIbWqeLatEnabled = true;
    ncclIbWqeLatThresholdNs = thresholdNs;
    ncclIbWqeLatReportEnabled = true;
  }
  ~ScopedWqeLatGlobals() {
    ncclIbWqeLatEnabled = savedEnabled;
    ncclIbWqeLatThresholdNs = savedThresholdNs;
    ncclIbWqeLatReportEnabled = savedReport;
  }
};

// Puts m in the state StampSend leaves behind for a single signaled WR posted
// at postNs on an idle QP, then completes it at postNs + deltaNs.
bool recordSample(struct ncclIbWqeLatMon* m, uint64_t postNs, uint64_t deltaNs) {
  m->tracking = true;
  m->trackedPostNs = postNs;
  m->pendingBefore = 0;
  m->inflight = 1;
  return ncclIbWqeLatMonOnComplete(m, postNs + deltaNs, nullptr, nullptr, nullptr);
}

// Feeds deltas as consecutive samples spaced kOneSecondNs + 1 apart so the
// slow-report rate limiter never interferes with the statistics under test.
void recordSamples(struct ncclIbWqeLatMon* m, const std::vector<uint64_t>& deltas) {
  uint64_t t = kBaseNs;
  for (uint64_t d : deltas) {
    recordSample(m, t, d);
    t += kOneSecondNs + 1;
  }
}

struct ncclIbWqeLatStats snapshot(const struct ncclIbWqeLatMon* m) {
  struct ncclIbWqeLatStats s;
  ncclIbWqeLatMonSnapshot(m, &s);
  return s;
}

// Builds a WR chain with the given signaled pattern, linked through next.
std::vector<struct ibv_send_wr> makeWrChain(const std::vector<bool>& signaled) {
  std::vector<struct ibv_send_wr> wrs(signaled.size());
  for (size_t i = 0; i < wrs.size(); i++) {
    std::memset(&wrs[i], 0, sizeof(wrs[i]));
    wrs[i].send_flags = signaled[i] ? IBV_SEND_SIGNALED : 0;
    wrs[i].next = (i + 1 < wrs.size()) ? &wrs[i + 1] : nullptr;
  }
  return wrs;
}

}  // namespace

// =============================================================================
// Test: InitClearsAllState
//
// Init must zero a monitor regardless of prior contents, and a fresh
// monitor's snapshot reports no samples and zero for every statistic.
// =============================================================================
TEST(NetIbWqeLatMonUnit, InitClearsAllState) {
  struct ncclIbWqeLatMon m;
  std::memset(&m, 0xa5, sizeof(m));
  ncclIbWqeLatMonInit(&m);

  EXPECT_FALSE(m.tracking);
  EXPECT_EQ(m.inflight, 0u);
  EXPECT_EQ(m.pendingBefore, 0u);
  EXPECT_EQ(m.trackedPostNs, 0u);
  EXPECT_EQ(m.lastWarnNs, 0u);
  EXPECT_EQ(m.lastStallWarnNs, 0u);

  struct ncclIbWqeLatStats s = snapshot(&m);
  EXPECT_EQ(s.count, 0u);
  EXPECT_EQ(s.slowCount, 0u);
  EXPECT_EQ(s.meanNs, 0.0);
  EXPECT_EQ(s.stddevNs, 0.0);
  EXPECT_EQ(s.maxNs, 0u);
  EXPECT_EQ(s.p50Ns, 0u);
  EXPECT_EQ(s.p999Ns, 0u);
}

// =============================================================================
// Test: WelfordMeanStddevAndMax
//
// {100,200,300,400,500} ns: mean 300, sample stddev (n-1 divisor)
// sqrt(25000) ~= 158.1139, max 500.
// =============================================================================
TEST(NetIbWqeLatMonUnit, WelfordMeanStddevAndMax) {
  ScopedWqeLatGlobals g(kOneSecondNs);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  recordSamples(&m, {300, 100, 500, 200, 400});

  struct ncclIbWqeLatStats s = snapshot(&m);
  EXPECT_EQ(s.count, 5u);
  EXPECT_NEAR(s.meanNs, 300.0, 1e-9);
  EXPECT_NEAR(s.stddevNs, 158.11388300841898, 1e-6);
  EXPECT_EQ(s.maxNs, 500u);
  EXPECT_EQ(s.slowCount, 0u);
}

// =============================================================================
// Test: SingleSampleHasZeroStddev
//
// With one sample the variance is undefined; the snapshot reports 0 and every
// percentile equals the sample.
// =============================================================================
TEST(NetIbWqeLatMonUnit, SingleSampleHasZeroStddev) {
  ScopedWqeLatGlobals g(kOneSecondNs);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  recordSamples(&m, {1234});

  struct ncclIbWqeLatStats s = snapshot(&m);
  EXPECT_EQ(s.count, 1u);
  EXPECT_NEAR(s.meanNs, 1234.0, 1e-9);
  EXPECT_EQ(s.stddevNs, 0.0);
  EXPECT_EQ(s.maxNs, 1234u);
  EXPECT_EQ(s.p50Ns, 1234u);
  EXPECT_EQ(s.p90Ns, 1234u);
  EXPECT_EQ(s.p99Ns, 1234u);
  EXPECT_EQ(s.p999Ns, 1234u);
}

// =============================================================================
// Test: SmallCountPercentilesUseNearestRank
//
// Below 5 samples the P^2 markers are not initialised yet and the estimate is
// the nearest-rank percentile of the raw samples: index ceil(p*n)-1 of the
// sorted values. For {400,100,300}: p50 -> idx 1 = 300, p90/p99/p99.9 -> idx 2
// = 400.
// =============================================================================
TEST(NetIbWqeLatMonUnit, SmallCountPercentilesUseNearestRank) {
  ScopedWqeLatGlobals g(kOneSecondNs);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  recordSamples(&m, {400, 100, 300});

  struct ncclIbWqeLatStats s = snapshot(&m);
  EXPECT_EQ(s.count, 3u);
  EXPECT_EQ(s.p50Ns, 300u);
  EXPECT_EQ(s.p90Ns, 400u);
  EXPECT_EQ(s.p99Ns, 400u);
  EXPECT_EQ(s.p999Ns, 400u);
  EXPECT_EQ(s.maxNs, 400u);
}

// =============================================================================
// Test: P2PercentilesTrackUniformDistribution
//
// A fixed-seed shuffle of 1..10000 ns. P^2 is an approximation, so the
// percentiles are checked against the true quantiles with a 2% tolerance, plus
// the ladder p50 <= p90 <= p99 <= p99.9 <= max that the summary line relies on.
// =============================================================================
TEST(NetIbWqeLatMonUnit, P2PercentilesTrackUniformDistribution) {
  ScopedWqeLatGlobals g(UINT64_MAX);
  constexpr uint64_t kN = 10000;
  std::vector<uint64_t> deltas(kN);
  for (uint64_t i = 0; i < kN; i++) deltas[i] = i + 1;
  std::mt19937 rng(20132013);
  std::shuffle(deltas.begin(), deltas.end(), rng);

  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  recordSamples(&m, deltas);

  struct ncclIbWqeLatStats s = snapshot(&m);
  EXPECT_EQ(s.count, kN);
  EXPECT_EQ(s.maxNs, kN);
  EXPECT_NEAR(s.meanNs, (kN + 1) / 2.0, 1e-6);
  const double tol = 0.02 * kN;
  EXPECT_NEAR((double)s.p50Ns, 0.50 * kN, tol);
  EXPECT_NEAR((double)s.p90Ns, 0.90 * kN, tol);
  EXPECT_NEAR((double)s.p99Ns, 0.99 * kN, tol);
  EXPECT_NEAR((double)s.p999Ns, 0.999 * kN, tol);

  EXPECT_LE(s.p50Ns, s.p90Ns);
  EXPECT_LE(s.p90Ns, s.p99Ns);
  EXPECT_LE(s.p99Ns, s.p999Ns);
  EXPECT_LE(s.p999Ns, s.maxNs);
}

// =============================================================================
// Test: P2PercentilesSeparateBimodalTail
//
// 99% of samples at ~1 us and 1% at ~1 ms (a typical "mostly fast, rare slow
// CQE" pattern): p50/p90 stay in the fast mode, max reports the slow mode.
// =============================================================================
TEST(NetIbWqeLatMonUnit, P2PercentilesSeparateBimodalTail) {
  ScopedWqeLatGlobals g(UINT64_MAX);
  std::vector<uint64_t> deltas;
  for (int i = 0; i < 9900; i++) deltas.push_back(1000 + (i % 100));
  for (int i = 0; i < 100; i++) deltas.push_back(1000000 + i);
  std::mt19937 rng(7);
  std::shuffle(deltas.begin(), deltas.end(), rng);

  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  recordSamples(&m, deltas);

  struct ncclIbWqeLatStats s = snapshot(&m);
  EXPECT_EQ(s.count, 10000u);
  EXPECT_GE(s.p50Ns, 1000u);
  EXPECT_LT(s.p50Ns, 1100u);
  EXPECT_LT(s.p90Ns, 1100u);
  EXPECT_GE(s.p999Ns, 1000000u);
  EXPECT_EQ(s.maxNs, 1000099u);
  EXPECT_LE(s.p99Ns, s.p999Ns);
}

// =============================================================================
// Test: CompletionWithoutTrackingOnlyDrainsInflight
//
// A CQE for a WQE posted while another was already being tracked is not a
// sample: it only decrements inflight, which must also not underflow at 0.
// =============================================================================
TEST(NetIbWqeLatMonUnit, CompletionWithoutTrackingOnlyDrainsInflight) {
  ScopedWqeLatGlobals g(0);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  m.inflight = 2;

  uint64_t delta = 1, post = 1;
  struct ncclIbWqeLatStats out;
  EXPECT_FALSE(ncclIbWqeLatMonOnComplete(&m, kBaseNs, &delta, &post, &out));
  EXPECT_EQ(m.inflight, 1u);
  EXPECT_EQ(delta, 0u);
  EXPECT_EQ(post, 0u);
  EXPECT_EQ(out.count, 0u);

  EXPECT_FALSE(ncclIbWqeLatMonOnComplete(&m, kBaseNs, nullptr, nullptr, nullptr));
  EXPECT_FALSE(ncclIbWqeLatMonOnComplete(&m, kBaseNs, nullptr, nullptr, nullptr));
  EXPECT_EQ(m.inflight, 0u);
  EXPECT_EQ(snapshot(&m).count, 0u);
  EXPECT_EQ(snapshot(&m).slowCount, 0u);
}

// =============================================================================
// Test: PendingBeforeSkipsOlderCompletions
//
// When the tracked WQE was posted behind 2 older in-flight WQEs, the first two
// CQEs on the QP belong to those and must be skipped; the third closes the
// sample with delta measured from the tracked post time.
// =============================================================================
TEST(NetIbWqeLatMonUnit, PendingBeforeSkipsOlderCompletions) {
  ScopedWqeLatGlobals g(UINT64_MAX);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  m.tracking = true;
  m.trackedPostNs = kBaseNs;
  m.pendingBefore = 2;
  m.inflight = 3;

  EXPECT_FALSE(ncclIbWqeLatMonOnComplete(&m, kBaseNs + 10, nullptr, nullptr, nullptr));
  EXPECT_FALSE(ncclIbWqeLatMonOnComplete(&m, kBaseNs + 20, nullptr, nullptr, nullptr));
  EXPECT_TRUE(m.tracking);
  EXPECT_EQ(m.pendingBefore, 0u);
  EXPECT_EQ(snapshot(&m).count, 0u);

  uint64_t delta = 0, post = 0;
  struct ncclIbWqeLatStats out;
  EXPECT_FALSE(ncclIbWqeLatMonOnComplete(&m, kBaseNs + 700, &delta, &post, &out));
  EXPECT_FALSE(m.tracking);
  EXPECT_EQ(m.inflight, 0u);
  EXPECT_EQ(delta, 700u);
  EXPECT_EQ(post, kBaseNs);
  EXPECT_EQ(out.count, 1u);
  EXPECT_EQ(out.maxNs, 700u);
}

// =============================================================================
// Test: PollBeforePostClampsDeltaToZero
//
// A poll timestamp older than the post timestamp (not expected with
// CLOCK_MONOTONIC, but guarded) records a zero-length sample rather than a
// wrapped huge delta.
// =============================================================================
TEST(NetIbWqeLatMonUnit, PollBeforePostClampsDeltaToZero) {
  ScopedWqeLatGlobals g(0);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  m.tracking = true;
  m.trackedPostNs = kBaseNs;
  m.inflight = 1;

  uint64_t delta = 1;
  EXPECT_FALSE(ncclIbWqeLatMonOnComplete(&m, kBaseNs - 5, &delta, nullptr, nullptr));
  EXPECT_EQ(delta, 0u);
  struct ncclIbWqeLatStats s = snapshot(&m);
  EXPECT_EQ(s.count, 1u);
  EXPECT_EQ(s.maxNs, 0u);
  EXPECT_EQ(s.slowCount, 0u);
}

// =============================================================================
// Test: SlowThresholdIsStrict
//
// A completion is slow only when delta > threshold: delta == threshold is not
// counted, threshold + 1 is counted and requests a report.
// =============================================================================
TEST(NetIbWqeLatMonUnit, SlowThresholdIsStrict) {
  constexpr uint64_t kThr = 5000;
  ScopedWqeLatGlobals g(kThr);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);

  EXPECT_FALSE(recordSample(&m, kBaseNs, kThr - 1));
  EXPECT_FALSE(recordSample(&m, kBaseNs + 10 * kOneSecondNs, kThr));
  EXPECT_EQ(snapshot(&m).slowCount, 0u);

  EXPECT_TRUE(recordSample(&m, kBaseNs + 20 * kOneSecondNs, kThr + 1));
  struct ncclIbWqeLatStats s = snapshot(&m);
  EXPECT_EQ(s.count, 3u);
  EXPECT_EQ(s.slowCount, 1u);
}

// =============================================================================
// Test: SlowReportsAreRateLimitedButAllCounted
//
// Slow completions within 1 s of the last report are still counted in
// slowCount (which feeds the close-time summary) but do not request another
// INFO line; the next slow completion at >= 1 s does.
// =============================================================================
TEST(NetIbWqeLatMonUnit, SlowReportsAreRateLimitedButAllCounted) {
  constexpr uint64_t kThr = 1000;
  ScopedWqeLatGlobals g(kThr);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);

  // First slow completion polled at kBaseNs + 2000 -> reported.
  EXPECT_TRUE(recordSample(&m, kBaseNs, 2000));
  const uint64_t firstReport = kBaseNs + 2000;
  EXPECT_EQ(m.lastWarnNs, firstReport);

  // Ten more slow completions inside the 1 s window -> counted, not reported.
  for (int i = 1; i <= 10; i++) {
    EXPECT_FALSE(recordSample(&m, kBaseNs + i * 1000000ULL, 2000)) << "i=" << i;
  }
  EXPECT_EQ(snapshot(&m).slowCount, 11u);
  EXPECT_EQ(m.lastWarnNs, firstReport);

  // A fast completion in between does not reset the limiter.
  EXPECT_FALSE(recordSample(&m, kBaseNs + 500000000ULL, 10));

  // Polled at exactly firstReport + 1 s -> outside the window, reported again.
  EXPECT_TRUE(recordSample(&m, firstReport + kOneSecondNs - 2000, 2000));
  EXPECT_EQ(m.lastWarnNs, firstReport + kOneSecondNs);
  struct ncclIbWqeLatStats s = snapshot(&m);
  EXPECT_EQ(s.slowCount, 12u);
  EXPECT_EQ(s.count, 13u);
}

// =============================================================================
// Test: StallCheckBoundaryAndRateLimit
//
// A tracked WQE is a stall once its age exceeds the threshold (strictly). The
// stall report is rate-limited to once per second, independently of the slow
// limiter, and reports the age and current inflight count.
// =============================================================================
TEST(NetIbWqeLatMonUnit, StallCheckBoundaryAndRateLimit) {
  constexpr uint64_t kThr = 1000000;
  ScopedWqeLatGlobals g(kThr);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  m.tracking = true;
  m.trackedPostNs = kBaseNs;
  m.inflight = 4;

  uint64_t age = 0;
  uint32_t inflight = 0;
  EXPECT_FALSE(ncclIbWqeLatMonCheckStall(&m, kBaseNs, &age, &inflight));
  EXPECT_FALSE(ncclIbWqeLatMonCheckStall(&m, kBaseNs + kThr, &age, &inflight));
  EXPECT_EQ(age, kThr);
  EXPECT_EQ(inflight, 4u);

  EXPECT_TRUE(ncclIbWqeLatMonCheckStall(&m, kBaseNs + kThr + 1, &age, &inflight));
  EXPECT_EQ(age, kThr + 1);
  EXPECT_EQ(inflight, 4u);

  EXPECT_FALSE(ncclIbWqeLatMonCheckStall(&m, kBaseNs + kThr + 1 + kOneSecondNs - 1, &age, &inflight));
  EXPECT_TRUE(ncclIbWqeLatMonCheckStall(&m, kBaseNs + kThr + 1 + kOneSecondNs, &age, &inflight));

  // The stall check never records samples or slow completions.
  struct ncclIbWqeLatStats s = snapshot(&m);
  EXPECT_EQ(s.count, 0u);
  EXPECT_EQ(s.slowCount, 0u);
  EXPECT_EQ(m.lastWarnNs, 0u);
}

// =============================================================================
// Test: StallCheckIgnoresUntrackedQp
//
// With nothing tracked (idle QP, or only untracked WQEs in flight) there is no
// stall regardless of how much time has passed.
// =============================================================================
TEST(NetIbWqeLatMonUnit, StallCheckIgnoresUntrackedQp) {
  ScopedWqeLatGlobals g(1);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  m.inflight = 3;

  uint64_t age = 1;
  uint32_t inflight = 1;
  EXPECT_FALSE(ncclIbWqeLatMonCheckStall(&m, kBaseNs + 100 * kOneSecondNs, &age, &inflight));
  EXPECT_EQ(age, 0u);
  EXPECT_EQ(inflight, 0u);
}

// =============================================================================
// Test: StampSendIgnoresUnsignaledWrs
//
// Only signaled WRs produce CQEs, so only they are counted in inflight; a
// chain without any signaled WR leaves the monitor untouched.
// =============================================================================
TEST(NetIbWqeLatMonUnit, StampSendIgnoresUnsignaledWrs) {
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);

  auto unsignaled = makeWrChain({false, false, false});
  ASSERT_EQ(ncclIbWqeLatTestStampSend(&m, unsignaled.data()), ncclSuccess);
  EXPECT_FALSE(m.tracking);
  EXPECT_EQ(m.inflight, 0u);

  // The net_ib send path: unsignaled data WRs followed by one signaled
  // RDMA_WRITE_WITH_IMM as the last WR of the chain.
  auto mixed = makeWrChain({false, false, true});
  const uint64_t before = ncclIbWqeLatMonNowNs();
  ASSERT_EQ(ncclIbWqeLatTestStampSend(&m, mixed.data()), ncclSuccess);
  const uint64_t after = ncclIbWqeLatMonNowNs();
  EXPECT_TRUE(m.tracking);
  EXPECT_EQ(m.inflight, 1u);
  EXPECT_EQ(m.pendingBefore, 0u);
  EXPECT_GE(m.trackedPostNs, before);
  EXPECT_LE(m.trackedPostNs, after);
}

// =============================================================================
// Test: StampSendTracksFirstSignaledWrOnly
//
// A chain with several signaled WRs tracks only the first; the rest just add
// to inflight. Completing the first CQE closes the sample and the remaining
// CQEs only drain inflight.
// =============================================================================
TEST(NetIbWqeLatMonUnit, StampSendTracksFirstSignaledWrOnly) {
  ScopedWqeLatGlobals g(UINT64_MAX);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);

  auto wrs = makeWrChain({true, false, true, true});
  ASSERT_EQ(ncclIbWqeLatTestStampSend(&m, wrs.data()), ncclSuccess);
  EXPECT_TRUE(m.tracking);
  EXPECT_EQ(m.pendingBefore, 0u);
  EXPECT_EQ(m.inflight, 3u);
  const uint64_t postNs = m.trackedPostNs;

  uint64_t delta = 0, post = 0;
  ncclIbWqeLatMonOnComplete(&m, postNs + 42, &delta, &post, nullptr);
  EXPECT_EQ(delta, 42u);
  EXPECT_EQ(post, postNs);
  EXPECT_FALSE(m.tracking);
  EXPECT_EQ(m.inflight, 2u);

  ncclIbWqeLatMonOnComplete(&m, postNs + 50, nullptr, nullptr, nullptr);
  ncclIbWqeLatMonOnComplete(&m, postNs + 60, nullptr, nullptr, nullptr);
  EXPECT_EQ(m.inflight, 0u);
  EXPECT_EQ(snapshot(&m).count, 1u);
}

// =============================================================================
// Test: PipelinedPostsSampleOnePerTrackingWindow
//
// Models posting on a QP while earlier WQEs are still in flight:
//   post A (tracked), post B (not tracked, A still tracked),
//   complete A -> sample, post C while B still in flight (tracked with
//   pendingBefore 1), complete B -> skipped, complete C -> sample.
// Exactly two samples are recorded and inflight returns to 0.
// =============================================================================
TEST(NetIbWqeLatMonUnit, PipelinedPostsSampleOnePerTrackingWindow) {
  ScopedWqeLatGlobals g(UINT64_MAX);
  struct ncclIbWqeLatMon m;
  ncclIbWqeLatMonInit(&m);
  auto a = makeWrChain({true});
  auto b = makeWrChain({true});
  auto c = makeWrChain({true});

  ASSERT_EQ(ncclIbWqeLatTestStampSend(&m, a.data()), ncclSuccess);
  const uint64_t postA = m.trackedPostNs;
  ASSERT_EQ(ncclIbWqeLatTestStampSend(&m, b.data()), ncclSuccess);
  EXPECT_EQ(m.trackedPostNs, postA);
  EXPECT_EQ(m.inflight, 2u);

  ncclIbWqeLatMonOnComplete(&m, ncclIbWqeLatMonNowNs(), nullptr, nullptr, nullptr);
  EXPECT_FALSE(m.tracking);
  EXPECT_EQ(m.inflight, 1u);
  EXPECT_EQ(snapshot(&m).count, 1u);

  ASSERT_EQ(ncclIbWqeLatTestStampSend(&m, c.data()), ncclSuccess);
  EXPECT_TRUE(m.tracking);
  EXPECT_EQ(m.pendingBefore, 1u);
  EXPECT_EQ(m.inflight, 2u);

  ncclIbWqeLatMonOnComplete(&m, ncclIbWqeLatMonNowNs(), nullptr, nullptr, nullptr);  // B
  EXPECT_TRUE(m.tracking);
  EXPECT_EQ(snapshot(&m).count, 1u);
  ncclIbWqeLatMonOnComplete(&m, ncclIbWqeLatMonNowNs(), nullptr, nullptr, nullptr);  // C
  EXPECT_FALSE(m.tracking);
  EXPECT_EQ(m.inflight, 0u);
  EXPECT_EQ(snapshot(&m).count, 2u);
}

#endif  // MPI_TESTS_ENABLED
