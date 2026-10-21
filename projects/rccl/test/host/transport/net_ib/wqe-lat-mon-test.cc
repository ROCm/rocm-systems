/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/wqe_lat_mon.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <deque>
#include <vector>

#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/socket_fakes.h"
#include "fakes/transport_net_ib_wqe_lat_fakes.h"
#include "transport/net_ib/common.h"
#include "../../../common/LogCapture.hpp"

namespace {

std::deque<uint64_t> g_wqeClockValues;
int g_wqeClockCalls = 0;
int g_wqeClockResult = 0;

int TestWqeClockGettime(clockid_t clockId, struct timespec* ts) {
  EXPECT_EQ(clockId, CLOCK_MONOTONIC);
  ++g_wqeClockCalls;
  if (g_wqeClockResult != 0) {
    errno = EIO;
    return -1;
  }
  uint64_t value = g_wqeClockValues.empty() ? 0 : g_wqeClockValues.front();
  if (!g_wqeClockValues.empty()) g_wqeClockValues.pop_front();
  ts->tv_sec = value / 1000000000ULL;
  ts->tv_nsec = value % 1000000000ULL;
  return 0;
}

}  // namespace

#define ncclParamIbWqeLatencyThresholdNs NetIbWqeLatencyThresholdNs
#define ncclParamIbWqeLatencyReport NetIbWqeLatencyReport
#define ncclIbWqeLatEnabled NetIbWqeLatEnabled
#define ncclIbWqeLatThresholdNs NetIbWqeLatThresholdNs
#define ncclIbWqeLatReportEnabled NetIbWqeLatReportEnabled
#define ncclIbWqeLatMonNowNs NetIbWqeLatMonNowNs
#define ncclIbWqeLatMonInit NetIbWqeLatMonInit
#define ncclIbWqeLatMonStampSend NetIbWqeLatMonStampSend
#define ncclIbWqeLatMonOnComplete NetIbWqeLatMonOnComplete
#define ncclIbWqeLatMonCheckStall NetIbWqeLatMonCheckStall
#define ncclIbWqeLatMonSnapshot NetIbWqeLatMonSnapshot
#define ncclIbWqeLatHandleCompletion NetIbWqeLatHandleCompletion
#define ncclIbWqeLatScanStalls NetIbWqeLatScanStalls
#define ncclIbWqeLatReportQpSummary NetIbWqeLatReportQpSummary
#define clock_gettime TestWqeClockGettime
#include NET_IB_WQE_LAT_MON_CC_PATH
#undef clock_gettime
#undef ncclIbWqeLatReportQpSummary
#undef ncclIbWqeLatScanStalls
#undef ncclIbWqeLatHandleCompletion
#undef ncclIbWqeLatMonSnapshot
#undef ncclIbWqeLatMonCheckStall
#undef ncclIbWqeLatMonOnComplete
#undef ncclIbWqeLatMonStampSend
#undef ncclIbWqeLatMonInit
#undef ncclIbWqeLatMonNowNs
#undef ncclIbWqeLatReportEnabled
#undef ncclIbWqeLatThresholdNs
#undef ncclIbWqeLatEnabled
#undef ncclParamIbWqeLatencyReport
#undef ncclParamIbWqeLatencyThresholdNs

namespace {

constexpr uint64_t kSecondNs = 1000000000ULL;
constexpr uint64_t kBaseNs = 100 * kSecondNs;

void InitMonitor(struct ncclIbWqeLatMon* monitor) {
  std::memset(monitor, 0, sizeof(*monitor));
  for (int i = 0; i < NCCL_IB_WQE_LAT_NUM_PCTL; ++i) p2Init(&monitor->p2[i], kPctlTargets[i]);
}

std::vector<ibv_send_wr> MakeWrChain(std::initializer_list<bool> signaled) {
  std::vector<ibv_send_wr> wrs(signaled.size());
  size_t index = 0;
  for (bool signal : signaled) {
    std::memset(&wrs[index], 0, sizeof(wrs[index]));
    wrs[index].send_flags = signal ? IBV_SEND_SIGNALED : 0;
    wrs[index].next = index + 1 < wrs.size() ? &wrs[index + 1] : nullptr;
    ++index;
  }
  return wrs;
}

bool RecordSample(struct ncclIbWqeLatMon* monitor, uint64_t postNs, uint64_t deltaNs,
                  struct ncclIbWqeLatStats* stats = nullptr) {
  monitor->tracking = true;
  monitor->trackedPostNs = postNs;
  monitor->pendingBefore = 0;
  monitor->inflight++;
  return NetIbWqeLatMonOnComplete(monitor, postNs + deltaNs, nullptr, nullptr, stats);
}

struct WqeCompletionScene {
  ncclIbSendComm send{};
  std::array<ibv_qp, 2> verbsQps{};

  WqeCompletionScene() {
    send.base.isSend = true;
    send.base.nqps = 2;
    send.base.vProps.ndevs = 2;
    send.base.nRemDevs = 1;
    for (int i = 0; i < 2; ++i) {
      verbsQps[i].qp_num = 100 + i;
      send.base.qps[i].qp = &verbsQps[i];
      send.base.qps[i].devIndex = i;
      send.base.qps[i].remDevIdx = 0;
      send.base.qps[i].rtrAttr.localIbPort = i + 1;
      send.base.activeQps[i] = &send.base.qps[i];
      send.devs[i].base.ibDevN = i;
      InitMonitor(&send.base.qps[i].latMon);
    }
  }
};

class NetIbWqeLatMonMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (int i = 0; i < 2; ++i) savedPortAttrs_[i] = ncclIbDevs[i].portAttr;
  }

  void TearDown() override {
    for (int i = 0; i < 2; ++i) ncclIbDevs[i].portAttr = savedPortAttrs_[i];
    g_wqeClockValues.clear();
    g_wqeClockCalls = 0;
    g_wqeClockResult = 0;
    NetIbWqeLatEnabled = false;
    NetIbWqeLatThresholdNs = 0;
    NetIbWqeLatReportEnabled = false;
    ResetSocketFakes();
    ResetNcclFakes();
  }

 private:
  std::array<decltype(ncclIbDevs[0].portAttr), 2> savedPortAttrs_{};
};

TEST_F(NetIbWqeLatMonMicrotest, Init_ClearsStateAndSeedsQuantileTargets) {
  EXPECT_EXIT(
    {
      g_loadParam = [](const char* name, int64_t fallback) {
        if (std::strcmp(name, "IB_WQE_LATENCY_THRESHOLD_NS") == 0) return int64_t{123};
        if (std::strcmp(name, "IB_WQE_LATENCY_REPORT") == 0) return int64_t{1};
        return fallback;
      };
      ncclIbWqeLatMon monitor;
      std::memset(&monitor, 0xa5, sizeof(monitor));
      NetIbWqeLatMonInit(&monitor);
      if (!NetIbWqeLatEnabled || NetIbWqeLatThresholdNs != 123 || !NetIbWqeLatReportEnabled) std::exit(1);
      if (monitor.tracking || monitor.inflight != 0 || monitor.count != 0 || monitor.slowCount != 0) std::exit(2);
      for (int i = 0; i < NCCL_IB_WQE_LAT_NUM_PCTL; ++i) {
        if (monitor.p2[i].p != kPctlTargets[i]) std::exit(3);
      }
      std::exit(0);
    },
    ::testing::ExitedWithCode(0), "");
}

TEST_F(NetIbWqeLatMonMicrotest, Init_NonPositiveThresholdLeavesMonitoringDisabled) {
  EXPECT_EXIT(
    {
      NetIbWqeLatEnabled = false;
      NetIbWqeLatThresholdNs = 0;
      NetIbWqeLatReportEnabled = false;
      g_loadParam = [](const char* name, int64_t fallback) {
        if (std::strcmp(name, "IB_WQE_LATENCY_THRESHOLD_NS") == 0) return int64_t{0};
        if (std::strcmp(name, "IB_WQE_LATENCY_REPORT") == 0) return int64_t{1};
        return fallback;
      };
      ncclIbWqeLatMon monitor{};
      NetIbWqeLatMonInit(&monitor);
      if (NetIbWqeLatEnabled || NetIbWqeLatThresholdNs != 0 || NetIbWqeLatReportEnabled) std::exit(1);
      std::exit(0);
    },
    ::testing::ExitedWithCode(0), "");
}

TEST_F(NetIbWqeLatMonMicrotest, NowNs_ConvertsMonotonicTimespec) {
  g_wqeClockValues.push_back(3 * kSecondNs + 47);

  EXPECT_EQ(NetIbWqeLatMonNowNs(), 3 * kSecondNs + 47);
  EXPECT_EQ(g_wqeClockCalls, 1);
}

TEST_F(NetIbWqeLatMonMicrotest, StampSend_NullAndUnsignaledChainsDoNotReadClock) {
  ncclIbQp qp{};
  InitMonitor(&qp.latMon);
  auto wrs = MakeWrChain({false, false});

  NetIbWqeLatMonStampSend(&qp, nullptr);
  NetIbWqeLatMonStampSend(&qp, wrs.data());

  EXPECT_EQ(g_wqeClockCalls, 0);
  EXPECT_FALSE(qp.latMon.tracking);
  EXPECT_EQ(qp.latMon.inflight, 0u);
}

TEST_F(NetIbWqeLatMonMicrotest, StampSend_IdleTracksFirstSignaledAndCountsAllSignaledWrs) {
  ncclIbQp qp{};
  InitMonitor(&qp.latMon);
  auto wrs = MakeWrChain({false, true, false, true});
  g_wqeClockValues.push_back(kBaseNs);

  NetIbWqeLatMonStampSend(&qp, wrs.data());

  EXPECT_EQ(g_wqeClockCalls, 1);
  EXPECT_TRUE(qp.latMon.tracking);
  EXPECT_EQ(qp.latMon.trackedPostNs, kBaseNs);
  EXPECT_EQ(qp.latMon.pendingBefore, 0u);
  EXPECT_EQ(qp.latMon.inflight, 2u);
}

TEST_F(NetIbWqeLatMonMicrotest, StampSend_WhileTrackingPreservesWindowAndOnlyAddsInflight) {
  ncclIbQp qp{};
  InitMonitor(&qp.latMon);
  qp.latMon.tracking = true;
  qp.latMon.trackedPostNs = kBaseNs;
  qp.latMon.inflight = 2;
  auto wrs = MakeWrChain({true, true});
  g_wqeClockValues.push_back(kBaseNs + 10);

  NetIbWqeLatMonStampSend(&qp, wrs.data());

  EXPECT_EQ(qp.latMon.trackedPostNs, kBaseNs);
  EXPECT_EQ(qp.latMon.inflight, 4u);
  EXPECT_EQ(qp.latMon.pendingBefore, 0u);
}

TEST_F(NetIbWqeLatMonMicrotest, StampSend_BackloggedQpSkipsOlderCompletionsBeforeSample) {
  ncclIbQp qp{};
  InitMonitor(&qp.latMon);
  qp.latMon.inflight = 2;
  auto wrs = MakeWrChain({true});
  g_wqeClockValues.push_back(kBaseNs);
  NetIbWqeLatThresholdNs = UINT64_MAX;

  NetIbWqeLatMonStampSend(&qp, wrs.data());
  ASSERT_TRUE(qp.latMon.tracking);
  EXPECT_EQ(qp.latMon.pendingBefore, 2u);
  EXPECT_EQ(qp.latMon.inflight, 3u);

  EXPECT_FALSE(NetIbWqeLatMonOnComplete(&qp.latMon, kBaseNs + 10, nullptr, nullptr, nullptr));
  EXPECT_FALSE(NetIbWqeLatMonOnComplete(&qp.latMon, kBaseNs + 20, nullptr, nullptr, nullptr));
  EXPECT_TRUE(qp.latMon.tracking);
  EXPECT_EQ(qp.latMon.count, 0u);
  EXPECT_FALSE(NetIbWqeLatMonOnComplete(&qp.latMon, kBaseNs + 30, nullptr, nullptr, nullptr));
  EXPECT_FALSE(qp.latMon.tracking);
  EXPECT_EQ(qp.latMon.inflight, 0u);
  EXPECT_EQ(qp.latMon.count, 1u);
  EXPECT_EQ(qp.latMon.maxNs, 30u);
}

TEST_F(NetIbWqeLatMonMicrotest, OnComplete_UntrackedCompletionOnlyDrainsInflightAndClearsOutputs) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  monitor.inflight = 1;
  uint64_t delta = 7;
  uint64_t post = 8;
  ncclIbWqeLatStats stats;
  std::memset(&stats, 0xa5, sizeof(stats));

  EXPECT_FALSE(NetIbWqeLatMonOnComplete(&monitor, kBaseNs, &delta, &post, &stats));
  EXPECT_EQ(monitor.inflight, 0u);
  EXPECT_EQ(delta, 0u);
  EXPECT_EQ(post, 0u);
  EXPECT_EQ(stats.count, 0u);
  EXPECT_EQ(stats.maxNs, 0u);
  EXPECT_FALSE(NetIbWqeLatMonOnComplete(&monitor, kBaseNs, nullptr, nullptr, nullptr));
  EXPECT_EQ(monitor.inflight, 0u);
}

TEST_F(NetIbWqeLatMonMicrotest, OnComplete_PendingBeforeSkipsOlderCompletions) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  NetIbWqeLatThresholdNs = UINT64_MAX;
  monitor.tracking = true;
  monitor.trackedPostNs = kBaseNs;
  monitor.pendingBefore = 2;
  monitor.inflight = 3;

  EXPECT_FALSE(NetIbWqeLatMonOnComplete(&monitor, kBaseNs + 10, nullptr, nullptr, nullptr));
  EXPECT_FALSE(NetIbWqeLatMonOnComplete(&monitor, kBaseNs + 20, nullptr, nullptr, nullptr));
  EXPECT_TRUE(monitor.tracking);
  EXPECT_EQ(monitor.pendingBefore, 0u);
  EXPECT_EQ(monitor.count, 0u);

  uint64_t delta = 0;
  EXPECT_FALSE(NetIbWqeLatMonOnComplete(&monitor, kBaseNs + 30, &delta, nullptr, nullptr));
  EXPECT_FALSE(monitor.tracking);
  EXPECT_EQ(monitor.inflight, 0u);
  EXPECT_EQ(delta, 30u);
  EXPECT_EQ(monitor.count, 1u);
}

TEST_F(NetIbWqeLatMonMicrotest, OnComplete_PollBeforePostClampsSampleToZero) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  monitor.tracking = true;
  monitor.trackedPostNs = kBaseNs;
  monitor.inflight = 1;
  uint64_t delta = 1;

  EXPECT_FALSE(NetIbWqeLatMonOnComplete(&monitor, kBaseNs - 1, &delta, nullptr, nullptr));
  EXPECT_EQ(delta, 0u);
  EXPECT_EQ(monitor.count, 1u);
  EXPECT_EQ(monitor.maxNs, 0u);
}

TEST_F(NetIbWqeLatMonMicrotest, OnComplete_StrictThresholdAndOneSecondLimiterCountEverySlowSample) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  NetIbWqeLatThresholdNs = 100;

  EXPECT_FALSE(RecordSample(&monitor, kBaseNs, 100));
  EXPECT_TRUE(RecordSample(&monitor, kBaseNs + kSecondNs, 101));
  uint64_t firstWarn = monitor.lastWarnNs;
  EXPECT_FALSE(RecordSample(&monitor, firstWarn + 1, 101));
  EXPECT_TRUE(RecordSample(&monitor, firstWarn + kSecondNs - 101, 101));

  EXPECT_EQ(monitor.count, 4u);
  EXPECT_EQ(monitor.slowCount, 3u);
  EXPECT_EQ(monitor.lastWarnNs, firstWarn + kSecondNs);
}

TEST_F(NetIbWqeLatMonMicrotest, CheckStall_StrictThresholdAndIndependentLimiter) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  monitor.tracking = true;
  monitor.trackedPostNs = kBaseNs;
  monitor.inflight = 3;
  NetIbWqeLatThresholdNs = 50;
  uint64_t age = 0;
  uint32_t inflight = 0;

  EXPECT_FALSE(NetIbWqeLatMonCheckStall(&monitor, kBaseNs + 50, &age, &inflight));
  EXPECT_EQ(age, 50u);
  EXPECT_EQ(inflight, 3u);
  EXPECT_TRUE(NetIbWqeLatMonCheckStall(&monitor, kBaseNs + 51, &age, &inflight));
  uint64_t firstWarn = monitor.lastStallWarnNs;
  EXPECT_FALSE(NetIbWqeLatMonCheckStall(&monitor, firstWarn + kSecondNs - 1, nullptr, nullptr));
  EXPECT_TRUE(NetIbWqeLatMonCheckStall(&monitor, firstWarn + kSecondNs, nullptr, nullptr));
  EXPECT_EQ(monitor.count, 0u);
  EXPECT_EQ(monitor.lastWarnNs, 0u);
}

TEST_F(NetIbWqeLatMonMicrotest, CheckStall_UntrackedOrNonForwardTimeClearsOutputsWithoutWarning) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  uint64_t age = 9;
  uint32_t inflight = 9;

  EXPECT_FALSE(NetIbWqeLatMonCheckStall(&monitor, kBaseNs, &age, &inflight));
  EXPECT_EQ(age, 0u);
  EXPECT_EQ(inflight, 0u);

  monitor.tracking = true;
  monitor.trackedPostNs = kBaseNs;
  monitor.inflight = 2;
  age = 9;
  inflight = 9;
  EXPECT_FALSE(NetIbWqeLatMonCheckStall(&monitor, kBaseNs, &age, &inflight));
  EXPECT_EQ(age, 0u);
  EXPECT_EQ(inflight, 0u);
  EXPECT_FALSE(NetIbWqeLatMonCheckStall(&monitor, kBaseNs - 1, &age, &inflight));
  EXPECT_EQ(monitor.lastStallWarnNs, 0u);
}

TEST_F(NetIbWqeLatMonMicrotest, Snapshot_EmptyAndSingleSampleReportExactStatistics) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  NetIbWqeLatThresholdNs = UINT64_MAX;
  ncclIbWqeLatStats stats{};

  NetIbWqeLatMonSnapshot(&monitor, &stats);
  EXPECT_EQ(stats.count, 0u);
  EXPECT_EQ(stats.p50Ns, 0u);

  RecordSample(&monitor, kBaseNs, 123);
  NetIbWqeLatMonSnapshot(&monitor, &stats);
  EXPECT_EQ(stats.count, 1u);
  EXPECT_DOUBLE_EQ(stats.meanNs, 123.0);
  EXPECT_DOUBLE_EQ(stats.stddevNs, 0.0);
  EXPECT_EQ(stats.maxNs, 123u);
  EXPECT_EQ(stats.p50Ns, 123u);
  EXPECT_EQ(stats.p999Ns, 123u);
}

TEST_F(NetIbWqeLatMonMicrotest, Snapshot_FiveSamplesReportWelfordAndSeededP2Statistics) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  NetIbWqeLatThresholdNs = UINT64_MAX;
  for (uint64_t delta : {300, 100, 500, 200, 400}) {
    RecordSample(&monitor, kBaseNs + monitor.count * (kSecondNs + 1000), delta);
  }
  ncclIbWqeLatStats stats{};

  NetIbWqeLatMonSnapshot(&monitor, &stats);

  EXPECT_EQ(stats.count, 5u);
  EXPECT_NEAR(stats.meanNs, 300.0, 1e-9);
  EXPECT_NEAR(stats.stddevNs, 158.11388300841898, 1e-6);
  EXPECT_EQ(stats.maxNs, 500u);
  EXPECT_EQ(stats.p50Ns, 300u);
  EXPECT_EQ(stats.p90Ns, 300u);
  EXPECT_EQ(stats.p99Ns, 300u);
  EXPECT_EQ(stats.p999Ns, 300u);
}

TEST_F(NetIbWqeLatMonMicrotest, Snapshot_AscendingSamplesProduceBoundedOrderedPercentiles) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  NetIbWqeLatThresholdNs = UINT64_MAX;
  for (uint64_t delta = 1; delta <= 30; ++delta) {
    RecordSample(&monitor, kBaseNs + delta * (kSecondNs + 100), delta);
  }
  ncclIbWqeLatStats stats{};

  NetIbWqeLatMonSnapshot(&monitor, &stats);

  EXPECT_EQ(stats.count, 30u);
  EXPECT_EQ(stats.maxNs, 30u);
  EXPECT_NEAR(stats.meanNs, 15.5, 1e-9);
  EXPECT_GE(stats.p50Ns, 12u);
  EXPECT_LE(stats.p50Ns, 18u);
  EXPECT_GE(stats.p90Ns, stats.p50Ns);
  EXPECT_LE(stats.p999Ns, stats.maxNs);
}

TEST_F(NetIbWqeLatMonMicrotest, Snapshot_DescendingSamplesProduceBoundedOrderedPercentiles) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  NetIbWqeLatThresholdNs = UINT64_MAX;
  for (uint64_t delta = 30; delta > 0; --delta) {
    RecordSample(&monitor, kBaseNs + (31 - delta) * (kSecondNs + 100), delta);
  }
  ncclIbWqeLatStats stats{};

  NetIbWqeLatMonSnapshot(&monitor, &stats);

  EXPECT_EQ(stats.count, 30u);
  EXPECT_EQ(stats.maxNs, 30u);
  EXPECT_NEAR(stats.meanNs, 15.5, 1e-9);
  EXPECT_GE(stats.p50Ns, 12u);
  EXPECT_LE(stats.p50Ns, 18u);
  EXPECT_GE(stats.p90Ns, stats.p50Ns);
  EXPECT_GE(stats.p99Ns, stats.p90Ns);
  EXPECT_GE(stats.p999Ns, stats.p99Ns);
  EXPECT_LE(stats.p999Ns, stats.maxNs);
}

TEST_F(NetIbWqeLatMonMicrotest, Snapshot_MixedUniformSamplesSeparateMedianAndTailPercentiles) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  NetIbWqeLatThresholdNs = UINT64_MAX;
  std::vector<uint64_t> values;
  for (uint64_t value = 1; value <= 50; ++value) {
    values.push_back(value);
    values.push_back(101 - value);
  }
  for (size_t i = 0; i < values.size(); ++i) {
    RecordSample(&monitor, kBaseNs + (i + 1) * (kSecondNs + 100), values[i]);
  }
  ncclIbWqeLatStats stats{};

  NetIbWqeLatMonSnapshot(&monitor, &stats);

  EXPECT_EQ(stats.count, 100u);
  EXPECT_EQ(stats.maxNs, 100u);
  EXPECT_NEAR(stats.p50Ns, 50.0, 10.0);
  EXPECT_NEAR(stats.p90Ns, 90.0, 10.0);
  EXPECT_GE(stats.p90Ns, stats.p50Ns + 20);
  EXPECT_GE(stats.p99Ns, stats.p90Ns + 3);
  EXPECT_GE(stats.p99Ns, 90u);
  EXPECT_GE(stats.p999Ns, stats.p99Ns);
  EXPECT_LE(stats.p999Ns, stats.maxNs);
}

TEST_F(NetIbWqeLatMonMicrotest, HandleCompletion_IgnoresRecvAndUnknownQpWithoutReadingClock) {
  WqeCompletionScene scene;
  auto& monitor = scene.send.base.qps[0].latMon;
  monitor.tracking = true;
  monitor.trackedPostNs = kBaseNs;
  monitor.inflight = 1;
  uint64_t pollNs = 0;
  bool pollNsValid = false;
  ibv_wc completion{};
  completion.opcode = IBV_WC_RECV;
  completion.qp_num = scene.verbsQps[0].qp_num;

  NetIbWqeLatHandleCompletion(&scene.send.base, 0, &completion, &pollNs, &pollNsValid);
  completion.opcode = IBV_WC_SEND;
  completion.qp_num = 999;
  NetIbWqeLatHandleCompletion(&scene.send.base, 0, &completion, &pollNs, &pollNsValid);

  EXPECT_EQ(g_wqeClockCalls, 0);
  EXPECT_FALSE(pollNsValid);
  EXPECT_TRUE(monitor.tracking);
  EXPECT_EQ(monitor.inflight, 1u);
}

TEST_F(NetIbWqeLatMonMicrotest, HandleCompletion_SendOpcodesShareOnePollTimestamp) {
  WqeCompletionScene scene;
  scene.send.base.vProps.ndevs = 1;
  scene.send.base.qps[1].devIndex = 0;
  NetIbWqeLatThresholdNs = UINT64_MAX;
  for (int i = 0; i < 2; ++i) {
    scene.send.base.qps[i].latMon.tracking = true;
    scene.send.base.qps[i].latMon.trackedPostNs = kBaseNs;
    scene.send.base.qps[i].latMon.inflight = 1;
  }
  g_wqeClockValues.push_back(kBaseNs + 42);
  uint64_t pollNs = 0;
  bool pollNsValid = false;
  ibv_wc first{};
  first.opcode = IBV_WC_RDMA_WRITE;
  first.qp_num = scene.verbsQps[0].qp_num;
  ibv_wc second{};
  second.opcode = IBV_WC_RDMA_READ;
  second.qp_num = scene.verbsQps[1].qp_num;

  NetIbWqeLatHandleCompletion(&scene.send.base, 0, &first, &pollNs, &pollNsValid);
  NetIbWqeLatHandleCompletion(&scene.send.base, 0, &second, &pollNs, &pollNsValid);

  EXPECT_TRUE(pollNsValid);
  EXPECT_EQ(pollNs, kBaseNs + 42);
  EXPECT_EQ(g_wqeClockCalls, 1);
  EXPECT_EQ(scene.send.base.qps[0].latMon.maxNs, 42u);
  EXPECT_EQ(scene.send.base.qps[1].latMon.maxNs, 42u);
}

TEST_F(NetIbWqeLatMonMicrotest, HandleCompletion_SlowSendRecordsSampleAndEmitsReportPath) {
  WqeCompletionScene scene;
  auto& monitor = scene.send.base.qps[0].latMon;
  monitor.tracking = true;
  monitor.trackedPostNs = kBaseNs;
  monitor.inflight = 1;
  NetIbWqeLatThresholdNs = 1;
  g_wqeClockValues.push_back(kBaseNs + 10);
  g_ncclSocketGetAddr = [](ncclSocket*, ncclSocketAddress*) { return ncclSystemError; };
  uint64_t pollNs = 0;
  bool pollNsValid = false;
  ibv_wc completion{};
  completion.opcode = IBV_WC_SEND;
  completion.qp_num = scene.verbsQps[0].qp_num;

  RcclUnitTesting::ScopedDebugLogging logging(NCCL_LOG_INFO, NCCL_ALL);
  std::string log = RcclUnitTesting::CaptureLog(
    [&] { NetIbWqeLatHandleCompletion(&scene.send.base, 0, &completion, &pollNs, &pollNsValid); });

  EXPECT_EQ(monitor.count, 1u);
  EXPECT_EQ(monitor.slowCount, 1u);
  EXPECT_EQ(monitor.maxNs, 10u);
  EXPECT_FALSE(monitor.tracking);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "WQE slow:"));
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "delta=10ns"));
}

TEST_F(NetIbWqeLatMonMicrotest, ScanStalls_VisitsOnlyRequestedInterleavedDevice) {
  WqeCompletionScene scene;
  for (auto& qp : scene.send.base.qps) {
    qp.latMon.tracking = true;
    qp.latMon.trackedPostNs = kBaseNs;
    qp.latMon.inflight = 1;
  }
  NetIbWqeLatThresholdNs = 1;
  g_wqeClockValues.push_back(kBaseNs + 10);
  g_ncclSocketGetAddr = [](ncclSocket*, ncclSocketAddress*) { return ncclSystemError; };

  RcclUnitTesting::ScopedDebugLogging logging(NCCL_LOG_INFO, NCCL_ALL);
  std::string log = RcclUnitTesting::CaptureLog([&] { NetIbWqeLatScanStalls(&scene.send.base, 1); });

  EXPECT_EQ(g_wqeClockCalls, 1);
  EXPECT_EQ(scene.send.base.qps[0].latMon.lastStallWarnNs, 0u);
  EXPECT_EQ(scene.send.base.qps[1].latMon.lastStallWarnNs, kBaseNs + 10);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "WQE stall (no CQE):"));
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "qpn=101"));
}

// AICOMRCCL-3131: stall scans should inspect the active failover QP, not a
// stale raw QP that has been replaced on the communicator.
TEST_F(NetIbWqeLatMonMicrotest, DISABLED_ScanStalls_ReplacedQpDoesNotReportStaleOriginal) {
  ncclIbSendComm send{};
  ibv_qp originalVerbs{};
  ibv_qp replacementVerbs{};
  ncclIbQp replacement{};
  send.base.isSend = true;
  send.base.nqps = 1;
  send.base.vProps.ndevs = 1;
  originalVerbs.qp_num = 10;
  replacementVerbs.qp_num = 11;
  send.base.qps[0].qp = &originalVerbs;
  replacement.qp = &replacementVerbs;
  send.base.activeQps[0] = &replacement;
  InitMonitor(&send.base.qps[0].latMon);
  InitMonitor(&replacement.latMon);
  send.base.qps[0].latMon.tracking = true;
  send.base.qps[0].latMon.trackedPostNs = kBaseNs;
  send.base.qps[0].latMon.inflight = 1;
  NetIbWqeLatThresholdNs = 1;
  g_wqeClockValues.push_back(kBaseNs + 10);
  g_ncclSocketGetAddr = [](ncclSocket*, ncclSocketAddress*) { return ncclSystemError; };

  NetIbWqeLatScanStalls(&send.base, 0);

  EXPECT_EQ(send.base.qps[0].latMon.lastStallWarnNs, 0u);
  EXPECT_EQ(replacement.latMon.lastStallWarnNs, 0u);
}

TEST_F(NetIbWqeLatMonMicrotest, ReportSummary_EitherGlobalGateDisabledEmitsNothing) {
  WqeCompletionScene scene;
  auto& monitor = scene.send.base.qps[0].latMon;
  InitMonitor(&monitor);
  NetIbWqeLatThresholdNs = UINT64_MAX;
  RecordSample(&monitor, kBaseNs, 25);
  g_ncclSocketGetAddr = [](ncclSocket*, ncclSocketAddress*) { return ncclSystemError; };

  RcclUnitTesting::ScopedDebugLogging logging(NCCL_LOG_INFO, NCCL_ALL);
  std::string log = RcclUnitTesting::CaptureLog([&] {
    NetIbWqeLatReportEnabled = true;
    NetIbWqeLatEnabled = false;
    NetIbWqeLatReportQpSummary(&scene.send.base, 0, &scene.send.base.qps[0]);
    NetIbWqeLatEnabled = true;
    NetIbWqeLatReportEnabled = false;
    NetIbWqeLatReportQpSummary(&scene.send.base, 0, &scene.send.base.qps[0]);
  });

  EXPECT_EQ(monitor.count, 1u);
  EXPECT_FALSE(RcclUnitTesting::LogHas(log, "WQE latency summary"));
}

TEST_F(NetIbWqeLatMonMicrotest, ReportSummary_NullVerbsOrEmptyQpEmitsNothing) {
  WqeCompletionScene scene;
  ncclIbQp nullVerbsQp{};
  InitMonitor(&nullVerbsQp.latMon);
  nullVerbsQp.latMon.count = 1;
  NetIbWqeLatEnabled = true;
  NetIbWqeLatReportEnabled = true;

  RcclUnitTesting::ScopedDebugLogging logging(NCCL_LOG_INFO, NCCL_ALL);
  std::string log = RcclUnitTesting::CaptureLog([&] {
    NetIbWqeLatReportQpSummary(&scene.send.base, 0, nullptr);
    NetIbWqeLatReportQpSummary(&scene.send.base, 0, &nullVerbsQp);
    NetIbWqeLatReportQpSummary(&scene.send.base, 0, &scene.send.base.qps[0]);
  });

  EXPECT_FALSE(RcclUnitTesting::LogHas(log, "WQE latency summary"));
}

TEST_F(NetIbWqeLatMonMicrotest, ReportSummary_EnabledSampleEmitsStatistics) {
  WqeCompletionScene scene;
  auto& monitor = scene.send.base.qps[0].latMon;
  InitMonitor(&monitor);
  NetIbWqeLatThresholdNs = UINT64_MAX;
  RecordSample(&monitor, kBaseNs, 25);
  NetIbWqeLatEnabled = true;
  NetIbWqeLatReportEnabled = true;
  g_ncclSocketGetAddr = [](ncclSocket*, ncclSocketAddress*) { return ncclSystemError; };

  RcclUnitTesting::ScopedDebugLogging logging(NCCL_LOG_INFO, NCCL_ALL);
  std::string log = RcclUnitTesting::CaptureLog(
    [&] { NetIbWqeLatReportQpSummary(&scene.send.base, 0, &scene.send.base.qps[0]); });

  EXPECT_EQ(monitor.count, 1u);
  EXPECT_EQ(monitor.maxNs, 25u);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "WQE latency summary [send"));
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "n=1"));
}

// AICOMRCCL-3131: clock failure currently arms a tracking window
// using the contents of a failed timespec read.
TEST_F(NetIbWqeLatMonMicrotest, DISABLED_StampSend_ClockFailureDoesNotStartTracking) {
  ncclIbQp qp{};
  InitMonitor(&qp.latMon);
  auto wrs = MakeWrChain({true});
  g_wqeClockValues.push_back(kBaseNs);
  g_wqeClockResult = EIO;

  NetIbWqeLatMonStampSend(&qp, wrs.data());

  EXPECT_FALSE(qp.latMon.tracking);
  EXPECT_EQ(qp.latMon.inflight, 0u);
}

// AICOMRCCL-3131: returned statistics should include the slow
// classification performed by the same completion.
TEST_F(NetIbWqeLatMonMicrotest, DISABLED_OnComplete_SlowSampleReturnsPostClassificationStatistics) {
  ncclIbWqeLatMon monitor{};
  InitMonitor(&monitor);
  NetIbWqeLatThresholdNs = 1;
  ncclIbWqeLatStats stats{};

  EXPECT_TRUE(RecordSample(&monitor, kBaseNs, 2, &stats));
  EXPECT_EQ(stats.count, 1u);
  EXPECT_EQ(stats.slowCount, 1u);
}

}  // namespace
