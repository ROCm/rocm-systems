/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// maxCTAs bounds the grid of every kernel, P2P ones included (NCCL 2.32.6). The grid and the comm's
// P2P channel pool are read from the "Launch P2P kernel: gridDim.x=... (p2pnChannels=..." line
// ncclLaunchKernel logs under NCCL_DEBUG_SUBSYS=COLL.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "common/DeviceBufferHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/TestChecks.hpp"

namespace RcclUnitTesting
{
namespace
{
constexpr int    kMaxGpus = 8;
constexpr int    kMinGpus = 4;  // enough rounds to spread one peer per channel past every cap below
constexpr size_t kCount   = 256 * 1024;  // floats per peer
constexpr int    kNoCap   = 1 << 30;
constexpr int    kTimeoutSeconds = 180;

enum class P2pMode { Exchange, ScatterGather };

struct P2pLaunch { int grid; int p2pnChannels; };

// gridDim.x and p2pnChannels of every P2P kernel launch logged past `offset` bytes of `path`.
std::vector<P2pLaunch> P2pLaunches(const std::string& path, std::streamoff offset)
{
  std::vector<P2pLaunch> launches;
  std::ifstream          f(path);
  f.seekg(offset);
  std::string line;
  while (std::getline(f, line))
  {
    P2pLaunch l{};
    auto      pos = line.find("Launch P2P kernel: gridDim.x=");
    if (pos != std::string::npos &&
        std::sscanf(line.c_str() + pos, "Launch P2P kernel: gridDim.x=%d blockDim.x=%*d (p2pnChannels=%d", &l.grid,
                    &l.p2pnChannels) == 2)
      launches.push_back(l);
  }
  return launches;
}

// Runs one P2P-lowered operation on every GPU and checks its data and the grid of each P2P launch.
// commMaxCTAs/callMaxCTAs <= 0 leave the comm and per-call configs at their defaults; a comm cap is
// also its minCTAs, so the comm runs exactly commMaxCTAs channels.
void RunAndCheckGrids(const char* log, P2pMode mode, int commMaxCTAs, int callMaxCTAs, int gridCap)
{
  ASSERT_NE(nullptr, log) << "NCCL_DEBUG_FILE is not set";
  int devCount = 0;
  HIP_CHECK(hipGetDeviceCount(&devCount));
  int const n = std::min(devCount, kMaxGpus);
  if (n < kMinGpus) GTEST_SKIP() << "Requires >= " << kMinGpus << " GPUs; found " << devCount;
  bool const exchange = mode == P2pMode::Exchange;

  std::vector<ncclComm_t>  comms(n);
  std::vector<hipStream_t> streams(n);
  std::vector<float*>      sendbuf(n), recvbuf(n);
  ncclUniqueId             id;
  ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id));
  ASSERT_EQ(ncclSuccess, ncclGroupStart());
  for (int r = 0; r < n; r++)
  {
    HIP_CHECK(hipSetDevice(r));
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    if (commMaxCTAs > 0) config.minCTAs = config.maxCTAs = commMaxCTAs;
    ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&comms[r], n, id, r, &config));
  }
  ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  for (int r = 0; r < n; r++)
  {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipStreamCreate(&streams[r]));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&sendbuf[r]), n * kCount * sizeof(float)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&recvbuf[r]), (n + 1) * kCount * sizeof(float)));
    // Slot p of rank r's sendbuf holds r * n + p.
    HIP_CHECK(RCCLTestHelpers::initializeBufferWithPattern<float>(
      sendbuf[r], n * kCount, [r, n](size_t i) { return static_cast<float>(r * n + i / kCount); }));
  }
  std::ifstream        sized(log, std::ios::ate);
  std::streamoff const afterInit = sized ? std::streamoff(sized.tellg()) : 0;  // skip what init logged

  ncclCollConfig_t callConfig = NCCL_COLLCONFIG_INITIALIZER;
  if (callMaxCTAs > 0) callConfig.maxCTAs = callMaxCTAs;
  // Exchange: every rank sends slot p to rank p. Otherwise root 0 scatters slot r to rank r's recvbuf slot 0,
  // then gathers every rank's sendbuf slot 0 into its recvbuf slots 1..n.
  for (int op = 0; op < (exchange ? 1 : 2); op++)
  {
    ASSERT_EQ(ncclSuccess, ncclGroupStart());
    for (int r = 0; r < n; r++)
    {
      if (!exchange && op == 0)
        ASSERT_EQ(ncclSuccess, ncclScatterConfig(sendbuf[r], recvbuf[r], kCount, ncclFloat, 0, comms[r], streams[r],
                                                 &callConfig));
      if (!exchange && op == 1)
        ASSERT_EQ(ncclSuccess, ncclGatherConfig(sendbuf[r], recvbuf[r] + kCount, kCount, ncclFloat, 0, comms[r],
                                                streams[r], &callConfig));
      for (int p = 0; exchange && p < n; p++)
      {
        ASSERT_EQ(ncclSuccess, ncclSend(sendbuf[r] + p * kCount, kCount, ncclFloat, p, comms[r], streams[r]));
        ASSERT_EQ(ncclSuccess, ncclRecv(recvbuf[r] + p * kCount, kCount, ncclFloat, p, comms[r], streams[r]));
      }
    }
    ASSERT_EQ(ncclSuccess, ncclGroupEnd());
  }
  for (int r = 0; r < n; r++)
  {
    HIP_CHECK(hipSetDevice(r));
    HIP_CHECK(hipStreamSynchronize(streams[r]));
    size_t const checked = exchange ? n * kCount : (r == 0 ? (n + 1) * kCount : kCount);
    EXPECT_TRUE(RCCLTestHelpers::verifyBufferData<float>(recvbuf[r], checked, [r, n, exchange](size_t i) {
      int const slot = static_cast<int>(i / kCount);
      return static_cast<float>(exchange ? slot * n + r : (slot == 0 ? r : (slot - 1) * n));
    }, 0, 0.0)) << "rank " << r << " received wrong data";
  }
  for (int r = 0; r < n; r++)
  {
    ASSERT_EQ(ncclSuccess, ncclCommDestroy(comms[r]));
    HIP_CHECK(hipFree(sendbuf[r]));
    HIP_CHECK(hipFree(recvbuf[r]));
    HIP_CHECK(hipStreamDestroy(streams[r]));
  }

  std::vector<P2pLaunch> const launches = P2pLaunches(log, afterInit);
  ASSERT_GE(static_cast<int>(launches.size()), n) << "expected a P2P kernel launch per rank in " << log;
  for (auto const& l : launches)
  {
    EXPECT_LE(l.grid, gridCap) << "P2P kernel launched wider than maxCTAs";
    if (commMaxCTAs > 0) EXPECT_LE(l.p2pnChannels, commMaxCTAs) << "P2P channel pool exceeds the comm's maxCTAs";
  }
}

// Runs RunAndCheckGrids in a fresh process whose NCCL log goes to its own file.
void RunIsolated(const char* name, P2pMode mode, int commMaxCTAs, int callMaxCTAs, int gridCap)
{
  std::string const log = std::string("/tmp/rccl_p2p_maxctas_") + name + "_" + std::to_string(getpid()) + ".log";
  ProcessIsolatedTestRunner::ExecutionOptions options;
  options.verboseLogging = true;
  RUN_ISOLATED_TESTS_WITH_OPTIONS(
    options,
    ProcessIsolatedTestRunner::TestConfig(
      name,
      // The child re-runs this body under its own pid, so it takes the log path from its environment.
      [=]() { RunAndCheckGrids(std::getenv("NCCL_DEBUG_FILE"), mode, commMaxCTAs, callMaxCTAs, gridCap); })
      .withNumGpus(kMaxGpus)
      // RCCL_FORCE_CE=0 keeps Scatter/Gather on the P2P kernel instead of the copy engine.
      .withEnvironment({{"NCCL_DEBUG", "INFO"}, {"NCCL_DEBUG_SUBSYS", "COLL"}, {"NCCL_DEBUG_FILE", log},
                        {"RCCL_FORCE_CE", "0"}})
      .withTimeout(std::chrono::seconds(kTimeoutSeconds)));
  std::remove(log.c_str());
}
}  // namespace

// Control: default caps; the log carries a P2P launch per rank and the exchange delivers.
TEST(P2pMaxCtas, DefaultCapsLaunchAndDeliverExchange)
{
  RunIsolated("DefaultCapsLaunchAndDeliverExchange", P2pMode::Exchange, 0, 0, kNoCap);
}

// A comm maxCTAs that is not a power of two: p2pnChannels used to round up past it (5 -> 8).
TEST(P2pMaxCtas, NonPow2CommCapBoundsSendRecvGrid)
{
  RunIsolated("NonPow2CommCapBoundsSendRecvGrid", P2pMode::Exchange, 5, 0, 5);
}

// A per-call maxCTAs on Scatter/Gather used to be dropped when they were lowered to P2P tasks.
TEST(P2pMaxCtas, PerCallCapBoundsScatterGatherGrid)
{
  RunIsolated("PerCallCapBoundsScatterGatherGrid", P2pMode::ScatterGather, 0, 1, 1);
}

}  // namespace RcclUnitTesting
