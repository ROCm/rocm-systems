/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// RasProgressCounters: with RAS on and NCCL_PROGRESS_COUNTERS=1, the generic kernel counts the work it
// completes per ncclFunc_t slot in comm->deviceCountersBlock (profiler() in src/device/common.h), and a
// per-device monitor thread copies that block into the pinned comm->hostCountersBlock
// (src/ras/progress_monitor.cc).

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "comm.h"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

constexpr int kOps = 3;
constexpr size_t kCount = 1024;

// Creates a one-rank communicator and sends to self kOps times. One-rank collectives are plain copies,
// so this is how a single rank reaches the generic kernel: each group is one launch with one P2P work.
void sendRecvToSelf(ncclComm_t* comm) {
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ncclUniqueId id;
  ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
  ASSERT_EQ(ncclCommInitRank(comm, 1, id, 0), ncclSuccess);
  std::vector<float> sent(kCount, 1.5f), received(kCount);
  float *sendBuf = nullptr, *recvBuf = nullptr;
  hipStream_t stream;
  ASSERT_EQ(hipMalloc(&sendBuf, kCount * sizeof(float)), hipSuccess);
  ASSERT_EQ(hipMalloc(&recvBuf, kCount * sizeof(float)), hipSuccess);
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
  ASSERT_EQ(hipMemcpy(sendBuf, sent.data(), kCount * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemsetAsync(recvBuf, 0, kCount * sizeof(float), stream), hipSuccess);
  for (int op = 0; op < kOps; op++) {
    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    ASSERT_EQ(ncclSend(sendBuf, kCount, ncclFloat, 0, *comm, stream), ncclSuccess);
    ASSERT_EQ(ncclRecv(recvBuf, kCount, ncclFloat, 0, *comm, stream), ncclSuccess);
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
  }
  ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
  ASSERT_EQ(hipMemcpy(received.data(), recvBuf, kCount * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_TRUE(received == sent) << "send/recv to self delivered the wrong data";
  (void)hipStreamDestroy(stream);
  (void)hipFree(sendBuf);
  (void)hipFree(recvBuf);
}

} // namespace

TEST(RasProgressCounters, CountSelfSendRecv) {
  RUN_ISOLATED_TEST_WITH_ENV("RasProgressCounters.CountSelfSendRecv", []() {
    int nGpus = 0;
    if (hipGetDeviceCount(&nGpus) != hipSuccess || nGpus < 1) GTEST_SKIP() << "requires at least one GPU";
    ncclComm_t comm = nullptr;
    ASSERT_NO_FATAL_FAILURE(sendRecvToSelf(&comm));
    ASSERT_NE(comm->deviceCountersBlock, nullptr) << "NCCL_PROGRESS_COUNTERS=1 allocated no device counters";
    ASSERT_NE(comm->hostCountersBlock, nullptr) << "NCCL_PROGRESS_COUNTERS=1 allocated no host mirror";

    // The stream has drained, so the device block is final; the monitor copies it to the host mirror
    // every NCCL_PROGRESS_COUNTER_MONITOR_POLL_MS.
    ncclProgressCountersBlock counters;
    ASSERT_EQ(hipMemcpy(&counters, comm->deviceCountersBlock, sizeof(counters), hipMemcpyDeviceToHost), hipSuccess);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (memcmp(comm->hostCountersBlock, &counters, sizeof(counters)) != 0 &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    bool mirrored = memcmp(comm->hostCountersBlock, &counters, sizeof(counters)) == 0;
    ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);

    // Send/recv counts in the synthetic P2P slot, once per op, on the work's first channel.
    constexpr int p2p = NCCL_PROGRESS_P2P_COUNTER_INDEX;
    EXPECT_EQ(counters.completedWorkCount[p2p], uint64_t(kOps)) << "P2P completions not counted";
    EXPECT_NE(counters.completedTimeNs[p2p], 0u) << "P2P completion not timestamped";
    for (int slot = 0; slot < p2p; slot++) EXPECT_EQ(counters.completedWorkCount[slot], 0u) << "slot " << slot;
    for (int c = 0; c < MAXCHANNELS; c++) EXPECT_EQ(counters.collOpActive[c], 0u) << "channel " << c << " active";
    EXPECT_TRUE(mirrored) << "the monitor did not copy the device counters into hostCountersBlock";
  }, {{"NCCL_RAS_ENABLE", "1"}, {"NCCL_PROGRESS_COUNTERS", "1"}, {"NCCL_PROGRESS_COUNTER_MONITOR_POLL_MS", "50"}});
}

// The same work with NCCL_PROGRESS_COUNTERS unset: counters are opt-in, so the communicator allocates
// none and the kernel runs without them.
TEST(RasProgressCounters, DisabledByDefault) {
  RUN_ISOLATED_TEST_WITH_ENV("RasProgressCounters.DisabledByDefault", []() {
    int nGpus = 0;
    if (hipGetDeviceCount(&nGpus) != hipSuccess || nGpus < 1) GTEST_SKIP() << "requires at least one GPU";
    ncclComm_t comm = nullptr;
    ASSERT_NO_FATAL_FAILURE(sendRecvToSelf(&comm));
    EXPECT_EQ(comm->deviceCountersBlock, nullptr);
    EXPECT_EQ(comm->hostCountersBlock, nullptr);
    ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
  }, {{"NCCL_RAS_ENABLE", "1"}});
}

} // namespace RcclUnitTesting
