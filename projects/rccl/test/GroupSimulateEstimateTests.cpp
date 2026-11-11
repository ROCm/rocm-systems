/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
// ncclGroupSimulateEnd must return a positive estimatedTime for the collectives
// PyTorch's time estimator issues: ncclGroupStart, one collective, ncclGroupSimulateEnd,
// nothing launched on the device. On a single 8-GPU MI355 node the built-in CSV tuner
// reports 0 for small and medium AllReduce, and Direct AllGather leaves the
// NCCL_SIM_INFO_INITIALIZER value of -1 because that path never costs the collective.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

#include "EnvVars.hpp"
#include "ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {

namespace {

bool parentShouldSkip(int minGpus) {
  if (std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) != nullptr) return false;
  return EnvVars().maxGpus < minGpus;
}

enum class CollOp { kAllReduce, kAllGather, kReduceScatter };

struct SimCase {
  CollOp op;
  size_t count;
  bool maxCtas;
};

const char* collName(CollOp op) {
  switch (op) {
    case CollOp::kAllReduce:
      return "AllReduce";
    case CollOp::kAllGather:
      return "AllGather";
    case CollOp::kReduceScatter:
      return "ReduceScatter";
  }
  return "?";
}

// One rank of a single-process clique, matching the PyTorch estimator: the group holds
// exactly one collective, and GroupSimulateEnd does not launch it.
void expectPositive(ncclComm_t comm, int rank, int nRanks, const SimCase& simCase) {
  ASSERT_EQ(hipSetDevice(rank), hipSuccess);
  float* buf = nullptr;
  hipStream_t stream = nullptr;
  ASSERT_EQ(hipMalloc(&buf, simCase.count * static_cast<size_t>(nRanks) * sizeof(float)), hipSuccess);
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

  ncclCollConfig_t cfg = NCCL_COLLCONFIG_INITIALIZER;
  if (simCase.maxCtas) cfg.maxCTAs = 2;

  ncclResult_t rc = ncclGroupStart();
  if (rc == ncclSuccess) {
    if (simCase.op == CollOp::kAllReduce) {
      rc = simCase.maxCtas ? ncclAllReduceConfig(buf, buf, simCase.count, ncclFloat, ncclSum, comm, stream, &cfg)
                           : ncclAllReduce(buf, buf, simCase.count, ncclFloat, ncclSum, comm, stream);
    } else if (simCase.op == CollOp::kAllGather) {
      rc = simCase.maxCtas ? ncclAllGatherConfig(buf, buf, simCase.count, ncclFloat, comm, stream, &cfg)
                           : ncclAllGather(buf, buf, simCase.count, ncclFloat, comm, stream);
    } else {
      rc = simCase.maxCtas ? ncclReduceScatterConfig(buf, buf, simCase.count, ncclFloat, ncclSum, comm, stream, &cfg)
                           : ncclReduceScatter(buf, buf, simCase.count, ncclFloat, ncclSum, comm, stream);
    }
  }
  ncclSimInfo_t info = NCCL_SIM_INFO_INITIALIZER;
  if (rc == ncclSuccess) rc = ncclGroupSimulateEnd(&info);

  EXPECT_EQ(rc, ncclSuccess) << collName(simCase.op) << " count=" << simCase.count
                             << " maxCTAs=" << (simCase.maxCtas ? 2 : 0) << " rank=" << rank;
  // 0 is the CSV tuner's "force this algo" sentinel, and -1 is "no estimate".
  // Neither is a usable duration.
  EXPECT_GT(info.estimatedTime, 0.f) << "ngpus=" << nRanks << " " << collName(simCase.op) << " count=" << simCase.count
                                     << " maxCTAs=" << (simCase.maxCtas ? 2 : 0) << " rank=" << rank
                                     << " estimatedTime=" << info.estimatedTime;

  EXPECT_EQ(hipFree(buf), hipSuccess);
  EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

void expectPositiveEstimates(int nRanks) {
  int deviceCount = 0;
  ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
  ASSERT_GE(deviceCount, nRanks);

  std::vector<int> devs(nRanks);
  for (int i = 0; i < nRanks; ++i) devs[i] = i;
  std::vector<ncclComm_t> comms(nRanks, nullptr);
  ASSERT_EQ(ncclCommInitAll(comms.data(), nRanks, devs.data()), ncclSuccess);

  // The 4 KiB and 1 MiB rows are the defect. The 64 MiB rows, maxCTAs=2 AllGather,
  // and every ReduceScatter already return a positive time and must stay that way.
  const SimCase cases[] = {
    {CollOp::kAllReduce, 1024, false},      {CollOp::kAllReduce, 262144, false},
    {CollOp::kAllReduce, 16777216, false},  {CollOp::kAllReduce, 1024, true},
    {CollOp::kAllGather, 1024, false},      {CollOp::kAllGather, 262144, false},
    {CollOp::kAllGather, 16777216, false},  {CollOp::kAllGather, 1024, true},
    {CollOp::kReduceScatter, 1024, false},  {CollOp::kReduceScatter, 262144, false},
    {CollOp::kReduceScatter, 16777216, false}, {CollOp::kReduceScatter, 1024, true},
  };

  for (const SimCase& simCase : cases) {
    for (int rank = 0; rank < nRanks; ++rank) {
      expectPositive(comms[rank], rank, nRanks, simCase);
    }
  }

  for (int rank = 0; rank < nRanks; ++rank) {
    if (comms[rank]) EXPECT_EQ(ncclCommDestroy(comms[rank]), ncclSuccess);
  }
}

}  // namespace

TEST(GroupSimulateEstimate, TwoGpuCollectivesReportPositiveTime) {
  if (parentShouldSkip(2)) GTEST_SKIP() << "Requires 2 GPUs";
  RUN_ISOLATED_TESTS(ProcessIsolatedTestRunner::TestConfig("GroupSimulateEstimate_TwoGpu",
                                                           []() { expectPositiveEstimates(2); })
                       .withTimeout(std::chrono::seconds(600)));
}

TEST(GroupSimulateEstimate, EightGpuCollectivesReportPositiveTime) {
  if (parentShouldSkip(8)) GTEST_SKIP() << "Requires 8 GPUs";
  RUN_ISOLATED_TESTS(ProcessIsolatedTestRunner::TestConfig("GroupSimulateEstimate_EightGpu",
                                                           []() { expectPositiveEstimates(8); })
                       .withTimeout(std::chrono::seconds(600)));
}

}  // namespace RcclUnitTesting
