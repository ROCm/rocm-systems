/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Host-only microtests for ncclTransportPatConnect (src/transport/generic.cc),
// compiled in via GENERIC_CC_PATH. ncclTransportP2pSetup is a blocking exchange
// with the rank's peers, so a rank that enters it while another rank skips it
// hangs (NVIDIA/nccl#2385).

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include GENERIC_CC_PATH

namespace {
int fakePatEnable;
int p2pSetupCalls;
}  // namespace

int ncclPatEnable(struct ncclComm*) { return fakePatEnable; }
ncclResult_t ncclTransportP2pConnect(struct ncclComm*, int, int, int*, int, int*, int) { return ncclSuccess; }
ncclResult_t ncclTransportP2pSetup(struct ncclComm*, struct ncclTopoGraph*, int, bool*) {
  p2pSetupCalls++;
  return ncclSuccess;
}

int ncclDebugLevel = 0;           // read by debug.h before NCCL 2.32.3
uint32_t ncclDebugLevelMask = 0;  // read by debug.h from NCCL 2.32.3
uint64_t ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

namespace {

// Runs ncclTransportPatConnect on every rank of a communicator whose node n has
// localRanksPerNode[n] ranks and returns each rank's ncclTransportP2pSetup calls.
std::vector<int> setupCallsPerRank(const std::vector<int>& localRanksPerNode, int patEnable) {
  fakePatEnable = patEnable;
  auto minMax = std::minmax_element(localRanksPerNode.begin(), localRanksPerNode.end());
  std::vector<int> calls;
  for (int node = 0; node < (int)localRanksPerNode.size(); node++) {
    for (int i = 0; i < localRanksPerNode[node]; i++) {
      auto comm = std::make_unique<ncclComm>();
      comm->nRanks = std::accumulate(localRanksPerNode.begin(), localRanksPerNode.end(), 0);
      comm->node = node;
      comm->nNodes = (int)localRanksPerNode.size();
      comm->localRanks = localRanksPerNode[node];
      comm->minLocalRanks = *minMax.first;
      comm->maxLocalRanks = *minMax.second;
      comm->isOneRPN = comm->maxLocalRanks == 1;
      comm->nChannels = 2;
      p2pSetupCalls = 0;
      EXPECT_EQ(ncclSuccess, ncclTransportPatConnect(comm.get()));
      calls.push_back(p2pSetupCalls);
    }
  }
  return calls;
}

}  // namespace

TEST(PatConnectMicrotest, OneRankPerNodeSetsUpEveryRank) {
  for (int calls : setupCallsPerRank({1, 1}, /*patEnable=*/1)) EXPECT_GT(calls, 0);
}

// Without NVLS a multi-rank node never connects PAT, so the layout needs a
// single-rank node for a rank that would.
TEST(PatConnectMicrotest, UnevenLocalRanksSkipSetupOnEveryRank) {
  EXPECT_EQ(std::vector<int>({0, 0, 0}), setupCallsPerRank({2, 1}, /*patEnable=*/1));
}

TEST(PatConnectMicrotest, PatDisabledSkipsSetupOnEveryRank) {
  EXPECT_EQ(std::vector<int>({0, 0}), setupCallsPerRank({1, 1}, /*patEnable=*/0));
}
