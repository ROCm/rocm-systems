/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Host-only microtests for ncclTransportPatConnect and ncclTransportRingConnect (src/transport/generic.cc),
// compiled in via GENERIC_CC_PATH. ncclTransportP2pSetup is a blocking exchange
// with the rank's peers, so a rank that enters it while another rank skips it
// hangs (NVIDIA/nccl#2385).

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include GENERIC_CC_PATH

namespace {
int fakePatEnable;
bool fakeShmEligible;
int p2pSetupCalls;
std::vector<int> connectConnIndices;
std::vector<std::pair<int, int>> setupSpecificCalls;  // (connIndex, transport)
}  // namespace

int ncclPatEnable(struct ncclComm*) { return fakePatEnable; }
bool rcclPolicyCanonicalShmEligible(const struct ncclComm*) { return fakeShmEligible; }
int64_t ncclParamLocalRegister() { return 0; }
int64_t ncclParamGraphRegister() { return 0; }
ncclResult_t bootstrapAllGather(void*, void*, int) { return ncclInternalError; }
ncclResult_t ncclTransportP2pConnect(struct ncclComm*, int, int, int*, int, int*, int connIndex) {
  connectConnIndices.push_back(connIndex);
  return ncclSuccess;
}
ncclResult_t ncclTransportP2pSetup(struct ncclComm*, struct ncclTopoGraph*, int, bool*) {
  p2pSetupCalls++;
  return ncclSuccess;
}
ncclResult_t ncclTransportP2pSetupSpecific(struct ncclComm*, struct ncclTopoGraph*, int connIndex, bool*,
                                           int transport) {
  setupSpecificCalls.emplace_back(connIndex, transport);
  return ncclSuccess;
}

uint32_t ncclDebugLevelMask = 0;  // read by debug.h
uint64_t ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

namespace {

// Runs ncclTransportPatConnect on every rank of a communicator whose node n has
// localRanksPerNode[n] ranks and returns each rank's ncclTransportP2pSetup calls.
// NVLS is usable, and nvls.nHeads is comm-wide as in production: the smallest
// node's head count (graph/connect.cc). So a node passes the NVLS check only if
// it has that many local ranks.
std::vector<int> setupCallsPerRank(const std::vector<int>& localRanksPerNode, int patEnable) {
  fakePatEnable = patEnable;
  auto minMax = std::minmax_element(localRanksPerNode.begin(), localRanksPerNode.end());
  // Sized for every node at the largest local-rank count, so peer lookups stay in bounds.
  std::vector<int> denseToUserRank(localRanksPerNode.size() * *minMax.second, 0);
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
      comm->nvlsSupport = 1;
      comm->nvlsChannels = 2;
      comm->channels[0].nvls.nHeads = comm->minLocalRanks;
      comm->channels[0].nvls.headRank = i;
      comm->denseToUserRank = denseToUserRank.data();
      p2pSetupCalls = 0;
      EXPECT_EQ(ncclSuccess, ncclTransportPatConnect(comm.get()));
      calls.push_back(p2pSetupCalls);
    }
  }
  return calls;
}

}  // namespace

TEST(GenericPatConnectMicrotest, OneRankPerNodeSetsUpEveryRank) {
  for (int calls : setupCallsPerRank({1, 1}, /*patEnable=*/1)) EXPECT_GT(calls, 0);
}

// Without the uneven-local-ranks guard, only the one-rank node passes the NVLS
// check, so its rank alone would enter setup ({0, 0, 1}): the hang.
TEST(GenericPatConnectMicrotest, UnevenLocalRanksSkipSetupOnEveryRank) {
  EXPECT_EQ(std::vector<int>({0, 0, 0}), setupCallsPerRank({2, 1}, /*patEnable=*/1));
}

// The guard above must be about unevenness: an even multi-rank layout still
// sets up every rank.
TEST(GenericPatConnectMicrotest, EvenMultiRankPerNodeSetsUpEveryRank) {
  for (int calls : setupCallsPerRank({2, 2}, /*patEnable=*/1)) EXPECT_GT(calls, 0);
}

TEST(GenericPatConnectMicrotest, PatDisabledSkipsSetupOnEveryRank) {
  EXPECT_EQ(std::vector<int>({0, 0}), setupCallsPerRank({1, 1}, /*patEnable=*/0));
}

namespace {

// Runs ncclTransportRingConnect on a two-rank, two-channel communicator.
void ringConnect(bool shmEligible) {
  fakeShmEligible = shmEligible;
  p2pSetupCalls = 0;
  connectConnIndices.clear();
  setupSpecificCalls.clear();
  auto comm = std::make_unique<ncclComm>();
  comm->nRanks = 2;
  comm->nChannels = 2;
  EXPECT_EQ(ncclSuccess, ncclTransportRingConnect(comm.get()));
}

}  // namespace

// The SHM decision comes from all-gathered peer info, so every rank takes the same arm;
// an ineligible communicator never touches the SHM connector.
TEST(GenericRingConnectMicrotest, ShmIneligibleConnectsOnlyTheDefaultConnector) {
  ringConnect(/*shmEligible=*/false);
  EXPECT_EQ(1, p2pSetupCalls);
  EXPECT_EQ(std::vector<int>({0, 0}), connectConnIndices);
  EXPECT_TRUE(setupSpecificCalls.empty());
}

TEST(GenericRingConnectMicrotest, ShmEligibleAlsoProvisionsTheShmConnectorOnEveryChannel) {
  ringConnect(/*shmEligible=*/true);
  EXPECT_EQ(1, p2pSetupCalls);
  EXPECT_EQ(std::vector<int>({0, 0, RCCL_CONN_IDX_COLL_SHM, RCCL_CONN_IDX_COLL_SHM}), connectConnIndices);
  EXPECT_EQ((std::vector<std::pair<int, int>>{{RCCL_CONN_IDX_COLL_SHM, TRANSPORT_SHM}}), setupSpecificCalls);
}
