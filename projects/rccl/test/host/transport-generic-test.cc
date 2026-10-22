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
#include <tuple>
#include <vector>

#include GENERIC_CC_PATH

namespace {
int fakePatEnable;

struct TransportCall {
  bool setup;     // ncclTransportP2pSetup; otherwise ncclTransportP2pConnect
  int channel;    // connect only
  int recvPeer;   // connect only
  int sendPeer;   // connect only
  int graphAlgo;  // setup only: the comm->graphs[] entry it was given
};
std::vector<TransportCall> transportCalls;
}  // namespace

int ncclPatEnable(struct ncclComm*) { return fakePatEnable; }
ncclResult_t ncclTransportP2pConnect(struct ncclComm*, int channel, int nrecv, int* peerRecv, int nsend,
                                     int* peerSend, int) {
  transportCalls.push_back({false, channel, nrecv > 0 ? peerRecv[0] : -1, nsend > 0 ? peerSend[0] : -1, -1});
  return ncclSuccess;
}
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int, bool*) {
  transportCalls.push_back({true, -1, -1, -1, static_cast<int>(graph - comm->graphs)});
  return ncclSuccess;
}

uint32_t ncclDebugLevelMask = 0;  // read by debug.h
uint64_t ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

namespace {

// Different counts, so a test can tell which one a rank connected over.
constexpr int kChannels = 3;
constexpr int kNvlsChannels = 2;

// Runs ncclTransportPatConnect on every rank of a communicator whose node n has
// localRanksPerNode[n] ranks and returns each rank's transport calls in order.
// NVLS is usable, and nvls.nHeads is comm-wide as in production: the smallest
// node's head count (graph/connect.cc). So a node passes the NVLS check only if
// it has that many local ranks.
std::vector<std::vector<TransportCall>> transportCallsPerRank(const std::vector<int>& localRanksPerNode,
                                                              int patEnable, bool patSharedQps = false) {
  fakePatEnable = patEnable;
  auto minMax = std::minmax_element(localRanksPerNode.begin(), localRanksPerNode.end());
  // Sized for every node at the largest local-rank count, so peer lookups stay in bounds.
  std::vector<int> denseToUserRank(localRanksPerNode.size() * *minMax.second, 0);
  std::vector<std::vector<TransportCall>> calls;
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
      comm->patSharedQps = patSharedQps;
      comm->nChannels = kChannels;
      comm->nvlsSupport = 1;
      comm->nvlsChannels = kNvlsChannels;
      comm->channels[0].nvls.nHeads = comm->minLocalRanks;
      comm->channels[0].nvls.headRank = i;
      comm->denseToUserRank = denseToUserRank.data();
      transportCalls.clear();
      EXPECT_EQ(ncclSuccess, ncclTransportPatConnect(comm.get()));
      calls.push_back(transportCalls);
    }
  }
  return calls;
}

int setupCount(const std::vector<TransportCall>& calls) {
  return (int)std::count_if(calls.begin(), calls.end(), [](const TransportCall& call) { return call.setup; });
}

std::vector<int> setupCallsPerRank(const std::vector<int>& localRanksPerNode, int patEnable) {
  std::vector<int> setups;
  for (const auto& calls : transportCallsPerRank(localRanksPerNode, patEnable)) setups.push_back(setupCount(calls));
  return setups;
}

// (channel, recv peer, send peer) of every connect call, sorted so the call order does not matter.
std::vector<std::tuple<int, int, int>> sortedConnections(const std::vector<TransportCall>& calls) {
  std::vector<std::tuple<int, int, int>> connections;
  for (const TransportCall& call : calls) {
    if (!call.setup) connections.emplace_back(call.channel, call.recvPeer, call.sendPeer);
  }
  std::sort(connections.begin(), connections.end());
  return connections;
}

// Every connection is queued before the rank's one setup, which runs on the given graph.
void expectOneSetupAfterEveryConnection(const std::vector<TransportCall>& calls, int graphAlgo) {
  ASSERT_FALSE(calls.empty());
  EXPECT_EQ(1, setupCount(calls));
  EXPECT_TRUE(calls.back().setup);
  EXPECT_EQ(graphAlgo, calls.back().graphAlgo);
}

}  // namespace

TEST(PatConnectMicrotest, OneRankPerNodeSetsUpEveryRank) {
  for (int calls : setupCallsPerRank({1, 1}, /*patEnable=*/1)) EXPECT_GT(calls, 0);
}

// Without the uneven-local-ranks guard, only the one-rank node passes the NVLS
// check, so its rank alone would enter setup ({0, 0, 1}): the hang.
TEST(PatConnectMicrotest, UnevenLocalRanksSkipSetupOnEveryRank) {
  EXPECT_EQ(std::vector<int>({0, 0, 0}), setupCallsPerRank({2, 1}, /*patEnable=*/1));
}

// The guard above must be about unevenness: an even multi-rank layout still
// sets up every rank.
TEST(PatConnectMicrotest, EvenMultiRankPerNodeSetsUpEveryRank) {
  for (int calls : setupCallsPerRank({2, 2}, /*patEnable=*/1)) EXPECT_GT(calls, 0);
}

TEST(PatConnectMicrotest, PatDisabledSkipsSetupOnEveryRank) {
  EXPECT_EQ(std::vector<int>({0, 0}), setupCallsPerRank({1, 1}, /*patEnable=*/0));
}

// Each binomial step's ReduceScatter and AllGather connections are queued, and
// one setup per rank connects them all, rather than one per step and direction.
TEST(PatConnectMicrotest, OneRankPerNodeSetsUpOnceAfterEveryConnection) {
  for (const auto& calls : transportCallsPerRank({1, 1, 1, 1}, /*patEnable=*/1)) {
    expectOneSetupAfterEveryConnection(calls, NCCL_ALGO_TREE);
  }
}

TEST(PatConnectMicrotest, SharedQpsSetUpOnceAfterEveryConnection) {
  for (const auto& calls : transportCallsPerRank({1, 1, 1, 1}, /*patEnable=*/1, /*patSharedQps=*/true)) {
    expectOneSetupAfterEveryConnection(calls, NCCL_ALGO_TREE);
  }
}

// Multi-rank nodes connect across nodes on the NVLS channels and set up on the
// NVLS graph, as NVLS_TREE does.
TEST(PatConnectMicrotest, MultiRankPerNodeSetsUpOnceOnTheNvlsGraph) {
  for (const auto& calls : transportCallsPerRank({2, 2, 2, 2}, /*patEnable=*/1)) {
    expectOneSetupAfterEveryConnection(calls, NCCL_ALGO_NVLS);
    // Two binomial steps, each with a ReduceScatter and an AllGather connection per NVLS channel.
    EXPECT_EQ(2 * 2 * kNvlsChannels, (int)sortedConnections(calls).size());
    for (const TransportCall& call : calls) {
      if (!call.setup) EXPECT_LT(call.channel, kNvlsChannels);
    }
  }
}

// However the setups are batched, a rank connects to each binomial peer in both
// directions on every channel.
TEST(PatConnectMicrotest, OneRankPerNodeConnectsBothWaysToEveryBinomialPeer) {
  const int nNodes = 8;
  const auto callsPerRank = transportCallsPerRank(std::vector<int>(nNodes, 1), /*patEnable=*/1);
  ASSERT_EQ(nNodes, (int)callsPerRank.size());
  for (int rank = 0; rank < nNodes; rank++) {
    std::vector<std::tuple<int, int, int>> expected;
    for (int mask = 1; mask < nNodes; mask <<= 1) {
      const int prevPeer = (rank + mask) % nNodes;
      const int nextPeer = (rank + nNodes - mask) % nNodes;
      for (int c = 0; c < kChannels; c++) {
        expected.emplace_back(c, prevPeer, nextPeer);  // ReduceScatter
        expected.emplace_back(c, nextPeer, prevPeer);  // AllGather
      }
    }
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(expected, sortedConnections(callsPerRank[rank])) << "rank " << rank;
  }
}

// A single node has no binomial peers, so no rank sets up.
TEST(PatConnectMicrotest, SingleNodeNeverSetsUp) {
  EXPECT_EQ(std::vector<int>({0, 0}), setupCallsPerRank({2}, /*patEnable=*/1));
}
