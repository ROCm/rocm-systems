/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/hier_host.cc: building the child comms when a step fails on this rank or a peer,
// the vote after each step, and teardown on the abort and revoke paths.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/comm_fakes.h"
#include "fakes/nccl_stubs.h"

#include HIER_HOST_CC_PATH

namespace {
std::vector<ncclComm_t> g_aborted;
}  // namespace

// The externals of hier_host.cc that no shared fake covers. The stages never run here.
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  g_aborted.push_back(comm);
  return ncclSuccess;
}
ncclResult_t ncclReduceScatter(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, hipStream_t) {
  ::abort();
}
ncclResult_t ncclAllReduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, hipStream_t) {
  ::abort();
}
ncclResult_t ncclAllGather(const void*, void*, size_t, ncclDataType_t, ncclComm_t, hipStream_t) { ::abort(); }
const char* ncclGetErrorString(ncclResult_t) { return "faked error"; }

namespace {

// Rank 0 of two hosts with two ranks each.
struct HierComm {
  std::unique_ptr<ncclComm> storage = std::make_unique<ncclComm>();
  ncclComm& comm = *storage;
  ncclPeerInfo peers[4]{};
  uint32_t abortFlag = 0;
  int bootstrapState = 0;

  HierComm() {
    std::memset(&comm, 0, sizeof(comm));
    comm.nRanks = 4;
    comm.rank = 0;
    for (int r = 0; r < 4; r++) peers[r].hostHash = 0x5eed0000u + r / 2;
    comm.peerInfo = peers;
    comm.peerInfoValid = true;
    comm.bootstrap = &bootstrapState;
    comm.abortFlag = &abortFlag;
    comm.hostHierState = rcclHostHierUnprobed;
  }
  ncclComm* get() { return &comm; }
};

// A child comm handed out by the faked ncclCommSplit. One rank, so initChildFabric skips it.
struct Children {
  std::vector<std::unique_ptr<ncclComm>> comms;
  ncclComm_t next() {
    comms.push_back(std::make_unique<ncclComm>());
    std::memset(comms.back().get(), 0, sizeof(ncclComm));
    comms.back()->nRanks = 1;
    return comms.back().get();
  }
};

// Votes: each call fills every peer's slot as ready, except the peer listed for that vote.
struct Votes {
  std::vector<int> failingPeer;  // per vote, -1 when every peer is ready
  int calls = 0;
};

auto VoteHook(Votes* votes) {
  return [votes](void*, void* allData, int size) {
    uint8_t* ready = static_cast<uint8_t*>(allData);
    const int vote = votes->calls++;
    const int failing = vote < static_cast<int>(votes->failingPeer.size()) ? votes->failingPeer[vote] : -1;
    for (int r = 1; r < 4; r++) ready[r * size] = r == failing ? 0 : 1;
    return ncclSuccess;
  };
}

auto SplitHook(Children* children, int* calls, int failOnCall) {
  return [children, calls, failOnCall](ncclComm_t, int, int, ncclComm_t* newcomm, ncclConfig_t*) {
    if ((*calls)++ == failOnCall) {
      *newcomm = nullptr;
      return ncclSystemError;
    }
    *newcomm = children->next();
    return ncclSuccess;
  };
}

class HostHierEnsureTest : public ::testing::Test {
 protected:
  void SetUp() override { g_aborted.clear(); }
  void TearDown() override { g_aborted.clear(); }
};

TEST_F(HostHierEnsureTest, SplitFailingHereStopsAfterTheFirstVote) {
  HierComm hc;
  Children children;
  Votes votes;
  int splits = 0;
  ScopedHook split(g_ncclCommSplit, SplitHook(&children, &splits, /*failOnCall=*/0));
  ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
  EXPECT_EQ(ncclSystemError, rcclHostHierEnsure(hc.get()));
  EXPECT_EQ(rcclHostHierFailed, hc.comm.hostHierState);
  EXPECT_EQ(1, splits) << "the inter-host split must not start";
  EXPECT_EQ(1, votes.calls);
  EXPECT_EQ(nullptr, hc.comm.hostHier->intra);
  rcclHostHierTeardown(hc.get());
}

// A peer failing its intra-host split makes this rank fall back too, before any other collective.
TEST_F(HostHierEnsureTest, PeerFailingIntraSplitAbortsTheChild) {
  HierComm hc;
  Children children;
  Votes votes{{/*vote 0*/ 2}};
  int splits = 0;
  int destroys = 0;
  ScopedHook split(g_ncclCommSplit, SplitHook(&children, &splits, /*failOnCall=*/-1));
  ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
  ScopedHook destroy(g_ncclCommDestroy, [&destroys](ncclComm_t) {
    destroys++;
    return ncclSuccess;
  });
  EXPECT_EQ(ncclRemoteError, rcclHostHierEnsure(hc.get()));
  EXPECT_EQ(rcclHostHierFailed, hc.comm.hostHierState);
  EXPECT_EQ(1, splits);
  EXPECT_EQ(1, votes.calls);
  ASSERT_EQ(1u, g_aborted.size());
  EXPECT_EQ(children.comms[0].get(), g_aborted[0]);
  EXPECT_EQ(0, destroys) << "a peer may be gone, so the child is aborted, not destroyed";
  rcclHostHierTeardown(hc.get());
}

TEST_F(HostHierEnsureTest, PeerFailingInterSplitAbortsBothChildren) {
  HierComm hc;
  Children children;
  Votes votes{{-1, /*vote 1*/ 3}};
  int splits = 0;
  ScopedHook split(g_ncclCommSplit, SplitHook(&children, &splits, /*failOnCall=*/-1));
  ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
  EXPECT_EQ(ncclRemoteError, rcclHostHierEnsure(hc.get()));
  EXPECT_EQ(rcclHostHierFailed, hc.comm.hostHierState);
  EXPECT_EQ(2, splits);
  EXPECT_EQ(2, votes.calls);
  EXPECT_EQ(2u, g_aborted.size());
  rcclHostHierTeardown(hc.get());
}

TEST_F(HostHierEnsureTest, ChildNotReadyHereFailsTheVote) {
  HierComm hc;
  Children children;
  Votes votes;
  int splits = 0;
  ScopedHook split(g_ncclCommSplit, SplitHook(&children, &splits, /*failOnCall=*/-1));
  ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
  g_commEnsureReadyResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, rcclHostHierEnsure(hc.get()));
  g_commEnsureReadyResult = ncclSuccess;
  EXPECT_EQ(rcclHostHierFailed, hc.comm.hostHierState);
  EXPECT_EQ(1, votes.calls);
  EXPECT_EQ(1u, g_aborted.size());
  rcclHostHierTeardown(hc.get());
}

// Every step votes once: the two splits and the two fabric inits.
TEST_F(HostHierEnsureTest, SuccessVotesAfterEveryStep) {
  HierComm hc;
  Children children;
  Votes votes;
  int splits = 0;
  ScopedHook split(g_ncclCommSplit, SplitHook(&children, &splits, /*failOnCall=*/-1));
  ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
  EXPECT_EQ(ncclSuccess, rcclHostHierEnsure(hc.get()));
  EXPECT_EQ(rcclHostHierReady, hc.comm.hostHierState);
  EXPECT_EQ(2, splits);
  EXPECT_EQ(4, votes.calls);
  EXPECT_EQ(rcclHostHierOff, hc.comm.hostHier->intra->hostHierState) << "children never split again";
  EXPECT_EQ(rcclHostHierOff, hc.comm.hostHier->inter->hostHierState);
  EXPECT_EQ(ncclSuccess, rcclHostHierEnsure(hc.get())) << "a ready comm does not vote again";
  EXPECT_EQ(4, votes.calls);
  rcclHostHierTeardown(hc.get());
}

// Builds ready children, then tears down with the given flags.
void TeardownWith(bool abortFlag, bool revoked, int* destroys) {
  HierComm hc;
  Children children;
  Votes votes;
  int splits = 0;
  {
    ScopedHook split(g_ncclCommSplit, SplitHook(&children, &splits, /*failOnCall=*/-1));
    ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
    ASSERT_EQ(ncclSuccess, rcclHostHierEnsure(hc.get()));
  }
  ScopedHook destroy(g_ncclCommDestroy, [destroys](ncclComm_t) {
    (*destroys)++;
    return ncclSuccess;
  });
  hc.abortFlag = abortFlag ? 1 : 0;
  hc.comm.revokedFlag = revoked ? 1 : 0;
  EXPECT_EQ(ncclSuccess, rcclHostHierTeardown(hc.get()));
  EXPECT_EQ(nullptr, hc.comm.hostHier);
}

TEST_F(HostHierEnsureTest, TeardownDestroysChildren) {
  int destroys = 0;
  TeardownWith(/*abortFlag=*/false, /*revoked=*/false, &destroys);
  EXPECT_EQ(2, destroys);
  EXPECT_TRUE(g_aborted.empty());
}

TEST_F(HostHierEnsureTest, TeardownAbortsChildrenWhenAborting) {
  int destroys = 0;
  TeardownWith(/*abortFlag=*/true, /*revoked=*/false, &destroys);
  EXPECT_EQ(0, destroys);
  EXPECT_EQ(2u, g_aborted.size());
}

// A revoke resets abortFlag but leaves revokedFlag set.
TEST_F(HostHierEnsureTest, TeardownAbortsChildrenWhenRevoked) {
  int destroys = 0;
  TeardownWith(/*abortFlag=*/false, /*revoked=*/true, &destroys);
  EXPECT_EQ(0, destroys);
  EXPECT_EQ(2u, g_aborted.size());
}

}  // namespace
