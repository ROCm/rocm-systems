/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/hier_host.cc: building the child comms when a step fails on this rank or a peer,
// the vote after each step, and teardown on the abort and revoke paths.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ScopedHook.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/comm_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_stubs.h"

#include HIER_HOST_CC_PATH

namespace {
std::vector<ncclComm_t> g_aborted;
// The comm whose children ncclCommAbort checks, and how many children it still listed at each abort.
ncclComm* g_abortParent = nullptr;
std::vector<int> g_childrenListedAtAbort;
// The three stages of rcclHostHierAllReduce, and the parent votes and scratch calls some tests log alongside them.
std::vector<std::string> g_log;
}  // namespace

// The externals of hier_host.cc that no shared fake covers.
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  g_aborted.push_back(comm);
  if (g_abortParent != nullptr) {
    ncclComm_t listed[2];
    g_childrenListedAtAbort.push_back(rcclHostHierChildren(g_abortParent, listed));
  }
  return ncclSuccess;
}
ncclResult_t ncclReduceScatter(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, hipStream_t) {
  g_log.push_back("ReduceScatter");
  return ncclSuccess;
}
ncclResult_t ncclAllReduce(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, hipStream_t) {
  g_log.push_back("AllReduce");
  return ncclSuccess;
}
ncclResult_t ncclAllGather(const void*, void*, size_t, ncclDataType_t, ncclComm_t, hipStream_t) {
  g_log.push_back("AllGather");
  return ncclSuccess;
}
// The mock comm has no DDA fabric, so the fused kernel never runs.
bool ncclAllReduceDdaFabricHierEligible(ncclComm*, size_t, ncclDataType_t, ncclRedOp_t, int, int) { return false; }
ncclResult_t ncclAllReduceDdaFabricHier(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t, int, int,
                                        int, int, const int*, const int*) {
  ::abort();
}
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

// A child comm handed out by the faked ncclCommSplit. With one rank, initChildFabric skips it.
struct Children {
  int ranks = 1;
  std::vector<std::unique_ptr<ncclComm>> comms;
  ncclComm_t next() {
    comms.push_back(std::make_unique<ncclComm>());
    std::memset(comms.back().get(), 0, sizeof(ncclComm));
    comms.back()->nRanks = ranks;
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
  void SetUp() override {
    g_aborted.clear();
    g_log.clear();
    g_hipEventCreateResult = hipSuccess;
    g_hipAsyncOpsResult = hipSuccess;  // stream capture mode exchange and event records on the scratch path
    g_hipEventSynchronize = [](hipEvent_t) { return hipSuccess; };
    // ncclCudaFree looks up the allocation's base before freeing it.
    g_hipMemGetAddressRange = [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      *base = ptr;
      *size = 0;
      return hipSuccess;
    };
  }
  void TearDown() override {
    g_aborted.clear();
    g_abortParent = nullptr;
    g_childrenListedAtAbort.clear();
    g_log.clear();
    ResetHipFakes();
  }
};

// Builds the child comms with every peer ready.
void EnsureReady(HierComm* hc, Children* children) {
  Votes votes;
  int splits = 0;
  ScopedHook split(g_ncclCommSplit, SplitHook(children, &splits, /*failOnCall=*/-1));
  ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
  ASSERT_EQ(ncclSuccess, rcclHostHierEnsure(hc->get()));
}

// Logs each parent vote and fills the peers' slots, with peer `failingPeer` reporting not ready.
auto LoggedVote(int failingPeer) {
  return [failingPeer](void*, void* allData, int size) {
    g_log.push_back("vote");
    uint8_t* ready = static_cast<uint8_t*>(allData);
    for (int r = 1; r < 4; r++) ready[r * size] = r == failingPeer ? 0 : 1;
    return ncclSuccess;
  };
}

// fp32 counts: the scratch holds one chunk per host-local rank, twice with the inter-host stage.
constexpr size_t kSmallCount = 1024;
constexpr size_t kLargeCount = 64 * 1024;

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

// A revoke on another thread sets the children's abort flags through rcclHostHierChildren, so a failed setup must
// unlink each child before it releases it.
TEST_F(HostHierEnsureTest, FailedSetupUnlinksTheChildrenBeforeAbortingThem) {
  HierComm hc;
  Children children;
  Votes votes{{-1, /*vote 1*/ 3}};
  int splits = 0;
  ScopedHook split(g_ncclCommSplit, SplitHook(&children, &splits, /*failOnCall=*/-1));
  ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
  g_abortParent = hc.get();
  EXPECT_EQ(ncclRemoteError, rcclHostHierEnsure(hc.get()));
  ASSERT_EQ(2u, g_aborted.size());
  EXPECT_EQ(std::vector<int>({0, 0}), g_childrenListedAtAbort);
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

// A peer failing the intra-host child's fabric init makes this rank abort both children at that vote.
TEST_F(HostHierEnsureTest, PeerFailingFabricInitAbortsBothChildren) {
  HierComm hc;
  Children children;
  Votes votes{{-1, -1, /*vote 2*/ 2}};
  int splits = 0;
  ScopedHook split(g_ncclCommSplit, SplitHook(&children, &splits, /*failOnCall=*/-1));
  ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
  EXPECT_EQ(ncclRemoteError, rcclHostHierEnsure(hc.get()));
  EXPECT_EQ(rcclHostHierFailed, hc.comm.hostHierState);
  EXPECT_EQ(3, votes.calls) << "the inter-host fabric init must not start";
  EXPECT_EQ(2u, g_aborted.size());
  rcclHostHierTeardown(hc.get());
}

// Two-rank children on the fabric path run DDA fabric init; when it fails here, this rank reports it at the vote.
TEST_F(HostHierEnsureTest, FabricInitFailingHereFailsTheVote) {
  HierComm hc;
  Children children;
  children.ranks = 2;
  Votes votes;
  int splits = 0;
  std::vector<ncclComm_t> inits;
  ScopedHook split(g_ncclCommSplit, SplitHook(&children, &splits, /*failOnCall=*/-1));
  ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
  ScopedHook fabricPath(g_ncclDdaUseFabricPath, [](ncclComm*) { return true; });
  ScopedHook fabricInit(g_ncclDdaFabricCommInit, [&inits](ncclComm* child) {
    inits.push_back(child);
    return ncclSystemError;
  });
  EXPECT_EQ(ncclSystemError, rcclHostHierEnsure(hc.get()));
  EXPECT_EQ(rcclHostHierFailed, hc.comm.hostHierState);
  ASSERT_EQ(1u, inits.size()) << "the inter-host child's fabric init must not start";
  EXPECT_EQ(children.comms[0].get(), inits[0]);
  EXPECT_EQ(3, votes.calls);
  EXPECT_EQ(2u, g_aborted.size());
  rcclHostHierTeardown(hc.get());
}

TEST_F(HostHierEnsureTest, TwoRankChildrenGetFabricInit) {
  HierComm hc;
  Children children;
  children.ranks = 2;
  Votes votes;
  int splits = 0;
  std::vector<ncclComm_t> inits;
  ScopedHook split(g_ncclCommSplit, SplitHook(&children, &splits, /*failOnCall=*/-1));
  ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
  ScopedHook fabricPath(g_ncclDdaUseFabricPath, [](ncclComm*) { return true; });
  ScopedHook fabricInit(g_ncclDdaFabricCommInit, [&inits](ncclComm* child) {
    inits.push_back(child);
    return ncclSuccess;
  });
  EXPECT_EQ(ncclSuccess, rcclHostHierEnsure(hc.get()));
  EXPECT_EQ(rcclHostHierReady, hc.comm.hostHierState);
  ASSERT_EQ(2u, inits.size());
  EXPECT_EQ(hc.comm.hostHier->intra, inits[0]);
  EXPECT_EQ(hc.comm.hostHier->inter, inits[1]);
  EXPECT_EQ(4, votes.calls);
  rcclHostHierTeardown(hc.get());
}

// The splits run blocking and unshared: a child sharing the parent's resources shares its abort flag, which the
// failure path would set. The parent's settings come back on success and on failure.
TEST_F(HostHierEnsureTest, SplitsRunBlockingAndUnsharedThenRestoreTheParent) {
  for (int failOnCall : {-1, 1}) {
    HierComm hc;
    hc.comm.config.blocking = 0;
    hc.comm.config.splitShare = 1;
    Children children;
    Votes votes;
    std::vector<std::pair<int, int>> seen;  // parent's {blocking, splitShare} at each split
    int splits = 0;
    auto inner = SplitHook(&children, &splits, failOnCall);
    ScopedHook split(g_ncclCommSplit, [&seen, inner](ncclComm_t parent, int color, int key, ncclComm_t* newcomm,
                                                     ncclConfig_t* config) {
      seen.emplace_back(parent->config.blocking, parent->config.splitShare);
      return inner(parent, color, key, newcomm, config);
    });
    ScopedHook vote(g_bootstrapAllGather, VoteHook(&votes));
    EXPECT_EQ(failOnCall < 0 ? ncclSuccess : ncclSystemError, rcclHostHierEnsure(hc.get()));
    ASSERT_EQ(2u, seen.size()) << "failOnCall " << failOnCall;
    for (const auto& s : seen) EXPECT_EQ(std::make_pair(1, 0), s) << "failOnCall " << failOnCall;
    EXPECT_EQ(0, hc.comm.config.blocking) << "failOnCall " << failOnCall;
    EXPECT_EQ(1, hc.comm.config.splitShare) << "failOnCall " << failOnCall;
    rcclHostHierTeardown(hc.get());
  }
}

// Every rank grows the scratch on the same call and votes before the first stage; a call that fits does not vote.
TEST_F(HostHierEnsureTest, ScratchGrowthVotesBeforeTheStages) {
  HierComm hc;
  Children children;
  EnsureReady(&hc, &children);
  ScopedHook vote(g_bootstrapAllGather, LoggedVote(/*failingPeer=*/-1));
  const std::vector<std::string> grow = {"vote", "ReduceScatter", "AllReduce", "AllGather"};
  const std::vector<std::string> fits = {"ReduceScatter", "AllReduce", "AllGather"};
  for (const auto& [count, expected] : {std::make_pair(kSmallCount, grow), std::make_pair(kSmallCount, fits),
                                        std::make_pair(kLargeCount, grow)}) {
    g_log.clear();
    ASSERT_EQ(ncclSuccess, rcclHostHierAllReduce(nullptr, nullptr, count, ncclFloat32, ncclSum, hc.get(), nullptr));
    EXPECT_EQ(expected, g_log) << "count " << count;
  }
  rcclHostHierTeardown(hc.get());
}

// After a failed grow vote the ranks may hold scratch of different sizes, so the comm stops using the path for good;
// the scratch is kept until teardown like any other.
void ExpectPathOffAfterFailedGrow(HierComm* hc) {
  EXPECT_EQ(rcclHostHierFailed, hc->comm.hostHierState);
  EXPECT_EQ(nullptr, rcclHostHierGetLayout(hc->get())) << "later calls take the default path";
}

TEST_F(HostHierEnsureTest, PeerFailingScratchGrowthLaunchesNothing) {
  HierComm hc;
  Children children;
  EnsureReady(&hc, &children);
  ScopedHook vote(g_bootstrapAllGather, LoggedVote(/*failingPeer=*/2));
  EXPECT_EQ(ncclRemoteError, rcclHostHierAllReduce(nullptr, nullptr, kSmallCount, ncclFloat32, ncclSum, hc.get(),
                                                   nullptr));
  EXPECT_EQ(std::vector<std::string>{"vote"}, g_log);
  ExpectPathOffAfterFailedGrow(&hc);
  rcclHostHierTeardown(hc.get());
}

TEST_F(HostHierEnsureTest, ScratchAllocationFailingHereLaunchesNothing) {
  HierComm hc;
  Children children;
  EnsureReady(&hc, &children);
  ScopedHook vote(g_bootstrapAllGather, LoggedVote(/*failingPeer=*/-1));
  ScopedHook alloc(g_hipExtMallocWithFlags, [](void**, size_t, unsigned) { return hipErrorOutOfMemory; });
  EXPECT_NE(ncclSuccess, rcclHostHierAllReduce(nullptr, nullptr, kSmallCount, ncclFloat32, ncclSum, hc.get(),
                                               nullptr));
  EXPECT_EQ(std::vector<std::string>{"vote"}, g_log) << "this rank still votes, then launches nothing";
  ExpectPathOffAfterFailedGrow(&hc);
  rcclHostHierTeardown(hc.get());
}

// A peer fails the vote after this rank grew: both buffers stay until teardown.
TEST_F(HostHierEnsureTest, PeerFailingRegrowKeepsTheBuffersUntilTeardown) {
  HierComm hc;
  Children children;
  EnsureReady(&hc, &children);
  {
    ScopedHook vote(g_bootstrapAllGather, LoggedVote(/*failingPeer=*/-1));
    ASSERT_EQ(ncclSuccess,
              rcclHostHierAllReduce(nullptr, nullptr, kSmallCount, ncclFloat32, ncclSum, hc.get(), nullptr));
  }
  ScopedHook vote(g_bootstrapAllGather, LoggedVote(/*failingPeer=*/3));
  ScopedHook release(g_hipFree, [](void* ptr) {
    g_log.push_back("free");
    std::free(ptr);  // DefaultHipExtMallocWithFlags allocates with std::malloc
    return hipSuccess;
  });
  g_log.clear();
  EXPECT_EQ(ncclRemoteError, rcclHostHierAllReduce(nullptr, nullptr, kLargeCount, ncclFloat32, ncclSum, hc.get(),
                                                   nullptr));
  EXPECT_EQ(std::vector<std::string>{"vote"}, g_log) << "no free while the comm lives";
  ExpectPathOffAfterFailedGrow(&hc);
  g_log.clear();
  rcclHostHierTeardown(hc.get());
  EXPECT_EQ((std::vector<std::string>{"free", "free"}), g_log) << "the outgrown and the new scratch";
}

// An outgrown stage buffer is kept until teardown rather than freed while the comm lives.
TEST_F(HostHierEnsureTest, RegrowKeepsTheOldBufferUntilTeardown) {
  HierComm hc;
  Children children;
  EnsureReady(&hc, &children);
  ScopedHook vote(g_bootstrapAllGather, LoggedVote(/*failingPeer=*/-1));
  ScopedHook release(g_hipFree, [](void* ptr) {
    g_log.push_back("free");
    std::free(ptr);  // DefaultHipExtMallocWithFlags allocates with std::malloc
    return hipSuccess;
  });
  ASSERT_EQ(ncclSuccess, rcclHostHierAllReduce(nullptr, nullptr, kSmallCount, ncclFloat32, ncclSum, hc.get(), nullptr));
  void* const first = hc.comm.hostHier->scratch;
  g_log.clear();
  ASSERT_EQ(ncclSuccess, rcclHostHierAllReduce(nullptr, nullptr, kLargeCount, ncclFloat32, ncclSum, hc.get(), nullptr));
  const std::vector<std::string> expected = {"vote", "ReduceScatter", "AllReduce", "AllGather"};
  EXPECT_EQ(expected, g_log) << "no free while the comm lives";
  EXPECT_NE(first, hc.comm.hostHier->scratch);
  ASSERT_EQ(1, hc.comm.hostHier->nRetired);
  EXPECT_EQ(first, hc.comm.hostHier->retired[0]);
  g_log.clear();
  rcclHostHierTeardown(hc.get());
  EXPECT_EQ((std::vector<std::string>{"free", "free"}), g_log) << "the kept and the current buffer";
}

// Buffers that are not 16-byte aligned go to the stages as they are, with no copy and no extra scratch.
TEST_F(HostHierEnsureTest, MisalignedBuffersGoStraightToTheStages) {
  HierComm hc;
  Children children;
  EnsureReady(&hc, &children);
  ScopedHook vote(g_bootstrapAllGather, LoggedVote(/*failingPeer=*/-1));
  ScopedHook copies(g_hipMemcpyAsync, [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) {
    g_log.push_back("copy");
    return hipSuccess;
  });
  ASSERT_EQ(ncclSuccess, rcclHostHierAllReduce(nullptr, nullptr, kSmallCount, ncclFloat32, ncclSum, hc.get(), nullptr));
  const size_t stageBytes = hc.comm.hostHier->scratchBytes;
  g_log.clear();
  void* const misaligned = reinterpret_cast<void*>(0x1004);
  ASSERT_EQ(ncclSuccess,
            rcclHostHierAllReduce(misaligned, misaligned, kSmallCount, ncclFloat32, ncclSum, hc.get(), nullptr));
  const std::vector<std::string> expected = {"ReduceScatter", "AllReduce", "AllGather"};
  EXPECT_EQ(expected, g_log);
  EXPECT_EQ(stageBytes, hc.comm.hostHier->scratchBytes);
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
