/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/ce_coll.cc: the hierarchical CE AllGather rail
 * (AICOMRCCL-2513).
 *************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

// fakes/ce_fakes.cc and fakes/collective_stubs.cc define these ce_coll.cc entry
// points for the rest of this binary. Rename the unit's copies before any header
// declares them, so declarations, default arguments and definitions move together.
#define ncclCeInit ncclCeInitUut
#define ncclCeImplemented ncclCeImplementedUut
#define ncclCeAvailable ncclCeAvailableUut
#define ncclCeScratchAvailable ncclCeScratchAvailableUut
#define ncclHierCeAvailable ncclHierCeAvailableUut
#define ncclCeInitBatchOpsParams ncclCeInitBatchOpsParamsUut
#define ncclCeLaunchBatchOps ncclCeLaunchBatchOpsUut
#define ncclCeFreeBatchOpsParams ncclCeFreeBatchOpsParamsUut
#define ncclLaunchCeColl ncclLaunchCeCollUut

#include "ScopedHook.h"
#include "fakes/nccl_fakes.h"  // g_loadParam, used by param_redirect.h

// Everything ce_coll.cc includes, so none of it is first seen under the seams below.
#include <hip/hip_runtime.h>
#include "comm.h"
#include "register_inline.h"
#include "rocmwrap.h"
#include "ce_coll.h"
#include "alltoallv_meta.h"
#include "group.h"
#include "alloc.h"
#include "ce_fault_inject.h"
#include "enqueue.h"
#include "tuning.h"
#include "dev_runtime.h"
#include "rma/rma_proxy.h"
#include "fakes/param_redirect.h"

// Every data put the rail builds: (peer, offset in the peer's recv window, bytes).
// RailSync's entry-barrier puts carry no data and are left out.
using Puts = std::vector<std::tuple<int, size_t, size_t>>;
static Puts g_hierCePuts;
static ncclResult_t HierCePutBuildOp(ncclComm*, ncclRmaProxyCtx*, int, bool, ncclDevrWindow*, size_t, ncclDevrWindow*,
                                     size_t peerOff, size_t size, int peer, int, ncclSignalMode_t,
                                     ncclRmaPutSignalOp*) {
  if (size > 0) g_hierCePuts.emplace_back(peer, peerOff, size);
  return ncclSuccess;
}
// The group builder and the enqueue take ownership, as rma_proxy.h documents.
static ncclResult_t HierCePutGroupBuildDesc(ncclComm*, ncclRmaProxyCtx*, ncclKernelPlan*, int,
                                            ncclRmaPutSignalOp** ops, int, ncclRmaProxyDesc*) {
  free(*ops);
  *ops = nullptr;
  return ncclSuccess;
}
static ncclResult_t HierCeEnqueueDesc(ncclRmaProxyCtx*, ncclRmaProxyDesc** desc) {
  free(*desc);
  *desc = nullptr;
  return ncclSuccess;
}
static ncclResult_t HierCeNotLegacyStream(hipStream_t, bool* isLegacy) {
  *isLegacy = false;
  return ncclSuccess;
}
template <typename... A> static ncclResult_t HierCeOk(A...) { return ncclSuccess; }
template <typename... A> static int HierCeOneOp(A...) { return 1; }
template <typename... A> static hipError_t HierCeHipOk(A...) { return hipSuccess; }

#define ncclRmaProxyPutBuildOp HierCePutBuildOp
#define ncclRmaProxyPutGroupBuildDesc HierCePutGroupBuildDesc
#define ncclRmaProxyEnqueueDesc HierCeEnqueueDesc
#define ncclRmaProxyWaitBuildDesc HierCeOk
#define ncclRmaProxyPutGroupStartParams HierCeOk
#define ncclRmaProxyPutGroupDoneParams HierCeOk
#define ncclRmaProxyWaitParams HierCeOk
#define ncclRmaProxyPutGroupStartNumOps HierCeOneOp
#define ncclRmaProxyPutGroupDoneNumOps HierCeOneOp
#define ncclRmaProxyWaitNumStreamOps HierCeOneOp
#define ncclCuStreamBatchMemOp HierCeOk
#define ncclProfilerStartCeSyncEvent HierCeOk
#define ncclProfilerStopCeSyncEvent HierCeOk
#define ncclProfilerStartCeBatchEvent HierCeOk
#define ncclProfilerStopCeBatchEvent HierCeOk
#define ncclCudaStreamIsLegacyNull HierCeNotLegacyStream
#define hipStreamBatchMemOp HierCeHipOk
#define hipStreamWriteValue32 HierCeHipOk
#define cuStreamWriteValue32 HierCeHipOk
#include CE_COLL_CC_PATH

namespace {

// Four nodes of one GPU each, seen from rank 1: the ring's next node is 2 and its
// previous node is 0, so node 3 is a rail peer only the Direct rail writes to.
constexpr int kNodes = 4;
constexpr int kRank = 1;
constexpr size_t kSlice = 64 * 1024;  // per-rank bytes, below the multi-context threshold
constexpr size_t kHalf = kSlice / 2;

class HierCeAllGatherMicrotest : public ::testing::Test {
protected:
  void SetUp() override {
    g_hierCePuts.clear();
    comm_->rank = comm_->node = kRank;
    comm_->nRanks = comm_->nNodes = kNodes;
    comm_->localRanks = 1;
    for (int n = 0; n < kNodes; n++) nodeRanks_[n] = {1, &railRank_[n]};
    comm_->nodeRanks = nodeRanks_;
    comm_->devrState.lsaSize = 1;
    comm_->devrState.lsaRankList = &railRank_[kRank];
    syncWin_.userPtr = syncFlags_;
    comm_->ceColl.ceSyncWin = &syncWin_;
    comm_->ceColl.baseUCSymReadyPtr = reinterpret_cast<uint8_t*>(syncFlags_);
    comm_->ceColl.baseUCSymComplPtr = reinterpret_cast<uint8_t*>(syncFlags_ + kNodes);
    // A zeroed graph reads as captured; the collective runs eagerly.
    comm_->planner.capturingGraph = ncclCudaGraphNone(0);
    comm_->rmaState.rmaProxyState.numIntCtx = 1;
    comm_->rmaState.rmaProxyState.rmaProxyCtxs = proxyCtxs_;
    // In place, so the self-copy is skipped and no HIP copy is issued.
    win_.userPtr = buf_.data();
    args_.func = ncclFuncAllGather;
    args_.nElts = kSlice;
    args_.eltSize = 1;
    args_.recvBuff = buf_.data();
    args_.sendBuff = buf_.data() + kRank * kSlice;
    args_.sendWin = args_.recvWin = &win_;
    plan_->ceCollArgs = &args_;
  }

  static Puts SortedPuts() {
    Puts puts = g_hierCePuts;
    std::sort(puts.begin(), puts.end());
    return puts;
  }

  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  std::unique_ptr<ncclKernelPlan> plan_ = std::make_unique<ncclKernelPlan>();
  ncclCeCollArgs args_{};
  ncclDevrWindow win_{}, syncWin_{};
  std::vector<uint8_t> buf_ = std::vector<uint8_t>(kNodes * kSlice);
  int railRank_[kNodes] = {0, 1, 2, 3};
  ncclNodeRanks nodeRanks_[kNodes];
  uint32_t syncFlags_[2 * kNodes] = {};
  void* proxyCtxs_[1] = {};
};

// NCCL_HIER_CE_COLL_AG_RAIL_RING_ENABLE=1 selects the bidirectional ring: half of
// every slice travels clockwise to the next node (rank 2: slices 1, 0, 3) and the
// other half counterclockwise to the previous node (rank 0: slices 1, 2, 3).
TEST_F(HierCeAllGatherMicrotest, RingEnabled_ForwardsHalvesToRingNeighboursOnly) {
  ScopedHook ring(g_loadParam, [](const char* env, int64_t deflt) -> int64_t {
    return std::string(env) == "HIER_CE_COLL_AG_RAIL_RING_ENABLE" ? 1 : deflt;
  });
  ASSERT_EQ(ncclHierCeAllGather(comm_.get(), plan_.get(), nullptr), ncclSuccess);
  EXPECT_EQ(SortedPuts(), (Puts{{0, kSlice + kHalf, kHalf}, {0, 2 * kSlice + kHalf, kHalf},
                                {0, 3 * kSlice + kHalf, kHalf}, {2, 0, kHalf}, {2, kSlice, kHalf},
                                {2, 3 * kSlice, kHalf}}));
}

// The switch is off by default: the Direct rail puts the whole own slice to every
// remote rail peer.
TEST_F(HierCeAllGatherMicrotest, DirectByDefault_PutsOwnSliceToEveryRemoteRailPeer) {
  ASSERT_EQ(ncclHierCeAllGather(comm_.get(), plan_.get(), nullptr), ncclSuccess);
  EXPECT_EQ(SortedPuts(), (Puts{{0, kSlice, kSlice}, {2, kSlice, kSlice}, {3, kSlice, kSlice}}));
}

}  // namespace
