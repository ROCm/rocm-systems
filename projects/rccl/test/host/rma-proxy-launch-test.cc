/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/rma/rma_proxy_launch.cc.
 *************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"

#include "nccl.h"
#include "comm.h"
#include "rma/rma_proxy.h"

// Other units in rccl-UnitTestsMicro need controllable doubles for these
// exported functions. Rename this translation unit's definitions at inclusion
// time so those doubles keep serving their existing callers while the tests
// below exercise the production implementation directly.
#define ncclCuStreamBatchMemOp ncclCuStreamBatchMemOpUut
#define ncclRmaProxyCircularBufEmpty ncclRmaProxyCircularBufEmptyUut
#define ncclRmaProxyDestroyDesc ncclRmaProxyDestroyDescUut
#define ncclRmaProxyPutLaunch ncclRmaProxyPutLaunchUut
#define ncclRmaProxyWaitLaunch ncclRmaProxyWaitLaunchUut
#define ncclRmaProxyReclaimPlan ncclRmaProxyReclaimPlanUut
#include RMA_PROXY_LAUNCH_CC_PATH
#undef ncclRmaProxyReclaimPlan
#undef ncclRmaProxyWaitLaunch
#undef ncclRmaProxyPutLaunch
#undef ncclRmaProxyDestroyDesc
#undef ncclRmaProxyCircularBufEmpty
#undef ncclCuStreamBatchMemOp

namespace {

// ---------------------------------------------------------------------------
// RCCL's HIP batch-memory-operation wrapper.
// ---------------------------------------------------------------------------

class RmaProxyBatchMemOpTest : public ::testing::Test {
protected:
  void SetUp() override { ResetHipFakes(); }
  void TearDown() override { ResetHipFakes(); }
};

TEST_F(RmaProxyBatchMemOpTest, MoreThanTheHipLimit_IsSubmittedInContiguousChunks) {
  struct Call {
    hipStream_t stream;
    unsigned int count;
    hipStreamBatchMemOpParams* params;
    unsigned int flags;
  };
  std::vector<Call> calls;
  std::vector<hipStreamBatchMemOpParams> params(600);
  hipStream_t stream = reinterpret_cast<hipStream_t>(0x1234);
  ScopedHook batch(g_hipStreamBatchMemOp,
                   [&](hipStream_t gotStream, unsigned int count,
                       hipStreamBatchMemOpParams* gotParams, unsigned int flags) {
                     calls.push_back({gotStream, count, gotParams, flags});
                     return hipSuccess;
                   });

  ASSERT_EQ(ncclSuccess,
            ncclCuStreamBatchMemOpUut(stream, params.size(), params.data()));

  ASSERT_EQ(3u, calls.size());
  EXPECT_EQ(3, batch.calls);
  EXPECT_EQ(Call({stream, 255, params.data(), 0}).stream, calls[0].stream);
  EXPECT_EQ(255u, calls[0].count);
  EXPECT_EQ(params.data(), calls[0].params);
  EXPECT_EQ(0u, calls[0].flags);
  EXPECT_EQ(255u, calls[1].count);
  EXPECT_EQ(params.data() + 255, calls[1].params);
  EXPECT_EQ(90u, calls[2].count);
  EXPECT_EQ(params.data() + 510, calls[2].params);
}

TEST_F(RmaProxyBatchMemOpTest, HipFailure_StopsBeforeSubmittingLaterChunks) {
  std::vector<hipStreamBatchMemOpParams*> submitted;
  std::vector<hipStreamBatchMemOpParams> params(600);
  ScopedHook batch(g_hipStreamBatchMemOp,
                   [&](hipStream_t, unsigned int, hipStreamBatchMemOpParams* gotParams,
                       unsigned int) {
                     submitted.push_back(gotParams);
                     return submitted.size() == 2 ? hipErrorInvalidValue : hipSuccess;
                   });

  EXPECT_EQ(ncclUnhandledCudaError,
            ncclCuStreamBatchMemOpUut(nullptr, params.size(), params.data()));
  ASSERT_EQ(2u, submitted.size());
  EXPECT_EQ(2, batch.calls);
  EXPECT_EQ(params.data(), submitted[0]);
  EXPECT_EQ(params.data() + 255, submitted[1]);
}

TEST_F(RmaProxyBatchMemOpTest, ZeroOperations_DoesNotCallHip) {
  ScopedHook batch(g_hipStreamBatchMemOp,
                   [](hipStream_t, unsigned int, hipStreamBatchMemOpParams*, unsigned int) {
                     ADD_FAILURE() << "zero operations must not reach HIP";
                     return hipErrorInvalidValue;
                   });

  EXPECT_EQ(ncclSuccess, ncclCuStreamBatchMemOpUut(nullptr, 0, nullptr));
  EXPECT_EQ(0, batch.calls);
}

// ---------------------------------------------------------------------------
// Descriptor construction and ownership.
// ---------------------------------------------------------------------------

class RmaProxyDescriptorTest : public ::testing::Test {
protected:
  static constexpr int kNRanks = 7;
  static constexpr int kRank = 4;
  static constexpr int kPeer = 3;
  static constexpr int kContext = 2;

  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclKernelPlan> plan_;
  std::unique_ptr<ncclRmaProxyCtx> ctx_;
  ncclDevrWindow srcWin_{};
  ncclDevrWindow dstWin_{};
  std::vector<uint64_t> opSeqs_;
  std::vector<uint64_t> readySeqs_;
  std::vector<uint64_t> readySeqsDev_;
  std::vector<uint64_t> doneSeqs_;
  std::vector<uint64_t> doneSeqsDev_;

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = kNRanks;
    comm_->rank = kRank;
    plan_ = std::make_unique<ncclKernelPlan>();

    opSeqs_.assign(kNRanks, 0);
    readySeqs_.assign(kNRanks, 0);
    readySeqsDev_.assign(kNRanks, 0);
    doneSeqs_.assign(kNRanks, 0);
    doneSeqsDev_.assign(kNRanks, 0);

    ctx_ = std::make_unique<ncclRmaProxyCtx>();
    ctx_->comm = comm_.get();
    ctx_->collCommIdx = kContext;
    ctx_->opSeqs = opSeqs_.data();
    ctx_->readySeqs = readySeqs_.data();
    ctx_->readySeqsDev = readySeqsDev_.data();
    ctx_->readySeqsGdrHandle = reinterpret_cast<void*>(0x1110);
    ctx_->doneSeqs = doneSeqs_.data();
    ctx_->doneSeqsDev = doneSeqsDev_.data();
    ctx_->doneSeqsGdrHandle = reinterpret_cast<void*>(0x2220);
    ctx_->signalsMhandle = reinterpret_cast<void*>(0x3330);
    ctx_->cpuAccessSignalsMhandle = reinterpret_cast<void*>(0x4440);

    // A non-symmetric window stores one host MR handle per physical RMA
    // connection directly on the window. Using context 2 distinguishes the
    // selected handle from the surrounding entries.
    srcWin_.rmaHostWins[kContext] = reinterpret_cast<void*>(0x5550);
    dstWin_.rmaHostWins[kContext] = reinterpret_cast<void*>(0x6660);
  }
};

TEST_F(RmaProxyDescriptorTest, PutOp_NoSignalCopiesTheDataOperationOnly) {
  ncclRmaPutSignalOp op{};

  ASSERT_EQ(ncclSuccess,
            ncclRmaProxyPutBuildOp(comm_.get(), ctx_.get(), kContext, false,
                                   &srcWin_, 17, &dstWin_, 29, 4096, kPeer, 1,
                                   NCCL_SIGNAL_NONE, &op));

  EXPECT_EQ(17u, op.srcOff);
  EXPECT_EQ(srcWin_.rmaHostWins[kContext], op.srcHandle);
  EXPECT_EQ(29u, op.dstOff);
  EXPECT_EQ(dstWin_.rmaHostWins[kContext], op.dstHandle);
  EXPECT_EQ(4096u, op.size);
  EXPECT_EQ(kPeer, op.targetRank);
  EXPECT_EQ(nullptr, op.request);
  EXPECT_EQ(0u, op.signal.op);
}

TEST_F(RmaProxyDescriptorTest, PutOp_NonPersistentSignalUsesTheOrdinarySignalHandle) {
  ncclRmaPutSignalOp op{};

  ASSERT_EQ(ncclSuccess,
            ncclRmaProxyPutBuildOp(comm_.get(), ctx_.get(), kContext, false,
                                   &srcWin_, 5, &dstWin_, 11, 64, kPeer, 2,
                                   NCCL_SIGNAL, &op));

  EXPECT_EQ(NCCL_NET_SIGNAL_OP_ADD, op.signal.op);
  EXPECT_EQ(ncclRmaSignalOffset(kNRanks, 2, kRank), op.signal.offset);
  EXPECT_EQ(ctx_->signalsMhandle, op.signal.signalMhandle);
  EXPECT_EQ(1u, op.signal.val);
}

TEST_F(RmaProxyDescriptorTest, PutOp_PersistentSignalUsesTheCpuAccessibleHandle) {
  ncclRmaPutSignalOp op{};

  ASSERT_EQ(ncclSuccess,
            ncclRmaProxyPutBuildOp(comm_.get(), ctx_.get(), kContext, true,
                                   &srcWin_, 0, &dstWin_, 0, 1, kPeer, 1,
                                   NCCL_SIGNAL, &op));

  EXPECT_EQ(ctx_->cpuAccessSignalsMhandle, op.signal.signalMhandle);
}

TEST_F(RmaProxyDescriptorTest, PutDesc_NonPersistentUsesTheTargetSequenceSlot) {
  opSeqs_[kPeer] = 8;
  ncclRmaProxyDesc desc{};

  ASSERT_EQ(ncclSuccess,
            ncclRmaProxyPutBuildDesc(comm_.get(), ctx_.get(), plan_.get(),
                                     &srcWin_, 7, &dstWin_, 13, 512, kPeer,
                                     kContext, 1, NCCL_SIGNAL, &desc));

  EXPECT_EQ(ncclRmaDescTypePutSignal, desc.rmaDescType);
  EXPECT_EQ(ncclRmaDescStateReady, desc.rmaDescState);
  EXPECT_EQ(9u, desc.opSeq);
  EXPECT_EQ(9u, opSeqs_[kPeer]);
  EXPECT_EQ(&readySeqs_[kPeer], desc.readySeq);
  EXPECT_EQ(&readySeqsDev_[kPeer], desc.readySeqDev);
  EXPECT_EQ(ctx_->readySeqsGdrHandle, desc.readySeqGdrHandle);
  EXPECT_EQ(&doneSeqs_[kPeer], desc.doneSeq);
  EXPECT_EQ(&doneSeqsDev_[kPeer], desc.doneSeqDev);
  EXPECT_EQ(ctx_->doneSeqsGdrHandle, desc.doneSeqGdrHandle);
  EXPECT_EQ(nullptr, desc.persistPlan);
  EXPECT_FALSE(desc.persistDescValid);
  EXPECT_EQ(kPeer, desc.putSignal.targetRank);
  EXPECT_EQ(512u, desc.putSignal.size);
}

TEST_F(RmaProxyDescriptorTest, PutGroupDesc_NonPersistentTakesOpsAndUsesTheLocalSequenceSlot) {
  opSeqs_[kRank] = 14;
  auto* ops = static_cast<ncclRmaPutSignalOp*>(std::calloc(3, sizeof(ncclRmaPutSignalOp)));
  ASSERT_NE(nullptr, ops);
  ncclRmaPutSignalOp* submittedOps = ops;
  ncclRmaProxyDesc desc{};

  ASSERT_EQ(ncclSuccess,
            ncclRmaProxyPutGroupBuildDesc(comm_.get(), ctx_.get(), plan_.get(),
                                          3, &submittedOps, kContext, &desc));

  EXPECT_EQ(nullptr, submittedOps);
  EXPECT_EQ(ncclRmaDescTypePutSignalGroup, desc.rmaDescType);
  EXPECT_EQ(ncclRmaDescStateReady, desc.rmaDescState);
  EXPECT_EQ(3, desc.putSignalGroup.nOps);
  EXPECT_EQ(ops, desc.putSignalGroup.ops);
  EXPECT_EQ(0, desc.putSignalGroup.nIssued);
  EXPECT_EQ(0, desc.putSignalGroup.nCompleted);
  EXPECT_EQ(15u, desc.opSeq);
  EXPECT_EQ(15u, opSeqs_[kRank]);
  EXPECT_EQ(&readySeqs_[kRank], desc.readySeq);
  EXPECT_EQ(&doneSeqs_[kRank], desc.doneSeq);

  std::free(ops);
}

TEST_F(RmaProxyDescriptorTest, WaitDesc_NonPersistentTakesAllCallerArrays) {
  auto* peers = static_cast<int*>(std::malloc(2 * sizeof(int)));
  auto* nsignals = static_cast<int*>(std::malloc(2 * sizeof(int)));
  auto* signalIdxs = static_cast<int*>(std::malloc(2 * sizeof(int)));
  ASSERT_NE(nullptr, peers);
  ASSERT_NE(nullptr, nsignals);
  ASSERT_NE(nullptr, signalIdxs);
  peers[0] = 1;
  peers[1] = 5;
  nsignals[0] = 2;
  nsignals[1] = 9;
  signalIdxs[0] = 3;
  signalIdxs[1] = 4;
  int* submittedPeers = peers;
  int* submittedSignals = nsignals;
  int* submittedSignalIdxs = signalIdxs;
  ncclRmaProxyDesc desc{};

  ASSERT_EQ(ncclSuccess,
            ncclRmaProxyWaitBuildDesc(comm_.get(), ctx_.get(), plan_.get(), 2,
                                      &submittedPeers, &submittedSignals,
                                      &submittedSignalIdxs, &desc));

  EXPECT_EQ(nullptr, submittedPeers);
  EXPECT_EQ(nullptr, submittedSignals);
  EXPECT_EQ(nullptr, submittedSignalIdxs);
  EXPECT_EQ(ncclRmaDescTypeWaitSignal, desc.rmaDescType);
  EXPECT_EQ(ncclRmaDescStateReady, desc.rmaDescState);
  EXPECT_EQ(2, desc.waitSignal.npeers);
  EXPECT_EQ(peers, desc.waitSignal.waitPeers);
  EXPECT_EQ(nsignals, desc.waitSignal.waitSignals);
  EXPECT_EQ(signalIdxs, desc.waitSignal.waitSignalIdxs);
  EXPECT_EQ(nullptr, desc.persistPlan);

  std::free(peers);
  std::free(nsignals);
  std::free(signalIdxs);
}

// ---------------------------------------------------------------------------
// Public launch orchestration.
// ---------------------------------------------------------------------------

class RmaProxyLaunchTest : public ::testing::Test {
protected:
  static constexpr int kNRanks = 4;
  static constexpr int kContexts = 2;
  static constexpr size_t kQueueSize = 8;

  struct ContextStorage {
    ncclRmaProxyCtx ctx{};
    std::array<uint32_t, kNRanks> pis{};
    std::array<uint32_t, kNRanks> cis{};
    std::array<uint64_t, kNRanks> opSeqs{};
    std::array<uint64_t, kNRanks> readySeqs{};
    std::array<uint64_t, kNRanks> readySeqsDev{};
    std::array<uint64_t, kNRanks> doneSeqs{};
    std::array<uint64_t, kNRanks> doneSeqsDev{};
    std::array<uint64_t, 3 * kNRanks> signalsHost{};
    std::array<uint64_t, 3 * kNRanks> signalsDev{};
    std::array<ncclRmaProxyDesc*, kNRanks * kQueueSize> circular{};
    std::array<ncclIntruQueue<ncclRmaProxyDesc, &ncclRmaProxyDesc::next>, kNRanks>
        persistent{};

    void Init(ncclComm* comm, int collCommIdx) {
      ctx.comm = comm;
      ctx.collCommIdx = collCommIdx;
      ctx.queueSize = kQueueSize;
      ctx.pis = pis.data();
      ctx.cis = cis.data();
      ctx.opSeqs = opSeqs.data();
      ctx.readySeqs = readySeqs.data();
      ctx.readySeqsDev = readySeqsDev.data();
      ctx.doneSeqs = doneSeqs.data();
      ctx.doneSeqsDev = doneSeqsDev.data();
      ctx.signalsHost = signalsHost.data();
      ctx.signalsDev = signalsDev.data();
      ctx.circularBuffers = circular.data();
      ctx.persistentQueues = persistent.data();
      for (auto& queue : persistent) ncclIntruQueueConstruct(&queue);
    }
  };

  struct BatchCall {
    hipStream_t stream;
    std::vector<hipStreamBatchMemOpParams> params;
  };

  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclKernelPlan> plan_;
  ncclRmaArgs args_{};
  std::array<ContextStorage, kContexts> contexts_;
  std::array<void*, kContexts> contextPtrs_{};
  ncclDevrWindow srcWin_{};
  ncclDevrWindow dstWin_{};
  std::vector<std::unique_ptr<ncclTaskRma>> tasks_;

  void SetUp() override {
    ResetHipFakes();
    comm_ = std::make_unique<ncclComm>();
    comm_->rank = 1;
    comm_->nRanks = kNRanks;
    ncclMemoryPoolConstruct(&comm_->memPool_ncclTaskRma);
    comm_->rmaState.rmaProxyState.connected = true;
    comm_->rmaState.rmaProxyState.rmaProxyCtxs = contextPtrs_.data();

    for (int i = 0; i < kContexts; i++) {
      contexts_[i].Init(comm_.get(), i);
      contextPtrs_[i] = &contexts_[i].ctx;
      srcWin_.rmaHostWins[i] = reinterpret_cast<void*>(0x1000 + i * 0x100);
      dstWin_.rmaHostWins[i] = reinterpret_cast<void*>(0x2000 + i * 0x100);
    }

    plan_ = std::make_unique<ncclKernelPlan>();
    plan_->rmaArgs = &args_;
    ncclIntruQueueConstruct(&plan_->rmaTaskQueueProxy);
  }

  void TearDown() override {
    for (auto& storage : contexts_) {
      for (ncclRmaProxyDesc*& desc : storage.circular) {
        if (desc != nullptr) {
          ASSERT_EQ(ncclSuccess, ncclRmaProxyDestroyDescUut(comm_.get(), &desc));
        }
      }
    }
    ResetHipFakes();
  }

  ncclTaskRma* PushPut(int context, int peer, size_t count,
                       ncclSignalMode_t signalMode = NCCL_SIGNAL_NONE) {
    auto task = std::make_unique<ncclTaskRma>();
    task->func = ncclFuncPutSignal;
    task->ctx = context;
    task->count = count;
    task->datatype = ncclUint8;
    task->srcWinOffset = 17 + context;
    task->srcWinHost = &srcWin_;
    task->peer = peer;
    task->peerWinOffset = 31 + context;
    task->peerWinHost = &dstWin_;
    task->signalMode = signalMode;
    task->signalIdx = 2;
    ncclTaskRma* raw = task.get();
    tasks_.push_back(std::move(task));
    ncclIntruQueueEnqueue(&plan_->rmaTaskQueueProxy, raw);
    args_.nRmaTasksProxy++;
    return raw;
  }

  ncclTaskRma* PushWait(ncclFunc_t func, ncclSignalMode_t signalMode,
                        std::vector<int> peers = {}, std::vector<int> nsignals = {},
                        std::vector<int> signalIdxs = {}) {
    auto task = std::make_unique<ncclTaskRma>();
    task->func = func;
    task->ctx = 0;
    task->signalMode = signalMode;
    task->npeers = peers.size();
    if (!peers.empty()) {
      task->peers = static_cast<int*>(std::malloc(peers.size() * sizeof(int)));
      task->nsignals = static_cast<int*>(std::malloc(nsignals.size() * sizeof(int)));
      task->signalIdxs = static_cast<int*>(std::malloc(signalIdxs.size() * sizeof(int)));
      std::copy(peers.begin(), peers.end(), task->peers);
      std::copy(nsignals.begin(), nsignals.end(), task->nsignals);
      std::copy(signalIdxs.begin(), signalIdxs.end(), task->signalIdxs);
    }
    ncclTaskRma* raw = task.get();
    tasks_.push_back(std::move(task));
    ncclIntruQueueEnqueue(&plan_->rmaTaskQueueProxy, raw);
    args_.nRmaTasksProxy++;
    return raw;
  }

  ScopedHook<hipError_t(hipStream_t, unsigned int, hipStreamBatchMemOpParams*, unsigned int)>
  RecordBatches(std::vector<BatchCall>* calls) {
    return ScopedHook(g_hipStreamBatchMemOp,
                      [calls](hipStream_t stream, unsigned int count,
                              hipStreamBatchMemOpParams* params, unsigned int) {
                        calls->push_back({stream, {params, params + count}});
                        return hipSuccess;
                      });
  }
};

TEST_F(RmaProxyLaunchTest, PutLaunch_DisconnectedProxyIsRejectedBeforeReadingThePlan) {
  comm_->rmaState.rmaProxyState.connected = false;

  EXPECT_EQ(ncclInternalError,
            ncclRmaProxyPutLaunchUut(comm_.get(), nullptr, nullptr));
}

TEST_F(RmaProxyLaunchTest, WaitLaunch_DisconnectedProxyIsRejectedBeforeReadingThePlan) {
  comm_->rmaState.rmaProxyState.connected = false;

  EXPECT_EQ(ncclInternalError,
            ncclRmaProxyWaitLaunchUut(comm_.get(), nullptr, nullptr));
}

TEST_F(RmaProxyLaunchTest, PutLaunch_NoTasksReturnsWithoutSubmittingMemops) {
  ScopedHook batch(g_hipStreamBatchMemOp,
                   [](hipStream_t, unsigned int, hipStreamBatchMemOpParams*, unsigned int) {
                     ADD_FAILURE() << "an empty plan must not submit stream operations";
                     return hipErrorInvalidValue;
                   });

  EXPECT_EQ(ncclSuccess, ncclRmaProxyPutLaunchUut(comm_.get(), plan_.get(), nullptr));
  EXPECT_EQ(0, batch.calls);
}

TEST_F(RmaProxyLaunchTest, PutLaunch_TasksFromDifferentContextsAreEnqueuedAndSubmittedTogether) {
  PushPut(1, 2, 64);
  PushPut(0, 3, 128, NCCL_SIGNAL);
  hipStream_t stream = reinterpret_cast<hipStream_t>(0x7770);
  std::vector<BatchCall> calls;
  auto batch = RecordBatches(&calls);

  ASSERT_EQ(ncclSuccess, ncclRmaProxyPutLaunchUut(comm_.get(), plan_.get(), stream));

  ASSERT_EQ(1u, calls.size());
  EXPECT_EQ(1, batch.calls);
  EXPECT_EQ(stream, calls[0].stream);
  ASSERT_EQ(4u, calls[0].params.size());
  EXPECT_EQ(hipStreamMemOpWriteValue64, calls[0].params[0].writeValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&contexts_[1].readySeqsDev[2]),
            calls[0].params[0].writeValue.address);
  EXPECT_EQ(hipStreamMemOpWriteValue64, calls[0].params[1].writeValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&contexts_[0].readySeqsDev[3]),
            calls[0].params[1].writeValue.address);
  EXPECT_EQ(hipStreamMemOpWaitValue64, calls[0].params[2].waitValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&contexts_[1].doneSeqsDev[2]),
            calls[0].params[2].waitValue.address);
  EXPECT_EQ(hipStreamMemOpWaitValue64, calls[0].params[3].waitValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&contexts_[0].doneSeqsDev[3]),
            calls[0].params[3].waitValue.address);

  ncclRmaProxyDesc* first = contexts_[1].circular[2 * kQueueSize];
  ncclRmaProxyDesc* second = contexts_[0].circular[3 * kQueueSize];
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  EXPECT_EQ(64u, first->putSignal.size);
  EXPECT_EQ(srcWin_.rmaHostWins[1], first->putSignal.srcHandle);
  EXPECT_EQ(dstWin_.rmaHostWins[1], first->putSignal.dstHandle);
  EXPECT_EQ(128u, second->putSignal.size);
  EXPECT_EQ(srcWin_.rmaHostWins[0], second->putSignal.srcHandle);
  EXPECT_EQ(dstWin_.rmaHostWins[0], second->putSignal.dstHandle);
  EXPECT_EQ(tasks_[1].get(), reinterpret_cast<ncclTaskRma*>(comm_->memPool_ncclTaskRma.head));
}

TEST_F(RmaProxyLaunchTest, WaitLaunch_NonWaitTaskIsRejectedAndReturnedToThePool) {
  ncclTaskRma* task = PushWait(ncclFuncPutSignal, NCCL_SIGNAL_NONE);

  EXPECT_EQ(ncclInternalError,
            ncclRmaProxyWaitLaunchUut(comm_.get(), plan_.get(), nullptr));
  EXPECT_EQ(task, reinterpret_cast<ncclTaskRma*>(comm_->memPool_ncclTaskRma.head));
}

TEST_F(RmaProxyLaunchTest, WaitLaunch_MoreThanOneTaskIsRejected) {
  ncclTaskRma* task = PushWait(ncclFuncWaitSignal, NCCL_SIGNAL_NONE);
  args_.nRmaTasksProxy = 2;

  EXPECT_EQ(ncclInternalError,
            ncclRmaProxyWaitLaunchUut(comm_.get(), plan_.get(), nullptr));
  EXPECT_EQ(task, reinterpret_cast<ncclTaskRma*>(comm_->memPool_ncclTaskRma.head));
}

TEST_F(RmaProxyLaunchTest, WaitLaunch_NoSignalReturnsTheTaskWithoutSubmittingMemops) {
  ncclTaskRma* task = PushWait(ncclFuncWaitSignal, NCCL_SIGNAL_NONE);
  ScopedHook batch(g_hipStreamBatchMemOp,
                   [](hipStream_t, unsigned int, hipStreamBatchMemOpParams*, unsigned int) {
                     ADD_FAILURE() << "a wait without signalling must not submit stream operations";
                     return hipErrorInvalidValue;
                   });

  EXPECT_EQ(ncclSuccess,
            ncclRmaProxyWaitLaunchUut(comm_.get(), plan_.get(), nullptr));
  EXPECT_EQ(0, batch.calls);
  EXPECT_EQ(task, reinterpret_cast<ncclTaskRma*>(comm_->memPool_ncclTaskRma.head));
}

TEST_F(RmaProxyLaunchTest, WaitLaunch_SignalTaskSubmitsAccumulatedPeerWaits) {
  ncclTaskRma* task = PushWait(ncclFuncWaitSignal, NCCL_SIGNAL,
                               {2, 3}, {5, 7}, {1, 2});
  const size_t firstSlot = ncclRmaSignalSlot(kNRanks, 1, 2);
  const size_t secondSlot = ncclRmaSignalSlot(kNRanks, 2, 3);
  contexts_[0].signalsHost[firstSlot] = 11;
  contexts_[0].signalsHost[secondSlot] = 13;
  hipStream_t stream = reinterpret_cast<hipStream_t>(0x8880);
  std::vector<BatchCall> calls;
  auto batch = RecordBatches(&calls);

  ASSERT_EQ(ncclSuccess,
            ncclRmaProxyWaitLaunchUut(comm_.get(), plan_.get(), stream));

  ASSERT_EQ(1u, calls.size());
  EXPECT_EQ(1, batch.calls);
  EXPECT_EQ(stream, calls[0].stream);
  ASSERT_EQ(2u, calls[0].params.size());
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&contexts_[0].signalsDev[firstSlot]),
            calls[0].params[0].waitValue.address);
  EXPECT_EQ(16u, calls[0].params[0].waitValue.value64);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&contexts_[0].signalsDev[secondSlot]),
            calls[0].params[1].waitValue.address);
  EXPECT_EQ(20u, calls[0].params[1].waitValue.value64);
  EXPECT_EQ(nullptr, task->peers);
  EXPECT_EQ(nullptr, task->nsignals);
  EXPECT_EQ(nullptr, task->signalIdxs);
  EXPECT_EQ(task, reinterpret_cast<ncclTaskRma*>(comm_->memPool_ncclTaskRma.head));
}

// ---------------------------------------------------------------------------
// Circular-buffer and descriptor-queue contracts.
// ---------------------------------------------------------------------------

class RmaProxyQueueTest : public ::testing::Test {
protected:
  static constexpr int kNRanks = 4;
  static constexpr size_t kQueueSize = 8;

  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclRmaProxyCtx> ctx_;
  std::vector<uint32_t> pis_;
  std::vector<uint32_t> cis_;
  std::vector<ncclRmaProxyDesc*> circular_;
  std::vector<ncclIntruQueue<ncclRmaProxyDesc, &ncclRmaProxyDesc::next>> persistent_;

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    comm_->rank = 1;
    comm_->nRanks = kNRanks;

    pis_.assign(kNRanks, 0);
    cis_.assign(kNRanks, 0);
    circular_.assign(kNRanks * kQueueSize, nullptr);
    persistent_.resize(kNRanks);
    for (auto& queue : persistent_) ncclIntruQueueConstruct(&queue);

    ctx_ = std::make_unique<ncclRmaProxyCtx>();
    ctx_->comm = comm_.get();
    ctx_->queueSize = kQueueSize;
    ctx_->pis = pis_.data();
    ctx_->cis = cis_.data();
    ctx_->circularBuffers = circular_.data();
    ctx_->persistentQueues = persistent_.data();
  }
};

TEST_F(RmaProxyQueueTest, CircularBuffer_ProducerAtConsumer_IsEmptyAndNotFull) {
  pis_[2] = 19;
  cis_[2] = 19;

  EXPECT_TRUE(ncclRmaProxyCircularBufEmptyUut(ctx_.get(), 2));
  EXPECT_FALSE(ncclRmaProxyCircularBufFull(ctx_.get(), 2));
}

TEST_F(RmaProxyQueueTest, CircularBuffer_PendingEntry_IsNotEmpty) {
  pis_[2] = 20;
  cis_[2] = 19;

  EXPECT_FALSE(ncclRmaProxyCircularBufEmptyUut(ctx_.get(), 2));
}

TEST_F(RmaProxyQueueTest, CircularBuffer_ExactlyAtCapacity_IsFull) {
  pis_[2] = 27;
  cis_[2] = 19;

  EXPECT_TRUE(ncclRmaProxyCircularBufFull(ctx_.get(), 2));
}

TEST_F(RmaProxyQueueTest, CircularBuffer_WrappedIndicesUseUnsignedDistance) {
  cis_[2] = UINT32_MAX - 2;
  pis_[2] = 3;
  EXPECT_FALSE(ncclRmaProxyCircularBufFull(ctx_.get(), 2));

  pis_[2] = 5;
  EXPECT_TRUE(ncclRmaProxyCircularBufFull(ctx_.get(), 2));
}

TEST_F(RmaProxyQueueTest, EnqueueFull_SinglePutUsesItsTargetRank) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignal;
  desc.putSignal.targetRank = 3;
  pis_[comm_->rank] = kQueueSize;

  EXPECT_FALSE(ncclRmaProxyEnqueueFull(ctx_.get(), &desc));

  pis_[3] = kQueueSize;
  EXPECT_TRUE(ncclRmaProxyEnqueueFull(ctx_.get(), &desc));
}

TEST_F(RmaProxyQueueTest, EnqueueFull_GroupPutUsesTheLocalRank) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignalGroup;
  pis_[3] = kQueueSize;

  EXPECT_FALSE(ncclRmaProxyEnqueueFull(ctx_.get(), &desc));

  pis_[comm_->rank] = kQueueSize;
  EXPECT_TRUE(ncclRmaProxyEnqueueFull(ctx_.get(), &desc));
}

TEST_F(RmaProxyQueueTest, EnqueueFull_WaitAndPersistentDescriptorsAreUnbounded) {
  ncclKernelPlan plan{};
  ncclRmaProxyDesc wait{};
  wait.rmaDescType = ncclRmaDescTypeWaitSignal;
  ncclRmaProxyDesc persistentPut{};
  persistentPut.rmaDescType = ncclRmaDescTypePutSignal;
  persistentPut.putSignal.targetRank = 3;
  persistentPut.persistPlan = &plan;
  ncclRmaProxyDesc capturedPut{};
  capturedPut.rmaDescType = ncclRmaDescTypePutSignal;
  capturedPut.putSignal.targetRank = 3;
  capturedPut.captured = true;
  pis_[3] = kQueueSize;

  EXPECT_FALSE(ncclRmaProxyEnqueueFull(ctx_.get(), &wait));
  EXPECT_FALSE(ncclRmaProxyEnqueueFull(ctx_.get(), &persistentPut));
  EXPECT_FALSE(ncclRmaProxyEnqueueFull(ctx_.get(), &capturedPut));
}

TEST_F(RmaProxyQueueTest, EnqueueDesc_NonPersistentPutPublishesAtProducerSlot) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignal;
  desc.putSignal.targetRank = 3;
  pis_[3] = 9;
  cis_[3] = 4;
  ncclRmaProxyDesc* submitted = &desc;

  ASSERT_EQ(ncclSuccess, ncclRmaProxyEnqueueDesc(ctx_.get(), &submitted));

  EXPECT_EQ(nullptr, submitted);
  EXPECT_EQ(&desc, circular_[3 * kQueueSize + 1]);
  EXPECT_EQ(10u, pis_[3]);
}

TEST_F(RmaProxyQueueTest, EnqueueDesc_NonPersistentGroupPublishesOnTheLocalRank) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignalGroup;
  pis_[comm_->rank] = 2;
  ncclRmaProxyDesc* submitted = &desc;

  ASSERT_EQ(ncclSuccess, ncclRmaProxyEnqueueDesc(ctx_.get(), &submitted));

  EXPECT_EQ(nullptr, submitted);
  EXPECT_EQ(&desc, circular_[comm_->rank * kQueueSize + 2]);
  EXPECT_EQ(3u, pis_[comm_->rank]);
}

TEST_F(RmaProxyQueueTest, EnqueueDesc_PersistentPutAppendsAndMarksValid) {
  ncclKernelPlan plan{};
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignal;
  desc.putSignal.targetRank = 3;
  desc.persistPlan = &plan;
  ncclRmaProxyDesc* submitted = &desc;

  ASSERT_EQ(ncclSuccess, ncclRmaProxyEnqueueDesc(ctx_.get(), &submitted));

  EXPECT_EQ(nullptr, submitted);
  EXPECT_EQ(&desc, ncclIntruQueueHead(&persistent_[3]));
  EXPECT_EQ(&desc, persistent_[3].tail);
  EXPECT_TRUE(desc.persistDescValid);
}

TEST_F(RmaProxyQueueTest, EnqueueDesc_CapturedWaitAppendsOnTheLocalRank) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypeWaitSignal;
  desc.captured = true;
  ncclRmaProxyDesc* submitted = &desc;

  ASSERT_EQ(ncclSuccess, ncclRmaProxyEnqueueDesc(ctx_.get(), &submitted));

  EXPECT_EQ(nullptr, submitted);
  EXPECT_EQ(&desc, ncclIntruQueueHead(&persistent_[comm_->rank]));
  EXPECT_TRUE(desc.persistDescValid);
}

TEST_F(RmaProxyQueueTest, EnqueueDesc_UnknownDescriptorTypeIsRejectedWithoutTransfer) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = static_cast<ncclRmaDescType_t>(99);
  ncclRmaProxyDesc* submitted = &desc;

  EXPECT_EQ(ncclInternalError, ncclRmaProxyEnqueueDesc(ctx_.get(), &submitted));
  EXPECT_EQ(&desc, submitted);
}

// ---------------------------------------------------------------------------
// Stream-memory-operation parameter construction.
// ---------------------------------------------------------------------------

class RmaProxyParamsTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclRmaProxyCtx> ctx_;
  std::vector<uint64_t> signalsHost_;
  std::vector<uint64_t> signalsDev_;

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = 4;
    ctx_ = std::make_unique<ncclRmaProxyCtx>();
    ctx_->comm = comm_.get();
    signalsHost_.assign(12, 0);
    signalsDev_.assign(12, 0);
    ctx_->signalsHost = signalsHost_.data();
    ctx_->signalsDev = signalsDev_.data();
  }
};

TEST_F(RmaProxyParamsTest, PutStart_WritesTheReadySequence) {
  uint64_t ready = 0;
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignal;
  desc.readySeqDev = &ready;
  desc.opSeq = 37;
  hipStreamBatchMemOpParams params{};

  EXPECT_EQ(1, ncclRmaProxyPutStartNumOps(false));
  EXPECT_EQ(1, ncclRmaProxyPutStartNumOps(true));
  ASSERT_EQ(ncclSuccess, ncclRmaProxyPutStartParams(&desc, &params));
  EXPECT_EQ(hipStreamMemOpWriteValue64, params.writeValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&ready), params.writeValue.address);
  EXPECT_EQ(37u, params.writeValue.value64);
  EXPECT_EQ(CU_STREAM_WRITE_VALUE_DEFAULT, params.writeValue.flags);
}

TEST_F(RmaProxyParamsTest, PutStart_NonPutDescriptorIsRejected) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypeWaitSignal;
  hipStreamBatchMemOpParams params{};

  EXPECT_EQ(ncclInternalError, ncclRmaProxyPutStartParams(&desc, &params));
}

TEST_F(RmaProxyParamsTest, PutDone_NonPersistentWaitsForTheDoneSequence) {
  uint64_t done = 0;
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignal;
  desc.doneSeqDev = &done;
  desc.opSeq = 41;
  std::array<hipStreamBatchMemOpParams, 2> params{};

  EXPECT_EQ(1, ncclRmaProxyPutDoneNumOps(false));
  ASSERT_EQ(ncclSuccess, ncclRmaProxyPutDoneParams(&desc, params.data()));
  EXPECT_EQ(hipStreamMemOpWaitValue64, params[0].waitValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&done), params[0].waitValue.address);
  EXPECT_EQ(41u, params[0].waitValue.value64);
  EXPECT_EQ(hipStreamWaitValueGte, params[0].waitValue.flags);
  EXPECT_EQ(0u, params[1].operation);
}

TEST_F(RmaProxyParamsTest, PutDone_PersistentWaitsThenResetsTheDoneSequence) {
  uint64_t done = 0;
  ncclKernelPlan plan{};
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignal;
  desc.doneSeqDev = &done;
  desc.opSeq = 43;
  desc.persistPlan = &plan;
  std::array<hipStreamBatchMemOpParams, 2> params{};

  EXPECT_EQ(2, ncclRmaProxyPutDoneNumOps(true));
  ASSERT_EQ(ncclSuccess, ncclRmaProxyPutDoneParams(&desc, params.data()));
  EXPECT_EQ(hipStreamMemOpWaitValue64, params[0].waitValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&done), params[0].waitValue.address);
  EXPECT_EQ(43u, params[0].waitValue.value64);
  EXPECT_EQ(hipStreamWaitValueGte, params[0].waitValue.flags);
  EXPECT_EQ(hipStreamMemOpWriteValue64, params[1].writeValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&done), params[1].writeValue.address);
  EXPECT_EQ(0u, params[1].writeValue.value64);
  EXPECT_EQ(CU_STREAM_WRITE_VALUE_DEFAULT, params[1].writeValue.flags);
}

TEST_F(RmaProxyParamsTest, PutDone_CapturedDescriptorAlsoResetsTheDoneSequence) {
  uint64_t done = 0;
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignal;
  desc.doneSeqDev = &done;
  desc.opSeq = 47;
  desc.captured = true;
  std::array<hipStreamBatchMemOpParams, 2> params{};

  ASSERT_EQ(ncclSuccess, ncclRmaProxyPutDoneParams(&desc, params.data()));
  EXPECT_EQ(hipStreamMemOpWriteValue64, params[1].writeValue.operation);
}

TEST_F(RmaProxyParamsTest, PutDone_NonPutDescriptorIsRejected) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignalGroup;
  hipStreamBatchMemOpParams params{};

  EXPECT_EQ(ncclInternalError, ncclRmaProxyPutDoneParams(&desc, &params));
}

TEST_F(RmaProxyParamsTest, GroupStart_WritesTheSharedReadySequence) {
  uint64_t ready = 0;
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignalGroup;
  desc.readySeqDev = &ready;
  desc.opSeq = 53;
  hipStreamBatchMemOpParams params{};

  EXPECT_EQ(1, ncclRmaProxyPutGroupStartNumOps(false));
  EXPECT_EQ(1, ncclRmaProxyPutGroupStartNumOps(true));
  ASSERT_EQ(ncclSuccess, ncclRmaProxyPutGroupStartParams(&desc, &params));
  EXPECT_EQ(hipStreamMemOpWriteValue64, params.writeValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&ready), params.writeValue.address);
  EXPECT_EQ(53u, params.writeValue.value64);
  EXPECT_EQ(CU_STREAM_WRITE_VALUE_DEFAULT, params.writeValue.flags);
}

TEST_F(RmaProxyParamsTest, GroupStart_NonGroupDescriptorIsRejected) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignal;
  hipStreamBatchMemOpParams params{};

  EXPECT_EQ(ncclInternalError, ncclRmaProxyPutGroupStartParams(&desc, &params));
}

TEST_F(RmaProxyParamsTest, GroupDone_PersistentWaitsThenResetsTheSharedDoneSequence) {
  uint64_t done = 0;
  ncclKernelPlan plan{};
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignalGroup;
  desc.doneSeqDev = &done;
  desc.opSeq = 59;
  desc.persistPlan = &plan;
  std::array<hipStreamBatchMemOpParams, 2> params{};

  EXPECT_EQ(1, ncclRmaProxyPutGroupDoneNumOps(false));
  EXPECT_EQ(2, ncclRmaProxyPutGroupDoneNumOps(true));
  ASSERT_EQ(ncclSuccess, ncclRmaProxyPutGroupDoneParams(&desc, params.data()));
  EXPECT_EQ(hipStreamMemOpWaitValue64, params[0].waitValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&done), params[0].waitValue.address);
  EXPECT_EQ(59u, params[0].waitValue.value64);
  EXPECT_EQ(hipStreamWaitValueGte, params[0].waitValue.flags);
  EXPECT_EQ(hipStreamMemOpWriteValue64, params[1].writeValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&done), params[1].writeValue.address);
  EXPECT_EQ(0u, params[1].writeValue.value64);
}

TEST_F(RmaProxyParamsTest, GroupDone_NonGroupDescriptorIsRejected) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignal;
  hipStreamBatchMemOpParams params{};

  EXPECT_EQ(ncclInternalError, ncclRmaProxyPutGroupDoneParams(&desc, &params));
}

TEST_F(RmaProxyParamsTest, Wait_NonPersistentAccumulatesEachSignalSlot) {
  std::array<int, 2> peers{2, 1};
  std::array<int, 2> nsignals{3, 7};
  std::array<int, 2> signalIdxs{1, 2};
  const size_t firstSlot = ncclRmaSignalSlot(comm_->nRanks, signalIdxs[0], peers[0]);
  const size_t secondSlot = ncclRmaSignalSlot(comm_->nRanks, signalIdxs[1], peers[1]);
  signalsHost_[firstSlot] = 10;
  signalsHost_[secondSlot] = 20;

  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypeWaitSignal;
  desc.waitSignal.npeers = peers.size();
  desc.waitSignal.waitPeers = peers.data();
  desc.waitSignal.waitSignals = nsignals.data();
  desc.waitSignal.waitSignalIdxs = signalIdxs.data();
  std::array<hipStreamBatchMemOpParams, 2> params{};

  EXPECT_EQ(2, ncclRmaProxyWaitNumStreamOps(&desc));
  ASSERT_EQ(ncclSuccess, ncclRmaProxyWaitParams(ctx_.get(), &desc, params.data()));

  EXPECT_EQ(13u, signalsHost_[firstSlot]);
  EXPECT_EQ(27u, signalsHost_[secondSlot]);
  EXPECT_EQ(hipStreamMemOpWaitValue64, params[0].waitValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&signalsDev_[firstSlot]), params[0].waitValue.address);
  EXPECT_EQ(13u, params[0].waitValue.value64);
  EXPECT_EQ(hipStreamWaitValueGte, params[0].waitValue.flags);
  EXPECT_EQ(hipStreamMemOpWaitValue64, params[1].waitValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&signalsDev_[secondSlot]), params[1].waitValue.address);
  EXPECT_EQ(27u, params[1].waitValue.value64);
  EXPECT_EQ(hipStreamWaitValueGte, params[1].waitValue.flags);
}

TEST_F(RmaProxyParamsTest, Wait_CapturedDescriptorSignalsWaitsAndResets) {
  uint64_t ready = 0;
  uint64_t done = 0;
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypeWaitSignal;
  desc.readySeqDev = &ready;
  desc.doneSeqDev = &done;
  desc.opSeq = 61;
  desc.captured = true;
  std::array<hipStreamBatchMemOpParams, 3> params{};

  EXPECT_EQ(3, ncclRmaProxyWaitNumStreamOps(&desc));
  ASSERT_EQ(ncclSuccess, ncclRmaProxyWaitParams(ctx_.get(), &desc, params.data()));
  EXPECT_EQ(hipStreamMemOpWriteValue64, params[0].writeValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&ready), params[0].writeValue.address);
  EXPECT_EQ(61u, params[0].writeValue.value64);
  EXPECT_EQ(hipStreamMemOpWaitValue64, params[1].waitValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&done), params[1].waitValue.address);
  EXPECT_EQ(61u, params[1].waitValue.value64);
  EXPECT_EQ(hipStreamWaitValueGte, params[1].waitValue.flags);
  EXPECT_EQ(hipStreamMemOpWriteValue64, params[2].writeValue.operation);
  EXPECT_EQ(reinterpret_cast<hipDeviceptr_t>(&done), params[2].writeValue.address);
  EXPECT_EQ(0u, params[2].writeValue.value64);
}

TEST_F(RmaProxyParamsTest, Wait_NonWaitDescriptorIsRejected) {
  ncclRmaProxyDesc desc{};
  desc.rmaDescType = ncclRmaDescTypePutSignal;
  hipStreamBatchMemOpParams params{};

  EXPECT_EQ(ncclInternalError, ncclRmaProxyWaitParams(ctx_.get(), &desc, &params));
}

// ---------------------------------------------------------------------------
// Persistent-descriptor reclaim.
// ---------------------------------------------------------------------------

class RmaProxyReclaimTest : public ::testing::Test {
protected:
  static constexpr int kNRanks = 3;
  static constexpr int kContexts = 3;

  struct ContextStorage {
    ncclRmaProxyCtx ctx{};
    std::array<ncclIntruQueue<ncclRmaProxyDesc, &ncclRmaProxyDesc::next>, kNRanks>
        persistent{};

    void Init(ncclComm* comm) {
      ctx.comm = comm;
      ctx.persistentQueues = persistent.data();
      for (auto& queue : persistent) ncclIntruQueueConstruct(&queue);
    }
  };

  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclKernelPlan> targetPlan_;
  std::unique_ptr<ncclKernelPlan> otherPlan_;
  std::array<ContextStorage, kContexts> contexts_;
  std::array<void*, kContexts> contextPtrs_{};

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = kNRanks;
    targetPlan_ = std::make_unique<ncclKernelPlan>();
    otherPlan_ = std::make_unique<ncclKernelPlan>();

    for (int i = 0; i < kContexts; i++) {
      contexts_[i].Init(comm_.get());
      contextPtrs_[i] = &contexts_[i].ctx;
    }
    comm_->rmaState.rmaProxyState.comm = comm_.get();
    comm_->rmaState.rmaProxyState.rmaProxyCtxCount = kContexts;
    comm_->rmaState.rmaProxyState.rmaProxyCtxs = contextPtrs_.data();
  }

  void TearDown() override {
    for (auto& storage : contexts_) {
      for (auto& queue : storage.persistent) {
        while (!ncclIntruQueueEmpty(&queue)) {
          ncclRmaProxyDesc* desc = ncclIntruQueueDequeue(&queue);
          std::free(desc);
        }
      }
    }
  }

  ncclRmaProxyDesc* Append(int context, int peer, ncclKernelPlan* plan) {
    auto* desc = static_cast<ncclRmaProxyDesc*>(std::calloc(1, sizeof(ncclRmaProxyDesc)));
    EXPECT_NE(nullptr, desc);
    if (desc == nullptr) return nullptr;
    desc->rmaDescType = ncclRmaDescTypePutSignal;
    desc->persistPlan = plan;
    ncclIntruQueueEnqueue(&contexts_[context].persistent[peer], desc);
    return desc;
  }
};

TEST_F(RmaProxyReclaimTest, ReclaimPersistDescs_RemovesOnlyTheRequestedPlansDescriptors) {
  ncclRmaProxyDesc* firstTarget = Append(0, 1, targetPlan_.get());
  ncclRmaProxyDesc* firstOther = Append(0, 1, otherPlan_.get());
  ncclRmaProxyDesc* middleTarget = Append(0, 1, targetPlan_.get());
  ncclRmaProxyDesc* secondOther = Append(0, 1, otherPlan_.get());
  ncclRmaProxyDesc* lastTarget = Append(0, 1, targetPlan_.get());
  ncclRmaProxyDesc* otherContextTarget = Append(2, 2, targetPlan_.get());
  ASSERT_NE(nullptr, firstTarget);
  ASSERT_NE(nullptr, firstOther);
  ASSERT_NE(nullptr, middleTarget);
  ASSERT_NE(nullptr, secondOther);
  ASSERT_NE(nullptr, lastTarget);
  ASSERT_NE(nullptr, otherContextTarget);
  contextPtrs_[1] = nullptr;

  ASSERT_EQ(ncclSuccess,
            ncclRmaProxyReclaimPersistDescs(&comm_->rmaState.rmaProxyState,
                                            targetPlan_.get()));

  EXPECT_EQ(firstOther, ncclIntruQueueHead(&contexts_[0].persistent[1]));
  EXPECT_EQ(secondOther, firstOther->next);
  EXPECT_EQ(nullptr, secondOther->next);
  EXPECT_EQ(secondOther, contexts_[0].persistent[1].tail);
  EXPECT_EQ(nullptr, ncclIntruQueueHead(&contexts_[2].persistent[2]));
  EXPECT_EQ(nullptr, contexts_[2].persistent[2].tail);
}

TEST_F(RmaProxyReclaimTest, ReclaimPlan_DisconnectedProxyReturnsWithoutPausing) {
  comm_->rmaState.rmaProxyState.connected = false;
  comm_->rmaState.rmaProxyState.rmaProgress = 7;

  EXPECT_EQ(ncclSuccess,
            ncclRmaProxyReclaimPlanUut(comm_.get(), targetPlan_.get()));
  EXPECT_EQ(7, comm_->rmaState.rmaProxyState.rmaProgress);
}

TEST_F(RmaProxyReclaimTest, ReclaimPlan_ConnectedProxyPausesReclaimsAndResumes) {
  ncclRmaProxyState* state = &comm_->rmaState.rmaProxyState;
  state->connected = true;
  state->rmaProgress = 1;
  ASSERT_NE(nullptr, Append(0, 2, targetPlan_.get()));

  std::thread proxy([&] {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->cond.wait(lock, [&] { return state->rmaProgress == 2; });
    state->rmaProgress = 0;
    state->cond.notify_one();
    state->cond.wait(lock, [&] { return state->rmaProgress == 1; });
  });

  EXPECT_EQ(ncclSuccess, ncclRmaProxyReclaimPlanUut(comm_.get(), targetPlan_.get()));
  proxy.join();

  EXPECT_EQ(1, state->rmaProgress);
  EXPECT_EQ(nullptr, ncclIntruQueueHead(&contexts_[0].persistent[2]));
}

}  // namespace
