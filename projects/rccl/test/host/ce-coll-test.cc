/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ce_coll.cc: the UUT is #include'd, so its launch entry points run
// against fakes/ with no GPU or HIP runtime.

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/allocator_fakes.h"
#include "fakes/dev_runtime_fakes.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"  // g_loadParam, used by param_redirect.h
#include "fakes/nccl_stubs.h"
#include "fakes/rccl_wrap_fakes.h"

#include "fakes/param_redirect.h"
#include "fakes/nvtx_redirect.h"

#include CE_COLL_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

constexpr char kLaunchNoStagingWarn[] = "launched without its staging buffer";

// Launch runs on one thread for every local rank, so a collective window registration there deadlocks (ROCM-32044).
class CeCollMicrotest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;
  ncclCudaStreamList streams_{};
  ncclCeCollArgs args_{};
  ncclKernelPlan plan_{};
  uint8_t staging_ = 0;
  uint8_t allocated_ = 0;
  size_t allocSize_ = 0;
  ncclWindow_vidmem devVidmem_{};
  ncclWindow_vidmem hostVidmem_{};
  ncclDevrWindow devrWin_{};
  ncclResult_t registerResult_ = ncclInternalError;
  ncclResult_t toHostResult_ = ncclSuccess;
  int expectedRegistrations_ = 0;  // TearDown checks it, so every launch test also guards against registering
  // Lets a regressed launch-time staging setup reach the window registration the tests count instead of aborting.
  ScopedHook<ncclResult_t(void**, size_t)> alloc_{g_ncclMemAlloc, [this](void** ptr, size_t size) {
    *ptr = &allocated_;
    allocSize_ = size;
    return ncclSuccess;
  }};
  ScopedHook<ncclResult_t(void*)> free_{g_ncclMemFree, [this](void* ptr) {
    EXPECT_EQ(&allocated_, ptr);
    return ncclSuccess;
  }};
  ScopedHook<ncclResult_t(ncclComm*, void*, size_t, int, ncclWindow_t*)> registration_{
      g_devrWindowRegisterInGroup, [this](ncclComm* comm, void* ptr, size_t size, int winFlags, ncclWindow_t* win) {
        EXPECT_EQ(comm_.get(), comm);
        EXPECT_EQ(&allocated_, ptr);
        EXPECT_EQ(allocSize_, size);
        EXPECT_EQ(NCCL_WIN_COLL_SYMMETRIC, winFlags);
        if (registerResult_ == ncclSuccess) {
          *win = &devVidmem_;
        }
        return registerResult_;
      }};
  ScopedHook<ncclResult_t(ncclShadowPool*, void*, void**)> toHost_{
      g_shadowPoolToHost, [this](ncclShadowPool*, void* devObj, void** hostObj) {
        EXPECT_EQ(&devVidmem_, devObj);
        if (toHostResult_ == ncclSuccess) {
          *hostObj = &hostVidmem_;
        }
        return toHostResult_;
      }};

  void SetUp() override {
    ResetRcclWrapFakes();
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = 4;
    comm_->nNodes = 1;
    comm_->planner.streams = &streams_;
    comm_->ceColl.ceArStagingBytes = 1024;
    hostVidmem_.winHost = &devrWin_;
    devrWin_.userPtr = &staging_;  // not &allocated_, so a test can tell the published window from the raw buffer
    args_.func = ncclFuncAllReduce;
    args_.datatype = ncclFloat32;
    args_.eltSize = sizeof(float);
    plan_.ceCollArgs = &args_;
  }
  void TearDown() override {
    EXPECT_EQ(expectedRegistrations_, registration_.calls);
    ResetRcclWrapFakes();
  }

  ncclResult_t Run(const std::function<ncclResult_t()>& body, std::string* log) {
    ncclResult_t res = ncclSuccess;
    *log = CaptureLog([&]() { res = body(); });
    return res;
  }
  ncclResult_t Launch(std::string* log) {
    return Run([&]() { return ncclLaunchCeColl(comm_.get(), &plan_); }, log);
  }
};

TEST_F(CeCollMicrotest, LaunchAllReduceWithoutStaging_FailsLoudlyWithoutRegistering) {
  std::string log;

  EXPECT_EQ(ncclInvalidUsage, Launch(&log));

  EXPECT_TRUE(LogHas(log, kLaunchNoStagingWarn)) << log;
}

TEST_F(CeCollMicrotest, LaunchNonStagingCollWithoutStaging_SkipsTheStagingCheck) {
  args_.func = ncclFuncBroadcast;  // no CE implementation, so the dispatch default arm rejects it after the check
  uint8_t userRecv = 0;
  args_.useDda = true;  // the DDA copy-back runs only past the staging check
  args_.ddaUserRecvBuff = &userRecv;
  ScopedHook copyBack(g_hipMemcpyAsync,
                      [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipSuccess; });
  std::string log;

  EXPECT_EQ(ncclInvalidUsage, Launch(&log));

  EXPECT_FALSE(LogHas(log, kLaunchNoStagingWarn)) << log;
  EXPECT_EQ(1, copyBack.calls);
}

TEST_F(CeCollMicrotest, LaunchAllReduceWithStaging_ReachesTheCollectiveWithoutRegistering) {
  comm_->ceColl.ceARTmpBuf = &staging_;
  args_.nElts = 0;  // stops ncclCeAllReduce at its layout check, before any copy is issued
  std::string log;

  EXPECT_EQ(ncclInvalidArgument, Launch(&log));

  EXPECT_TRUE(LogHas(log, "CE AllReduce: no valid chunk layout")) << log;
}

// A single-rank CE AllGather that reaches the copy path, logging every stream operation.
class CeCollDoneEventMicrotest : public CeCollMicrotest {
 protected:
  static constexpr uintptr_t kCopyStreamBase = 0x5300;
  static constexpr uintptr_t kCopyEventBase = 0x5400;
  hipStream_t const launchStream_ = reinterpret_cast<hipStream_t>(0x5100);
  hipEvent_t const doneEvent_ = reinterpret_cast<hipEvent_t>(0x5200);
  int lsaRankList_[1] = {0};
  uint32_t syncFlags_[2] = {};
  ncclDevrWindow syncWin_{};
  float send_ = 1.0f;
  float recv_ = 0.0f;

  struct StreamOp {
    enum Kind { kCopy, kSync, kRecord, kWait } kind;
    hipStream_t stream;
    hipEvent_t event;
  };
  std::vector<StreamOp> ops_;
  int copies_ = 0;
  int syncs_ = 0;
  int failCopy_ = 0;  // 1-based; 0 never fails
  bool failSyncAfterCopy_ = false;
  bool failDoneRecord_ = false;

  ScopedHook<hipError_t(void*, const void*, size_t, hipMemcpyKind, hipStream_t)> copyHook_{
      g_hipMemcpyAsync, [this](void*, const void*, size_t, hipMemcpyKind, hipStream_t stream) {
        ops_.push_back({StreamOp::kCopy, stream, nullptr});
        return ++copies_ == failCopy_ ? hipErrorInvalidValue : hipSuccess;
      }};
  ScopedHook<hipError_t(hipStream_t, unsigned int, hipStreamBatchMemOpParams*, unsigned int)> syncHook_{
      g_hipStreamBatchMemOp, [this](hipStream_t stream, unsigned int, hipStreamBatchMemOpParams*, unsigned int) {
        ops_.push_back({StreamOp::kSync, stream, nullptr});
        ++syncs_;
        return failSyncAfterCopy_ && copies_ > 0 ? hipErrorInvalidValue : hipSuccess;
      }};
  ScopedHook<hipError_t(hipEvent_t, hipStream_t)> recordHook_{
      g_hipEventRecord, [this](hipEvent_t event, hipStream_t stream) {
        ops_.push_back({StreamOp::kRecord, stream, event});
        return failDoneRecord_ && event == doneEvent_ ? hipErrorInvalidValue : hipSuccess;
      }};
  ScopedHook<hipError_t(hipStream_t, hipEvent_t, unsigned int)> waitHook_{
      g_hipStreamWaitEvent, [this](hipStream_t stream, hipEvent_t event, unsigned int) {
        ops_.push_back({StreamOp::kWait, stream, event});
        return hipSuccess;
      }};

  void SetUp() override {
    CeCollMicrotest::SetUp();
    streams_.stream = launchStream_;
    comm_->planner.capturingGraph.graphId = ULLONG_MAX;  // not capturing
    comm_->doneEvent = doneEvent_;
    comm_->nRanks = 1;
    comm_->devrState.lsaSize = 1;
    comm_->devrState.lsaSelf = 0;
    comm_->devrState.lsaRankList = lsaRankList_;
    syncWin_.userPtr = syncFlags_;
    comm_->ceColl.baseUCSymReadyPtr = reinterpret_cast<uint8_t*>(&syncFlags_[0]);
    comm_->ceColl.baseUCSymComplPtr = reinterpret_cast<uint8_t*>(&syncFlags_[1]);
    comm_->ceColl.ceSyncWin = &syncWin_;
    comm_->ceColl.intraBatchSyncFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
    comm_->ceColl.intraBatchSyncMsgThreshold = CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD;
    args_.func = ncclFuncAllGather;
    args_.nElts = 1;
    args_.sendBuff = reinterpret_cast<uint8_t*>(&send_);
    args_.recvBuff = reinterpret_cast<uint8_t*>(&recv_);
  }

  int DoneRecords() const {
    return static_cast<int>(std::count_if(ops_.begin(), ops_.end(), [this](const StreamOp& op) {
      return op.kind == StreamOp::kRecord && op.event == doneEvent_;
    }));
  }
  // doneEvent must be the last operation on the launch stream, so a wait on it covers every copy.
  void ExpectDoneEventRecordedLast() const {
    ASSERT_FALSE(ops_.empty());
    EXPECT_EQ(StreamOp::kRecord, ops_.back().kind);
    EXPECT_EQ(doneEvent_, ops_.back().event);
    EXPECT_EQ(launchStream_, ops_.back().stream);
    EXPECT_EQ(1, DoneRecords());
  }
};

TEST_F(CeCollDoneEventMicrotest, LaunchSucceeds_RecordsDoneEventAndStreamTagLast) {
  std::string log;

  ASSERT_EQ(ncclSuccess, Launch(&log)) << log;

  EXPECT_GT(copies_, 0) << "the launch must reach the CE copy path";
  ExpectDoneEventRecordedLast();
  EXPECT_EQ(ncclStreamTag(launchStream_), comm_->lastStreamTag);
}

TEST_F(CeCollDoneEventMicrotest, LaunchFailsAfterQueuingCopies_StillRecordsDoneEventLast) {
  failSyncAfterCopy_ = true;  // the completion barrier, after the copies are queued
  std::string log;

  EXPECT_NE(ncclSuccess, Launch(&log));

  EXPECT_GT(copies_, 0) << "the failure must come after the CE copies";
  ExpectDoneEventRecordedLast();
  EXPECT_EQ(ncclStreamTag(launchStream_), comm_->lastStreamTag);
}

TEST_F(CeCollDoneEventMicrotest, LaunchRejectedBeforeAnyCopy_StillRecordsDoneEvent) {
  args_.func = ncclFuncBroadcast;  // rejected by the dispatch default arm
  std::string log;

  EXPECT_EQ(ncclInvalidUsage, Launch(&log));

  EXPECT_EQ(0, copies_);
  ExpectDoneEventRecordedLast();
}

TEST_F(CeCollDoneEventMicrotest, DoneEventRecordFails_FailsTheLaunchAndKeepsTheStreamTag) {
  failDoneRecord_ = true;
  std::string log;

  EXPECT_EQ(ncclUnhandledCudaError, Launch(&log));

  EXPECT_TRUE(LogHas(log, "doneEvent record failed")) << log;
  EXPECT_EQ(0u, comm_->lastStreamTag);
}

TEST_F(CeCollDoneEventMicrotest, MultiStreamCopyFails_RejoinsForkedCopyStreamsToTheLaunchStream) {
  constexpr int kStreams = 2;
  comm_->ceColl.nCopyStreams = kStreams;
  for (int s = 0; s < kStreams; ++s) {
    comm_->ceColl.copyStreams[s] = reinterpret_cast<hipStream_t>(kCopyStreamBase + s);
    comm_->ceColl.copyEvents[s] = reinterpret_cast<hipEvent_t>(kCopyEventBase + s);
  }
  uint8_t src[3] = {}, dst[3] = {};
  void* srcs[3] = {&src[0], &src[1], &src[2]};
  void* dsts[3] = {&dst[0], &dst[1], &dst[2]};
  size_t sizes[3] = {1, 1, 1};
  ncclCeBatchOpsParams params = {};
  params.srcs = srcs;
  params.dsts = dsts;
  params.sizes = sizes;
  params.numOps = 3;
  failCopy_ = 3;  // copies 1 and 2 are already queued on both copy streams

  EXPECT_NE(ncclSuccess, ncclCeLaunchBatchOps(comm_.get(), &params, launchStream_, &args_));

  ASSERT_EQ(3, copies_) << "the multi-stream path must be taken";
  auto failed = std::find_if(ops_.rbegin(), ops_.rend(), [](const StreamOp& op) {
                  return op.kind == StreamOp::kCopy;
                }).base();
  for (int s = 0; s < kStreams; ++s) {
    hipStream_t const copyStream = comm_->ceColl.copyStreams[s];
    hipEvent_t const copyEvent = comm_->ceColl.copyEvents[s];
    auto rec = std::find_if(failed, ops_.end(), [&](const StreamOp& op) {
      return op.kind == StreamOp::kRecord && op.stream == copyStream && op.event == copyEvent;
    });
    ASSERT_NE(ops_.end(), rec) << "copy stream " << s << " is not joined after the failure";
    auto wait = std::find_if(rec, ops_.end(), [&](const StreamOp& op) {
      return op.kind == StreamOp::kWait && op.stream == launchStream_ && op.event == copyEvent;
    });
    EXPECT_NE(ops_.end(), wait) << "launch stream does not wait on copy stream " << s;
  }
}

// The eager 2-shot path calls ncclCeAllReduce directly, bypassing ncclLaunchCeColl.
TEST_F(CeCollMicrotest, CeAllReduceWithoutStaging_FailsLoudlyWithoutRegistering) {
  std::string log;

  EXPECT_EQ(ncclInvalidUsage, Run([&]() {
    return ncclCeAllReduce(comm_.get(), nullptr, nullptr, 4, ncclFloat32, ncclSum, nullptr);
  }, &log));

  EXPECT_TRUE(LogHas(log, "CE AllReduce staging is not available")) << log;
}

TEST_F(CeCollMicrotest, EnsureStaging_PublishesTheRegisteredWindowOnce) {
  registerResult_ = ncclSuccess;
  expectedRegistrations_ = 1;

  EXPECT_EQ(ncclSuccess, ncclCeEnsureAllReduceStaging(comm_.get()));
  EXPECT_EQ(ncclSuccess, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(1, alloc_.calls);
  EXPECT_EQ(NCCL_CE_NUM_SLOTS * comm_->ceColl.ceArStagingBytes, allocSize_);
  EXPECT_EQ(&devrWin_, comm_->ceColl.ceARTmpWin);
  EXPECT_EQ(&staging_, comm_->ceColl.ceARTmpBuf);
  EXPECT_EQ(0, free_.calls);
}

TEST_F(CeCollMicrotest, EnsureStaging_CeAllReduceDisabled_LeavesStagingUnset) {
  // Staging is shared by CE AllReduce and CE ReduceScatter; both params must be off.
  g_rcclParamCeAllReduce = 0;
  g_rcclParamCeReduceScatter = 0;

  EXPECT_EQ(ncclSuccess, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(nullptr, comm_->ceColl.ceARTmpBuf);
  EXPECT_EQ(0, alloc_.calls);
}

TEST_F(CeCollMicrotest, EnsureStaging_CeReduceScatterEnabled_AllocatesWhenAllReduceDisabled) {
  registerResult_ = ncclSuccess;
  expectedRegistrations_ = 1;
  g_rcclParamCeAllReduce = 0;
  g_rcclParamCeReduceScatter = 1;

  EXPECT_EQ(ncclSuccess, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(1, alloc_.calls);
  EXPECT_EQ(&devrWin_, comm_->ceColl.ceARTmpWin);
  EXPECT_EQ(&staging_, comm_->ceColl.ceARTmpBuf);
}

TEST_F(CeCollMicrotest, EnsureStaging_RegistrationFails_FreesTheBufferAndReturnsTheError) {
  expectedRegistrations_ = 1;

  EXPECT_EQ(ncclInternalError, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(1, free_.calls);
  EXPECT_EQ(nullptr, comm_->ceColl.ceARTmpBuf);
  EXPECT_EQ(nullptr, comm_->ceColl.ceARTmpWin);
}

TEST_F(CeCollMicrotest, EnsureStaging_ToHostFails_DeregistersAndFreesOnceAndReturnsTheError) {
  registerResult_ = ncclSuccess;
  toHostResult_ = ncclSystemError;
  expectedRegistrations_ = 1;
  ScopedHook deregister(g_devrNcclCommWindowDeregister, [this](ncclComm_t comm, ncclWindow_t win) {
    EXPECT_EQ(comm_.get(), comm);
    EXPECT_EQ(&devVidmem_, win);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSystemError, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(1, deregister.calls);
  EXPECT_EQ(1, free_.calls);
  EXPECT_EQ(nullptr, comm_->ceColl.ceARTmpBuf);
  EXPECT_EQ(nullptr, comm_->ceColl.ceARTmpWin);
}

TEST_F(CeCollMicrotest, EnsureStaging_AlreadySetUp_ReturnsWithoutRegistering) {
  comm_->ceColl.ceARTmpBuf = &allocated_;

  EXPECT_EQ(ncclSuccess, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(&allocated_, comm_->ceColl.ceARTmpBuf);
  EXPECT_EQ(0, alloc_.calls);
}

// ncclCeInit on 2 ranks with ncclMemAlloc returning stale 1s, the value the first UC sync waits for.
class CeInitMicrotest : public ::testing::Test {
 protected:
  static constexpr uint32_t kStaleFlag = 1;
  struct Allocation {
    std::vector<uint32_t> words;
    std::vector<uint32_t> atRegistration;  // after the registration's barrier, peers may signal into the window
    ncclWindow_vidmem vidmem{};
    ncclDevrWindow win{};
  };
  // A device fill lands when its stream is synchronized. hipMemset of device memory only queues on the null stream.
  struct Fill {
    hipStream_t stream;
    void* dst;
    int value;
    size_t bytes;
  };
  Allocation allocs_[2];  // the signal buffer, then the ready/complete flag window
  int nAllocs_ = 0;
  std::vector<Fill> queued_;
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  ScopedHook<ncclResult_t(void**, size_t)> alloc_{g_ncclMemAlloc, [this](void** ptr, size_t size) {
    if (nAllocs_ == 2) return ncclInternalError;
    Allocation& a = allocs_[nAllocs_++];
    a.words.assign((size + sizeof(uint32_t) - 1) / sizeof(uint32_t), kStaleFlag);
    *ptr = a.words.data();
    return ncclSuccess;
  }};
  ScopedHook<ncclResult_t(ncclComm*, void*, size_t, int, ncclWindow_t*)> registration_{
      g_devrWindowRegisterInGroup, [this](ncclComm*, void* ptr, size_t, int, ncclWindow_t* win) {
        Allocation* a = Find(ptr);
        if (a == nullptr) return ncclInvalidArgument;
        a->atRegistration = a->words;
        a->win.userPtr = ptr;
        a->vidmem.winHost = &a->win;
        *win = &a->vidmem;
        return ncclSuccess;
      }};
  ScopedHook<ncclResult_t(ncclShadowPool*, void*, void**)> toHost_{
      g_shadowPoolToHost, [](ncclShadowPool*, void* devObj, void** hostObj) {
        *hostObj = devObj;
        return ncclSuccess;
      }};
  ScopedHook<size_t(const ncclComm*)> nonRegCap_{g_rcclCeArNonRegCap, [](const ncclComm*) { return size_t{0}; }};
  ScopedHook<hipError_t(void*, int, size_t)> memset_{g_hipMemset, [this](void* dst, int value, size_t bytes) {
    queued_.push_back({nullptr, dst, value, bytes});
    return hipSuccess;
  }};
  ScopedHook<hipError_t(void*, int, size_t, hipStream_t)> memsetAsync_{
      g_hipMemsetAsync, [this](void* dst, int value, size_t bytes, hipStream_t stream) {
        queued_.push_back({stream, dst, value, bytes});
        return hipSuccess;
      }};
  ScopedHook<hipError_t(hipStream_t)> streamSync_{g_hipStreamSynchronize, [this](hipStream_t stream) {
    Land(stream);
    return hipSuccess;
  }};

  void SetUp() override {
    comm_->nRanks = 2;
    // The rest of ncclCeInit: its stream and event, and the seq-number buffer it fills through a side stream.
    g_hipStreamCreateResult = hipSuccess;
    g_hipEventCreateResult = hipSuccess;
    g_hipAsyncOpsResult = hipSuccess;
    g_hipDeviceGetPCIBusIdResult = hipSuccess;
  }
  void TearDown() override {
    EXPECT_EQ(hipSuccess, hipFree(comm_->ceColl.d_barrierSync));
    EXPECT_EQ(hipSuccess, hipFree(comm_->ceColl.ceSeqNumDev));
    ResetHipFakes();
  }
  Allocation* Find(const void* ptr) {
    for (Allocation& a : allocs_) {
      if (!a.words.empty() && a.words.data() == ptr) return &a;
    }
    return nullptr;
  }
  void Land(hipStream_t stream) {
    for (auto it = queued_.begin(); it != queued_.end();) {
      if (it->stream != stream) {
        ++it;
        continue;
      }
      std::memset(it->dst, it->value, it->bytes);
      it = queued_.erase(it);
    }
  }
};

TEST_F(CeInitMicrotest, StaleSyncFlags_ClearsThemBeforeRegisteringTheWindow) {
  ASSERT_EQ(ncclSuccess, ncclCeInit(comm_.get()));

  const Allocation* flags = Find(comm_->ceColl.ceSyncWin->userPtr);
  ASSERT_NE(nullptr, flags);
  EXPECT_EQ(std::vector<uint32_t>(flags->words.size(), 0), flags->atRegistration)
      << "a stale " << kStaleFlag << " lets the first UC sync pass before its peer arrives";
}

TEST_F(CeInitMicrotest, StaleSignalBuffer_ClearsIt) {
  ASSERT_EQ(ncclSuccess, ncclCeInit(comm_.get()));
  Land(nullptr);

  const Allocation* signal = Find(comm_->ceColl.signalBuffer);
  ASSERT_NE(nullptr, signal);
  EXPECT_EQ(std::vector<uint32_t>(signal->words.size(), 0), signal->words);
  EXPECT_NE(nullptr, Find(comm_->ceColl.ceSyncWin->userPtr));
}

constexpr size_t kMiB = 1024 * 1024;
constexpr uintptr_t kSendBase = 0x100000000000;
constexpr uintptr_t kRecvBase = 0x200000000000;
constexpr uintptr_t kPeerWindowBase = 0x300000000000;
constexpr uintptr_t kPeerWindowStride = 0x10000000000;
constexpr unsigned long long kCaptureGraphId = 7;
constexpr int kBatchAsyncDriverVersion = ROCM_VER_7_12_0;

enum class LaunchPath { kSingleStream, kMultiStream, kBatchAsync, kLegacyNullStream, kGraphCapture };

std::string LaunchPathName(const ::testing::TestParamInfo<LaunchPath>& info) {
  switch (info.param) {
    case LaunchPath::kSingleStream:
      return "SingleStream";
    case LaunchPath::kMultiStream:
      return "MultiStream";
    case LaunchPath::kBatchAsync:
      return "BatchAsync";
    case LaunchPath::kLegacyNullStream:
      return "LegacyNullStream";
    case LaunchPath::kGraphCapture:
      return "GraphCapture";
  }
  return "Unknown";
}

using SizeMatrix = std::vector<std::vector<size_t>>;

// Row r sends totalBytes over peerCounts[r] consecutive destinations from itself; the first takes any remainder.
SizeMatrix EqualTotalsMatrix(const std::vector<int>& peerCounts, size_t totalBytes) {
  const int nRanks = static_cast<int>(peerCounts.size());
  SizeMatrix send(nRanks, std::vector<size_t>(nRanks, 0));
  for (int src = 0; src < nRanks; ++src) {
    for (int j = 0; j < peerCounts[src]; ++j) {
      send[src][(src + j) % nRanks] = totalBytes / peerCounts[src] + (j == 0 ? totalBytes % peerCounts[src] : 0);
    }
  }
  return send;
}

std::vector<int> CyclingPeerCounts(int nRanks, int minPeers, int span) {
  std::vector<int> counts(nRanks);
  for (int r = 0; r < nRanks; ++r) {
    counts[r] = minPeers + r % span;
  }
  return counts;
}

std::vector<size_t> GatherSizes(const SizeMatrix& send) {
  const int nRanks = static_cast<int>(send.size());
  std::vector<size_t> gathered(4 * static_cast<size_t>(nRanks) * nRanks);
  for (int rank = 0; rank < nRanks; ++rank) {
    std::vector<size_t> sendDispls(nRanks);
    std::vector<size_t> recvSizes(nRanks);
    std::vector<size_t> recvDispls(nRanks);
    size_t sendOff = 0;
    size_t recvOff = 0;
    for (int peer = 0; peer < nRanks; ++peer) {
      sendDispls[peer] = sendOff;
      sendOff += send[rank][peer];
      recvSizes[peer] = send[peer][rank];
      recvDispls[peer] = recvOff;
      recvOff += recvSizes[peer];
    }
    ncclAlltoAllvPackLocalSizes(ncclAlltoAllvRankMetaBlock(gathered.data(), rank, nRanks), nRanks,
                                send[rank].data(), sendDispls.data(), recvSizes.data(), recvDispls.data());
  }
  return gathered;
}

struct RankRun {
  uint32_t barriers = 0;
  size_t copies = 0;
  size_t bytes = 0;
  size_t batchCopies = 0;
  size_t copyStreamCopies = 0;
  size_t copyCalls = 0;
  int syncCalls = 0;
  std::vector<uintptr_t> destinations;
  // Barriers already issued when the copy to each destination slot was enqueued; 0 for a slot with nothing to send.
  std::vector<uint32_t> slotEpochs;
};

constexpr uintptr_t kCopyStreamBase = 0x6000;

class CeAlltoAllvSyncMicrotest : public ::testing::TestWithParam<LaunchPath> {
 protected:
  void SetUp() override {
    ResetAllFakes();
#ifndef CE_BATCH_ASYNC_SUPPORTED
    if (GetParam() == LaunchPath::kBatchAsync) {
      GTEST_SKIP() << "hip_runtime_api.h has no hipMemcpyBatchAsync, so ce_coll.cc builds without the batch path";
    }
#endif
    g_hipStreamBatchMemOp = [this](hipStream_t, unsigned int, hipStreamBatchMemOpParams*, unsigned int) {
      return ++current_.syncCalls == failSyncCall_ ? hipErrorInvalidValue : hipSuccess;
    };
    g_hipMemcpyAsync = [this](void* dst, const void*, size_t bytes, hipMemcpyKind, hipStream_t stream) {
      if (++current_.copyCalls == failCopyCall_) {
        return hipErrorInvalidValue;
      }
      RecordCopy(dst, bytes);
      const uintptr_t s = reinterpret_cast<uintptr_t>(stream);
      if (s >= kCopyStreamBase && s < kCopyStreamBase + RCCL_CE_NUM_COPY_STREAMS) {
        ++current_.copyStreamCopies;
      }
      return hipSuccess;
    };
#ifdef CE_BATCH_ASYNC_SUPPORTED
    g_hipMemcpyBatchAsync = [this](void** dsts, void**, size_t* sizes, size_t count, hipMemcpyAttributes*,
                                   size_t*, size_t, size_t*, hipStream_t) {
      if (++current_.copyCalls == failCopyCall_) {
        return hipErrorInvalidValue;
      }
      for (size_t i = 0; i < count; ++i) {
        RecordCopy(dsts[i], sizes[i]);
      }
      current_.batchCopies += count;
      return hipSuccess;
    };
#endif
    g_hipEventRecord = [](hipEvent_t, hipStream_t) { return hipSuccess; };
    g_hipStreamWaitEvent = [](hipStream_t, hipEvent_t, unsigned int) { return hipSuccess; };
    g_devrGetLsaRankPtr = [](ncclComm* comm, ncclDevrWindow*, size_t offset, int lsaRank, void** outPtr) {
      if (lsaRank < 0 || lsaRank >= comm->devrState.lsaSize) {
        return ncclInvalidArgument;
      }
      *outPtr = reinterpret_cast<void*>(kPeerWindowBase + lsaRank * kPeerWindowStride + offset);
      return ncclSuccess;
    };
    const bool batchAsync = GetParam() == LaunchPath::kBatchAsync;
    g_loadParam = [batchAsync](const char* env, int64_t deftVal) -> int64_t {
      if (std::strcmp(env, "RCCL_CE_BATCH_ASYNC_ENABLE") == 0) {
        return batchAsync ? 1 : 0;
      }
      return deftVal;
    };
    if (batchAsync) {
      ncclCudaDriverVersionCache = kBatchAsyncDriverVersion;
    }
  }

  void TearDown() override { ResetAllFakes(); }

  static void ResetAllFakes() {
    ResetHipFakes();
    ResetDevRuntimeFakes();
    ResetNcclFakes();
    ResetNcclStubs();
    ResetEnvFakes();
  }

  void RecordCopy(void* dst, size_t bytes) {
    ++current_.copies;
    current_.bytes += bytes;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(dst);
    current_.destinations.push_back(addr);
    const int nRanks = activeComm_->nRanks;
    const int dstRank =
        addr >= kPeerWindowBase ? static_cast<int>((addr - kPeerWindowBase) / kPeerWindowStride) : activeComm_->rank;
    current_.slotEpochs[(dstRank - activeComm_->rank + nRanks) % nRanks] = activeComm_->ceColl.ceSeqNum;
  }

  // Runs ncclCeAlltoAllv once per simulated rank; every rank sees the same gathered size matrix, as in production.
  std::vector<RankRun> Run(const SizeMatrix& send, uint32_t freq, ncclResult_t expected = ncclSuccess) {
    const int nRanks = static_cast<int>(send.size());
    std::vector<size_t> gathered = GatherSizes(send);
    if (corruptGathered_) {
      corruptGathered_(gathered);
    }
    std::vector<int> lsaRankList(nRanks);
    for (int r = 0; r < nRanks; ++r) {
      lsaRankList[r] = r;
    }
    std::vector<uint32_t> syncFlags(2 * nRanks, 0);
    ncclDevrWindow syncWin = {};
    syncWin.userPtr = syncFlags.data();
    ncclDevrWindow recvWin = {};
    recvWin.userPtr = reinterpret_cast<void*>(kRecvBase);

    const LaunchPath path = GetParam();
    hipStream_t stream = path == LaunchPath::kLegacyNullStream ? nullptr : reinterpret_cast<hipStream_t>(0x5717);
    std::vector<RankRun> runs;
    for (int rank = 0; rank < nRanks; ++rank) {
      auto comm = std::make_unique<ncclComm>();
      comm->rank = rank;
      comm->nRanks = nRanks;
      comm->devrState.lsaSize = nRanks;
      comm->devrState.lsaSelf = rank;
      comm->devrState.lsaRankList = lsaRankList.data();
      comm->symkState.hasLsaMultimem = false;
      comm->planner.capturingGraph.graphId = path == LaunchPath::kGraphCapture ? kCaptureGraphId : ULLONG_MAX;
      comm->ceColl.intraBatchSyncFreq = freq;
      comm->ceColl.intraBatchSyncMsgThreshold = CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD;
      comm->ceColl.baseUCSymReadyPtr = reinterpret_cast<uint8_t*>(syncFlags.data());
      comm->ceColl.baseUCSymComplPtr = reinterpret_cast<uint8_t*>(syncFlags.data() + nRanks);
      comm->ceColl.ceSyncWin = &syncWin;
      if (path == LaunchPath::kMultiStream) {
        comm->ceColl.nCopyStreams = RCCL_CE_NUM_COPY_STREAMS;
        for (int s = 0; s < RCCL_CE_NUM_COPY_STREAMS; ++s) {
          comm->ceColl.copyStreams[s] = reinterpret_cast<hipStream_t>(kCopyStreamBase + s);
          comm->ceColl.copyEvents[s] = reinterpret_cast<hipEvent_t>(0x7000 + s);
        }
      }

      ncclCeCollArgs args = {};
      args.func = ncclFuncAlltoAllv;
      args.sendBuff = reinterpret_cast<uint8_t*>(kSendBase);
      args.recvBuff = reinterpret_cast<uint8_t*>(kRecvBase);
      args.recvWin = &recvWin;
      args.sizes = gathered.data();

      current_ = RankRun{};
      current_.slotEpochs.assign(nRanks, 0);
      activeComm_ = comm.get();
      EXPECT_EQ(ncclCeAlltoAllv(comm.get(), &args, stream), expected) << "rank " << rank;
      activeComm_ = nullptr;
      current_.barriers = comm->ceColl.ceSeqNum;
      runs.push_back(current_);
    }
    return runs;
  }

  static std::vector<uint32_t> Barriers(const std::vector<RankRun>& runs) {
    std::vector<uint32_t> barriers;
    for (const RankRun& run : runs) {
      barriers.push_back(run.barriers);
    }
    return barriers;
  }

  // Each rank copies exactly its nonzero row, slot s after the ready barrier and s / roundSlots round barriers.
  static void ExpectRowsCopied(const SizeMatrix& send, const std::vector<RankRun>& runs, uint32_t roundSlots) {
    const int nRanks = static_cast<int>(send.size());
    for (int rank = 0; rank < nRanks; ++rank) {
      size_t copies = 0;
      size_t bytes = 0;
      std::vector<uint32_t> epochs(nRanks, 0);
      for (int slot = 0; slot < nRanks; ++slot) {
        const size_t chunk = send[rank][(rank + slot) % nRanks];
        if (chunk != 0) {
          ++copies;
          bytes += chunk;
          epochs[slot] = 1 + slot / roundSlots;
        }
      }
      EXPECT_EQ(runs[rank].copies, copies) << "rank " << rank;
      EXPECT_EQ(runs[rank].bytes, bytes) << "rank " << rank;
      EXPECT_EQ(runs[rank].slotEpochs, epochs) << "rank " << rank;
    }
  }

  // Copies in rounds of two or more ops; a one-op round takes the single-stream fallback even with copy streams.
  static size_t MultiOpRoundCopies(const RankRun& run) {
    std::map<uint32_t, size_t> roundOps;
    for (uint32_t epoch : run.slotEpochs) {
      if (epoch != 0) {
        ++roundOps[epoch];
      }
    }
    size_t copies = 0;
    for (const auto& [epoch, ops] : roundOps) {
      if (ops > 1) {
        copies += ops;
      }
    }
    return copies;
  }

  // Guards the parameter itself: each arm must really reach the copy API it is named for.
  void ExpectLaunchPathTaken(const std::vector<RankRun>& runs) {
    for (size_t rank = 0; rank < runs.size(); ++rank) {
      const size_t expectedBatch = GetParam() == LaunchPath::kBatchAsync ? runs[rank].copies : 0;
      EXPECT_EQ(runs[rank].batchCopies, expectedBatch) << "rank " << rank;
      const size_t expectedCopyStream = GetParam() == LaunchPath::kMultiStream ? MultiOpRoundCopies(runs[rank]) : 0;
      EXPECT_EQ(runs[rank].copyStreamCopies, expectedCopyStream) << "rank " << rank;
    }
  }

  // Ready and complete barriers around the transfers, plus one per further round of freq destination slots.
  static uint32_t RoundSyncBarriers(int nRanks, uint32_t freq) { return 2 + (nRanks - 1) / freq; }

  // Graph capture rounds up to an even count, so back-to-back replays keep alternating the ready/complete flags.
  void ExpectRoundSyncBarriers(const std::vector<RankRun>& runs, int nRanks, uint32_t freq) {
    uint32_t expected = RoundSyncBarriers(nRanks, freq);
    if (GetParam() == LaunchPath::kGraphCapture && expected % 2 != 0) {
      ++expected;
    }
    EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(nRanks, expected));
  }

  // hipStreamBatchMemOp calls per barrier: the wait batch, plus the flag write and the reset batch under graph capture.
  int SyncCallsPerBarrier() const { return GetParam() == LaunchPath::kGraphCapture ? 3 : 1; }

  RankRun current_;
  ncclComm* activeComm_ = nullptr;
  size_t failCopyCall_ = 0;  // 1-based copy API call on each rank that fails; 0 never fails
  int failSyncCall_ = 0;     // 1-based hipStreamBatchMemOp call on each rank that fails; 0 never fails
  std::function<void(std::vector<size_t>&)> corruptGathered_;  // edits the gathered matrix before any rank runs
};

TEST_P(CeAlltoAllvSyncMicrotest, Sparse16Ranks_EqualTotalsAt512MiB_EveryRankIssuesSameBarrierCount) {
  const int kNRanks = 16;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  const SizeMatrix send = EqualTotalsMatrix(CyclingPeerCounts(kNRanks, 6, 5), CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD);

  const std::vector<RankRun> runs = Run(send, kFreq);

  ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
  ExpectRowsCopied(send, runs, kFreq);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, Sparse32Ranks_EveryRankOverFreqButUnevenRounds_EveryRankIssuesSameBarrierCount) {
  const int kNRanks = 32;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  const SizeMatrix send = EqualTotalsMatrix(CyclingPeerCounts(kNRanks, 12, 7), 12252240ull * 64);

  const std::vector<RankRun> runs = Run(send, kFreq);

  ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
  ExpectRowsCopied(send, runs, kFreq);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, Sparse4RanksFreq2_EqualTotalsOver512MiB_EveryRankIssuesSameBarrierCount) {
  const int kNRanks = 4;
  const uint32_t kFreq = 2;
  const SizeMatrix send = EqualTotalsMatrix({2, 3, 4, 2}, 12 * 48 * kMiB);

  const std::vector<RankRun> runs = Run(send, kFreq);

  ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
  ExpectRowsCopied(send, runs, kFreq);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, OneHeavyRank_OthersUnderThreshold_EveryRankJoinsTheRoundBarriers) {
  const int kNRanks = 16;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  for (int heavyRank : {0, kNRanks - 1}) {
    SCOPED_TRACE("heavy rank " + std::to_string(heavyRank));
    SizeMatrix send = EqualTotalsMatrix(std::vector<int>(kNRanks, 4), 4 * kMiB);
    for (int j = 0; j < 10; ++j) {
      send[heavyRank][(heavyRank + j) % kNRanks] = 64 * kMiB;
    }

    const std::vector<RankRun> runs = Run(send, kFreq);

    ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
    ExpectRowsCopied(send, runs, kFreq);
    ExpectLaunchPathTaken(runs);
  }
}

TEST_P(CeAlltoAllvSyncMicrotest, Sparse16Ranks_TotalsUnder512MiB_OnlyReadyAndCompleteBarriers) {
  const int kNRanks = 16;
  const SizeMatrix send = EqualTotalsMatrix(CyclingPeerCounts(kNRanks, 6, 5), 2520 * 100 * 1024);

  const std::vector<RankRun> runs = Run(send, CE_COLL_INTRA_BATCH_SYNC_FREQ);

  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, 2));
  ExpectRowsCopied(send, runs, kNRanks);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, ManyPeersEachUnder512MiB_NoRankCrossesBothGates_OnlyReadyAndCompleteBarriers) {
  const int kNRanks = 16;
  const SizeMatrix send = EqualTotalsMatrix(std::vector<int>(kNRanks, kNRanks), kNRanks * 31 * kMiB);

  const std::vector<RankRun> runs = Run(send, CE_COLL_INTRA_BATCH_SYNC_FREQ);

  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, 2));
  ExpectRowsCopied(send, runs, kNRanks);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, Dense16Ranks_Over512MiB_EveryRankIssuesSameBarrierCount) {
  const int kNRanks = 16;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  const SizeMatrix send = EqualTotalsMatrix(std::vector<int>(kNRanks, kNRanks), kNRanks * 40 * kMiB);

  const std::vector<RankRun> runs = Run(send, kFreq);

  ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
  ExpectRowsCopied(send, runs, kFreq);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, SixteenRanks_EachSendsToExactlyFreqPeers_OnlyReadyAndCompleteBarriers) {
  const int kNRanks = 16;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  const SizeMatrix send = EqualTotalsMatrix(std::vector<int>(kNRanks, kFreq), 1024 * kMiB);

  const std::vector<RankRun> runs = Run(send, kFreq);

  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, 2));
  ExpectRowsCopied(send, runs, kNRanks);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, SeventeenRanksFreq8_EvenRoundsPartialLastRound_EveryRankIssuesSameBarrierCount) {
  const int kNRanks = 17;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  const SizeMatrix send = EqualTotalsMatrix(CyclingPeerCounts(kNRanks, 9, 9), CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD);

  const std::vector<RankRun> runs = Run(send, kFreq);

  ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
  ExpectRowsCopied(send, runs, kFreq);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, OneRankSendsNothing_OthersCrossBothGates_SilentRankIssuesSameBarrierCount) {
  const int kNRanks = 16;
  const int kSilentRank = 5;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  SizeMatrix send = EqualTotalsMatrix(CyclingPeerCounts(kNRanks, 9, 5), CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD);
  send[kSilentRank].assign(kNRanks, 0);

  const std::vector<RankRun> runs = Run(send, kFreq);

  ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
  ExpectRowsCopied(send, runs, kFreq);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, FirstRoundCopyFails_ReturnsTheErrorBeforeTheRoundBarrier) {
  const int kNRanks = 16;
  const SizeMatrix send = EqualTotalsMatrix(CyclingPeerCounts(kNRanks, 9, 5), CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD);
  failCopyCall_ = 1;

  const std::vector<RankRun> runs = Run(send, CE_COLL_INTRA_BATCH_SYNC_FREQ, ncclUnhandledCudaError);

  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, 1));
}

TEST_P(CeAlltoAllvSyncMicrotest, GatheredRecvSizeDisagreesWithPeerSendSize_EveryRankRejectsBeforeAnyBarrierOrCopy) {
  const int kNRanks = 4;
  const int kSrc = 2;
  const int kDst = 3;
  const SizeMatrix send = EqualTotalsMatrix(std::vector<int>(kNRanks, kNRanks), kNRanks * kMiB);
  ASSERT_NE(send[kSrc][kDst], 0u);
  corruptGathered_ = [&](std::vector<size_t>& gathered) {
    ncclAlltoAllvRecvSizes(gathered.data(), kDst, kNRanks)[kSrc] = send[kSrc][kDst] + 1;
  };

  const std::vector<RankRun> runs = Run(send, CE_COLL_INTRA_BATCH_SYNC_FREQ, ncclInvalidUsage);

  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, 0));
  for (int rank = 0; rank < kNRanks; ++rank) {
    EXPECT_EQ(runs[rank].syncCalls, 0) << "rank " << rank;
    EXPECT_EQ(runs[rank].copyCalls, 0u) << "rank " << rank;
    EXPECT_EQ(runs[rank].copies, 0u) << "rank " << rank;
  }
}

TEST_P(CeAlltoAllvSyncMicrotest, SenderWritesAtTheReceiversGatheredDisplacement) {
  const int kNRanks = 4;
  const int kSrc = 2;
  const int kDst = 3;
  const SizeMatrix send = EqualTotalsMatrix(std::vector<int>(kNRanks, kNRanks), kNRanks * kMiB);
  size_t recvDisplacement = 0;
  corruptGathered_ = [&](std::vector<size_t>& gathered) {
    recvDisplacement = ++ncclAlltoAllvRecvDispls(gathered.data(), kDst, kNRanks)[kSrc];
  };

  const std::vector<RankRun> runs = Run(send, CE_COLL_INTRA_BATCH_SYNC_FREQ);

  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, 2));
  ExpectRowsCopied(send, runs, kNRanks);
  ExpectLaunchPathTaken(runs);
  const uintptr_t expectedDestination = kPeerWindowBase + kDst * kPeerWindowStride + recvDisplacement;
  EXPECT_EQ(std::count(runs[kSrc].destinations.begin(), runs[kSrc].destinations.end(), expectedDestination), 1);
}

TEST_P(CeAlltoAllvSyncMicrotest, FirstRoundBarrierFails_ReturnsTheErrorWithoutLaunchingTheNextRound) {
  const int kNRanks = 16;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  const SizeMatrix send = EqualTotalsMatrix(CyclingPeerCounts(kNRanks, 9, 5), CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD);
  failSyncCall_ = SyncCallsPerBarrier() + 1;

  const std::vector<RankRun> runs = Run(send, kFreq, ncclUnhandledCudaError);

  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, 2));
  for (int rank = 0; rank < kNRanks; ++rank) {
    EXPECT_EQ(runs[rank].copies, kFreq) << "rank " << rank;
  }
}

INSTANTIATE_TEST_SUITE_P(LaunchPaths, CeAlltoAllvSyncMicrotest,
                         ::testing::Values(LaunchPath::kSingleStream, LaunchPath::kMultiStream, LaunchPath::kBatchAsync,
                                           LaunchPath::kLegacyNullStream, LaunchPath::kGraphCapture),
                         LaunchPathName);

}  // namespace
