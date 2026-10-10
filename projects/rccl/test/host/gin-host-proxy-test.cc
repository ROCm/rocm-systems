/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the src/gin/gin_host_proxy.cc progress loop: GFD draining per
// NCCL_GIN_PROXY_POLL_BATCH and the ncclRmaOptFlagsAggregateRequests hint. Includes the
// hipified source (GIN_HOST_PROXY_CC_PATH) and records calls into a fake RMA backend.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "nccl.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

#include GIN_HOST_PROXY_CC_PATH

namespace {

enum class RmaCall { Put, PutSignal, Get, Flush };

struct RecordedCall {
  RmaCall call;
  uint32_t rank;
  uint32_t optFlags;
};

std::vector<RecordedCall> g_calls;
int g_requestToken = 0;
int g_testCalls = 0;

ncclResult_t FakeIput(void*, int, uint64_t, void*, size_t, uint64_t, void*, uint32_t rank, uint32_t optFlags,
                      void** request) {
  g_calls.push_back({RmaCall::Put, rank, optFlags});
  *request = &g_requestToken;
  return ncclSuccess;
}

ncclResult_t FakeIputSignal(void*, int, uint64_t, void*, size_t, uint64_t, void*, uint32_t rank, uint64_t, void*,
                            uint64_t, uint32_t, bool, uint32_t optFlags, void** request) {
  g_calls.push_back({RmaCall::PutSignal, rank, optFlags});
  *request = &g_requestToken;
  return ncclSuccess;
}

ncclResult_t FakeIget(void*, int, uint64_t, void*, size_t, uint64_t, void*, uint32_t rank, uint32_t optFlags,
                      void** request) {
  g_calls.push_back({RmaCall::Get, rank, optFlags});
  *request = &g_requestToken;
  return ncclSuccess;
}

// Completes immediately with no request, so the proxy must mark the op done itself.
ncclResult_t FakeIflush(void*, int, void*, uint32_t rank, void** request) {
  g_calls.push_back({RmaCall::Flush, rank, ncclRmaOptFlagsDefault});
  *request = nullptr;
  return ncclSuccess;
}

ncclResult_t FakeTest(void*, void*, int* done) {
  g_testCalls++;
  *done = 1;
  return ncclSuccess;
}

constexpr uint32_t kAgg = ncclRmaOptFlagsAggregateRequests;
constexpr uint32_t kDef = ncclRmaOptFlagsDefault;

// One GIN context whose peer queues live in host memory.
class GinHostProxyBatchTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kQueueSize = 64;

  void SetUp() override {
    ResetNcclFakes();
    g_calls.clear();
    g_testCalls = 0;
    std::memset(&rma_, 0, sizeof(rma_));
    rma_.name = "Recorder";
    rma_.iput = FakeIput;
    rma_.iputSignal = FakeIputSignal;
    rma_.iget = FakeIget;
    rma_.iflush = FakeIflush;
    rma_.test = FakeTest;
    savedBackend_ = rmaBackend;
    rmaBackend = &rma_;
  }

  void TearDown() override {
    rmaBackend = savedBackend_;
    ResetNcclFakes();
  }

  void Init(int nRanks, int pollBatch) {
    queues_.assign(static_cast<size_t>(nRanks) * kQueueSize, ncclGinProxyGfd_t{});
    states_.assign(static_cast<size_t>(nRanks) * kQueueSize, ginProxyGfdState{});
    inlines_.assign(static_cast<size_t>(nRanks) * kQueueSize, 0);
    pis_.assign(nRanks, 0);
    cis_.assign(nRanks, 0);
    cisShadow_.assign(nRanks, 0);
    sis_.assign(nRanks, 0);

    std::memset(&hostGpuCtx_, 0, sizeof(hostGpuCtx_));
    hostGpuCtx_.contextId = 0;
    hostGpuCtx_.queueSize = kQueueSize;
    hostGpuCtx_.queues = queues_.data();
    hostGpuCtx_.pis = pis_.data();
    hostGpuCtx_.cis = cis_.data();
    hostGpuCtx_.cisShadow = cisShadow_.data();
    hostGpuCtx_.sis = sis_.data();
    hostGpuCtx_.states = states_.data();
    hostGpuCtx_.inlines = inlines_.data();

    std::memset(&ctx_, 0, sizeof(ctx_));
    ctx_.nRanks = nRanks;
    ctx_.nContexts = 1;
    ctx_.hostGpuCtx = &hostGpuCtx_;
    ctx_.nSignalsPerContext = 1;
    ctx_.pollBatch = pollBatch;
  }

  // The proxy polls each qword's flag bit, so all of them must be set.
  void Enqueue(int rank, uint16_t op) {
    uint32_t idx = pis_[rank] & (kQueueSize - 1);
    ncclGinProxyGfd_t& gfd = queues_[static_cast<size_t>(rank) * kQueueSize + idx];
    for (int k = 0; k < ncclGinProxyGfdQwords; k++) gfd.qword[k].raw = 1;
    gfd.qword[ncclGinProxyGfdHeader].header.version = NCCL_GIN_PROXY_GFD_VERSION;
    gfd.qword[ncclGinProxyGfdHeader].header.size = 8;
    gfd.qword[ncclGinProxyGfdHeaderExt].headerExt.op = op;
    if (op & (ncclGinProxyOpWithSignalInc | ncclGinProxyOpWithSignalAdd)) {
      gfd.qword[ncclGinProxyGfdCompletion].completion.signalId = 1;
    }
    pis_[rank]++;
  }

  void EnqueuePuts(int rank, int n) {
    for (int i = 0; i < n; i++) Enqueue(rank, ncclGinProxyOpPut);
  }

  void Tick() { ASSERT_EQ(ncclSuccess, ncclGinProxyProgress(&ctx_)); }

  std::vector<uint32_t> FlagsFor(int rank) const {
    std::vector<uint32_t> flags;
    for (const auto& c : g_calls) {
      if (c.rank == static_cast<uint32_t>(rank)) flags.push_back(c.optFlags);
    }
    return flags;
  }

  ncclRma_t rma_;
  ncclRma_t* savedBackend_ = nullptr;
  ginProxyHostGpuCtx hostGpuCtx_;
  ginProxyCtx ctx_;
  std::vector<ncclGinProxyGfd_t> queues_;
  std::vector<ginProxyGfdState> states_;
  std::vector<uint64_t> inlines_;
  std::vector<uint32_t> pis_, cis_, cisShadow_, sis_;
};

TEST_F(GinHostProxyBatchTest, PollBatchDefaultsTo32) {
  EXPECT_EQ(32, ncclParamGinProxyPollBatch());
  g_loadParam = [](const char* env, int64_t deft) -> int64_t {
    return std::string(env) == "GIN_PROXY_POLL_BATCH" ? 4 : deft;
  };
  EXPECT_EQ(4, ncclParamGinProxyPollBatch());
}

TEST_F(GinHostProxyBatchTest, DrainsAtMostPollBatchPerTick) {
  Init(/*nRanks=*/1, /*pollBatch=*/4);
  EnqueuePuts(0, 10);

  Tick();
  EXPECT_EQ(4u, g_calls.size());
  EXPECT_EQ(4u, sis_[0]);
  Tick();
  EXPECT_EQ(8u, g_calls.size());
  Tick();
  EXPECT_EQ(10u, g_calls.size());
  Tick();
  EXPECT_EQ(10u, g_calls.size());
}

TEST_F(GinHostProxyBatchTest, PollBatchOneIsSinglePullWithoutHint) {
  Init(1, 1);
  EnqueuePuts(0, 3);

  Tick();
  EXPECT_EQ(std::vector<uint32_t>({kDef}), FlagsFor(0));
  Tick();
  Tick();
  EXPECT_EQ(std::vector<uint32_t>({kDef, kDef, kDef}), FlagsFor(0));
}

TEST_F(GinHostProxyBatchTest, HintsAggregationWhileMoreAreQueued) {
  Init(1, 32);
  EnqueuePuts(0, 5);

  Tick();
  EXPECT_EQ(std::vector<uint32_t>({kAgg, kAgg, kAgg, kAgg, kDef}), FlagsFor(0));
}

TEST_F(GinHostProxyBatchTest, LastOpOfABatchIsNeverHinted) {
  Init(1, 3);
  EnqueuePuts(0, 5);

  Tick();
  // The third op has a queued successor, but the tick ends after it.
  EXPECT_EQ(std::vector<uint32_t>({kAgg, kAgg, kDef}), FlagsFor(0));
  Tick();
  EXPECT_EQ(std::vector<uint32_t>({kAgg, kAgg, kDef, kAgg, kDef}), FlagsFor(0));
}

TEST_F(GinHostProxyBatchTest, HintReachesPutSignalAndGet) {
  Init(1, 32);
  Enqueue(0, ncclGinProxyOpPut);
  Enqueue(0, ncclGinProxyOpPut | ncclGinProxyOpWithSignalInc);
  Enqueue(0, ncclGinProxyOpGet);
  Enqueue(0, ncclGinProxyOpVASignal | ncclGinProxyOpWithSignalAdd);
  Enqueue(0, ncclGinProxyOpPut);

  Tick();
  ASSERT_EQ(5u, g_calls.size());
  const RmaCall expected[] = {RmaCall::Put, RmaCall::PutSignal, RmaCall::Get, RmaCall::PutSignal, RmaCall::Put};
  for (size_t i = 0; i < 5; i++) {
    EXPECT_EQ(expected[i], g_calls[i].call) << "call " << i;
    EXPECT_EQ(i < 4 ? kAgg : kDef, g_calls[i].optFlags) << "call " << i;
  }
}

TEST_F(GinHostProxyBatchTest, FlushEndsAHintedRun) {
  Init(1, 32);
  EnqueuePuts(0, 2);
  Enqueue(0, ncclGinProxyOpFlush);

  Tick();
  ASSERT_EQ(3u, g_calls.size());
  EXPECT_EQ(RmaCall::Put, g_calls[0].call);
  EXPECT_EQ(RmaCall::Put, g_calls[1].call);
  EXPECT_EQ(RmaCall::Flush, g_calls[2].call);
  // Both puts are hinted; the iflush that follows is what tells the backend to submit them.
  EXPECT_EQ(kAgg, g_calls[0].optFlags);
  EXPECT_EQ(kAgg, g_calls[1].optFlags);
  Tick();
  // A flush with no request is complete at issue, so only the two puts are tested.
  EXPECT_EQ(2, g_testCalls);
  EXPECT_EQ(3u, cis_[0]);
}

TEST_F(GinHostProxyBatchTest, BatchAndHintArePerPeer) {
  Init(2, 2);
  EnqueuePuts(0, 3);
  EnqueuePuts(1, 1);

  Tick();
  EXPECT_EQ(std::vector<uint32_t>({kAgg, kDef}), FlagsFor(0));
  EXPECT_EQ(std::vector<uint32_t>({kDef}), FlagsFor(1));
  Tick();
  EXPECT_EQ(std::vector<uint32_t>({kAgg, kDef, kDef}), FlagsFor(0));
}

TEST_F(GinHostProxyBatchTest, DrainedSlotsAreReleasedToTheProducer) {
  Init(1, 32);
  EnqueuePuts(0, 4);

  Tick();
  ASSERT_EQ(4u, g_calls.size());
  for (uint32_t i = 0; i < 4; i++) {
    for (int k = 0; k < ncclGinProxyGfdQwords; k++) {
      EXPECT_EQ(0u, queues_[i].qword[k].raw) << "slot " << i << " qword " << k << " was not cleared";
    }
  }
  EXPECT_EQ(0u, cis_[0]);
  // Completions, and with them the consumed index, are polled on the next tick.
  Tick();
  EXPECT_EQ(4u, cis_[0]);
}

}  // namespace
