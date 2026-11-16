/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/profiler/profiler_v6.cc, the adapter that serves a v6 plugin through the
// v7 interface. The plugin is a fake exported from a FakeLib (see profiler-test.cc's ncclOsDlsym).

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "comm.h"
#include "nccl_profiler.h"

#include PROFILER_V6_CC_PATH

namespace {

using FakeLib = std::map<std::string, void*>;
void* H(uintptr_t v) { return reinterpret_cast<void*>(v); }

struct FakeV6 {
  int initCalls = 0;
  void** initCtx = nullptr;
  uint64_t initCommId = 0;
  std::string initName;
  int initNNodes = -1, initNRanks = -1, initRank = -1;
  ncclDebugLogger_t initLogfn = nullptr;
  int setMask = 0;
  ncclResult_t initResult = ncclSuccess;
  std::vector<ncclProfilerEventDescr_v6_t> starts;
  std::vector<std::pair<ncclProfilerEventState_v6_t, ncclProfilerEventStateArgs_v6_t>> records;
  ncclProfiler_v6_t vtable{"FakeV6", &Init, &Start, &Stop, &Record, &Finalize};

  static FakeV6*& cur() {
    static FakeV6* p = nullptr;
    return p;
  }
  static ncclResult_t Init(void** ctx, uint64_t commId, int* mask, const char* name, int nNodes, int nRanks, int rank,
                           ncclDebugLogger_t logfn) {
    FakeV6& f = *cur();
    f.initCalls++;
    f.initCtx = ctx;
    f.initCommId = commId;
    f.initName = name;
    f.initNNodes = nNodes;
    f.initNRanks = nRanks;
    f.initRank = rank;
    f.initLogfn = logfn;
    if (mask) *mask = f.setMask;
    return f.initResult;
  }
  static ncclResult_t Start(void*, void** eHandle, ncclProfilerEventDescr_v6_t* d) {
    cur()->starts.push_back(*d);
    *eHandle = H(0x1);
    return ncclSuccess;
  }
  static ncclResult_t Stop(void*) { return ncclSuccess; }
  static ncclResult_t Record(void*, ncclProfilerEventState_v6_t s, ncclProfilerEventStateArgs_v6_t* a) {
    cur()->records.push_back({s, *a});
    return ncclSuccess;
  }
  static ncclResult_t Finalize(void*) { return ncclSuccess; }
};

void FakeLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

class ProfilerV6Microtest : public ::testing::Test {
 protected:
  FakeV6 fake_;
  FakeLib lib_{{"ncclProfiler_v6", &fake_.vtable}};
  ncclProfiler_t* prof_ = nullptr;
  void* ctx_ = nullptr;
  int mask_ = 0;

  void SetUp() override {
    FakeV6::cur() = &fake_;
    prof_ = getNcclProfiler_v6(&lib_);
    ASSERT_NE(prof_, nullptr);
    ASSERT_EQ(ncclSuccess, prof_->init(&ctx_, 0xABCD, &mask_, "c0", 2, 8, 3, &FakeLog));
  }
  void TearDown() override { FakeV6::cur() = nullptr; }

  const ncclProfilerEventDescr_v6_t* Start(ncclProfilerEventDescr_t& d) {
    void* h = H(0xBAD);
    size_t before = fake_.starts.size();
    EXPECT_EQ(ncclSuccess, prof_->startEvent(H(0xC0), &h, &d));
    if (fake_.starts.size() == before) {
      EXPECT_EQ(h, nullptr);
      return nullptr;
    }
    EXPECT_EQ(h, H(0x1));
    return &fake_.starts.back();
  }
};

TEST_F(ProfilerV6Microtest, GetProfiler_FindsOnlyTheV6Symbol) {
  FakeLib other{{"ncclProfiler_v5", &fake_.vtable}};
  EXPECT_EQ(getNcclProfiler_v6(&other), nullptr);
  EXPECT_STREQ(prof_->name, "FakeV6");
}

TEST_F(ProfilerV6Microtest, Init_ForwardsTheCommIdentityAndWiresTheCallbacks) {
  EXPECT_EQ(fake_.initCalls, 1);
  EXPECT_EQ(fake_.initCtx, &ctx_);
  EXPECT_EQ(fake_.initCommId, 0xABCDu);
  EXPECT_EQ(fake_.initName, "c0");
  EXPECT_EQ(fake_.initNNodes, 2);
  EXPECT_EQ(fake_.initNRanks, 8);
  EXPECT_EQ(fake_.initRank, 3);
  EXPECT_EQ(fake_.initLogfn, &FakeLog);
  EXPECT_EQ(prof_->stopEvent, &FakeV6::Stop);
  EXPECT_EQ(prof_->finalize, &FakeV6::Finalize);
  EXPECT_EQ(prof_->startEvent, &ncclProfiler_startEvent);
  EXPECT_EQ(prof_->recordEventState, &ncclProfiler_recordEventState);
}

TEST_F(ProfilerV6Microtest, Init_ClearsOnlyKernelPhaseAndReturnsThePluginsResult) {
  fake_.setMask = ~0;
  fake_.initResult = ncclSystemError;
  EXPECT_EQ(prof_->init(&ctx_, 0, &mask_, "c0", 2, 8, 3, nullptr), ncclSystemError);
  EXPECT_EQ(mask_, ~ncclProfileKernelPhase);
  EXPECT_EQ(prof_->init(&ctx_, 0, nullptr, "c0", 2, 8, 3, nullptr), ncclSystemError);
}

TEST_F(ProfilerV6Microtest, StartEvent_ApiEvents) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileGroupApi;
  d.parentObj = H(0x42);
  d.rank = 3;
  d.groupApi.graphCaptured = true;
  d.groupApi.groupDepth = 2;
  const auto* v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_EQ(v6->type, ncclProfileGroupApi);
  EXPECT_EQ(v6->parentObj, H(0x42));
  EXPECT_EQ(v6->rank, 3);
  EXPECT_TRUE(v6->groupApi.graphCaptured);
  EXPECT_EQ(v6->groupApi.groupDepth, 2);

  d = {};
  d.type = ncclProfileCollApi;
  d.collApi = {"AllReduce", 42, "ncclFloat32", 6, H(0x55), true, 9};
  v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_STREQ(v6->collApi.func, "AllReduce");
  EXPECT_EQ(v6->collApi.count, 42u);
  EXPECT_STREQ(v6->collApi.datatype, "ncclFloat32");
  EXPECT_EQ(v6->collApi.root, 6);
  EXPECT_EQ(v6->collApi.stream, H(0x55));
  EXPECT_TRUE(v6->collApi.graphCaptured);

  d = {};
  d.type = ncclProfileP2pApi;
  d.p2pApi = {"Send", 43, "ncclInt8", H(0x56), true, 9};
  v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_STREQ(v6->p2pApi.func, "Send");
  EXPECT_EQ(v6->p2pApi.count, 43u);
  EXPECT_STREQ(v6->p2pApi.datatype, "ncclInt8");
  EXPECT_EQ(v6->p2pApi.stream, H(0x56));
  EXPECT_TRUE(v6->p2pApi.graphCaptured);

  d = {};
  d.type = ncclProfileKernelLaunch;
  d.kernelLaunch.stream = H(0x57);
  v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_EQ(v6->kernelLaunch.stream, H(0x57));
}

TEST_F(ProfilerV6Microtest, StartEvent_TaskEvents) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileColl;
  d.parentObj = H(0x31);
  d.coll.parentGroup = H(0x30);
  d.coll.seqNumber = 7;
  d.coll.func = "AllGather";
  d.coll.sendBuff = H(0x100);
  d.coll.recvBuff = H(0x200);
  d.coll.count = 1000;
  d.coll.root = 2;
  d.coll.datatype = "ncclInt32";
  d.coll.nChannels = 4;
  d.coll.nWarps = 8;
  d.coll.algo = "RING";
  d.coll.proto = "LL";
  d.coll.kernelVariant = "AGxLL_R";
  const auto* v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_EQ(v6->parentObj, H(0x31));
  EXPECT_EQ(v6->coll.parentGroup, H(0x30));
  EXPECT_EQ(v6->coll.seqNumber, 7u);
  EXPECT_STREQ(v6->coll.func, "AllGather");
  EXPECT_EQ(v6->coll.sendBuff, H(0x100));
  EXPECT_EQ(v6->coll.recvBuff, H(0x200));
  EXPECT_EQ(v6->coll.count, 1000u);
  EXPECT_EQ(v6->coll.root, 2);
  EXPECT_STREQ(v6->coll.datatype, "ncclInt32");
  EXPECT_EQ(v6->coll.nChannels, 4);
  EXPECT_EQ(v6->coll.nWarps, 8);
  EXPECT_STREQ(v6->coll.algo, "RING");
  EXPECT_STREQ(v6->coll.proto, "LL");

  d = {};
  d.type = ncclProfileP2p;
  d.p2p.parentGroup = H(0x30);
  d.p2p.func = "Recv";
  d.p2p.buff = H(0x300);
  d.p2p.count = 77;
  d.p2p.datatype = "ncclUint64";
  d.p2p.peer = 4;
  d.p2p.nChannels = 3;
  v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_EQ(v6->p2p.parentGroup, H(0x30));
  EXPECT_STREQ(v6->p2p.func, "Recv");
  EXPECT_EQ(v6->p2p.buff, H(0x300));
  EXPECT_EQ(v6->p2p.count, 77u);
  EXPECT_STREQ(v6->p2p.datatype, "ncclUint64");
  EXPECT_EQ(v6->p2p.peer, 4);
  EXPECT_EQ(v6->p2p.nChannels, 3);
}

TEST_F(ProfilerV6Microtest, StartEvent_ProxyOp_CarriesTheRcclDiagFields) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileProxyOp;
  d.proxyOp = {99, 7, 5, 3, 200, 1, 0xAB, 12, 1, 2, 3, 4, 4096, 6};
  const auto* v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  const auto& p = v6->proxyOp;
  EXPECT_EQ(p.pid, 99);
  EXPECT_EQ(p.channelId, 7);
  EXPECT_EQ(p.peer, 5);
  EXPECT_EQ(p.nSteps, 3);
  EXPECT_EQ(p.chunkSize, 200);
  EXPECT_EQ(p.isSend, 1);
  EXPECT_EQ(p.commHash, 0xABu);
  EXPECT_EQ(p.opCount, 12);
  EXPECT_EQ(p.traceFuncIdx, 1);
  EXPECT_EQ(p.traceProtocol, 2);
  EXPECT_EQ(p.tracePattern, 3);
  EXPECT_EQ(p.traceTotalBytes, 4u);
  EXPECT_EQ(p.proxyNbytes, 4096u);
  EXPECT_EQ(p.rawProxyNsteps, 6);
}

TEST_F(ProfilerV6Microtest, StartEvent_StepKernelNetAndControlEvents) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileProxyStep;
  d.proxyStep.step = 9;
  const auto* v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_EQ(v6->proxyStep.step, 9);

  d = {};
  d.type = ncclProfileKernelCh;
  d.kernelCh.channelId = 9;
  d.kernelCh.pTimer = 111;
  v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_EQ(v6->kernelCh.channelId, 9);
  EXPECT_EQ(v6->kernelCh.pTimer, 111u);

  d = {};
  d.type = ncclProfileNetPlugin;
  d.netPlugin.id = 17;
  d.netPlugin.data = H(0xE0);
  v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_EQ(v6->netPlugin.id, 17);
  EXPECT_EQ(v6->netPlugin.data, H(0xE0));

  for (uint64_t type : {ncclProfileGroup, ncclProfileProxyCtrl}) {
    d = {};
    d.type = type;
    v6 = Start(d);
    ASSERT_NE(v6, nullptr) << type;
    EXPECT_EQ(v6->type, type);
  }
}

TEST_F(ProfilerV6Microtest, StartEvent_CeEvents) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileCeColl;
  d.parentObj = H(0x31);
  d.ceColl = {21, "AllGather", H(0x100), H(0x200), 64, 1, "ncclFloat32", "MC", true, 5, 6, 21, H(0x57), 13};
  const auto* v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  const auto& c = v6->ceColl;
  EXPECT_EQ(v6->parentObj, H(0x31));
  EXPECT_EQ(c.seqNumber, 21u);
  EXPECT_STREQ(c.func, "AllGather");
  EXPECT_EQ(c.sendBuff, H(0x100));
  EXPECT_EQ(c.recvBuff, H(0x200));
  EXPECT_EQ(c.count, 64u);
  EXPECT_EQ(c.root, 1);
  EXPECT_STREQ(c.datatype, "ncclFloat32");
  EXPECT_STREQ(c.syncStrategy, "MC");
  EXPECT_TRUE(c.intraBatchSync);
  EXPECT_EQ(c.batchSize, 5u);
  EXPECT_EQ(c.numBatches, 6u);
  EXPECT_EQ(c.ceSeqNum, 21u);
  EXPECT_EQ(c.stream, H(0x57));

  d = {};
  d.type = ncclProfileCeSync;
  d.ceCollSync = {true, 8};
  v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_TRUE(v6->ceCollSync.isComplete);
  EXPECT_EQ(v6->ceCollSync.nRanks, 8);

  d = {};
  d.type = ncclProfileCeBatch;
  d.ceCollBatch = {3, 600, true};
  v6 = Start(d);
  ASSERT_NE(v6, nullptr);
  EXPECT_EQ(v6->ceCollBatch.numOps, 3);
  EXPECT_EQ(v6->ceCollBatch.totalBytes, 600u);
  EXPECT_TRUE(v6->ceCollBatch.useIntraSync);
}

TEST_F(ProfilerV6Microtest, StartEvent_KernelPhase_IsDroppedWithANullHandle) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileKernelPhase;
  EXPECT_EQ(Start(d), nullptr);
}

TEST_F(ProfilerV6Microtest, RecordEventState_ForwardsEachStateWithItsArgs) {
  ncclProfilerEventStateArgs_t a{};
  a.proxyStep.transSize = 4096;
  for (ncclProfilerEventState_t s :
       {ncclProfilerProxyStepSendGPUWait, ncclProfilerProxyStepSendPeerWait_v4, ncclProfilerProxyStepSendWait,
        ncclProfilerProxyStepRecvWait, ncclProfilerProxyStepRecvFlushWait, ncclProfilerProxyStepRecvGPUWait}) {
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
    EXPECT_EQ(fake_.records.back().first, s);
    EXPECT_EQ(fake_.records.back().second.proxyStep.transSize, 4096u) << s;
  }
  a = {};
  a.proxyCtrl.appendedProxyOps = 6;
  for (ncclProfilerEventState_t s :
       {ncclProfilerProxyCtrlIdle, ncclProfilerProxyCtrlActive, ncclProfilerProxyCtrlSleep, ncclProfilerProxyCtrlWakeup,
        ncclProfilerProxyCtrlAppend, ncclProfilerProxyCtrlAppendEnd}) {
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
    EXPECT_EQ(fake_.records.back().first, s);
    EXPECT_EQ(fake_.records.back().second.proxyCtrl.appendedProxyOps, 6) << s;
  }
  a = {};
  a.netPlugin.data = H(0xE1);
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerNetPluginUpdate, &a));
  EXPECT_EQ(fake_.records.back().second.netPlugin.data, H(0xE1));
  a = {};
  a.kernelCh.pTimer = 222;
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerKernelChStop, &a));
  EXPECT_EQ(fake_.records.back().second.kernelCh.pTimer, 222u);
  a = {};
  a.proxyDiag = {9, 0, 3, -5, 1LL << 40};
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerProxyDiagUpdate, &a));
  EXPECT_EQ(fake_.records.back().first, ncclProfilerProxyDiagUpdate);
  EXPECT_EQ(fake_.records.back().second.proxyDiag.counterKind, 9);
  EXPECT_EQ(fake_.records.back().second.proxyDiag.flags, 3);
  EXPECT_EQ(fake_.records.back().second.proxyDiag.value, -5);
  EXPECT_EQ(fake_.records.back().second.proxyDiag.value2, 1LL << 40);

  // States without a v6 payload are forwarded with zeroed args, null or not.
  a.proxyStep.transSize = 4096;
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerCeCollStart, &a));
  EXPECT_EQ(fake_.records.back().first, ncclProfilerCeCollStart);
  EXPECT_EQ(fake_.records.back().second.proxyStep.transSize, 0u);
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerGroupStartApiStop, nullptr));
  EXPECT_EQ(fake_.records.back().first, ncclProfilerGroupStartApiStop);
  EXPECT_EQ(fake_.records.back().second.proxyStep.transSize, 0u);
}

TEST_F(ProfilerV6Microtest, RecordEventState_KernelPhaseStop_IsDropped) {
  ncclProfilerEventStateArgs_t a{};
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerKernelPhaseStop, &a));
  EXPECT_TRUE(fake_.records.empty());
}

}  // namespace
