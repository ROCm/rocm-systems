/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/profiler/profiler_v7.cc. The internal interface is v8, so v7 is now served
// through a shim that drops the v8 timestamps, stop descriptor and init config. The plugin is a FakeLib entry (see
// profiler-test.cc's ncclOsDlsym).

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "comm.h"
#include "nccl_profiler.h"

#include PROFILER_V7_CC_PATH

namespace {

using FakeLib = std::map<std::string, void*>;
void* H(uintptr_t v) { return reinterpret_cast<void*>(v); }

struct FakeV7 {
  int initCalls = 0;
  void** initCtx = nullptr;
  uint64_t initCommId = 0;
  int* initMask = nullptr;
  std::string initName;
  int initNNodes = -1, initNRanks = -1, initRank = -1;
  ncclDebugLogger_t initLogfn = nullptr;
  ncclResult_t initResult = ncclSuccess;
  int stopCalls = 0;
  void* stopHandle = nullptr;
  int finalizeCalls = 0;
  void* finalizeCtx = nullptr;
  void* startCtx = nullptr;
  std::vector<ncclProfilerEventDescr_v7_t> starts;
  std::vector<std::pair<ncclProfilerEventState_v7_t, ncclProfilerEventStateArgs_v7_t>> records;
  ncclProfiler_v7_t vtable{"FakeV7", &Init, &Start, &Stop, &Record, &Finalize};

  static FakeV7*& cur() {
    static FakeV7* p = nullptr;
    return p;
  }
  static ncclResult_t Init(void** ctx, uint64_t commId, int* mask, const char* name, int nNodes, int nRanks, int rank,
                           ncclDebugLogger_t logfn) {
    FakeV7& f = *cur();
    f.initCalls++;
    f.initCtx = ctx;
    f.initCommId = commId;
    f.initMask = mask;
    f.initName = name;
    f.initNNodes = nNodes;
    f.initNRanks = nRanks;
    f.initRank = rank;
    f.initLogfn = logfn;
    return f.initResult;
  }
  static ncclResult_t Start(void* ctx, void** eHandle, ncclProfilerEventDescr_v7_t* d) {
    cur()->startCtx = ctx;
    cur()->starts.push_back(*d);
    *eHandle = H(0x1);
    return ncclSuccess;
  }
  static ncclResult_t Stop(void* h) {
    cur()->stopCalls++;
    cur()->stopHandle = h;
    return ncclSuccess;
  }
  static ncclResult_t Record(void*, ncclProfilerEventState_v7_t s, ncclProfilerEventStateArgs_v7_t* a) {
    cur()->records.push_back({s, *a});
    return ncclSuccess;
  }
  static ncclResult_t Finalize(void* ctx) {
    cur()->finalizeCalls++;
    cur()->finalizeCtx = ctx;
    return ncclSuccess;
  }
};

void FakeLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

class ProfilerV7Microtest : public ::testing::Test {
 protected:
  FakeV7 fake_;
  FakeLib lib_{{"ncclProfiler_v7", &fake_.vtable}};
  ncclProfiler_t* prof_ = nullptr;
  void* target_ = nullptr;
  void* ctx_ = nullptr;
  int mask_ = 0;

  void SetUp() override {
    FakeV7::cur() = &fake_;
    prof_ = getNcclProfiler_v7(&lib_, &target_);
    ASSERT_NE(prof_, nullptr);
    ASSERT_EQ(target_, &fake_.vtable);
    setNcclProfiler_v7(target_);  // what profilerActivate() does before each call
  }
  void TearDown() override {
    setNcclProfiler_v7(nullptr);
    FakeV7::cur() = nullptr;
  }

  const ncclProfilerEventDescr_v7_t* Start(ncclProfilerEventDescr_t& d) {
    void* h = H(0xBAD);
    size_t before = fake_.starts.size();
    EXPECT_EQ(ncclSuccess, prof_->startEvent(H(0xC0), &h, &d));
    if (fake_.starts.size() == before) return nullptr;
    EXPECT_EQ(h, H(0x1));
    EXPECT_EQ(fake_.startCtx, H(0xC0));
    return &fake_.starts.back();
  }
};

TEST_F(ProfilerV7Microtest, GetProfiler_ReturnsTheShimAndHandsBackThePluginAsTarget) {
  EXPECT_NE(prof_, reinterpret_cast<ncclProfiler_t*>(&fake_.vtable));
  EXPECT_STREQ(prof_->name, "FakeV7");
  EXPECT_EQ(prof_->init, &ncclProfiler_init);
  EXPECT_EQ(prof_->startEvent, &ncclProfiler_startEvent);
  EXPECT_EQ(prof_->stopEvent, &ncclProfiler_stopEvent);
  EXPECT_EQ(prof_->recordEventState, &ncclProfiler_recordEventState);
  EXPECT_EQ(prof_->finalize, &ncclProfiler_finalize);

  FakeLib older{{"ncclProfiler_v6", &fake_.vtable}};
  void* target = H(0xBAD);
  EXPECT_EQ(getNcclProfiler_v7(&older, &target), nullptr);
  EXPECT_EQ(target, nullptr);
}

TEST_F(ProfilerV7Microtest, Init_ForwardsTheCommIdentityAndLeavesTheConfigAtItsDefaults) {
  ncclProfilerConfig_t config{};
  ASSERT_EQ(ncclSuccess, prof_->init(&ctx_, 0xABCD, &mask_, "c0", 2, 8, 3, &FakeLog, &config));
  EXPECT_EQ(fake_.initCalls, 1);
  EXPECT_EQ(fake_.initCtx, &ctx_);
  EXPECT_EQ(fake_.initCommId, 0xABCDu);
  EXPECT_EQ(fake_.initMask, &mask_);
  EXPECT_EQ(fake_.initName, "c0");
  EXPECT_EQ(fake_.initNNodes, 2);
  EXPECT_EQ(fake_.initNRanks, 8);
  EXPECT_EQ(fake_.initRank, 3);
  EXPECT_EQ(fake_.initLogfn, &FakeLog);
  EXPECT_EQ(config.cpuClock, (uint32_t)ncclProfilerCpuClockNone);
  EXPECT_EQ(config.gpuClockSync, 0u);

  fake_.initResult = ncclSystemError;
  EXPECT_EQ(prof_->init(&ctx_, 0, &mask_, "c0", 2, 8, 3, nullptr, nullptr), ncclSystemError);
}

TEST_F(ProfilerV7Microtest, StopAndFinalize_ForwardToThePluginAndDropTheStopDescriptor) {
  ncclProfilerEventDescr_t stopDescr{};
  stopDescr.tsCpu = 123;
  ASSERT_EQ(ncclSuccess, prof_->stopEvent(H(0x5), &stopDescr));
  ASSERT_EQ(ncclSuccess, prof_->stopEvent(H(0x6), nullptr));
  EXPECT_EQ(fake_.stopCalls, 2);
  EXPECT_EQ(fake_.stopHandle, H(0x6));
  ASSERT_EQ(ncclSuccess, prof_->finalize(H(0xC1)));
  EXPECT_EQ(fake_.finalizeCalls, 1);
  EXPECT_EQ(fake_.finalizeCtx, H(0xC1));
}

// Unlike the v1-v6 shims, the v7 startEvent reports a missing active plugin as an internal error.
TEST_F(ProfilerV7Microtest, NoActivePlugin_InitAndStartFailAndOtherCallbacksDoNotReachThePlugin) {
  setNcclProfiler_v7(nullptr);
  EXPECT_EQ(prof_->init(&ctx_, 0, &mask_, "c0", 2, 8, 3, nullptr, nullptr), ncclInternalError);
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileGroup;
  void* h = nullptr;
  EXPECT_EQ(prof_->startEvent(H(0xC0), &h, &d), ncclInternalError);
  ncclProfilerEventStateArgs_t a{};
  EXPECT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerProxyCtrlIdle, &a));
  EXPECT_EQ(ncclSuccess, prof_->stopEvent(H(0x5), nullptr));
  EXPECT_EQ(ncclSuccess, prof_->finalize(H(0xC1)));
  EXPECT_EQ(fake_.initCalls, 0);
  EXPECT_TRUE(fake_.starts.empty());
  EXPECT_TRUE(fake_.records.empty());
  EXPECT_EQ(fake_.stopCalls, 0);
  EXPECT_EQ(fake_.finalizeCalls, 0);
}

TEST_F(ProfilerV7Microtest, StartEvent_ApiEventsCarryTheUserTag) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileGroupApi;
  d.parentObj = H(0x42);
  d.rank = 3;
  d.tsCpu = 1000;
  d.groupApi.graphCaptured = true;
  d.groupApi.groupDepth = 2;
  const auto* v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->type, ncclProfileGroupApi);
  EXPECT_EQ(v7->parentObj, H(0x42));
  EXPECT_EQ(v7->rank, 3);
  EXPECT_TRUE(v7->groupApi.graphCaptured);
  EXPECT_EQ(v7->groupApi.groupDepth, 2);

  d = {};
  d.type = ncclProfileCollApi;
  d.collApi = {"AllReduce", 42, "ncclFloat32", 6, H(0x55), true, 9};
  v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_STREQ(v7->collApi.func, "AllReduce");
  EXPECT_EQ(v7->collApi.count, 42u);
  EXPECT_STREQ(v7->collApi.datatype, "ncclFloat32");
  EXPECT_EQ(v7->collApi.root, 6);
  EXPECT_EQ(v7->collApi.stream, H(0x55));
  EXPECT_TRUE(v7->collApi.graphCaptured);
  EXPECT_EQ(v7->collApi.userTag, 9u);

  d = {};
  d.type = ncclProfileP2pApi;
  d.p2pApi = {"Send", 43, "ncclInt8", H(0x56), false, 10};
  v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_STREQ(v7->p2pApi.func, "Send");
  EXPECT_EQ(v7->p2pApi.count, 43u);
  EXPECT_STREQ(v7->p2pApi.datatype, "ncclInt8");
  EXPECT_EQ(v7->p2pApi.stream, H(0x56));
  EXPECT_FALSE(v7->p2pApi.graphCaptured);
  EXPECT_EQ(v7->p2pApi.userTag, 10u);

  d = {};
  d.type = ncclProfileKernelLaunch;
  d.kernelLaunch.stream = H(0x57);
  v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->kernelLaunch.stream, H(0x57));
}

TEST_F(ProfilerV7Microtest, StartEvent_TaskEventsCarryTheV7Fields) {
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
  d.coll.algo = "LL_R";
  d.coll.proto = "";
  d.coll.kernelVariant = "AGxLL_R";
  d.coll.isSymColl = true;
  d.coll.userTag = 11;
  const auto* v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->parentObj, H(0x31));
  EXPECT_EQ(v7->coll.parentGroup, H(0x30));
  EXPECT_EQ(v7->coll.seqNumber, 7u);
  EXPECT_STREQ(v7->coll.func, "AllGather");
  EXPECT_EQ(v7->coll.sendBuff, H(0x100));
  EXPECT_EQ(v7->coll.recvBuff, H(0x200));
  EXPECT_EQ(v7->coll.count, 1000u);
  EXPECT_EQ(v7->coll.root, 2);
  EXPECT_STREQ(v7->coll.datatype, "ncclInt32");
  EXPECT_EQ(v7->coll.nChannels, 4);
  EXPECT_EQ(v7->coll.nWarps, 8);
  EXPECT_STREQ(v7->coll.algo, "LL_R");
  EXPECT_STREQ(v7->coll.proto, "");
  EXPECT_STREQ(v7->coll.kernelVariant, "AGxLL_R");
  EXPECT_TRUE(v7->coll.isSymColl);
  EXPECT_EQ(v7->coll.userTag, 11u);

  d = {};
  d.type = ncclProfileP2p;
  d.p2p.parentGroup = H(0x30);
  d.p2p.func = "Recv";
  d.p2p.buff = H(0x300);
  d.p2p.count = 77;
  d.p2p.datatype = "ncclUint64";
  d.p2p.peer = 4;
  d.p2p.nChannels = 3;
  d.p2p.userTag = 12;
  v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->p2p.parentGroup, H(0x30));
  EXPECT_STREQ(v7->p2p.func, "Recv");
  EXPECT_EQ(v7->p2p.buff, H(0x300));
  EXPECT_EQ(v7->p2p.count, 77u);
  EXPECT_STREQ(v7->p2p.datatype, "ncclUint64");
  EXPECT_EQ(v7->p2p.peer, 4);
  EXPECT_EQ(v7->p2p.nChannels, 3);
  EXPECT_EQ(v7->p2p.userTag, 12u);
}

TEST_F(ProfilerV7Microtest, StartEvent_ProxyKernelNetAndControlEvents) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileProxyOp;
  d.parentObj = H(0x41);
  d.proxyOp.pid = 99;
  d.proxyOp.channelId = 7;
  d.proxyOp.peer = 5;
  d.proxyOp.nSteps = 3;
  d.proxyOp.chunkSize = 200;
  d.proxyOp.isSend = 1;
  const auto* v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->parentObj, H(0x41));
  EXPECT_EQ(v7->proxyOp.pid, 99);
  EXPECT_EQ(v7->proxyOp.channelId, 7);
  EXPECT_EQ(v7->proxyOp.peer, 5);
  EXPECT_EQ(v7->proxyOp.nSteps, 3);
  EXPECT_EQ(v7->proxyOp.chunkSize, 200);
  EXPECT_EQ(v7->proxyOp.isSend, 1);

  d = {};
  d.type = ncclProfileProxyStep;
  d.proxyStep.step = 9;
  v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->proxyStep.step, 9);

  d = {};
  d.type = ncclProfileKernelCh;
  d.kernelCh.channelId = 9;
  d.kernelCh.pTimer = 111;
  v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->kernelCh.channelId, 9);
  EXPECT_EQ(v7->kernelCh.pTimer, 111u);

  d = {};
  d.type = ncclProfileKernelPhase;
  d.kernelPhase.channelId = 2;
  d.kernelPhase.phaseId = 1;
  d.kernelPhase.phaseName = "compute";
  d.kernelPhase.pTimer = 333;
  v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->kernelPhase.channelId, 2);
  EXPECT_EQ(v7->kernelPhase.phaseId, 1);
  EXPECT_STREQ(v7->kernelPhase.phaseName, "compute");
  EXPECT_EQ(v7->kernelPhase.pTimer, 333u);

  d = {};
  d.type = ncclProfileNetPlugin;
  d.netPlugin.id = 17;
  d.netPlugin.data = H(0xE0);
  v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->netPlugin.id, 17);
  EXPECT_EQ(v7->netPlugin.data, H(0xE0));

  for (uint64_t type : {ncclProfileGroup, ncclProfileProxyCtrl}) {
    d = {};
    d.type = type;
    d.parentObj = H(0x43);
    v7 = Start(d);
    ASSERT_NE(v7, nullptr) << type;
    EXPECT_EQ(v7->type, type);
    EXPECT_EQ(v7->parentObj, H(0x43));
  }

  // An event type the adapter does not know is dropped without reaching the plugin.
  d = {};
  d.type = 1ull << 40;
  EXPECT_EQ(Start(d), nullptr);
}

TEST_F(ProfilerV7Microtest, StartEvent_CeEvents) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileCeColl;
  d.parentObj = H(0x31);
  d.ceColl = {21, "AllGather", H(0x100), H(0x200), 64, 1, "ncclFloat32", "MC", true, 5, 6, 21, H(0x57), 13};
  const auto* v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  const auto& c = v7->ceColl;
  EXPECT_EQ(v7->parentObj, H(0x31));
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
  EXPECT_EQ(c.userTag, 13u);

  d = {};
  d.type = ncclProfileCeSync;
  d.ceCollSync = {true, 8};
  v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_TRUE(v7->ceCollSync.isComplete);
  EXPECT_EQ(v7->ceCollSync.nRanks, 8);

  d = {};
  d.type = ncclProfileCeBatch;
  d.ceCollBatch = {3, 600, true};
  v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->ceCollBatch.numOps, 3);
  EXPECT_EQ(v7->ceCollBatch.totalBytes, 600u);
  EXPECT_TRUE(v7->ceCollBatch.useIntraSync);
}

TEST_F(ProfilerV7Microtest, RecordEventState_FieldCopiesThePayloadAndDropsTheTimestamp) {
  ncclProfilerEventStateArgs_t a{};
  a.tsCpu = 1000;
  a.tsCpuUncertainty = 5;
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
  for (ncclProfilerEventState_t s : {ncclProfilerKernelChStop, ncclProfilerKernelPhaseStop}) {
    a = {};
    a.kernelCh.pTimer = 222;
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
    EXPECT_EQ(fake_.records.back().first, s);
    EXPECT_EQ(fake_.records.back().second.kernelCh.pTimer, 222u) << s;
  }

  // States without a v7 payload, and a null args pointer, are forwarded with zeroed args.
  a = {};
  a.proxyStep.transSize = 4096;
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerCeCollStart, &a));
  EXPECT_EQ(fake_.records.back().first, ncclProfilerCeCollStart);
  EXPECT_EQ(fake_.records.back().second.proxyStep.transSize, 0u);
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerGroupStartApiStop, nullptr));
  EXPECT_EQ(fake_.records.back().first, ncclProfilerGroupStartApiStop);
  EXPECT_EQ(fake_.records.back().second.proxyStep.transSize, 0u);
}

// RCCL's proxy-diag extensions in the v7 header (the proxyOp trace fields and the ProxyDiagUpdate payload) must
// reach a v7 plugin through the v8 shim, as they do through the v6 adapter.
TEST_F(ProfilerV7Microtest, RcclProxyDiagFieldsAreCarriedThroughTheShim) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileProxyOp;
  d.proxyOp = {99, 7, 5, 3, 200, 1, 0xAB, 12, 1, 2, 3, 4, 4096, 6};
  const auto* v7 = Start(d);
  ASSERT_NE(v7, nullptr);
  EXPECT_EQ(v7->proxyOp.pid, 99);
  EXPECT_EQ(v7->proxyOp.commHash, 0xABu);
  EXPECT_EQ(v7->proxyOp.opCount, 12);
  EXPECT_EQ(v7->proxyOp.traceFuncIdx, 1);
  EXPECT_EQ(v7->proxyOp.traceProtocol, 2);
  EXPECT_EQ(v7->proxyOp.tracePattern, 3);
  EXPECT_EQ(v7->proxyOp.traceTotalBytes, 4u);
  EXPECT_EQ(v7->proxyOp.proxyNbytes, 4096u);
  EXPECT_EQ(v7->proxyOp.rawProxyNsteps, 6);

  ncclProfilerEventStateArgs_t a{};
  a.proxyDiag = {9, 0, 3, -5, 1LL << 40};
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerProxyDiagUpdate, &a));
  EXPECT_EQ(fake_.records.back().first, ncclProfilerProxyDiagUpdate);
  EXPECT_EQ(fake_.records.back().second.proxyDiag.counterKind, 9);
  EXPECT_EQ(fake_.records.back().second.proxyDiag.flags, 3);
  EXPECT_EQ(fake_.records.back().second.proxyDiag.value, -5);
  EXPECT_EQ(fake_.records.back().second.proxyDiag.value2, 1LL << 40);
}

}  // namespace
