/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/profiler/profiler_v5.cc, the adapter that serves a v5 plugin through the
// v7 interface. The plugin is a fake exported from a FakeLib (see profiler-test.cc's ncclOsDlsym).

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "comm.h"
#include "nccl_profiler.h"

#include PROFILER_V5_CC_PATH

namespace {

using FakeLib = std::map<std::string, void*>;
void* H(uintptr_t v) { return reinterpret_cast<void*>(v); }

struct FakeV5 {
  int initCalls = 0;
  void** initCtx = nullptr;
  uint64_t initCommId = 0;
  std::string initName;
  int initNNodes = -1, initNRanks = -1, initRank = -1;
  ncclDebugLogger_t initLogfn = nullptr;
  int setMask = 0;
  ncclResult_t initResult = ncclSuccess;
  std::vector<ncclProfilerEventDescr_v5_t> starts;
  std::vector<std::pair<ncclProfilerEventState_v5_t, ncclProfilerEventStateArgs_v5_t>> records;
  ncclProfiler_v5_t vtable{"FakeV5", &Init, &Start, &Stop, &Record, &Finalize};

  static FakeV5*& cur() {
    static FakeV5* p = nullptr;
    return p;
  }
  static ncclResult_t Init(void** ctx, uint64_t commId, int* mask, const char* name, int nNodes, int nRanks, int rank,
                           ncclDebugLogger_t logfn) {
    FakeV5& f = *cur();
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
  static ncclResult_t Start(void*, void** eHandle, ncclProfilerEventDescr_v5_t* d) {
    cur()->starts.push_back(*d);
    *eHandle = H(0x1);
    return ncclSuccess;
  }
  static ncclResult_t Stop(void*) { return ncclSuccess; }
  static ncclResult_t Record(void*, ncclProfilerEventState_v5_t s, ncclProfilerEventStateArgs_v5_t* a) {
    cur()->records.push_back({s, *a});
    return ncclSuccess;
  }
  static ncclResult_t Finalize(void*) { return ncclSuccess; }
};

void FakeLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

class ProfilerV5Microtest : public ::testing::Test {
 protected:
  FakeV5 fake_;
  FakeLib lib_{{"ncclProfiler_v5", &fake_.vtable}};
  ncclProfiler_t* prof_ = nullptr;
  void* ctx_ = nullptr;
  int mask_ = 0;

  void SetUp() override {
    FakeV5::cur() = &fake_;
    prof_ = getNcclProfiler_v5(&lib_);
    ASSERT_NE(prof_, nullptr);
    ASSERT_EQ(ncclSuccess, prof_->init(&ctx_, 0xABCD, &mask_, "c0", 2, 8, 3, &FakeLog));
  }
  void TearDown() override { FakeV5::cur() = nullptr; }

  const ncclProfilerEventDescr_v5_t* Start(ncclProfilerEventDescr_t& d, void** h) {
    size_t before = fake_.starts.size();
    EXPECT_EQ(ncclSuccess, prof_->startEvent(H(0xC0), h, &d));
    return fake_.starts.size() == before ? nullptr : &fake_.starts.back();
  }
  const ncclProfilerEventDescr_v5_t* Start(ncclProfilerEventDescr_t& d) {
    void* h = nullptr;
    return Start(d, &h);
  }
};

TEST_F(ProfilerV5Microtest, GetProfiler_FindsOnlyTheV5Symbol) {
  FakeLib other{{"ncclProfiler_v4", &fake_.vtable}};
  EXPECT_EQ(getNcclProfiler_v5(&other), nullptr);
  EXPECT_STREQ(prof_->name, "FakeV5");
}

TEST_F(ProfilerV5Microtest, Init_ForwardsTheCommIdentityAndWiresTheCallbacks) {
  EXPECT_EQ(fake_.initCalls, 1);
  EXPECT_EQ(fake_.initCtx, &ctx_);
  EXPECT_EQ(fake_.initCommId, 0xABCDu);
  EXPECT_EQ(fake_.initName, "c0");
  EXPECT_EQ(fake_.initNNodes, 2);
  EXPECT_EQ(fake_.initNRanks, 8);
  EXPECT_EQ(fake_.initRank, 3);
  EXPECT_EQ(fake_.initLogfn, &FakeLog);
  EXPECT_EQ(prof_->stopEvent, &FakeV5::Stop);
  EXPECT_EQ(prof_->finalize, &FakeV5::Finalize);
  EXPECT_EQ(prof_->startEvent, &ncclProfiler_startEvent);
  EXPECT_EQ(prof_->recordEventState, &ncclProfiler_recordEventState);

  fake_.initResult = ncclSystemError;
  EXPECT_EQ(prof_->init(&ctx_, 0, &mask_, "c0", 2, 8, 3, nullptr), ncclSystemError);
}

TEST_F(ProfilerV5Microtest, Init_ClearsTheEventBitsV5CannotReceive) {
  fake_.setMask = ~0;
  ASSERT_EQ(ncclSuccess, prof_->init(&ctx_, 0, &mask_, "c0", 2, 8, 3, nullptr));
  EXPECT_EQ(mask_, ~(ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch | ncclProfileProxyDiag |
                     ncclProfileKernelPhase));
  EXPECT_EQ(ncclSuccess, prof_->init(&ctx_, 0, nullptr, "c0", 2, 8, 3, nullptr));
}

TEST_F(ProfilerV5Microtest, StartEvent_ApiEvents) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileGroupApi;
  d.parentObj = H(0x42);
  d.rank = 3;
  d.groupApi.graphCaptured = true;
  d.groupApi.groupDepth = 2;
  const auto* v5 = Start(d);
  ASSERT_NE(v5, nullptr);
  EXPECT_EQ(v5->type, ncclProfileGroupApi);
  EXPECT_EQ(v5->parentObj, H(0x42));
  EXPECT_EQ(v5->rank, 3);
  EXPECT_TRUE(v5->groupApi.graphCaptured);
  EXPECT_EQ(v5->groupApi.groupDepth, 2);

  d = {};
  d.type = ncclProfileCollApi;
  d.collApi = {"AllReduce", 42, "ncclFloat32", 6, H(0x55), true, 9};
  v5 = Start(d);
  ASSERT_NE(v5, nullptr);
  EXPECT_STREQ(v5->collApi.func, "AllReduce");
  EXPECT_EQ(v5->collApi.count, 42u);
  EXPECT_STREQ(v5->collApi.datatype, "ncclFloat32");
  EXPECT_EQ(v5->collApi.root, 6);
  EXPECT_EQ(v5->collApi.stream, H(0x55));
  EXPECT_TRUE(v5->collApi.graphCaptured);

  d = {};
  d.type = ncclProfileP2pApi;
  d.p2pApi = {"Send", 43, "ncclInt8", H(0x56), true, 9};
  v5 = Start(d);
  ASSERT_NE(v5, nullptr);
  EXPECT_STREQ(v5->p2pApi.func, "Send");
  EXPECT_EQ(v5->p2pApi.count, 43u);
  EXPECT_STREQ(v5->p2pApi.datatype, "ncclInt8");
  EXPECT_EQ(v5->p2pApi.stream, H(0x56));
  EXPECT_TRUE(v5->p2pApi.graphCaptured);

  d = {};
  d.type = ncclProfileKernelLaunch;
  d.kernelLaunch.stream = H(0x57);
  v5 = Start(d);
  ASSERT_NE(v5, nullptr);
  EXPECT_EQ(v5->kernelLaunch.stream, H(0x57));
}

TEST_F(ProfilerV5Microtest, StartEvent_TaskEvents_KeepTheParentGroupField) {
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
  const auto* v5 = Start(d);
  ASSERT_NE(v5, nullptr);
  EXPECT_EQ(v5->parentObj, H(0x31));
  EXPECT_EQ(v5->coll.parentGroup, H(0x30));
  EXPECT_EQ(v5->coll.seqNumber, 7u);
  EXPECT_STREQ(v5->coll.func, "AllGather");
  EXPECT_EQ(v5->coll.sendBuff, H(0x100));
  EXPECT_EQ(v5->coll.recvBuff, H(0x200));
  EXPECT_EQ(v5->coll.count, 1000u);
  EXPECT_EQ(v5->coll.root, 2);
  EXPECT_STREQ(v5->coll.datatype, "ncclInt32");
  EXPECT_EQ(v5->coll.nChannels, 4);
  EXPECT_EQ(v5->coll.nWarps, 8);
  EXPECT_STREQ(v5->coll.algo, "RING");
  EXPECT_STREQ(v5->coll.proto, "LL");

  d = {};
  d.type = ncclProfileP2p;
  d.parentObj = H(0x32);
  d.p2p.parentGroup = H(0x30);
  d.p2p.func = "Recv";
  d.p2p.buff = H(0x300);
  d.p2p.count = 77;
  d.p2p.datatype = "ncclUint64";
  d.p2p.peer = 4;
  d.p2p.nChannels = 3;
  v5 = Start(d);
  ASSERT_NE(v5, nullptr);
  EXPECT_EQ(v5->parentObj, H(0x32));
  EXPECT_EQ(v5->p2p.parentGroup, H(0x30));
  EXPECT_STREQ(v5->p2p.func, "Recv");
  EXPECT_EQ(v5->p2p.buff, H(0x300));
  EXPECT_EQ(v5->p2p.count, 77u);
  EXPECT_STREQ(v5->p2p.datatype, "ncclUint64");
  EXPECT_EQ(v5->p2p.peer, 4);
  EXPECT_EQ(v5->p2p.nChannels, 3);
}

TEST_F(ProfilerV5Microtest, StartEvent_ProxyKernelAndNetEvents) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileProxyOp;
  d.proxyOp = {99, 7, 5, 3, 200, 1};
  const auto* v5 = Start(d);
  ASSERT_NE(v5, nullptr);
  EXPECT_EQ(v5->proxyOp.pid, 99);
  EXPECT_EQ(v5->proxyOp.channelId, 7);
  EXPECT_EQ(v5->proxyOp.peer, 5);
  EXPECT_EQ(v5->proxyOp.nSteps, 3);
  EXPECT_EQ(v5->proxyOp.chunkSize, 200);
  EXPECT_EQ(v5->proxyOp.isSend, 1);

  d = {};
  d.type = ncclProfileProxyStep;
  d.proxyStep.step = 9;
  v5 = Start(d);
  ASSERT_NE(v5, nullptr);
  EXPECT_EQ(v5->proxyStep.step, 9);

  d = {};
  d.type = ncclProfileKernelCh;
  d.kernelCh.channelId = 9;
  d.kernelCh.pTimer = 111;
  v5 = Start(d);
  ASSERT_NE(v5, nullptr);
  EXPECT_EQ(v5->kernelCh.channelId, 9);
  EXPECT_EQ(v5->kernelCh.pTimer, 111u);

  d = {};
  d.type = ncclProfileNetPlugin;
  d.netPlugin.id = 17;
  d.netPlugin.data = H(0xE0);
  v5 = Start(d);
  ASSERT_NE(v5, nullptr);
  EXPECT_EQ(v5->netPlugin.id, 17);
  EXPECT_EQ(v5->netPlugin.data, H(0xE0));

  for (uint64_t type : {ncclProfileGroup, ncclProfileProxyCtrl}) {
    d = {};
    d.type = type;
    v5 = Start(d);
    ASSERT_NE(v5, nullptr) << type;
    EXPECT_EQ(v5->type, type);
  }
}

TEST_F(ProfilerV5Microtest, StartEvent_CeAndKernelPhaseEvents_AreDropped) {
  for (uint64_t type : {ncclProfileCeColl, ncclProfileCeSync, ncclProfileCeBatch}) {
    ncclProfilerEventDescr_t d{};
    d.type = type;
    void* h = H(0xBAD);
    EXPECT_EQ(Start(d, &h), nullptr) << type;
    EXPECT_EQ(h, nullptr);
  }
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileKernelPhase;
  EXPECT_EQ(Start(d), nullptr);
}

TEST_F(ProfilerV5Microtest, RecordEventState_ForwardsEachStateWithItsArgs) {
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
  for (ncclProfilerEventState_t s :
       {ncclProfilerProxyOpInProgress_v4, ncclProfilerGroupStartApiStop, ncclProfilerGroupEndApiStart}) {
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
    EXPECT_EQ(fake_.records.back().first, s);
  }

  size_t n = fake_.records.size();
  for (ncclProfilerEventState_t s :
       {ncclProfilerCeCollStart, ncclProfilerCeCollComplete, ncclProfilerCeSyncStart, ncclProfilerCeSyncComplete,
        ncclProfilerCeBatchStart, ncclProfilerCeBatchComplete, ncclProfilerProxyDiagUpdate,
        ncclProfilerKernelPhaseStop})
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
  EXPECT_EQ(fake_.records.size(), n);
}

// The core records group-API states with null args, which this adapter drops; a fix flips this pin.
TEST_F(ProfilerV5Microtest, RecordEventState_KnownBug_GroupApiStatesWithNullArgs_NeverReachThePlugin) {
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerGroupStartApiStop, nullptr));
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerGroupEndApiStart, nullptr));
  EXPECT_TRUE(fake_.records.empty());
}

}  // namespace
