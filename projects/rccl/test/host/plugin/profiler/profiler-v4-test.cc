/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/profiler/profiler_v4.cc, the adapter that serves a v4 plugin through the
// v7 interface. The plugin is a fake exported from a FakeLib (see profiler-test.cc's ncclOsDlsym).

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "comm.h"
#include "nccl_profiler.h"

#include PROFILER_V4_CC_PATH

namespace {

using FakeLib = std::map<std::string, void*>;
void* H(uintptr_t v) { return reinterpret_cast<void*>(v); }

struct FakeV4 {
  int initCalls = 0;
  void** initCtx = nullptr;
  int* initMask = nullptr;
  std::string initName;
  uint64_t initHash = 0;
  int initNNodes = -1, initNRanks = -1, initRank = -1;
  ncclDebugLogger_t initLogfn = nullptr;
  ncclResult_t initResult = ncclSuccess;
  std::vector<ncclProfilerEventDescr_v4_t> starts;
  std::vector<std::pair<ncclProfilerEventState_v4_t, ncclProfilerEventStateArgs_v4_t>> records;
  ncclProfiler_v4_t vtable{"FakeV4", &Init, &Start, &Stop, &Record, &Finalize};

  static FakeV4*& cur() {
    static FakeV4* p = nullptr;
    return p;
  }
  static ncclResult_t Init(void** ctx, int* mask, const char* name, uint64_t hash, int nNodes, int nRanks, int rank,
                           ncclDebugLogger_t logfn) {
    FakeV4& f = *cur();
    f.initCalls++;
    f.initCtx = ctx;
    f.initMask = mask;
    f.initName = name;
    f.initHash = hash;
    f.initNNodes = nNodes;
    f.initNRanks = nRanks;
    f.initRank = rank;
    f.initLogfn = logfn;
    return f.initResult;
  }
  static ncclResult_t Start(void*, void** eHandle, ncclProfilerEventDescr_v4_t* d) {
    cur()->starts.push_back(*d);
    *eHandle = H(0x1);
    return ncclSuccess;
  }
  static ncclResult_t Stop(void*) { return ncclSuccess; }
  static ncclResult_t Record(void*, ncclProfilerEventState_v4_t s, ncclProfilerEventStateArgs_v4_t* a) {
    cur()->records.push_back({s, *a});
    return ncclSuccess;
  }
  static ncclResult_t Finalize(void*) { return ncclSuccess; }
};

void FakeLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

class ProfilerV4Microtest : public ::testing::Test {
 protected:
  FakeV4 fake_;
  FakeLib lib_{{"ncclProfiler_v4", &fake_.vtable}};
  ncclProfiler_t* prof_ = nullptr;
  void* ctx_ = nullptr;
  int mask_ = 0;

  void SetUp() override {
    FakeV4::cur() = &fake_;
    prof_ = getNcclProfiler_v4(&lib_);
    ASSERT_NE(prof_, nullptr);
    ASSERT_EQ(ncclSuccess, prof_->init(&ctx_, 0xABCD, &mask_, "c0", 2, 8, 3, &FakeLog));
  }
  void TearDown() override { FakeV4::cur() = nullptr; }

  const ncclProfilerEventDescr_v4_t* Start(ncclProfilerEventDescr_t& d) {
    void* h = nullptr;
    size_t before = fake_.starts.size();
    EXPECT_EQ(ncclSuccess, prof_->startEvent(H(0xC0), &h, &d));
    return fake_.starts.size() == before ? nullptr : &fake_.starts.back();
  }
};

TEST_F(ProfilerV4Microtest, GetProfiler_FindsOnlyTheV4Symbol) {
  FakeLib other{{"ncclProfiler_v3", &fake_.vtable}};
  EXPECT_EQ(getNcclProfiler_v4(&other), nullptr);
  EXPECT_STREQ(prof_->name, "FakeV4");
}

TEST_F(ProfilerV4Microtest, Init_ForwardsTheCommIdentityInV4ArgumentOrder) {
  EXPECT_EQ(fake_.initCalls, 1);
  EXPECT_EQ(fake_.initCtx, &ctx_);
  EXPECT_EQ(fake_.initMask, &mask_);
  EXPECT_EQ(fake_.initName, "c0");
  EXPECT_EQ(fake_.initHash, 0xABCDu);
  EXPECT_EQ(fake_.initNNodes, 2);
  EXPECT_EQ(fake_.initNRanks, 8);
  EXPECT_EQ(fake_.initRank, 3);
  EXPECT_EQ(fake_.initLogfn, &FakeLog);
  EXPECT_EQ(prof_->stopEvent, &FakeV4::Stop);
  EXPECT_EQ(prof_->finalize, &FakeV4::Finalize);
  EXPECT_EQ(prof_->startEvent, &ncclProfiler_startEvent);
  EXPECT_EQ(prof_->recordEventState, &ncclProfiler_recordEventState);

  fake_.initResult = ncclSystemError;
  EXPECT_EQ(prof_->init(&ctx_, 0, &mask_, "c0", 2, 8, 3, nullptr), ncclSystemError);
}

TEST_F(ProfilerV4Microtest, StartEvent_Coll_ParentsToTheGroup) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileColl;
  d.parentObj = H(0x31);
  d.rank = 3;
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
  const auto* v4 = Start(d);
  ASSERT_NE(v4, nullptr);
  EXPECT_EQ(v4->type, ncclProfileColl);
  EXPECT_EQ(v4->parentObj, H(0x30));
  EXPECT_EQ(v4->rank, 3);
  EXPECT_EQ(v4->coll.seqNumber, 7u);
  EXPECT_STREQ(v4->coll.func, "AllGather");
  EXPECT_EQ(v4->coll.sendBuff, H(0x100));
  EXPECT_EQ(v4->coll.recvBuff, H(0x200));
  EXPECT_EQ(v4->coll.count, 1000u);
  EXPECT_EQ(v4->coll.root, 2);
  EXPECT_STREQ(v4->coll.datatype, "ncclInt32");
  EXPECT_EQ(v4->coll.nChannels, 4);
  EXPECT_EQ(v4->coll.nWarps, 8);
  EXPECT_STREQ(v4->coll.algo, "RING");
  EXPECT_STREQ(v4->coll.proto, "LL");
}

TEST_F(ProfilerV4Microtest, StartEvent_P2p_ParentsToTheGroup) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileP2p;
  d.parentObj = H(0x32);
  d.p2p.parentGroup = H(0x30);
  d.p2p.func = "Recv";
  d.p2p.buff = H(0x300);
  d.p2p.count = 77;
  d.p2p.datatype = "ncclUint64";
  d.p2p.peer = 4;
  const auto* v4 = Start(d);
  ASSERT_NE(v4, nullptr);
  EXPECT_EQ(v4->parentObj, H(0x30));
  EXPECT_STREQ(v4->p2p.func, "Recv");
  EXPECT_EQ(v4->p2p.buff, H(0x300));
  EXPECT_EQ(v4->p2p.count, 77u);
  EXPECT_STREQ(v4->p2p.datatype, "ncclUint64");
  EXPECT_EQ(v4->p2p.peer, 4);
}

TEST_F(ProfilerV4Microtest, StartEvent_ProxyKernelAndNetEvents) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileProxyOp;
  d.parentObj = H(0x41);
  d.proxyOp = {99, 7, 5, 3, 200, 1};
  const auto* v4 = Start(d);
  ASSERT_NE(v4, nullptr);
  EXPECT_EQ(v4->parentObj, H(0x41));
  EXPECT_EQ(v4->proxyOp.pid, 99);
  EXPECT_EQ(v4->proxyOp.channelId, 7);
  EXPECT_EQ(v4->proxyOp.peer, 5);
  EXPECT_EQ(v4->proxyOp.nSteps, 3);
  EXPECT_EQ(v4->proxyOp.chunkSize, 200);
  EXPECT_EQ(v4->proxyOp.isSend, 1);

  d = {};
  d.type = ncclProfileProxyStep;
  d.proxyStep.step = 9;
  v4 = Start(d);
  ASSERT_NE(v4, nullptr);
  EXPECT_EQ(v4->proxyStep.step, 9);

  d = {};
  d.type = ncclProfileKernelCh;
  d.kernelCh.channelId = 9;
  d.kernelCh.pTimer = 111;
  v4 = Start(d);
  ASSERT_NE(v4, nullptr);
  EXPECT_EQ(v4->kernelCh.channelId, 9);
  EXPECT_EQ(v4->kernelCh.pTimer, 111u);

  d = {};
  d.type = ncclProfileNetPlugin;
  d.netPlugin.id = 17;
  d.netPlugin.data = H(0xE0);
  v4 = Start(d);
  ASSERT_NE(v4, nullptr);
  EXPECT_EQ(v4->netPlugin.id, 17);
  EXPECT_EQ(v4->netPlugin.data, H(0xE0));

  for (uint64_t type : {ncclProfileGroup, ncclProfileProxyCtrl}) {
    d = {};
    d.type = type;
    d.parentObj = H(0x42);
    d.rank = 3;
    v4 = Start(d);
    ASSERT_NE(v4, nullptr) << type;
    EXPECT_EQ(v4->type, type);
    EXPECT_EQ(v4->parentObj, H(0x42));
    EXPECT_EQ(v4->rank, 3);
  }
}

TEST_F(ProfilerV4Microtest, StartEvent_EventsNewerThanV4_AreDropped) {
  for (uint64_t type : {ncclProfileGroupApi, ncclProfileCollApi, ncclProfileKernelLaunch, ncclProfileCeColl}) {
    ncclProfilerEventDescr_t d{};
    d.type = type;
    EXPECT_EQ(Start(d), nullptr) << type;
  }
}

TEST_F(ProfilerV4Microtest, RecordEventState_ForwardsEachStateWithItsArgs) {
  ncclProfilerEventStateArgs_t a{};
  ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), ncclProfilerProxyOpInProgress_v4, &a));
  EXPECT_EQ(fake_.records.back().first, ncclProfilerProxyOpInProgress_v4);

  a.proxyStep.transSize = 4096;
  for (ncclProfilerEventState_t s :
       {ncclProfilerProxyStepSendGPUWait, ncclProfilerProxyStepSendPeerWait_v4, ncclProfilerProxyStepSendWait,
        ncclProfilerProxyStepRecvWait, ncclProfilerProxyStepRecvFlushWait, ncclProfilerProxyStepRecvGPUWait}) {
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
    EXPECT_EQ(fake_.records.back().first, s);
    EXPECT_EQ(fake_.records.back().second.proxyStep.transSize, 4096u) << s;
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
  a.proxyCtrl.appendedProxyOps = 6;
  for (ncclProfilerEventState_t s :
       {ncclProfilerProxyCtrlIdle, ncclProfilerProxyCtrlActive, ncclProfilerProxyCtrlSleep, ncclProfilerProxyCtrlWakeup,
        ncclProfilerProxyCtrlAppend, ncclProfilerProxyCtrlAppendEnd}) {
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
    EXPECT_EQ(fake_.records.back().first, s);
    EXPECT_EQ(fake_.records.back().second.proxyCtrl.appendedProxyOps, 6) << s;
  }

  size_t n = fake_.records.size();
  for (ncclProfilerEventState_t s : {ncclProfilerGroupStartApiStop, ncclProfilerGroupEndApiStart,
                                     ncclProfilerCeCollStart, ncclProfilerKernelPhaseStop, ncclProfilerProxyDiagUpdate})
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
  EXPECT_EQ(fake_.records.size(), n);
}

}  // namespace
