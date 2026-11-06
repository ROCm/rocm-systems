/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/profiler/profiler_v3.cc, the adapter that serves a v3 plugin through the
// v7 interface. The plugin is a fake exported from a FakeLib (see profiler-test.cc's ncclOsDlsym).

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

#include "comm.h"
#include "nccl_profiler.h"

#include PROFILER_V3_CC_PATH

namespace {

using FakeLib = std::map<std::string, void*>;
void* H(uintptr_t v) { return reinterpret_cast<void*>(v); }

struct FakeV3 {
  int initCalls = 0;
  void** initCtx = nullptr;
  int* initMask = nullptr;
  ncclResult_t initResult = ncclSuccess;
  std::vector<ncclProfilerEventDescr_v3_t> starts;
  std::vector<std::pair<ncclProfilerEventState_v3_t, ncclProfilerEventStateArgs_v3_t>> records;
  ncclProfiler_v3_t vtable{"FakeV3", &Init, &Start, &Stop, &Record, &Finalize};

  static FakeV3*& cur() {
    static FakeV3* p = nullptr;
    return p;
  }
  static ncclResult_t Init(void** ctx, int* mask) {
    cur()->initCalls++;
    cur()->initCtx = ctx;
    cur()->initMask = mask;
    return cur()->initResult;
  }
  static ncclResult_t Start(void*, void** eHandle, ncclProfilerEventDescr_v3_t* d) {
    cur()->starts.push_back(*d);
    *eHandle = H(0x1);
    return ncclSuccess;
  }
  static ncclResult_t Stop(void*) { return ncclSuccess; }
  static ncclResult_t Record(void*, ncclProfilerEventState_v3_t s, ncclProfilerEventStateArgs_v3_t* a) {
    cur()->records.push_back({s, *a});
    return ncclSuccess;
  }
  static ncclResult_t Finalize(void*) { return ncclSuccess; }
};

class ProfilerV3Microtest : public ::testing::Test {
 protected:
  FakeV3 fake_;
  FakeLib lib_{{"ncclProfiler_v3", &fake_.vtable}};
  ncclProfiler_t* prof_ = nullptr;
  void* ctx_ = nullptr;
  int mask_ = 0;

  void SetUp() override {
    FakeV3::cur() = &fake_;
    prof_ = getNcclProfiler_v3(&lib_);
    ASSERT_NE(prof_, nullptr);
    ASSERT_EQ(ncclSuccess, prof_->init(&ctx_, 0xABCD, &mask_, "c0", 2, 8, 3, nullptr));
  }
  void TearDown() override { FakeV3::cur() = nullptr; }

  const ncclProfilerEventDescr_v3_t* Start(ncclProfilerEventDescr_t& d) {
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

TEST_F(ProfilerV3Microtest, GetProfiler_FindsOnlyTheV3Symbol) {
  FakeLib other{{"ncclProfiler_v2", &fake_.vtable}};
  EXPECT_EQ(getNcclProfiler_v3(&other), nullptr);
  EXPECT_STREQ(prof_->name, "FakeV3");
}

TEST_F(ProfilerV3Microtest, Init_ForwardsTheMaskAndWiresTheCallbacks) {
  EXPECT_EQ(fake_.initCalls, 1);
  EXPECT_EQ(fake_.initCtx, &ctx_);
  EXPECT_EQ(fake_.initMask, &mask_);
  EXPECT_EQ(prof_->stopEvent, &FakeV3::Stop);
  EXPECT_EQ(prof_->finalize, &FakeV3::Finalize);
  EXPECT_EQ(prof_->startEvent, &ncclProfiler_startEvent);
  EXPECT_EQ(prof_->recordEventState, &ncclProfiler_recordEventState);

  fake_.initResult = ncclSystemError;
  EXPECT_EQ(prof_->init(&ctx_, 0, &mask_, "c0", 2, 8, 3, nullptr), ncclSystemError);
}

TEST_F(ProfilerV3Microtest, StartEvent_Coll_KeepsTheStringsAndParentsToTheGroup) {
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
  const auto* v3 = Start(d);
  ASSERT_NE(v3, nullptr);
  EXPECT_EQ(v3->type, ncclProfileColl);
  EXPECT_EQ(v3->parentObj, H(0x30));
  EXPECT_EQ(v3->rank, 3);
  EXPECT_EQ(v3->coll.name, nullptr);
  EXPECT_EQ(v3->coll.commHash, 0u);
  EXPECT_EQ(v3->coll.seqNumber, 7u);
  EXPECT_STREQ(v3->coll.func, "AllGather");
  EXPECT_EQ(v3->coll.sendBuff, H(0x100));
  EXPECT_EQ(v3->coll.recvBuff, H(0x200));
  EXPECT_EQ(v3->coll.count, 1000u);
  EXPECT_EQ(v3->coll.root, 2);
  EXPECT_STREQ(v3->coll.datatype, "ncclInt32");
  EXPECT_EQ(v3->coll.nMaxChannels, 4);
  EXPECT_EQ(v3->coll.nWarps, 8);
  EXPECT_STREQ(v3->coll.algo, "RING");
  EXPECT_STREQ(v3->coll.proto, "LL");
}

TEST_F(ProfilerV3Microtest, StartEvent_P2pProxyAndGroupEvents) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileP2p;
  d.parentObj = H(0x32);
  d.p2p.parentGroup = H(0x30);
  d.p2p.func = "Recv";
  d.p2p.buff = H(0x300);
  d.p2p.count = 77;
  d.p2p.datatype = "ncclUint64";
  d.p2p.peer = 4;
  const auto* v3 = Start(d);
  ASSERT_NE(v3, nullptr);
  EXPECT_EQ(v3->parentObj, H(0x30));
  EXPECT_EQ(v3->p2p.name, nullptr);
  EXPECT_EQ(v3->p2p.commHash, 0u);
  EXPECT_STREQ(v3->p2p.func, "Recv");
  EXPECT_EQ(v3->p2p.buff, H(0x300));
  EXPECT_EQ(v3->p2p.count, 77u);
  EXPECT_STREQ(v3->p2p.datatype, "ncclUint64");
  EXPECT_EQ(v3->p2p.peer, 4);

  d = {};
  d.type = ncclProfileProxyOp;
  d.parentObj = H(0x41);
  d.proxyOp = {99, 7, 5, 3, 200, 1};
  v3 = Start(d);
  ASSERT_NE(v3, nullptr);
  EXPECT_EQ(v3->parentObj, H(0x41));
  EXPECT_EQ(v3->proxyOp.pid, 99);
  EXPECT_EQ(v3->proxyOp.channelId, 7);
  EXPECT_EQ(v3->proxyOp.peer, 5);
  EXPECT_EQ(v3->proxyOp.nSteps, 3);
  EXPECT_EQ(v3->proxyOp.chunkSize, 200);
  EXPECT_EQ(v3->proxyOp.isSend, 1);

  d = {};
  d.type = ncclProfileProxyStep;
  d.proxyStep.step = 9;
  v3 = Start(d);
  ASSERT_NE(v3, nullptr);
  EXPECT_EQ(v3->proxyStep.step, 9);

  for (uint64_t type : {ncclProfileGroup, ncclProfileProxyCtrl}) {
    d = {};
    d.type = type;
    d.parentObj = H(0x42);
    v3 = Start(d);
    ASSERT_NE(v3, nullptr) << type;
    EXPECT_EQ(v3->type, type);
    EXPECT_EQ(v3->parentObj, H(0x42));
  }
}

TEST_F(ProfilerV3Microtest, StartEvent_KernelChAndNetPlugin) {
  ncclProfilerEventDescr_t d{};
  d.type = ncclProfileKernelCh;
  d.parentObj = H(0x51);
  d.kernelCh.channelId = 9;
  d.kernelCh.pTimer = 111;  // v3 has no start timestamp
  const auto* v3 = Start(d);
  ASSERT_NE(v3, nullptr);
  EXPECT_EQ(v3->parentObj, H(0x51));
  EXPECT_EQ(v3->kernelCh.channelId, 9);

  d = {};
  d.type = ncclProfileNetPlugin;
  d.netPlugin.id = 17;
  d.netPlugin.data = H(0xE0);
  v3 = Start(d);
  ASSERT_NE(v3, nullptr);
  EXPECT_EQ(v3->netPlugin.id, 17);
  EXPECT_EQ(v3->netPlugin.data, H(0xE0));
}

TEST_F(ProfilerV3Microtest, StartEvent_EventsNewerThanV3_AreDroppedWithANullHandle) {
  for (uint64_t type : {ncclProfileGroupApi, ncclProfileKernelLaunch, ncclProfileCeColl, ncclProfileKernelPhase}) {
    ncclProfilerEventDescr_t d{};
    d.type = type;
    EXPECT_EQ(Start(d), nullptr) << type;
  }
}

TEST_F(ProfilerV3Microtest, RecordEventState_ForwardsProxyStatesOnly) {
  ncclProfilerEventStateArgs_t a{};
  a.proxyCtrl.appendedProxyOps = 6;
  for (ncclProfilerEventState_t s :
       {ncclProfilerProxyCtrlIdle, ncclProfilerProxyCtrlActive, ncclProfilerProxyCtrlSleep, ncclProfilerProxyCtrlWakeup,
        ncclProfilerProxyCtrlAppend, ncclProfilerProxyCtrlAppendEnd}) {
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
    ASSERT_FALSE(fake_.records.empty());
    EXPECT_EQ(fake_.records.back().first, s);
    EXPECT_EQ(fake_.records.back().second.proxyCtrl.appendedProxyOps, 6) << s;
  }
  a.proxyStep.transSize = 4096;
  for (ncclProfilerEventState_t s : {ncclProfilerProxyStepSendGPUWait, ncclProfilerProxyStepSendWait,
                                     ncclProfilerProxyStepRecvWait, ncclProfilerProxyStepRecvFlushWait,
                                     ncclProfilerProxyStepRecvGPUWait}) {
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
    EXPECT_EQ(fake_.records.back().first, s);
    EXPECT_EQ(fake_.records.back().second.proxyOp.transSize, 0u) << s;
  }
  size_t n = fake_.records.size();
  for (ncclProfilerEventState_t s : {ncclProfilerProxyOpInProgress_v4, ncclProfilerProxyStepSendPeerWait_v4,
                                     ncclProfilerNetPluginUpdate, ncclProfilerKernelChStop,
                                     ncclProfilerGroupStartApiStop, ncclProfilerProxyDiagUpdate})
    ASSERT_EQ(ncclSuccess, prof_->recordEventState(H(0x1), s, &a));
  EXPECT_EQ(fake_.records.size(), n);
}

}  // namespace
