/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests; the UUT is #include'd. Hipify prepends one line, so hipified N is source N-1.

#include <gtest/gtest.h>

#include <cstring>
#include <initializer_list>
#include <vector>

#include "ScopedHook.h"
#include "TaskPrepScene.h"

#include TASK_PRETUNING_CC_PATH

namespace {

constexpr int kPeer = 3;
constexpr size_t kFloatSize = sizeof(float);
constexpr size_t kDoubleSize = sizeof(double);
constexpr size_t kCollSendBytes = kCount * kFloatSize;
constexpr size_t kAllGatherRecvBytes = kRanks * kCount * kFloatSize;
constexpr size_t kAllGatherQueriedSendBytes = kCount;
constexpr size_t kAllGatherQueriedRecvBytes = kRanks * kCount;
constexpr size_t kReduceScatterSendBytes = kRanks * kCount * kFloatSize;
constexpr int kEnvCTAs = 7;
constexpr int kCallCTAs = 5;
constexpr int kCommCTAs = 3;
constexpr int kSmallerCTAs = 5;
constexpr int kLargerCTAs = 7;
constexpr int kBelowRangeCTAs = 0;
constexpr int kOutOfRangeCTAs = MAXCHANNELS + 1;
constexpr ncclSymRegType_t kProbedRegType = ncclSymSendRegRecvReg;

constexpr uintptr_t kSendAddr = 0x40024ULL;
constexpr uintptr_t kRecvAddr = 0x50028ULL;
constexpr uintptr_t kRecvWindowAlignedAddr = 0x5002CULL;
constexpr uintptr_t kSendWindowAddr = 0x40004ULL;
constexpr uintptr_t kRecvWindowAddr = 0x50008ULL;
constexpr size_t kAlignedGap = 16;
constexpr size_t kMisalignedGap = 8;

const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x5eedULL);

void* TaskPreTuning_Addr(uintptr_t address) {
  return reinterpret_cast<void*>(address);
}

// Records which registration each probe was handed, so passing one buffer's reg twice cannot pass.
class TaskPreTuning_LocallyValidRegistrations {
 public:
  TaskPreTuning_LocallyValidRegistrations()
    : hook_(g_regLocalIsValid, [this](struct ncclReg* reg, bool* out) {
        probed_.push_back(reg);
        *out = true;
        return ncclSuccess;
      }) {}

  const std::vector<struct ncclReg*>& probed() const { return probed_; }

 private:
  std::vector<struct ncclReg*> probed_;
  ScopedHook<ncclResult_t(struct ncclReg*, bool*)> hook_;
};

// Routes ncclDevrFindWindow to a distinct window per buffer so send and recv cannot be confused.
class TaskPreTuning_Windows {
 public:
  TaskPreTuning_Windows(const void* sendbuff, void* sendUserPtr, const void* recvbuff, void* recvUserPtr)
    : sendbuff_(sendbuff), recvbuff_(recvbuff),
      hook_(g_devrFindWindow, [this](struct ncclComm*, void const* ptr, struct ncclDevrWindow** out) {
        *out = ptr == sendbuff_ ? sendWindow() : (ptr == recvbuff_ ? recvWindow() : nullptr);
        return ncclSuccess;
      }) {
    sendWindow_.userPtr = sendUserPtr;
    recvWindow_.userPtr = recvUserPtr;
  }

  struct ncclDevrWindow* sendWindow() { return sendbuff_ != nullptr ? &sendWindow_ : nullptr; }
  struct ncclDevrWindow* recvWindow() { return recvbuff_ != nullptr ? &recvWindow_ : nullptr; }

 private:
  const void* sendbuff_;
  const void* recvbuff_;
  struct ncclDevrWindow sendWindow_{};
  struct ncclDevrWindow recvWindow_{};
  ScopedHook<ncclResult_t(struct ncclComm*, void const*, struct ncclDevrWindow**)> hook_;
};

std::vector<struct ncclRawTask*> TaskPreTuning_RawTasks(
  struct ncclIntruQueue<struct ncclRawTask, &ncclRawTask::next>* queue) {
  std::vector<struct ncclRawTask*> raws;
  for (struct ncclRawTask* raw = ncclIntruQueueHead(queue); raw != nullptr; raw = raw->next) {
    raws.push_back(raw);
  }
  return raws;
}

std::vector<struct ncclRawTask*> TaskPreTuning_TunedRaws(struct ncclTaskTuningInfoQueue* tiq) {
  std::vector<struct ncclRawTask*> raws;
  for (struct ncclTaskTuningInfo* task : QueueTasks(&tiq->queue)) {
    raws.push_back(task->raw);
  }
  return raws;
}

std::vector<struct ncclRawTask*> TaskPreTuning_List(std::initializer_list<struct ncclRawTask*> raws) {
  return std::vector<struct ncclRawTask*>(raws);
}

// TaskPrepScene starts non-capturing, so the fixture needs no SetUp of its own beyond the fakes reset.
class TaskPreTuningMicrotest : public TaskPrepFakesFixture {
 protected:
  TaskPrepScene scene_;
};

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_UnscaledColl_CopiesTheRawFieldsAndSizesNBytes) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllReduce);
  raw->coll.opHost = ncclProd;
  raw->coll.opDev.op = ncclDevProd;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(scene_.comm(), in.comm);
  EXPECT_EQ(NCCL_TUNING_MASK_SYM_KERNELS | NCCL_TUNING_MASK_GENERAL_KERNELS, in.tuningMask);
  EXPECT_EQ(ncclFuncAllReduce, in.func);
  EXPECT_EQ(ncclProd, in.redOp);
  EXPECT_EQ(ncclDevProd, in.devRedOp);
  EXPECT_EQ(ncclFloat32, in.datatype);
  EXPECT_EQ(kCount, in.count);
  EXPECT_EQ(kCount, in.countMax);
  EXPECT_EQ(1, in.nWorks);
  EXPECT_EQ(1, in.numPipeOps);
  EXPECT_EQ(kCollSendBytes, in.nBytes);
  // fillCollTuningInput is a partial fill: these four stay the caller's, and the scheduler resolves them.
  EXPECT_EQ(TaskPrep_Poisoned<int>(), in.collNetSupport);
  EXPECT_EQ(TaskPrep_Poisoned<int>(), in.captured);
  EXPECT_EQ(TaskPrep_Poisoned<int>(), in.inPlace);
  EXPECT_EQ(TaskPrep_Poisoned<int>(), in.CTAPolicy);
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_AllGather_RescalesTheCountToBytesAndSwitchesToInt8) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllGather);
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(ncclInt8, in.datatype);
  EXPECT_EQ(kCollSendBytes, in.count);
  EXPECT_EQ(kCollSendBytes, in.countMax);
  EXPECT_EQ(kAllGatherRecvBytes, in.nBytes);
  EXPECT_EQ(kCount, raw->coll.count) << "the raw task must not be rewritten";
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_Broadcast_RescalesTheCountToBytesAndSwitchesToInt8) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncBroadcast, ncclFloat64);
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(ncclInt8, in.datatype);
  EXPECT_EQ(kCount * kDoubleSize, in.count);
  EXPECT_EQ(kCount * kDoubleSize, in.countMax);
  EXPECT_EQ(kCount * kDoubleSize, in.nBytes);
}

// Built at two ranks rather than kRanks, so a literal 4 in place of comm->nRanks turns nBytes red.
TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_ReduceScatter_SizesNBytesAcrossEveryRanksSlice) {
  constexpr int kTwoRanks = 2;
  TaskPrepScene scene(kTwoRanks, 0);
  struct ncclRawTask* raw = scene.NewColl(ncclFuncReduceScatter);
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

  EXPECT_EQ(ncclFloat32, in.datatype);
  EXPECT_EQ(kCount, in.count);
  EXPECT_EQ(kTwoRanks * kCount * kFloatSize, in.nBytes);
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_FindWindowFails_PropagatesForEitherBuffer) {
  for (int failingCall : {1, 2}) {
    TaskPrepScene scene;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
    int seen = 0;
    ScopedHook find(g_devrFindWindow,
                    [&seen, failingCall](struct ncclComm*, void const*, struct ncclDevrWindow** out) {
                      if (++seen == failingCall) {
                        return ncclInternalError;
                      }
                      *out = nullptr;
                      return ncclSuccess;
                    });
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    EXPECT_EQ(ncclInternalError, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(failingCall, seen) << "must stop at the first failing lookup";
    EXPECT_EQ(TaskPrep_Poisoned<int>(), in.regBuff);
  }
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_NeitherBufferInAWindow_SkipsTheSymRegTypeQuery) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllReduce);
  ScopedHook regType(g_getSymRegType,
                     [](struct ncclDevrWindow*, struct ncclDevrWindow*, ncclSymRegType_t* out) {
                       *out = kProbedRegType;
                       return ncclSuccess;
                     });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();
  const ncclSymRegType_t untouched = in.winRegType;

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(0, regType.calls);
  EXPECT_EQ(untouched, in.winRegType) << "winRegType must keep the caller's value";
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_EitherBufferInAWindow_QueriesSymRegTypeWithBothWindows) {
  for (bool sendSide : {true, false}) {
    TaskPrepScene scene;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
    TaskPreTuning_Windows windows(sendSide ? raw->coll.sendbuff : nullptr, TaskPreTuning_Addr(kSendWindowAddr),
                                  sendSide ? nullptr : raw->coll.recvbuff,
                                  TaskPreTuning_Addr(kRecvWindowAddr));
    struct ncclDevrWindow* seenSend = nullptr;
    struct ncclDevrWindow* seenRecv = nullptr;
    ScopedHook regType(g_getSymRegType, [&](struct ncclDevrWindow* sendWin,
                                            struct ncclDevrWindow* recvWin, ncclSymRegType_t* out) {
      seenSend = sendWin;
      seenRecv = recvWin;
      *out = kProbedRegType;
      return ncclSuccess;
    });
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(1, regType.calls) << "sendSide = " << sendSide;
    EXPECT_EQ(windows.sendWindow(), seenSend);
    EXPECT_EQ(windows.recvWindow(), seenRecv);
    EXPECT_EQ(kProbedRegType, in.winRegType);
  }
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_SymRegTypeFails_Propagates) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllReduce);
  TaskPreTuning_Windows windows(raw->coll.sendbuff, TaskPreTuning_Addr(kSendWindowAddr),
                                raw->coll.recvbuff, TaskPreTuning_Addr(kRecvWindowAddr));
  ScopedHook regType(g_getSymRegType,
                     [](struct ncclDevrWindow*, struct ncclDevrWindow*, ncclSymRegType_t*) {
                       return ncclInvalidUsage;
                     });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  EXPECT_EQ(ncclInvalidUsage, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(TaskPrep_Poisoned<int>(), in.nvlsSupport) << "the nvls arm must not have run";
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_NvlsSupport_NeedsTheCommFlagAndAnEligibleOpOrAllGather) {
  struct Case {
    int commNvlsSupport;
    ncclFunc_t func;
    ncclDataType_t datatype;
    ncclDevRedOp_t devRedOp;
    int expected;
  };
  // The Broadcast row rescales in->datatype to ncclInt8, so reading that instead of raw->datatype fails it.
  const std::vector<Case> cases = {{0, ncclFuncAllGather, ncclInt8, ncclDevSum, 0},
                                   {1, ncclFuncAllReduce, ncclFloat32, ncclDevSum, 1},
                                   {1, ncclFuncAllReduce, ncclInt8, ncclDevSum, 0},
                                   {1, ncclFuncAllGather, ncclInt8, ncclDevSum, 1},
                                   {1, ncclFuncAllReduce, ncclFloat32, ncclDevProd, 0},
                                   {1, ncclFuncBroadcast, ncclFloat32, ncclDevSum, 1}};
  for (const Case& c : cases) {
    TaskPrepScene scene;
    scene.comm()->nvlsSupport = c.commNvlsSupport;
    struct ncclRawTask* raw = scene.NewColl(c.func, c.datatype);
    raw->coll.opDev.op = c.devRedOp;
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(c.expected, in.nvlsSupport) << "nvlsSupport = " << c.commNvlsSupport << " func = " << c.func
                                          << " datatype = " << c.datatype << " devRedOp = " << c.devRedOp;
  }
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_NoWindows_MeasuresTheAlignmentFromTheRawPointers) {
  for (size_t recvGap : {kAlignedGap, kMisalignedGap}) {
    TaskPrepScene scene;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
    raw->coll.sendbuff = TaskPreTuning_Addr(kSendAddr);
    raw->coll.recvbuff = TaskPreTuning_Addr(kSendAddr + recvGap);
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(recvGap == kAlignedGap, in.symAligned16B) << "recvGap = " << recvGap;
  }
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_BuffersInWindows_MeasuresTheAlignmentFromTheUserPtr) {
  for (bool useWindows : {false, true}) {
    TaskPrepScene scene;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
    raw->coll.sendbuff = TaskPreTuning_Addr(kSendAddr);
    raw->coll.recvbuff = TaskPreTuning_Addr(kRecvAddr);
    TaskPreTuning_Windows windows(useWindows ? TaskPreTuning_Addr(kSendAddr) : nullptr,
                                  TaskPreTuning_Addr(kSendWindowAddr),
                                  useWindows ? TaskPreTuning_Addr(kRecvAddr) : nullptr,
                                  TaskPreTuning_Addr(kRecvWindowAddr));
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(useWindows, in.symAligned16B) << "useWindows = " << useWindows;
  }
}

// One window only: a single (sendWin && recvWin) gate over both offsets would read this pair as misaligned.
TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_OnlyTheRecvBufferInAWindow_StillOffsetsByItsUserPtr) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllReduce);
  raw->coll.sendbuff = TaskPreTuning_Addr(kSendAddr);
  raw->coll.recvbuff = TaskPreTuning_Addr(kRecvWindowAlignedAddr);
  TaskPreTuning_Windows windows(nullptr, TaskPreTuning_Addr(kSendWindowAddr),
                                TaskPreTuning_Addr(kRecvWindowAlignedAddr),
                                TaskPreTuning_Addr(kRecvWindowAddr));
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_TRUE(in.symAligned16B);
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_UnscaledCollRegistrationsCoverBothBuffers_SetsRegBuff) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllReduce);
  RegisteredRanges registered(&scene_, {{raw->coll.sendbuff, kCollSendBytes},
                                       {raw->coll.recvbuff, kCollSendBytes}});
  TaskPreTuning_LocallyValidRegistrations isValid;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(1, in.regBuff);
  ASSERT_EQ(2u, isValid.probed().size());
  // Each buffer must be probed with its own registration; handing regSendBuf to both would pass otherwise.
  EXPECT_NE(isValid.probed().front(), isValid.probed().back());
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_UnscaledCollRegistrationOneByteShort_ClearsRegBuff) {
  for (bool shortenSend : {true, false}) {
    TaskPrepScene scene;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
    RegisteredRanges registered(
      &scene, {{raw->coll.sendbuff, shortenSend ? kCollSendBytes - 1 : kCollSendBytes},
               {raw->coll.recvbuff, shortenSend ? kCollSendBytes : kCollSendBytes - 1}});
    TaskPreTuning_LocallyValidRegistrations isValid;
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(0, in.regBuff) << "shortenSend = " << shortenSend;
  }
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_RegistrationsNotLocallyValid_ClearsRegBuff) {
  for (bool sendValid : {true, false}) {
    TaskPrepScene scene;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
    RegisteredRanges registered(&scene, {{raw->coll.sendbuff, kCollSendBytes},
                                         {raw->coll.recvbuff, kCollSendBytes}});
    std::vector<struct ncclReg*> probed;
    ScopedHook isValid(g_regLocalIsValid, [&probed, sendValid](struct ncclReg* reg, bool* out) {
      const bool isSendBuffer = probed.empty();
      probed.push_back(reg);
      *out = isSendBuffer ? sendValid : !sendValid;
      return ncclSuccess;
    });
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(0, in.regBuff) << "sendValid = " << sendValid;
    ASSERT_EQ(2u, probed.size());
    EXPECT_NE(probed.front(), probed.back());
  }
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_CapturingGraphWithGraphRegister_SetsRegBuff) {
  scene_.SetGraphCapture(true);
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllReduce);
  ScopedHook param(g_loadParam, [](const char* env, int64_t deft) -> int64_t {
    return std::strcmp(env, "GRAPH_REGISTER") == 0 ? 1 : deft;
  });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(1, in.regBuff);
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_GraphArmHalfSatisfied_ClearsRegBuff) {
  for (bool capturing : {true, false}) {
    const int64_t graphRegister = capturing ? 0 : 1;
    TaskPrepScene scene;
    scene.SetGraphCapture(capturing);
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
    ScopedHook param(g_loadParam, [graphRegister](const char* env, int64_t deft) -> int64_t {
      return std::strcmp(env, "GRAPH_REGISTER") == 0 ? graphRegister : deft;
    });
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(0, in.regBuff) << "capturing = " << capturing;
  }
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_RegLocalIsValidFails_PropagatesAtTheFirstFailingProbe) {
  for (int failingCall : {1, 2}) {
    TaskPrepScene scene;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
    int seen = 0;
    ScopedHook isValid(g_regLocalIsValid,
                       [&seen, failingCall](struct ncclReg*, bool* out) -> ncclResult_t {
                         if (++seen == failingCall) {
                           return ncclInternalError;
                         }
                         *out = true;
                         return ncclSuccess;
                       });
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    EXPECT_EQ(ncclInternalError, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(failingCall, seen) << "must stop at the first failing probe";
    EXPECT_EQ(TaskPrep_Poisoned<int>(), in.regBuff);
    EXPECT_EQ(kCollSendBytes, in.nBytes) << "the earlier writes must still have landed";
  }
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_AllGatherRegistrationsCoverTheRealExtent_SetsRegBuff) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllGather);
  RegisteredRanges registered(&scene_, {{raw->coll.sendbuff, kCollSendBytes},
                                       {raw->coll.recvbuff, kAllGatherRecvBytes}});
  TaskPreTuning_LocallyValidRegistrations isValid;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(1, in.regBuff);
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_AllGatherRegistrationShorterThanTheQueriedSize_ClearsRegBuff) {
  for (bool shortenSend : {true, false}) {
    TaskPrepScene scene;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllGather);
    RegisteredRanges registered(
      &scene,
      {{raw->coll.sendbuff, shortenSend ? kAllGatherQueriedSendBytes - 1 : kCollSendBytes},
       {raw->coll.recvbuff, shortenSend ? kAllGatherRecvBytes : kAllGatherQueriedRecvBytes - 1}});
    TaskPreTuning_LocallyValidRegistrations isValid;
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(0, in.regBuff) << "shortenSend = " << shortenSend;
  }
}

TEST_F(TaskPreTuningMicrotest,
       FillCollTuningInput_ReduceScatterRegistrationShorterThanTheQueriedSize_ClearsRegBuff) {
  for (bool shortenSend : {true, false}) {
    TaskPrepScene scene;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncReduceScatter);
    RegisteredRanges registered(
      &scene,
      {{raw->coll.sendbuff, shortenSend ? kReduceScatterSendBytes - 1 : kReduceScatterSendBytes},
       {raw->coll.recvbuff, shortenSend ? kCollSendBytes : kCollSendBytes - 1}});
    TaskPreTuning_LocallyValidRegistrations isValid;
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(0, in.regBuff) << "shortenSend = " << shortenSend;
  }
}

// AICOMRCCL-2421: task_pretuning.cc:68-69 should size the lookup from the real extent, so this must become 0.
TEST_F(TaskPreTuningMicrotest,
       FillCollTuningInput_AllGatherRegistrationShorterThanTheRealExtent_TodaySetsRegBuff) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllGather);
  RegisteredRanges registered(&scene_, {{raw->coll.sendbuff, kAllGatherQueriedSendBytes},
                                       {raw->coll.recvbuff, kAllGatherQueriedRecvBytes}});
  TaskPreTuning_LocallyValidRegistrations isValid;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(1, in.regBuff);
  EXPECT_EQ(kAllGatherRecvBytes, in.nBytes) << "nBytes uses the scaled count and stays correct";
}

// AICOMRCCL-2421 again: Broadcast takes the same elementSize = 1 branch, so an AllGather-only fix misses it.
TEST_F(TaskPreTuningMicrotest,
       FillCollTuningInput_BroadcastRegistrationShorterThanTheRealExtent_TodaySetsRegBuff) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncBroadcast, ncclFloat64);
  RegisteredRanges registered(&scene_, {{raw->coll.sendbuff, kCount}, {raw->coll.recvbuff, kCount}});
  TaskPreTuning_LocallyValidRegistrations isValid;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(1, in.regBuff);
  EXPECT_EQ(kCount * kDoubleSize, in.nBytes) << "nBytes uses the scaled count and stays correct";
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_CTABoundsInRange_PreferEnvThenCallThenComm) {
  struct Case {
    int64_t env;
    int call;
    int expected;
  };
  const std::vector<Case> cases = {{kEnvCTAs, kCallCTAs, kEnvCTAs},
                                   {kOutOfRangeCTAs, kCallCTAs, kCallCTAs},
                                   {NCCL_CONFIG_UNDEF_INT, kOutOfRangeCTAs, kCommCTAs},
                                   {NCCL_CONFIG_UNDEF_INT, kBelowRangeCTAs, kCommCTAs},
                                   {NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, kCommCTAs}};
  for (const Case& c : cases) {
    TaskPrepScene scene;
    scene.comm()->config.minCTAs = kCommCTAs;
    scene.comm()->config.maxCTAs = MAXCHANNELS;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
    raw->coll.collConfig.minCTAs = c.call;
    ScopedHook param(g_loadParam, [&c](const char* env, int64_t deft) -> int64_t {
      return std::strcmp(env, "MIN_CTAS") == 0 ? c.env : deft;
    });
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(c.expected, in.minCTAs) << "env = " << c.env << " call = " << c.call;
  }
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_MaxCTAsFromTheEnv_OutranksTheCallAndCommBounds) {
  scene_.comm()->config.maxCTAs = kCommCTAs;
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllReduce);
  raw->coll.collConfig.maxCTAs = kCallCTAs;
  ScopedHook param(g_loadParam, [](const char* env, int64_t deft) -> int64_t {
    return std::strcmp(env, "MAX_CTAS") == 0 ? kEnvCTAs : deft;
  });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(kEnvCTAs, in.maxCTAs);
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_MaxCTAs_TakesTheSmallerOfTheCallAndCommBounds) {
  for (bool callIsSmaller : {true, false}) {
    TaskPrepScene scene;
    scene.comm()->config.maxCTAs = callIsSmaller ? kLargerCTAs : kSmallerCTAs;
    struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
    raw->coll.collConfig.maxCTAs = callIsSmaller ? kSmallerCTAs : kLargerCTAs;
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene.comm(), &raw->coll, &in));

    EXPECT_EQ(kSmallerCTAs, in.maxCTAs) << "callIsSmaller = " << callIsSmaller;
  }
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_MinCTAsAboveMaxCTAs_ResetsMinCTAsToOne) {
  scene_.comm()->config.minCTAs = kLargerCTAs;
  scene_.comm()->config.maxCTAs = kSmallerCTAs;
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllReduce);
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(1, in.minCTAs);
  EXPECT_EQ(kSmallerCTAs, in.maxCTAs);
}

TEST_F(TaskPreTuningMicrotest, FillCollTuningInput_MinCTAsAtMaxCTAs_KeepsMinCTAs) {
  scene_.comm()->config.minCTAs = kCommCTAs;
  scene_.comm()->config.maxCTAs = kCommCTAs;
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllReduce);
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillCollTuningInput(scene_.comm(), &raw->coll, &in));

  EXPECT_EQ(kCommCTAs, in.minCTAs);
}

TEST_F(TaskPreTuningMicrotest, FillSendRecvTuningInput_UnregisteredBuffer_CopiesTheRawFieldsAndClearsRegBuff) {
  struct ncclRawTask* raw = scene_.NewSendRecv(ncclFuncSend, kPeer);
  raw->sendRecv.datatype = ncclFloat64;
  raw->sendRecv.bytes = kCount * kDoubleSize;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillSendRecvTuningInput(scene_.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(scene_.comm(), in.comm);
  EXPECT_EQ(NCCL_TUNING_MASK_ALL, in.tuningMask);
  EXPECT_EQ(ncclFuncSend, in.func);
  EXPECT_EQ(ncclFloat64, in.datatype);
  EXPECT_EQ(kCount, in.count);
  EXPECT_EQ(kCount, in.countMax);
  EXPECT_EQ(1, in.nWorks);
  EXPECT_EQ(1, in.numPipeOps);
  EXPECT_EQ(kCount * kDoubleSize, in.nBytes);
  EXPECT_EQ(0, in.regBuff);
}

TEST_F(TaskPreTuningMicrotest, FillSendRecvTuningInput_RegistrationCoversTheBuffer_SetsRegBuff) {
  struct ncclRawTask* raw = scene_.NewSendRecv(ncclFuncRecv, kPeer);
  RegisteredRanges registered(&scene_, {{raw->sendRecv.buff, raw->sendRecv.bytes}});
  struct ncclReg* probed = nullptr;
  ScopedHook isValid(g_regLocalIsValid, [&probed](struct ncclReg* reg, bool* out) {
    probed = reg;
    *out = true;
    return ncclSuccess;
  });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillSendRecvTuningInput(scene_.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(1, in.regBuff);
  EXPECT_EQ(1, isValid.calls);
  EXPECT_NE(nullptr, probed);
}

TEST_F(TaskPreTuningMicrotest, FillSendRecvTuningInput_RegistrationShorterThanTheBuffer_ClearsRegBuff) {
  struct ncclRawTask* raw = scene_.NewSendRecv(ncclFuncSend, kPeer);
  RegisteredRanges registered(&scene_, {{raw->sendRecv.buff, raw->sendRecv.bytes - 1}});
  TaskPreTuning_LocallyValidRegistrations isValid;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillSendRecvTuningInput(scene_.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(0, in.regBuff);
}

TEST_F(TaskPreTuningMicrotest, FillSendRecvTuningInput_RegistrationNotLocallyValid_ClearsRegBuff) {
  struct ncclRawTask* raw = scene_.NewSendRecv(ncclFuncSend, kPeer);
  RegisteredRanges registered(&scene_, {{raw->sendRecv.buff, raw->sendRecv.bytes}});
  struct ncclReg* probed = nullptr;
  ScopedHook isValid(g_regLocalIsValid, [&probed](struct ncclReg* reg, bool* out) {
    probed = reg;
    *out = false;
    return ncclSuccess;
  });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillSendRecvTuningInput(scene_.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(0, in.regBuff);
  EXPECT_NE(nullptr, probed) << "the registration must still have been found";
}

TEST_F(TaskPreTuningMicrotest, FillSendRecvTuningInput_CapturingGraphWithGraphRegister_SetsRegBuff) {
  scene_.SetGraphCapture(true);
  struct ncclRawTask* raw = scene_.NewSendRecv(ncclFuncSend, kPeer);
  ScopedHook param(g_loadParam, [](const char* env, int64_t deft) -> int64_t {
    return std::strcmp(env, "GRAPH_REGISTER") == 0 ? 1 : deft;
  });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillSendRecvTuningInput(scene_.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(1, in.regBuff);
}

TEST_F(TaskPreTuningMicrotest, FillSendRecvTuningInput_GraphArmHalfSatisfied_ClearsRegBuff) {
  for (bool capturing : {true, false}) {
    const int64_t graphRegister = capturing ? 0 : 1;
    TaskPrepScene scene;
    scene.SetGraphCapture(capturing);
    struct ncclRawTask* raw = scene.NewSendRecv(ncclFuncSend, kPeer);
    ScopedHook param(g_loadParam, [graphRegister](const char* env, int64_t deft) -> int64_t {
      return std::strcmp(env, "GRAPH_REGISTER") == 0 ? graphRegister : deft;
    });
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillSendRecvTuningInput(scene.comm(), &raw->sendRecv, &in));

    EXPECT_EQ(0, in.regBuff) << "capturing = " << capturing;
  }
}

TEST_F(TaskPreTuningMicrotest, FillSendRecvTuningInput_RegLocalIsValidFails_PropagatesBeforeWritingRegBuff) {
  struct ncclRawTask* raw = scene_.NewSendRecv(ncclFuncSend, kPeer);
  ScopedHook isValid(g_regLocalIsValid, [](struct ncclReg*, bool*) { return ncclInternalError; });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  EXPECT_EQ(ncclInternalError, fillSendRecvTuningInput(scene_.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(TaskPrep_Poisoned<int>(), in.regBuff);
  EXPECT_EQ(kCount * kFloatSize, in.nBytes);
}

TEST_F(TaskPreTuningMicrotest, FillRmaTuningInput_PutSignal_TakesTheCountAndDatatypeFromThePutSignalOp) {
  struct ncclRawTask* raw = scene_.NewRma(ncclFuncPutSignal);
  raw->rma.rmaOp.putSignal.datatype = ncclFloat64;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillRmaTuningInput(scene_.comm(), &raw->rma, &in));

  EXPECT_EQ(scene_.comm(), in.comm);
  EXPECT_EQ(NCCL_TUNING_MASK_ALL, in.tuningMask);
  EXPECT_EQ(ncclFuncPutSignal, in.func);
  EXPECT_EQ(1, in.nWorks);
  EXPECT_EQ(1, in.numPipeOps);
  EXPECT_EQ(ncclFloat64, in.datatype);
  EXPECT_EQ(kCount, in.count);
  EXPECT_EQ(kCount, in.countMax);
  EXPECT_EQ(kCount * kDoubleSize, in.nBytes);
}

TEST_F(TaskPreTuningMicrotest, FillRmaTuningInput_SignalAndWaitSignal_CarryNoPayload) {
  for (ncclFunc_t func : {ncclFuncSignal, ncclFuncWaitSignal}) {
    TaskPrepScene scene;
    struct ncclRawTask* raw = scene.NewRma(func);
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, fillRmaTuningInput(scene.comm(), &raw->rma, &in));

    EXPECT_EQ(func, in.func) << "func = " << func;
    EXPECT_EQ(ncclInt8, in.datatype);
    EXPECT_EQ(0u, in.count);
    EXPECT_EQ(0u, in.countMax);
    EXPECT_EQ(0u, in.nBytes);
  }
}

TEST_F(TaskPreTuningMicrotest, FillRmaTuningInput_UnknownRmaFunc_LeavesThePayloadFieldsUntouched) {
  struct ncclRawTask* raw = scene_.NewRma(ncclFuncAllReduce);
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillRmaTuningInput(scene_.comm(), &raw->rma, &in));

  EXPECT_EQ(ncclFuncAllReduce, in.func);
  EXPECT_EQ(TaskPrep_Poisoned<ncclDataType_t>(), in.datatype);
  EXPECT_EQ(TaskPrep_Poisoned<size_t>(), in.count);
  EXPECT_EQ(TaskPrep_Poisoned<size_t>(), in.countMax);
  EXPECT_EQ(TaskPrep_Poisoned<size_t>(), in.nBytes);
}

TEST_F(TaskPreTuningMicrotest, PreTuningRawTaskPtrEqual_MatchesOnIdentityNotOnContents) {
  struct ncclRawTask* first = scene_.NewColl(ncclFuncBroadcast);
  struct ncclRawTask* second = scene_.NewColl(ncclFuncBroadcast);

  EXPECT_TRUE(preTuningRawTaskPtrEqual(first, first));
  EXPECT_FALSE(preTuningRawTaskPtrEqual(first, second));
  EXPECT_FALSE(preTuningRawTaskPtrEqual(first, nullptr));
  EXPECT_TRUE(preTuningRawTaskPtrEqual(nullptr, nullptr));
}

TEST_F(TaskPreTuningMicrotest, PreTuningBcastFallsBack_EveryBroadcastShape_IsUnconditionallyTrue) {
  for (int root : {0, kPeer}) {
    for (size_t count : {size_t{0}, kCount}) {
      struct ncclRawTask* raw = scene_.NewColl(ncclFuncBroadcast);
      raw->coll.root = root;
      raw->coll.count = count;
      raw->coll.stream = kStream;
      EXPECT_TRUE(preTuningBcastFallsBack(&raw->coll)) << "root = " << root << " count = " << count;
    }
  }
}

TEST_F(TaskPreTuningMicrotest, PreTuningRemoveBcastRawFromQueue_WithoutFree_UnlinksAndKeepsThePoolEmpty) {
  struct ncclRawTask* first = scene_.NewColl(ncclFuncBroadcast);
  struct ncclRawTask* second = scene_.NewColl(ncclFuncBroadcast);
  struct ncclRawTask* third = scene_.NewColl(ncclFuncBroadcast);
  for (struct ncclRawTask* raw : {first, second, third}) {
    scene_.EnqueueBcast(raw);
  }

  ASSERT_EQ(ncclSuccess, preTuningRemoveBcastRawFromQueue(
                           scene_.comm(), &scene_.comm()->rawTaskQueue.bcastQueue, second, false));

  EXPECT_EQ(TaskPreTuning_List({first, third}),
            TaskPreTuning_RawTasks(&scene_.comm()->rawTaskQueue.bcastQueue));
  EXPECT_EQ(nullptr, scene_.comm()->memPool_ncclRawTask.head);
}

TEST_F(TaskPreTuningMicrotest, PreTuningRemoveBcastRawFromQueue_WithFree_ReturnsTheRawTaskToThePool) {
  struct ncclRawTask* only = scene_.NewColl(ncclFuncBroadcast);
  scene_.EnqueueBcast(only);
  ASSERT_EQ(nullptr, scene_.comm()->memPool_ncclRawTask.head);

  ASSERT_EQ(ncclSuccess, preTuningRemoveBcastRawFromQueue(
                           scene_.comm(), &scene_.comm()->rawTaskQueue.bcastQueue, only, true));

  EXPECT_TRUE(ncclIntruQueueEmpty(&scene_.comm()->rawTaskQueue.bcastQueue));
  EXPECT_EQ(static_cast<void*>(only), static_cast<void*>(scene_.comm()->memPool_ncclRawTask.head));
}

TEST_F(TaskPreTuningMicrotest, PreTuningRemoveBcastRawFromQueue_TailRemoved_LeavesTheQueueAppendable) {
  struct ncclRawTask* first = scene_.NewColl(ncclFuncBroadcast);
  struct ncclRawTask* tail = scene_.NewColl(ncclFuncBroadcast);
  struct ncclRawTask* appended = scene_.NewColl(ncclFuncBroadcast);
  scene_.EnqueueBcast(first);
  scene_.EnqueueBcast(tail);

  ASSERT_EQ(ncclSuccess, preTuningRemoveBcastRawFromQueue(
                           scene_.comm(), &scene_.comm()->rawTaskQueue.bcastQueue, tail, false));
  scene_.EnqueueBcast(appended);

  EXPECT_EQ(TaskPreTuning_List({first, appended}),
            TaskPreTuning_RawTasks(&scene_.comm()->rawTaskQueue.bcastQueue));
}

TEST_F(TaskPreTuningMicrotest, PreTuningRemoveBcastRawFromQueue_RawTaskNotQueued_LeavesTheQueueIntact) {
  struct ncclRawTask* queued = scene_.NewColl(ncclFuncBroadcast);
  struct ncclRawTask* stranger = scene_.NewColl(ncclFuncBroadcast);
  scene_.EnqueueBcast(queued);

  ASSERT_EQ(ncclSuccess, preTuningRemoveBcastRawFromQueue(
                           scene_.comm(), &scene_.comm()->rawTaskQueue.bcastQueue, stranger, false));

  EXPECT_EQ(TaskPreTuning_List({queued}),
            TaskPreTuning_RawTasks(&scene_.comm()->rawTaskQueue.bcastQueue));
}

TEST_F(TaskPreTuningMicrotest, PreTuningMergeBcastQueue_EmptyQueue_LeavesTheTuningQueueEmpty) {
  struct ncclTaskTuningInfoQueue tiq;
  ncclIntruQueueConstruct(&tiq.queue);

  ASSERT_EQ(ncclSuccess,
            preTuningMergeBcastQueue(scene_.comm(), &scene_.comm()->rawTaskQueue.bcastQueue, &tiq));

  EXPECT_TRUE(ncclIntruQueueEmpty(&tiq.queue));
}

// preTuningBcastFallsBack is hardcoded true, so every broadcast takes the fallback arm and agvRaw stays null.
TEST_F(TaskPreTuningMicrotest, PreTuningMergeBcastQueue_EveryBroadcastFallsBack_DrainsThemInQueueOrder) {
  struct ncclRawTask* first = scene_.NewColl(ncclFuncBroadcast);
  struct ncclRawTask* second = scene_.NewColl(ncclFuncBroadcast, ncclFloat64);
  struct ncclRawTask* third = scene_.NewColl(ncclFuncBroadcast);
  for (struct ncclRawTask* raw : {first, second, third}) {
    scene_.EnqueueBcast(raw);
  }
  struct ncclTaskTuningInfoQueue tiq;
  ncclIntruQueueConstruct(&tiq.queue);

  ASSERT_EQ(ncclSuccess,
            preTuningMergeBcastQueue(scene_.comm(), &scene_.comm()->rawTaskQueue.bcastQueue, &tiq));

  EXPECT_EQ(TaskPreTuning_List({first, second, third}), TaskPreTuning_TunedRaws(&tiq));
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene_.comm()->rawTaskQueue.bcastQueue));
  EXPECT_EQ(nullptr, scene_.comm()->memPool_ncclRawTask.head) << "the fallback arm must not free the raw task";
  std::vector<struct ncclTaskTuningInfo*> tuned = QueueTasks(&tiq.queue);
  ASSERT_EQ(3u, tuned.size());
  EXPECT_EQ(kCollSendBytes, tuned[0]->tuningIn.count);
  EXPECT_EQ(kCount * kDoubleSize, tuned[1]->tuningIn.count);
  EXPECT_TRUE(CarriesNoTuningEstimate(tuned[0]->tuningOut));
}

TEST_F(TaskPreTuningMicrotest, PreTuningMergeBcastQueue_TuningInputFails_PropagatesAndKeepsTheRemainder) {
  struct ncclRawTask* first = scene_.NewColl(ncclFuncBroadcast);
  struct ncclRawTask* second = scene_.NewColl(ncclFuncBroadcast);
  scene_.EnqueueBcast(first);
  scene_.EnqueueBcast(second);
  struct ncclTaskTuningInfoQueue tiq;
  ncclIntruQueueConstruct(&tiq.queue);
  ScopedHook isValid(g_regLocalIsValid, [](struct ncclReg*, bool*) { return ncclInternalError; });

  EXPECT_EQ(ncclInternalError,
            preTuningMergeBcastQueue(scene_.comm(), &scene_.comm()->rawTaskQueue.bcastQueue, &tiq));

  EXPECT_TRUE(ncclIntruQueueEmpty(&tiq.queue));
  EXPECT_EQ(TaskPreTuning_List({first, second}),
            TaskPreTuning_RawTasks(&scene_.comm()->rawTaskQueue.bcastQueue));
}

TEST_F(TaskPreTuningMicrotest, TaskPreTuning_NullArgument_ReturnsInvalidArgument) {
  struct ncclTaskTuningInfoQueue tiq;
  ncclIntruQueueConstruct(&tiq.queue);
  scene_.EnqueueGeneric(scene_.NewColl(ncclFuncAllReduce));

  EXPECT_EQ(ncclInvalidArgument, ncclTaskPreTuning(nullptr, &scene_.comm()->rawTaskQueue, &tiq));
  EXPECT_EQ(ncclInvalidArgument, ncclTaskPreTuning(scene_.comm(), nullptr, &tiq));
  EXPECT_EQ(ncclInvalidArgument, ncclTaskPreTuning(scene_.comm(), &scene_.comm()->rawTaskQueue, nullptr));
  EXPECT_TRUE(ncclIntruQueueEmpty(&tiq.queue));
  EXPECT_FALSE(ncclIntruQueueEmpty(&scene_.comm()->rawTaskQueue.genericQueue));
}

TEST_F(TaskPreTuningMicrotest, TaskPreTuning_BothQueues_DrainsTheGenericQueueBeforeTheBroadcasts) {
  struct ncclRawTask* coll = scene_.NewColl(ncclFuncAllReduce);
  struct ncclRawTask* sendRecv = scene_.NewSendRecv(ncclFuncSend, kPeer);
  struct ncclRawTask* bcast = scene_.NewColl(ncclFuncBroadcast);
  scene_.EnqueueGeneric(coll);
  scene_.EnqueueGeneric(sendRecv);
  scene_.EnqueueBcast(bcast);
  struct ncclTaskTuningInfoQueue tiq;
  ncclIntruQueueConstruct(&tiq.queue);

  ASSERT_EQ(ncclSuccess, ncclTaskPreTuning(scene_.comm(), &scene_.comm()->rawTaskQueue, &tiq));

  EXPECT_EQ(TaskPreTuning_List({coll, sendRecv, bcast}), TaskPreTuning_TunedRaws(&tiq));
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene_.comm()->rawTaskQueue.genericQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene_.comm()->rawTaskQueue.bcastQueue));
}

TEST_F(TaskPreTuningMicrotest, TaskPreTuning_EachTaskKind_FillsItsOwnTuningInput) {
  struct ncclRawTask* rma = scene_.NewRma(ncclFuncPutSignal);
  rma->rma.rmaOp.putSignal.datatype = ncclFloat64;
  scene_.EnqueueGeneric(scene_.NewColl(ncclFuncAllReduce));
  scene_.EnqueueGeneric(scene_.NewSendRecv(ncclFuncSend, kPeer));
  scene_.EnqueueGeneric(rma);
  scene_.EnqueueGeneric(scene_.NewAllGatherV());
  struct ncclTaskTuningInfoQueue tiq;
  ncclIntruQueueConstruct(&tiq.queue);

  ASSERT_EQ(ncclSuccess, ncclTaskPreTuning(scene_.comm(), &scene_.comm()->rawTaskQueue, &tiq));

  std::vector<struct ncclTaskTuningInfo*> tuned = QueueTasks(&tiq.queue);
  ASSERT_EQ(4u, tuned.size());
  const std::vector<ncclFunc_t> expectedFuncs = {ncclFuncAllReduce, ncclFuncSend, ncclFuncPutSignal,
                                                 ncclFuncAllGatherV};
  const std::vector<size_t> expectedBytes = {kCollSendBytes, kCollSendBytes, kCount * kDoubleSize,
                                             kCount};
  for (size_t i = 0; i < expectedFuncs.size(); i++) {
    EXPECT_EQ(expectedFuncs[i], tuned[i]->tuningIn.func) << "index = " << i;
    EXPECT_EQ(expectedBytes[i], tuned[i]->tuningIn.nBytes) << "index = " << i;
    EXPECT_EQ(scene_.comm(), tuned[i]->tuningIn.comm);
    EXPECT_TRUE(CarriesNoTuningEstimate(tuned[i]->tuningOut));
  }
}

TEST_F(TaskPreTuningMicrotest, TaskPreTuning_UnknownRawTaskKind_ReturnsInternalError) {
  struct ncclRawTask* raw = scene_.NewColl(ncclFuncAllReduce);
  raw->kind = TaskPrep_Poisoned<ncclTaskKind>();
  scene_.EnqueueGeneric(raw);
  struct ncclTaskTuningInfoQueue tiq;
  ncclIntruQueueConstruct(&tiq.queue);

  EXPECT_EQ(ncclInternalError, ncclTaskPreTuning(scene_.comm(), &scene_.comm()->rawTaskQueue, &tiq));

  EXPECT_TRUE(ncclIntruQueueEmpty(&tiq.queue));
}

TEST_F(TaskPreTuningMicrotest, TaskPreTuning_GenericQueueFails_PropagatesBeforeTheBroadcasts) {
  scene_.EnqueueGeneric(scene_.NewColl(ncclFuncAllReduce));
  struct ncclRawTask* bcast = scene_.NewColl(ncclFuncBroadcast);
  scene_.EnqueueBcast(bcast);
  struct ncclTaskTuningInfoQueue tiq;
  ncclIntruQueueConstruct(&tiq.queue);
  ScopedHook isValid(g_regLocalIsValid, [](struct ncclReg*, bool*) { return ncclInternalError; });

  EXPECT_EQ(ncclInternalError, ncclTaskPreTuning(scene_.comm(), &scene_.comm()->rawTaskQueue, &tiq));

  EXPECT_TRUE(ncclIntruQueueEmpty(&tiq.queue));
  // Only a drained generic queue proves the generic arm ran first; a bcast-first drain would leave it full.
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene_.comm()->rawTaskQueue.genericQueue));
  EXPECT_EQ(TaskPreTuning_List({bcast}),
            TaskPreTuning_RawTasks(&scene_.comm()->rawTaskQueue.bcastQueue));
}

// Nothing on the generic queue, so the only path to a failure is the NCCLCHECK around the bcast merge.
TEST_F(TaskPreTuningMicrotest, TaskPreTuning_BcastQueueFails_PropagatesThroughTheMerge) {
  struct ncclRawTask* bcast = scene_.NewColl(ncclFuncBroadcast);
  scene_.EnqueueBcast(bcast);
  struct ncclTaskTuningInfoQueue tiq;
  ncclIntruQueueConstruct(&tiq.queue);
  ScopedHook isValid(g_regLocalIsValid, [](struct ncclReg*, bool*) { return ncclInternalError; });

  EXPECT_EQ(ncclInternalError, ncclTaskPreTuning(scene_.comm(), &scene_.comm()->rawTaskQueue, &tiq));

  EXPECT_TRUE(ncclIntruQueueEmpty(&tiq.queue));
  EXPECT_EQ(TaskPreTuning_List({bcast}),
            TaskPreTuning_RawTasks(&scene_.comm()->rawTaskQueue.bcastQueue));
}

// T3: preTuningBcastFallsBack is hardcoded true, so production never reaches the statics below.
constexpr int kOwnRoot = 1;
constexpr int kOtherRoot = 2;
constexpr int kSpareRoot = 3;
constexpr int kRootBelowRange = -1;
constexpr int kRootAtTheRankCount = kRanks;
constexpr size_t kSmallSliceCount = 8;
constexpr size_t kLargeSliceCount = 40;
constexpr size_t kPresetMaxCount = 999;
constexpr uintptr_t kBcastSendAddr = 0x61000ULL;
constexpr uintptr_t kBcastRecvAddr = 0x72000ULL;

const hipStream_t kOtherStream = reinterpret_cast<hipStream_t>(0xfeedULL);
const hipStream_t kInvalidAggregateStream = reinterpret_cast<hipStream_t>(static_cast<intptr_t>(-1));

struct ncclRawTask* TaskPreTuning_PoisonedRaw(TaskPrepScene* scene) {
  struct ncclRawTask* raw = scene->NewRaw(ncclTaskKindColl);
  std::memset(raw, kPoison, sizeof(*raw));
  return raw;
}

struct ncclRawTaskAllGatherV* TaskPreTuning_NewAggregate(TaskPrepScene* scene) {
  struct ncclRawTask* raw = TaskPreTuning_PoisonedRaw(scene);
  EXPECT_EQ(ncclSuccess, preTuningInitAllGatherVRaw(scene->comm(), raw));
  return &raw->allGatherV;
}

struct ncclRawTaskColl* TaskPreTuning_NewBcast(TaskPrepScene* scene, int root, size_t count,
                                               ncclDataType_t datatype = ncclFloat64) {
  struct ncclRawTask* raw = scene->NewColl(ncclFuncBroadcast, datatype);
  raw->coll.root = root;
  raw->coll.count = count;
  raw->coll.stream = kStream;
  raw->coll.sendbuff = TaskPreTuning_Addr(kBcastSendAddr + root);
  raw->coll.recvbuff = TaskPreTuning_Addr(kBcastRecvAddr + root);
  return &raw->coll;
}

TEST_F(TaskPreTuningMicrotest,
       UnreachableMerge_InitAllGatherVRaw_PoisonedRawTask_ResetsEveryFieldAndClearsBothSliceArrays) {
  struct ncclRawTask* raw = TaskPreTuning_PoisonedRaw(&scene_);

  ASSERT_EQ(ncclSuccess, preTuningInitAllGatherVRaw(scene_.comm(), raw));

  struct ncclRawTaskAllGatherV* agv = &raw->allGatherV;
  EXPECT_EQ(ncclTaskKindAllGatherV, raw->kind);
  EXPECT_EQ(ncclFuncAllGatherV, agv->func);
  EXPECT_EQ(kRanks, agv->nRanks);
  EXPECT_EQ(nullptr, agv->sendbuff);
  EXPECT_EQ(0u, agv->maxCount);
  EXPECT_EQ(ncclInt8, agv->datatype);
  EXPECT_EQ(kInvalidAggregateStream, agv->stream);
  ASSERT_NE(nullptr, agv->recvbuff);
  ASSERT_NE(nullptr, agv->counts);
  for (int root = 0; root < kRanks; root++) {
    EXPECT_EQ(nullptr, agv->recvbuff[root]) << "root = " << root;
    EXPECT_EQ(0u, agv->counts[root]) << "root = " << root;
  }
}

TEST_F(TaskPreTuningMicrotest,
       UnreachableMerge_InitAllGatherVRaw_AnyRankCount_GivesEveryRankASliceEntryNoLaterAllocationOverlaps) {
  struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene_);
  for (int root = 0; root < kRanks; root++) {
    agv->recvbuff[root] = TaskPreTuning_Addr(kBcastRecvAddr + root);
    agv->counts[root] = kSmallSliceCount + root;
  }

  size_t* later = ncclMemoryStackAlloc<size_t>(&scene_.comm()->memScoped, kRanks);
  std::memset(later, kPoison, kRanks * sizeof(size_t));

  for (int root = 0; root < kRanks; root++) {
    EXPECT_EQ(TaskPreTuning_Addr(kBcastRecvAddr + root), agv->recvbuff[root]) << "root = " << root;
    EXPECT_EQ(kSmallSliceCount + root, agv->counts[root]) << "root = " << root;
  }
}

TEST_F(TaskPreTuningMicrotest, UnreachableMerge_InitAllGatherVRaw_TwoAggregates_GetSliceArraysOfTheirOwn) {
  struct ncclRawTaskAllGatherV* first = TaskPreTuning_NewAggregate(&scene_);
  struct ncclRawTaskAllGatherV* second = TaskPreTuning_NewAggregate(&scene_);
  first->counts[kOtherRoot] = kLargeSliceCount;

  EXPECT_NE(first->counts, second->counts);
  EXPECT_NE(first->recvbuff, second->recvbuff);
  EXPECT_EQ(0u, second->counts[kOtherRoot]);
}

TEST_F(TaskPreTuningMicrotest, UnreachableMerge_AllGatherVRootUsed_OneRootCarriesASlice_IsTrueForThatRootAlone) {
  struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene_);
  agv->counts[kOtherRoot] = kSmallSliceCount;

  for (int root = 0; root < kRanks; root++) {
    EXPECT_EQ(root == kOtherRoot, preTuningAllGatherVRootUsed(agv, root)) << "root = " << root;
  }
}

TEST_F(TaskPreTuningMicrotest, UnreachableMerge_BcastFitsAllGatherV_SameStreamAndUnusedRoot_AcceptsTheBroadcast) {
  struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene_);
  agv->stream = kStream;
  agv->counts[kSpareRoot] = kSmallSliceCount;
  struct ncclRawTaskColl* bcast = TaskPreTuning_NewBcast(&scene_, kOtherRoot, kSmallSliceCount);

  EXPECT_TRUE(preTuningBcastFitsAllGatherV(agv, bcast));
}

TEST_F(TaskPreTuningMicrotest, UnreachableMerge_BcastFitsAllGatherV_AnotherStream_RejectsTheBroadcastOnAnUnusedRoot) {
  struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene_);
  agv->stream = kOtherStream;
  struct ncclRawTaskColl* bcast = TaskPreTuning_NewBcast(&scene_, kOtherRoot, kSmallSliceCount);

  EXPECT_FALSE(preTuningBcastFitsAllGatherV(agv, bcast));
}

TEST_F(TaskPreTuningMicrotest, UnreachableMerge_BcastFitsAllGatherV_RootAlreadyCarriesASlice_RejectsTheBroadcast) {
  struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene_);
  agv->stream = kStream;
  agv->counts[kOtherRoot] = kSmallSliceCount;
  struct ncclRawTaskColl* bcast = TaskPreTuning_NewBcast(&scene_, kOtherRoot, kLargeSliceCount);

  EXPECT_FALSE(preTuningBcastFitsAllGatherV(agv, bcast));
}

TEST_F(TaskPreTuningMicrotest,
       UnreachableMerge_AddBcastToAllGatherV_RootOutsideTheRankRange_RejectsItWithoutTouchingTheAggregate) {
  for (int root : {kRootBelowRange, kRootAtTheRankCount}) {
    TaskPrepScene scene(kRanks, kOwnRoot);
    struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene);
    agv->maxCount = kPresetMaxCount;
    struct ncclRawTaskColl* bcast = TaskPreTuning_NewBcast(&scene, kOtherRoot, kLargeSliceCount);
    bcast->root = root;

    EXPECT_EQ(ncclInvalidArgument, preTuningAddBcastToAllGatherV(scene.comm(), agv, bcast)) << "root = " << root;

    EXPECT_EQ(kInvalidAggregateStream, agv->stream) << "root = " << root;
    EXPECT_EQ(kPresetMaxCount, agv->maxCount) << "root = " << root;
    EXPECT_EQ(nullptr, agv->sendbuff) << "root = " << root;
  }
}

TEST_F(TaskPreTuningMicrotest, UnreachableMerge_AddBcastToAllGatherV_RootIsThisRank_TakesTheBroadcastSourceBuffer) {
  TaskPrepScene scene(kRanks, kOwnRoot);
  struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene);
  struct ncclRawTaskColl* bcast = TaskPreTuning_NewBcast(&scene, kOwnRoot, kSmallSliceCount);

  ASSERT_EQ(ncclSuccess, preTuningAddBcastToAllGatherV(scene.comm(), agv, bcast));

  EXPECT_EQ(TaskPreTuning_Addr(kBcastSendAddr + kOwnRoot), agv->sendbuff);
  EXPECT_EQ(TaskPreTuning_Addr(kBcastRecvAddr + kOwnRoot), agv->recvbuff[kOwnRoot]);
}

TEST_F(TaskPreTuningMicrotest, UnreachableMerge_AddBcastToAllGatherV_RootIsAnotherRank_LeavesTheSourceBufferUnset) {
  TaskPrepScene scene(kRanks, kOwnRoot);
  struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene);
  struct ncclRawTaskColl* bcast = TaskPreTuning_NewBcast(&scene, kOtherRoot, kSmallSliceCount);

  ASSERT_EQ(ncclSuccess, preTuningAddBcastToAllGatherV(scene.comm(), agv, bcast));

  EXPECT_EQ(nullptr, agv->sendbuff);
  EXPECT_EQ(TaskPreTuning_Addr(kBcastRecvAddr + kOtherRoot), agv->recvbuff[kOtherRoot]);
}

TEST_F(TaskPreTuningMicrotest,
       UnreachableMerge_AddBcastToAllGatherV_AnyRoot_FillsThatRootsSliceInBytesAndAdoptsTheBroadcastStream) {
  struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene_);
  struct ncclRawTaskColl* bcast = TaskPreTuning_NewBcast(&scene_, kOtherRoot, kSmallSliceCount);

  ASSERT_EQ(ncclSuccess, preTuningAddBcastToAllGatherV(scene_.comm(), agv, bcast));

  EXPECT_EQ(kStream, agv->stream);
  EXPECT_EQ(kSmallSliceCount * kDoubleSize, agv->counts[kOtherRoot]);
  EXPECT_EQ(kSmallSliceCount * kDoubleSize, agv->maxCount);
  EXPECT_EQ(kSmallSliceCount, bcast->count) << "the broadcast must not be rewritten";
  for (int root = 0; root < kRanks; root++) {
    if (root == kOtherRoot) {
      continue;
    }
    EXPECT_EQ(nullptr, agv->recvbuff[root]) << "root = " << root;
    EXPECT_EQ(0u, agv->counts[root]) << "root = " << root;
  }
}

TEST_F(TaskPreTuningMicrotest,
       UnreachableMerge_AddBcastToAllGatherV_SeveralRoots_FillTheirOwnSlicesAndTrackTheLargest) {
  struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene_);
  struct ncclRawTaskColl* small = TaskPreTuning_NewBcast(&scene_, kSpareRoot, kSmallSliceCount);
  struct ncclRawTaskColl* large = TaskPreTuning_NewBcast(&scene_, kOtherRoot, kLargeSliceCount);

  ASSERT_EQ(ncclSuccess, preTuningAddBcastToAllGatherV(scene_.comm(), agv, small));
  ASSERT_EQ(ncclSuccess, preTuningAddBcastToAllGatherV(scene_.comm(), agv, large));

  EXPECT_EQ(kSmallSliceCount * kDoubleSize, agv->counts[kSpareRoot]);
  EXPECT_EQ(kLargeSliceCount * kDoubleSize, agv->counts[kOtherRoot]);
  EXPECT_EQ(TaskPreTuning_Addr(kBcastRecvAddr + kSpareRoot), agv->recvbuff[kSpareRoot]);
  EXPECT_EQ(TaskPreTuning_Addr(kBcastRecvAddr + kOtherRoot), agv->recvbuff[kOtherRoot]);
  EXPECT_EQ(kLargeSliceCount * kDoubleSize, agv->maxCount);
}

TEST_F(TaskPreTuningMicrotest,
       UnreachableMerge_AddBcastToAllGatherV_SmallerFollowingBroadcast_LeavesTheLargestSliceAlone) {
  struct ncclRawTaskAllGatherV* agv = TaskPreTuning_NewAggregate(&scene_);
  struct ncclRawTaskColl* large = TaskPreTuning_NewBcast(&scene_, kOtherRoot, kLargeSliceCount);
  struct ncclRawTaskColl* small = TaskPreTuning_NewBcast(&scene_, kSpareRoot, kSmallSliceCount);

  ASSERT_EQ(ncclSuccess, preTuningAddBcastToAllGatherV(scene_.comm(), agv, large));
  ASSERT_EQ(ncclSuccess, preTuningAddBcastToAllGatherV(scene_.comm(), agv, small));

  EXPECT_EQ(kSmallSliceCount * kDoubleSize, agv->counts[kSpareRoot]);
  EXPECT_EQ(kLargeSliceCount * kDoubleSize, agv->maxCount);
}

TEST_F(TaskPreTuningMicrotest,
       UnreachableMerge_FillAllGatherVTuningInput_AnyAggregate_CopiesTheRawFieldsAndSizesNBytesFromTheLargestSlice) {
  struct ncclRawTask* raw = scene_.NewAllGatherV();
  raw->allGatherV.datatype = ncclFloat64;
  raw->allGatherV.maxCount = kLargeSliceCount;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillAllGatherVTuningInput(scene_.comm(), &raw->allGatherV, &in));

  EXPECT_EQ(scene_.comm(), in.comm);
  EXPECT_EQ(NCCL_TUNING_MASK_ALL, in.tuningMask);
  EXPECT_EQ(ncclFuncAllGatherV, in.func);
  EXPECT_EQ(ncclFloat64, in.datatype);
  EXPECT_EQ(1, in.nWorks);
  EXPECT_EQ(1, in.numPipeOps);
  EXPECT_EQ(kLargeSliceCount, in.count);
  EXPECT_EQ(kLargeSliceCount, in.countMax);
  EXPECT_EQ(kLargeSliceCount * kDoubleSize, in.nBytes);
}

TEST_F(TaskPreTuningMicrotest,
       UnreachableMerge_FillAllGatherVTuningInput_AggregateCarriesAnotherFunc_StillReportsAllGatherV) {
  struct ncclRawTask* raw = scene_.NewAllGatherV();
  raw->allGatherV.func = ncclFuncBroadcast;
  raw->allGatherV.maxCount = kLargeSliceCount;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillAllGatherVTuningInput(scene_.comm(), &raw->allGatherV, &in));

  EXPECT_EQ(ncclFuncAllGatherV, in.func);
}

TEST_F(TaskPreTuningMicrotest, UnreachableMerge_FillAllGatherVTuningInput_EmptyAggregate_ReportsNoPayload) {
  struct ncclRawTask* raw = scene_.NewAllGatherV();
  raw->allGatherV.maxCount = 0;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillAllGatherVTuningInput(scene_.comm(), &raw->allGatherV, &in));

  EXPECT_EQ(ncclInt8, in.datatype);
  EXPECT_EQ(0u, in.count);
  EXPECT_EQ(0u, in.countMax);
  EXPECT_EQ(0u, in.nBytes);
}

TEST_F(TaskPreTuningMicrotest,
       UnreachableMerge_FillAllGatherVTuningInput_AnyAggregate_LeavesTheCollectiveOnlyFieldsToTheCaller) {
  struct ncclRawTask* raw = scene_.NewAllGatherV();
  raw->allGatherV.maxCount = kLargeSliceCount;
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, fillAllGatherVTuningInput(scene_.comm(), &raw->allGatherV, &in));

  EXPECT_EQ(TaskPrep_Poisoned<int>(), in.regBuff);
  EXPECT_EQ(TaskPrep_Poisoned<int>(), in.nvlsSupport);
  EXPECT_EQ(TaskPrep_Poisoned<ncclRedOp_t>(), in.redOp);
}

}  // namespace
