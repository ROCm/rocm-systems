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

}  // namespace
