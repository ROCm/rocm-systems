/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/tuner/tuner_v5.cc, the adapter that presents a v5 tuner plugin as the
// current ncclTuner_t: v7 <-> v5 constants in init(), and a constants copy that outlives init(), since a v5
// plugin may keep the pointer it was given until finalize().

#include <gtest/gtest.h>

#include <cstring>

#include "nccl_tuner.h"
#include "os.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x5);
ncclTuner_v5_t* g_export = nullptr;

void* FakeDlsym(void* lib, const char* symbol) {
  if (lib != kLib || std::strcmp(symbol, "ncclTunerPlugin_v5")) return nullptr;
  return g_export;
}

}  // namespace

#define ncclOsDlsym FakeDlsym
#include TUNER_V5_CC_PATH
#undef ncclOsDlsym

namespace {

constexpr double kCore = 3.25;
constexpr double kOverride = 42.5;
int g_pluginContext;

// What the v5 plugin was last called with.
struct Rec {
  int finalizeCalls = 0;
  ncclResult_t initResult = ncclSuccess;
  ncclTunerConstants_v5_t* constants = nullptr;
  void* collInfoContext = nullptr;
  void* finalizeContext = reinterpret_cast<void*>(0x1);
  double treeLlAtCollInfo = 0;
};
Rec g;

ncclTuner_v5_t MakeTuner() {
  ncclTuner_v5_t t{};
  t.name = "Fake5";
  t.init = [](void** context, uint64_t, size_t, size_t, ncclDebugLogger_t, ncclNvlDomainInfo_v5_t*,
              ncclTunerConstants_v5_t* constants) {
    g.constants = constants;
    constants->baseLatencies[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = kOverride;
    *context = &g_pluginContext;
    return g.initResult;
  };
  t.getCollInfo = [](void* context, ncclFunc_t, size_t, int, float**, int, int, int, int*) {
    g.collInfoContext = context;
    g.treeLlAtCollInfo = g.constants->baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_LL];
    return ncclSuccess;
  };
  t.finalize = [](void* context) {
    ++g.finalizeCalls;
    g.finalizeContext = context;
    return ncclSuccess;
  };
  return t;
}

// Reuses the stack init() ran on, as the core's later calls do.
__attribute__((noinline)) void ClobberStack() {
  volatile unsigned char junk[64 << 10];
  for (size_t i = 0; i < sizeof(junk); ++i) junk[i] = 0xa5;
}

class TunerV5 : public ::testing::Test {
 protected:
  void SetUp() override {
    plugin = MakeTuner();
    g_export = &plugin;
    tuner = getNcclTuner_v5(kLib);
    ASSERT_NE(tuner, nullptr);
    core.baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_LL] = kCore;
  }
  ncclResult_t Init() { return tuner->init(&context, 1, 8, 1, nullptr, nullptr, &core); }

  ncclTuner_v5_t plugin{};
  ncclTuner_t* tuner = nullptr;
  ncclTunerConstants_t core{};
  void* context = nullptr;
};

TEST_F(TunerV5, MissingSymbolIsNotAV5Tuner) {
  g_export = nullptr;
  EXPECT_EQ(getNcclTuner_v5(kLib), nullptr);
}

TEST_F(TunerV5, KeepsThePluginNameAndHasNoChunkSizeHook) {
  EXPECT_STREQ(tuner->name, "Fake5");
  EXPECT_EQ(tuner->getChunkSize, nullptr);
}

TEST_F(TunerV5, InitSeesCoreConstantsAndCoreSeesItsWrites) {
  g = Rec{};
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_NE(g.constants, nullptr);
  EXPECT_EQ(g.constants->baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_LL], kCore);
  EXPECT_EQ(core.baseLatencies[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE], kOverride);
  EXPECT_EQ(tuner->finalize(context), ncclSuccess);
}

TEST_F(TunerV5, ConstantsPointerStaysValidUntilFinalize) {
  g = Rec{};
  ASSERT_EQ(Init(), ncclSuccess);
  ClobberStack();
  ASSERT_EQ(tuner->getCollInfo(context, ncclFuncAllReduce, 1024, 1, nullptr, 0, 0, 0, nullptr), ncclSuccess);
  EXPECT_EQ(g.treeLlAtCollInfo, kCore) << "the constants init() got did not outlive init()";
  EXPECT_EQ(tuner->finalize(context), ncclSuccess);
}

TEST_F(TunerV5, PluginContextReachesEveryCall) {
  g = Rec{};
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_EQ(tuner->getCollInfo(context, ncclFuncAllReduce, 1024, 1, nullptr, 0, 0, 0, nullptr), ncclSuccess);
  ASSERT_EQ(tuner->finalize(context), ncclSuccess);
  EXPECT_EQ(g.collInfoContext, &g_pluginContext);
  EXPECT_EQ(g.finalizeCalls, 1);
  EXPECT_EQ(g.finalizeContext, &g_pluginContext);
}

TEST_F(TunerV5, InitFailureIsReturnedAndFinalizeStillWorks) {
  g = Rec{};
  g.initResult = ncclInternalError;
  EXPECT_EQ(Init(), ncclInternalError);
  EXPECT_NE(context, nullptr);
  EXPECT_EQ(tuner->finalize(context), ncclSuccess);
  // As with a v7 plugin, finalize() gets the context init() stored before it failed.
  EXPECT_EQ(g.finalizeContext, &g_pluginContext);
}

}  // namespace
