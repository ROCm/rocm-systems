/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/tuner/tuner_v6.cc, the adapter that presents a v6 tuner plugin as the
// current ncclTuner_t: v7 <-> v5 constants in init(), and a constants copy that outlives init(), since a v6
// plugin may keep the pointer it was given until finalize().

#include <gtest/gtest.h>
#include <dlfcn.h>

#include <cstring>

#include "nccl_tuner.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x6);
ncclTuner_v6_t* g_export = nullptr;

void* FakeDlsym(void* lib, const char* symbol) {
  if (lib != kLib || std::strcmp(symbol, "ncclTunerPlugin_v6")) return nullptr;
  return g_export;
}

}  // namespace

#define dlsym FakeDlsym
#include TUNER_V6_CC_PATH
#undef dlsym

namespace {

constexpr double kCore = 3.25;
constexpr double kOverride = 42.5;
int g_pluginContext;

// What the v6 plugin was last called with.
struct Rec {
  int initCalls = 0;
  int finalizeCalls = 0;
  ncclResult_t initResult = ncclSuccess;
  ncclTunerConstants_v6_t* constants = nullptr;
  void* collInfoContext = nullptr;
  void* chunkSizeContext = nullptr;
  void* finalizeContext = reinterpret_cast<void*>(0x1);
  double treeLlAtCollInfo = 0;
};
Rec g;

ncclTuner_v6_t MakeTuner(bool withChunkSize) {
  ncclTuner_v6_t t{};
  t.name = "Fake6";
  t.init = [](void** context, uint64_t, size_t, size_t, ncclDebugLogger_t, ncclNvlDomainInfo_v6_t*,
              ncclTunerConstants_v6_t* constants) {
    ++g.initCalls;
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
  if (withChunkSize) {
    t.getChunkSize = [](void* context, ncclFunc_t, size_t, int, int, int, size_t* chunkSize) {
      g.chunkSizeContext = context;
      *chunkSize = 4096;
      return ncclSuccess;
    };
  }
  return t;
}

// Reuses the stack init() ran on, as the core's later calls do.
__attribute__((noinline)) void ClobberStack() {
  volatile unsigned char junk[64 << 10];
  for (size_t i = 0; i < sizeof(junk); ++i) junk[i] = 0xa5;
}

class TunerV6 : public ::testing::Test {
 protected:
  void Load(bool withChunkSize = true) {
    g = Rec{};
    plugin = MakeTuner(withChunkSize);
    g_export = &plugin;
    tuner = getNcclTuner_v6(kLib);
    ASSERT_NE(tuner, nullptr);
    core = {};
    core.baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_LL] = kCore;
  }
  ncclResult_t Init() { return tuner->init(&context, 1, 8, 1, nullptr, nullptr, &core); }

  ncclTuner_v6_t plugin{};
  ncclTuner_t* tuner = nullptr;
  ncclTunerConstants_t core{};
  void* context = nullptr;
};

TEST_F(TunerV6, MissingSymbolIsNotAV6Tuner) {
  g_export = nullptr;
  EXPECT_EQ(getNcclTuner_v6(kLib), nullptr);
}

TEST_F(TunerV6, KeepsThePluginName) {
  Load();
  EXPECT_STREQ(tuner->name, "Fake6");
}

TEST_F(TunerV6, InitSeesCoreConstantsAndCoreSeesItsWrites) {
  Load();
  ASSERT_EQ(Init(), ncclSuccess);
  ASSERT_NE(g.constants, nullptr);
  EXPECT_EQ(g.constants->baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_LL], kCore);
  EXPECT_EQ(core.baseLatencies[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE], kOverride);
  EXPECT_EQ(tuner->finalize(context), ncclSuccess);
}

TEST_F(TunerV6, ConstantsPointerStaysValidUntilFinalize) {
  Load();
  ASSERT_EQ(Init(), ncclSuccess);
  ClobberStack();
  ASSERT_EQ(tuner->getCollInfo(context, ncclFuncAllReduce, 1024, 1, nullptr, 0, 0, 0, nullptr), ncclSuccess);
  EXPECT_EQ(g.treeLlAtCollInfo, kCore) << "the constants init() got did not outlive init()";
  EXPECT_EQ(tuner->finalize(context), ncclSuccess);
}

TEST_F(TunerV6, PluginContextReachesEveryCall) {
  Load();
  ASSERT_EQ(Init(), ncclSuccess);
  size_t chunkSize = 0;
  ASSERT_EQ(tuner->getCollInfo(context, ncclFuncAllReduce, 1024, 1, nullptr, 0, 0, 0, nullptr), ncclSuccess);
  ASSERT_NE(tuner->getChunkSize, nullptr);
  ASSERT_EQ(tuner->getChunkSize(context, ncclFuncAllReduce, 1024, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 1, &chunkSize),
            ncclSuccess);
  ASSERT_EQ(tuner->finalize(context), ncclSuccess);
  EXPECT_EQ(g.collInfoContext, &g_pluginContext);
  EXPECT_EQ(g.chunkSizeContext, &g_pluginContext);
  EXPECT_EQ(chunkSize, 4096u);
  EXPECT_EQ(g.finalizeCalls, 1);
  EXPECT_EQ(g.finalizeContext, &g_pluginContext);
}

TEST_F(TunerV6, NoChunkSizeHookWhenThePluginHasNone) {
  Load(/*withChunkSize=*/false);
  ASSERT_EQ(Init(), ncclSuccess);
  EXPECT_EQ(tuner->getChunkSize, nullptr);
  EXPECT_EQ(tuner->finalize(context), ncclSuccess);
}

TEST_F(TunerV6, InitFailureIsReturnedAndFinalizeStillWorks) {
  Load();
  g.initResult = ncclInternalError;
  EXPECT_EQ(Init(), ncclInternalError);
  EXPECT_EQ(context, nullptr);
  EXPECT_EQ(tuner->finalize(context), ncclSuccess);
  EXPECT_EQ(g.finalizeContext, nullptr);
}

}  // namespace
