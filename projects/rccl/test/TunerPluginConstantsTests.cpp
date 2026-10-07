/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// TunerPluginConstants: a tuner plugin's init() takes the cost-model constants as an input/output argument
// (plugin/tuner/tuner_v6.h). They must already hold the core's values when init() runs, and what init() writes
// must survive the cost-model setup that follows it in ncclTuningInit() (src/tuning/tuning.cc). The tuner below
// is built into rccl-UnitTests, loaded via NCCL_TUNER_PLUGIN=STATIC_PLUGIN; test/CMakeLists.txt exports it.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "plugin/nccl_tuner.h"

namespace {

// Exactly representable and not one of the defaults.
constexpr double kOverride = 42.5;

int initCalls = 0;
ncclTunerConstants_v6_t seenAtInit = {};
ncclTunerConstants_v6_t* coreConstants = nullptr; // the core's own copy; valid until ncclCommDestroy()

ncclResult_t tunerInit(void** context, uint64_t, size_t, size_t, ncclDebugLogger_t, ncclNvlDomainInfo_v6_t*,
                       ncclTunerConstants_v6_t* constants) {
  *context = nullptr;
  ++initCalls;
  if (constants == nullptr) return ncclSuccess;
  seenAtInit = *constants;
  coreConstants = constants;
  constants->baseLatencies[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = kOverride;
  return ncclSuccess;
}

ncclResult_t tunerGetCollInfo(void*, ncclFunc_t, size_t, int, float**, int, int, int, int*) { return ncclSuccess; }
ncclResult_t tunerFinalize(void*) { return ncclSuccess; }

// Returns the reason this host cannot run the test, or "" when it can.
// GTEST_SKIP() must be issued by the caller: it expands to a bare return and
// would otherwise only leave this helper, letting the test body run on.
std::string gpuSkipReason() {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1) return "requires at least one GPU";
  return "";
}

// Runs `check` while a one-rank communicator that loaded the test tuner is alive.
// Callers must be isolated, since the tuner load outcome is latched in a process global,
// and must skip on gpuSkipReason() first.
template <typename Check>
void withTestTunerComm(Check check) {
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(setenv("NCCL_TUNER_PLUGIN", "STATIC_PLUGIN", 1), 0);
  ncclUniqueId id;
  ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
  ncclComm_t comm = nullptr;
  ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
  ASSERT_EQ(initCalls, 1) << "the test tuner was not loaded, or its init() did not run exactly once";
  ASSERT_NE(coreConstants, nullptr) << "init() was given no constants";
  check();
  ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

} // namespace

// Not const: clang would then also emit it for the GPU, where the host functions it points to don't exist.
extern "C" __attribute__((visibility("default"))) ncclTuner_v6_t ncclTunerPlugin_v6 = {
  .name = "TunerPluginConstantsTest",
  .init = tunerInit,
  .getCollInfo = tunerGetCollInfo,
  .finalize = tunerFinalize,
  .getChunkSize = nullptr,
};

namespace RcclUnitTesting {

// Control: the plugin loads and initializes on any tree, so a failure below is about the constants.
TEST(TunerPluginConstants, TestTunerIsLoaded) {
  RUN_ISOLATED_TEST("TunerPluginConstants.TestTunerIsLoaded", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    withTestTunerComm([] {});
  });
}

TEST(TunerPluginConstants, InitSeesCoreConstantsAndKeepsItsWrites) {
  RUN_ISOLATED_TEST("TunerPluginConstants.InitSeesCoreConstantsAndKeepsItsWrites", []() {
    if (auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
    withTestTunerComm([] {
      // The comparison below would also hold if the constants were never populated at all.
      EXPECT_GT(seenAtInit.baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_LL], 0.0) << "init() saw unpopulated constants";
      ncclTunerConstants_v6_t core = *coreConstants;
      double& written = core.baseLatencies[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
      EXPECT_EQ(written, kOverride) << "the cost model overwrote the constants init() set";
      written = seenAtInit.baseLatencies[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
      EXPECT_EQ(memcmp(&core, &seenAtInit, sizeof(core)), 0)
        << "init() did not get the constants the cost model uses; e.g. Tree/LL base latency was "
        << seenAtInit.baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_LL] << ", the core's is "
        << core.baseLatencies[NCCL_ALGO_TREE][NCCL_PROTO_LL];
    });
  });
}

} // namespace RcclUnitTesting
