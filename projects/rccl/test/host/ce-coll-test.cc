/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ce_coll.cc: the UUT is #include'd, so its launch entry points run
// against fakes/ with no GPU or HIP runtime.

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/dev_runtime_fakes.h"
#include "fakes/nccl_fakes.h"  // g_loadParam, used by param_redirect.h
#include "fakes/nccl_stubs.h"

#include "fakes/param_redirect.h"
#include "fakes/nvtx_redirect.h"

#include CE_COLL_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

constexpr char kLaunchNoStagingWarn[] = "launched without its staging buffer";

// Launch runs on one thread for every local rank, so a collective window registration there deadlocks (ROCM-32044).
class CeCollMicrotest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;
  ncclCudaStreamList streams_{};
  ncclCeCollArgs args_{};
  ncclKernelPlan plan_{};
  uint8_t staging_ = 0;
  uint8_t allocated_ = 0;
  // Lets a regressed launch-time staging setup reach the window registration the tests count instead of aborting.
  ScopedHook<ncclResult_t(void**, size_t)> alloc_{g_ncclMemAlloc, [this](void** ptr, size_t) {
    *ptr = &allocated_;
    return ncclSuccess;
  }};

  void SetUp() override {
    ResetDevRuntimeFakes();
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = 4;
    comm_->nNodes = 1;
    comm_->planner.streams = &streams_;
    args_.func = ncclFuncAllReduce;
    args_.datatype = ncclFloat32;
    args_.eltSize = sizeof(float);
    plan_.ceCollArgs = &args_;
  }
  void TearDown() override { ResetDevRuntimeFakes(); }

  ncclResult_t Run(const std::function<ncclResult_t()>& body, std::string* log) {
    ncclResult_t res = ncclSuccess;
    *log = CaptureLog([&]() { res = body(); });
    return res;
  }
  ncclResult_t Launch(std::string* log) {
    return Run([&]() { return ncclLaunchCeColl(comm_.get(), &plan_); }, log);
  }
};

TEST_F(CeCollMicrotest, LaunchAllReduceWithoutStaging_FailsLoudlyWithoutRegistering) {
  ScopedHook registration(g_devrWindowRegisterInGroup,
                          [](ncclComm*, void*, size_t, int, ncclWindow_t*) { return ncclInternalError; });
  std::string log;

  EXPECT_EQ(ncclInvalidUsage, Launch(&log));

  EXPECT_TRUE(LogHas(log, kLaunchNoStagingWarn)) << log;
  EXPECT_EQ(0, registration.calls);
}

TEST_F(CeCollMicrotest, LaunchNonStagingCollWithoutStaging_SkipsTheStagingCheck) {
  args_.func = ncclFuncBroadcast;  // no CE implementation, so the dispatch default arm rejects it after the check
  std::string log;

  EXPECT_EQ(ncclInvalidUsage, Launch(&log));

  EXPECT_FALSE(LogHas(log, kLaunchNoStagingWarn)) << log;
}

TEST_F(CeCollMicrotest, LaunchAllReduceWithStaging_ReachesTheCollectiveWithoutRegistering) {
  comm_->ceColl.ceARTmpBuf = &staging_;
  args_.nElts = 0;  // stops ncclCeAllReduce at its layout check, before any copy is issued
  ScopedHook registration(g_devrWindowRegisterInGroup,
                          [](ncclComm*, void*, size_t, int, ncclWindow_t*) { return ncclInternalError; });
  std::string log;

  EXPECT_EQ(ncclInvalidArgument, Launch(&log));

  EXPECT_TRUE(LogHas(log, "CE AllReduce: no valid chunk layout")) << log;
  EXPECT_EQ(0, registration.calls);
}

// The eager 2-shot path calls ncclCeAllReduce directly, bypassing ncclLaunchCeColl.
TEST_F(CeCollMicrotest, CeAllReduceWithoutStaging_FailsLoudlyWithoutRegistering) {
  ScopedHook registration(g_devrWindowRegisterInGroup,
                          [](ncclComm*, void*, size_t, int, ncclWindow_t*) { return ncclInternalError; });
  std::string log;

  EXPECT_EQ(ncclInvalidUsage, Run([&]() {
    return ncclCeAllReduce(comm_.get(), nullptr, nullptr, 4, ncclFloat32, ncclSum, nullptr);
  }, &log));

  EXPECT_TRUE(LogHas(log, "CE AllReduce staging is not available")) << log;
  EXPECT_EQ(0, registration.calls);
}

}  // namespace
