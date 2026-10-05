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
#include "fakes/allocator_fakes.h"
#include "fakes/dev_runtime_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"  // g_loadParam, used by param_redirect.h
#include "fakes/nccl_stubs.h"
#include "fakes/rccl_wrap_fakes.h"

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
  size_t allocSize_ = 0;
  ncclWindow_vidmem devVidmem_{};
  ncclWindow_vidmem hostVidmem_{};
  ncclDevrWindow devrWin_{};
  ncclResult_t registerResult_ = ncclInternalError;
  ncclResult_t toHostResult_ = ncclSuccess;
  int expectedRegistrations_ = 0;  // TearDown checks it, so every launch test also guards against registering
  // Lets a regressed launch-time staging setup reach the window registration the tests count instead of aborting.
  ScopedHook<ncclResult_t(void**, size_t)> alloc_{g_ncclMemAlloc, [this](void** ptr, size_t size) {
    *ptr = &allocated_;
    allocSize_ = size;
    return ncclSuccess;
  }};
  ScopedHook<ncclResult_t(void*)> free_{g_ncclMemFree, [this](void* ptr) {
    EXPECT_EQ(&allocated_, ptr);
    return ncclSuccess;
  }};
  ScopedHook<ncclResult_t(ncclComm*, void*, size_t, int, ncclWindow_t*)> registration_{
      g_devrWindowRegisterInGroup, [this](ncclComm* comm, void* ptr, size_t size, int winFlags, ncclWindow_t* win) {
        EXPECT_EQ(comm_.get(), comm);
        EXPECT_EQ(&allocated_, ptr);
        EXPECT_EQ(allocSize_, size);
        EXPECT_EQ(NCCL_WIN_COLL_SYMMETRIC, winFlags);
        if (registerResult_ == ncclSuccess) {
          *win = &devVidmem_;
        }
        return registerResult_;
      }};
  ScopedHook<ncclResult_t(ncclShadowPool*, void*, void**)> toHost_{
      g_shadowPoolToHost, [this](ncclShadowPool*, void* devObj, void** hostObj) {
        EXPECT_EQ(&devVidmem_, devObj);
        if (toHostResult_ == ncclSuccess) {
          *hostObj = &hostVidmem_;
        }
        return toHostResult_;
      }};

  void SetUp() override {
    ResetRcclWrapFakes();
    comm_ = std::make_unique<ncclComm>();
    comm_->nRanks = 4;
    comm_->nNodes = 1;
    comm_->planner.streams = &streams_;
    comm_->ceColl.ceArStagingBytes = 1024;
    hostVidmem_.winHost = &devrWin_;
    devrWin_.userPtr = &staging_;  // not &allocated_, so a test can tell the published window from the raw buffer
    args_.func = ncclFuncAllReduce;
    args_.datatype = ncclFloat32;
    args_.eltSize = sizeof(float);
    plan_.ceCollArgs = &args_;
  }
  void TearDown() override {
    EXPECT_EQ(expectedRegistrations_, registration_.calls);
    ResetRcclWrapFakes();
  }

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
  std::string log;

  EXPECT_EQ(ncclInvalidUsage, Launch(&log));

  EXPECT_TRUE(LogHas(log, kLaunchNoStagingWarn)) << log;
}

TEST_F(CeCollMicrotest, LaunchNonStagingCollWithoutStaging_SkipsTheStagingCheck) {
  args_.func = ncclFuncBroadcast;  // no CE implementation, so the dispatch default arm rejects it after the check
  uint8_t userRecv = 0;
  args_.useDda = true;  // the DDA copy-back runs only past the staging check
  args_.ddaUserRecvBuff = &userRecv;
  ScopedHook copyBack(g_hipMemcpyAsync,
                      [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipSuccess; });
  std::string log;

  EXPECT_EQ(ncclInvalidUsage, Launch(&log));

  EXPECT_FALSE(LogHas(log, kLaunchNoStagingWarn)) << log;
  EXPECT_EQ(1, copyBack.calls);
}

TEST_F(CeCollMicrotest, LaunchAllReduceWithStaging_ReachesTheCollectiveWithoutRegistering) {
  comm_->ceColl.ceARTmpBuf = &staging_;
  args_.nElts = 0;  // stops ncclCeAllReduce at its layout check, before any copy is issued
  std::string log;

  EXPECT_EQ(ncclInvalidArgument, Launch(&log));

  EXPECT_TRUE(LogHas(log, "CE AllReduce: no valid chunk layout")) << log;
}

// The eager 2-shot path calls ncclCeAllReduce directly, bypassing ncclLaunchCeColl.
TEST_F(CeCollMicrotest, CeAllReduceWithoutStaging_FailsLoudlyWithoutRegistering) {
  std::string log;

  EXPECT_EQ(ncclInvalidUsage, Run([&]() {
    return ncclCeAllReduce(comm_.get(), nullptr, nullptr, 4, ncclFloat32, ncclSum, nullptr);
  }, &log));

  EXPECT_TRUE(LogHas(log, "CE AllReduce staging is not available")) << log;
}

TEST_F(CeCollMicrotest, EnsureStaging_PublishesTheRegisteredWindowOnce) {
  registerResult_ = ncclSuccess;
  expectedRegistrations_ = 1;

  EXPECT_EQ(ncclSuccess, ncclCeEnsureAllReduceStaging(comm_.get()));
  EXPECT_EQ(ncclSuccess, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(1, alloc_.calls);
  EXPECT_EQ(NCCL_CE_NUM_SLOTS * comm_->ceColl.ceArStagingBytes, allocSize_);
  EXPECT_EQ(&devrWin_, comm_->ceColl.ceARTmpWin);
  EXPECT_EQ(&staging_, comm_->ceColl.ceARTmpBuf);
  EXPECT_EQ(0, free_.calls);
}

TEST_F(CeCollMicrotest, EnsureStaging_CeAllReduceDisabled_LeavesStagingUnset) {
  g_rcclParamCeAllReduce = 0;

  EXPECT_EQ(ncclSuccess, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(nullptr, comm_->ceColl.ceARTmpBuf);
  EXPECT_EQ(0, alloc_.calls);
}

TEST_F(CeCollMicrotest, EnsureStaging_RegistrationFails_FreesTheBufferAndReturnsTheError) {
  expectedRegistrations_ = 1;

  EXPECT_EQ(ncclInternalError, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(1, free_.calls);
  EXPECT_EQ(nullptr, comm_->ceColl.ceARTmpBuf);
  EXPECT_EQ(nullptr, comm_->ceColl.ceARTmpWin);
}

TEST_F(CeCollMicrotest, EnsureStaging_ToHostFails_DeregistersAndFreesOnceAndReturnsTheError) {
  registerResult_ = ncclSuccess;
  toHostResult_ = ncclSystemError;
  expectedRegistrations_ = 1;
  ScopedHook deregister(g_devrNcclCommWindowDeregister, [this](ncclComm_t comm, ncclWindow_t win) {
    EXPECT_EQ(comm_.get(), comm);
    EXPECT_EQ(&devVidmem_, win);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSystemError, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(1, deregister.calls);
  EXPECT_EQ(1, free_.calls);
  EXPECT_EQ(nullptr, comm_->ceColl.ceARTmpBuf);
  EXPECT_EQ(nullptr, comm_->ceColl.ceARTmpWin);
}

TEST_F(CeCollMicrotest, EnsureStaging_AlreadySetUp_ReturnsWithoutRegistering) {
  comm_->ceColl.ceARTmpBuf = &allocated_;

  EXPECT_EQ(ncclSuccess, ncclCeEnsureAllReduceStaging(comm_.get()));

  EXPECT_EQ(&allocated_, comm_->ceColl.ceARTmpBuf);
  EXPECT_EQ(0, alloc_.calls);
}

}  // namespace
