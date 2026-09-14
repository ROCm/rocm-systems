/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/rma/rma_ce.cc (AICOMRCCL-2347).
 *************************************************************************/

#include <gtest/gtest.h>

#include <memory>

#include "ScopedHook.h"
#include "fakes/ce_fakes.h"
#include "fakes/dev_runtime_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/rma_fakes.h"

#include "nccl.h"
#include "comm.h"
#include "rma/rma_ce.h"

// rma_ce.cc defines ncclRmaCePutLaunch / ncclRmaCeWaitLaunch, which rma-test.cc
// needs as controllable seams in this same binary (rma.cc dispatches to them).
// Rename the unit's own entry points so both can coexist: rma.cc keeps binding
// to the seams in rma_fakes.cc, and the tests below call the real ones.
// The static helpers (ncclRmaCePutLaunchPersist/NonPersist) are distinct tokens
// and so are untouched.
#define ncclRmaCePutLaunch ncclRmaCePutLaunchUut
#define ncclRmaCeWaitLaunch ncclRmaCeWaitLaunchUut
#include RMA_CE_CC_PATH
#undef ncclRmaCePutLaunch
#undef ncclRmaCeWaitLaunch

namespace {

// Minimal comm for the uninitialised guard: both entry points read
// rmaState.rmaCeState.initialized before anything else. ncclComm is ~3.8 MB, so
// it is heap-allocated rather than held by value.
class RmaCeLaunchTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclKernelPlan> plan_;

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();   // value-initialised, so initialized == false
    plan_ = std::make_unique<ncclKernelPlan>();
  }

  void TearDown() override {
    ResetCeFakes();
    ResetDevRuntimeFakes();
    ResetRmaFakes();
    ResetHipFakes();
  }
};

// Both entry points refuse to touch a communicator whose CE state was never
// brought up, rather than dereferencing it.
TEST_F(RmaCeLaunchTest, PutLaunch_CeNotInitialised_ReturnsInternalError) {
  EXPECT_EQ(ncclRmaCePutLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
}

TEST_F(RmaCeLaunchTest, WaitLaunch_CeNotInitialised_ReturnsInternalError) {
  EXPECT_EQ(ncclRmaCeWaitLaunchUut(comm_.get(), plan_.get(), nullptr), ncclInternalError);
}

}  // namespace
