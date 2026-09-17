/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/rma/rma_ce.cc (AICOMRCCL-2347).
 *************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ce_fakes.h"
#include "fakes/dev_runtime_micro_fakes.h"
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

// Release the windows a test left registered. ncclRmaCeFinalize deregisters a
// window but never frees the host object, and a test whose teardown path fails
// deliberately leaves it registered as well.
//
// dev-runtime-test.cc has a fuller ReclaimDevrWindows, but it is static to that
// TU, walks devrState.winSorted (whose element type is private to
// dev_runtime.cc) and drains memHead through the equally private
// symMemoryDestroy. Every window here comes from the non-symmetric path, so it
// has no backing ncclDevrMemory and memHead stays empty; the windows are
// tracked as they are created rather than recovered from winSorted.
void ReclaimWindow(ncclDevrWindow* w) {
  if (w == nullptr) return;
  free(w->ipcPeerPtrs);
  free(w->ipcPeerPtrsAllocBase);
  free(w);
}

// Minimal comm for the uninitialised guard: both entry points read
// rmaState.rmaCeState.initialized before anything else. ncclComm is ~3.8 MB, so
// it is heap-allocated rather than held by value.
class RmaCeLaunchTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclKernelPlan> plan_;

  void SetUp() override {
    // Reset on entry as well as in TearDown: the seams are process-wide, the
    // fixtures install lambdas capturing `this`, and a test that dies mid-body
    // never reaches its TearDown. Same reason rma-test.cc gives.
    ResetCeFakes();
    ResetDevRuntimeMicroFakes();
    ResetRmaFakes();
    ResetHipFakes();

    comm_ = std::make_unique<ncclComm>();   // value-initialised, so initialized == false
    plan_ = std::make_unique<ncclKernelPlan>();
  }

  void TearDown() override {
    ResetCeFakes();
    ResetDevRuntimeMicroFakes();
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
