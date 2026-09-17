/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/rma/rma_proxy_launch.cc.
 *************************************************************************/

#include <gtest/gtest.h>

#include <memory>

#include "nccl.h"
#include "comm.h"
#include "rma/rma_proxy.h"

// Other units in rccl-UnitTestsMicro need controllable doubles for these
// exported functions. Rename this translation unit's definitions at inclusion
// time so those doubles keep serving their existing callers while the tests
// below exercise the production implementation directly.
#define ncclCuStreamBatchMemOp ncclCuStreamBatchMemOpUut
#define ncclRmaProxyCircularBufEmpty ncclRmaProxyCircularBufEmptyUut
#define ncclRmaProxyDestroyDesc ncclRmaProxyDestroyDescUut
#define ncclRmaProxyPutLaunch ncclRmaProxyPutLaunchUut
#define ncclRmaProxyWaitLaunch ncclRmaProxyWaitLaunchUut
#define ncclRmaProxyReclaimPlan ncclRmaProxyReclaimPlanUut
#include RMA_PROXY_LAUNCH_CC_PATH
#undef ncclRmaProxyReclaimPlan
#undef ncclRmaProxyWaitLaunch
#undef ncclRmaProxyPutLaunch
#undef ncclRmaProxyDestroyDesc
#undef ncclRmaProxyCircularBufEmpty
#undef ncclCuStreamBatchMemOp

namespace {

// The connection guard is the first branch of both launch entry points and is
// the smallest proof that the host-only harness reaches the production unit.
class RmaProxyLaunchTest : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> comm_;

  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    comm_->rmaState.rmaProxyState.connected = false;
  }
};

TEST_F(RmaProxyLaunchTest, PutLaunch_DisconnectedProxyIsRejectedBeforeReadingThePlan) {
  EXPECT_EQ(ncclInternalError,
            ncclRmaProxyPutLaunchUut(comm_.get(), nullptr, nullptr));
}

TEST_F(RmaProxyLaunchTest, WaitLaunch_DisconnectedProxyIsRejectedBeforeReadingThePlan) {
  EXPECT_EQ(ncclInternalError,
            ncclRmaProxyWaitLaunchUut(comm_.get(), nullptr, nullptr));
}

}  // namespace
