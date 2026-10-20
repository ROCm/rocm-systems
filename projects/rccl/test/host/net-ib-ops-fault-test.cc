/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/net_ib_ops_fault.cc (via NET_IB_OPS_FAULT_CC_PATH): the
// ibv_context ops shims, driven through a fake verbs device.

#include <gtest/gtest.h>

#include <cerrno>
#include <string>

#include "../common/LogCapture.hpp"
#include "IbVerbsScene.h"

#include NET_IB_OPS_FAULT_CC_PATH

namespace {

using RcclUnitTesting::CaptureLog;

class NetIbOpsFaultMicrotest : public ::testing::Test {
 protected:
  void TearDown() override { ncclIbOpsFaultRemove(&scene_.ctx); }

  void Install() { ASSERT_EQ(ncclIbOpsFaultInstall(&scene_.ctx), ncclSuccess); }

  int Send(int qp, ibv_send_wr** bad) { return scene_.ctx.ops.post_send(&scene_.qps[qp], &wr_, bad); }
  int Recv(int qp, ibv_recv_wr** bad) { return scene_.ctx.ops.post_recv(&scene_.qps[qp], &rwr_, bad); }
  int Poll(int n = 4) { return scene_.ctx.ops.poll_cq(&scene_.cq, n, wcs_); }
  uint32_t QpNum(int qp) const { return scene_.qps[qp].qp_num; }

  IbVerbsScene scene_;
  ibv_send_wr wr_{};
  ibv_recv_wr rwr_{};
  ibv_wc wcs_[4]{};
};

TEST_F(NetIbOpsFaultMicrotest, NullContext_IsRejectedByEveryEntryPoint) {
  EXPECT_EQ(ncclIbOpsFaultInstall(nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbOpsFaultRemove(nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbOpsFaultArmPostSend(nullptr, 0, EIO), ncclInvalidArgument);
  EXPECT_EQ(ncclIbOpsFaultArmPostRecv(nullptr, 0, EIO), ncclInvalidArgument);
  EXPECT_EQ(ncclIbOpsFaultArmPollCq(nullptr, 0, IBV_WC_GENERAL_ERR, 1, false), ncclInvalidArgument);
  EXPECT_EQ(ncclIbOpsFaultClear(nullptr), ncclInvalidArgument);
}

TEST_F(NetIbOpsFaultMicrotest, Install_SwapsTheThreeOpsAndRemoveRestoresThem) {
  const ibv_context_ops before = scene_.ctx.ops;
  Install();
  EXPECT_NE(scene_.ctx.ops.post_send, before.post_send);
  EXPECT_NE(scene_.ctx.ops.post_recv, before.post_recv);
  EXPECT_NE(scene_.ctx.ops.poll_cq, before.poll_cq);
  EXPECT_EQ(scene_.ctx.ops.modify_qp, before.modify_qp);

  EXPECT_EQ(ncclIbOpsFaultRemove(&scene_.ctx), ncclSuccess);
  EXPECT_EQ(scene_.ctx.ops.post_send, before.post_send);
  EXPECT_EQ(scene_.ctx.ops.post_recv, before.post_recv);
  EXPECT_EQ(scene_.ctx.ops.poll_cq, before.poll_cq);
  EXPECT_EQ(ncclIbOpsFaultRemove(&scene_.ctx), ncclSuccess);
}

// A second install must not save the shims as the "real" ops, or Remove would leave the shims behind.
TEST_F(NetIbOpsFaultMicrotest, Install_Twice_StillRestoresTheOriginalOps) {
  const ibv_context_ops before = scene_.ctx.ops;
  Install();
  const ibv_context_ops shimmed = scene_.ctx.ops;
  Install();
  EXPECT_EQ(scene_.ctx.ops.post_send, shimmed.post_send);
  ASSERT_EQ(ncclIbOpsFaultRemove(&scene_.ctx), ncclSuccess);
  EXPECT_EQ(scene_.ctx.ops.post_send, before.post_send);
  EXPECT_EQ(scene_.ctx.ops.post_recv, before.post_recv);
  EXPECT_EQ(scene_.ctx.ops.poll_cq, before.poll_cq);
}

TEST_F(NetIbOpsFaultMicrotest, Install_NullOp_LeavesTheContextUnregistered) {
  for (int which = 0; which < 3; which++) {
    IbVerbsScene scene;
    if (which == 0) scene.ctx.ops.post_send = nullptr;
    if (which == 1) scene.ctx.ops.post_recv = nullptr;
    if (which == 2) scene.ctx.ops.poll_cq = nullptr;
    const ibv_context_ops before = scene.ctx.ops;
    ncclResult_t res = ncclInternalError;
    const std::string log = CaptureLog([&] { res = ncclIbOpsFaultInstall(&scene.ctx); });
    EXPECT_EQ(res, ncclSuccess) << which;
    EXPECT_NE(log.find("has NULL post_send/post_recv/poll_cq op; skipping install"), std::string::npos) << which;
    EXPECT_EQ(scene.ctx.ops.post_send, before.post_send) << which;
    EXPECT_EQ(scene.ctx.ops.post_recv, before.post_recv) << which;
    EXPECT_EQ(scene.ctx.ops.poll_cq, before.poll_cq) << which;
    EXPECT_EQ(ncclIbOpsFaultArmPostSend(&scene.ctx, 0, EIO), ncclInvalidArgument) << which;
  }
}

TEST_F(NetIbOpsFaultMicrotest, Arm_NotInstalled_IsRejectedButClearAndRemoveAreNoOps) {
  EXPECT_EQ(ncclIbOpsFaultArmPostSend(&scene_.ctx, 0, EIO), ncclInvalidArgument);
  EXPECT_EQ(ncclIbOpsFaultArmPostRecv(&scene_.ctx, 0, EIO), ncclInvalidArgument);
  EXPECT_EQ(ncclIbOpsFaultArmPollCq(&scene_.ctx, 0, IBV_WC_GENERAL_ERR, 1, true), ncclInvalidArgument);
  EXPECT_EQ(ncclIbOpsFaultClear(&scene_.ctx), ncclSuccess);
  EXPECT_EQ(ncclIbOpsFaultRemove(&scene_.ctx), ncclSuccess);
}

TEST_F(NetIbOpsFaultMicrotest, PostSend_Unarmed_ForwardsAndReturnsTheRealResult) {
  Install();
  wr_.wr_id = 0x42;
  ibv_send_wr* bad = nullptr;
  EXPECT_EQ(Send(0, &bad), 0);
  ASSERT_EQ(scene_.sends.size(), 1u);
  EXPECT_EQ(scene_.sends[0].qp, &scene_.qps[0]);
  EXPECT_EQ(scene_.sends[0].chain[0].wr.wr_id, 0x42u);

  scene_.postSendErr = ENOMEM;
  EXPECT_EQ(Send(0, &bad), ENOMEM);
  EXPECT_EQ(bad, &wr_);
}

TEST_F(NetIbOpsFaultMicrotest, PostSend_ArmedQp_FailsWithoutPostingAndOthersForward) {
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPostSend(&scene_.ctx, QpNum(1), EAGAIN), ncclSuccess);
  ibv_send_wr* bad = nullptr;
  EXPECT_EQ(Send(1, &bad), EAGAIN);
  EXPECT_EQ(bad, &wr_);
  EXPECT_EQ(Send(1, nullptr), EAGAIN);
  EXPECT_TRUE(scene_.sends.empty());

  EXPECT_EQ(Send(0, &bad), 0);
  EXPECT_EQ(scene_.sends.size(), 1u);

  ASSERT_EQ(ncclIbOpsFaultArmPostSend(&scene_.ctx, QpNum(1), 0), ncclSuccess);
  EXPECT_EQ(Send(1, &bad), 0);
  EXPECT_EQ(scene_.sends.size(), 2u);
}

// findFault prefers an exact qp_num entry, so a QP disarmed explicitly is exempt from a context-wide fault.
TEST_F(NetIbOpsFaultMicrotest, PostSend_ExactQpEntryOverridesAny) {
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPostSend(&scene_.ctx, NCCL_IB_OPS_FAULT_QP_ANY, EIO), ncclSuccess);
  ASSERT_EQ(ncclIbOpsFaultArmPostSend(&scene_.ctx, QpNum(2), 0), ncclSuccess);
  ibv_send_wr* bad = nullptr;
  EXPECT_EQ(Send(0, &bad), EIO);
  EXPECT_EQ(Send(3, &bad), EIO);
  EXPECT_EQ(Send(2, &bad), 0);
  EXPECT_EQ(scene_.sends.size(), 1u);
}

TEST_F(NetIbOpsFaultMicrotest, PostRecv_ArmedQp_FailsWithoutPostingAndOthersForward) {
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPostRecv(&scene_.ctx, QpNum(0), ENOMEM), ncclSuccess);
  rwr_.wr_id = 9;
  ibv_recv_wr* bad = nullptr;
  EXPECT_EQ(Recv(0, &bad), ENOMEM);
  EXPECT_EQ(bad, &rwr_);
  EXPECT_EQ(Recv(0, nullptr), ENOMEM);
  EXPECT_TRUE(scene_.recvs.empty());

  EXPECT_EQ(Recv(1, &bad), 0);
  ASSERT_EQ(scene_.recvs.size(), 1u);
  EXPECT_EQ(scene_.recvs[0].qp, &scene_.qps[1]);
  EXPECT_EQ(scene_.recvs[0].wrId, 9u);

  scene_.postRecvErr = EINVAL;
  EXPECT_EQ(Recv(1, &bad), EINVAL);
}

TEST_F(NetIbOpsFaultMicrotest, PostRecv_AnyArmsEveryQpAndLeavesSendAlone) {
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPostRecv(&scene_.ctx, NCCL_IB_OPS_FAULT_QP_ANY, EAGAIN), ncclSuccess);
  ibv_recv_wr* rbad = nullptr;
  ibv_send_wr* sbad = nullptr;
  EXPECT_EQ(Recv(0, &rbad), EAGAIN);
  EXPECT_EQ(Recv(5, &rbad), EAGAIN);
  EXPECT_EQ(Send(5, &sbad), 0);
}

// A shim reached through a context that is not registered has no real op to call.
TEST_F(NetIbOpsFaultMicrotest, Shims_UnregisteredContext_FailInsteadOfCrashing) {
  Install();
  IbVerbsScene other("mlx5_1");
  other.ctx.ops = scene_.ctx.ops;
  ibv_send_wr* sbad = nullptr;
  ibv_recv_wr* rbad = nullptr;
  EXPECT_EQ(other.ctx.ops.post_send(&other.qps[0], &wr_, &sbad), EFAULT);
  EXPECT_EQ(sbad, &wr_);
  EXPECT_EQ(other.ctx.ops.post_send(&other.qps[0], &wr_, nullptr), EFAULT);
  EXPECT_EQ(other.ctx.ops.post_recv(&other.qps[0], &rwr_, &rbad), EFAULT);
  EXPECT_EQ(rbad, &rwr_);
  EXPECT_EQ(other.ctx.ops.post_recv(&other.qps[0], &rwr_, nullptr), EFAULT);
  EXPECT_EQ(other.ctx.ops.poll_cq(&other.cq, 4, wcs_), -1);
  EXPECT_TRUE(scene_.sends.empty() && other.sends.empty() && scene_.recvs.empty() && other.recvs.empty());
}

TEST_F(NetIbOpsFaultMicrotest, PollCq_Unarmed_ReturnsTheRealCompletions) {
  Install();
  scene_.Complete(7, IBV_WC_RDMA_WRITE, QpNum(0));
  ASSERT_EQ(Poll(), 1);
  EXPECT_EQ(wcs_[0].wr_id, 7u);
  EXPECT_EQ(wcs_[0].status, IBV_WC_SUCCESS);
  EXPECT_EQ(Poll(), 0);

  scene_.pollErr = -5;
  EXPECT_EQ(Poll(), -5);
}

TEST_F(NetIbOpsFaultMicrotest, PollCq_ArmedQp_RewritesOnlyItsCompletions) {
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPollCq(&scene_.ctx, QpNum(0), IBV_WC_RETRY_EXC_ERR, -1, false), ncclSuccess);
  scene_.Complete(1, IBV_WC_RDMA_WRITE, QpNum(0));
  scene_.Complete(2, IBV_WC_RDMA_WRITE, QpNum(1));
  scene_.Complete(3, IBV_WC_RECV_RDMA_WITH_IMM, QpNum(0));
  ASSERT_EQ(Poll(), 3);
  EXPECT_EQ(wcs_[0].status, IBV_WC_RETRY_EXC_ERR);
  EXPECT_EQ(wcs_[0].wr_id, 1u);
  EXPECT_EQ(wcs_[1].status, IBV_WC_SUCCESS);
  EXPECT_EQ(wcs_[2].status, IBV_WC_RETRY_EXC_ERR);
  EXPECT_EQ(wcs_[2].opcode, IBV_WC_RECV_RDMA_WITH_IMM);
}

TEST_F(NetIbOpsFaultMicrotest, PollCq_InjectCount_CorruptsExactlyThatMany) {
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPollCq(&scene_.ctx, QpNum(0), IBV_WC_WR_FLUSH_ERR, 2, false), ncclSuccess);
  for (int i = 0; i < 3; i++) scene_.Complete(i, IBV_WC_RDMA_WRITE, QpNum(0));
  ASSERT_EQ(Poll(), 3);
  EXPECT_EQ(wcs_[0].status, IBV_WC_WR_FLUSH_ERR);
  EXPECT_EQ(wcs_[1].status, IBV_WC_WR_FLUSH_ERR);
  EXPECT_EQ(wcs_[2].status, IBV_WC_SUCCESS);
  scene_.Complete(9, IBV_WC_RDMA_WRITE, QpNum(0));
  ASSERT_EQ(Poll(), 1);
  EXPECT_EQ(wcs_[0].status, IBV_WC_SUCCESS);
}

TEST_F(NetIbOpsFaultMicrotest, PollCq_SuccessStatus_DisarmsTheRewrite) {
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPollCq(&scene_.ctx, QpNum(0), IBV_WC_SUCCESS, -1, true), ncclSuccess);
  scene_.Complete(1, IBV_WC_RDMA_WRITE, QpNum(0));
  ASSERT_EQ(Poll(), 1);
  EXPECT_EQ(wcs_[0].status, IBV_WC_SUCCESS);
  EXPECT_EQ(Poll(), 0);
}

TEST_F(NetIbOpsFaultMicrotest, PollCq_IdleInjection_SynthesizesOneErrorCompletion) {
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPollCq(&scene_.ctx, QpNum(3), IBV_WC_WR_FLUSH_ERR, 1, true), ncclSuccess);
  wcs_[0].byte_len = 0xdead;
  wcs_[0].opcode = IBV_WC_RDMA_READ;
  ASSERT_EQ(Poll(), 1);
  EXPECT_EQ(wcs_[0].status, IBV_WC_WR_FLUSH_ERR);
  EXPECT_EQ(wcs_[0].qp_num, QpNum(3));
  EXPECT_EQ(wcs_[0].wr_id, NCCL_IB_OPS_FAULT_SYNTH_WR_ID);
  EXPECT_EQ(wcs_[0].byte_len, 0u);
  EXPECT_EQ(wcs_[0].opcode, static_cast<ibv_wc_opcode>(0));
  EXPECT_EQ(Poll(), 0);
}

TEST_F(NetIbOpsFaultMicrotest, PollCq_IdleInjection_UnlimitedOnAnyReportsQpNumZero) {
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPollCq(&scene_.ctx, NCCL_IB_OPS_FAULT_QP_ANY, IBV_WC_GENERAL_ERR, -1, true),
            ncclSuccess);
  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(Poll(), 1) << i;
    EXPECT_EQ(wcs_[0].qp_num, 0u);
    EXPECT_EQ(wcs_[0].status, IBV_WC_GENERAL_ERR);
  }
}

TEST_F(NetIbOpsFaultMicrotest, PollCq_IdleInjection_NeedsTheFlagAnEntryAndAnIdleQueue) {
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPollCq(&scene_.ctx, QpNum(0), IBV_WC_GENERAL_ERR, -1, false), ncclSuccess);
  EXPECT_EQ(Poll(), 0);
  ASSERT_EQ(ncclIbOpsFaultArmPollCq(&scene_.ctx, QpNum(0), IBV_WC_GENERAL_ERR, 0, true), ncclSuccess);
  EXPECT_EQ(Poll(), 0);
  ASSERT_EQ(ncclIbOpsFaultArmPollCq(&scene_.ctx, QpNum(0), IBV_WC_GENERAL_ERR, -1, true), ncclSuccess);
  EXPECT_EQ(Poll(0), 0);
  // A real completion on another QP means the queue was not idle: nothing is synthesized.
  scene_.Complete(5, IBV_WC_RDMA_WRITE, QpNum(1));
  ASSERT_EQ(Poll(), 1);
  EXPECT_EQ(wcs_[0].wr_id, 5u);
  EXPECT_EQ(wcs_[0].status, IBV_WC_SUCCESS);
  scene_.pollErr = -1;
  EXPECT_EQ(Poll(), -1);
}

TEST_F(NetIbOpsFaultMicrotest, Clear_DisarmsEveryFaultButKeepsTheShims) {
  const ibv_context_ops before = scene_.ctx.ops;
  Install();
  ASSERT_EQ(ncclIbOpsFaultArmPostSend(&scene_.ctx, NCCL_IB_OPS_FAULT_QP_ANY, EIO), ncclSuccess);
  ASSERT_EQ(ncclIbOpsFaultArmPostRecv(&scene_.ctx, QpNum(0), EIO), ncclSuccess);
  ASSERT_EQ(ncclIbOpsFaultArmPollCq(&scene_.ctx, QpNum(0), IBV_WC_GENERAL_ERR, -1, true), ncclSuccess);
  ASSERT_EQ(ncclIbOpsFaultClear(&scene_.ctx), ncclSuccess);

  EXPECT_NE(scene_.ctx.ops.post_send, before.post_send);
  ibv_send_wr* sbad = nullptr;
  ibv_recv_wr* rbad = nullptr;
  EXPECT_EQ(Send(0, &sbad), 0);
  EXPECT_EQ(Recv(0, &rbad), 0);
  EXPECT_EQ(Poll(), 0);
  EXPECT_EQ(scene_.sends.size(), 1u);
  EXPECT_EQ(scene_.recvs.size(), 1u);
}

// Contexts are registered independently: arming or removing one leaves the other's faults in place.
TEST_F(NetIbOpsFaultMicrotest, Registry_IsKeyedPerContext) {
  IbVerbsScene other("mlx5_1");
  Install();
  ASSERT_EQ(ncclIbOpsFaultInstall(&other.ctx), ncclSuccess);
  ASSERT_EQ(ncclIbOpsFaultArmPostSend(&other.ctx, NCCL_IB_OPS_FAULT_QP_ANY, EIO), ncclSuccess);
  ibv_send_wr* bad = nullptr;
  EXPECT_EQ(Send(0, &bad), 0);
  EXPECT_EQ(other.ctx.ops.post_send(&other.qps[0], &wr_, &bad), EIO);
  ASSERT_EQ(ncclIbOpsFaultRemove(&scene_.ctx), ncclSuccess);
  EXPECT_EQ(other.ctx.ops.post_send(&other.qps[0], &wr_, &bad), EIO);
  ASSERT_EQ(ncclIbOpsFaultRemove(&other.ctx), ncclSuccess);
  EXPECT_EQ(other.ctx.ops.post_send(&other.qps[0], &wr_, &bad), 0);
}

}  // namespace
