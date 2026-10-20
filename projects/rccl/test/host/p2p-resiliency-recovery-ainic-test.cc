/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/p2p_resiliency_recovery_ainic.cc (via
// P2P_RESILIENCY_RECOVERY_AINIC_CC_PATH): destroy+recreate recovery and the QPN exchange.

#include <gtest/gtest.h>

#include <cerrno>
#include <cstring>
#include <memory>
#include <vector>

#include "P2pResiliencyScene.h"
#include "fakes/param_redirect.h"

#include P2P_RESILIENCY_RECOVERY_AINIC_CC_PATH

namespace {

class P2pResiliencyRecoveryAinicMicrotest : public ::testing::Test {
 protected:
  P2pResiliencyScene scene_;
  P2pResiliencyScene::CommPtr<ncclIbSendComm> send_;
  P2pResiliencyScene::CommPtr<ncclIbRecvComm> recv_;
  std::unique_ptr<ncclIbResiliency> res_ = std::make_unique<ncclIbResiliency>();
  ncclIbPortRecoveryContext ctx_ = {};
  std::vector<ncclIbQpCreateAttr> creates_;

  // QpCreate clears telQpStats in production; the hook does too, so restoring it is observable.
  ScopedHook<ncclResult_t(ncclIbQp*, ncclIbQpCreateAttr*)> create_{
      g_ibCastQpCreate, [this](ncclIbQp* qp, ncclIbQpCreateAttr* attr) {
        qp->telQpStats = nullptr;
        qp->qp = scene_.NewQp();
        creates_.push_back(*attr);
        return ncclSuccess;
      }};

  void SetUp() override { IbCastAinicRoce = true; }

  ncclIbNetCommBase* Make(bool isSend, int ndevs, int nqps, int dev) {
    ncclIbNetCommBase* base;
    if (isSend) {
      send_ = scene_.NewComm<ncclIbSendComm>(true, ndevs, nqps);
      base = &send_->base;
    } else {
      recv_ = scene_.NewComm<ncclIbRecvComm>(false, ndevs, nqps);
      base = &recv_->base;
    }
    res_->baseComm = base;
    res_->ndevs = ndevs;
    ctx_.resCtx = res_.get();
    ctx_.devIndex = dev;
    return base;
  }

  // Receiver whose device dev has a flush QP.
  ncclIbQp* WithFlushQp(int dev) {
    recv_->flushEnabled = 1;
    ncclIbQp* flush = &recv_->devs[dev].gpuFlush.qp;
    flush->qp = scene_.NewQp();
    return flush;
  }
};

TEST_F(P2pResiliencyRecoveryAinicMicrotest, RestoreAinic_RecreatesEachQpOfTheDeviceOnItsOldPipeline) {
  ncclIbNetCommBase* base = Make(true, 2, 4, 1);
  RcclQpStats* stats = reinterpret_cast<RcclQpStats*>(0x1230);
  base->qps[1].channelId = 5;
  base->qps[1].isDataQp = true;
  base->qps[1].telQpStats = stats;
  base->qps[3].channelId = 6;
  ibv_qp* old1 = base->qps[1].qp;
  ibv_qp* old3 = base->qps[3].qp;
  ibv_qp* keep0 = base->qps[0].qp;
  ASSERT_EQ(IbCastPortRecoveryQpsRestoreAinic(&ctx_), ncclSuccess);
  EXPECT_EQ(scene_.destroyedQps, (std::vector<ibv_qp*>{old1, old3}));
  EXPECT_EQ(base->qps[0].qp, keep0);
  ASSERT_EQ(scene_.buildAttrCalls.size(), 2u);
  EXPECT_EQ(scene_.buildAttrCalls[0], std::make_pair(base, 1));
  ASSERT_EQ(creates_.size(), 2u);
  EXPECT_EQ(creates_[0].channelId, 5);
  EXPECT_TRUE(creates_[0].isDataQp);
  EXPECT_EQ(creates_[1].channelId, 6);
  EXPECT_FALSE(creates_[1].isDataQp);
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(creates_[i].maxSendWorkRequest, 77u);  // the rest comes from IbCastBuildDataQpCreateAttr
    EXPECT_FALSE(creates_[i].isQpSharingEnabled);
    EXPECT_EQ(creates_[i].qpSharingGroupIdx, -1);
    EXPECT_EQ(creates_[i].cqDepthMultiplier, 1);
    EXPECT_EQ(creates_[i].qpIdx, i);  // QP 1 is the device's first QP, QP 3 its second
  }
  EXPECT_EQ(base->qps[1].telQpStats, stats);
  ASSERT_EQ(scene_.transitions.size(), 2u);
  EXPECT_EQ(scene_.transitions[0].step, "INIT");
  EXPECT_EQ(scene_.transitions[1].qp, &base->qps[3]);
  ASSERT_EQ(ctx_.nLocalQpns, 2);
  EXPECT_EQ(ctx_.localQpns[0].qpIndex, 1u);
  EXPECT_EQ(ctx_.localQpns[0].newQpn, base->qps[1].qp->qp_num);
  EXPECT_EQ(ctx_.localQpns[1].qpIndex, 3u);
  EXPECT_EQ(ctx_.localQpns[1].newQpn, base->qps[3].qp->qp_num);
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, RestoreAinic_Receiver_AlsoRecreatesTheDevicesFlushQp) {
  ncclIbNetCommBase* base = Make(false, 2, 2, 1);
  ncclIbQp* flush = WithFlushQp(1);
  ncclIbQp* otherFlush = &recv_->devs[0].gpuFlush.qp;
  otherFlush->qp = scene_.NewQp();
  ibv_qp* otherFlushQp = otherFlush->qp;
  flush->channelId = 9;
  ibv_qp* oldData = base->qps[1].qp;
  ibv_qp* oldFlush = flush->qp;
  ASSERT_EQ(IbCastPortRecoveryQpsRestoreAinic(&ctx_), ncclSuccess);
  ASSERT_EQ(creates_.size(), 2u);
  const ncclIbQpCreateAttr& attr = creates_[1];
  EXPECT_EQ(attr.maxRecvWorkRequest, 0u);
  EXPECT_EQ(attr.maxSendWorkRequest, (uint32_t)NET_IB_MAX_REQUESTS);
  EXPECT_EQ(attr.isP2p, 0);
  EXPECT_EQ(attr.channelId, 9);
  EXPECT_EQ(attr.qpSharingGroupIdx, -1);
  EXPECT_EQ(scene_.destroyedQps, (std::vector<ibv_qp*>{oldData, oldFlush}));
  EXPECT_NE(flush->qp, oldFlush);
  EXPECT_EQ(otherFlush->qp, otherFlushQp);
  ASSERT_EQ(scene_.transitions.size(), 2u);
  EXPECT_EQ(scene_.transitions[1].qp, flush);
  EXPECT_EQ(ctx_.nLocalQpns, 1);  // the flush QP talks to itself: its QPN is not exchanged
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, RestoreAinic_FlushDisabled_LeavesFlushQpAlone) {
  Make(false, 1, 1, 0);
  ncclIbQp* flush = WithFlushQp(0);
  recv_->flushEnabled = 0;
  ibv_qp* qp = flush->qp;
  ASSERT_EQ(IbCastPortRecoveryQpsRestoreAinic(&ctx_), ncclSuccess);
  EXPECT_EQ(flush->qp, qp);
  EXPECT_EQ(creates_.size(), 1u);
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, RestoreAinic_StepFails_ReturnsTheError) {
  // Data QP destroy, create, init; then the same three for the flush QP.
  for (int failAt = 0; failAt < 6; failAt++) {
    Make(false, 1, 1, 0);
    WithFlushQp(0);
    int destroys = 0, creates = 0, inits = 0;
    ScopedHook destroy(g_ibvDestroyQp, [&](ibv_qp*) {
      return (failAt == 0 && destroys++ == 0) || (failAt == 3 && destroys++ == 1) ? ncclSystemError : ncclSuccess;
    });
    ScopedHook create(g_ibCastQpCreate, [&](ncclIbQp* qp, ncclIbQpCreateAttr*) {
      qp->qp = scene_.NewQp();
      int n = creates++;
      return (failAt == 1 && n == 0) || (failAt == 4 && n == 1) ? ncclSystemError : ncclSuccess;
    });
    ScopedHook init(g_ibCastQpInit, [&](ncclIbQp*) {
      int n = inits++;
      return (failAt == 2 && n == 0) || (failAt == 5 && n == 1) ? ncclSystemError : ncclSuccess;
    });
    EXPECT_EQ(IbCastPortRecoveryQpsRestoreAinic(&ctx_), ncclSystemError) << failAt;
  }
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, QpsToRtsAinic_NotAinic_DoesNothing) {
  Make(true, 1, 1, 0);
  IbCastAinicRoce = false;
  ScopedHook devBase(g_ibCastGetNetCommDevBase, P2pResiliencyNoDevBase);
  EXPECT_EQ(IbCastPortRecoveryQpsToRtsAinic(&ctx_), ncclSuccess);
  EXPECT_TRUE(scene_.transitions.empty());
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, QpsToRtsAinic_NoDeviceBase_ReturnsInternalError) {
  Make(true, 1, 1, 0);
  ScopedHook devBase(g_ibCastGetNetCommDevBase, P2pResiliencyNoDevBase);
  EXPECT_EQ(IbCastPortRecoveryQpsToRtsAinic(&ctx_), ncclInternalError);
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, QpsToRtsAinic_Sender_ConnectsEachQpToThePeersNewQpn) {
  ncclIbNetCommBase* base = Make(true, 2, 4, 1);
  base->qps[3].eceSupported = 1;
  send_->devs[1].base.gidInfo.localGidIndex = 31;
  send_->devs[1].base.gidInfo.localGid.global.interface_id = 0x3131;
  ctx_.remoteQpns[0] = {3, 0x803};  // looked up by QP index, not by position
  ctx_.remoteQpns[1] = {1, 0x801};
  ctx_.nRemoteQpns = 2;
  ASSERT_EQ(IbCastPortRecoveryQpsToRtsAinic(&ctx_), ncclSuccess);
  EXPECT_EQ(base->qps[1].rtrAttr.remoteQpNum, 0x801u);
  EXPECT_EQ(base->qps[3].rtrAttr.remoteQpNum, 0x803u);
  EXPECT_EQ(base->qps[0].rtrAttr.remoteQpNum, 0u);
  EXPECT_EQ(base->qps[1].rtrAttr.localGidIndex, 31);
  EXPECT_EQ(base->qps[1].rtrAttr.localGid.global.interface_id, 0x3131u);
  EXPECT_EQ(scene_.eceSets, (std::vector<ibv_qp*>{base->qps[3].qp}));
  ASSERT_EQ(scene_.transitions.size(), 4u);
  EXPECT_EQ(scene_.transitions[0].step, "RTR");
  EXPECT_EQ(scene_.transitions[0].qp, &base->qps[1]);
  EXPECT_EQ(scene_.transitions[1].step, "RTS");
  EXPECT_TRUE(scene_.postedRecvWrQps.empty());
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, QpsToRtsAinic_MissingRemoteQpn_ReturnsInternalError) {
  Make(true, 1, 2, 0);
  ctx_.remoteQpns[0] = {0, 0x800};
  ctx_.remoteQpns[1] = {1, 0x801};
  ctx_.nRemoteQpns = 1;  // the entry for QP 1 is not part of the ACK
  EXPECT_EQ(IbCastPortRecoveryQpsToRtsAinic(&ctx_), ncclInternalError);
  EXPECT_EQ(scene_.transitions.size(), 2u);  // QP 0 got connected first
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, QpsToRtsAinic_Receiver_PostsReceivesAndLoopsTheFlushQpBack) {
  ncclIbNetCommBase* base = Make(false, 1, 2, 0);
  ncclIbQp* flush = WithFlushQp(0);
  recv_->devs[0].base.gidInfo.localGidIndex = 12;
  ctx_.remoteQpns[0] = {0, 0x700};
  ctx_.remoteQpns[1] = {1, 0x701};
  ctx_.nRemoteQpns = 2;
  ASSERT_EQ(IbCastPortRecoveryQpsToRtsAinic(&ctx_), ncclSuccess);
  EXPECT_EQ(scene_.postedRecvWrQps, (std::vector<ncclIbQp*>{&base->qps[0], &base->qps[1]}));
  EXPECT_EQ(flush->rtrAttr.remoteQpNum, flush->qp->qp_num);
  EXPECT_EQ(flush->rtrAttr.localGidIndex, 12);
  EXPECT_EQ(flush->rtrAttr.remoteGid.global.interface_id, recv_->devs[0].base.gidInfo.localGid.global.interface_id);
  EXPECT_EQ(flush->rtrAttr.localGid.global.interface_id, recv_->devs[0].base.gidInfo.localGid.global.interface_id);
  ASSERT_EQ(scene_.transitions.size(), 6u);
  EXPECT_EQ(scene_.transitions[4].qp, flush);
  EXPECT_EQ(scene_.transitions[5].step, "RTS");
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, QpsToRtsAinic_StepFails_ReturnsTheError) {
  // ECE, RTR, RTS, receive posting; then RTR and RTS of the flush QP.
  for (int failAt = 0; failAt < 6; failAt++) {
    ncclIbNetCommBase* base = Make(false, 1, 1, 0);
    WithFlushQp(0);
    base->qps[0].eceSupported = 1;
    ctx_.remoteQpns[0] = {0, 0x700};
    ctx_.nRemoteQpns = 1;
    int rtrs = 0, rtss = 0;
    ScopedHook ece(g_ibvSetEce, [&](ibv_qp*, ibv_ece*, int*) { return failAt == 0 ? ncclSystemError : ncclSuccess; });
    ScopedHook rtr(g_ibCastQpRtr, [&](ncclIbQp*) {
      int n = rtrs++;
      return (failAt == 1 && n == 0) || (failAt == 4 && n == 1) ? ncclSystemError : ncclSuccess;
    });
    ScopedHook rts(g_ibCastQpRts, [&](ncclIbQp*) {
      int n = rtss++;
      return (failAt == 2 && n == 0) || (failAt == 5 && n == 1) ? ncclSystemError : ncclSuccess;
    });
    ScopedHook post(g_ibCastPostReceiveWorkRequestsOnQp,
                    [&](ncclIbRecvComm*, ncclIbQp*) { return failAt == 3 ? ncclSystemError : ncclSuccess; });
    EXPECT_EQ(IbCastPortRecoveryQpsToRtsAinic(&ctx_), ncclSystemError) << failAt;
  }
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, PostAckWithQpns_SendsTheLocalQpnsAsPayload) {
  Make(true, 2, 2, 1);
  res_->devs[1].portRecoveryQpnMr = scene_.NewMr(0x77);
  res_->portRecoveryQps[1].qp = scene_.NewQp();
  res_->portRecoveryAh[1] = scene_.NewAh();
  res_->portRecoveryRemoteQpn[1] = 0x655;
  ctx_.aliveMsgNextId = 12;
  ctx_.localQpns[0] = {1, 0x1111};
  ctx_.localQpns[1] = {3, 0x3333};
  ctx_.nLocalQpns = 2;
  bool ok = false;
  ASSERT_EQ(IbCastPortRecoveryPostAckWithQpnsAinic(&ctx_, &ok), ncclSuccess);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(ctx_.ackPosted);
  EXPECT_EQ(std::memcmp(res_->portRecoveryQpnBuf, ctx_.localQpns, 2 * sizeof(RecoveryQpnEntry)), 0);
  ASSERT_EQ(scene_.sends.size(), 1u);
  const auto& ack = scene_.sends[0];
  EXPECT_EQ(ack.qp, res_->portRecoveryQps[1].qp);
  EXPECT_EQ(ack.wr.wr_id, 12u);
  EXPECT_EQ(ack.wr.opcode, IBV_WR_SEND_WITH_IMM);
  EXPECT_EQ(ack.wr.imm_data, (uint32_t)NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
  EXPECT_EQ(ack.wr.send_flags, IBV_SEND_SIGNALED);
  EXPECT_EQ(ack.wr.wr.ud.ah, res_->portRecoveryAh[1]);
  EXPECT_EQ(ack.wr.wr.ud.remote_qpn, 0x655u);
  EXPECT_EQ(ack.wr.wr.ud.remote_qkey, (uint32_t)NCCL_IB_RECOVERY_QKEY);
  ASSERT_EQ(ack.sges.size(), 1u);
  EXPECT_EQ(ack.sges[0].addr, (uint64_t)res_->portRecoveryQpnBuf);
  EXPECT_EQ(ack.sges[0].length, 2 * sizeof(RecoveryQpnEntry));
  EXPECT_EQ(ack.sges[0].lkey, 0x77u);
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, PostAckWithQpns_TooManyEntriesOrPostFails_IsReported) {
  Make(true, 1, 1, 0);
  res_->devs[0].portRecoveryQpnMr = scene_.NewMr(1);
  res_->portRecoveryQps[0].qp = scene_.NewQp();
  bool ok = true;
  ctx_.nLocalQpns = NCCL_IB_MAX_RECOVERY_QPN_ENTRIES + 1;
  EXPECT_EQ(IbCastPortRecoveryPostAckWithQpnsAinic(&ctx_, &ok), ncclInternalError);
  EXPECT_TRUE(scene_.sends.empty());
  ctx_.nLocalQpns = NCCL_IB_MAX_RECOVERY_QPN_ENTRIES;
  ASSERT_EQ(IbCastPortRecoveryPostAckWithQpnsAinic(&ctx_, &ok), ncclSuccess);
  EXPECT_TRUE(ok);
  ctx_.ackPosted = false;
  scene_.postSend = [](ibv_qp*, ibv_send_wr*) { return EIO; };
  ASSERT_EQ(IbCastPortRecoveryPostAckWithQpnsAinic(&ctx_, &ok), ncclSuccess);
  EXPECT_FALSE(ok);
  EXPECT_FALSE(ctx_.ackPosted);
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, ExtractQpns_ReadsTheEntriesAfterTheGrh) {
  Make(false, 1, 1, 0);
  uint8_t* payload = res_->devs[0].portRecoveryGrhBuf + NCCL_IB_UD_GRH_SIZE;
  RecoveryQpnEntry entries[3] = {{0, 0x10}, {2, 0x12}, {4, 0x14}};
  std::memcpy(payload, entries, sizeof(entries));
  ibv_wc wc = P2pResiliencyWc(0, IBV_WC_SUCCESS, IBV_WC_RECV, 0, NCCL_IB_UD_GRH_SIZE + sizeof(entries) + 3);
  IbCastPortRecoveryExtractQpnsAinic(&ctx_, &wc);
  ASSERT_EQ(ctx_.nRemoteQpns, 3);  // a trailing partial entry is dropped
  EXPECT_EQ(ctx_.remoteQpns[2].qpIndex, 4u);
  EXPECT_EQ(ctx_.remoteQpns[2].newQpn, 0x14u);
  wc.byte_len = NCCL_IB_UD_GRH_SIZE;
  IbCastPortRecoveryExtractQpnsAinic(&ctx_, &wc);
  EXPECT_EQ(ctx_.nRemoteQpns, 0);
  wc.byte_len = 0;
  ctx_.nRemoteQpns = 5;
  IbCastPortRecoveryExtractQpnsAinic(&ctx_, &wc);
  EXPECT_EQ(ctx_.nRemoteQpns, 0);
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, ExtractQpns_OversizedPayload_IsCappedAtTheQpLimit) {
  Make(false, 1, 1, 0);
  ibv_wc wc = P2pResiliencyWc(0, IBV_WC_SUCCESS, IBV_WC_RECV, 0,
                              NCCL_IB_UD_GRH_SIZE + (NCCL_IB_MAX_QPS + 5) * sizeof(RecoveryQpnEntry));
  IbCastPortRecoveryExtractQpnsAinic(&ctx_, &wc);
  EXPECT_EQ(ctx_.nRemoteQpns, NCCL_IB_MAX_QPS);
}

TEST_F(P2pResiliencyRecoveryAinicMicrotest, ExtractQpns_NotAinic_LeavesTheContextAlone) {
  Make(false, 1, 1, 0);
  IbCastAinicRoce = false;
  ctx_.nRemoteQpns = 7;
  ibv_wc wc = P2pResiliencyWc(0, IBV_WC_SUCCESS, IBV_WC_RECV, 0, NCCL_IB_UD_GRH_SIZE + sizeof(RecoveryQpnEntry));
  IbCastPortRecoveryExtractQpnsAinic(&ctx_, &wc);
  EXPECT_EQ(ctx_.nRemoteQpns, 7);
}

}  // namespace
