/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/p2p_resiliency.cc (via P2P_RESILIENCY_CC_PATH).

#include <gtest/gtest.h>

#include <cerrno>
#include <cstring>
#include <string>

#include "../common/LogCapture.hpp"
#include "P2pResiliencyScene.h"
#include "fakes/param_redirect.h"

#define clockNano P2pResiliencyTestClockNano
#include P2P_RESILIENCY_CC_PATH
#undef clockNano

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

constexpr uint64_t kProbeDelayMs = 10;  // IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY default

class P2pResiliencyMicrotest : public ::testing::Test {
 protected:
  P2pResiliencyScene scene_;
  P2pResiliencyScene::CommPtr<ncclIbSendComm> send_;
  P2pResiliencyScene::CommPtr<ncclIbRecvComm> recv_;
  ncclIbResiliency* res_ = nullptr;

  void TearDown() override { IbCastResiliencyDestroy(&res_); }

  // A context over ndevs devices on both sides, with every device initialised.
  ncclIbResiliency* Make(bool isSend, int ndevs, int nqps, bool recovery = false) {
    ncclIbNetCommBase* base;
    if (isSend) {
      send_ = scene_.NewComm<ncclIbSendComm>(true, ndevs, nqps);
      base = &send_->base;
    } else {
      recv_ = scene_.NewComm<ncclIbRecvComm>(false, ndevs, nqps);
      base = &recv_->base;
    }
    res_ = P2pResiliencyAttach(base, ndevs, recovery);
    return res_;
  }
  ncclIbResiliencySend* MakeSender(int ndevs, int nqps) { return (ncclIbResiliencySend*)Make(true, ndevs, nqps); }
  ncclIbResiliency* MakeReceiver(int ndevs, int nqps) { return Make(false, ndevs, nqps); }

  // The nreqs requests of a (multi-)send in slot id % NET_IB_MAX_REQUESTS.
  ncclIbRequest* SendRequest(uint64_t id, int nreqs = 1) {
    int slot = id % NET_IB_MAX_REQUESTS;
    for (int r = 0; r < nreqs; r++) {
      ncclIbRequest* req = &send_->base.reqs[(slot + r * 37) % NET_IB_MAX_REQUESTS];
      req->base = &send_->base;
      req->type = NCCL_NET_IB_REQ_SEND;
      req->id = id;
      req->nreqs = nreqs;
      send_->sendReqs[slot][r] = req;
    }
    return send_->sendReqs[slot][0];
  }

  ncclIbResiliencyDevState State(int d) { return res_->devs[d].state.load(); }

  void SetState(int d, ncclIbResiliencyDevState s) { res_->devs[d].state.store(s); }
};

TEST_F(P2pResiliencyMicrotest, Init_PortFailoverOff_LeavesResiliencyNull) {
  scene_.params["IB_RESILIENCY_PORT_FAILOVER"] = 0;
  send_ = scene_.NewComm<ncclIbSendComm>(true, 2, 2);
  res_ = (ncclIbResiliency*)0x1;
  EXPECT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  EXPECT_EQ(res_, nullptr);
}

TEST_F(P2pResiliencyMicrotest, Init_ByOrderMatching_LeavesResiliencyNull) {
  ScopedHook byOrder(g_ibCastByOrderRequested, [] { return true; });
  send_ = scene_.NewComm<ncclIbSendComm>(true, 2, 2);
  EXPECT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  EXPECT_EQ(res_, nullptr);
}

TEST_F(P2pResiliencyMicrotest, Init_QpSharingEnabled_LeavesResiliencyNull) {
  IbCastQpSharingGlobalEnable = true;
  recv_ = scene_.NewComm<ncclIbRecvComm>(false, 2, 2);
  EXPECT_EQ(IbCastResiliencyInit(&recv_->base, &res_), ncclSuccess);
  EXPECT_EQ(res_, nullptr);
}

TEST_F(P2pResiliencyMicrotest, Init_Sender_AllocatesSendContextWithoutRecovery) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  ASSERT_NE(res_, nullptr);
  EXPECT_EQ(res_->baseComm, &send_->base);
  EXPECT_FALSE(res_->inProgress);
  EXPECT_FALSE(res_->recoveryEnabled);
}

TEST_F(P2pResiliencyMicrotest, Init_Receiver_AllocatesBaseContext) {
  recv_ = scene_.NewComm<ncclIbRecvComm>(false, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&recv_->base, &res_), ncclSuccess);
  ASSERT_NE(res_, nullptr);
  EXPECT_EQ(res_->baseComm, &recv_->base);
  EXPECT_FALSE(res_->recoveryEnabled);
}

TEST_F(P2pResiliencyMicrotest, Init_RecoveryThreadRunning_EnablesRecovery) {
  scene_.params["IB_RESILIENCY_PORT_RECOVERY"] = 1;
  ASSERT_EQ(IbCastPortRecoveryThreadStart(), ncclSuccess);
  send_ = scene_.NewComm<ncclIbSendComm>(true, 2, 2);
  ncclResult_t ret = IbCastResiliencyInit(&send_->base, &res_);
  P2pResiliencyStopRecoveryThread();
  ASSERT_EQ(ret, ncclSuccess);
  EXPECT_TRUE(res_->recoveryEnabled);
  EXPECT_EQ(res_->outstandingRecovery, 0);
}

TEST_F(P2pResiliencyMicrotest, Destroy_FreesAndClearsTheContext) {
  ncclIbResiliency* none = nullptr;
  EXPECT_EQ(IbCastResiliencyDestroy(nullptr), ncclSuccess);
  EXPECT_EQ(IbCastResiliencyDestroy(&none), ncclSuccess);
  MakeReceiver(2, 2);
  EXPECT_EQ(IbCastResiliencyDestroy(&res_), ncclSuccess);
  EXPECT_EQ(res_, nullptr);
}

TEST_F(P2pResiliencyMicrotest, DeviceNumSet_Sender_ProbesOnEveryLocalDevice) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 2, 3), ncclSuccess);
  EXPECT_EQ(res_->ndevs, 2);
  EXPECT_EQ(res_->nProbingQps, 2);
  EXPECT_EQ(res_->nPortRecoveryQps, 0);
}

TEST_F(P2pResiliencyMicrotest, DeviceNumSet_Receiver_MatchesTheSendersProbingQps) {
  recv_ = scene_.NewComm<ncclIbRecvComm>(false, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&recv_->base, &res_), ncclSuccess);
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 2, 3), ncclSuccess);
  EXPECT_EQ(res_->ndevs, 2);
  EXPECT_EQ(res_->nProbingQps, 3);
}

TEST_F(P2pResiliencyMicrotest, DeviceNumSet_RecoveryEnabled_SizesRecoveryQpsForTheLargerSide) {
  recv_ = scene_.NewComm<ncclIbRecvComm>(false, 3, 3);
  ASSERT_EQ(IbCastResiliencyInit(&recv_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 3, 2), ncclSuccess);
  EXPECT_EQ(res_->nPortRecoveryQps, 3);
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 2, 3), ncclSuccess);
  EXPECT_EQ(res_->nPortRecoveryQps, 3);
}

TEST_F(P2pResiliencyMicrotest, DeviceNumSet_SingleProbingQp_IsStillConfigured) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 1, 2), ncclSuccess);
  EXPECT_EQ(res_->nProbingQps, 1);
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 2, 1), ncclSuccess);
  EXPECT_EQ(res_->nProbingQps, 2);
}

TEST_F(P2pResiliencyMicrotest, DataCqSizeGet_Sender_HoldsEveryRequestOnOneSurvivingDevice) {
  MakeSender(2, 4);
  int cqSize = 0;
  ASSERT_EQ(IbCastResiliencyDataCqSizeGet(res_, 0, &cqSize), ncclSuccess);
  EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * 1 * 2);
  send_->base.schedParms.splitData = true;
  ASSERT_EQ(IbCastResiliencyDataCqSizeGet(res_, 1, &cqSize), ncclSuccess);
  EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * 4 * 2);
}

TEST_F(P2pResiliencyMicrotest, DataCqSizeGet_Receiver_CountsTheCtsCompletionToo) {
  MakeReceiver(3, 3);
  int cqSize = 0;
  ASSERT_EQ(IbCastResiliencyDataCqSizeGet(res_, 0, &cqSize), ncclSuccess);
  EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * 2 * 1 * 3);
}

TEST_F(P2pResiliencyMicrotest, DataRqSizeGet_Receiver_HoldsEveryRequestOnOneQp) {
  MakeReceiver(2, 4);
  uint32_t rqSize = 0;
  ASSERT_EQ(IbCastResiliencyDataRqSizeGet(res_, 0, &rqSize), ncclSuccess);
  EXPECT_EQ(rqSize, (uint32_t)NET_IB_MAX_REQUESTS);
  recv_->base.schedParms.doWrr = true;
  ASSERT_EQ(IbCastResiliencyDataRqSizeGet(res_, 0, &rqSize), ncclSuccess);
  EXPECT_EQ(rqSize, (uint32_t)NET_IB_MAX_REQUESTS * 4);
}

TEST_F(P2pResiliencyMicrotest, DevInit_Sender_RegistersProbingResultsAndCreatesProbingCq) {
  ncclIbResiliencySend* sendRes = MakeSender(2, 2);
  ASSERT_EQ(scene_.regMrs.size(), 2u);
  ASSERT_EQ(scene_.createCqs.size(), 2u);
  for (int d = 0; d < 2; d++) {
    EXPECT_EQ(scene_.regMrs[d].pd, send_->devs[d].base.pd);
    EXPECT_EQ(scene_.regMrs[d].addr, (void*)&sendRes->probingResults);
    EXPECT_EQ(scene_.regMrs[d].length, sizeof(sendRes->probingResults));
    EXPECT_EQ(scene_.regMrs[d].access, IBV_ACCESS_LOCAL_WRITE);
    EXPECT_EQ(res_->devs[d].probingResultMr, scene_.regMrs[d].mr);
    EXPECT_EQ(scene_.createCqs[d].cqe, NET_IB_MAX_REQUESTS);
    EXPECT_EQ(scene_.createCqs[d].context, scene_.Dev(d)->context);
    EXPECT_EQ(scene_.createCqs[d].cqContext, (void*)&send_->base.stats);
    EXPECT_EQ(res_->devs[d].probingCq, scene_.createCqs[d].cq);
    EXPECT_EQ(res_->devs[d].state.load(), ncclIbResiliencyDevStateOk);
    EXPECT_EQ(res_->devs[d].recoveryCount, 0);
  }
}

TEST_F(P2pResiliencyMicrotest, DevInit_Receiver_CreatesSingleEntryCqWithoutMr) {
  MakeReceiver(2, 2);
  EXPECT_TRUE(scene_.regMrs.empty());
  ASSERT_EQ(scene_.createCqs.size(), 2u);
  EXPECT_EQ(scene_.createCqs[0].cqe, 1);
  EXPECT_EQ(res_->devs[1].probingCq, scene_.createCqs[1].cq);
}

TEST_F(P2pResiliencyMicrotest, DevInit_VerbsFailure_ReturnsTheError) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 1, 1);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 1, 1), ncclSuccess);
  {
    ScopedHook regMr(g_ibvRegMr, [](ibv_mr** ret, ibv_pd*, void*, size_t, int) { return ncclSystemError; });
    EXPECT_EQ(IbCastResiliencyDevInit(res_, 0, scene_.Dev(0)), ncclSystemError);
    EXPECT_TRUE(scene_.createCqs.empty());
  }
  ScopedHook createCq(g_ibvCreateCq,
                      [](ibv_cq**, ibv_context*, int, void*, ibv_comp_channel*, int) { return ncclSystemError; });
  EXPECT_EQ(IbCastResiliencyDevInit(res_, 0, scene_.Dev(0)), ncclSystemError);
}

TEST_F(P2pResiliencyMicrotest, DevInit_RecoveryEnabled_CreatesRecoveryCqAndBuffers) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 2, 2), ncclSuccess);
  ASSERT_EQ(IbCastResiliencyDevInit(res_, 0, scene_.Dev(0)), ncclSuccess);
  EXPECT_TRUE(res_->recoveryEnabled);
  ASSERT_EQ(scene_.createCqs.size(), 2u);
  EXPECT_EQ(scene_.createCqs[1].cqe, NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE);
  EXPECT_EQ(res_->devs[0].portRecoveryCq, scene_.createCqs[1].cq);
  ASSERT_EQ(scene_.regMrs.size(), 3u);
  EXPECT_EQ(scene_.regMrs[1].addr, (void*)res_->devs[0].portRecoveryGrhBuf);
  EXPECT_EQ(scene_.regMrs[2].addr, (void*)res_->portRecoveryQpnBuf);
}

TEST_F(P2pResiliencyMicrotest, DevInit_SomeDeviceLacksUd_TurnsRecoveryOff) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 2, 2), ncclSuccess);
  ASSERT_EQ(res_->nPortRecoveryQps, 2);
  scene_.Dev(1)->udSupported = 0;  // the check covers every device, not just the one being initialised
  ASSERT_EQ(IbCastResiliencyDevInit(res_, 0, scene_.Dev(0)), ncclSuccess);
  EXPECT_FALSE(res_->recoveryEnabled);
  EXPECT_EQ(res_->nPortRecoveryQps, 0);
  EXPECT_EQ(scene_.createCqs.size(), 1u);
}

TEST_F(P2pResiliencyMicrotest, DevInit_LaterDevice_DoesNotRecheckUd) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 2, 2), ncclSuccess);
  scene_.Dev(1)->udSupported = 0;
  ASSERT_EQ(IbCastResiliencyDevInit(res_, 1, scene_.Dev(1)), ncclSuccess);
  EXPECT_TRUE(res_->recoveryEnabled);
  EXPECT_EQ(scene_.createCqs.size(), 2u);
}

TEST_F(P2pResiliencyMicrotest, DevDestroy_Sender_ReleasesCqAndMr) {
  MakeSender(2, 2);
  ASSERT_EQ(IbCastResiliencyDevDestroy(res_, 1), ncclSuccess);
  ASSERT_EQ(scene_.destroyedCqs.size(), 1u);
  EXPECT_EQ(scene_.destroyedCqs[0], res_->devs[1].probingCq);
  ASSERT_EQ(scene_.deregMrs.size(), 1u);
  EXPECT_EQ(scene_.deregMrs[0], res_->devs[1].probingResultMr);
}

TEST_F(P2pResiliencyMicrotest, DevDestroy_Receiver_ReleasesOnlyTheCq) {
  MakeReceiver(2, 2);
  ASSERT_EQ(IbCastResiliencyDevDestroy(res_, 0), ncclSuccess);
  EXPECT_EQ(scene_.destroyedCqs.size(), 1u);
  EXPECT_TRUE(scene_.deregMrs.empty());
}

TEST_F(P2pResiliencyMicrotest, DevDestroy_NothingCreated_ReleasesNothing) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 1, 1);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  ASSERT_EQ(IbCastResiliencyDevDestroy(res_, 0), ncclSuccess);
  EXPECT_TRUE(scene_.destroyedCqs.empty());
  EXPECT_TRUE(scene_.deregMrs.empty());
}

TEST_F(P2pResiliencyMicrotest, DevDestroy_VerbsFailure_ReturnsTheError) {
  MakeSender(1, 1);
  {
    ScopedHook destroyCq(g_ibvDestroyCq, [](ibv_cq*) { return ncclSystemError; });
    EXPECT_EQ(IbCastResiliencyDevDestroy(res_, 0), ncclSystemError);
  }
  ScopedHook dereg(g_ibvDeregMr, [](ibv_mr*) { return ncclSystemError; });
  EXPECT_EQ(IbCastResiliencyDevDestroy(res_, 0), ncclSystemError);
}

TEST_F(P2pResiliencyMicrotest, DevDestroy_RecoveryEnabled_ReleasesRecoveryResources) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 1, 1);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 1, 1), ncclSuccess);
  ASSERT_EQ(IbCastResiliencyDevInit(res_, 0, scene_.Dev(0)), ncclSuccess);
  ASSERT_EQ(IbCastResiliencyDevDestroy(res_, 0), ncclSuccess);
  EXPECT_EQ(scene_.destroyedCqs.size(), 2u);
  EXPECT_EQ(scene_.deregMrs.size(), 3u);
  EXPECT_EQ(res_->devs[0].portRecoveryCq, nullptr);
}

TEST_F(P2pResiliencyMicrotest, SenderCreateQps_CreatesOneRcProbingQpPerLocalDevice) {
  scene_.params["IB_PKEY"] = 3;
  MakeSender(2, 2);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  ASSERT_EQ(scene_.qpCreates.size(), 2u);
  for (int d = 0; d < 2; d++) {
    const ncclIbQpCreateAttr& attr = scene_.qpCreates[d].attr;
    EXPECT_EQ(scene_.qpCreates[d].qp, &res_->probingQps[d]);
    EXPECT_EQ(attr.type, IBV_QPT_RC);
    EXPECT_EQ(attr.maxRecvWorkRequest, 0u);
    EXPECT_EQ(attr.maxSendWorkRequest, (uint32_t)NET_IB_MAX_REQUESTS);
    EXPECT_EQ(attr.cq, res_->devs[d].probingCq);
    EXPECT_EQ(attr.pd, send_->devs[d].base.pd);
    EXPECT_EQ(attr.qpContext, (void*)&send_->base.stats);
    EXPECT_FALSE(attr.isQpSharingEnabled);
    EXPECT_EQ(attr.qpSharingGroupIdx, -1);
    EXPECT_EQ(res_->probingQps[d].devIndex, d);
    EXPECT_EQ(info.probingQpsInfo[d].qpn, res_->probingQps[d].qp->qp_num);
    EXPECT_EQ(info.probingQpsInfo[d].devIndex, d);
    EXPECT_EQ(res_->probingQps[d].initAttr.state, IBV_QPS_INIT);
    EXPECT_EQ(res_->probingQps[d].initAttr.pkeyIndex, 3);
    EXPECT_EQ(res_->probingQps[d].initAttr.portNum, scene_.Dev(d)->portNum);
    EXPECT_EQ(res_->probingQps[d].initAttr.qpAccessFlags, IBV_ACCESS_LOCAL_WRITE);
  }
  ASSERT_EQ(scene_.transitions.size(), 2u);
  EXPECT_EQ(scene_.transitions[1].step, "INIT");
  EXPECT_EQ(scene_.transitions[1].qp, &res_->probingQps[1]);
}

TEST_F(P2pResiliencyMicrotest, SenderCreateQps_QpCreateOrInitFails_ReturnsTheError) {
  MakeSender(2, 2);
  ncclIbResiliencyInfo info = {};
  {
    ScopedHook create(g_ibCastQpCreate, [](ncclIbQp*, ncclIbQpCreateAttr*) { return ncclSystemError; });
    EXPECT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSystemError);
  }
  ScopedHook init(g_ibCastQpInit, [](ncclIbQp*) { return ncclInternalError; });
  EXPECT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclInternalError);
}

TEST_F(P2pResiliencyMicrotest, SenderQpsToRts_ConnectsEachProbingQpToItsRemoteDevice) {
  scene_.params["IB_TIMEOUT"] = 14;
  scene_.params["IB_RETRY_CNT"] = 5;
  MakeSender(2, 2);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, false);
  remote.devs[1].link_layer = IBV_LINK_LAYER_INFINIBAND;
  remote.devs[0].mtu = IBV_MTU_4096;
  scene_.Dev(1)->portAttr.active_mtu = IBV_MTU_2048;
  scene_.transitions.clear();
  ASSERT_EQ(IbCastResiliencySenderQpsToRts(res_, &remote), ncclSuccess);
  // Probing QP 0 talks to remote device 1 (InfiniBand), QP 1 to remote device 0 (RoCE).
  const ncclIbQpRtrAttr& rtr0 = res_->probingQps[0].rtrAttr;
  EXPECT_EQ(res_->probingQps[0].remDevIdx, 1);
  EXPECT_EQ(rtr0.mtu, IBV_MTU_1024);
  EXPECT_EQ(rtr0.linkLayer, IBV_LINK_LAYER_INFINIBAND);
  EXPECT_EQ(rtr0.tc, (uint8_t)-1);
  EXPECT_EQ(rtr0.sl, 3);
  EXPECT_EQ(rtr0.remoteQpNum, 0x500u);
  EXPECT_EQ(rtr0.remoteLid, 0x301u);
  EXPECT_EQ(rtr0.remoteGid.global.interface_id, 0xC1u);
  EXPECT_EQ(rtr0.localIbPort, 6);
  EXPECT_EQ(rtr0.localPortFlags, 0x40);
  EXPECT_EQ(rtr0.localGid.global.interface_id, 0xA0u);
  EXPECT_EQ(rtr0.localGidIndex, 10);
  const ncclIbQpRtrAttr& rtr1 = res_->probingQps[1].rtrAttr;
  EXPECT_EQ(res_->probingQps[1].remDevIdx, 0);
  EXPECT_EQ(rtr1.mtu, IBV_MTU_2048);
  EXPECT_EQ(rtr1.tc, 106);
  EXPECT_EQ(rtr1.remoteQpNum, 0x501u);
  EXPECT_EQ(rtr1.localGidIndex, 11);
  for (int q = 0; q < 2; q++) {
    EXPECT_EQ(res_->probingQps[q].rtsAttr.timeout, 14);
    EXPECT_EQ(res_->probingQps[q].rtsAttr.retryCnt, 5);
  }
  ASSERT_EQ(scene_.transitions.size(), 4u);
  EXPECT_EQ(scene_.transitions[0].step, "RTR");
  EXPECT_EQ(scene_.transitions[1].step, "RTS");
  EXPECT_EQ(scene_.transitions[1].rtsAttr.timeout, 14);
}

TEST_F(P2pResiliencyMicrotest, SenderQpsToRts_RtrOrRtsFails_ReturnsTheError) {
  MakeSender(1, 1);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(1, false);
  {
    ScopedHook rtr(g_ibCastQpRtr, [](ncclIbQp*) { return ncclSystemError; });
    EXPECT_EQ(IbCastResiliencySenderQpsToRts(res_, &remote), ncclSystemError);
  }
  ScopedHook rts(g_ibCastQpRts, [](ncclIbQp*) { return ncclSystemError; });
  EXPECT_EQ(IbCastResiliencySenderQpsToRts(res_, &remote), ncclSystemError);
}

TEST_F(P2pResiliencyMicrotest, SenderCreateQps_RecoveryEnabled_AlsoCreatesUdRecoveryQps) {
  Make(true, 2, 2, true);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  ASSERT_EQ(scene_.qpCreates.size(), 4u);
  EXPECT_EQ(scene_.qpCreates[2].attr.type, IBV_QPT_UD);
  EXPECT_EQ(scene_.qpCreates[2].qp, &res_->portRecoveryQps[0]);
  EXPECT_EQ(info.portRecoveryQpsInfo[1].qpn, res_->portRecoveryQps[1].qp->qp_num);
}

TEST_F(P2pResiliencyMicrotest, SenderQpsToRts_PeerWithoutRecovery_TurnsRecoveryOffAndFreesIt) {
  Make(true, 2, 2, true);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  ibv_qp* recoveryQp0 = res_->portRecoveryQps[0].qp;
  ibv_qp* recoveryQp1 = res_->portRecoveryQps[1].qp;
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, false);
  CaptureLog([&] { ASSERT_EQ(IbCastResiliencySenderQpsToRts(res_, &remote), ncclSuccess); });
  EXPECT_FALSE(res_->recoveryEnabled);
  EXPECT_EQ(res_->nPortRecoveryQps, 0);
  EXPECT_EQ(scene_.destroyedQps, (std::vector<ibv_qp*>{recoveryQp0, recoveryQp1}));
  EXPECT_EQ(res_->portRecoveryQps[0].qp, nullptr);
  EXPECT_EQ(res_->portRecoveryQps[1].qp, nullptr);
  EXPECT_EQ(res_->devs[0].portRecoveryCq, nullptr);
  EXPECT_EQ(res_->devs[1].portRecoveryCq, nullptr);
  EXPECT_TRUE(scene_.ahAttrs.empty());
}

TEST_F(P2pResiliencyMicrotest, SenderQpsToRts_PeerWithRecovery_ConnectsRecoveryQps) {
  Make(true, 2, 2, true);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, true);
  ASSERT_EQ(IbCastResiliencySenderQpsToRts(res_, &remote), ncclSuccess);
  EXPECT_TRUE(res_->recoveryEnabled);
  EXPECT_EQ(scene_.ahAttrs.size(), 2u);
  EXPECT_EQ(res_->portRecoveryRemoteQpn[1], 0x601u);
}

TEST_F(P2pResiliencyMicrotest, ReceiverQpsCreateToRts_StripesProbingQpsOverLocalDevices) {
  scene_.params["IB_PKEY"] = 2;
  recv_ = scene_.NewComm<ncclIbRecvComm>(false, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&recv_->base, &res_), ncclSuccess);
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 2, 3), ncclSuccess);
  for (int d = 0; d < 2; d++) ASSERT_EQ(IbCastResiliencyDevInit(res_, d, scene_.Dev(d)), ncclSuccess);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(3, false);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencyReceiverQpsCreateToRts(res_, &remote, &info), ncclSuccess);
  ASSERT_EQ(scene_.qpCreates.size(), 3u);
  const int expectDev[3] = {0, 1, 0};
  for (int q = 0; q < 3; q++) {
    const ncclIbQp& qp = res_->probingQps[q];
    EXPECT_EQ(qp.devIndex, expectDev[q]);
    EXPECT_EQ(scene_.qpCreates[q].attr.cq, res_->devs[expectDev[q]].probingCq);
    EXPECT_EQ(scene_.qpCreates[q].attr.pd, recv_->devs[expectDev[q]].base.pd);
    EXPECT_EQ(scene_.qpCreates[q].attr.maxSendWorkRequest, 0u);
    EXPECT_EQ(scene_.qpCreates[q].attr.maxRecvWorkRequest, 0u);
    EXPECT_EQ(info.probingQpsInfo[q].devIndex, expectDev[q]);
    EXPECT_EQ(info.probingQpsInfo[q].qpn, qp.qp->qp_num);
    EXPECT_EQ(qp.initAttr.qpAccessFlags, IBV_ACCESS_REMOTE_READ);
    EXPECT_EQ(qp.initAttr.pkeyIndex, 2);
    EXPECT_EQ(qp.initAttr.portNum, scene_.Dev(expectDev[q])->portNum);
    EXPECT_EQ(qp.rtrAttr.remoteQpNum, 0x500u + q);
    EXPECT_EQ(qp.rtrAttr.remoteLid, 0x300u + (2 - q % 3));
    EXPECT_EQ(qp.rtrAttr.localGidIndex, 10 + expectDev[q]);
    EXPECT_EQ(qp.rtrAttr.localPortFlags, 0x40 + expectDev[q]);
    EXPECT_EQ(qp.rtsAttr.timeout, 20);
    EXPECT_EQ(qp.rtsAttr.retryCnt, 7);
  }
  ASSERT_EQ(scene_.transitions.size(), 9u);
  EXPECT_EQ(scene_.transitions[0].step, "INIT");
  EXPECT_EQ(scene_.transitions[1].step, "RTR");
  EXPECT_EQ(scene_.transitions[2].step, "RTS");
}

TEST_F(P2pResiliencyMicrotest, ReceiverQpsCreateToRts_StepFails_ReturnsTheError) {
  MakeReceiver(1, 1);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(1, false);
  ncclIbResiliencyInfo info = {};
  for (auto* hook : {&g_ibCastQpInit, &g_ibCastQpRtr, &g_ibCastQpRts}) {
    ScopedHook fail(*hook, [](ncclIbQp*) { return ncclSystemError; });
    EXPECT_EQ(IbCastResiliencyReceiverQpsCreateToRts(res_, &remote, &info), ncclSystemError);
  }
  ScopedHook create(g_ibCastQpCreate, [](ncclIbQp*, ncclIbQpCreateAttr*) { return ncclSystemError; });
  EXPECT_EQ(IbCastResiliencyReceiverQpsCreateToRts(res_, &remote, &info), ncclSystemError);
}

TEST_F(P2pResiliencyMicrotest, ReceiverQpsCreateToRts_PeerWithoutRecovery_TurnsRecoveryOff) {
  Make(false, 2, 2, true);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, false);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencyReceiverQpsCreateToRts(res_, &remote, &info), ncclSuccess);
  EXPECT_FALSE(res_->recoveryEnabled);
  EXPECT_EQ(res_->nPortRecoveryQps, 0);
  // Only the probing QPs exist: no recovery QP was created or destroyed.
  EXPECT_EQ(scene_.qpCreates.size(), 2u);
  EXPECT_TRUE(scene_.destroyedQps.empty());
  EXPECT_EQ(res_->devs[0].portRecoveryCq, nullptr);
  EXPECT_EQ(info.portRecoveryQpsInfo[0].qpn, 0u);
}

TEST_F(P2pResiliencyMicrotest, ReceiverQpsCreateToRts_PeerWithRecovery_CreatesRecoveryQps) {
  Make(false, 2, 2, true);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, true);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencyReceiverQpsCreateToRts(res_, &remote, &info), ncclSuccess);
  EXPECT_TRUE(res_->recoveryEnabled);
  ASSERT_EQ(scene_.qpCreates.size(), 4u);
  EXPECT_EQ(scene_.qpCreates[3].attr.type, IBV_QPT_UD);
  EXPECT_EQ(info.portRecoveryQpsInfo[1].qpn, res_->portRecoveryQps[1].qp->qp_num);
}

TEST_F(P2pResiliencyMicrotest, QpReconfigure_MissingArgument_ReturnsInternalError) {
  MakeSender(1, 1);
  ncclIbQp qp = send_->base.qps[0];
  ncclIbQp noVerbsQp = {};
  ncclIbNetCommDevBase* devBase = &send_->devs[0].base;
  bool ok = true;
  EXPECT_EQ(IbCastResiliencyQpReconfigure(nullptr, &qp, devBase, 0, &ok), ncclInternalError);
  EXPECT_EQ(IbCastResiliencyQpReconfigure(res_, nullptr, devBase, 0, &ok), ncclInternalError);
  EXPECT_EQ(IbCastResiliencyQpReconfigure(res_, &noVerbsQp, devBase, 0, &ok), ncclInternalError);
  EXPECT_EQ(IbCastResiliencyQpReconfigure(res_, &qp, nullptr, 0, &ok), ncclInternalError);
  EXPECT_EQ(IbCastResiliencyQpReconfigure(res_, &qp, devBase, 0, nullptr), ncclInternalError);
  EXPECT_TRUE(scene_.transitions.empty());
}

TEST_F(P2pResiliencyMicrotest, QpReconfigure_Success_ResetsAndReconnectsWithTheCurrentGid) {
  MakeSender(1, 1);
  ncclIbQp* qp = &send_->base.qps[0];
  ncclIbNetCommDevBase devBase = send_->devs[0].base;
  devBase.gidInfo.localGidIndex = 42;
  devBase.gidInfo.localGid.global.interface_id = 0x4242;
  bool ok = false;
  ASSERT_EQ(IbCastResiliencyQpReconfigure(res_, qp, &devBase, 0, &ok), ncclSuccess);
  EXPECT_TRUE(ok);
  ASSERT_EQ(scene_.transitions.size(), 4u);
  EXPECT_EQ(scene_.transitions[0].step, "RESET");
  EXPECT_EQ(scene_.transitions[1].step, "INIT");
  EXPECT_EQ(scene_.transitions[2].step, "RTR");
  EXPECT_EQ(scene_.transitions[3].step, "RTS");
  EXPECT_EQ(scene_.transitions[2].rtrAttr.localGidIndex, 42);
  EXPECT_EQ(scene_.transitions[2].rtrAttr.localGid.global.interface_id, 0x4242u);
  EXPECT_TRUE(scene_.eceSets.empty());
}

TEST_F(P2pResiliencyMicrotest, QpReconfigure_EceSupported_RestoresEceBeforeRtr) {
  MakeSender(1, 1);
  ncclIbQp* qp = &send_->base.qps[0];
  qp->eceSupported = 1;
  size_t transitionsAtEce = 0;
  ScopedHook ece(g_ibvSetEce, [&](ibv_qp* q, ibv_ece*, int*) {
    EXPECT_EQ(q, qp->qp);
    transitionsAtEce = scene_.transitions.size();
    return ncclSuccess;
  });
  bool ok = false;
  ASSERT_EQ(IbCastResiliencyQpReconfigure(res_, qp, &send_->devs[0].base, 0, &ok), ncclSuccess);
  EXPECT_TRUE(ok);
  EXPECT_EQ(ece.calls, 1);
  EXPECT_EQ(transitionsAtEce, 2u);  // after RESET and INIT
}

// Each step's failure is reported through *success, not as an error, and stops the sequence.
TEST_F(P2pResiliencyMicrotest, QpReconfigure_StepFails_ReportsNotReconfigured) {
  const char* steps[] = {"RESET", "INIT", "ECE", "RTR", "RTS"};
  for (size_t failAt = 0; failAt < 5; failAt++) {
    const std::string failing = steps[failAt];
    MakeSender(1, 1);
    ncclIbQp* qp = &send_->base.qps[0];
    qp->eceSupported = 1;
    auto step = [&](const char* name) {
      return [&, name](ncclIbQp* q) {
        scene_.transitions.push_back({q, name, {}, {}, {}});
        return failing == name ? ncclSystemError : ncclSuccess;
      };
    };
    ScopedHook reset(g_ibCastQpReset, step("RESET"));
    ScopedHook init(g_ibCastQpInit, step("INIT"));
    ScopedHook rtr(g_ibCastQpRtr, step("RTR"));
    ScopedHook rts(g_ibCastQpRts, step("RTS"));
    ScopedHook ece(g_ibvSetEce, [&](ibv_qp*, ibv_ece*, int*) {
      scene_.transitions.push_back({qp, "ECE", {}, {}, {}});
      return failing == "ECE" ? ncclSystemError : ncclSuccess;
    });
    bool ok = true;
    ASSERT_EQ(IbCastResiliencyQpReconfigure(res_, qp, &send_->devs[0].base, 0, &ok), ncclSuccess) << failing;
    EXPECT_FALSE(ok) << failing;
    ASSERT_EQ(scene_.transitions.size(), failAt + 1) << failing;
    EXPECT_EQ(scene_.transitions.back().step, failing);
    scene_.transitions.clear();
    IbCastResiliencyDestroy(&res_);
  }
}

TEST_F(P2pResiliencyMicrotest, QpsReconfigure_NoDeviceBase_ReturnsInternalError) {
  MakeSender(1, 1);
  ScopedHook devBase(g_ibCastGetNetCommDevBase, P2pResiliencyNoDevBase);
  bool ok = true;
  EXPECT_EQ(IbCastResiliencyQpsReconfigure(res_, 0, &ok), ncclInternalError);
  EXPECT_FALSE(ok);
}

TEST_F(P2pResiliencyMicrotest, QpsReconfigure_RefreshesGidAndReconfiguresThatDevicesProbingQps) {
  MakeSender(2, 2);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  res_->nProbingQps = 3;  // a third, never-created probing QP is skipped
  scene_.Dev(1)->gidInfo.localGidIndex = 77;
  scene_.transitions.clear();
  bool ok = false;
  ASSERT_EQ(IbCastResiliencyQpsReconfigure(res_, 1, &ok), ncclSuccess);
  EXPECT_TRUE(ok);
  EXPECT_EQ(send_->devs[1].base.gidInfo.localGidIndex, 77);
  EXPECT_EQ(send_->devs[0].base.gidInfo.localGidIndex, 10);
  ASSERT_EQ(scene_.transitions.size(), 4u);
  for (const auto& t : scene_.transitions) EXPECT_EQ(t.qp, &res_->probingQps[1]);
  EXPECT_EQ(scene_.transitions[2].rtrAttr.localGidIndex, 77);
}

TEST_F(P2pResiliencyMicrotest, QpsReconfigure_Ainic_LeavesProbingQpsAlone) {
  MakeSender(1, 1);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  IbCastAinicRoce = true;
  scene_.transitions.clear();
  bool ok = false;
  ASSERT_EQ(IbCastResiliencyQpsReconfigure(res_, 0, &ok), ncclSuccess);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(scene_.transitions.empty());
}

TEST_F(P2pResiliencyMicrotest, QpsReconfigure_ProbingQpFails_ReportsNotReconfigured) {
  MakeSender(1, 1);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  ScopedHook rtr(g_ibCastQpRtr, [](ncclIbQp*) { return ncclSystemError; });
  bool ok = true;
  ASSERT_EQ(IbCastResiliencyQpsReconfigure(res_, 0, &ok), ncclSuccess);
  EXPECT_FALSE(ok);
}

TEST_F(P2pResiliencyMicrotest, Close_NullContext_IsTolerated) {
  EXPECT_EQ(IbCastResiliencyClose(nullptr), ncclSuccess);
}

TEST_F(P2pResiliencyMicrotest, Close_DestroysEveryCreatedProbingQp) {
  MakeSender(2, 2);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  res_->probingQps[0].qp = nullptr;
  ASSERT_EQ(IbCastResiliencyClose(res_), ncclSuccess);
  EXPECT_EQ(scene_.destroyedQps, (std::vector<ibv_qp*>{res_->probingQps[1].qp}));
  ScopedHook destroy(g_ibvDestroyQp, [](ibv_qp*) { return ncclSystemError; });
  EXPECT_EQ(IbCastResiliencyClose(res_), ncclSystemError);
}

TEST_F(P2pResiliencyMicrotest, Close_RecoveryEnabled_DestroysRecoveryQpsAndAhs) {
  Make(true, 2, 2, true);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, true);
  ASSERT_EQ(IbCastResiliencySenderQpsToRts(res_, &remote), ncclSuccess);
  res_->outstandingRecovery = 1;
  std::string log = CaptureLog([&] { ASSERT_EQ(IbCastResiliencyClose(res_), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, "still 1 outstanding recovery")) << log;
  EXPECT_EQ(scene_.destroyedQps.size(), 4u);
  EXPECT_EQ(scene_.destroyedAhs.size(), 2u);
  EXPECT_EQ(res_->portRecoveryAh[0], nullptr);
}

TEST_F(P2pResiliencyMicrotest, RemoteCompletionRecordsSet_StoresKeyAndAddressPerDevice) {
  ncclIbResiliencySend* sendRes = MakeSender(2, 2);
  ASSERT_EQ(IbCastResiliencyRemoteCompletionRecordsSet(res_, 0xAB, 0x1000, 1), ncclSuccess);
  EXPECT_EQ(sendRes->remCmplRecordsInfo[1].rkey, 0xABu);
  EXPECT_EQ(sendRes->remCmplRecordsInfo[1].addr, 0x1000u);
  EXPECT_EQ(sendRes->remCmplRecordsInfo[0].addr, 0u);
}

TEST_F(P2pResiliencyMicrotest, RequestIsComplete_NullRequest_IsNotComplete) {
  bool done = true;
  ASSERT_EQ(IbCastResiliencyRequestIsComplete(nullptr, &done), ncclSuccess);
  EXPECT_FALSE(done);
}

TEST_F(P2pResiliencyMicrotest, RequestIsComplete_BalancesSurplusAgainstFailedDevices) {
  MakeSender(3, 3);
  ncclIbRequest* req = SendRequest(4);
  struct Case {
    int events[4];
    bool complete;
  } cases[] = {
    {{0, 0, 0, 0}, true},    // every device done
    {{1, 0, 0, 0}, false},   // one completion outstanding
    {{2, -2, 0, 0}, true},   // device 1 served device 0's two completions
    {{2, -1, 0, 0}, false},  // only one of them so far
    {{0, -1, 0, 0}, true},   // nothing outstanding anywhere
    {{1, -2, 0, 0}, false},  // more surplus than outstanding
    {{0, 0, 0, 5}, true},    // counters past ndevs are not consulted
  };
  for (const Case& c : cases) {
    std::memcpy(req->events, c.events, sizeof(c.events));
    bool done = !c.complete;
    ASSERT_EQ(IbCastResiliencyRequestIsComplete(req, &done), ncclSuccess);
    EXPECT_EQ(done, c.complete) << c.events[0] << "," << c.events[1] << "," << c.events[2] << "," << c.events[3];
  }
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_SecondFailedDevice_IsFatal) {
  MakeReceiver(2, 2);
  SetState(0, ncclIbResiliencyDevStateError);
  ibv_wc wc = P2pResiliencyWc(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_WR_FLUSH_ERR, IBV_WC_RECV);
  std::string log =
      CaptureLog([&] { EXPECT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 1), ncclRemoteError); });
  EXPECT_TRUE(LogHas(log, "Detected 2 failed devices")) << log;
  EXPECT_EQ(State(1), ncclIbResiliencyDevStateOk);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_LastDevice_IsFatal) {
  MakeReceiver(1, 1);
  ibv_wc wc = P2pResiliencyWc(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_RETRY_EXC_ERR, IBV_WC_RECV);
  std::string log =
      CaptureLog([&] { EXPECT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclRemoteError); });
  EXPECT_TRUE(LogHas(log, "No functional devices left")) << log;
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_UnsupportedStatus_IsFatal) {
  MakeReceiver(2, 2);
  ibv_wc wc = P2pResiliencyWc(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_LOC_LEN_ERR, IBV_WC_RECV);
  std::string log =
      CaptureLog([&] { EXPECT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclRemoteError); });
  EXPECT_TRUE(LogHas(log, "Fatal error status in work completion")) << log;
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateOk);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_FlushOrRetryExceeded_FailsTheDeviceOver) {
  for (ibv_wc_status status : {IBV_WC_WR_FLUSH_ERR, IBV_WC_RETRY_EXC_ERR}) {
    MakeReceiver(2, 4);
    ibv_wc wc = P2pResiliencyWc(NCCL_IB_RECV_WR_ID_DUMMY, status, IBV_WC_RECV);
    if (status == IBV_WC_RETRY_EXC_ERR) wc.wr_id = NET_IB_MAX_REQUESTS - 1;
    recv_->recvReqs[NET_IB_MAX_REQUESTS - 1] = &recv_->base.reqs[0];
    ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess) << status;
    EXPECT_EQ(State(0), ncclIbResiliencyDevStateError);
    // QPs 0 and 2 lived on device 0; each moves to the next QP, which is on device 1.
    EXPECT_EQ(recv_->base.activeQps[0], &recv_->base.qps[1]);
    EXPECT_EQ(recv_->base.activeQps[1], &recv_->base.qps[1]);
    EXPECT_EQ(recv_->base.activeQps[2], &recv_->base.qps[3]);
    EXPECT_EQ(recv_->base.activeQps[3], &recv_->base.qps[3]);
    IbCastResiliencyDestroy(&res_);
  }
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_DeviceAlreadyFailed_KeepsItsReplacement) {
  MakeReceiver(2, 2);
  ibv_wc wc = P2pResiliencyWc(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_WR_FLUSH_ERR, IBV_WC_RECV);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  ncclIbQp* sentinel = &recv_->base.qps[7];
  recv_->base.activeQps[0] = sentinel;
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  EXPECT_EQ(recv_->base.activeQps[0], sentinel);
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateError);
}

TEST_F(P2pResiliencyMicrotest, ReplaceQps_SkipsQpsOnOtherUnhealthyDevices) {
  MakeReceiver(3, 3);
  SetState(0, ncclIbResiliencyDevStateError);
  SetState(1, ncclIbResiliencyDevStateRecoveryInProgress);
  CaptureLog([&] { ASSERT_EQ(IbCastResiliencyReplaceQps(res_, 0), ncclSuccess); });
  EXPECT_EQ(recv_->base.activeQps[0], &recv_->base.qps[2]);
}

TEST_F(P2pResiliencyMicrotest, ReplaceQps_NoHealthyQp_ReturnsInternalError) {
  MakeReceiver(2, 2);
  recv_->base.qps[1].devIndex = 0;  // both QPs on the failed device
  SetState(0, ncclIbResiliencyDevStateError);
  CaptureLog([&] { EXPECT_EQ(IbCastResiliencyReplaceQps(res_, 0), ncclInternalError); });
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_RecoveryCannotStart_MarksDevicePermanent) {
  Make(false, 2, 2, true);
  ScopedHook qpError(g_ibCastQpError, [](ncclIbQp*) { return ncclSystemError; });
  ibv_wc wc = P2pResiliencyWc(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_WR_FLUSH_ERR, IBV_WC_RECV);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 1), ncclSuccess);
  EXPECT_EQ(State(1), ncclIbResiliencyDevStateErrorPermanent);
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateOk);
  EXPECT_EQ(res_->outstandingRecovery, 0);
  EXPECT_FALSE(res_->inProgress);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_RecoveryStarts_QueuesTheDevice) {
  Make(false, 2, 2, true);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, true);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencyReceiverQpsCreateToRts(res_, &remote, &info), ncclSuccess);
  ibv_wc wc = P2pResiliencyWc(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_WR_FLUSH_ERR, IBV_WC_RECV);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 1), ncclSuccess);
  EXPECT_EQ(State(1), ncclIbResiliencyDevStateRecoveryInProgress);
  EXPECT_EQ(res_->outstandingRecovery, 1);
  EXPECT_TRUE(res_->inProgress);
  EXPECT_EQ(P2pResiliencyDrainRecoveryInbox(), 1);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_SenderStaleSlot_IsIgnored) {
  ncclIbResiliencySend* sendRes = MakeSender(2, 2);
  ibv_wc wc = P2pResiliencyWc(0x1205, IBV_WC_WR_FLUSH_ERR, IBV_WC_SEND);
  CaptureLog([&] { ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess); });
  EXPECT_EQ(res_->outstandingRequests, 0);
  EXPECT_EQ(sendRes->failedRequests[5].request, nullptr);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_Sender_TracksTheRequestForProbing) {
  ncclIbResiliencySend* sendRes = MakeSender(2, 2);
  const int slot = 133;  // the slot is the wr_id's low byte; bits above it carry other data
  const uint64_t id = 2 * NET_IB_MAX_REQUESTS + slot;
  ncclIbRequest* req = SendRequest(id);
  ibv_wc wc = P2pResiliencyWc(0x7700 | slot, IBV_WC_RETRY_EXC_ERR, IBV_WC_SEND);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 1), ncclSuccess);
  const ncclIbResiliencyRequestSend& tracked = sendRes->failedRequests[slot];
  EXPECT_EQ(tracked.request, req);
  EXPECT_EQ(tracked.state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(tracked.errorInfo.devIndex, 1);
  EXPECT_EQ(tracked.errorInfo.time, g_p2pResiliencyNowNs.load());
  EXPECT_EQ(tracked.failedAttempts, 0);
  EXPECT_EQ(tracked.id, id + 1);
  EXPECT_EQ(res_->outstandingRequests, 1);
  EXPECT_TRUE(res_->inProgress);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_SenderSecondErrorOfSameRequest_TracksItOnce) {
  ncclIbResiliencySend* sendRes = MakeSender(3, 3);
  SendRequest(9, 2);
  ibv_wc wc = P2pResiliencyWc(9, IBV_WC_WR_FLUSH_ERR, IBV_WC_SEND);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  sendRes->failedRequests[9].failedAttempts = 1;
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  EXPECT_EQ(res_->outstandingRequests, 1);
  EXPECT_EQ(sendRes->failedRequests[9].failedAttempts, 1);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_SenderRequestAlreadyHandled_IsIgnored) {
  ncclIbResiliencySend* sendRes = MakeSender(2, 2);
  SendRequest(3);
  sendRes->failedRequests[3].id = 4;  // tracked and freed already: id kept, request cleared
  ibv_wc wc = P2pResiliencyWc(3, IBV_WC_WR_FLUSH_ERR, IBV_WC_SEND);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  EXPECT_EQ(sendRes->failedRequests[3].request, nullptr);
  EXPECT_EQ(res_->outstandingRequests, 0);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_SenderOlderRequestThanTracked_ReturnsInternalError) {
  ncclIbResiliencySend* sendRes = MakeSender(2, 2);
  SendRequest(3);
  sendRes->failedRequests[3].id = 3 + NET_IB_MAX_REQUESTS + 1;
  ibv_wc wc = P2pResiliencyWc(3, IBV_WC_WR_FLUSH_ERR, IBV_WC_SEND);
  EXPECT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclInternalError);
  // One slot-generation newer than tracked is a new failure, not an old one.
  sendRes->failedRequests[3].id = 3;
  EXPECT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  EXPECT_EQ(res_->outstandingRequests, 1);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_SenderNonSendRequest_ReturnsInternalError) {
  MakeSender(2, 2);
  SendRequest(6)->type = NCCL_NET_IB_REQ_RECV;
  ibv_wc wc = P2pResiliencyWc(6, IBV_WC_WR_FLUSH_ERR, IBV_WC_SEND);
  EXPECT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclInternalError);
  EXPECT_EQ(res_->outstandingRequests, 0);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_ReceiverUnknownWrId_ReturnsInternalError) {
  MakeReceiver(2, 2);
  for (uint64_t wrId : {(uint64_t)NET_IB_MAX_REQUESTS + 1, (uint64_t)NCCL_IB_FLUSH_REQ_WR_ID_OFFSET - 1,
                        (uint64_t)NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + NET_IB_MAX_REQUESTS}) {
    ibv_wc wc = P2pResiliencyWc(wrId, IBV_WC_WR_FLUSH_ERR, IBV_WC_RECV);
    EXPECT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclInternalError) << wrId;
  }
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_ReceiverDummyWrId_IsIgnored) {
  MakeReceiver(2, 2);
  ibv_wc wc = P2pResiliencyWc(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_WR_FLUSH_ERR, IBV_WC_RECV);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  EXPECT_TRUE(scene_.postFifos.empty());
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_ReceiverFlush_DropsOnlyThatDevicesEvents) {
  MakeReceiver(2, 2);
  ncclIbRequest* flush = &recv_->base.reqs[NET_IB_MAX_REQUESTS - 1];
  flush->base = &recv_->base;
  flush->type = NCCL_NET_IB_REQ_FLUSH;
  flush->events[0] = 1;
  flush->events[1] = 1;
  ibv_wc wc = P2pResiliencyWc(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + NET_IB_MAX_REQUESTS - 1, IBV_WC_WR_FLUSH_ERR,
                              IBV_WC_RDMA_READ);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 1), ncclSuccess);
  EXPECT_EQ(flush->events[1], 0);
  EXPECT_EQ(flush->events[0], 1);
  EXPECT_TRUE(scene_.postFifos.empty());
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_ReceiverCts_RepostsTheCts) {
  MakeReceiver(2, 2);
  ncclIbRequest* req = &recv_->base.reqs[40];
  req->base = &recv_->base;
  req->type = NCCL_NET_IB_REQ_RECV;
  req->id = NET_IB_MAX_REQUESTS + 9;
  req->nreqs = 3;
  recv_->recvReqs[0] = req;
  ibv_wc wc = P2pResiliencyWc(0, IBV_WC_RETRY_EXC_ERR, IBV_WC_SEND);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  ASSERT_EQ(scene_.postFifos.size(), 1u);
  EXPECT_EQ(scene_.postFifos[0].comm, recv_.get());
  EXPECT_EQ(scene_.postFifos[0].req, req);
  EXPECT_EQ(scene_.postFifos[0].slot, 9);
  EXPECT_EQ(scene_.postFifos[0].n, 3);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_ReceiverCtsRepostFails_ReturnsTheError) {
  MakeReceiver(2, 2);
  ncclIbRequest* req = &recv_->base.reqs[1];
  req->base = &recv_->base;
  req->type = NCCL_NET_IB_REQ_RECV;
  recv_->recvReqs[1] = req;
  ScopedHook postFifo(g_ibCastPostFifo, [](ncclIbRecvComm*, ncclIbRequest*, int, int) { return ncclSystemError; });
  ibv_wc wc = P2pResiliencyWc(1, IBV_WC_RETRY_EXC_ERR, IBV_WC_SEND);
  EXPECT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSystemError);
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_ReceiverCompletedRequest_IsTolerated) {
  MakeReceiver(2, 2);
  recv_->recvReqs[2] = &recv_->base.reqs[2];  // type UNUSED: completed before the CTS error surfaced
  ibv_wc wc = P2pResiliencyWc(2, IBV_WC_RETRY_EXC_ERR, IBV_WC_SEND);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  EXPECT_TRUE(scene_.postFifos.empty());
}

TEST_F(P2pResiliencyMicrotest, HandleCompletionError_ReceiverUnexpectedType_ReturnsInternalError) {
  MakeReceiver(2, 2);
  recv_->base.reqs[2].type = NCCL_NET_IB_REQ_SEND;
  recv_->recvReqs[2] = &recv_->base.reqs[2];
  ibv_wc wc = P2pResiliencyWc(2, IBV_WC_RETRY_EXC_ERR, IBV_WC_SEND);
  EXPECT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclInternalError);
}

class P2pResiliencyProbeMicrotest : public P2pResiliencyMicrotest {
 protected:
  static constexpr int kSlot = 17;
  static constexpr uint64_t kId = 3 * NET_IB_MAX_REQUESTS + kSlot;
  ncclIbResiliencySend* sendRes_ = nullptr;
  ncclIbRequest* req_ = nullptr;

  // A 2-device sender whose request in kSlot failed on device 0 and is now tracked.
  void FailRequestOnDevice0(int nqps = 2, int nreqs = 1) {
    sendRes_ = MakeSender(2, nqps);
    ncclIbResiliencyInfo info = {};
    ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
    ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, false);
    ASSERT_EQ(IbCastResiliencySenderQpsToRts(res_, &remote), ncclSuccess);
    ASSERT_EQ(IbCastResiliencyRemoteCompletionRecordsSet(res_, 0x51, 0x100000, 0), ncclSuccess);
    ASSERT_EQ(IbCastResiliencyRemoteCompletionRecordsSet(res_, 0x52, 0x200000, 1), ncclSuccess);
    req_ = SendRequest(kId, nreqs);
    ibv_wc wc = P2pResiliencyWc(kSlot, IBV_WC_RETRY_EXC_ERR, IBV_WC_SEND);
    ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  }

  void CompleteProbe(int dev, ibv_wc_status status = IBV_WC_SUCCESS) {
    scene_.PushCqe(res_->devs[dev].probingCq, P2pResiliencyWc(kSlot, status, IBV_WC_RDMA_READ));
  }
};

TEST_F(P2pResiliencyProbeMicrotest, Progress_NothingInProgress_DoesNothing) {
  FailRequestOnDevice0();
  res_->inProgress = false;
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  EXPECT_TRUE(scene_.sends.empty());
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_BeforeProbeDelay_WaitsWithoutPosting) {
  FailRequestOnDevice0();
  P2pResiliencyAdvanceMs(kProbeDelayMs - 1);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  EXPECT_TRUE(scene_.sends.empty());
  EXPECT_EQ(sendRes_->failedRequests[kSlot].state, ncclIbResiliencyRequestStatePending);
  EXPECT_TRUE(res_->inProgress);
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_ProbeDelayElapsed_ReadsTheSlotsCompletionRecordFromAHealthyDevice) {
  FailRequestOnDevice0();
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  ASSERT_EQ(scene_.sends.size(), 1u);
  const auto& probe = scene_.sends[0];
  // Device 0 failed, so the probe goes out on device 1's probing QP, which talks to remote device 0.
  EXPECT_EQ(probe.qp, res_->probingQps[1].qp);
  EXPECT_EQ(probe.wr.opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(probe.wr.wr_id, (uint64_t)kSlot);
  EXPECT_EQ(probe.wr.send_flags, IBV_SEND_SIGNALED);
  ASSERT_EQ(probe.sges.size(), 1u);
  EXPECT_EQ(probe.sges[0].addr, (uint64_t)&sendRes_->probingResults[kSlot]);
  EXPECT_EQ(probe.sges[0].length, (uint32_t)NCCL_IB_MAX_QPS * sizeof(bool));
  EXPECT_EQ(probe.sges[0].lkey, res_->devs[1].probingResultMr->lkey);
  EXPECT_EQ(probe.wr.wr.rdma.rkey, 0x51u);
  EXPECT_EQ(probe.wr.wr.rdma.remote_addr,
            0x100000 + sizeof(ncclIbRequestCompletionRecord) * kSlot +
                offsetof(ncclIbRequestCompletionRecord, completions));
  EXPECT_EQ(sendRes_->failedRequests[kSlot].state, ncclIbResiliencyRequestStateProbePosted);
  EXPECT_EQ(res_->devs[1].nOutstandingProbes, 1);
  // Posted once: the next progress waits for the completion.
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  EXPECT_EQ(scene_.sends.size(), 1u);
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_NoHealthyDeviceForProbe_ReturnsInternalError) {
  FailRequestOnDevice0();
  SetState(1, ncclIbResiliencyDevStateErrorPermanent);
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  EXPECT_EQ(IbCastResiliencyProgress(res_), ncclInternalError);
  EXPECT_TRUE(scene_.sends.empty());
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_ProbeAttemptsExhausted_ReturnsRemoteError) {
  scene_.params["IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS"] = 2;
  FailRequestOnDevice0();
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  sendRes_->failedRequests[kSlot].failedAttempts = 2;
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  EXPECT_EQ(scene_.sends.size(), 1u);
  sendRes_->failedRequests[kSlot].state = ncclIbResiliencyRequestStatePending;
  sendRes_->failedRequests[kSlot].failedAttempts = 3;
  EXPECT_EQ(IbCastResiliencyProgress(res_), ncclRemoteError);
  EXPECT_EQ(scene_.sends.size(), 1u);
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_ProbePostFails_ReturnsTheError) {
  FailRequestOnDevice0();
  scene_.postSend = [](ibv_qp*, ibv_send_wr*) { return EIO; };
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  EXPECT_EQ(IbCastResiliencyProgress(res_), ncclSystemError);
  EXPECT_EQ(sendRes_->failedRequests[kSlot].state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(res_->devs[1].nOutstandingProbes, 0);
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_ProbeShowsEverythingDelivered_CompletesWithoutResending) {
  FailRequestOnDevice0(4, 2);
  req_->send.sentData[1] = true;
  req_->send.sentData[2] = true;
  sendRes_->probingResults[kSlot][1] = true;
  sendRes_->probingResults[kSlot][2] = true;
  for (int r = 0; r < 2; r++) send_->sendReqs[kSlot][r]->events[0] = send_->sendReqs[kSlot][r]->events[1] = 1;
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  CompleteProbe(1);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  EXPECT_TRUE(scene_.multiSends.empty());
  for (int r = 0; r < 2; r++) {
    EXPECT_EQ(send_->sendReqs[kSlot][r]->events[0], 0);
    EXPECT_EQ(send_->sendReqs[kSlot][r]->events[1], 0);
  }
  EXPECT_EQ(sendRes_->failedRequests[kSlot].request, nullptr);
  EXPECT_EQ(sendRes_->failedRequests[kSlot].id, kId + 1);
  EXPECT_EQ(res_->outstandingRequests, 0);
  EXPECT_EQ(res_->devs[1].nOutstandingProbes, 0);
  EXPECT_FALSE(res_->inProgress);
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_ProbeShowsMissingData_ResendsOnlyTheUndeliveredQps) {
  FailRequestOnDevice0(4, 2);
  send_->base.schedParms.splitData = true;  // every request spans all 4 QPs
  for (int q = 0; q < 4; q++) req_->send.sentData[q] = true;
  sendRes_->probingResults[kSlot][1] = true;
  sendRes_->probingResults[kSlot][3] = true;
  req_->desc.nqps = 4;
  req_->desc.startQpIndex = 2;
  req_->desc.wrrSched = true;
  for (int r = 0; r < 2; r++) send_->sendReqs[kSlot][r]->events[1] = 9;
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  CompleteProbe(1);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  // Device 0's QPs 0 and 2 now map to QPs 1 and 3, so the undelivered parts resend on device 1.
  ASSERT_EQ(scene_.multiSends.size(), 1u);
  EXPECT_EQ(scene_.multiSends[0].comm, send_.get());
  EXPECT_EQ(scene_.multiSends[0].slot, kSlot);
  EXPECT_EQ(scene_.multiSends[0].nqps, 4);
  EXPECT_EQ(scene_.multiSends[0].startQpIndex, 2);
  EXPECT_TRUE(scene_.multiSends[0].wrrSched);
  EXPECT_FALSE(scene_.multiSends[0].useWriteOp);
  EXPECT_EQ(res_->repostCount, 1);
  for (int r = 0; r < 2; r++) {
    ncclIbRequest* sub = send_->sendReqs[kSlot][r];
    EXPECT_EQ(sub->events[1], 0);  // cleared before the resent parts are counted again
  }
  // Two undelivered QPs (0 and 2) for each of the two requests.
  ASSERT_EQ(scene_.addEvents.size(), 4u);
  for (const auto& e : scene_.addEvents) EXPECT_EQ(e.devIndex, 1);
  EXPECT_EQ(scene_.addEvents[0].req, send_->sendReqs[kSlot][0]);
  EXPECT_EQ(scene_.addEvents[3].req, send_->sendReqs[kSlot][1]);
  EXPECT_FALSE(req_->send.sentData[0]);
  EXPECT_TRUE(req_->send.sentData[1]);
  EXPECT_FALSE(req_->send.sentData[2]);
  EXPECT_TRUE(req_->send.sentData[3]);
  EXPECT_EQ(sendRes_->failedRequests[kSlot].request, nullptr);
  EXPECT_FALSE(res_->inProgress);
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_ResendFails_ReturnsTheError) {
  FailRequestOnDevice0();
  req_->send.sentData[0] = true;
  ScopedHook multiSend(g_ibCastMultiSend,
                       [](ncclIbSendComm*, int, int, int, bool, bool) { return ncclSystemError; });
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  CompleteProbe(1);
  EXPECT_EQ(IbCastResiliencyProgress(res_), ncclSystemError);
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_ProbeFailsOnItsDevice_RetriesOnTheNextHealthyDevice) {
  sendRes_ = MakeSender(3, 3);
  ncclIbResiliencyInfo info = {};
  ASSERT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(3, false);
  ASSERT_EQ(IbCastResiliencySenderQpsToRts(res_, &remote), ncclSuccess);
  req_ = SendRequest(kId);
  ibv_wc wc = P2pResiliencyWc(kSlot, IBV_WC_RETRY_EXC_ERR, IBV_WC_SEND);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 2), ncclSuccess);
  // Device 2 recovers before the probe, so only device 0 fails afterwards.
  SetState(2, ncclIbResiliencyDevStateOk);
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  ASSERT_EQ(scene_.sends.size(), 1u);
  EXPECT_EQ(scene_.sends[0].qp, res_->probingQps[0].qp);
  CompleteProbe(0, IBV_WC_RETRY_EXC_ERR);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateError);
  // The failed probe is reposted at once, on device 1.
  ASSERT_EQ(scene_.sends.size(), 2u);
  EXPECT_EQ(scene_.sends[1].qp, res_->probingQps[1].qp);
  EXPECT_EQ(sendRes_->failedRequests[kSlot].failedAttempts, 1);
  EXPECT_EQ(sendRes_->failedRequests[kSlot].state, ncclIbResiliencyRequestStateProbePosted);
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_ProbeFailsWithAnotherDeviceDown_ReturnsRemoteError) {
  FailRequestOnDevice0();
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  CompleteProbe(1, IBV_WC_RETRY_EXC_ERR);
  EXPECT_EQ(IbCastResiliencyProgress(res_), ncclRemoteError);
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_PollFails_ReturnsTheError) {
  FailRequestOnDevice0();
  P2pResiliencyAdvanceMs(kProbeDelayMs);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  scene_.pollCq = [](ibv_cq*, int, ibv_wc*) { return -1; };
  EXPECT_EQ(IbCastResiliencyProgress(res_), ncclSystemError);
}

TEST_F(P2pResiliencyProbeMicrotest, Progress_ManyProbeCompletions_DrainsThemAll) {
  FailRequestOnDevice0();
  const int kProbes = 6;  // more than one poll batch
  for (int i = 0; i < kProbes; i++) {
    sendRes_->failedRequests[i].state = ncclIbResiliencyRequestStateProbePosted;
    scene_.PushCqe(res_->devs[1].probingCq, P2pResiliencyWc(i, IBV_WC_SUCCESS, IBV_WC_RDMA_READ));
  }
  res_->devs[1].nOutstandingProbes = kProbes;
  ASSERT_EQ(IbCastResiliencyProbeProgress(sendRes_), ncclSuccess);
  EXPECT_EQ(res_->devs[1].nOutstandingProbes, 0);
  for (int i = 0; i < kProbes; i++) {
    EXPECT_EQ(sendRes_->failedRequests[i].state, ncclIbResiliencyRequestStateProbeCompleted);
  }
}

TEST_F(P2pResiliencyMicrotest, Progress_RecoveredDevice_GetsItsQpsBackAndReturnsToOk) {
  MakeReceiver(2, 4);
  ibv_wc wc = P2pResiliencyWc(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_WR_FLUSH_ERR, IBV_WC_RECV);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  ASSERT_EQ(recv_->base.activeQps[2], &recv_->base.qps[3]);
  recv_->base.activeQps[1] = &recv_->base.qps[3];  // device 1's own mapping is not touched
  res_->outstandingRecovery = 1;
  res_->inProgress = true;
  SetState(0, ncclIbResiliencyDevStateRecoveryInProgress);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  EXPECT_TRUE(res_->inProgress);  // still recovering
  SetState(0, ncclIbResiliencyDevStateRecovered);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateOk);
  EXPECT_EQ(res_->devs[0].recoveryCount, 1);
  EXPECT_EQ(res_->outstandingRecovery, 0);
  EXPECT_FALSE(res_->inProgress);
  EXPECT_EQ(recv_->base.activeQps[0], &recv_->base.qps[0]);
  EXPECT_EQ(recv_->base.activeQps[1], &recv_->base.qps[3]);
  EXPECT_EQ(recv_->base.activeQps[2], &recv_->base.qps[2]);
  EXPECT_EQ(recv_->base.activeQps[3], &recv_->base.qps[3]);
}

TEST_F(P2pResiliencyMicrotest, Progress_RecoveryFailed_MarksDevicePermanentAndKeepsReplacements) {
  MakeReceiver(2, 2);
  ibv_wc wc = P2pResiliencyWc(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_WR_FLUSH_ERR, IBV_WC_RECV);
  ASSERT_EQ(IbCastResiliencyHandleCompletionError(res_, &wc, 0), ncclSuccess);
  res_->outstandingRecovery = 1;
  res_->inProgress = true;
  SetState(0, ncclIbResiliencyDevStateRecoveryFailed);
  ASSERT_EQ(IbCastResiliencyProgress(res_), ncclSuccess);
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateErrorPermanent);
  EXPECT_EQ(res_->devs[0].recoveryCount, 0);
  EXPECT_EQ(res_->outstandingRecovery, 0);
  EXPECT_EQ(recv_->base.activeQps[0], &recv_->base.qps[1]);
  EXPECT_FALSE(res_->inProgress);
}

TEST_F(P2pResiliencyMicrotest, RepostRequest_UnusedOrUnknownType_ReturnsInternalError) {
  MakeSender(1, 1);
  ncclIbRequest req = {};
  req.base = &send_->base;
  EXPECT_EQ(IbCastResiliencyRepostRequest(&req), ncclInternalError);
  req.type = NCCL_NET_IB_REQ_FLUSH;
  EXPECT_EQ(IbCastResiliencyRepostRequest(&req), ncclInternalError);
}

TEST_F(P2pResiliencyMicrotest, SendRequestFree_UntrackedSlot_ReturnsInternalError) {
  ncclIbResiliencySend* sendRes = MakeSender(1, 1);
  EXPECT_EQ(IbCastResiliencySendRequestFree(sendRes, &sendRes->failedRequests[4]), ncclInternalError);
  EXPECT_EQ(res_->outstandingRequests, 0);
}

}  // namespace
