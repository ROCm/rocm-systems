/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/p2p_resiliency_recovery.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <deque>
#include <functional>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ibv_ops_fakes.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/transport_net_ib_recovery_fakes.h"
#include "fakes/transport_net_ib_resiliency_fakes.h"
#include "transport/net_ib/NetIbQpSeams.h"

namespace {

int g_recoveryQpErrorCalls = 0;
int g_recoveryDataRecvPostCalls = 0;
std::vector<ncclIbQp*> g_recoveryErroredQps;
std::vector<ncclIbQp*> g_recoveryDataRecvQps;

ncclResult_t TestRecoveryQpError(ncclIbQp* qp) {
  ++g_recoveryQpErrorCalls;
  g_recoveryErroredQps.push_back(qp);
  return ncclSuccess;
}
ncclResult_t TestRecoveryPostDataReceives(ncclIbRecvComm*, ncclIbQp* qp) {
  ++g_recoveryDataRecvPostCalls;
  g_recoveryDataRecvQps.push_back(qp);
  return ncclSuccess;
}

}  // namespace

#define ncclParamIbResiliencyPortRecovery NetIbRecoveryParamEnable
#define ncclParamIbResiliencyPortRecoveryStartDelay NetIbRecoveryParamStartDelay
#define ncclParamIbResiliencyPortRecoveryAliveMsgBatchInterval NetIbRecoveryParamBatchInterval
#define ncclParamIbResiliencyPortRecoveryAliveMsgBatchSize NetIbRecoveryParamBatchSize
#define ncclParamIbResiliencyPortRecoveryAliveMsgSequenceSize NetIbRecoveryParamSequenceSize
#define ncclParamIbResiliencyPortRecoveryAliveMsgTimeout NetIbRecoveryParamAliveTimeout
#define ncclParamIbResiliencyPortRecoveryAckTimeout NetIbRecoveryParamAckTimeout
#define ncclParamIbResiliencyPortRecoveryAttemptsMax NetIbRecoveryParamAttemptsMax
#define ncclIbPortRecoveryThreadStart NetIbRecoveryThreadStart
#define ncclIbPortRecoveryThreadStop NetIbRecoveryThreadStop
#define ncclIbPortRecoveryInit NetIbRecoveryInit
#define ncclIbPortRecoveryClose NetIbRecoveryClose
#define ncclIbPortRecoveryDevInit NetIbRecoveryDevInit
#define ncclIbPortRecoveryDevDestroy NetIbRecoveryDevDestroy
#define ncclIbPortRecoverySenderQpsCreate NetIbRecoverySenderQpsCreate
#define ncclIbPortRecoverySenderQpsToRts NetIbRecoverySenderQpsToRts
#define ncclIbPortRecoveryReceiverQpsCreateToRts NetIbRecoveryReceiverQpsCreateToRts
#define ncclIbPortRecoveryQpsDestroy NetIbRecoveryQpsDestroy
#define ncclIbPortRecoveryQpsReconfigure NetIbRecoveryQpsReconfigure
#define ncclIbPortRecoveryHandleFailure NetIbRecoveryHandleFailure
#define ncclIbPortRecoveryAsyncThread NetIbRecoveryAsyncThread
#define ncclIbPortRecoveryAsyncThreadMain NetIbRecoveryAsyncThreadMain
#define clockNano NetIbTestClockNano
#define ncclIbQpError TestRecoveryQpError
#define ncclIbQpCreate NetIbTestQpCreate
#define ncclIbQpInit NetIbTestQpInit
#define ncclIbQpRtr NetIbTestQpRtr
#define ncclIbQpRts NetIbTestQpRts
#define ncclIbPostReceiveWorkRequestsOnQp TestRecoveryPostDataReceives
#include NET_IB_P2P_RESILIENCY_RECOVERY_CC_PATH
#undef ncclIbPostReceiveWorkRequestsOnQp
#undef ncclIbQpRts
#undef ncclIbQpRtr
#undef ncclIbQpInit
#undef ncclIbQpCreate
#undef ncclIbQpError
#undef clockNano
#undef ncclIbPortRecoveryAsyncThreadMain
#undef ncclIbPortRecoveryAsyncThread
#undef ncclIbPortRecoveryHandleFailure
#undef ncclIbPortRecoveryQpsReconfigure
#undef ncclIbPortRecoveryQpsDestroy
#undef ncclIbPortRecoveryReceiverQpsCreateToRts
#undef ncclIbPortRecoverySenderQpsToRts
#undef ncclIbPortRecoverySenderQpsCreate
#undef ncclIbPortRecoveryDevDestroy
#undef ncclIbPortRecoveryDevInit
#undef ncclIbPortRecoveryClose
#undef ncclIbPortRecoveryInit
#undef ncclIbPortRecoveryThreadStop
#undef ncclIbPortRecoveryThreadStart
#undef ncclParamIbResiliencyPortRecoveryAttemptsMax
#undef ncclParamIbResiliencyPortRecoveryAckTimeout
#undef ncclParamIbResiliencyPortRecoveryAliveMsgTimeout
#undef ncclParamIbResiliencyPortRecoveryAliveMsgSequenceSize
#undef ncclParamIbResiliencyPortRecoveryAliveMsgBatchSize
#undef ncclParamIbResiliencyPortRecoveryAliveMsgBatchInterval
#undef ncclParamIbResiliencyPortRecoveryStartDelay
#undef ncclParamIbResiliencyPortRecovery

namespace {

struct RecoveryScene {
  ncclIbSendComm send{};
  ncclIbRecvComm recv{};
  ncclIbResiliency sendRes{};
  ncclIbResiliency recvRes{};
  std::array<FakeIbvVerbs, 2> verbs{{FakeIbvVerbs(100, IBV_QPT_UC), FakeIbvVerbs(101, IBV_QPT_UC)}};

  RecoveryScene() {
    send.base.isSend = true;
    recv.base.isSend = false;
    sendRes.baseComm = &send.base;
    recvRes.baseComm = &recv.base;
    send.base.resiliency = &sendRes;
    recv.base.resiliency = &recvRes;
    sendRes.ndevs = recvRes.ndevs = 2;
    sendRes.nPortRecoveryQps = recvRes.nPortRecoveryQps = 2;
    for (int i = 0; i < 2; ++i) {
      sendRes.devs[i].state.store(ncclIbResiliencyDevStateOk, std::memory_order_release);
      recvRes.devs[i].state.store(ncclIbResiliencyDevStateOk, std::memory_order_release);
      sendRes.devs[i].portRecoveryCq = &verbs[i].cq;
      recvRes.devs[i].portRecoveryCq = &verbs[i].cq;
      sendRes.portRecoveryQps[i].qp = &verbs[i].qp;
      recvRes.portRecoveryQps[i].qp = &verbs[i].qp;
      sendRes.portRecoveryQps[i].devIndex = i;
      recvRes.portRecoveryQps[i].devIndex = i;
      send.base.qps[i].qp = &verbs[i].qp;
      recv.base.qps[i].qp = &verbs[i].qp;
      send.base.qps[i].devIndex = i;
      recv.base.qps[i].devIndex = i;
      send.base.activeQps[i] = &send.base.qps[i];
      recv.base.activeQps[i] = &recv.base.qps[i];
    }
    send.base.nqps = recv.base.nqps = 2;
    send.base.vProps.ndevs = recv.base.vProps.ndevs = 2;
  }
};

class NetIbP2pResiliencyRecoveryMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (int i = 0; i < 2; ++i) {
      savedPortNums_[i] = ncclIbDevs[i].portNum;
      savedMtus_[i] = ncclIbDevs[i].portAttr.active_mtu;
    }
  }

  void TearDown() override {
    {
      // Leftover contexts reference a destroyed RecoveryScene, and
      // ncclIbPortRecoveryContextDestroy dereferences resCtx, so free them directly.
      std::lock_guard<std::mutex> lock(ncclIbPortRecoveryMutex);
      for (auto* context : recoveryInbox) free(context);
      recoveryInbox.clear();
    }
    for (int i = 0; i < 2; ++i) {
      ncclIbDevs[i].portNum = savedPortNums_[i];
      ncclIbDevs[i].portAttr.active_mtu = savedMtus_[i];
    }
    g_recoveryQpErrorCalls = 0;
    g_recoveryDataRecvPostCalls = 0;
    g_recoveryErroredQps.clear();
    g_recoveryDataRecvQps.clear();
    ResetNetIbQpSeams();
    ResetIbvOpsFakes();
    ResetTransportNetIbResiliencyFakes();
    ResetTransportNetIbRecoveryFakes();
    ResetIbvwrapFakes();
    ResetNcclFakes();
  }

 private:
  std::array<decltype(ncclIbDevs[0].portNum), 2> savedPortNums_{};
  std::array<decltype(ncclIbDevs[0].portAttr.active_mtu), 2> savedMtus_{};
};

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsToError_TransitionsOnlyFailedDeviceQps) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 1;

  EXPECT_EQ(ncclIbPortRecoveryQpsToError(&context), ncclSuccess);
  EXPECT_EQ(g_recoveryQpErrorCalls, 1);
  ASSERT_EQ(g_recoveryErroredQps.size(), 1u);
  EXPECT_EQ(g_recoveryErroredQps[0], &scene.send.base.qps[1]);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsReconfigure_SenderPostsOneReceiveAndReportsSuccess) {
  RecoveryScene scene;
  ScopedHook reconfigure(g_ncclIbResiliencyQpReconfigure,
                         [&](ncclIbResiliency*, ncclIbQp* qp, ncclIbNetCommDevBase*, int, bool* success) {
                           EXPECT_EQ(qp, &scene.sendRes.portRecoveryQps[0]);
                           *success = true;
                           return ncclSuccess;
                         });
  bool success = false;

  ASSERT_EQ(NetIbRecoveryQpsReconfigure(&scene.sendRes, 0, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(reconfigure.calls, 1);
  ASSERT_EQ(g_ibvPostRecvQps.size(), 1u);
  EXPECT_EQ(g_ibvPostRecvQps[0], scene.sendRes.portRecoveryQps[0].qp);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsReconfigure_ReceiverPostsTenReceives) {
  RecoveryScene scene;
  ScopedHook reconfigure(g_ncclIbResiliencyQpReconfigure,
                         [&](ncclIbResiliency*, ncclIbQp* qp, ncclIbNetCommDevBase*, int, bool* success) {
                           EXPECT_EQ(qp, &scene.recvRes.portRecoveryQps[1]);
                           *success = true;
                           return ncclSuccess;
                         });
  bool success = false;

  ASSERT_EQ(NetIbRecoveryQpsReconfigure(&scene.recvRes, 1, &success), ncclSuccess);
  EXPECT_TRUE(success);
  ASSERT_EQ(g_ibvPostRecvQps.size(), NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
  for (auto* qp : g_ibvPostRecvQps) EXPECT_EQ(qp, scene.recvRes.portRecoveryQps[1].qp);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DrainCq_RepostsOneReceiveForEveryDrainedCompletion) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.recoveryCq = &scene.verbs[0].cq;
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV;
  g_ibvCompletions.push_back(completion);
  g_ibvCompletions.push_back(completion);
  bool success = false;

  EXPECT_EQ(ncclIbPortRecoveryDrainCqAndPostReceiveWRs(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(g_ibvPostRecvQps.size(), 2u);
  ASSERT_FALSE(g_ibvPolledCqs.empty());
  EXPECT_EQ(g_ibvPolledCqs[0], context.recoveryCq);
  for (auto* qp : g_ibvPostRecvQps) EXPECT_EQ(qp, scene.recvRes.portRecoveryQps[0].qp);
}

// AICOMRCCL-3128: SEND completions do not consume receive WQEs and should not
// cause the recovery QP to over-post its receive queue.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_DrainCq_SendCompletionDoesNotPostReceive) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.recoveryCq = &scene.verbs[0].cq;
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_SEND;
  g_ibvCompletions.push_back(completion);
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryDrainCqAndPostReceiveWRs(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_TRUE(g_ibvPostRecvQps.empty());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextInit_SenderInitializesStateAndPostsAckReceive) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext* context = nullptr;
  g_netIbNowNs = 5000;

  ASSERT_EQ(ncclIbPortRecoveryContextInit(&scene.sendRes, 0, &context), ncclSuccess);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->resCtx, &scene.sendRes);
  EXPECT_EQ(context->devIndex, 0);
  EXPECT_EQ(context->state, ncclIbPortRecoveryStateInit);
  EXPECT_EQ(context->timeInit, 5000u);
  EXPECT_EQ(g_recoveryQpErrorCalls, 1);
  ASSERT_EQ(g_recoveryErroredQps.size(), 1u);
  EXPECT_EQ(g_recoveryErroredQps[0], &scene.send.base.qps[0]);
  ASSERT_EQ(g_ibvPostRecvQps.size(), 1u);
  EXPECT_EQ(g_ibvPostRecvQps[0], scene.sendRes.portRecoveryQps[0].qp);
  EXPECT_EQ(ncclIbPortRecoveryContextDestroy(context), ncclSuccess);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextInit_ReceiverPostsAliveAndFinalAckReceives) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext* context = nullptr;

  ASSERT_EQ(ncclIbPortRecoveryContextInit(&scene.recvRes, 1, &context), ncclSuccess);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(g_ibvPostRecvQps.size(), NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
  for (auto* qp : g_ibvPostRecvQps) EXPECT_EQ(qp, scene.recvRes.portRecoveryQps[1].qp);
  EXPECT_EQ(ncclIbPortRecoveryContextDestroy(context), ncclSuccess);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextInit_PostFailureReturnsErrorWithoutPublishingContext) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext* context = reinterpret_cast<ncclIbPortRecoveryContext*>(0x1);
  g_ibvPostRecvResult = EIO;

  EXPECT_EQ(ncclIbPortRecoveryContextInit(&scene.sendRes, 0, &context), ncclSystemError);
  EXPECT_EQ(context, nullptr);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DevInitAndDestroyManageRecoveryCq) {
  RecoveryScene scene;
  ibv_context verbsContext{};
  ncclIbDev dev{};
  dev.context = &verbsContext;
  ibv_cq cq{};
  ScopedHook create(g_ibvCreateCq,
                    [&cq](ibv_cq** out, ibv_context*, int size, void*, ibv_comp_channel*, int) {
                      EXPECT_EQ(size, NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE);
                      *out = &cq;
                      return ncclSuccess;
                    });
  ScopedHook destroy(g_ibvDestroyCq, [&cq](ibv_cq* seen) {
    EXPECT_EQ(seen, &cq);
    return ncclSuccess;
  });
  scene.sendRes.recoveryEnabled = true;

  ASSERT_EQ(NetIbRecoveryDevInit(&scene.sendRes, 0, &dev), ncclSuccess);
  EXPECT_EQ(scene.sendRes.devs[0].portRecoveryCq, &cq);
  EXPECT_EQ(NetIbRecoveryDevDestroy(&scene.sendRes, 0), ncclSuccess);
  EXPECT_EQ(create.calls, 1);
  EXPECT_EQ(destroy.calls, 1);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsDestroy_DestroysOnlyMaterializedQps) {
  RecoveryScene scene;
  scene.sendRes.portRecoveryQps[1].qp = nullptr;
  ScopedHook destroy(g_ibvDestroyQp, [&](ibv_qp* qp) {
    EXPECT_EQ(qp, &scene.verbs[0].qp);
    return ncclSuccess;
  });

  EXPECT_EQ(NetIbRecoveryQpsDestroy(&scene.sendRes, 2), ncclSuccess);
  EXPECT_EQ(destroy.calls, 1);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, SenderRecoveryQpsCreationPublishesStripedMetadata) {
  RecoveryScene scene;
  scene.sendRes.recoveryEnabled = true;
  ibv_pd pds[2] = {};
  int created = 0;
  for (int i = 0; i < 2; ++i) {
    scene.send.devs[i].base.ibDevN = i;
    scene.send.devs[i].base.pd = &pds[i];
    scene.sendRes.devs[i].portRecoveryCq = &scene.verbs[i].cq;
    ncclIbDevs[i].portNum = i + 1;
  }
  g_netIbQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr* attr) {
    EXPECT_EQ(attr->type, IBV_QPT_UC);
    qp->qp = &scene.verbs[created++].qp;
    return ncclSuccess;
  };
  std::vector<ncclIbQp*> initialized;
  g_netIbQpInit = [&](ncclIbQp* qp) {
    initialized.push_back(qp);
    return ncclSuccess;
  };
  ncclIbQpInfo info[2] = {};

  ASSERT_EQ(NetIbRecoverySenderQpsCreate(&scene.sendRes, info, 2), ncclSuccess);
  EXPECT_EQ(created, 2);
  EXPECT_EQ(scene.sendRes.portRecoveryQps[0].devIndex, 0);
  EXPECT_EQ(scene.sendRes.portRecoveryQps[1].devIndex, 1);
  EXPECT_EQ(info[0].qpn, 100u);
  EXPECT_EQ(info[1].qpn, 101u);
  EXPECT_EQ(initialized, (std::vector<ncclIbQp*>{&scene.sendRes.portRecoveryQps[0],
                                                &scene.sendRes.portRecoveryQps[1]}));
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, SenderRecoveryQpsToRtsUsesRemoteMetadata) {
  RecoveryScene scene;
  scene.sendRes.recoveryEnabled = true;
  scene.send.devs[0].base.ibDevN = 0;
  scene.sendRes.portRecoveryQps[0].qp = &scene.verbs[0].qp;
  scene.sendRes.portRecoveryQps[0].devIndex = 0;
  ncclIbDevs[0].portNum = 1;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  ncclIbConnectionMetadata remote{};
  remote.sl = 2;
  remote.tc = 3;
  remote.resiliencyInfo.portRecoveryQpsInfo[0].qpn = 200;
  remote.resiliencyInfo.portRecoveryQpsInfo[0].devIndex = 0;
  remote.devs[0].mtu = IBV_MTU_2048;
  remote.devs[0].link_layer = IBV_LINK_LAYER_ETHERNET;
  remote.devs[0].ib_port = 1;
  std::vector<char> transitions;
  g_netIbQpRtr = [&](ncclIbQp* qp) {
    EXPECT_EQ(qp, &scene.sendRes.portRecoveryQps[0]);
    transitions.push_back('T');
    return ncclSuccess;
  };
  g_netIbQpRts = [&](ncclIbQp* qp) {
    EXPECT_EQ(qp, &scene.sendRes.portRecoveryQps[0]);
    transitions.push_back('S');
    return ncclSuccess;
  };

  ASSERT_EQ(NetIbRecoverySenderQpsToRts(&scene.sendRes, &remote, 1), ncclSuccess);
  EXPECT_EQ(scene.sendRes.portRecoveryQps[0].rtrAttr.remoteQpNum, 200u);
  EXPECT_EQ(scene.sendRes.portRecoveryQps[0].rtrAttr.mtu, IBV_MTU_2048);
  EXPECT_EQ(transitions, (std::vector<char>{'T', 'S'}));
}

// AICOMRCCL-3129: the recovery AV port is local and must not be copied from
// the remote device metadata.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_SenderRecoveryQpsToRtsUsesLocalIbPort) {
  RecoveryScene scene;
  scene.sendRes.recoveryEnabled = true;
  scene.send.devs[0].base.ibDevN = 0;
  ncclIbDevs[0].portNum = 3;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  ncclIbConnectionMetadata remote{};
  remote.resiliencyInfo.portRecoveryQpsInfo[0].qpn = 200;
  remote.resiliencyInfo.portRecoveryQpsInfo[0].devIndex = 0;
  remote.devs[0].mtu = IBV_MTU_2048;
  remote.devs[0].link_layer = IBV_LINK_LAYER_ETHERNET;
  remote.devs[0].ib_port = 7;
  g_netIbQpRtr = [](ncclIbQp*) { return ncclSuccess; };
  g_netIbQpRts = [](ncclIbQp*) { return ncclSuccess; };

  ASSERT_EQ(NetIbRecoverySenderQpsToRts(&scene.sendRes, &remote, 1), ncclSuccess);
  EXPECT_EQ(scene.sendRes.portRecoveryQps[0].rtrAttr.localIbPort, 3);
}

// AICOMRCCL-3129: receiver recovery QPs have the same local AV-port contract.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_ReceiverRecoveryQpsToRtsUsesLocalIbPort) {
  RecoveryScene scene;
  scene.recv.base.vProps.ndevs = 1;
  scene.recv.devs[0].base.ibDevN = 0;
  ibv_pd pd{};
  scene.recv.devs[0].base.pd = &pd;
  scene.recvRes.devs[0].portRecoveryCq = &scene.verbs[0].cq;
  ncclIbDevs[0].portNum = 3;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  g_netIbQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr*) {
    qp->qp = &scene.verbs[0].qp;
    return ncclSuccess;
  };
  g_netIbQpInit = [](ncclIbQp*) { return ncclSuccess; };
  g_netIbQpRtr = [](ncclIbQp*) { return ncclSuccess; };
  g_netIbQpRts = [](ncclIbQp*) { return ncclSuccess; };
  ncclIbConnectionMetadata remote{};
  remote.resiliencyInfo.portRecoveryQpsInfo[0].qpn = 300;
  remote.resiliencyInfo.portRecoveryQpsInfo[0].devIndex = 0;
  remote.devs[0].mtu = IBV_MTU_2048;
  remote.devs[0].link_layer = IBV_LINK_LAYER_ETHERNET;
  remote.devs[0].ib_port = 7;
  ncclIbQpInfo local[1] = {};

  ASSERT_EQ(NetIbRecoveryReceiverQpsCreateToRts(&scene.recvRes, &remote, local, 1), ncclSuccess);
  EXPECT_EQ(scene.recvRes.portRecoveryQps[0].rtrAttr.localIbPort, 3);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ReceiverRecoveryQpsCreationTransitionsAndPublishesInfo) {
  RecoveryScene scene;
  scene.recv.base.vProps.ndevs = 1;
  scene.recv.devs[0].base.ibDevN = 0;
  ibv_pd pd{};
  scene.recv.devs[0].base.pd = &pd;
  scene.recvRes.devs[0].portRecoveryCq = &scene.verbs[0].cq;
  ncclIbDevs[0].portNum = 1;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  int created = 0;
  g_netIbQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr*) {
    qp->qp = &scene.verbs[created++].qp;
    return ncclSuccess;
  };
  std::vector<char> transitions;
  g_netIbQpInit = [&](ncclIbQp* qp) {
    EXPECT_EQ(qp, &scene.recvRes.portRecoveryQps[0]);
    transitions.push_back('I');
    return ncclSuccess;
  };
  g_netIbQpRtr = [&](ncclIbQp* qp) {
    EXPECT_EQ(qp, &scene.recvRes.portRecoveryQps[0]);
    transitions.push_back('T');
    return ncclSuccess;
  };
  g_netIbQpRts = [&](ncclIbQp* qp) {
    EXPECT_EQ(qp, &scene.recvRes.portRecoveryQps[0]);
    transitions.push_back('S');
    return ncclSuccess;
  };
  ncclIbConnectionMetadata remote{};
  remote.resiliencyInfo.portRecoveryQpsInfo[0].qpn = 300;
  remote.resiliencyInfo.portRecoveryQpsInfo[0].devIndex = 0;
  remote.devs[0].mtu = IBV_MTU_2048;
  remote.devs[0].link_layer = IBV_LINK_LAYER_ETHERNET;
  ncclIbQpInfo local[1] = {};

  ASSERT_EQ(NetIbRecoveryReceiverQpsCreateToRts(&scene.recvRes, &remote, local, 1), ncclSuccess);
  EXPECT_EQ(created, 1);
  EXPECT_EQ(transitions, (std::vector<char>{'I', 'T', 'S'}));
  EXPECT_EQ(local[0].qpn, scene.verbs[0].qp.qp_num);
  EXPECT_EQ(scene.recvRes.portRecoveryQps[0].rtrAttr.remoteQpNum, 300u);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsRestore_ReconfiguresOnlyFailedDeviceDataQps) {
  RecoveryScene scene;
  scene.send.base.nqps = 2;
  scene.send.base.qps[0].devIndex = 0;
  scene.send.base.qps[1].devIndex = 1;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  ScopedHook reconfigure(
      g_ncclIbResiliencyQpReconfigure,
      [&](ncclIbResiliency*, ncclIbQp* qp, ncclIbNetCommDevBase* devBase, int devIndex, bool* success) {
        EXPECT_EQ(qp, &scene.send.base.qps[0]);
        EXPECT_EQ(devBase, &scene.send.devs[0].base);
        EXPECT_EQ(devIndex, 0);
        *success = qp == &scene.send.base.qps[0] && devBase == &scene.send.devs[0].base;
        return ncclSuccess;
      });
  // Callers currently pass success=true from the preceding CQ operation.
  // AICOMRCCL-3128 tracks making this a self-contained output contract.
  bool success = true;

  EXPECT_EQ(ncclIbPortRecoveryQpsRestore(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(reconfigure.calls, 1);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsRestore_ReconfigureFailureReportsFailureAndStops) {
  RecoveryScene scene;
  scene.send.base.qps[0].devIndex = 0;
  scene.send.base.qps[1].devIndex = 0;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  ScopedHook reconfigure(g_ncclIbResiliencyQpReconfigure,
                         [](ncclIbResiliency*, ncclIbQp*, ncclIbNetCommDevBase*, int, bool* success) {
                           *success = false;
                           return ncclSuccess;
                         });
  bool success = true;

  EXPECT_EQ(ncclIbPortRecoveryQpsRestore(&context, &success), ncclSuccess);
  EXPECT_FALSE(success);
  EXPECT_EQ(reconfigure.calls, 1);
}

// AICOMRCCL-3128: a successful restore should publish its own success result,
// rather than depending on the value left by a previous operation.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_QpsRestore_SetsSuccessOutputOnSuccessfulRestore) {
  RecoveryScene scene;
  scene.send.base.nqps = 0;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryQpsRestore(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, CompletionReceiver_InOrderAliveMessageAdvancesAndReposts) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.aliveMsgNextId = 7;
  context.recv.nInOrderMsgsReceived = 2;
  ibv_wc completion{};
  completion.opcode = IBV_WC_RECV;
  completion.imm_data = 7;
  bool success = false;
  g_netIbNowNs = 9000;

  EXPECT_EQ(ncclIbPortRecoveryHandleCompletionReceiver(&context, completion, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(context.recv.nInOrderMsgsReceived, 3);
  EXPECT_EQ(context.aliveMsgNextId, 8u);
  EXPECT_EQ(context.timeLastMsg, 9000u);
  EXPECT_EQ(g_ibvPostRecvQps.size(), 1u);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, CompletionSender_AliveSendAndAckMessagesUpdateState) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.recoveryCq = &scene.verbs[0].cq;
  ibv_wc completion{};
  completion.opcode = IBV_WC_SEND;
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryHandleCompletionSender(&context, completion, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_TRUE(context.send.aliveMsgCompleted);

  context.state = ncclIbPortRecoveryStateAck;
  completion.opcode = IBV_WC_RECV;
  completion.imm_data = NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID;
  ASSERT_EQ(ncclIbPortRecoveryHandleCompletionSender(&context, completion, &success), ncclSuccess);
  EXPECT_TRUE(context.ackReceived);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, PollCq_ErrorStatusReportsFailure) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  context.recoveryCq = &scene.verbs[0].cq;
  ibv_wc completion{};
  completion.status = IBV_WC_REM_ACCESS_ERR;
  g_ibvCompletions.push_back(completion);
  bool success = true;

  EXPECT_EQ(ncclIbPortRecoveryPollCq(&context, &success), ncclSuccess);
  EXPECT_FALSE(success);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, PostAliveMessages_SequencePostsOnlyFinalMessageSignaled) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE") == 0) return int64_t{3};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0) return int64_t{3};
    return fallback;
  });
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryPostAliveMessages(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  ASSERT_EQ(g_ibvPostedSends.size(), 3u);
  EXPECT_EQ(g_ibvPostedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_ibvPostedSends[1].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_NE(g_ibvPostedSends[2].sendFlags & IBV_SEND_SIGNALED, 0);
  for (size_t i = 0; i < g_ibvPostedSends.size(); ++i) {
    EXPECT_EQ(g_ibvPostedSends[i].opcode, IBV_WR_SEND_WITH_IMM);
    EXPECT_EQ(g_ibvPostedSends[i].wrId, i);
    EXPECT_EQ(g_ibvPostedSends[i].immData, i);
  }
  ASSERT_EQ(g_ibvPostSendQps.size(), 1u);
  EXPECT_EQ(g_ibvPostSendQps[0], scene.sendRes.portRecoveryQps[0].qp);
  EXPECT_EQ(context.aliveMsgNextId, 3u);
  EXPECT_TRUE(context.send.aliveMsgPosted);
}

// AICOMRCCL-3128: one batch should be controlled by ALIVE_MSG_BATCH_SIZE;
// ALIVE_MSG_SEQUENCE_SIZE controls how many messages establish liveness.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_PostAliveMessages_UsesBatchSizeIndependently) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE") == 0) return int64_t{2};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0) return int64_t{5};
    return fallback;
  });
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryPostAliveMessages(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(g_ibvPostedSends.size(), 2u);
}

// AICOMRCCL-3128: a zero-sized batch must be rejected before wr[0] is posted.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_PostAliveMessages_ZeroBatchIsRejectedWithoutPosting) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE") == 0) return int64_t{0};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0) return int64_t{5};
    return fallback;
  });
  bool success = false;

  EXPECT_NE(ncclIbPortRecoveryPostAliveMessages(&context, &success), ncclSuccess);
  EXPECT_TRUE(g_ibvPostedSends.empty());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, PostAliveMessages_AboveMaximumIsRejectedWithoutPosting) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE") == 0)
      return int64_t{NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX + 1};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0)
      return int64_t{NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX + 1};
    return fallback;
  });
  bool success = false;

  EXPECT_EQ(ncclIbPortRecoveryPostAliveMessages(&context, &success), ncclInternalError);
  EXPECT_TRUE(g_ibvPostedSends.empty());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, PostAck_PublishesSignaledAckAndMarksPosted) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryPostAck(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_TRUE(context.ackPosted);
  ASSERT_EQ(g_ibvPostedSends.size(), 1u);
  EXPECT_EQ(g_ibvPostedSends[0].immData, NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
  EXPECT_NE(g_ibvPostedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  ASSERT_EQ(g_ibvPostSendQps.size(), 1u);
  EXPECT_EQ(g_ibvPostSendQps[0], scene.recvRes.portRecoveryQps[0].qp);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, AliveProgress_SenderPostsThenCompletesBatch) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.recoveryCq = &scene.verbs[0].cq;
  g_netIbNowNs = 10000;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_INTERVAL") == 0) return int64_t{0};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0) return int64_t{1};
    return fallback;
  });
  ncclIbPortRecoveryStateProgressResult result = ncclIbPortRecoveryStateProgressResultFailed;

  ASSERT_EQ(ncclIbPortRecoveryProgressAliveMessagesSender(&context, &result), ncclSuccess);
  EXPECT_EQ(result, ncclIbPortRecoveryStateProgressResultInProgress);
  EXPECT_TRUE(context.send.aliveMsgPosted);

  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_SEND;
  g_ibvCompletions.push_back(completion);
  ASSERT_EQ(ncclIbPortRecoveryProgressAliveMessagesSender(&context, &result), ncclSuccess);
  EXPECT_EQ(result, ncclIbPortRecoveryStateProgressResultGoToNextState);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, AliveProgress_ReceiverAfterCurrentThresholdRestoresAndPostsAck) {
  RecoveryScene scene;
  scene.recv.base.nqps = 0;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.recoveryCq = &scene.verbs[0].cq;
  context.recv.nInOrderMsgsReceived = 2;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0 ? int64_t{1} : fallback;
  });
  ncclIbPortRecoveryStateProgressResult result = ncclIbPortRecoveryStateProgressResultFailed;

  ASSERT_EQ(ncclIbPortRecoveryProgressAliveMessagesReceiver(&context, &result), ncclSuccess);
  EXPECT_EQ(result, ncclIbPortRecoveryStateProgressResultInProgress);
  EXPECT_TRUE(context.ackPosted);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, AliveProgress_ReceiverBelowSequenceKeepsWaiting) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.recoveryCq = &scene.verbs[0].cq;
  context.recv.nInOrderMsgsReceived = 0;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0 ? int64_t{1} : fallback;
  });
  ncclIbPortRecoveryStateProgressResult result = ncclIbPortRecoveryStateProgressResultFailed;

  ASSERT_EQ(ncclIbPortRecoveryProgressAliveMessagesReceiver(&context, &result), ncclSuccess);
  EXPECT_EQ(result, ncclIbPortRecoveryStateProgressResultInProgress);
  EXPECT_FALSE(context.ackPosted);
}

// AICOMRCCL-3128: timing out while waiting for the receiver's local ACK
// completion should restart the alive phase instead of returning an error.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_ContextProgress_ReceiverAckTimeoutRestartsAlivePhase) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.recoveryCq = &scene.verbs[0].cq;
  context.ackPosted = true;
  context.timeLastMsg = 0;
  g_netIbNowNs = 100;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ACK_TIMEOUT") == 0) return int64_t{0};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ATTEMPTS_MAX") == 0) return int64_t{2};
    return fallback;
  });
  bool done = false;

  EXPECT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_FALSE(done);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateAliveMessages);
  EXPECT_FALSE(context.ackPosted);
  EXPECT_EQ(context.nFailedAttempts, 1);
  EXPECT_TRUE(g_ibvPostRecvQps.empty());
}

// AICOMRCCL-3128: exact configured sequence currently requires one extra
// message because production uses '>' instead of '>='.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest,
       DISABLED_AliveProgress_ReceiverExactSequenceRestoresAndPostsAck) {
  RecoveryScene scene;
  scene.recv.base.nqps = 0;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.recoveryCq = &scene.verbs[0].cq;
  context.recv.nInOrderMsgsReceived = 1;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0 ? int64_t{1} : fallback;
  });
  ncclIbPortRecoveryStateProgressResult result = ncclIbPortRecoveryStateProgressResultFailed;

  ASSERT_EQ(ncclIbPortRecoveryProgressAliveMessagesReceiver(&context, &result), ncclSuccess);
  EXPECT_TRUE(context.ackPosted);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, AckProgress_SenderReceivesAckPostsFinalAckThenCompletes) {
  RecoveryScene scene;
  scene.send.base.nqps = 0;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAck;
  context.recoveryCq = &scene.verbs[0].cq;
  ibv_wc ack{};
  ack.status = IBV_WC_SUCCESS;
  ack.opcode = IBV_WC_RECV;
  ack.imm_data = NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID;
  g_ibvCompletions.push_back(ack);
  ncclIbPortRecoveryStateProgressResult result = ncclIbPortRecoveryStateProgressResultFailed;

  ASSERT_EQ(ncclIbPortRecoveryProgressAckSender(&context, &result), ncclSuccess);
  EXPECT_TRUE(context.ackReceived);
  EXPECT_TRUE(context.ackPosted);
  EXPECT_EQ(result, ncclIbPortRecoveryStateProgressResultInProgress);

  ibv_wc finalAck{};
  finalAck.status = IBV_WC_SUCCESS;
  finalAck.opcode = IBV_WC_SEND;
  g_ibvCompletions.push_back(finalAck);
  ASSERT_EQ(ncclIbPortRecoveryProgressAckSender(&context, &result), ncclSuccess);
  EXPECT_TRUE(context.ackCompleted);
  EXPECT_EQ(result, ncclIbPortRecoveryStateProgressResultGoToNextState);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_SenderCompletesAliveAndAckPhases) {
  RecoveryScene scene;
  scene.send.base.nqps = 0;
  scene.sendRes.devs[0].state.store(ncclIbResiliencyDevStateRecoveryInProgress, std::memory_order_release);
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.recoveryCq = &scene.verbs[0].cq;
  context.send.aliveMsgPosted = true;
  ibv_wc alive{};
  alive.status = IBV_WC_SUCCESS;
  alive.opcode = IBV_WC_SEND;
  ibv_wc remoteAck{};
  remoteAck.status = IBV_WC_SUCCESS;
  remoteAck.opcode = IBV_WC_RECV;
  remoteAck.imm_data = NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID;
  ibv_wc finalAck{};
  finalAck.status = IBV_WC_SUCCESS;
  finalAck.opcode = IBV_WC_SEND;
  g_ibvCompletions = {alive, remoteAck, finalAck};
  bool done = false;

  ASSERT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_TRUE(done);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateSuccess);
  EXPECT_TRUE(context.send.aliveMsgCompleted);
  EXPECT_TRUE(context.ackReceived);
  EXPECT_TRUE(context.ackCompleted);
  EXPECT_EQ(scene.sendRes.devs[0].state.load(), ncclIbResiliencyDevStateRecovered);
  ASSERT_EQ(g_ibvPostSendQps.size(), 1u);
  EXPECT_EQ(g_ibvPostSendQps[0], scene.sendRes.portRecoveryQps[0].qp);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_ReceiverCompletesAliveAndAckPhases) {
  RecoveryScene scene;
  scene.recv.base.nqps = 0;
  scene.recvRes.devs[0].state.store(ncclIbResiliencyDevStateRecoveryInProgress, std::memory_order_release);
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.recoveryCq = &scene.verbs[0].cq;
  context.recv.nInOrderMsgsReceived = 2;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0 ? int64_t{1} : fallback;
  });
  bool done = false;

  ASSERT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_FALSE(done);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateAliveMessages);
  EXPECT_TRUE(context.ackPosted);

  ibv_wc localAck{};
  localAck.status = IBV_WC_SUCCESS;
  localAck.opcode = IBV_WC_SEND;
  ibv_wc finalAck{};
  finalAck.status = IBV_WC_SUCCESS;
  finalAck.opcode = IBV_WC_RECV;
  finalAck.imm_data = NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID;
  g_ibvCompletions = {localAck, finalAck};
  ASSERT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_TRUE(done);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateSuccess);
  EXPECT_TRUE(context.ackCompleted);
  EXPECT_TRUE(context.ackReceived);
  EXPECT_EQ(scene.recvRes.devs[0].state.load(), ncclIbResiliencyDevStateRecovered);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_AckTimeoutRetriesThenFailsAtLimit) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAck;
  context.recoveryCq = &scene.verbs[0].cq;
  g_netIbNowNs = 100;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ACK_TIMEOUT") == 0) return int64_t{0};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ATTEMPTS_MAX") == 0) return int64_t{2};
    return fallback;
  });
  bool done = false;

  ASSERT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_FALSE(done);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateAliveMessages);
  EXPECT_EQ(context.nFailedAttempts, 1);

  context.state = ncclIbPortRecoveryStateAck;
  ASSERT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_TRUE(done);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateFailed);
  EXPECT_EQ(context.nFailedAttempts, 2);
  EXPECT_EQ(scene.sendRes.devs[0].state.load(), ncclIbResiliencyDevStateRecoveryFailed);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, HandleFailure_QueuesInitializedRecoveryContext) {
  RecoveryScene scene;
  scene.sendRes.recoveryEnabled = true;
  scene.sendRes.devs[0].state.store(ncclIbResiliencyDevStateRecoveryInProgress, std::memory_order_release);

  ASSERT_EQ(NetIbRecoveryHandleFailure(&scene.sendRes, 0), ncclSuccess);
  {
    std::lock_guard<std::mutex> lock(ncclIbPortRecoveryMutex);
    ASSERT_EQ(recoveryInbox.size(), 1u);
    auto* context = recoveryInbox.front();
    recoveryInbox.pop_front();
    EXPECT_EQ(context->resCtx, &scene.sendRes);
    EXPECT_EQ(context->devIndex, 0);
    EXPECT_EQ(context->state, ncclIbPortRecoveryStateInit);
    EXPECT_EQ(context->recoveryCq, &scene.verbs[0].cq);
    EXPECT_EQ(ncclIbPortRecoveryContextDestroy(context), ncclSuccess);
  }
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_BeforeStartDelayRemainsPending) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateInit;
  context.timeInit = 1000;
  g_netIbNowNs = 1000 + 50 * MSEC_TO_NSEC;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_START_DELAY") == 0 ? int64_t{100} : fallback;
  });
  bool done = true;

  EXPECT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_FALSE(done);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateInit);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_ReconfigureFailureTransitionsToFailedAndDone) {
  RecoveryScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateInit;
  g_netIbNowNs = MSEC_TO_NSEC;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_START_DELAY") == 0 ? int64_t{0} : fallback;
  });
  ScopedHook reconfigure(g_ncclIbResiliencyQpsReconfigure, [](ncclIbResiliency*, int, bool* success) {
    *success = false;
    return ncclSuccess;
  });
  bool done = false;

  EXPECT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_TRUE(done);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateFailed);
  EXPECT_EQ(scene.sendRes.devs[0].state.load(), ncclIbResiliencyDevStateRecoveryFailed);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, Init_DisabledLeavesRecoveryOffWithoutStartingThread) {
  RecoveryScene scene;
  scene.sendRes.recoveryEnabled = true;

  EXPECT_EQ(NetIbRecoveryInit(&scene.sendRes), ncclSuccess);
  EXPECT_FALSE(scene.sendRes.recoveryEnabled);
  EXPECT_FALSE(ncclIbPortRecoveryThreadActive.load());
  EXPECT_FALSE(NetIbRecoveryAsyncThread.joinable());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, Init_ActiveRecoveryThreadEnablesRecovery) {
  RecoveryScene scene;
  ncclIbPortRecoveryThreadActive.store(true);

  ncclResult_t result = NetIbRecoveryInit(&scene.recvRes);
  ncclIbPortRecoveryThreadActive.store(false);

  EXPECT_EQ(result, ncclSuccess);
  EXPECT_TRUE(scene.recvRes.recoveryEnabled);
}

}  // namespace
