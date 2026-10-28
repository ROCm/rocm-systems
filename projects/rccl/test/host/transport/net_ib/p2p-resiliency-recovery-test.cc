/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/p2p_resiliency_recovery.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <vector>

#include "IbVerbsScene.h"
#include "ScopedHook.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/transport_net_ib_recovery_fakes.h"
#include "fakes/transport_net_ib_resiliency_fakes.h"
#include "transport/net_ib/NetIbGlobals.h"
#include "transport/net_ib/NetIbQpSeams.h"
#include "transport/net_ib/NetIbResiliencyScene.h"

namespace {

std::vector<ncclIbQp*> g_recoveryErroredQps;
std::vector<ncclIbQp*> g_recoveryDataRecvQps;

ncclResult_t TestRecoveryQpError(ncclIbQp* qp) {
  g_recoveryErroredQps.push_back(qp);
  return ncclSuccess;
}
ncclResult_t TestRecoveryPostDataReceives(ncclIbRecvComm*, ncclIbQp* qp) {
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

class NetIbP2pResiliencyRecoveryMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    {
      // Leftover contexts reference a destroyed NetIbResiliencyScene, and
      // ncclIbPortRecoveryContextDestroy dereferences resCtx, so free them directly.
      std::lock_guard<std::mutex> lock(ncclIbPortRecoveryMutex);
      for (auto* context : recoveryInbox) free(context);
      recoveryInbox.clear();
    }
    NetIbResetGlobals();
    g_recoveryErroredQps.clear();
    g_recoveryDataRecvQps.clear();
    ResetNetIbQpSeams();
    ResetTransportNetIbResiliencyFakes();
    ResetTransportNetIbRecoveryFakes();
    ResetIbvwrapFakes();
    ResetNcclFakes();
  }
};

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsToError_TransitionsOnlyFailedDeviceQps) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  context.devIndex = 1;

  EXPECT_EQ(ncclIbPortRecoveryQpsToError(&context), ncclSuccess);
  EXPECT_EQ(g_recoveryErroredQps, (std::vector<ncclIbQp*>{&scene.send.base.qps[1], &scene.send.base.qps[3]}));
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsReconfigure_SenderPostsOneReceiveAndReportsSuccess) {
  NetIbResiliencyScene scene;
  ScopedHook reconfigure(g_ncclIbResiliencyQpReconfigure,
                         [&](ncclIbResiliency*, ncclIbQp* qp, ncclIbNetCommDevBase*, int, bool* success) {
                           EXPECT_EQ(qp, &scene.sendRes.base.portRecoveryQps[0]);
                           *success = true;
                           return ncclSuccess;
                         });
  bool success = false;

  ASSERT_EQ(NetIbRecoveryQpsReconfigure(&scene.sendRes.base, 0, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(reconfigure.calls, 1);
  ASSERT_EQ(scene.verbs[0].postRecvQps.size(), 1u);
  EXPECT_EQ(scene.verbs[0].postRecvQps[0], scene.sendRes.base.portRecoveryQps[0].qp);
  EXPECT_TRUE(scene.verbs[1].postRecvQps.empty());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsReconfigure_ReceiverPostsTenReceives) {
  NetIbResiliencyScene scene;
  ScopedHook reconfigure(g_ncclIbResiliencyQpReconfigure,
                         [&](ncclIbResiliency*, ncclIbQp* qp, ncclIbNetCommDevBase*, int, bool* success) {
                           EXPECT_EQ(qp, &scene.recvRes.portRecoveryQps[1]);
                           *success = true;
                           return ncclSuccess;
                         });
  bool success = false;

  ASSERT_EQ(NetIbRecoveryQpsReconfigure(&scene.recvRes, 1, &success), ncclSuccess);
  EXPECT_TRUE(success);
  ASSERT_EQ(scene.verbs[1].postRecvQps.size(), NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
  for (auto* qp : scene.verbs[1].postRecvQps) EXPECT_EQ(qp, scene.recvRes.portRecoveryQps[1].qp);
  EXPECT_TRUE(scene.verbs[0].postRecvQps.empty());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsReconfigure_FailedQpOrReceivePostReportsFailure) {
  for (bool qpFails : {true, false}) {
    NetIbResiliencyScene scene;
    if (!qpFails) scene.verbs[0].postRecvErr = EIO;
    ScopedHook reconfigure(g_ncclIbResiliencyQpReconfigure,
                           [qpFails](ncclIbResiliency*, ncclIbQp*, ncclIbNetCommDevBase*, int, bool* success) {
                             *success = !qpFails;
                             return ncclSuccess;
                           });
    bool success = true;

    ASSERT_EQ(NetIbRecoveryQpsReconfigure(&scene.sendRes.base, 0, &success), ncclSuccess) << qpFails;
    EXPECT_FALSE(success) << qpFails;
    EXPECT_EQ(scene.verbs[0].postRecvQps.size(), qpFails ? 0u : 1u) << qpFails;
  }
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DrainCq_RepostsOneReceiveForEveryDrainedCompletion) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.recoveryCq = &scene.verbs[0].cq;
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV;
  scene.verbs[0].wcs.push_back(completion);
  scene.verbs[0].wcs.push_back(completion);
  bool success = false;

  EXPECT_EQ(ncclIbPortRecoveryDrainCqAndPostReceiveWRs(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(scene.verbs[0].postRecvQps.size(), 2u);
  ASSERT_FALSE(scene.verbs[0].polledCqs.empty());
  EXPECT_EQ(scene.verbs[0].polledCqs[0], context.recoveryCq);
  for (auto* qp : scene.verbs[0].postRecvQps) EXPECT_EQ(qp, scene.recvRes.portRecoveryQps[0].qp);
}

// AICOMRCCL-3128: SEND completions do not consume receive WQEs and should not
// cause the recovery QP to over-post its receive queue.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_DrainCq_SendCompletionDoesNotPostReceive) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.recoveryCq = &scene.verbs[0].cq;
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_SEND;
  scene.verbs[0].wcs.push_back(completion);
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryDrainCqAndPostReceiveWRs(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_TRUE(scene.verbs[0].postRecvQps.empty());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextInit_SenderInitializesStateAndPostsAckReceive) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext* context = nullptr;
  g_netIbNowNs = 5000;

  ASSERT_EQ(ncclIbPortRecoveryContextInit(&scene.sendRes.base, 0, &context), ncclSuccess);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->resCtx, &scene.sendRes.base);
  EXPECT_EQ(context->devIndex, 0);
  EXPECT_EQ(context->state, ncclIbPortRecoveryStateInit);
  EXPECT_EQ(context->timeInit, 5000u);
  EXPECT_EQ(g_recoveryErroredQps, (std::vector<ncclIbQp*>{&scene.send.base.qps[0], &scene.send.base.qps[2]}));
  ASSERT_EQ(scene.verbs[0].postRecvQps.size(), 1u);
  EXPECT_EQ(scene.verbs[0].postRecvQps[0], scene.sendRes.base.portRecoveryQps[0].qp);
  EXPECT_EQ(ncclIbPortRecoveryContextDestroy(context), ncclSuccess);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextInit_ReceiverPostsAliveAndFinalAckReceives) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext* context = nullptr;

  ASSERT_EQ(ncclIbPortRecoveryContextInit(&scene.recvRes, 1, &context), ncclSuccess);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->recoveryCq, &scene.verbs[1].cq);
  EXPECT_EQ(scene.verbs[1].postRecvQps.size(), NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
  for (auto* qp : scene.verbs[1].postRecvQps) EXPECT_EQ(qp, scene.recvRes.portRecoveryQps[1].qp);
  EXPECT_EQ(ncclIbPortRecoveryContextDestroy(context), ncclSuccess);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextInit_PostFailureReturnsErrorWithoutPublishingContext) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext* context = reinterpret_cast<ncclIbPortRecoveryContext*>(0x1);
  scene.verbs[0].postRecvErr = EIO;

  EXPECT_EQ(ncclIbPortRecoveryContextInit(&scene.sendRes.base, 0, &context), ncclSystemError);
  EXPECT_EQ(context, nullptr);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DevInitAndDestroyManageRecoveryCq) {
  NetIbResiliencyScene scene;
  ncclIbDev dev{};
  dev.context = &scene.verbs[0].ctx;
  ibv_cq cq{};
  ScopedHook create(g_ibvCreateCq,
                    [&](ibv_cq** out, ibv_context* context, int size, void*, ibv_comp_channel*, int) {
                      EXPECT_EQ(context, &scene.verbs[0].ctx);
                      EXPECT_EQ(size, NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE);
                      *out = &cq;
                      return ncclSuccess;
                    });
  ScopedHook destroy(g_ibvDestroyCq, [&cq](ibv_cq* seen) {
    EXPECT_EQ(seen, &cq);
    return ncclSuccess;
  });
  scene.sendRes.base.recoveryEnabled = true;

  ASSERT_EQ(NetIbRecoveryDevInit(&scene.sendRes.base, 0, &dev), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.devs[0].portRecoveryCq, &cq);
  EXPECT_EQ(NetIbRecoveryDevDestroy(&scene.sendRes.base, 0), ncclSuccess);
  EXPECT_EQ(create.calls, 1);
  EXPECT_EQ(destroy.calls, 1);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsDestroy_DestroysOnlyMaterializedQps) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.portRecoveryQps[1].qp = nullptr;
  ScopedHook destroy(g_ibvDestroyQp, [&](ibv_qp* qp) {
    EXPECT_EQ(qp, scene.sendRes.base.portRecoveryQps[0].qp);
    return ncclSuccess;
  });

  EXPECT_EQ(NetIbRecoveryQpsDestroy(&scene.sendRes.base, 2), ncclSuccess);
  EXPECT_EQ(destroy.calls, 1);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, SenderRecoveryQpsCreationPublishesStripedMetadata) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.recoveryEnabled = true;
  for (int d = 0; d < 2; ++d) ncclIbDevs[d].portNum = d + 1;
  g_netIbQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr* attr) {
    EXPECT_EQ(attr->type, IBV_QPT_UC);
    return scene.CreateQp(qp, attr);
  };
  std::vector<ncclIbQp*> initialized;
  g_netIbQpInit = [&](ncclIbQp* qp) {
    initialized.push_back(qp);
    return ncclSuccess;
  };
  ncclIbQpInfo info[2] = {};

  ASSERT_EQ(NetIbRecoverySenderQpsCreate(&scene.sendRes.base, info, 2), ncclSuccess);
  for (int d = 0; d < 2; ++d) {
    const ncclIbQp& qp = scene.sendRes.base.portRecoveryQps[d];
    EXPECT_EQ(qp.devIndex, d);
    EXPECT_EQ(qp.qp->context, &scene.verbs[d].ctx);
    EXPECT_EQ(qp.initAttr.portNum, d + 1);
    EXPECT_EQ(qp.initAttr.qpAccessFlags, IBV_ACCESS_LOCAL_WRITE);
    EXPECT_EQ(info[d].qpn, qp.qp->qp_num);
    EXPECT_EQ(info[d].devIndex, d);
  }
  EXPECT_EQ(initialized, (std::vector<ncclIbQp*>{&scene.sendRes.base.portRecoveryQps[0],
                                                &scene.sendRes.base.portRecoveryQps[1]}));
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, SenderRecoveryQpsToRtsUsesRemoteMetadata) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.recoveryEnabled = true;
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
    EXPECT_EQ(qp, &scene.sendRes.base.portRecoveryQps[0]);
    transitions.push_back('T');
    return ncclSuccess;
  };
  g_netIbQpRts = [&](ncclIbQp* qp) {
    EXPECT_EQ(qp, &scene.sendRes.base.portRecoveryQps[0]);
    transitions.push_back('S');
    return ncclSuccess;
  };

  ASSERT_EQ(NetIbRecoverySenderQpsToRts(&scene.sendRes.base, &remote, 1), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.portRecoveryQps[0].rtrAttr.remoteQpNum, 200u);
  EXPECT_EQ(scene.sendRes.base.portRecoveryQps[0].rtrAttr.mtu, IBV_MTU_2048);
  EXPECT_EQ(transitions, (std::vector<char>{'T', 'S'}));
}

// AICOMRCCL-3129: the recovery AV port is local and must not be copied from
// the remote device metadata.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_SenderRecoveryQpsToRtsUsesLocalIbPort) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.recoveryEnabled = true;
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

  ASSERT_EQ(NetIbRecoverySenderQpsToRts(&scene.sendRes.base, &remote, 1), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.portRecoveryQps[0].rtrAttr.localIbPort, 3);
}

// AICOMRCCL-3129: receiver recovery QPs have the same local AV-port contract.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_ReceiverRecoveryQpsToRtsUsesLocalIbPort) {
  NetIbResiliencyScene scene;
  ncclIbDevs[0].portNum = 3;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  g_netIbQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr* attr) { return scene.CreateQp(qp, attr); };
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
  NetIbResiliencyScene scene;
  ncclIbDevs[0].portNum = 1;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  g_netIbQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr* attr) { return scene.CreateQp(qp, attr); };
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
  const ncclIbQp& qp = scene.recvRes.portRecoveryQps[0];
  EXPECT_EQ(transitions, (std::vector<char>{'I', 'T', 'S'}));
  EXPECT_EQ(qp.qp->context, &scene.verbs[0].ctx);
  EXPECT_EQ(qp.initAttr.portNum, 1);
  EXPECT_EQ(qp.initAttr.qpAccessFlags, IBV_ACCESS_LOCAL_WRITE);
  EXPECT_EQ(local[0].qpn, qp.qp->qp_num);
  EXPECT_EQ(qp.rtrAttr.remoteQpNum, 300u);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsRestore_ReconfiguresOnlyFailedDeviceDataQps) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  context.devIndex = 0;
  std::vector<ncclIbQp*> restored;
  ScopedHook reconfigure(
      g_ncclIbResiliencyQpReconfigure,
      [&](ncclIbResiliency*, ncclIbQp* qp, ncclIbNetCommDevBase* devBase, int devIndex, bool* success) {
        EXPECT_EQ(devBase, &scene.send.devs[0].base);
        EXPECT_EQ(devIndex, 0);
        restored.push_back(qp);
        *success = true;
        return ncclSuccess;
      });
  // Callers currently pass success=true from the preceding CQ operation.
  // AICOMRCCL-3128 tracks making this a self-contained output contract.
  bool success = true;

  EXPECT_EQ(ncclIbPortRecoveryQpsRestore(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(restored, (std::vector<ncclIbQp*>{&scene.send.base.qps[0], &scene.send.base.qps[2]}));
  EXPECT_TRUE(g_recoveryDataRecvQps.empty());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsRestore_ReceiverRepostsReceivesOnEachRestoredDataQp) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  ScopedHook reconfigure(g_ncclIbResiliencyQpReconfigure,
                         [](ncclIbResiliency*, ncclIbQp*, ncclIbNetCommDevBase*, int, bool* success) {
                           *success = true;
                           return ncclSuccess;
                         });
  bool success = true;

  EXPECT_EQ(ncclIbPortRecoveryQpsRestore(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(reconfigure.calls, 2);
  EXPECT_EQ(g_recoveryDataRecvQps, (std::vector<ncclIbQp*>{&scene.recv.base.qps[0], &scene.recv.base.qps[2]}));
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsRestore_ReceiverRestoresRecoveredDeviceFlushQpAsLoopback) {
  NetIbResiliencyScene scene;
  scene.recv.base.nqps = 0;
  scene.recv.flushEnabled = 1;
  scene.recv.flushQpsCreated = true;
  for (int d = 0; d < 2; ++d) scene.recv.devs[d].base.gidInfo.localGid.global.interface_id = 0x10 + d;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 1;
  std::vector<ncclIbQp*> restored;
  ScopedHook reconfigure(
      g_ncclIbResiliencyQpReconfigure,
      [&](ncclIbResiliency*, ncclIbQp* qp, ncclIbNetCommDevBase* devBase, int devIndex, bool* success) {
        EXPECT_EQ(devBase, &scene.recv.devs[1].base);
        EXPECT_EQ(devIndex, 1);
        EXPECT_EQ(qp->rtrAttr.remoteGid.global.interface_id, 0x11u);
        restored.push_back(qp);
        *success = true;
        return ncclSuccess;
      });
  bool success = true;

  EXPECT_EQ(ncclIbPortRecoveryQpsRestore(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(restored, (std::vector<ncclIbQp*>{&scene.recv.devs[1].gpuFlush.qp}));
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, QpsRestore_ReconfigureFailureReportsFailureAndStops) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
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
  NetIbResiliencyScene scene;
  scene.send.base.nqps = 0;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  context.devIndex = 0;
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryQpsRestore(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, CompletionReceiver_InOrderAliveMessageAdvancesAndReposts) {
  NetIbResiliencyScene scene;
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
  EXPECT_EQ(scene.verbs[0].postRecvQps, (std::vector<ibv_qp*>{scene.recvRes.portRecoveryQps[0].qp}));
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, CompletionReceiver_OutOfOrderAliveMessageRestartsSequence) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.aliveMsgNextId = 7;
  context.recv.nInOrderMsgsReceived = 2;
  ibv_wc completion{};
  completion.opcode = IBV_WC_RECV;
  completion.imm_data = 9;
  bool success = false;

  EXPECT_EQ(ncclIbPortRecoveryHandleCompletionReceiver(&context, completion, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(context.recv.nInOrderMsgsReceived, 0);
  EXPECT_EQ(context.aliveMsgNextId, 10u);
  EXPECT_EQ(scene.verbs[0].postRecvQps.size(), 1u);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, CompletionSender_AliveSendAndAckMessagesUpdateState) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
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

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, CompletionSender_DuplicateAliveBatchCompletionFails) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  context.state = ncclIbPortRecoveryStateAliveMessages;
  context.send.aliveMsgCompleted = true;
  ibv_wc completion{};
  completion.opcode = IBV_WC_SEND;
  bool success = true;

  ASSERT_EQ(ncclIbPortRecoveryHandleCompletionSender(&context, completion, &success), ncclSuccess);
  EXPECT_FALSE(success);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, PollCq_ErrorStatusReportsFailure) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  context.devIndex = 0;
  context.recoveryCq = &scene.verbs[0].cq;
  ibv_wc completion{};
  completion.status = IBV_WC_REM_ACCESS_ERR;
  scene.verbs[0].wcs.push_back(completion);
  bool success = true;

  EXPECT_EQ(ncclIbPortRecoveryPollCq(&context, &success), ncclSuccess);
  EXPECT_FALSE(success);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, PostAliveMessages_SequencePostsOnlyFinalMessageSignaled) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  context.devIndex = 0;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE") == 0) return int64_t{3};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0) return int64_t{3};
    return fallback;
  });
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryPostAliveMessages(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  ASSERT_EQ(scene.verbs[0].posted.size(), 3u);
  EXPECT_EQ(scene.verbs[0].posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(scene.verbs[0].posted[1].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_NE(scene.verbs[0].posted[2].sendFlags & IBV_SEND_SIGNALED, 0);
  for (size_t i = 0; i < scene.verbs[0].posted.size(); ++i) {
    EXPECT_EQ(scene.verbs[0].posted[i].opcode, IBV_WR_SEND_WITH_IMM);
    EXPECT_EQ(scene.verbs[0].posted[i].wrId, i);
    EXPECT_EQ(scene.verbs[0].posted[i].immData, i);
  }
  ASSERT_EQ(scene.verbs[0].postSendQps.size(), 1u);
  EXPECT_EQ(scene.verbs[0].postSendQps[0], scene.sendRes.base.portRecoveryQps[0].qp);
  EXPECT_EQ(context.aliveMsgNextId, 3u);
  EXPECT_TRUE(context.send.aliveMsgPosted);
}

// AICOMRCCL-3128: one batch should be controlled by ALIVE_MSG_BATCH_SIZE;
// ALIVE_MSG_SEQUENCE_SIZE controls how many messages establish liveness.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_PostAliveMessages_UsesBatchSizeIndependently) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  context.devIndex = 0;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE") == 0) return int64_t{2};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0) return int64_t{5};
    return fallback;
  });
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryPostAliveMessages(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(scene.verbs[0].posted.size(), 2u);
}

// AICOMRCCL-3128: a zero-sized batch must be rejected before wr[0] is posted.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest, DISABLED_PostAliveMessages_ZeroBatchIsRejectedWithoutPosting) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  context.devIndex = 0;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE") == 0) return int64_t{0};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0) return int64_t{5};
    return fallback;
  });
  bool success = false;

  EXPECT_NE(ncclIbPortRecoveryPostAliveMessages(&context, &success), ncclSuccess);
  EXPECT_TRUE(scene.verbs[0].posted.empty());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, PostAliveMessages_AboveMaximumIsRejectedWithoutPosting) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE") == 0)
      return int64_t{NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX + 1};
    if (std::strcmp(name, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE") == 0)
      return int64_t{NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX + 1};
    return fallback;
  });
  bool success = false;

  EXPECT_EQ(ncclIbPortRecoveryPostAliveMessages(&context, &success), ncclInternalError);
  EXPECT_TRUE(scene.verbs[0].posted.empty());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, PostAck_PublishesSignaledAckAndMarksPosted) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  bool success = false;

  ASSERT_EQ(ncclIbPortRecoveryPostAck(&context, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_TRUE(context.ackPosted);
  ASSERT_EQ(scene.verbs[0].posted.size(), 1u);
  EXPECT_EQ(scene.verbs[0].posted[0].immData, NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
  EXPECT_NE(scene.verbs[0].posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
  ASSERT_EQ(scene.verbs[0].postSendQps.size(), 1u);
  EXPECT_EQ(scene.verbs[0].postSendQps[0], scene.recvRes.portRecoveryQps[0].qp);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, AliveProgress_SenderPostsThenCompletesBatch) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
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
  scene.verbs[0].wcs.push_back(completion);
  ASSERT_EQ(ncclIbPortRecoveryProgressAliveMessagesSender(&context, &result), ncclSuccess);
  EXPECT_EQ(result, ncclIbPortRecoveryStateProgressResultGoToNextState);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, AliveProgress_ReceiverBelowSequenceKeepsWaiting) {
  NetIbResiliencyScene scene;
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
  NetIbResiliencyScene scene;
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
  EXPECT_TRUE(scene.verbs[0].postRecvQps.empty());
}

// AICOMRCCL-3128: exact configured sequence currently requires one extra
// message because production uses '>' instead of '>='.
TEST_F(NetIbP2pResiliencyRecoveryMicrotest,
       DISABLED_AliveProgress_ReceiverExactSequenceRestoresAndPostsAck) {
  NetIbResiliencyScene scene;
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

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, AckProgress_SenderWaitsInProgressForFinalAckCompletion) {
  NetIbResiliencyScene scene;
  scene.send.base.nqps = 0;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAck;
  context.recoveryCq = &scene.verbs[0].cq;
  ibv_wc ack{};
  ack.status = IBV_WC_SUCCESS;
  ack.opcode = IBV_WC_RECV;
  ack.imm_data = NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID;
  scene.verbs[0].wcs.push_back(ack);
  ncclIbPortRecoveryStateProgressResult result = ncclIbPortRecoveryStateProgressResultFailed;

  ASSERT_EQ(ncclIbPortRecoveryProgressAckSender(&context, &result), ncclSuccess);
  EXPECT_TRUE(context.ackReceived);
  EXPECT_TRUE(context.ackPosted);
  EXPECT_FALSE(context.ackCompleted);
  EXPECT_EQ(result, ncclIbPortRecoveryStateProgressResultInProgress);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_SenderCompletesAliveAndAckPhases) {
  NetIbResiliencyScene scene;
  scene.send.base.nqps = 0;
  scene.sendRes.base.devs[0].state.store(ncclIbResiliencyDevStateRecoveryInProgress, std::memory_order_release);
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
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
  scene.verbs[0].wcs = {alive, remoteAck, finalAck};
  bool done = false;

  ASSERT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_TRUE(done);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateSuccess);
  EXPECT_TRUE(context.send.aliveMsgCompleted);
  EXPECT_TRUE(context.ackReceived);
  EXPECT_TRUE(context.ackCompleted);
  EXPECT_EQ(scene.sendRes.base.devs[0].state.load(), ncclIbResiliencyDevStateRecovered);
  ASSERT_EQ(scene.verbs[0].postSendQps.size(), 1u);
  EXPECT_EQ(scene.verbs[0].postSendQps[0], scene.sendRes.base.portRecoveryQps[0].qp);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_ReceiverCompletesAliveAndAckPhases) {
  NetIbResiliencyScene scene;
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
  scene.verbs[0].wcs = {localAck, finalAck};
  ASSERT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_TRUE(done);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateSuccess);
  EXPECT_TRUE(context.ackCompleted);
  EXPECT_TRUE(context.ackReceived);
  EXPECT_EQ(scene.recvRes.devs[0].state.load(), ncclIbResiliencyDevStateRecovered);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_AckTimeoutRetriesThenFailsAtLimit) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
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
  EXPECT_EQ(scene.sendRes.base.devs[0].state.load(), ncclIbResiliencyDevStateRecoveryFailed);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_ReceiverFinalAckTimeoutRestartsAliveSequence) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.recvRes;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateAck;
  context.recoveryCq = &scene.verbs[0].cq;
  context.ackPosted = true;
  context.ackCompleted = true;
  context.recv.nInOrderMsgsReceived = 6;
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
  EXPECT_FALSE(context.ackPosted);
  EXPECT_FALSE(context.ackCompleted);
  EXPECT_EQ(context.recv.nInOrderMsgsReceived, 0);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, HandleFailure_QueuesInitializedRecoveryContext) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.recoveryEnabled = true;
  scene.sendRes.base.devs[0].state.store(ncclIbResiliencyDevStateRecoveryInProgress, std::memory_order_release);

  ASSERT_EQ(NetIbRecoveryHandleFailure(&scene.sendRes.base, 0), ncclSuccess);
  {
    std::lock_guard<std::mutex> lock(ncclIbPortRecoveryMutex);
    ASSERT_EQ(recoveryInbox.size(), 1u);
    auto* context = recoveryInbox.front();
    recoveryInbox.pop_front();
    EXPECT_EQ(context->resCtx, &scene.sendRes.base);
    EXPECT_EQ(context->devIndex, 0);
    EXPECT_EQ(context->state, ncclIbPortRecoveryStateInit);
    EXPECT_EQ(context->recoveryCq, &scene.verbs[0].cq);
    EXPECT_EQ(ncclIbPortRecoveryContextDestroy(context), ncclSuccess);
  }
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, HandleCloseRequests_DestroysOnlyTheClosingContextsQueueEntries) {
  NetIbResiliencyScene scene;
  auto newContext = [](ncclIbResiliency* resiliency) {
    auto* context = static_cast<ncclIbPortRecoveryContext*>(calloc(1, sizeof(ncclIbPortRecoveryContext)));
    context->resCtx = resiliency;
    return context;
  };
  ncclIbPortRecoveryContext* kept = newContext(&scene.recvRes);
  std::list<ncclIbPortRecoveryContext*> queue{newContext(&scene.sendRes.base), kept,
                                              newContext(&scene.sendRes.base)};
  ncclIbPortRecoveryCloseRequest closing{&scene.sendRes.base, false};
  std::vector<ncclIbPortRecoveryCloseRequest*> closeRequests{&closing};

  ncclIbPortRecoveryHandleCloseRequests(closeRequests, queue);
  EXPECT_EQ(queue, (std::list<ncclIbPortRecoveryContext*>{kept}));
  for (auto* context : queue) free(context);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_BeforeStartDelayRemainsPending) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
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
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
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
  EXPECT_EQ(scene.sendRes.base.devs[0].state.load(), ncclIbResiliencyDevStateRecoveryFailed);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, ContextProgress_ReconfiguredSenderStartsAlivePhaseAndPostsBatch) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryContext context{};
  context.resCtx = &scene.sendRes.base;
  context.devIndex = 0;
  context.state = ncclIbPortRecoveryStateInit;
  context.recoveryCq = &scene.verbs[0].cq;
  // Past the 200 ms start delay and the 500 ms alive-batch interval.
  g_netIbNowNs = 1000 * MSEC_TO_NSEC;
  ScopedHook reconfigure(g_ncclIbResiliencyQpsReconfigure, [](ncclIbResiliency*, int devIndex, bool* success) {
    EXPECT_EQ(devIndex, 0);
    *success = true;
    return ncclSuccess;
  });
  bool done = false;

  ASSERT_EQ(ncclIbPortRecoveryContextProgress(&context, &done), ncclSuccess);
  EXPECT_FALSE(done);
  EXPECT_EQ(reconfigure.calls, 1);
  EXPECT_EQ(context.state, ncclIbPortRecoveryStateAliveMessages);
  EXPECT_TRUE(context.send.aliveMsgPosted);
  EXPECT_EQ(scene.verbs[0].postSendQps, (std::vector<ibv_qp*>{scene.sendRes.base.portRecoveryQps[0].qp}));
  EXPECT_EQ(scene.verbs[0].posted.size(), 5u);
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, Init_DisabledLeavesRecoveryOffWithoutStartingThread) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.recoveryEnabled = true;

  EXPECT_EQ(NetIbRecoveryInit(&scene.sendRes.base), ncclSuccess);
  EXPECT_FALSE(scene.sendRes.base.recoveryEnabled);
  EXPECT_FALSE(ncclIbPortRecoveryThreadActive.load());
  EXPECT_FALSE(NetIbRecoveryAsyncThread.joinable());
}

TEST_F(NetIbP2pResiliencyRecoveryMicrotest, Init_ActiveRecoveryThreadEnablesRecovery) {
  NetIbResiliencyScene scene;
  ncclIbPortRecoveryThreadActive.store(true);

  ncclResult_t result = NetIbRecoveryInit(&scene.recvRes);
  ncclIbPortRecoveryThreadActive.store(false);

  EXPECT_EQ(result, ncclSuccess);
  EXPECT_TRUE(scene.recvRes.recoveryEnabled);
}

}  // namespace
