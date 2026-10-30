/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/p2p_resiliency.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <string>
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
#include "transport/net_ib/connect.h"
#include "transport/net_ib/p2p.h"

namespace {

int g_resiliencyMultiSendCalls = 0;
int g_resiliencyPostFifoCalls = 0;
int g_resiliencyAddEventCalls = 0;
std::vector<size_t> g_resiliencyIbMallocSizes;
bool g_resiliencyIbMallocPoison = false;

ncclResult_t TestResiliencyIbMalloc(void** ptr, size_t size) {
  g_resiliencyIbMallocSizes.push_back(size);
  ncclResult_t result = ncclIbMallocDebug(ptr, size, __FILE__, __LINE__, __func__);
  if (result == ncclSuccess && g_resiliencyIbMallocPoison) std::memset(*ptr, 0x01, size);
  return result;
}

template <typename T>
bool AllBytesZero(const T& value) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  return std::all_of(bytes, bytes + sizeof(T), [](unsigned char byte) { return byte == 0; });
}

ncclResult_t TestResiliencyMultiSend(ncclIbSendComm*, int) {
  ++g_resiliencyMultiSendCalls;
  return ncclSuccess;
}
ncclResult_t TestResiliencyPostFifo(ncclIbRecvComm*, ncclIbRequest*, int) {
  ++g_resiliencyPostFifoCalls;
  return ncclSuccess;
}
void TestResiliencyAddEvent(ncclIbRequest* request, int devIndex) {
  ++g_resiliencyAddEventCalls;
  request->events[devIndex]++;
}
}  // namespace

#define ncclParamIbResiliencyPortFailover NetIbResiliencyParamPortFailover
#define ncclParamIbResiliencyPortFailoverMaxAttempts NetIbResiliencyParamMaxAttempts
#define ncclParamIbResiliencyPortFailoverProbeDelay NetIbResiliencyParamProbeDelay
#define ncclIbResiliencyInit NetIbResiliencyInit
#define ncclIbResiliencyDestroy NetIbResiliencyDestroy
#define ncclIbResiliencyDevInit NetIbResiliencyDevInit
#define ncclIbResiliencyDevDestroy NetIbResiliencyDevDestroy
#define ncclIbResiliencyDataCqSizeGet NetIbResiliencyDataCqSizeGet
#define ncclIbResiliencyDataRqSizeGet NetIbResiliencyDataRqSizeGet
#define ncclIbResiliencyDeviceNumSet NetIbResiliencyDeviceNumSet
#define ncclIbResiliencySenderCreateQps NetIbResiliencySenderCreateQps
#define ncclIbResiliencySenderQpsToRts NetIbResiliencySenderQpsToRts
#define ncclIbResiliencyReceiverQpsCreateToRts NetIbResiliencyReceiverQpsCreateToRts
#define ncclIbResiliencyQpReconfigure NetIbResiliencyQpReconfigure
#define ncclIbResiliencyQpsReconfigure NetIbResiliencyQpsReconfigure
#define ncclIbResiliencyClose NetIbResiliencyClose
#define ncclIbResiliencyRemoteCompletionRecordsSet NetIbResiliencyRemoteCompletionRecordsSet
#define ncclIbResiliencyRequestIsComplete NetIbResiliencyRequestIsComplete
#define ncclIbResiliencyHandleCompletionError NetIbResiliencyHandleCompletionError
#define ncclIbResiliencyProgress NetIbResiliencyProgress
#define clockNano NetIbTestClockNano
#define ncclIbMultiSend TestResiliencyMultiSend
#define ncclIbPostFifo TestResiliencyPostFifo
#define ncclIbAddEvent TestResiliencyAddEvent
#define ncclIbQpCreate NetIbTestQpCreate
#define ncclIbQpReset NetIbTestQpReset
#define ncclIbQpInit NetIbTestQpInit
#define ncclIbQpRtr NetIbTestQpRtr
#define ncclIbQpRts NetIbTestQpRts
#pragma push_macro("ncclIbMalloc")
#undef ncclIbMalloc
#define ncclIbMalloc TestResiliencyIbMalloc
#include NET_IB_P2P_RESILIENCY_CC_PATH
#undef ncclIbMalloc
#pragma pop_macro("ncclIbMalloc")
#undef ncclIbQpRts
#undef ncclIbQpRtr
#undef ncclIbQpInit
#undef ncclIbQpReset
#undef ncclIbQpCreate
#undef ncclIbAddEvent
#undef ncclIbPostFifo
#undef ncclIbMultiSend
#undef clockNano
#undef ncclIbResiliencyProgress
#undef ncclIbResiliencyHandleCompletionError
#undef ncclIbResiliencyRequestIsComplete
#undef ncclIbResiliencyRemoteCompletionRecordsSet
#undef ncclIbResiliencyClose
#undef ncclIbResiliencyQpsReconfigure
#undef ncclIbResiliencyQpReconfigure
#undef ncclIbResiliencyReceiverQpsCreateToRts
#undef ncclIbResiliencySenderQpsToRts
#undef ncclIbResiliencySenderCreateQps
#undef ncclIbResiliencyDeviceNumSet
#undef ncclIbResiliencyDataRqSizeGet
#undef ncclIbResiliencyDataCqSizeGet
#undef ncclIbResiliencyDevDestroy
#undef ncclIbResiliencyDevInit
#undef ncclIbResiliencyDestroy
#undef ncclIbResiliencyInit
#undef ncclParamIbResiliencyPortFailoverProbeDelay
#undef ncclParamIbResiliencyPortFailoverMaxAttempts
#undef ncclParamIbResiliencyPortFailover

namespace {

class NetIbP2pResiliencyMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    NetIbResetGlobals();
    g_resiliencyMultiSendCalls = 0;
    g_resiliencyPostFifoCalls = 0;
    g_resiliencyAddEventCalls = 0;
    ResetNetIbQpSeams();
    g_resiliencyIbMallocSizes.clear();
    g_resiliencyIbMallocPoison = false;
    ResetTransportNetIbRecoveryFakes();
    ResetTransportNetIbResiliencyFakes();
    ResetIbvwrapFakes();
    ResetNcclFakes();
  }
};

TEST_F(NetIbP2pResiliencyMicrotest, ErrorAdmission_FirstRetryOrFlushFailureWithHealthyPeerIsRecoverable) {
  NetIbResiliencyScene scene;
  ibv_wc completion{};
  completion.status = IBV_WC_RETRY_EXC_ERR;

  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&scene.sendRes.base, &completion, 0), ncclSuccess);
  completion.status = IBV_WC_WR_FLUSH_ERR;
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&scene.sendRes.base, &completion, 0), ncclSuccess);
}

TEST_F(NetIbP2pResiliencyMicrotest, ErrorAdmission_UnsupportedStatusOrSecondFailureIsFatal) {
  NetIbResiliencyScene scene;
  ibv_wc completion{};
  completion.status = IBV_WC_REM_ACCESS_ERR;
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&scene.sendRes.base, &completion, 0), ncclRemoteError);

  completion.status = IBV_WC_RETRY_EXC_ERR;
  scene.sendRes.base.devs[1].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&scene.sendRes.base, &completion, 0), ncclRemoteError);
}

TEST_F(NetIbP2pResiliencyMicrotest, ReplaceQps_FailedDeviceMovesToFirstHealthyAlternative) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.devs[0].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);

  ASSERT_EQ(ncclIbResiliencyReplaceQps(&scene.sendRes.base, 0), ncclSuccess);
  EXPECT_EQ(scene.send.base.activeQps[0], &scene.send.base.qps[1]);
  EXPECT_EQ(scene.send.base.activeQps[1], &scene.send.base.qps[1]);
  EXPECT_EQ(scene.send.base.activeQps[2], &scene.send.base.qps[3]);
  EXPECT_EQ(scene.send.base.activeQps[3], &scene.send.base.qps[3]);
}

TEST_F(NetIbP2pResiliencyMicrotest, ReplaceQps_NoHealthyAlternativeReturnsError) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.devs[0].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);
  scene.sendRes.base.devs[1].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);

  EXPECT_EQ(ncclIbResiliencyReplaceQps(&scene.sendRes.base, 0), ncclInternalError);
}

TEST_F(NetIbP2pResiliencyMicrotest, SendRequestTracking_NewGenerationIncrementsAndFreePreservesGeneration) {
  NetIbResiliencyScene scene;
  auto& request = scene.send.base.reqs[3];
  request.base = &scene.send.base;
  request.id = 3;
  request.type = NCCL_NET_IB_REQ_SEND;
  g_netIbNowNs = 9000;

  ASSERT_EQ(ncclIbResiliencySendRequestInit(&scene.sendRes, &request, 1), ncclSuccess);
  auto& tracked = scene.sendRes.failedRequests[3];
  EXPECT_EQ(tracked.request, &request);
  EXPECT_EQ(tracked.id, 4u);
  EXPECT_EQ(tracked.errorInfo.devIndex, 1);
  EXPECT_EQ(tracked.errorInfo.time, 9000u);
  EXPECT_EQ(scene.sendRes.base.outstandingRequests, 1);
  EXPECT_TRUE(scene.sendRes.base.inProgress);

  ASSERT_EQ(ncclIbResiliencySendRequestFree(&scene.sendRes, &tracked), ncclSuccess);
  EXPECT_EQ(tracked.request, nullptr);
  EXPECT_EQ(tracked.id, 4u);
  EXPECT_EQ(scene.sendRes.base.outstandingRequests, 0);
}

TEST_F(NetIbP2pResiliencyMicrotest, SendRequestTracking_DuplicateAndOldGenerationAreHandled) {
  NetIbResiliencyScene scene;
  auto& request = scene.send.base.reqs[5];
  request.base = &scene.send.base;
  request.id = 5;
  request.type = NCCL_NET_IB_REQ_SEND;
  auto& tracked = scene.sendRes.failedRequests[5];
  tracked.id = 6;
  tracked.request = &request;

  EXPECT_EQ(ncclIbResiliencySendRequestInit(&scene.sendRes, &request, 0), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.outstandingRequests, 0);
  tracked.request = nullptr;
  EXPECT_EQ(ncclIbResiliencySendRequestInit(&scene.sendRes, &request, 0), ncclSuccess);
  tracked.id = 7;
  EXPECT_EQ(ncclIbResiliencySendRequestInit(&scene.sendRes, &request, 0), ncclInternalError);
}

TEST_F(NetIbP2pResiliencyMicrotest, RepostRequest_ReceiveRepostsCtsOnce) {
  NetIbResiliencyScene scene;
  auto& request = scene.recv.base.reqs[2];
  request.base = &scene.recv.base;
  request.id = 2;
  request.type = NCCL_NET_IB_REQ_RECV;

  EXPECT_EQ(ncclIbResiliencyRepostRequest(&request), ncclSuccess);
  EXPECT_EQ(g_resiliencyPostFifoCalls, 1);
}

TEST_F(NetIbP2pResiliencyMicrotest, RepostRequest_SendSkipsDeliveredQpsAndReaddsMissingEvents) {
  NetIbResiliencyScene scene;
  scene.send.base.nqps = 2;
  scene.send.base.nDataQps = 2;
  scene.send.base.splitDataOnQps = 1;
  auto& request = scene.send.base.reqs[1];
  request.base = &scene.send.base;
  request.id = 1;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.nreqs = 1;
  request.send.sentData[0] = true;
  request.send.sentData[1] = true;
  scene.send.sendReqs[1][0] = &request;
  scene.sendRes.probingResults[1][0] = true;
  scene.sendRes.probingResults[1][1] = false;

  EXPECT_EQ(ncclIbResiliencyRepostRequest(&request), ncclSuccess);
  EXPECT_EQ(request.events[0], 0);
  EXPECT_EQ(request.events[1], 1);
  EXPECT_TRUE(request.send.sentData[0]);
  EXPECT_FALSE(request.send.sentData[1]);
  EXPECT_EQ(g_resiliencyMultiSendCalls, 1);
  EXPECT_EQ(g_resiliencyAddEventCalls, 1);
}

TEST_F(NetIbP2pResiliencyMicrotest, RequestIsComplete_ZeroOrBalancedFailoverEventsComplete) {
  NetIbResiliencyScene scene;
  ncclIbRequest request{};
  request.base = &scene.send.base;
  bool complete = false;

  ASSERT_EQ(NetIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
  EXPECT_TRUE(complete);
  request.events[0] = 2;
  request.events[1] = -2;
  ASSERT_EQ(NetIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
  EXPECT_TRUE(complete);
  request.events[1] = -1;
  ASSERT_EQ(NetIbResiliencyRequestIsComplete(&request, &complete), ncclSuccess);
  EXPECT_FALSE(complete);
}

TEST_F(NetIbP2pResiliencyMicrotest, RequestIsComplete_NullRequestIsIncompleteAndNullOutputIsRejected) {
  bool complete = true;
  EXPECT_EQ(NetIbResiliencyRequestIsComplete(nullptr, &complete), ncclSuccess);
  EXPECT_FALSE(complete);
  EXPECT_EQ(NetIbResiliencyRequestIsComplete(nullptr, nullptr), ncclInternalError);
}

TEST_F(NetIbP2pResiliencyMicrotest, DeviceNumSet_AssignsProbeAndRecoveryQpCounts) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.recoveryEnabled = true;
  ASSERT_EQ(NetIbResiliencyDeviceNumSet(&scene.sendRes.base, 2, 3), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.ndevs, 2);
  EXPECT_EQ(scene.sendRes.base.nProbingQps, 2);
  EXPECT_EQ(scene.sendRes.base.nPortRecoveryQps, 3);
  ASSERT_EQ(NetIbResiliencyDeviceNumSet(&scene.sendRes.base, 3, 2), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.ndevs, 3);
  EXPECT_EQ(scene.sendRes.base.nProbingQps, 3);
  EXPECT_EQ(scene.sendRes.base.nPortRecoveryQps, 3);

  scene.recvRes.recoveryEnabled = true;
  ASSERT_EQ(NetIbResiliencyDeviceNumSet(&scene.recvRes, 2, 3), ncclSuccess);
  EXPECT_EQ(scene.recvRes.ndevs, 2);
  EXPECT_EQ(scene.recvRes.nProbingQps, 3);
  EXPECT_EQ(scene.recvRes.nPortRecoveryQps, 3);
  scene.recvRes.recoveryEnabled = false;
  ASSERT_EQ(NetIbResiliencyDeviceNumSet(&scene.recvRes, 2, 3), ncclSuccess);
  EXPECT_EQ(scene.recvRes.nPortRecoveryQps, 0);
}

TEST_F(NetIbP2pResiliencyMicrotest, DeviceNumSet_InvalidCountsAreRejected) {
  NetIbResiliencyScene scene;
  EXPECT_EQ(NetIbResiliencyDeviceNumSet(nullptr, 1, 1), ncclInternalError);
  EXPECT_EQ(NetIbResiliencyDeviceNumSet(&scene.recvRes, 0, 1), ncclInternalError);
  EXPECT_EQ(NetIbResiliencyDeviceNumSet(&scene.recvRes, 1, 0), ncclInternalError);
  EXPECT_EQ(NetIbResiliencyDeviceNumSet(&scene.recvRes, 1, NCCL_IB_MAX_DEVS_PER_NIC + 1), ncclInternalError);
  scene.recvRes.recoveryEnabled = true;
  EXPECT_EQ(NetIbResiliencyDeviceNumSet(&scene.recvRes, NCCL_IB_MAX_DEVS_PER_NIC + 1, 1), ncclInternalError);
}

TEST_F(NetIbP2pResiliencyMicrotest, DataQueueSizing_AccountsForDirectionQpsAndFailoverDevices) {
  NetIbResiliencyScene scene;
  const int nqps = 2;
  const int ndevs = 3;
  for (ncclIbNetCommBase* base : {&scene.send.base, &scene.recv.base}) {
    base->nqps = base->nDataQps = nqps;
    base->splitDataOnQps = 1;
  }
  scene.sendRes.base.ndevs = scene.recvRes.ndevs = ndevs;
  int cqSize = 0;
  ASSERT_EQ(NetIbResiliencyDataCqSizeGet(&scene.sendRes.base, 0, &cqSize), ncclSuccess);
  EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * nqps * ndevs);

  ASSERT_EQ(NetIbResiliencyDataCqSizeGet(&scene.recvRes, 0, &cqSize), ncclSuccess);
  EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * 2 * nqps * ndevs);
  uint32_t rqSize = 0;
  ASSERT_EQ(NetIbResiliencyDataRqSizeGet(&scene.recvRes, 0, &rqSize), ncclSuccess);
  EXPECT_EQ(rqSize, static_cast<uint32_t>(NET_IB_MAX_REQUESTS * nqps));
}

TEST_F(NetIbP2pResiliencyMicrotest, RemoteCompletionRecords_SendOnlyStoresPerDeviceInfo) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.nProbingQps = 2;

  ASSERT_EQ(NetIbResiliencyRemoteCompletionRecordsSet(&scene.sendRes.base, 0x55, 0x123400, 1), ncclSuccess);
  EXPECT_EQ(scene.sendRes.remCmplRecordsInfo[1].rkey, 0x55u);
  EXPECT_EQ(scene.sendRes.remCmplRecordsInfo[1].addr, 0x123400u);
  EXPECT_EQ(NetIbResiliencyRemoteCompletionRecordsSet(&scene.recvRes, 1, 2, 0), ncclInternalError);
}

// AICOMRCCL-3129: sender probing-QP count is local-device based, while remote
// completion records are supplied once per remote device.
TEST_F(NetIbP2pResiliencyMicrotest,
       DISABLED_RemoteCompletionRecords_AllRemoteDevicesSupportedWhenCountsAreAsymmetric) {
  NetIbResiliencyScene scene;
  ASSERT_EQ(NetIbResiliencyDeviceNumSet(&scene.sendRes.base, 1, 3), ncclSuccess);
  scene.send.base.nRemDevs = 3;
  ASSERT_EQ(scene.sendRes.base.nProbingQps, 1);
  ASSERT_EQ(scene.sendRes.base.nPortRecoveryQps, 0);

  EXPECT_EQ(NetIbResiliencyRemoteCompletionRecordsSet(&scene.sendRes.base, 0x55, 0x123400, 2), ncclSuccess);
  EXPECT_EQ(scene.sendRes.remCmplRecordsInfo[2].rkey, 0x55u);
  EXPECT_EQ(scene.sendRes.remCmplRecordsInfo[2].addr, 0x123400u);
  EXPECT_EQ(NetIbResiliencyRemoteCompletionRecordsSet(&scene.sendRes.base, 0x66, 0x234500, 3),
            ncclInternalError);
}

TEST_F(NetIbP2pResiliencyMicrotest, Init_DisabledReturnsNullAndEnabledInitializesDirectionSpecificContext) {
  ncclIbNetCommBase sendBase{};
  sendBase.isSend = true;
  ncclIbNetCommBase recvBase{};
  ncclIbResiliency* resiliency = reinterpret_cast<ncclIbResiliency*>(0x1);
  EXPECT_EQ(NetIbResiliencyInit(&sendBase, &resiliency), ncclSuccess);
  EXPECT_EQ(resiliency, nullptr);
  EXPECT_TRUE(g_resiliencyIbMallocSizes.empty());

  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "IB_RESILIENCY_PORT_FAILOVER") == 0 ? int64_t{1} : fallback;
  });
  ScopedHook recovery(g_ncclIbPortRecoveryInit, [](ncclIbResiliency* context) {
    context->recoveryEnabled = !context->baseComm->isSend;
    return ncclSuccess;
  });
  // ncclIbMalloc zero-fills, so poison the allocation to observe Init's explicit resets.
  g_resiliencyIbMallocPoison = true;

  ASSERT_EQ(NetIbResiliencyInit(&sendBase, &resiliency), ncclSuccess);
  ASSERT_NE(resiliency, nullptr);
  ASSERT_EQ(g_resiliencyIbMallocSizes.size(), 1u);
  EXPECT_EQ(g_resiliencyIbMallocSizes[0], sizeof(ncclIbResiliencySend));
  EXPECT_EQ(resiliency->baseComm, &sendBase);
  EXPECT_FALSE(resiliency->inProgress);
  auto* sendResiliency = reinterpret_cast<ncclIbResiliencySend*>(resiliency);
  EXPECT_TRUE(AllBytesZero(sendResiliency->failedRequests));
  EXPECT_TRUE(AllBytesZero(sendResiliency->probingResults));
  EXPECT_TRUE(AllBytesZero(sendResiliency->remCmplRecordsInfo));
  EXPECT_EQ(NetIbResiliencyDestroy(&resiliency), ncclSuccess);
  EXPECT_EQ(resiliency, nullptr);

  ASSERT_EQ(NetIbResiliencyInit(&recvBase, &resiliency), ncclSuccess);
  ASSERT_NE(resiliency, nullptr);
  ASSERT_EQ(g_resiliencyIbMallocSizes.size(), 2u);
  EXPECT_EQ(g_resiliencyIbMallocSizes[1], sizeof(ncclIbResiliency));
  EXPECT_EQ(resiliency->baseComm, &recvBase);
  EXPECT_FALSE(resiliency->inProgress);
  EXPECT_EQ(resiliency->outstandingRecovery, 0);
  EXPECT_EQ(NetIbResiliencyDestroy(&resiliency), ncclSuccess);
  EXPECT_EQ(resiliency, nullptr);
}

TEST_F(NetIbP2pResiliencyMicrotest, QpReconfigure_RunsResetInitRtrRtsInOrder) {
  NetIbResiliencyScene scene;
  ncclIbNetCommDevBase devBase{};
  devBase.gidInfo.localGidIndex = 4;
  std::array<char, 4> order{};
  int cursor = 0;
  g_netIbQpReset = [&order, &cursor](ncclIbQp*) {
    order[cursor++] = 'R';
    return ncclSuccess;
  };
  g_netIbQpInit = [&order, &cursor](ncclIbQp*) {
    order[cursor++] = 'I';
    return ncclSuccess;
  };
  g_netIbQpRtr = [&order, &cursor](ncclIbQp*) {
    order[cursor++] = 'T';
    return ncclSuccess;
  };
  g_netIbQpRts = [&order, &cursor](ncclIbQp*) {
    order[cursor++] = 'S';
    return ncclSuccess;
  };
  bool success = false;

  ASSERT_EQ(NetIbResiliencyQpReconfigure(&scene.sendRes.base, &scene.send.base.qps[0], &devBase, 0, &success),
            ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(order, (std::array<char, 4>{{'R', 'I', 'T', 'S'}}));
  EXPECT_EQ(scene.send.base.qps[0].rtrAttr.localGidIndex, 4);
}

TEST_F(NetIbP2pResiliencyMicrotest, QpReconfigure_StageFailureStopsBeforeLaterStages) {
  const std::string stages = "RIETS";  // reset, init, ECE, RTR, RTS
  for (char failing : stages) {
    NetIbResiliencyScene scene;
    ncclIbNetCommDevBase devBase{};
    ncclIbQp& qp = scene.send.base.qps[0];
    qp.eceSupported = 1;
    std::string order;
    auto stage = [&order, failing](char name) {
      return [&order, failing, name](ncclIbQp*) {
        order.push_back(name);
        return name == failing ? ncclSystemError : ncclSuccess;
      };
    };
    g_netIbQpReset = stage('R');
    g_netIbQpInit = stage('I');
    g_netIbQpRtr = stage('T');
    g_netIbQpRts = stage('S');
    ScopedHook ece(g_ibvSetEce, [&order, failing](ibv_qp*, ibv_ece*, int*) {
      order.push_back('E');
      return failing == 'E' ? ncclSystemError : ncclSuccess;
    });
    bool success = true;

    ASSERT_EQ(NetIbResiliencyQpReconfigure(&scene.sendRes.base, &qp, &devBase, 0, &success), ncclSuccess) << failing;
    EXPECT_FALSE(success) << failing;
    EXPECT_EQ(order, stages.substr(0, stages.find(failing) + 1));
  }
}

TEST_F(NetIbP2pResiliencyMicrotest, Progress_RecoveredDeviceRestoresOriginalQpsAndCompletes) {
  NetIbResiliencyScene scene;
  scene.recvRes.inProgress = true;
  scene.recvRes.outstandingRecovery = 1;
  scene.recv.base.activeQps[0] = &scene.recv.base.qps[1];
  scene.recv.base.activeQps[2] = &scene.recv.base.qps[3];
  auto* healthyQp1 = scene.recv.base.activeQps[1];
  auto* healthyQp3 = scene.recv.base.activeQps[3];
  scene.recvRes.devs[0].state.store(ncclIbResiliencyDevStateRecovered, std::memory_order_release);

  ASSERT_EQ(NetIbResiliencyProgress(&scene.recvRes), ncclSuccess);
  EXPECT_EQ(scene.recv.base.activeQps[0], &scene.recv.base.qps[0]);
  EXPECT_EQ(scene.recv.base.activeQps[2], &scene.recv.base.qps[2]);
  EXPECT_EQ(scene.recv.base.activeQps[1], healthyQp1);
  EXPECT_EQ(scene.recv.base.activeQps[3], healthyQp3);
  EXPECT_EQ(scene.recvRes.devs[0].state.load(), ncclIbResiliencyDevStateOk);
  EXPECT_EQ(scene.recvRes.outstandingRecovery, 0);
  EXPECT_FALSE(scene.recvRes.inProgress);
}

TEST_F(NetIbP2pResiliencyMicrotest, Progress_RecoveryFailureBecomesPermanent) {
  NetIbResiliencyScene scene;
  scene.recvRes.inProgress = true;
  scene.recvRes.outstandingRecovery = 1;
  scene.recvRes.devs[1].state.store(ncclIbResiliencyDevStateRecoveryFailed, std::memory_order_release);

  ASSERT_EQ(NetIbResiliencyProgress(&scene.recvRes), ncclSuccess);
  EXPECT_EQ(scene.recvRes.devs[1].state.load(), ncclIbResiliencyDevStateErrorPermanent);
  EXPECT_FALSE(scene.recvRes.inProgress);
}

TEST_F(NetIbP2pResiliencyMicrotest, Progress_SenderProbesThenFreesDeliveredRequest) {
  NetIbResiliencyScene scene;
  auto& request = scene.send.base.reqs[1];
  request.base = &scene.send.base;
  request.id = 1;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.nreqs = 1;
  request.send.sentData[0] = true;
  request.events[0] = 1;
  scene.send.sendReqs[1][0] = &request;
  ibv_mr mr{};
  scene.sendRes.base.devs[1].probingResultMr = &mr;
  ibv_wc error{};
  error.status = IBV_WC_RETRY_EXC_ERR;
  error.wr_id = 1;
  ASSERT_EQ(NetIbResiliencyHandleCompletionError(&scene.sendRes.base, &error, 0), ncclSuccess);

  g_netIbNowNs += 10 * MSEC_TO_NSEC;
  ASSERT_EQ(NetIbResiliencyProgress(&scene.sendRes.base), ncclSuccess);
  EXPECT_TRUE(scene.verbs[0].posted.empty());
  ASSERT_EQ(scene.verbs[1].posted.size(), 1u);
  // The probe is an RDMA read of the peer's completion record: land QP 0's "delivered" flag where it points.
  *reinterpret_cast<bool*>(scene.verbs[1].posted[0].sge.addr) = true;
  scene.verbs[1].Complete(1, IBV_WC_RDMA_READ, scene.sendRes.base.probingQps[1].qp->qp_num);
  ASSERT_EQ(NetIbResiliencyProgress(&scene.sendRes.base), ncclSuccess);

  EXPECT_EQ(scene.sendRes.failedRequests[1].request, nullptr);
  EXPECT_EQ(scene.sendRes.base.outstandingRequests, 0);
  EXPECT_EQ(scene.sendRes.base.devs[1].nOutstandingProbes, 0);
  EXPECT_FALSE(scene.sendRes.base.inProgress);
  EXPECT_EQ(request.events[0], 0);
  EXPECT_EQ(g_resiliencyMultiSendCalls, 0);
}

TEST_F(NetIbP2pResiliencyMicrotest, DevInitAndDestroyManageProbeMrAndCq) {
  NetIbResiliencyScene scene;
  ibv_mr mr{};
  ibv_cq cq{};
  ncclIbDev dev{};
  dev.context = &scene.verbs[0].ctx;
  ScopedHook reg(g_ibvRegMr, [&](ibv_mr** out, ibv_pd* pd, void*, size_t, int) {
    EXPECT_EQ(pd, &scene.verbs[0].pd);
    *out = &mr;
    return ncclSuccess;
  });
  ScopedHook create(g_ibvCreateCq,
                    [&](ibv_cq** out, ibv_context* context, int size, void*, ibv_comp_channel*, int) {
                      EXPECT_EQ(context, &scene.verbs[0].ctx);
                      EXPECT_EQ(size, NET_IB_MAX_REQUESTS);
                      *out = &cq;
                      return ncclSuccess;
                    });
  ScopedHook destroyCq(g_ibvDestroyCq, [](ibv_cq*) { return ncclSuccess; });
  ScopedHook dereg(g_ibvDeregMr, [](ibv_mr*) { return ncclSuccess; });

  ASSERT_EQ(NetIbResiliencyDevInit(&scene.sendRes.base, 0, &dev), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.devs[0].probingResultMr, &mr);
  EXPECT_EQ(scene.sendRes.base.devs[0].probingCq, &cq);
  EXPECT_EQ(NetIbResiliencyDevDestroy(&scene.sendRes.base, 0), ncclSuccess);
  EXPECT_EQ(reg.calls, 1);
  EXPECT_EQ(create.calls, 1);
  EXPECT_EQ(destroyCq.calls, 1);
  EXPECT_EQ(dereg.calls, 1);
}

TEST_F(NetIbP2pResiliencyMicrotest, SenderProbeQpsCreationStripesDevicesAndPublishesInfo) {
  NetIbResiliencyScene scene;
  for (int d = 0; d < 2; ++d) {
    scene.send.devs[d].base.pkeyIndex = 10 + d;
    ncclIbDevs[d].portNum = d + 1;
  }
  g_netIbQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr* attr) {
    EXPECT_EQ(attr->type, IBV_QPT_RC);
    EXPECT_EQ(attr->maxRecvWorkRequest, 0u);
    EXPECT_EQ(attr->maxSendWorkRequest, static_cast<uint32_t>(NET_IB_MAX_REQUESTS));
    return scene.CreateQp(qp, attr);
  };
  std::vector<ncclIbQp*> initialized;
  g_netIbQpInit = [&](ncclIbQp* qp) {
    initialized.push_back(qp);
    return ncclSuccess;
  };
  ncclIbResiliencyInfo info{};

  ASSERT_EQ(NetIbResiliencySenderCreateQps(&scene.sendRes.base, &info), ncclSuccess);
  for (int d = 0; d < 2; ++d) {
    const ncclIbQp& qp = scene.sendRes.base.probingQps[d];
    EXPECT_EQ(qp.devIndex, d);
    EXPECT_EQ(qp.qp->context, &scene.verbs[d].ctx);
    EXPECT_EQ(qp.qp->send_cq, &scene.verbs[d].cq);
    EXPECT_EQ(qp.initAttr.pkeyIndex, 10 + d);
    EXPECT_EQ(qp.initAttr.portNum, d + 1);
    EXPECT_EQ(qp.initAttr.qpAccessFlags, IBV_ACCESS_LOCAL_WRITE);
    EXPECT_EQ(info.probingQpsInfo[d].qpn, qp.qp->qp_num);
    EXPECT_EQ(info.probingQpsInfo[d].devIndex, d);
  }
  EXPECT_EQ(initialized, (std::vector<ncclIbQp*>{&scene.sendRes.base.probingQps[0],
                                                &scene.sendRes.base.probingQps[1]}));
}

TEST_F(NetIbP2pResiliencyMicrotest, SenderProbeQpsToRtsUsesRemoteMetadata) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.nProbingQps = 1;
  scene.send.devs[0].base.gidInfo.localGidIndex = 3;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  ncclIbDevs[0].portNum = 1;
  ncclIbConnectionMetadata remote{};
  remote.tc = 7;
  remote.sl = 2;
  remote.resiliencyInfo.probingQpsInfo[0].qpn = 300;
  remote.resiliencyInfo.probingQpsInfo[0].devIndex = 0;
  remote.devs[0].mtu = IBV_MTU_2048;
  remote.devs[0].link_layer = IBV_LINK_LAYER_ETHERNET;
  remote.devs[0].ib_port = 1;
  std::vector<char> transitions;
  g_netIbQpRtr = [&](ncclIbQp* qp) {
    EXPECT_EQ(qp, &scene.sendRes.base.probingQps[0]);
    transitions.push_back('T');
    return ncclSuccess;
  };
  g_netIbQpRts = [&](ncclIbQp* qp) {
    EXPECT_EQ(qp, &scene.sendRes.base.probingQps[0]);
    transitions.push_back('S');
    return ncclSuccess;
  };

  ASSERT_EQ(NetIbResiliencySenderQpsToRts(&scene.sendRes.base, &remote), ncclSuccess);
  auto& qp = scene.sendRes.base.probingQps[0];
  EXPECT_EQ(qp.remDevIdx, 0);
  EXPECT_EQ(qp.rtrAttr.remoteQpNum, 300u);
  EXPECT_EQ(qp.rtrAttr.mtu, IBV_MTU_2048);
  EXPECT_EQ(qp.rtrAttr.tc, 7);
  EXPECT_EQ(qp.rtrAttr.sl, 2);
  EXPECT_EQ(qp.rtrAttr.localGidIndex, 3);
  EXPECT_EQ(transitions, (std::vector<char>{'T', 'S'}));
}

// AICOMRCCL-3129: the AV port is a local attribute and must not be copied
// from the remote device metadata.
TEST_F(NetIbP2pResiliencyMicrotest, DISABLED_SenderProbeQpsToRtsUsesLocalIbPort) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.nProbingQps = 1;
  ncclIbDevs[0].portNum = 3;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  ncclIbConnectionMetadata remote{};
  remote.resiliencyInfo.probingQpsInfo[0].qpn = 300;
  remote.resiliencyInfo.probingQpsInfo[0].devIndex = 0;
  remote.devs[0].mtu = IBV_MTU_2048;
  remote.devs[0].link_layer = IBV_LINK_LAYER_ETHERNET;
  remote.devs[0].ib_port = 7;
  g_netIbQpRtr = [](ncclIbQp*) { return ncclSuccess; };
  g_netIbQpRts = [](ncclIbQp*) { return ncclSuccess; };

  ASSERT_EQ(NetIbResiliencySenderQpsToRts(&scene.sendRes.base, &remote), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.probingQps[0].rtrAttr.localIbPort, 3);
}

// AICOMRCCL-3129: receiver probing QPs have the same local AV-port contract.
TEST_F(NetIbP2pResiliencyMicrotest, DISABLED_ReceiverProbeQpsToRtsUsesLocalIbPort) {
  NetIbResiliencyScene scene;
  scene.recvRes.nProbingQps = 1;
  ncclIbDevs[0].portNum = 3;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  g_netIbQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr* attr) { return scene.CreateQp(qp, attr); };
  g_netIbQpInit = [](ncclIbQp*) { return ncclSuccess; };
  g_netIbQpRtr = [](ncclIbQp*) { return ncclSuccess; };
  g_netIbQpRts = [](ncclIbQp*) { return ncclSuccess; };
  ncclIbConnectionMetadata remote{};
  remote.resiliencyInfo.probingQpsInfo[0].qpn = 300;
  remote.resiliencyInfo.probingQpsInfo[0].devIndex = 0;
  remote.devs[0].mtu = IBV_MTU_2048;
  remote.devs[0].link_layer = IBV_LINK_LAYER_ETHERNET;
  remote.devs[0].ib_port = 7;
  ncclIbResiliencyInfo local{};

  ASSERT_EQ(NetIbResiliencyReceiverQpsCreateToRts(&scene.recvRes, &remote, &local), ncclSuccess);
  EXPECT_EQ(scene.recvRes.probingQps[0].rtrAttr.localIbPort, 3);
}

TEST_F(NetIbP2pResiliencyMicrotest, ReceiverProbeQpsCreationStripesAndTransitions) {
  NetIbResiliencyScene scene;
  scene.recvRes.nProbingQps = 3;
  for (int d = 0; d < 2; ++d) {
    ncclIbDevs[d].portNum = d + 1;
    ncclIbDevs[d].portAttr.active_mtu = IBV_MTU_4096;
  }
  g_netIbQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr* attr) { return scene.CreateQp(qp, attr); };
  std::vector<char> transitions;
  std::vector<ncclIbQp*> transitionedQps;
  g_netIbQpInit = [&](ncclIbQp* qp) {
    transitions.push_back('I');
    transitionedQps.push_back(qp);
    return ncclSuccess;
  };
  g_netIbQpRtr = [&](ncclIbQp* qp) {
    transitions.push_back('T');
    transitionedQps.push_back(qp);
    return ncclSuccess;
  };
  g_netIbQpRts = [&](ncclIbQp* qp) {
    transitions.push_back('S');
    transitionedQps.push_back(qp);
    return ncclSuccess;
  };
  ncclIbConnectionMetadata remote{};
  for (int i = 0; i < 3; ++i) {
    remote.resiliencyInfo.probingQpsInfo[i].qpn = 400 + i;
    remote.resiliencyInfo.probingQpsInfo[i].devIndex = i % 2;
  }
  for (int i = 0; i < 2; ++i) {
    remote.devs[i].mtu = IBV_MTU_2048;
    remote.devs[i].link_layer = IBV_LINK_LAYER_ETHERNET;
    remote.devs[i].ib_port = i + 1;
  }
  ncclIbResiliencyInfo local{};

  ASSERT_EQ(NetIbResiliencyReceiverQpsCreateToRts(&scene.recvRes, &remote, &local), ncclSuccess);
  for (int i = 0; i < 3; ++i) {
    const ncclIbQp& qp = scene.recvRes.probingQps[i];
    EXPECT_EQ(qp.devIndex, i % 2);
    EXPECT_EQ(qp.qp->context, &scene.verbs[i % 2].ctx);
    EXPECT_EQ(qp.qp->send_cq, &scene.verbs[i % 2].cq);
    EXPECT_EQ(qp.initAttr.portNum, i % 2 + 1);
    EXPECT_EQ(qp.initAttr.qpAccessFlags, IBV_ACCESS_REMOTE_READ);
    EXPECT_EQ(local.probingQpsInfo[i].qpn, qp.qp->qp_num);
    EXPECT_EQ(local.probingQpsInfo[i].devIndex, i % 2);
  }
  EXPECT_EQ(transitions, (std::vector<char>{'I', 'T', 'S', 'I', 'T', 'S', 'I', 'T', 'S'}));
  EXPECT_EQ(transitionedQps,
            (std::vector<ncclIbQp*>{&scene.recvRes.probingQps[0], &scene.recvRes.probingQps[0],
                                    &scene.recvRes.probingQps[0], &scene.recvRes.probingQps[1],
                                    &scene.recvRes.probingQps[1], &scene.recvRes.probingQps[1],
                                    &scene.recvRes.probingQps[2], &scene.recvRes.probingQps[2],
                                    &scene.recvRes.probingQps[2]}));
}

TEST_F(NetIbP2pResiliencyMicrotest, QpsReconfigure_DelegatesRecoveryAfterProbeQpsSucceed) {
  NetIbResiliencyScene scene;
  ncclIbDevs[0].gidInfo.localGidIndex = 6;
  std::vector<ncclIbQp*> reset;
  g_netIbQpReset = [&reset](ncclIbQp* qp) {
    reset.push_back(qp);
    return ncclSuccess;
  };
  g_netIbQpInit = [](ncclIbQp*) { return ncclSuccess; };
  g_netIbQpRtr = [](ncclIbQp*) { return ncclSuccess; };
  g_netIbQpRts = [](ncclIbQp*) { return ncclSuccess; };
  ScopedHook recovery(g_ncclIbPortRecoveryQpsReconfigure,
                      [](ncclIbResiliency*, int, bool* success) {
                        *success = true;
                        return ncclSuccess;
                      });
  bool success = false;

  ASSERT_EQ(NetIbResiliencyQpsReconfigure(&scene.sendRes.base, 0, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(recovery.calls, 1);
  EXPECT_EQ(reset, (std::vector<ncclIbQp*>{&scene.sendRes.base.probingQps[0]}));
  EXPECT_EQ(scene.sendRes.base.probingQps[0].rtrAttr.localGidIndex, 6);
}

TEST_F(NetIbP2pResiliencyMicrotest, QpsReconfigure_ProbeQpFailureSkipsRecovery) {
  NetIbResiliencyScene scene;
  g_netIbQpReset = [](ncclIbQp*) { return ncclSuccess; };
  g_netIbQpInit = [](ncclIbQp*) { return ncclSystemError; };
  ScopedHook recovery(g_ncclIbPortRecoveryQpsReconfigure,
                      [](ncclIbResiliency*, int, bool* success) {
                        *success = true;
                        return ncclSuccess;
                      });
  bool success = true;

  ASSERT_EQ(NetIbResiliencyQpsReconfigure(&scene.sendRes.base, 0, &success), ncclSuccess);
  EXPECT_FALSE(success);
  EXPECT_EQ(recovery.calls, 0);
}

TEST_F(NetIbP2pResiliencyMicrotest, Close_DestroysProbeQpsAndRecoveryResources) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.nPortRecoveryQps = 3;
  scene.sendRes.base.recoveryEnabled = true;
  std::vector<ibv_qp*> destroyed;
  ScopedHook destroy(g_ibvDestroyQp, [&](ibv_qp* qp) {
    destroyed.push_back(qp);
    return ncclSuccess;
  });
  ScopedHook close(g_ncclIbPortRecoveryClose, [](ncclIbResiliency*) { return ncclSuccess; });
  ScopedHook destroyRecovery(g_ncclIbPortRecoveryQpsDestroy,
                             [](ncclIbResiliency*, int count) {
                               EXPECT_EQ(count, 3);
                               return ncclSuccess;
                             });

  EXPECT_EQ(NetIbResiliencyClose(&scene.sendRes.base), ncclSuccess);
  EXPECT_EQ(destroyed, (std::vector<ibv_qp*>{scene.sendRes.base.probingQps[0].qp,
                                             scene.sendRes.base.probingQps[1].qp}));
  EXPECT_EQ(close.calls, 1);
  EXPECT_EQ(destroyRecovery.calls, 1);
}

TEST_F(NetIbP2pResiliencyMicrotest, HandleCompletionError_SenderMarksDeviceAndTracksRequest) {
  NetIbResiliencyScene scene;
  auto& request = scene.send.base.reqs[1];
  request.base = &scene.send.base;
  request.id = 1;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.nreqs = 1;
  scene.send.sendReqs[1][0] = &request;
  ibv_wc completion{};
  completion.status = IBV_WC_RETRY_EXC_ERR;
  completion.wr_id = 1;
  g_netIbNowNs = 7000;

  ASSERT_EQ(NetIbResiliencyHandleCompletionError(&scene.sendRes.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.devs[0].state.load(), ncclIbResiliencyDevStateError);
  EXPECT_EQ(scene.sendRes.failedRequests[1].request, &request);
  EXPECT_EQ(scene.sendRes.base.outstandingRequests, 1);
}

TEST_F(NetIbP2pResiliencyMicrotest, HandleCompletionError_RecoveryHandOffTracksRecoveryOrFailsDevicePermanently) {
  for (ncclResult_t handOff : {ncclSuccess, ncclSystemError}) {
    NetIbResiliencyScene scene;
    scene.recvRes.recoveryEnabled = true;
    ScopedHook recovery(g_ncclIbPortRecoveryHandleFailure, [&](ncclIbResiliency* resiliency, int devIndex) {
      EXPECT_EQ(resiliency, &scene.recvRes);
      EXPECT_EQ(devIndex, 0);
      EXPECT_EQ(resiliency->devs[0].state.load(), ncclIbResiliencyDevStateRecoveryInProgress);
      return handOff;
    });
    ibv_wc completion{};
    completion.status = IBV_WC_WR_FLUSH_ERR;
    completion.wr_id = NCCL_IB_RECV_WR_ID_DUMMY;

    ASSERT_EQ(NetIbResiliencyHandleCompletionError(&scene.recvRes, &completion, 0), ncclSuccess) << handOff;
    const bool queued = handOff == ncclSuccess;
    EXPECT_EQ(recovery.calls, 1);
    EXPECT_EQ(scene.recvRes.devs[0].state.load(),
              queued ? ncclIbResiliencyDevStateRecoveryInProgress : ncclIbResiliencyDevStateErrorPermanent)
        << handOff;
    EXPECT_EQ(scene.recvRes.devs[1].state.load(), ncclIbResiliencyDevStateOk) << handOff;
    EXPECT_EQ(scene.recvRes.outstandingRecovery, queued ? 1 : 0) << handOff;
    EXPECT_EQ(scene.recvRes.inProgress, queued) << handOff;
  }
}

// AICOMRCCL-3127: slot-only wr_id values cannot distinguish a delayed CQE
// from the request generation that subsequently reused the same slot.
TEST_F(NetIbP2pResiliencyMicrotest, DISABLED_HandleCompletionError_StaleGenerationDoesNotTrackCurrentRequest) {
  NetIbResiliencyScene scene;
  constexpr uint64_t slot = 1;
  auto& current = scene.send.base.reqs[slot];
  current.base = &scene.send.base;
  current.id = NET_IB_MAX_REQUESTS + slot;
  current.type = NCCL_NET_IB_REQ_SEND;
  current.nreqs = 1;
  scene.send.sendReqs[slot][0] = &current;
  ibv_wc stale{};
  stale.status = IBV_WC_RETRY_EXC_ERR;
  stale.wr_id = slot;

  ASSERT_EQ(NetIbResiliencyHandleCompletionError(&scene.sendRes.base, &stale, 0), ncclSuccess);
  EXPECT_EQ(scene.sendRes.failedRequests[slot].request, nullptr);
  EXPECT_EQ(scene.sendRes.base.outstandingRequests, 0);
  EXPECT_EQ(scene.sendRes.base.devs[0].state.load(), ncclIbResiliencyDevStateOk);
  EXPECT_EQ(scene.send.base.activeQps[0], &scene.send.base.qps[0]);
  EXPECT_EQ(scene.send.base.activeQps[2], &scene.send.base.qps[2]);
}

TEST_F(NetIbP2pResiliencyMicrotest, HandleCompletionError_ReceiverFlushClearsFailedDeviceEvent) {
  NetIbResiliencyScene scene;
  auto& request = scene.recv.base.reqs[2];
  request.base = &scene.recv.base;
  request.id = 2;
  request.type = NCCL_NET_IB_REQ_FLUSH;
  request.events[0] = 1;
  ibv_wc completion{};
  completion.status = IBV_WC_RETRY_EXC_ERR;
  completion.wr_id = NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + 2;

  ASSERT_EQ(NetIbResiliencyHandleCompletionError(&scene.recvRes, &completion, 0), ncclSuccess);
  EXPECT_EQ(request.events[0], 0);
  EXPECT_EQ(scene.recvRes.devs[0].state.load(), ncclIbResiliencyDevStateError);
}

TEST_F(NetIbP2pResiliencyMicrotest, ProbePost_BeforeDelayIsNoOpAndBoundaryPostsExactRecordAddress) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.devs[0].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);
  scene.sendRes.base.probingQps[1].remDevIdx = 1;
  ibv_mr mr{};
  mr.lkey = 0x51;
  scene.sendRes.base.devs[1].probingResultMr = &mr;
  scene.sendRes.remCmplRecordsInfo[1].addr = 0x100000;
  scene.sendRes.remCmplRecordsInfo[1].rkey = 0x52;
  auto& request = scene.send.base.reqs[3];
  request.base = &scene.send.base;
  request.id = 3;
  request.type = NCCL_NET_IB_REQ_SEND;
  auto& failed = scene.sendRes.failedRequests[3];
  failed.request = &request;
  failed.state = ncclIbResiliencyRequestStatePending;
  failed.errorInfo.time = 1000;
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    if (std::strcmp(name, "IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY") == 0) return int64_t{10};
    return fallback;
  });

  g_netIbNowNs = 1000 + 10 * MSEC_TO_NSEC - 1;
  ASSERT_EQ(ncclIbResiliencyProbePost(&scene.sendRes, &failed), ncclSuccess);
  EXPECT_TRUE(scene.verbs[1].posted.empty());
  EXPECT_EQ(failed.state, ncclIbResiliencyRequestStatePending);

  g_netIbNowNs++;
  ASSERT_EQ(ncclIbResiliencyProbePost(&scene.sendRes, &failed), ncclSuccess);
  EXPECT_TRUE(scene.verbs[0].posted.empty());
  ASSERT_EQ(scene.verbs[1].posted.size(), 1u);
  const auto& probe = scene.verbs[1].posted[0];
  EXPECT_EQ(probe.qp, scene.sendRes.base.probingQps[1].qp);
  EXPECT_EQ(probe.wrId, 3u);
  EXPECT_EQ(probe.remoteAddress,
            0x100000u + sizeof(ncclIbRequestCompletionRecord) * 3 + sizeof(int) * NCCL_NET_IB_MAX_RECVS);
  EXPECT_EQ(probe.rkey, 0x52u);
  EXPECT_EQ(probe.sge.lkey, 0x51u);
  EXPECT_EQ(failed.state, ncclIbResiliencyRequestStateProbePosted);
  EXPECT_EQ(scene.sendRes.base.devs[1].nOutstandingProbes, 1);
}

TEST_F(NetIbP2pResiliencyMicrotest, ProbePost_ReprobesUpToMaxAttemptsThenFails) {
  NetIbResiliencyScene scene;
  ibv_mr mr{};
  scene.sendRes.base.devs[0].probingResultMr = &mr;
  auto& request = scene.send.base.reqs[2];
  request.id = 2;
  auto& failed = scene.sendRes.failedRequests[2];
  failed.request = &request;
  failed.state = ncclIbResiliencyRequestStatePending;
  // IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS defaults to 1.
  failed.failedAttempts = 1;
  g_netIbNowNs += 10 * MSEC_TO_NSEC;

  ASSERT_EQ(ncclIbResiliencyProbePost(&scene.sendRes, &failed), ncclSuccess);
  EXPECT_EQ(scene.verbs[0].posted.size(), 1u);
  EXPECT_EQ(failed.state, ncclIbResiliencyRequestStateProbePosted);

  failed.state = ncclIbResiliencyRequestStatePending;
  failed.failedAttempts = 2;
  EXPECT_EQ(ncclIbResiliencyProbePost(&scene.sendRes, &failed), ncclRemoteError);
  EXPECT_EQ(scene.verbs[0].posted.size(), 1u);
  EXPECT_EQ(failed.state, ncclIbResiliencyRequestStatePending);
}

TEST_F(NetIbP2pResiliencyMicrotest, ProbeCompletion_SuccessMarksRequestCompleted) {
  NetIbResiliencyScene scene;
  auto& request = scene.send.base.reqs[4];
  request.id = 4;
  auto& failed = scene.sendRes.failedRequests[4];
  failed.request = &request;
  failed.state = ncclIbResiliencyRequestStateProbePosted;
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.wr_id = 4;

  EXPECT_EQ(ncclIbResiliencyProbeHandleCompletionEvent(&scene.sendRes, &completion, 1), ncclSuccess);
  EXPECT_EQ(failed.state, ncclIbResiliencyRequestStateProbeCompleted);
}

TEST_F(NetIbP2pResiliencyMicrotest, ProbeCompletion_ErrorFailsDeviceAndRequeuesProbe) {
  NetIbResiliencyScene scene;
  auto& request = scene.send.base.reqs[4];
  request.id = 4;
  auto& failed = scene.sendRes.failedRequests[4];
  failed.request = &request;
  failed.state = ncclIbResiliencyRequestStateProbePosted;
  ibv_wc completion{};
  completion.status = IBV_WC_RETRY_EXC_ERR;
  completion.wr_id = 4;

  EXPECT_EQ(ncclIbResiliencyProbeHandleCompletionEvent(&scene.sendRes, &completion, 1), ncclSuccess);
  EXPECT_EQ(failed.state, ncclIbResiliencyRequestStatePending);
  EXPECT_EQ(failed.failedAttempts, 1);
  EXPECT_EQ(scene.sendRes.base.devs[1].state.load(), ncclIbResiliencyDevStateError);
}

TEST_F(NetIbP2pResiliencyMicrotest, ProbeProgress_DrainsCompletionAndDecrementsOutstandingCount) {
  NetIbResiliencyScene scene;
  scene.sendRes.base.devs[1].nOutstandingProbes = 1;
  auto& request = scene.send.base.reqs[5];
  request.id = 5;
  scene.sendRes.failedRequests[5].request = &request;
  scene.sendRes.failedRequests[5].state = ncclIbResiliencyRequestStateProbePosted;
  scene.verbs[1].Complete(5, IBV_WC_RDMA_READ, scene.sendRes.base.probingQps[1].qp->qp_num);

  ASSERT_EQ(ncclIbResiliencyProbeProgress(&scene.sendRes), ncclSuccess);
  EXPECT_TRUE(scene.verbs[0].polledCqs.empty());
  EXPECT_EQ(scene.verbs[1].polledCqs, (std::vector<ibv_cq*>{&scene.verbs[1].cq, &scene.verbs[1].cq}));
  EXPECT_EQ(scene.sendRes.base.devs[1].nOutstandingProbes, 0);
  EXPECT_EQ(scene.sendRes.failedRequests[5].state, ncclIbResiliencyRequestStateProbeCompleted);
}

TEST_F(NetIbP2pResiliencyMicrotest, ProbeConclusion_AllDeliveredClearsSiblingEvents) {
  NetIbResiliencyScene scene;
  auto& request = scene.send.base.reqs[6];
  auto& sibling = scene.send.base.reqs[7];
  request.base = sibling.base = &scene.send.base;
  request.id = sibling.id = 6;
  request.type = sibling.type = NCCL_NET_IB_REQ_SEND;
  request.nreqs = sibling.nreqs = 2;
  request.send.sentData[0] = true;
  sibling.send.sentData[0] = true;
  request.events[0] = sibling.events[0] = 1;
  scene.send.sendReqs[6][0] = &request;
  scene.send.sendReqs[6][1] = &sibling;
  scene.sendRes.probingResults[6][0] = true;
  auto& failed = scene.sendRes.failedRequests[6];
  failed.request = &request;

  ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(&scene.sendRes, &failed), ncclSuccess);
  EXPECT_EQ(request.events[0], 0);
  EXPECT_EQ(sibling.events[0], 0);
  EXPECT_EQ(g_resiliencyMultiSendCalls, 0);
}

TEST_F(NetIbP2pResiliencyMicrotest, ProbeConclusion_MissingDataRepostsRequest) {
  NetIbResiliencyScene scene;
  auto& request = scene.send.base.reqs[6];
  request.base = &scene.send.base;
  request.id = 6;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.nreqs = 1;
  request.send.sentData[0] = true;
  scene.send.sendReqs[6][0] = &request;
  scene.sendRes.probingResults[6][0] = false;
  auto& failed = scene.sendRes.failedRequests[6];
  failed.request = &request;

  ASSERT_EQ(ncclIbResiliencyHandleProbeCompleted(&scene.sendRes, &failed), ncclSuccess);
  EXPECT_EQ(g_resiliencyMultiSendCalls, 1);
}

}  // namespace
