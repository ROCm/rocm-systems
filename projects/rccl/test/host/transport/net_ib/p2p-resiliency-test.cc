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
#include <deque>
#include <functional>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/transport_net_ib_recovery_fakes.h"
#include "fakes/transport_net_ib_resiliency_fakes.h"
#include "transport/net_ib/connect.h"
#include "transport/net_ib/p2p.h"

namespace {

uint64_t g_resiliencyNow = 1000;
int g_resiliencyMultiSendCalls = 0;
int g_resiliencyPostFifoCalls = 0;
int g_resiliencyAddEventCalls = 0;
struct ResiliencyProbePost {
  uint64_t wrId;
  uint64_t remoteAddress;
  uint32_t rkey;
  uint64_t localAddress;
  uint32_t length;
  uint32_t lkey;
};
std::vector<ResiliencyProbePost> g_resiliencyProbePosts;
std::deque<ibv_wc> g_resiliencyProbeCompletions;
int g_resiliencyProbePostResult = 0;
std::function<ncclResult_t(ncclIbQp*)> g_resiliencyQpReset = [](ncclIbQp*) { return ncclSystemError; };
std::function<ncclResult_t(ncclIbQp*, ncclIbQpCreateAttr*)> g_resiliencyQpCreate =
    [](ncclIbQp*, ncclIbQpCreateAttr*) { return ncclSystemError; };
std::function<ncclResult_t(ncclIbQp*)> g_resiliencyQpInit = [](ncclIbQp*) { return ncclSystemError; };
std::function<ncclResult_t(ncclIbQp*)> g_resiliencyQpRtr = [](ncclIbQp*) { return ncclSystemError; };
std::function<ncclResult_t(ncclIbQp*)> g_resiliencyQpRts = [](ncclIbQp*) { return ncclSystemError; };
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

uint64_t TestResiliencyClockNano() { return g_resiliencyNow; }
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
ncclResult_t TestResiliencyQpReset(ncclIbQp* qp) { return g_resiliencyQpReset(qp); }
ncclResult_t TestResiliencyQpCreate(ncclIbQp* qp, ncclIbQpCreateAttr* attr) {
  return g_resiliencyQpCreate(qp, attr);
}
ncclResult_t TestResiliencyQpInit(ncclIbQp* qp) { return g_resiliencyQpInit(qp); }
ncclResult_t TestResiliencyQpRtr(ncclIbQp* qp) { return g_resiliencyQpRtr(qp); }
ncclResult_t TestResiliencyQpRts(ncclIbQp* qp) { return g_resiliencyQpRts(qp); }

int ResiliencyProbePostSend(ibv_qp*, ibv_send_wr* wr, ibv_send_wr** badWr) {
  if (g_resiliencyProbePostResult != 0) {
    if (badWr) *badWr = wr;
    return g_resiliencyProbePostResult;
  }
  ResiliencyProbePost post{};
  post.wrId = wr->wr_id;
  post.remoteAddress = wr->wr.rdma.remote_addr;
  post.rkey = wr->wr.rdma.rkey;
  if (wr->num_sge && wr->sg_list) {
    post.localAddress = wr->sg_list->addr;
    post.length = wr->sg_list->length;
    post.lkey = wr->sg_list->lkey;
  }
  g_resiliencyProbePosts.push_back(post);
  return 0;
}

int ResiliencyProbePollCq(ibv_cq*, int maxCount, ibv_wc* completions) {
  int count = std::min<int>(maxCount, g_resiliencyProbeCompletions.size());
  for (int i = 0; i < count; ++i) {
    completions[i] = g_resiliencyProbeCompletions.front();
    g_resiliencyProbeCompletions.pop_front();
  }
  return count;
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
#define clockNano TestResiliencyClockNano
#define ncclIbMultiSend TestResiliencyMultiSend
#define ncclIbPostFifo TestResiliencyPostFifo
#define ncclIbAddEvent TestResiliencyAddEvent
#define ncclIbQpCreate TestResiliencyQpCreate
#define ncclIbQpReset TestResiliencyQpReset
#define ncclIbQpInit TestResiliencyQpInit
#define ncclIbQpRtr TestResiliencyQpRtr
#define ncclIbQpRts TestResiliencyQpRts
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

struct ResiliencyScene {
  ncclIbSendComm send{};
  ncclIbRecvComm recv{};
  ncclIbResiliencySend sendRes{};
  ncclIbResiliency recvRes{};
  std::array<ibv_qp, 4> verbsQps{};

  ResiliencyScene() {
    send.base.isSend = true;
    recv.base.isSend = false;
    send.base.resiliency = &sendRes.base;
    recv.base.resiliency = &recvRes;
    sendRes.base.baseComm = &send.base;
    recvRes.baseComm = &recv.base;
    SetDeviceCount(&sendRes.base, 2);
    SetDeviceCount(&recvRes, 2);
    for (int i = 0; i < 4; ++i) {
      verbsQps[i].qp_num = 100 + i;
      send.base.qps[i].qp = &verbsQps[i];
      send.base.qps[i].devIndex = i % 2;
      send.base.activeQps[i] = &send.base.qps[i];
      recv.base.qps[i].qp = &verbsQps[i];
      recv.base.qps[i].devIndex = i % 2;
      recv.base.activeQps[i] = &recv.base.qps[i];
    }
    send.base.nqps = recv.base.nqps = 4;
    send.base.nDataQps = recv.base.nDataQps = 2;
    send.base.vProps.ndevs = recv.base.vProps.ndevs = 2;
  }

  static void SetDeviceCount(ncclIbResiliency* resiliency, int count) {
    resiliency->ndevs = count;
    for (int i = 0; i < count; ++i) {
      resiliency->devs[i].state.store(ncclIbResiliencyDevStateOk, std::memory_order_release);
    }
  }
};

class NetIbP2pResiliencyMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (int i = 0; i < 2; ++i) {
      savedPortNums_[i] = ncclIbDevs[i].portNum;
      savedMtus_[i] = ncclIbDevs[i].portAttr.active_mtu;
      savedGidIndices_[i] = ncclIbDevs[i].gidInfo.localGidIndex;
    }
  }

  void TearDown() override {
    for (int i = 0; i < 2; ++i) {
      ncclIbDevs[i].portNum = savedPortNums_[i];
      ncclIbDevs[i].portAttr.active_mtu = savedMtus_[i];
      ncclIbDevs[i].gidInfo.localGidIndex = savedGidIndices_[i];
    }
    g_resiliencyNow = 1000;
    g_resiliencyMultiSendCalls = 0;
    g_resiliencyPostFifoCalls = 0;
    g_resiliencyAddEventCalls = 0;
    g_resiliencyProbePosts.clear();
    g_resiliencyProbeCompletions.clear();
    g_resiliencyProbePostResult = 0;
    g_resiliencyQpReset = [](ncclIbQp*) { return ncclSystemError; };
    g_resiliencyQpCreate = [](ncclIbQp*, ncclIbQpCreateAttr*) { return ncclSystemError; };
    g_resiliencyQpInit = [](ncclIbQp*) { return ncclSystemError; };
    g_resiliencyQpRtr = [](ncclIbQp*) { return ncclSystemError; };
    g_resiliencyQpRts = [](ncclIbQp*) { return ncclSystemError; };
    g_resiliencyIbMallocSizes.clear();
    g_resiliencyIbMallocPoison = false;
    ResetTransportNetIbRecoveryFakes();
    ResetTransportNetIbResiliencyFakes();
    ResetIbvwrapFakes();
    ResetNcclFakes();
  }

 private:
  std::array<decltype(ncclIbDevs[0].portNum), 2> savedPortNums_{};
  std::array<decltype(ncclIbDevs[0].portAttr.active_mtu), 2> savedMtus_{};
  std::array<decltype(ncclIbDevs[0].gidInfo.localGidIndex), 2> savedGidIndices_{};
};

TEST_F(NetIbP2pResiliencyMicrotest, ErrorAdmission_FirstRetryOrFlushFailureWithHealthyPeerIsRecoverable) {
  ResiliencyScene scene;
  ibv_wc completion{};
  completion.status = IBV_WC_RETRY_EXC_ERR;

  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&scene.sendRes.base, &completion, 0), ncclSuccess);
  completion.status = IBV_WC_WR_FLUSH_ERR;
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&scene.sendRes.base, &completion, 0), ncclSuccess);
}

TEST_F(NetIbP2pResiliencyMicrotest, ErrorAdmission_UnsupportedStatusOrSecondFailureIsFatal) {
  ResiliencyScene scene;
  ibv_wc completion{};
  completion.status = IBV_WC_REM_ACCESS_ERR;
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&scene.sendRes.base, &completion, 0), ncclRemoteError);

  completion.status = IBV_WC_RETRY_EXC_ERR;
  scene.sendRes.base.devs[1].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);
  EXPECT_EQ(ncclIbResiliencyCheckErrorNotFatal(&scene.sendRes.base, &completion, 0), ncclRemoteError);
}

TEST_F(NetIbP2pResiliencyMicrotest, ReplaceQps_FailedDeviceMovesToFirstHealthyAlternative) {
  ResiliencyScene scene;
  scene.sendRes.base.devs[0].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);

  ASSERT_EQ(ncclIbResiliencyReplaceQps(&scene.sendRes.base, 0), ncclSuccess);
  EXPECT_EQ(scene.send.base.activeQps[0], &scene.send.base.qps[1]);
  EXPECT_EQ(scene.send.base.activeQps[1], &scene.send.base.qps[1]);
  EXPECT_EQ(scene.send.base.activeQps[2], &scene.send.base.qps[3]);
  EXPECT_EQ(scene.send.base.activeQps[3], &scene.send.base.qps[3]);
}

TEST_F(NetIbP2pResiliencyMicrotest, ReplaceQps_NoHealthyAlternativeReturnsError) {
  ResiliencyScene scene;
  scene.sendRes.base.devs[0].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);
  scene.sendRes.base.devs[1].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);

  EXPECT_EQ(ncclIbResiliencyReplaceQps(&scene.sendRes.base, 0), ncclInternalError);
}

TEST_F(NetIbP2pResiliencyMicrotest, SendRequestTracking_NewGenerationIncrementsAndFreePreservesGeneration) {
  ResiliencyScene scene;
  auto& request = scene.send.base.reqs[3];
  request.base = &scene.send.base;
  request.id = 3;
  request.type = NCCL_NET_IB_REQ_SEND;
  g_resiliencyNow = 9000;

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
  ResiliencyScene scene;
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
  ResiliencyScene scene;
  auto& request = scene.recv.base.reqs[2];
  request.base = &scene.recv.base;
  request.id = 2;
  request.type = NCCL_NET_IB_REQ_RECV;

  EXPECT_EQ(ncclIbResiliencyRepostRequest(&request), ncclSuccess);
  EXPECT_EQ(g_resiliencyPostFifoCalls, 1);
}

TEST_F(NetIbP2pResiliencyMicrotest, RepostRequest_SendSkipsDeliveredQpsAndReaddsMissingEvents) {
  ResiliencyScene scene;
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
  ResiliencyScene scene;
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

TEST_F(NetIbP2pResiliencyMicrotest, DeviceNumSet_AssignsSenderAndReceiverProbeCounts) {
  ResiliencyScene scene;
  scene.sendRes.base.recoveryEnabled = true;
  ASSERT_EQ(NetIbResiliencyDeviceNumSet(&scene.sendRes.base, 2, 3), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.ndevs, 2);
  EXPECT_EQ(scene.sendRes.base.nProbingQps, 2);
  EXPECT_EQ(scene.sendRes.base.nPortRecoveryQps, 3);

  scene.recvRes.recoveryEnabled = true;
  ASSERT_EQ(NetIbResiliencyDeviceNumSet(&scene.recvRes, 2, 3), ncclSuccess);
  EXPECT_EQ(scene.recvRes.ndevs, 2);
  EXPECT_EQ(scene.recvRes.nProbingQps, 3);
  EXPECT_EQ(scene.recvRes.nPortRecoveryQps, 3);
}

TEST_F(NetIbP2pResiliencyMicrotest, DeviceNumSet_InvalidCountsAreRejected) {
  ResiliencyScene scene;
  EXPECT_EQ(NetIbResiliencyDeviceNumSet(nullptr, 1, 1), ncclInternalError);
  EXPECT_EQ(NetIbResiliencyDeviceNumSet(&scene.recvRes, 0, 1), ncclInternalError);
  EXPECT_EQ(NetIbResiliencyDeviceNumSet(&scene.recvRes, 1, 0), ncclInternalError);
  EXPECT_EQ(NetIbResiliencyDeviceNumSet(&scene.recvRes, 1, NCCL_IB_MAX_DEVS_PER_NIC + 1), ncclInternalError);
}

TEST_F(NetIbP2pResiliencyMicrotest, DataQueueSizing_AccountsForDirectionQpsAndFailoverDevices) {
  ResiliencyScene scene;
  scene.send.base.nqps = scene.send.base.nDataQps = 2;
  scene.send.base.splitDataOnQps = 1;
  int cqSize = 0;
  ASSERT_EQ(NetIbResiliencyDataCqSizeGet(&scene.sendRes.base, 0, &cqSize), ncclSuccess);
  EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * 2 * 2);

  scene.recv.base.nqps = scene.recv.base.nDataQps = 2;
  scene.recv.base.splitDataOnQps = 1;
  ASSERT_EQ(NetIbResiliencyDataCqSizeGet(&scene.recvRes, 0, &cqSize), ncclSuccess);
  EXPECT_EQ(cqSize, NET_IB_MAX_REQUESTS * 2 * 2 * 2);
  uint32_t rqSize = 0;
  ASSERT_EQ(NetIbResiliencyDataRqSizeGet(&scene.recvRes, 0, &rqSize), ncclSuccess);
  EXPECT_EQ(rqSize, NET_IB_MAX_REQUESTS * 2);
}

TEST_F(NetIbP2pResiliencyMicrotest, RemoteCompletionRecords_SendOnlyStoresPerDeviceInfo) {
  ResiliencyScene scene;
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
  ResiliencyScene scene;
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
  ResiliencyScene scene;
  ncclIbNetCommDevBase devBase{};
  devBase.gidInfo.localGidIndex = 4;
  std::array<char, 4> order{};
  int cursor = 0;
  g_resiliencyQpReset = [&order, &cursor](ncclIbQp*) {
    order[cursor++] = 'R';
    return ncclSuccess;
  };
  g_resiliencyQpInit = [&order, &cursor](ncclIbQp*) {
    order[cursor++] = 'I';
    return ncclSuccess;
  };
  g_resiliencyQpRtr = [&order, &cursor](ncclIbQp*) {
    order[cursor++] = 'T';
    return ncclSuccess;
  };
  g_resiliencyQpRts = [&order, &cursor](ncclIbQp*) {
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

TEST_F(NetIbP2pResiliencyMicrotest, QpReconfigure_StageFailureReportsNotReconfigured) {
  ResiliencyScene scene;
  ncclIbNetCommDevBase devBase{};
  g_resiliencyQpReset = [](ncclIbQp*) { return ncclSuccess; };
  g_resiliencyQpInit = [](ncclIbQp*) { return ncclSystemError; };
  bool success = true;

  EXPECT_EQ(NetIbResiliencyQpReconfigure(&scene.sendRes.base, &scene.send.base.qps[0], &devBase, 0, &success),
            ncclSuccess);
  EXPECT_FALSE(success);
}

TEST_F(NetIbP2pResiliencyMicrotest, Progress_RecoveredDeviceRestoresOriginalQpsAndCompletes) {
  ResiliencyScene scene;
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
  ResiliencyScene scene;
  scene.recvRes.inProgress = true;
  scene.recvRes.outstandingRecovery = 1;
  scene.recvRes.devs[1].state.store(ncclIbResiliencyDevStateRecoveryFailed, std::memory_order_release);

  ASSERT_EQ(NetIbResiliencyProgress(&scene.recvRes), ncclSuccess);
  EXPECT_EQ(scene.recvRes.devs[1].state.load(), ncclIbResiliencyDevStateErrorPermanent);
  EXPECT_FALSE(scene.recvRes.inProgress);
}

TEST_F(NetIbP2pResiliencyMicrotest, DevInitAndDestroyManageProbeMrAndCq) {
  ResiliencyScene scene;
  ibv_context context{};
  ibv_pd pd{};
  ibv_mr mr{};
  ibv_cq cq{};
  scene.send.devs[0].base.pd = &pd;
  ncclIbDev dev{};
  dev.context = &context;
  ScopedHook reg(g_ibvRegMr, [&mr](ibv_mr** out, ibv_pd*, void*, size_t, int) {
    *out = &mr;
    return ncclSuccess;
  });
  ScopedHook create(g_ibvCreateCq,
                    [&cq](ibv_cq** out, ibv_context*, int size, void*, ibv_comp_channel*, int) {
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
  ResiliencyScene scene;
  scene.sendRes.base.nProbingQps = 2;
  ibv_pd pds[2] = {};
  ibv_cq cqs[2] = {};
  int created = 0;
  for (int i = 0; i < 2; ++i) {
    scene.send.devs[i].base.ibDevN = i;
    scene.send.devs[i].base.pd = &pds[i];
    scene.sendRes.base.devs[i].probingCq = &cqs[i];
    ncclIbDevs[i].portNum = i + 1;
    scene.verbsQps[i].qp_num = 200 + i;
  }
  g_resiliencyQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr* attr) {
    EXPECT_EQ(attr->maxRecvWorkRequest, 0u);
    EXPECT_EQ(attr->maxSendWorkRequest, static_cast<uint32_t>(NET_IB_MAX_REQUESTS));
    qp->qp = &scene.verbsQps[created++];
    return ncclSuccess;
  };
  std::vector<ncclIbQp*> initialized;
  g_resiliencyQpInit = [&](ncclIbQp* qp) {
    initialized.push_back(qp);
    return ncclSuccess;
  };
  ncclIbResiliencyInfo info{};

  ASSERT_EQ(NetIbResiliencySenderCreateQps(&scene.sendRes.base, &info), ncclSuccess);
  EXPECT_EQ(created, 2);
  EXPECT_EQ(scene.sendRes.base.probingQps[0].devIndex, 0);
  EXPECT_EQ(scene.sendRes.base.probingQps[1].devIndex, 1);
  EXPECT_EQ(info.probingQpsInfo[0].qpn, 200u);
  EXPECT_EQ(info.probingQpsInfo[1].qpn, 201u);
  EXPECT_EQ(initialized, (std::vector<ncclIbQp*>{&scene.sendRes.base.probingQps[0],
                                                &scene.sendRes.base.probingQps[1]}));
}

TEST_F(NetIbP2pResiliencyMicrotest, SenderProbeQpsToRtsUsesRemoteMetadata) {
  ResiliencyScene scene;
  scene.sendRes.base.nProbingQps = 1;
  scene.sendRes.base.probingQps[0].qp = &scene.verbsQps[0];
  scene.sendRes.base.probingQps[0].devIndex = 0;
  scene.send.devs[0].base.ibDevN = 0;
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
  g_resiliencyQpRtr = [&](ncclIbQp* qp) {
    EXPECT_EQ(qp, &scene.sendRes.base.probingQps[0]);
    transitions.push_back('T');
    return ncclSuccess;
  };
  g_resiliencyQpRts = [&](ncclIbQp* qp) {
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
  EXPECT_EQ(transitions, (std::vector<char>{'T', 'S'}));
}

// AICOMRCCL-3129: the AV port is a local attribute and must not be copied
// from the remote device metadata.
TEST_F(NetIbP2pResiliencyMicrotest, DISABLED_SenderProbeQpsToRtsUsesLocalIbPort) {
  ResiliencyScene scene;
  scene.sendRes.base.nProbingQps = 1;
  scene.sendRes.base.probingQps[0].qp = &scene.verbsQps[0];
  scene.send.devs[0].base.ibDevN = 0;
  ncclIbDevs[0].portNum = 3;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  ncclIbConnectionMetadata remote{};
  remote.resiliencyInfo.probingQpsInfo[0].qpn = 300;
  remote.resiliencyInfo.probingQpsInfo[0].devIndex = 0;
  remote.devs[0].mtu = IBV_MTU_2048;
  remote.devs[0].link_layer = IBV_LINK_LAYER_ETHERNET;
  remote.devs[0].ib_port = 7;
  g_resiliencyQpRtr = [](ncclIbQp*) { return ncclSuccess; };
  g_resiliencyQpRts = [](ncclIbQp*) { return ncclSuccess; };

  ASSERT_EQ(NetIbResiliencySenderQpsToRts(&scene.sendRes.base, &remote), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.probingQps[0].rtrAttr.localIbPort, 3);
}

// AICOMRCCL-3129: receiver probing QPs have the same local AV-port contract.
TEST_F(NetIbP2pResiliencyMicrotest, DISABLED_ReceiverProbeQpsToRtsUsesLocalIbPort) {
  ResiliencyScene scene;
  scene.recvRes.nProbingQps = 1;
  scene.recv.base.vProps.ndevs = 1;
  scene.recv.devs[0].base.ibDevN = 0;
  ibv_pd pd{};
  ibv_cq cq{};
  scene.recv.devs[0].base.pd = &pd;
  scene.recvRes.devs[0].probingCq = &cq;
  ncclIbDevs[0].portNum = 3;
  ncclIbDevs[0].portAttr.active_mtu = IBV_MTU_4096;
  g_resiliencyQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr*) {
    qp->qp = &scene.verbsQps[0];
    return ncclSuccess;
  };
  g_resiliencyQpInit = [](ncclIbQp*) { return ncclSuccess; };
  g_resiliencyQpRtr = [](ncclIbQp*) { return ncclSuccess; };
  g_resiliencyQpRts = [](ncclIbQp*) { return ncclSuccess; };
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
  ResiliencyScene scene;
  scene.recvRes.nProbingQps = 3;
  scene.recv.base.vProps.ndevs = 2;
  ibv_pd pds[2] = {};
  ibv_cq cqs[2] = {};
  for (int i = 0; i < 2; ++i) {
    scene.recv.devs[i].base.ibDevN = i;
    scene.recv.devs[i].base.pd = &pds[i];
    scene.recvRes.devs[i].probingCq = &cqs[i];
    ncclIbDevs[i].portNum = i + 1;
    ncclIbDevs[i].portAttr.active_mtu = IBV_MTU_4096;
  }
  int created = 0;
  g_resiliencyQpCreate = [&](ncclIbQp* qp, ncclIbQpCreateAttr*) {
    qp->qp = &scene.verbsQps[created++];
    return ncclSuccess;
  };
  std::vector<char> transitions;
  std::vector<ncclIbQp*> transitionedQps;
  g_resiliencyQpInit = [&](ncclIbQp* qp) {
    transitions.push_back('I');
    transitionedQps.push_back(qp);
    return ncclSuccess;
  };
  g_resiliencyQpRtr = [&](ncclIbQp* qp) {
    transitions.push_back('T');
    transitionedQps.push_back(qp);
    return ncclSuccess;
  };
  g_resiliencyQpRts = [&](ncclIbQp* qp) {
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
  EXPECT_EQ(created, 3);
  EXPECT_EQ(scene.recvRes.probingQps[0].devIndex, 0);
  EXPECT_EQ(scene.recvRes.probingQps[1].devIndex, 1);
  EXPECT_EQ(scene.recvRes.probingQps[2].devIndex, 0);
  EXPECT_EQ(local.probingQpsInfo[0].qpn, scene.verbsQps[0].qp_num);
  EXPECT_EQ(local.probingQpsInfo[1].qpn, scene.verbsQps[1].qp_num);
  EXPECT_EQ(local.probingQpsInfo[2].qpn, scene.verbsQps[2].qp_num);
  EXPECT_EQ(transitions, (std::vector<char>{'I', 'T', 'S', 'I', 'T', 'S', 'I', 'T', 'S'}));
  EXPECT_EQ(transitionedQps,
            (std::vector<ncclIbQp*>{&scene.recvRes.probingQps[0], &scene.recvRes.probingQps[0],
                                    &scene.recvRes.probingQps[0], &scene.recvRes.probingQps[1],
                                    &scene.recvRes.probingQps[1], &scene.recvRes.probingQps[1],
                                    &scene.recvRes.probingQps[2], &scene.recvRes.probingQps[2],
                                    &scene.recvRes.probingQps[2]}));
}

TEST_F(NetIbP2pResiliencyMicrotest, QpsReconfigure_DelegatesRecoveryAfterProbeQpsSucceed) {
  ResiliencyScene scene;
  scene.sendRes.base.nProbingQps = 1;
  scene.sendRes.base.probingQps[0].qp = &scene.verbsQps[0];
  scene.sendRes.base.probingQps[0].devIndex = 0;
  scene.send.devs[0].base.ibDevN = 0;
  ncclIbDevs[0].gidInfo.localGidIndex = 6;
  g_resiliencyQpReset = [](ncclIbQp*) { return ncclSuccess; };
  g_resiliencyQpInit = [](ncclIbQp*) { return ncclSuccess; };
  g_resiliencyQpRtr = [](ncclIbQp*) { return ncclSuccess; };
  g_resiliencyQpRts = [](ncclIbQp*) { return ncclSuccess; };
  ScopedHook recovery(g_ncclIbPortRecoveryQpsReconfigure,
                      [](ncclIbResiliency*, int, bool* success) {
                        *success = true;
                        return ncclSuccess;
                      });
  bool success = false;

  ASSERT_EQ(NetIbResiliencyQpsReconfigure(&scene.sendRes.base, 0, &success), ncclSuccess);
  EXPECT_TRUE(success);
  EXPECT_EQ(recovery.calls, 1);
  EXPECT_EQ(scene.sendRes.base.probingQps[0].rtrAttr.localGidIndex, 6);
}

TEST_F(NetIbP2pResiliencyMicrotest, Close_DestroysProbeQpsAndRecoveryResources) {
  ResiliencyScene scene;
  scene.sendRes.base.nProbingQps = 2;
  scene.sendRes.base.nPortRecoveryQps = 2;
  scene.sendRes.base.recoveryEnabled = true;
  scene.sendRes.base.probingQps[0].qp = &scene.verbsQps[0];
  scene.sendRes.base.probingQps[1].qp = &scene.verbsQps[1];
  std::vector<ibv_qp*> destroyed;
  ScopedHook destroy(g_ibvDestroyQp, [&](ibv_qp* qp) {
    destroyed.push_back(qp);
    return ncclSuccess;
  });
  ScopedHook close(g_ncclIbPortRecoveryClose, [](ncclIbResiliency*) { return ncclSuccess; });
  ScopedHook destroyRecovery(g_ncclIbPortRecoveryQpsDestroy,
                             [](ncclIbResiliency*, int count) {
                               EXPECT_EQ(count, 2);
                               return ncclSuccess;
                             });

  EXPECT_EQ(NetIbResiliencyClose(&scene.sendRes.base), ncclSuccess);
  EXPECT_EQ(destroy.calls, 2);
  EXPECT_EQ(destroyed, (std::vector<ibv_qp*>{&scene.verbsQps[0], &scene.verbsQps[1]}));
  EXPECT_EQ(close.calls, 1);
  EXPECT_EQ(destroyRecovery.calls, 1);
}

TEST_F(NetIbP2pResiliencyMicrotest, HandleCompletionError_SenderMarksDeviceAndTracksRequest) {
  ResiliencyScene scene;
  auto& request = scene.send.base.reqs[1];
  request.base = &scene.send.base;
  request.id = 1;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.nreqs = 1;
  scene.send.sendReqs[1][0] = &request;
  ibv_wc completion{};
  completion.status = IBV_WC_RETRY_EXC_ERR;
  completion.wr_id = 1;
  g_resiliencyNow = 7000;

  ASSERT_EQ(NetIbResiliencyHandleCompletionError(&scene.sendRes.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.devs[0].state.load(), ncclIbResiliencyDevStateError);
  EXPECT_EQ(scene.sendRes.failedRequests[1].request, &request);
  EXPECT_EQ(scene.sendRes.base.outstandingRequests, 1);
}

// AICOMRCCL-3127: slot-only wr_id values cannot distinguish a delayed CQE
// from the request generation that subsequently reused the same slot.
TEST_F(NetIbP2pResiliencyMicrotest, DISABLED_HandleCompletionError_StaleGenerationDoesNotTrackCurrentRequest) {
  ResiliencyScene scene;
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
  ResiliencyScene scene;
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
  ResiliencyScene scene;
  ibv_context context{};
  context.ops.post_send = ResiliencyProbePostSend;
  scene.verbsQps[1].context = &context;
  scene.sendRes.base.devs[0].state.store(ncclIbResiliencyDevStateError, std::memory_order_release);
  scene.sendRes.base.probingQps[1].qp = &scene.verbsQps[1];
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

  g_resiliencyNow = 1000 + 10 * MSEC_TO_NSEC - 1;
  ASSERT_EQ(ncclIbResiliencyProbePost(&scene.sendRes, &failed), ncclSuccess);
  EXPECT_TRUE(g_resiliencyProbePosts.empty());
  EXPECT_EQ(failed.state, ncclIbResiliencyRequestStatePending);

  g_resiliencyNow++;
  ASSERT_EQ(ncclIbResiliencyProbePost(&scene.sendRes, &failed), ncclSuccess);
  ASSERT_EQ(g_resiliencyProbePosts.size(), 1u);
  EXPECT_EQ(g_resiliencyProbePosts[0].wrId, 3u);
  EXPECT_EQ(g_resiliencyProbePosts[0].remoteAddress,
            0x100000u + sizeof(ncclIbRequestCompletionRecord) * 3 + sizeof(int) * NCCL_NET_IB_MAX_RECVS);
  EXPECT_EQ(g_resiliencyProbePosts[0].rkey, 0x52u);
  EXPECT_EQ(g_resiliencyProbePosts[0].lkey, 0x51u);
  EXPECT_EQ(failed.state, ncclIbResiliencyRequestStateProbePosted);
  EXPECT_EQ(scene.sendRes.base.devs[1].nOutstandingProbes, 1);
}

TEST_F(NetIbP2pResiliencyMicrotest, ProbeCompletion_SuccessMarksRequestCompleted) {
  ResiliencyScene scene;
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

TEST_F(NetIbP2pResiliencyMicrotest, ProbeProgress_DrainsCompletionAndDecrementsOutstandingCount) {
  ResiliencyScene scene;
  ibv_context context{};
  context.ops.poll_cq = ResiliencyProbePollCq;
  ibv_cq cq{};
  cq.context = &context;
  scene.sendRes.base.devs[1].probingCq = &cq;
  scene.sendRes.base.devs[1].nOutstandingProbes = 1;
  auto& request = scene.send.base.reqs[5];
  request.id = 5;
  scene.sendRes.failedRequests[5].request = &request;
  scene.sendRes.failedRequests[5].state = ncclIbResiliencyRequestStateProbePosted;
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.wr_id = 5;
  g_resiliencyProbeCompletions.push_back(completion);

  ASSERT_EQ(ncclIbResiliencyProbeProgress(&scene.sendRes), ncclSuccess);
  EXPECT_EQ(scene.sendRes.base.devs[1].nOutstandingProbes, 0);
  EXPECT_EQ(scene.sendRes.failedRequests[5].state, ncclIbResiliencyRequestStateProbeCompleted);
}

TEST_F(NetIbP2pResiliencyMicrotest, ProbeConclusion_AllDeliveredClearsSiblingEvents) {
  ResiliencyScene scene;
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

}  // namespace
