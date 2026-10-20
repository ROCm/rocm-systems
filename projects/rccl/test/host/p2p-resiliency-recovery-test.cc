/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/p2p_resiliency_recovery.cc (via
// P2P_RESILIENCY_RECOVERY_CC_PATH): the UD recovery QPs, the alive/ACK protocol and its thread.

#include <gtest/gtest.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <list>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common/LogCapture.hpp"
#include "P2pResiliencyScene.h"
#include "fakes/param_redirect.h"

// The recovery context is the unit's only plain malloc; the seam lets a test fail it.
static bool g_recoveryMallocFails = false;
static void* P2pResiliencyRecoveryTestMalloc(size_t size) { return g_recoveryMallocFails ? nullptr : malloc(size); }

#define clockNano P2pResiliencyTestClockNano
#define malloc P2pResiliencyRecoveryTestMalloc
#include P2P_RESILIENCY_RECOVERY_CC_PATH
#undef malloc
#undef clockNano

void ncclSetThreadName(pthread_t, const char*, ...) {}

// ThreadStop clears the flag without the mutex, so its wakeup can be lost (AICOMRCCL-3103).
// Clearing it under the mutex first makes the stop deterministic.
void P2pResiliencyStopRecoveryThread() {
  if (IbCastPortRecoveryRefCount.load() == 1) {
    std::lock_guard<std::mutex> lock(IbCastPortRecoveryMutex);
    IbCastPortRecoveryThreadActive.store(false);
  }
  EXPECT_EQ(IbCastPortRecoveryThreadStop(), ncclSuccess);
}

int P2pResiliencyDrainRecoveryInbox() {
  std::list<ncclIbPortRecoveryContext*> queued;
  {
    std::lock_guard<std::mutex> lock(IbCastPortRecoveryMutex);
    queued.swap(recoveryInbox);
  }
  for (ncclIbPortRecoveryContext* ctx : queued) IbCastPortRecoveryContextDestroy(ctx);
  return queued.size();
}

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

// Protocol timings this suite runs with; each differs from production's default.
constexpr int64_t kStartDelayMs = 50;
constexpr int64_t kBatchIntervalMs = 30;
constexpr int64_t kAliveTimeoutMs = 400;
constexpr int64_t kAckTimeoutMs = 600;
constexpr int64_t kAttemptsMax = 3;
constexpr int64_t kSequenceSize = 4;

class P2pResiliencyRecoveryMicrotest : public ::testing::Test {
 protected:
  P2pResiliencyScene scene_;
  P2pResiliencyScene::CommPtr<ncclIbSendComm> send_;
  P2pResiliencyScene::CommPtr<ncclIbRecvComm> recv_;
  ncclIbResiliency* res_ = nullptr;
  ncclIbPortRecoveryContext* ctx_ = nullptr;
  std::function<int(ibv_cq*, int, ibv_wc*)> defaultPoll_;
  std::function<int(ibv_qp*, ibv_send_wr*)> defaultSend_;
  std::function<int(ibv_qp*, ibv_recv_wr*)> defaultRecv_;

  void SetUp() override {
    defaultPoll_ = scene_.pollCq;
    defaultSend_ = scene_.postSend;
    defaultRecv_ = scene_.postRecv;
    scene_.params["IB_RESILIENCY_PORT_RECOVERY_START_DELAY"] = kStartDelayMs;
    scene_.params["IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_INTERVAL"] = kBatchIntervalMs;
    scene_.params["IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_TIMEOUT"] = kAliveTimeoutMs;
    scene_.params["IB_RESILIENCY_PORT_RECOVERY_ACK_TIMEOUT"] = kAckTimeoutMs;
    scene_.params["IB_RESILIENCY_PORT_RECOVERY_ATTEMPTS_MAX"] = kAttemptsMax;
    scene_.params["IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE"] = kSequenceSize;
  }

  void TearDown() override {
    if (ctx_ != nullptr) IbCastPortRecoveryContextDestroy(ctx_);
    IbCastResiliencyDestroy(&res_);
  }

  // Back to no comm, for tests that loop over several setups.
  void Reset() {
    scene_.pollCq = defaultPoll_;
    scene_.postSend = defaultSend_;
    scene_.postRecv = defaultRecv_;
    if (ctx_ != nullptr) IbCastPortRecoveryContextDestroy(ctx_);
    ctx_ = nullptr;
    IbCastResiliencyDestroy(&res_);
    send_.reset();
    recv_.reset();
  }

  ncclIbNetCommBase* Base() { return send_ ? &send_->base : &recv_->base; }

  // A comm with port recovery fully connected, nqps data QPs striped over ndevs devices.
  ncclIbResiliency* Make(bool isSend, int ndevs, int nqps) {
    if (isSend) {
      send_ = scene_.NewComm<ncclIbSendComm>(true, ndevs, nqps);
    } else {
      recv_ = scene_.NewComm<ncclIbRecvComm>(false, ndevs, nqps);
    }
    res_ = P2pResiliencyAttach(Base(), ndevs, true);
    ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(ndevs, true);
    ncclIbResiliencyInfo info = {};
    if (isSend) {
      EXPECT_EQ(IbCastResiliencySenderCreateQps(res_, &info), ncclSuccess);
      EXPECT_EQ(IbCastResiliencySenderQpsToRts(res_, &remote), ncclSuccess);
    } else {
      EXPECT_EQ(IbCastResiliencyReceiverQpsCreateToRts(res_, &remote, &info), ncclSuccess);
    }
    ClearRecords();
    return res_;
  }

  void ClearRecords() {
    scene_.sends.clear();
    scene_.recvs.clear();
    scene_.modifies.clear();
    scene_.transitions.clear();
    scene_.qpCreates.clear();
    scene_.destroyedQps.clear();
    scene_.postedRecvWrQps.clear();
  }

  // Recovery of device dev begins: the context exists and the start delay has passed.
  void Start(int dev) {
    res_->devs[dev].state.store(ncclIbResiliencyDevStateRecoveryInProgress);
    ASSERT_EQ(IbCastPortRecoveryContextInit(res_, dev, &ctx_), ncclSuccess);
    ClearRecords();
    P2pResiliencyAdvanceMs(kStartDelayMs);
  }

  bool Step(ncclResult_t expect = ncclSuccess) {
    bool done = false;
    EXPECT_EQ(IbCastPortRecoveryContextProgress(ctx_, &done), expect);
    return done;
  }

  void Cqe(ibv_wc_opcode opcode, uint32_t imm = 0, ibv_wc_status status = IBV_WC_SUCCESS) {
    scene_.PushCqe(ctx_->recoveryCq, P2pResiliencyWc(0, status, opcode, imm));
  }

  ncclIbResiliencyDevState State(int d) { return res_->devs[d].state.load(); }

  int QpsOnDevice(int dev) {
    int n = 0;
    for (int q = 0; q < Base()->nqps; q++) n += Base()->qps[q].devIndex == dev;
    return n;
  }

  // Sender: from start through a completed alive batch, into the ACK state.
  void SenderToAck(int dev) {
    Start(dev);
    ASSERT_FALSE(Step());
    ASSERT_EQ(ctx_->state, ncclIbPortRecoveryStateAliveMessages);
    Cqe(IBV_WC_SEND);
    ASSERT_FALSE(Step());
    ASSERT_EQ(ctx_->state, ncclIbPortRecoveryStateAck);
  }

  // Receiver: from start through a full alive window to a posted (not yet completed) ACK.
  void ReceiverToAckPosted(int dev) {
    Start(dev);
    for (uint32_t id = 0; id < kSequenceSize; id++) {
      Cqe(IBV_WC_RECV, id);
      ASSERT_FALSE(Step());
    }
    ASSERT_TRUE(ctx_->ackPosted);
    ASSERT_EQ(ctx_->nFailedAttempts, 0);
  }
};

TEST_F(P2pResiliencyRecoveryMicrotest, QpUdToRts_ProgramsInitRtrAndRtsInOrder) {
  scene_.params["IB_PKEY"] = 4;
  ncclIbQp qp = {};
  qp.qp = scene_.NewQp();
  ASSERT_EQ(IbCastPortRecoveryQpUdToRts(&qp, 2), ncclSuccess);
  ASSERT_EQ(scene_.modifies.size(), 3u);
  EXPECT_EQ(scene_.modifies[0].attr.qp_state, IBV_QPS_INIT);
  EXPECT_EQ(scene_.modifies[0].attr.pkey_index, 4);
  EXPECT_EQ(scene_.modifies[0].attr.port_num, 2);
  EXPECT_EQ(scene_.modifies[0].attr.qkey, (uint32_t)NCCL_IB_RECOVERY_QKEY);
  EXPECT_EQ(scene_.modifies[0].mask, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
  EXPECT_EQ(scene_.modifies[1].attr.qp_state, IBV_QPS_RTR);
  EXPECT_EQ(scene_.modifies[1].mask, IBV_QP_STATE);
  EXPECT_EQ(scene_.modifies[2].attr.qp_state, IBV_QPS_RTS);
  EXPECT_EQ(scene_.modifies[2].attr.sq_psn, 0u);
  EXPECT_EQ(scene_.modifies[2].mask, IBV_QP_STATE | IBV_QP_SQ_PSN);
  for (const auto& m : scene_.modifies) EXPECT_EQ(m.qp, qp.qp);
}

TEST_F(P2pResiliencyRecoveryMicrotest, QpUdToRts_TransitionFails_StopsThere) {
  for (int failAt = 0; failAt < 3; failAt++) {
    ncclIbQp qp = {};
    qp.qp = scene_.NewQp();
    int n = 0;
    ScopedHook modify(g_ibvModifyQp,
                      [&](ibv_qp*, ibv_qp_attr*, int) { return n++ == failAt ? ncclSystemError : ncclSuccess; });
    EXPECT_EQ(IbCastPortRecoveryQpUdToRts(&qp, 1), ncclSystemError) << failAt;
    EXPECT_EQ(n, failAt + 1);
  }
}

TEST_F(P2pResiliencyRecoveryMicrotest, DevInit_CreatesRecoveryCqAndRegistersBuffersOnTheDevicePd) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 1, 1);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  ibv_pd devicePd = {};
  scene_.Dev(0)->pd = &devicePd;
  std::memset(res_->devs[0].portRecoveryGrhBuf, 0xEE, sizeof(res_->devs[0].portRecoveryGrhBuf));
  ASSERT_EQ(IbCastPortRecoveryDevInit(res_, 0, scene_.Dev(0)), ncclSuccess);
  ASSERT_EQ(scene_.createCqs.size(), 1u);
  EXPECT_EQ(scene_.createCqs[0].context, scene_.Dev(0)->context);
  EXPECT_EQ(scene_.createCqs[0].cqe, NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE);
  EXPECT_EQ(scene_.createCqs[0].cqContext, (void*)&send_->base.stats);
  EXPECT_EQ(res_->devs[0].portRecoveryCq, scene_.createCqs[0].cq);
  ASSERT_EQ(scene_.regMrs.size(), 2u);
  EXPECT_EQ(scene_.regMrs[0].pd, &devicePd);
  EXPECT_EQ(scene_.regMrs[0].addr, (void*)res_->devs[0].portRecoveryGrhBuf);
  EXPECT_EQ(scene_.regMrs[0].length, sizeof(res_->devs[0].portRecoveryGrhBuf));
  EXPECT_EQ(scene_.regMrs[0].access, IBV_ACCESS_LOCAL_WRITE);
  EXPECT_EQ(res_->devs[0].portRecoveryGrhMr, scene_.regMrs[0].mr);
  EXPECT_EQ(scene_.regMrs[1].pd, &devicePd);
  EXPECT_EQ(scene_.regMrs[1].addr, (void*)res_->portRecoveryQpnBuf);
  EXPECT_EQ(scene_.regMrs[1].length, sizeof(res_->portRecoveryQpnBuf));
  EXPECT_EQ(res_->devs[0].portRecoveryQpnMr, scene_.regMrs[1].mr);
  for (uint8_t b : res_->devs[0].portRecoveryGrhBuf) ASSERT_EQ(b, 0);
}

TEST_F(P2pResiliencyRecoveryMicrotest, DevInit_VerbsFailure_ReturnsTheError) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 1, 1);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  {
    ScopedHook cq(g_ibvCreateCq,
                  [](ibv_cq**, ibv_context*, int, void*, ibv_comp_channel*, int) { return ncclSystemError; });
    EXPECT_EQ(IbCastPortRecoveryDevInit(res_, 0, scene_.Dev(0)), ncclSystemError);
    EXPECT_TRUE(scene_.regMrs.empty());
  }
  for (int failAt = 0; failAt < 2; failAt++) {
    int n = 0;
    ScopedHook reg(g_ibvRegMr, [&](ibv_mr** ret, ibv_pd*, void*, size_t, int) {
      *ret = scene_.NewMr(1);
      return n++ == failAt ? ncclSystemError : ncclSuccess;
    });
    EXPECT_EQ(IbCastPortRecoveryDevInit(res_, 0, scene_.Dev(0)), ncclSystemError) << failAt;
    EXPECT_EQ(n, failAt + 1);
  }
}

TEST_F(P2pResiliencyRecoveryMicrotest, DevDestroy_ReleasesEverythingAndClearsIt) {
  Make(true, 1, 1);
  ncclIbResiliencyDev* dev = &res_->devs[0];
  ibv_cq* cq = dev->portRecoveryCq;
  std::vector<ibv_mr*> mrs = {dev->portRecoveryGrhMr, dev->portRecoveryQpnMr};
  scene_.deregMrs.clear();
  ASSERT_EQ(IbCastPortRecoveryDevDestroy(res_, 0), ncclSuccess);
  EXPECT_EQ(scene_.destroyedCqs, (std::vector<ibv_cq*>{cq}));
  EXPECT_EQ(scene_.deregMrs, mrs);
  EXPECT_EQ(dev->portRecoveryCq, nullptr);
  EXPECT_EQ(dev->portRecoveryGrhMr, nullptr);
  EXPECT_EQ(dev->portRecoveryQpnMr, nullptr);
  ASSERT_EQ(IbCastPortRecoveryDevDestroy(res_, 0), ncclSuccess);
  EXPECT_EQ(scene_.destroyedCqs.size(), 1u);
  EXPECT_EQ(scene_.deregMrs.size(), 2u);
}

TEST_F(P2pResiliencyRecoveryMicrotest, DevDestroy_DestroyCqFails_ReturnsTheError) {
  Make(true, 1, 1);
  ScopedHook cq(g_ibvDestroyCq, [](ibv_cq*) { return ncclSystemError; });
  EXPECT_EQ(IbCastPortRecoveryDevDestroy(res_, 0), ncclSystemError);
  EXPECT_NE(res_->devs[0].portRecoveryCq, nullptr);
}

TEST_F(P2pResiliencyRecoveryMicrotest, SenderQpsCreate_StripesUdQpsOverDevicesAndInitsThem) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 2, 3), ncclSuccess);
  for (int d = 0; d < 2; d++) ASSERT_EQ(IbCastResiliencyDevInit(res_, d, scene_.Dev(d)), ncclSuccess);
  ncclIbQpInfo info[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  ASSERT_EQ(IbCastPortRecoverySenderQpsCreate(res_, info, 3), ncclSuccess);
  ASSERT_EQ(scene_.qpCreates.size(), 3u);
  ASSERT_EQ(scene_.modifies.size(), 3u);
  const int expectDev[3] = {0, 1, 0};
  for (int q = 0; q < 3; q++) {
    const ncclIbQpCreateAttr& attr = scene_.qpCreates[q].attr;
    EXPECT_EQ(attr.type, IBV_QPT_UD);
    EXPECT_EQ(attr.maxRecvWorkRequest, 1u);
    EXPECT_EQ(attr.maxSendWorkRequest, (uint32_t)NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
    EXPECT_EQ(attr.cq, res_->devs[expectDev[q]].portRecoveryCq);
    EXPECT_EQ(attr.pd, send_->devs[expectDev[q]].base.pd);
    EXPECT_EQ(attr.qpContext, (void*)&send_->base.stats);
    EXPECT_EQ(attr.qpSharingGroupIdx, -1);
    EXPECT_EQ(res_->portRecoveryQps[q].devIndex, expectDev[q]);
    EXPECT_EQ(info[q].qpn, res_->portRecoveryQps[q].qp->qp_num);
    EXPECT_EQ(info[q].devIndex, expectDev[q]);
    EXPECT_EQ(scene_.modifies[q].qp, res_->portRecoveryQps[q].qp);
    EXPECT_EQ(scene_.modifies[q].attr.qp_state, IBV_QPS_INIT);
    EXPECT_EQ(scene_.modifies[q].attr.port_num, scene_.Dev(expectDev[q])->portNum);
  }
}

TEST_F(P2pResiliencyRecoveryMicrotest, SenderQpsCreate_CreateOrInitFails_ReturnsTheError) {
  send_ = scene_.NewComm<ncclIbSendComm>(true, 1, 1);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 1, 1), ncclSuccess);
  ncclIbQpInfo info[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  {
    ScopedHook create(g_ibCastQpCreate, [](ncclIbQp*, ncclIbQpCreateAttr*) { return ncclSystemError; });
    EXPECT_EQ(IbCastPortRecoverySenderQpsCreate(res_, info, 1), ncclSystemError);
  }
  ScopedHook modify(g_ibvModifyQp, [](ibv_qp*, ibv_qp_attr*, int) { return ncclSystemError; });
  EXPECT_EQ(IbCastPortRecoverySenderQpsCreate(res_, info, 1), ncclSystemError);
}

TEST_F(P2pResiliencyRecoveryMicrotest, SenderQpsToRts_RoceAndInfinibandPeersGetMatchingAddressHandles) {
  Make(true, 2, 2);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, true);
  remote.devs[1].link_layer = IBV_LINK_LAYER_INFINIBAND;
  scene_.ahAttrs.clear();
  ASSERT_EQ(IbCastPortRecoverySenderQpsToRts(res_, &remote, 2), ncclSuccess);
  ASSERT_EQ(scene_.modifies.size(), 4u);
  EXPECT_EQ(scene_.modifies[0].attr.qp_state, IBV_QPS_RTR);
  EXPECT_EQ(scene_.modifies[1].attr.qp_state, IBV_QPS_RTS);
  ASSERT_EQ(scene_.ahAttrs.size(), 2u);
  const ibv_ah_attr& roce = scene_.ahAttrs[0];
  EXPECT_EQ(roce.is_global, 1);
  EXPECT_EQ(roce.grh.dgid.global.interface_id, 0xC0u);
  EXPECT_EQ(roce.grh.sgid_index, 10);
  EXPECT_EQ(roce.grh.hop_limit, 255);
  EXPECT_EQ(roce.grh.traffic_class, 106);
  EXPECT_EQ(roce.sl, 3);
  EXPECT_EQ(roce.port_num, scene_.Dev(0)->portNum);
  EXPECT_EQ(roce.dlid, 0);
  const ibv_ah_attr& ib = scene_.ahAttrs[1];
  EXPECT_EQ(ib.is_global, 0);
  EXPECT_EQ(ib.dlid, 0x301);
  EXPECT_EQ(ib.port_num, scene_.Dev(1)->portNum);
  EXPECT_EQ(res_->portRecoveryAhAttr[1].dlid, 0x301);
  EXPECT_EQ(res_->portRecoveryRemoteQpn[0], 0x600u);
  EXPECT_EQ(res_->portRecoveryRemoteQpn[1], 0x601u);
  EXPECT_EQ(res_->portRecoveryQps[1].remDevIdx, 1);
}

TEST_F(P2pResiliencyRecoveryMicrotest, SenderQpsToRts_VerbsFailure_ReturnsTheError) {
  Make(true, 1, 1);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(1, true);
  for (int failAt = 0; failAt < 2; failAt++) {
    int n = 0;
    ScopedHook modify(g_ibvModifyQp,
                      [&](ibv_qp*, ibv_qp_attr*, int) { return n++ == failAt ? ncclSystemError : ncclSuccess; });
    EXPECT_EQ(IbCastPortRecoverySenderQpsToRts(res_, &remote, 1), ncclSystemError);
  }
  scene_.createAh = [](ibv_pd*, ibv_ah_attr*) -> ibv_ah* { return nullptr; };
  EXPECT_EQ(IbCastPortRecoverySenderQpsToRts(res_, &remote, 1), ncclSystemError);
}

TEST_F(P2pResiliencyRecoveryMicrotest, ReceiverQpsCreateToRts_CreatesConnectsAndAddressesUdQps) {
  scene_.params["IB_PKEY"] = 6;
  recv_ = scene_.NewComm<ncclIbRecvComm>(false, 2, 2);
  ASSERT_EQ(IbCastResiliencyInit(&recv_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 2, 2), ncclSuccess);
  for (int d = 0; d < 2; d++) ASSERT_EQ(IbCastResiliencyDevInit(res_, d, scene_.Dev(d)), ncclSuccess);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, true);
  ncclIbQpInfo info[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  ASSERT_EQ(IbCastPortRecoveryReceiverQpsCreateToRts(res_, &remote, info, 2), ncclSuccess);
  ASSERT_EQ(scene_.qpCreates.size(), 2u);
  for (int q = 0; q < 2; q++) {
    const ncclIbQpCreateAttr& attr = scene_.qpCreates[q].attr;
    EXPECT_EQ(attr.type, IBV_QPT_UD);
    EXPECT_EQ(attr.maxRecvWorkRequest, (uint32_t)NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
    EXPECT_EQ(attr.maxSendWorkRequest, 1u);
    EXPECT_EQ(attr.cq, res_->devs[q].portRecoveryCq);
    EXPECT_EQ(attr.pd, recv_->devs[q].base.pd);
    EXPECT_EQ(attr.qpContext, (void*)&recv_->base.stats);
    EXPECT_EQ(info[q].qpn, res_->portRecoveryQps[q].qp->qp_num);
    EXPECT_EQ(info[q].devIndex, q);
    EXPECT_EQ(res_->portRecoveryQps[q].devIndex, q);
    EXPECT_EQ(res_->portRecoveryRemoteQpn[q], 0x600u + q);
    EXPECT_EQ(res_->portRecoveryAhAttr[q].grh.sgid_index, 10 + q);
    EXPECT_EQ(res_->portRecoveryAhAttr[q].grh.dgid.global.interface_id, 0xC0u + q);
  }
  ASSERT_EQ(scene_.modifies.size(), 6u);
  EXPECT_EQ(scene_.modifies[0].attr.qp_state, IBV_QPS_INIT);
  EXPECT_EQ(scene_.modifies[0].attr.pkey_index, 6);
  EXPECT_EQ(scene_.modifies[1].attr.qp_state, IBV_QPS_RTR);
  EXPECT_EQ(scene_.modifies[2].attr.qp_state, IBV_QPS_RTS);
  remote.devs[0].link_layer = IBV_LINK_LAYER_INFINIBAND;
  ASSERT_EQ(IbCastPortRecoveryReceiverQpsCreateToRts(res_, &remote, info, 1), ncclSuccess);
  EXPECT_EQ(res_->portRecoveryAhAttr[0].is_global, 0);
  EXPECT_EQ(res_->portRecoveryAhAttr[0].dlid, 0x300);
}

TEST_F(P2pResiliencyRecoveryMicrotest, ReceiverQpsCreateToRts_StepFails_ReturnsTheError) {
  Make(false, 1, 1);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(1, true);
  ncclIbQpInfo info[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  {
    ScopedHook create(g_ibCastQpCreate, [](ncclIbQp*, ncclIbQpCreateAttr*) { return ncclSystemError; });
    EXPECT_EQ(IbCastPortRecoveryReceiverQpsCreateToRts(res_, &remote, info, 1), ncclSystemError);
  }
  for (int failAt = 0; failAt < 3; failAt++) {
    int n = 0;
    ScopedHook modify(g_ibvModifyQp,
                      [&](ibv_qp*, ibv_qp_attr*, int) { return n++ == failAt ? ncclSystemError : ncclSuccess; });
    EXPECT_EQ(IbCastPortRecoveryReceiverQpsCreateToRts(res_, &remote, info, 1), ncclSystemError) << failAt;
  }
  scene_.createAh = [](ibv_pd*, ibv_ah_attr*) -> ibv_ah* { return nullptr; };
  EXPECT_EQ(IbCastPortRecoveryReceiverQpsCreateToRts(res_, &remote, info, 1), ncclSystemError);
}

TEST_F(P2pResiliencyRecoveryMicrotest, QpsDestroy_ReleasesAddressHandlesAndQps) {
  Make(true, 2, 2);
  std::vector<ibv_ah*> ahs = {res_->portRecoveryAh[0], res_->portRecoveryAh[1]};
  std::vector<ibv_qp*> qps = {res_->portRecoveryQps[0].qp, res_->portRecoveryQps[1].qp};
  ASSERT_EQ(IbCastPortRecoveryQpsDestroy(res_, 2), ncclSuccess);
  EXPECT_EQ(scene_.destroyedAhs, ahs);
  EXPECT_EQ(scene_.destroyedQps, qps);
  EXPECT_EQ(res_->portRecoveryAh[0], nullptr);
  EXPECT_EQ(res_->portRecoveryAh[1], nullptr);
  res_->portRecoveryQps[1].qp = nullptr;
  scene_.destroyedQps.clear();
  ASSERT_EQ(IbCastPortRecoveryQpsDestroy(res_, 2), ncclSuccess);
  EXPECT_EQ(scene_.destroyedQps, (std::vector<ibv_qp*>{qps[0]}));
}

TEST_F(P2pResiliencyRecoveryMicrotest, QpsDestroy_VerbsFailure_ReturnsTheError) {
  Make(true, 1, 1);
  {
    ScopedHook destroy(g_ibvDestroyQp, [](ibv_qp*) { return ncclSystemError; });
    EXPECT_EQ(IbCastPortRecoveryQpsDestroy(res_, 1), ncclSystemError);
  }
  res_->portRecoveryAh[0] = scene_.NewAh();
  scene_.destroyAh = [](ibv_ah*) { return EBUSY; };
  EXPECT_EQ(IbCastPortRecoveryQpsDestroy(res_, 1), ncclSystemError);
}

// Pins AICOMRCCL-3100; a fix flips this pin.
TEST_F(P2pResiliencyRecoveryMicrotest, SenderQpsToRts_KnownBug_MoreRecoveryQpsThanDevicesLeakAnAh) {
  // One local device against two remote ones: both recovery QPs live on device 0.
  send_ = scene_.NewComm<ncclIbSendComm>(true, 1, 1);
  ASSERT_EQ(IbCastResiliencyInit(&send_->base, &res_), ncclSuccess);
  res_->recoveryEnabled = true;
  ASSERT_EQ(IbCastResiliencyDeviceNumSet(res_, 1, 2), ncclSuccess);
  ASSERT_EQ(res_->nPortRecoveryQps, 2);
  ASSERT_EQ(IbCastResiliencyDevInit(res_, 0, scene_.Dev(0)), ncclSuccess);
  ncclIbQpInfo info[NCCL_IB_MAX_DEVS_PER_NIC] = {};
  ASSERT_EQ(IbCastPortRecoverySenderQpsCreate(res_, info, 2), ncclSuccess);
  ncclIbConnectionMetadata remote = P2pResiliencyRemoteInfo(2, true);
  ASSERT_EQ(IbCastPortRecoverySenderQpsToRts(res_, &remote, 2), ncclSuccess);
  ASSERT_EQ(IbCastPortRecoveryQpsDestroy(res_, 2), ncclSuccess);
  EXPECT_EQ(scene_.ahAttrs.size(), 2u);       // two AHs created...
  EXPECT_EQ(scene_.destroyedAhs.size(), 1u);  // ...one destroyed: the first is overwritten in slot 0
}

TEST_F(P2pResiliencyRecoveryMicrotest, PortQpsReconfigure_NoDeviceBase_ReturnsInternalError) {
  Make(true, 1, 1);
  ScopedHook devBase(g_ibCastGetNetCommDevBase, P2pResiliencyNoDevBase);
  bool ok = true;
  EXPECT_EQ(IbCastPortRecoveryQpsReconfigure(res_, 0, &ok), ncclInternalError);
  EXPECT_FALSE(ok);
}

TEST_F(P2pResiliencyRecoveryMicrotest, PortQpsReconfigure_NoGlobalAh_HasNothingToDo) {
  Make(true, 1, 1);
  bool ok = false;
  res_->portRecoveryAhAttr[0].is_global = 0;
  ASSERT_EQ(IbCastPortRecoveryQpsReconfigure(res_, 0, &ok), ncclSuccess);
  EXPECT_TRUE(ok);
  ibv_ah* ah = res_->portRecoveryAh[0];
  res_->portRecoveryAh[0] = nullptr;
  res_->portRecoveryAhAttr[0].is_global = 1;
  ok = false;
  ASSERT_EQ(IbCastPortRecoveryQpsReconfigure(res_, 0, &ok), ncclSuccess);
  EXPECT_TRUE(ok);
  res_->portRecoveryAh[0] = ah;
  EXPECT_EQ(scene_.ahAttrs.size(), 1u);
  EXPECT_TRUE(scene_.destroyedAhs.empty());
}

TEST_F(P2pResiliencyRecoveryMicrotest, PortQpsReconfigure_GlobalAh_IsRecreatedWithTheCurrentGidIndex) {
  Make(true, 2, 2);
  ibv_ah* stale = res_->portRecoveryAh[1];
  send_->devs[1].base.gidInfo.localGidIndex = 33;
  bool ok = false;
  ASSERT_EQ(IbCastPortRecoveryQpsReconfigure(res_, 1, &ok), ncclSuccess);
  EXPECT_TRUE(ok);
  ASSERT_EQ(scene_.ahAttrs.size(), 3u);
  EXPECT_EQ(scene_.ahAttrs[2].grh.sgid_index, 33);
  EXPECT_EQ(scene_.ahAttrs[2].grh.dgid.global.interface_id, 0xC1u);
  EXPECT_EQ(res_->portRecoveryAhAttr[1].grh.sgid_index, 33);
  EXPECT_EQ(scene_.destroyedAhs, (std::vector<ibv_ah*>{stale}));
  EXPECT_NE(res_->portRecoveryAh[1], stale);
}

TEST_F(P2pResiliencyRecoveryMicrotest, PortQpsReconfigure_CreateFails_KeepsTheOldAhAndReportsFailure) {
  Make(true, 1, 1);
  ibv_ah* stale = res_->portRecoveryAh[0];
  scene_.createAh = [](ibv_pd*, ibv_ah_attr*) -> ibv_ah* { return nullptr; };
  bool ok = true;
  ASSERT_EQ(IbCastPortRecoveryQpsReconfigure(res_, 0, &ok), ncclSuccess);
  EXPECT_FALSE(ok);
  EXPECT_EQ(res_->portRecoveryAh[0], stale);
  EXPECT_TRUE(scene_.destroyedAhs.empty());
}

TEST_F(P2pResiliencyRecoveryMicrotest, PortQpsReconfigure_StaleAhDestroyFails_StillUsesTheNewAh) {
  Make(true, 1, 1);
  ibv_ah* stale = res_->portRecoveryAh[0];
  scene_.destroyAh = [](ibv_ah*) { return EBUSY; };
  bool ok = false;
  ASSERT_EQ(IbCastPortRecoveryQpsReconfigure(res_, 0, &ok), ncclSuccess);
  EXPECT_TRUE(ok);
  EXPECT_NE(res_->portRecoveryAh[0], stale);
  EXPECT_NE(res_->portRecoveryAh[0], nullptr);
}

TEST_F(P2pResiliencyRecoveryMicrotest, ContextInit_Sender_ErrorsTheDevicesQpsAndPostsOneAckBuffer) {
  Make(true, 2, 4);
  ASSERT_EQ(IbCastPortRecoveryContextInit(res_, 1, &ctx_), ncclSuccess);
  EXPECT_EQ(ctx_->resCtx, res_);
  EXPECT_EQ(ctx_->devIndex, 1);
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateInit);
  EXPECT_EQ(ctx_->recoveryCq, res_->devs[1].portRecoveryCq);
  EXPECT_EQ(ctx_->timeInit, g_p2pResiliencyNowNs.load());
  ASSERT_EQ(scene_.transitions.size(), 2u);
  EXPECT_EQ(scene_.transitions[0].step, "ERROR");
  EXPECT_EQ(scene_.transitions[0].qp, &send_->base.qps[1]);
  EXPECT_EQ(scene_.transitions[1].qp, &send_->base.qps[3]);
  ASSERT_EQ(scene_.recvs.size(), 1u);
  EXPECT_EQ(scene_.recvs[0].qp, res_->portRecoveryQps[1].qp);
  EXPECT_EQ(scene_.recvs[0].wr.wr_id, (uint64_t)NCCL_IB_PORT_RECOVERY_WR_ID);
  EXPECT_EQ(scene_.recvs[0].wr.num_sge, 1);
  EXPECT_EQ(scene_.recvs[0].sge.addr, (uint64_t)res_->devs[1].portRecoveryGrhBuf);
  EXPECT_EQ(scene_.recvs[0].sge.length, sizeof(res_->devs[1].portRecoveryGrhBuf));
  EXPECT_EQ(scene_.recvs[0].sge.lkey, res_->devs[1].portRecoveryGrhMr->lkey);
}

TEST_F(P2pResiliencyRecoveryMicrotest, ContextInit_Receiver_PostsAFullBatchOfBuffers) {
  Make(false, 2, 2);
  ASSERT_EQ(IbCastPortRecoveryContextInit(res_, 0, &ctx_), ncclSuccess);
  EXPECT_EQ(scene_.recvs.size(), (size_t)NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX);
  EXPECT_EQ(ctx_->recv.windowBase, 0u);
  EXPECT_EQ(ctx_->recv.receivedBits, 0u);
}

TEST_F(P2pResiliencyRecoveryMicrotest, ContextInit_Failure_ReturnsTheErrorAndNoContext) {
  for (bool isSend : {true, false}) {
    Make(isSend, 1, 1);
    {
      ScopedHook qpError(g_ibCastQpError, [](ncclIbQp*) { return ncclSystemError; });
      ctx_ = (ncclIbPortRecoveryContext*)0x1;
      EXPECT_EQ(IbCastPortRecoveryContextInit(res_, 0, &ctx_), ncclSystemError);
      EXPECT_EQ(ctx_, nullptr);
    }
    int posts = 0;
    scene_.postRecv = [&](ibv_qp*, ibv_recv_wr*) { return ++posts == 2 ? EIO : 0; };
    EXPECT_EQ(IbCastPortRecoveryContextInit(res_, 0, &ctx_), isSend ? ncclSuccess : ncclSystemError);
    if (ctx_) IbCastPortRecoveryContextDestroy(ctx_);
    ctx_ = nullptr;
    scene_.postRecv = [](ibv_qp*, ibv_recv_wr*) { return EIO; };
    EXPECT_EQ(IbCastPortRecoveryContextInit(res_, 0, &ctx_), ncclSystemError);
    EXPECT_EQ(ctx_, nullptr);
    Reset();
  }
  EXPECT_EQ(IbCastPortRecoveryContextInit(nullptr, 0, nullptr), ncclInternalError);
}

TEST_F(P2pResiliencyRecoveryMicrotest, ContextInit_AllocationFails_ReturnsInternalError) {
  Make(true, 1, 1);
  g_recoveryMallocFails = true;
  ncclResult_t ret = IbCastPortRecoveryContextInit(res_, 0, &ctx_);
  g_recoveryMallocFails = false;
  EXPECT_EQ(ret, ncclInternalError);
  EXPECT_EQ(ctx_, nullptr);
  EXPECT_TRUE(scene_.transitions.empty());
}

TEST_F(P2pResiliencyRecoveryMicrotest, HandleFailure_QueuesTheDeviceForTheThread) {
  Make(true, 1, 1);
  res_->devs[0].state.store(ncclIbResiliencyDevStateRecoveryInProgress);
  ASSERT_EQ(IbCastPortRecoveryHandleFailure(res_, 0), ncclSuccess);
  std::list<ncclIbPortRecoveryContext*> queued;
  {
    std::lock_guard<std::mutex> lock(IbCastPortRecoveryMutex);
    queued.swap(recoveryInbox);
  }
  ASSERT_EQ(queued.size(), 1u);
  ctx_ = queued.front();
  EXPECT_EQ(ctx_->resCtx, res_);
  EXPECT_EQ(ctx_->devIndex, 0);
  ScopedHook qpError(g_ibCastQpError, [](ncclIbQp*) { return ncclSystemError; });
  EXPECT_EQ(IbCastPortRecoveryHandleFailure(res_, 0), ncclSystemError);
  std::lock_guard<std::mutex> lock(IbCastPortRecoveryMutex);
  EXPECT_TRUE(recoveryInbox.empty());
}

TEST_F(P2pResiliencyRecoveryMicrotest, Progress_BeforeStartDelay_Waits) {
  Make(true, 2, 2);
  res_->devs[1].state.store(ncclIbResiliencyDevStateRecoveryInProgress);
  ASSERT_EQ(IbCastPortRecoveryContextInit(res_, 1, &ctx_), ncclSuccess);
  ClearRecords();
  P2pResiliencyAdvanceMs(kStartDelayMs - 1);
  EXPECT_FALSE(Step());
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateInit);
  EXPECT_TRUE(scene_.transitions.empty());
  P2pResiliencyAdvanceMs(1);
  EXPECT_FALSE(Step());
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateAliveMessages);
  // The start reconfigures the device's probing QP before talking to the peer.
  ASSERT_EQ(scene_.transitions.size(), 4u);
  EXPECT_EQ(scene_.transitions[0].qp, &res_->probingQps[1]);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Progress_ReconfigureFails_FailsTheRecovery) {
  Make(false, 2, 2);
  ScopedHook rtr(g_ibCastQpRtr, [](ncclIbQp*) { return ncclSystemError; });
  Start(0);
  EXPECT_TRUE(Step());
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateFailed);
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed);
  EXPECT_EQ(State(1), ncclIbResiliencyDevStateOk);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Sender_PostsOneChainedSignaledAliveBatchAtOnce) {
  Make(true, 2, 2);
  Start(1);
  EXPECT_FALSE(Step());
  ASSERT_EQ(scene_.sends.size(), (size_t)kSequenceSize);
  for (uint32_t i = 0; i < kSequenceSize; i++) {
    const ibv_send_wr& wr = scene_.sends[i].wr;
    EXPECT_EQ(scene_.sends[i].qp, res_->portRecoveryQps[1].qp);
    EXPECT_EQ(wr.opcode, IBV_WR_SEND_WITH_IMM);
    EXPECT_EQ(wr.imm_data, i);
    EXPECT_EQ(wr.wr_id, i);
    EXPECT_EQ(wr.num_sge, 0);
    EXPECT_EQ(wr.send_flags, i == kSequenceSize - 1 ? IBV_SEND_SIGNALED : 0);
    EXPECT_EQ(wr.wr.ud.ah, res_->portRecoveryAh[1]);
    EXPECT_EQ(wr.wr.ud.remote_qpn, 0x601u);
    EXPECT_EQ(wr.wr.ud.remote_qkey, (uint32_t)NCCL_IB_RECOVERY_QKEY);
  }
  EXPECT_EQ(ctx_->aliveMsgNextId, (uint32_t)kSequenceSize);
  // Waits for the batch's completion rather than posting another.
  P2pResiliencyAdvanceMs(kBatchIntervalMs);
  EXPECT_FALSE(Step());
  EXPECT_EQ(scene_.sends.size(), (size_t)kSequenceSize);
  Cqe(IBV_WC_SEND);
  EXPECT_FALSE(Step());
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateAck);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Sender_SequenceLongerThanTheQueue_ReturnsInternalError) {
  scene_.params["IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE"] =
      NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX + 1;
  Make(true, 1, 1);
  Start(0);
  Step(ncclInternalError);
  EXPECT_TRUE(scene_.sends.empty());
}

TEST_F(P2pResiliencyRecoveryMicrotest, Sender_AliveBatchPostFails_FailsTheRecovery) {
  Make(true, 1, 1);
  Start(0);
  scene_.postSend = [](ibv_qp*, ibv_send_wr*) { return ENOMEM; };
  EXPECT_TRUE(Step());
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Sender_BadCompletionWhileSendingAlive_FailsTheRecovery) {
  struct Case {
    ibv_wc_opcode opcode;
    ibv_wc_status status;
  } cases[] = {{IBV_WC_RECV, IBV_WC_SUCCESS}, {IBV_WC_SEND, IBV_WC_RETRY_EXC_ERR}};
  for (const Case& c : cases) {
    Make(true, 1, 1);
    Start(0);
    ASSERT_FALSE(Step());
    Cqe(c.opcode, 0, c.status);
    EXPECT_TRUE(Step());
    EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateFailed);
    Reset();
  }
}

TEST_F(P2pResiliencyRecoveryMicrotest, Sender_PollFails_FailsTheRecovery) {
  Make(true, 1, 1);
  Start(0);
  ASSERT_FALSE(Step());
  scene_.pollCq = [](ibv_cq*, int, ibv_wc*) { return -1; };
  EXPECT_TRUE(Step());
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Sender_NoAckInTime_ResendsAliveBatchUntilAttemptsRunOut) {
  Make(true, 1, 1);
  SenderToAck(0);
  for (int attempt = 1; attempt < kAttemptsMax; attempt++) {
    P2pResiliencyAdvanceMs(kAckTimeoutMs);
    EXPECT_FALSE(Step());  // exactly at the timeout: still waiting
    EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateAck);
    P2pResiliencyAdvanceMs(1);
    EXPECT_FALSE(Step());
    EXPECT_EQ(ctx_->nFailedAttempts, attempt);
    EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateAliveMessages);
    EXPECT_FALSE(ctx_->send.aliveMsgPosted);
    EXPECT_FALSE(ctx_->send.aliveMsgCompleted);
    // The next batch goes out at once and continues the alive ids.
    EXPECT_FALSE(Step());
    EXPECT_EQ(scene_.sends.back().wr.imm_data, (uint32_t)(kSequenceSize * (attempt + 1) - 1));
    Cqe(IBV_WC_SEND);
    EXPECT_FALSE(Step());
    ASSERT_EQ(ctx_->state, ncclIbPortRecoveryStateAck);
  }
  P2pResiliencyAdvanceMs(kAckTimeoutMs + 1);
  EXPECT_TRUE(Step());
  EXPECT_EQ(ctx_->nFailedAttempts, kAttemptsMax);
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Sender_WaitsBatchIntervalBetweenAliveBatches) {
  Make(true, 1, 1);
  Start(0);
  ctx_->state = ncclIbPortRecoveryStateAliveMessages;
  ctx_->timeLastMsg = g_p2pResiliencyNowNs.load();
  P2pResiliencyAdvanceMs(kBatchIntervalMs - 1);
  EXPECT_FALSE(Step());
  EXPECT_TRUE(scene_.sends.empty());
  P2pResiliencyAdvanceMs(1);
  EXPECT_FALSE(Step());
  EXPECT_EQ(scene_.sends.size(), (size_t)kSequenceSize);
  EXPECT_EQ(ctx_->timeLastMsg, g_p2pResiliencyNowNs.load());
}

TEST_F(P2pResiliencyRecoveryMicrotest, Sender_AckReceived_RestoresQpsAcksAndRecovers) {
  Make(true, 2, 4);
  SenderToAck(1);
  ClearRecords();
  Cqe(IBV_WC_RECV, NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
  EXPECT_FALSE(Step());
  EXPECT_TRUE(ctx_->ackReceived);
  // Device 1's data QPs (1 and 3) go through RESET, INIT, RTR, RTS.
  ASSERT_EQ(scene_.transitions.size(), 8u);
  EXPECT_EQ(scene_.transitions[0].qp, &send_->base.qps[1]);
  EXPECT_EQ(scene_.transitions[0].step, "RESET");
  EXPECT_EQ(scene_.transitions[4].qp, &send_->base.qps[3]);
  EXPECT_EQ(scene_.transitions[7].step, "RTS");
  EXPECT_TRUE(scene_.postedRecvWrQps.empty());
  ASSERT_EQ(scene_.sends.size(), 1u);
  const ibv_send_wr& ack = scene_.sends[0].wr;
  EXPECT_EQ(ack.opcode, IBV_WR_SEND_WITH_IMM);
  EXPECT_EQ(ack.imm_data, (uint32_t)NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
  EXPECT_EQ(ack.send_flags, IBV_SEND_SIGNALED);
  EXPECT_EQ(ack.num_sge, 0);
  EXPECT_EQ(ack.wr_id, (uint64_t)kSequenceSize);
  EXPECT_EQ(ack.wr.ud.ah, res_->portRecoveryAh[1]);
  EXPECT_EQ(ack.wr.ud.remote_qpn, 0x601u);
  EXPECT_EQ(ack.wr.ud.remote_qkey, (uint32_t)NCCL_IB_RECOVERY_QKEY);
  EXPECT_EQ(scene_.sends[0].qp, res_->portRecoveryQps[1].qp);
  EXPECT_EQ(State(1), ncclIbResiliencyDevStateRecoveryInProgress);
  Cqe(IBV_WC_SEND);
  EXPECT_TRUE(Step());
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateSuccess);
  EXPECT_EQ(State(1), ncclIbResiliencyDevStateRecovered);
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateOk);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Sender_RestoreOrAckPostFails_FailsTheRecovery) {
  // The QP restore, the ACK post, then the poll for the ACK's completion.
  for (int failing = 0; failing < 3; failing++) {
    Make(true, 1, 1);
    SenderToAck(0);
    ScopedHook rts(g_ibCastQpRts, [&](ncclIbQp*) { return failing == 0 ? ncclSystemError : ncclSuccess; });
    if (failing == 1) scene_.postSend = [](ibv_qp*, ibv_send_wr*) { return EIO; };
    if (failing == 2) scene_.pollCq = [n = 0](ibv_cq*, int, ibv_wc* wc) mutable {
      wc[0] = P2pResiliencyWc(0, IBV_WC_SUCCESS, IBV_WC_RECV, NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
      return n++ == 0 ? 1 : -1;
    };
    if (failing != 2) Cqe(IBV_WC_RECV, NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
    EXPECT_TRUE(Step()) << failing;
    EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed) << failing;
    Reset();
  }
}

TEST_F(P2pResiliencyRecoveryMicrotest, Receiver_FullAliveWindow_DrainsRestoresAndAcks) {
  Make(false, 2, 4);
  recv_->flushEnabled = 1;
  recv_->devs[1].gpuFlush.qp.qp = scene_.NewQp();
  recv_->devs[0].gpuFlush.qp.qp = scene_.NewQp();
  Start(1);
  ASSERT_FALSE(Step());
  for (uint32_t id = 0; id < kSequenceSize - 1; id++) {
    Cqe(IBV_WC_RECV, id);
    ASSERT_FALSE(Step());
    EXPECT_EQ(ctx_->timeLastMsg, g_p2pResiliencyNowNs.load());
  }
  EXPECT_FALSE(ctx_->ackPosted);
  EXPECT_EQ(scene_.recvs.size(), (size_t)kSequenceSize - 1);  // one buffer back per message
  // Two duplicates are already queued behind the last message; the drain consumes them.
  Cqe(IBV_WC_RECV, kSequenceSize - 1);
  Cqe(IBV_WC_RECV, 1);
  Cqe(IBV_WC_RECV, 2);
  ClearRecords();
  ASSERT_FALSE(Step());
  EXPECT_EQ(scene_.PendingCqes(ctx_->recoveryCq), 0u);
  EXPECT_EQ(scene_.recvs.size(), 3u);
  // Data QPs 1 and 3, then device 1's flush QP, are restored; data QPs get their receives back.
  ASSERT_EQ(scene_.transitions.size(), 12u);
  EXPECT_EQ(scene_.transitions[0].qp, &recv_->base.qps[1]);
  EXPECT_EQ(scene_.transitions[4].qp, &recv_->base.qps[3]);
  EXPECT_EQ(scene_.transitions[8].qp, &recv_->devs[1].gpuFlush.qp);
  EXPECT_EQ(scene_.transitions[10].step, "RTR");
  EXPECT_EQ(scene_.transitions[10].rtrAttr.remoteGid.global.interface_id,
            recv_->devs[1].base.gidInfo.localGid.global.interface_id);
  EXPECT_EQ(scene_.postedRecvWrQps, (std::vector<ncclIbQp*>{&recv_->base.qps[1], &recv_->base.qps[3]}));
  ASSERT_EQ(scene_.sends.size(), 1u);
  EXPECT_EQ(scene_.sends[0].wr.imm_data, (uint32_t)NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
  EXPECT_EQ(scene_.sends[0].wr.wr.ud.remote_qpn, 0x601u);
  EXPECT_TRUE(ctx_->ackPosted);
  // A late alive message while the ACK is in flight just gets its buffer back.
  Cqe(IBV_WC_RECV, 3);
  EXPECT_FALSE(Step());
  EXPECT_EQ(scene_.recvs.size(), 4u);
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateAliveMessages);
  Cqe(IBV_WC_SEND);
  EXPECT_FALSE(Step());
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateAck);
  EXPECT_TRUE(ctx_->ackCompleted);
  // The sender's final ACK completes the recovery.
  Cqe(IBV_WC_RECV, NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
  EXPECT_TRUE(Step());
  EXPECT_EQ(State(1), ncclIbResiliencyDevStateRecovered);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Receiver_AliveWindow_TracksTheNewestSequence) {
  Make(false, 1, 1);
  Start(0);
  auto deliver = [&](uint32_t id) {
    Cqe(IBV_WC_RECV, id);
    EXPECT_FALSE(Step());
  };
  deliver(1);
  deliver(2);
  EXPECT_EQ(ctx_->recv.windowBase, 0u);
  EXPECT_EQ(ctx_->recv.receivedBits, 0x6u);
  // The first id past the window starts the next sequence's window.
  deliver(kSequenceSize);
  EXPECT_EQ(ctx_->recv.windowBase, (uint32_t)kSequenceSize);
  EXPECT_EQ(ctx_->recv.receivedBits, 0x1u);
  deliver(kSequenceSize + 2);
  EXPECT_EQ(ctx_->recv.receivedBits, 0x5u);
  deliver(1);  // stale: ignored
  EXPECT_EQ(ctx_->recv.windowBase, (uint32_t)kSequenceSize);
  EXPECT_EQ(ctx_->recv.receivedBits, 0x5u);
  // An id further ahead jumps to the window of its own sequence.
  deliver(3 * kSequenceSize + 1);
  EXPECT_EQ(ctx_->recv.windowBase, (uint32_t)(3 * kSequenceSize));
  EXPECT_EQ(ctx_->recv.receivedBits, 0x2u);
  deliver(3 * kSequenceSize);
  deliver(4 * kSequenceSize - 1);
  EXPECT_FALSE(ctx_->ackPosted);
  deliver(3 * kSequenceSize + 2);
  EXPECT_TRUE(ctx_->ackPosted);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Receiver_NoAliveMessages_FailsAfterTheAttempts) {
  Make(false, 1, 1);
  Start(0);
  ctx_->state = ncclIbPortRecoveryStateAliveMessages;
  ctx_->timeLastMsg = g_p2pResiliencyNowNs.load();
  for (int attempt = 1; attempt < kAttemptsMax; attempt++) {
    P2pResiliencyAdvanceMs(kAliveTimeoutMs);
    EXPECT_FALSE(Step());
    EXPECT_EQ(ctx_->nFailedAttempts, attempt - 1);
    P2pResiliencyAdvanceMs(1);
    EXPECT_FALSE(Step());
    EXPECT_EQ(ctx_->nFailedAttempts, attempt);
  }
  P2pResiliencyAdvanceMs(kAliveTimeoutMs + 1);
  EXPECT_TRUE(Step());
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed);
}

// Pins AICOMRCCL-3099; a fix flips this pin.
TEST_F(P2pResiliencyRecoveryMicrotest, Receiver_KnownBug_FirstAlivePollCountsAsAFailedAttempt) {
  Make(false, 1, 1);
  Start(0);
  // The receiver never starts its alive timer, so the first poll without a message
  // already exceeds the timeout.
  EXPECT_FALSE(Step());
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateAliveMessages);
  EXPECT_EQ(ctx_->nFailedAttempts, 1);
}

// Pins AICOMRCCL-3098; a fix flips this pin.
TEST_F(P2pResiliencyRecoveryMicrotest, Receiver_KnownBug_AckCompletionTimeoutReturnsInternalError) {
  Make(false, 1, 1);
  ReceiverToAckPosted(0);
  P2pResiliencyAdvanceMs(kAckTimeoutMs + 1);
  // Meant to restart the alive phase; the AliveMessages state rejects GoToPrevState instead,
  // which ends the recovery thread.
  std::string log = CaptureLog([&] { Step(ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "Unexpected GoToPrevState")) << log;
  EXPECT_FALSE(ctx_->ackPosted);
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateAliveMessages);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Receiver_AckCompletionTimeoutAtMaxAttempts_FailsTheRecovery) {
  Make(false, 1, 1);
  ReceiverToAckPosted(0);
  ctx_->nFailedAttempts = kAttemptsMax - 1;
  P2pResiliencyAdvanceMs(kAckTimeoutMs);
  EXPECT_FALSE(Step());
  P2pResiliencyAdvanceMs(1);
  EXPECT_TRUE(Step());
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Receiver_NoFinalAck_RestartsAliveThenFails) {
  Make(false, 1, 1);
  ReceiverToAckPosted(0);
  Cqe(IBV_WC_SEND);
  ASSERT_FALSE(Step());
  ASSERT_EQ(ctx_->state, ncclIbPortRecoveryStateAck);
  P2pResiliencyAdvanceMs(kAckTimeoutMs);
  EXPECT_FALSE(Step());
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateAck);
  ClearRecords();
  // The progress poll finds nothing; the drain on the way back finds one stray message.
  scene_.pollCq = [n = 0](ibv_cq*, int, ibv_wc* wc) mutable {
    wc[0] = P2pResiliencyWc(0, IBV_WC_SUCCESS, IBV_WC_RECV, 1);
    return ++n == 2 ? 1 : 0;
  };
  P2pResiliencyAdvanceMs(1);
  EXPECT_FALSE(Step());
  EXPECT_EQ(ctx_->state, ncclIbPortRecoveryStateAliveMessages);
  EXPECT_EQ(ctx_->nFailedAttempts, 1);
  EXPECT_FALSE(ctx_->ackPosted);
  EXPECT_FALSE(ctx_->ackCompleted);
  EXPECT_EQ(ctx_->recv.receivedBits, 0u);
  EXPECT_EQ(scene_.recvs.size(), 1u);
  ctx_->state = ncclIbPortRecoveryStateAck;
  ctx_->nFailedAttempts = kAttemptsMax - 1;
  P2pResiliencyAdvanceMs(kAckTimeoutMs + 1);
  EXPECT_TRUE(Step());
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Receiver_NoFinalAckDrainFails_FailsTheRecovery) {
  Make(false, 1, 1);
  ReceiverToAckPosted(0);
  Cqe(IBV_WC_SEND);
  ASSERT_FALSE(Step());
  scene_.pollCq = [n = 0](ibv_cq*, int, ibv_wc*) mutable { return ++n == 2 ? -1 : 0; };
  P2pResiliencyAdvanceMs(kAckTimeoutMs + 1);
  EXPECT_TRUE(Step());
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed);
}

TEST_F(P2pResiliencyRecoveryMicrotest, Receiver_RestoreStepFails_FailsTheRecovery) {
  for (int failing = 0; failing < 6; failing++) {
    Make(false, 1, 1);
    recv_->flushEnabled = 1;
    recv_->devs[0].gpuFlush.qp.qp = scene_.NewQp();
    Start(0);
    ASSERT_FALSE(Step());
    for (uint32_t id = 0; id < kSequenceSize - 1; id++) {
      Cqe(IBV_WC_RECV, id);
      ASSERT_FALSE(Step());
    }
    int rts = 0;
    ScopedHook rtsHook(g_ibCastQpRts, [&](ncclIbQp*) {
      ++rts;
      return (failing == 0 && rts == 1) || (failing == 1 && rts == 2) ? ncclSystemError : ncclSuccess;
    });
    ScopedHook postWrs(g_ibCastPostReceiveWorkRequestsOnQp,
                       [&](ncclIbRecvComm*, ncclIbQp*) { return failing == 2 ? ncclSystemError : ncclSuccess; });
    if (failing == 3) scene_.postSend = [](ibv_qp*, ibv_send_wr*) { return EIO; };
    int polls = 0;
    if (failing == 4) {
      scene_.pollCq = [&](ibv_cq*, int, ibv_wc* wc) {
        wc[0] = P2pResiliencyWc(0, IBV_WC_SUCCESS, IBV_WC_RECV, kSequenceSize - 1);
        return ++polls == 1 ? 1 : -1;  // the drain's poll fails
      };
    } else {
      Cqe(IBV_WC_RECV, kSequenceSize - 1);
      if (failing == 5) {
        Cqe(IBV_WC_RECV, 0);
        // The last message's buffer goes back fine; the drain's repost fails.
        scene_.postRecv = [posts = 0](ibv_qp*, ibv_recv_wr*) mutable { return ++posts == 2 ? EIO : 0; };
      }
    }
    EXPECT_TRUE(Step()) << failing;
    EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed) << failing;
    Reset();
  }
}


TEST_F(P2pResiliencyRecoveryMicrotest, Progress_PollFailsInAnyWaitingState_FailsTheRecovery) {
  // Receiver waiting for alive messages, receiver waiting for its ACK's completion,
  // receiver waiting for the final ACK, sender waiting for the ACK.
  for (int where = 0; where < 4; where++) {
    Make(where == 3, 1, 1);
    if (where == 0) Start(0);
    if (where == 1) ReceiverToAckPosted(0);
    if (where == 2) {
      ReceiverToAckPosted(0);
      Cqe(IBV_WC_SEND);
      ASSERT_FALSE(Step());
    }
    if (where == 3) SenderToAck(0);
    scene_.pollCq = [](ibv_cq*, int, ibv_wc*) { return -1; };
    EXPECT_TRUE(Step()) << where;
    EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed) << where;
    Reset();
  }
}

TEST_F(P2pResiliencyRecoveryMicrotest, Receiver_BufferRepostFails_FailsTheRecovery) {
  // Reposting after an alive message, after a late alive message, and on the way back after an ACK timeout.
  for (int where = 0; where < 3; where++) {
    Make(false, 1, 1);
    if (where == 0) {
      Start(0);
      Cqe(IBV_WC_RECV, 0);
    } else {
      ReceiverToAckPosted(0);
      if (where == 1) Cqe(IBV_WC_RECV, 1);
      if (where == 2) P2pResiliencyAdvanceMs(kAckTimeoutMs + 1);
    }
    scene_.postRecv = [](ibv_qp*, ibv_recv_wr*) { return EIO; };
    EXPECT_TRUE(Step()) << where;
    EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed) << where;
    Reset();
  }
}

TEST_F(P2pResiliencyRecoveryMicrotest, Progress_RestoreWithoutDeviceBase_ReturnsInternalError) {
  Make(true, 1, 1);
  SenderToAck(0);
  Cqe(IBV_WC_RECV, NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID);
  ScopedHook devBase(g_ibCastGetNetCommDevBase, P2pResiliencyNoDevBase);
  Step(ncclInternalError);
}

class P2pResiliencyRecoveryAinicFlowMicrotest : public P2pResiliencyRecoveryMicrotest {
 protected:
  void SetUp() override {
    P2pResiliencyRecoveryMicrotest::SetUp();
    IbCastAinicRoce = true;
  }

  // An ACK as received on device dev, carrying the peer's new QPNs after the GRH.
  void AckWithQpns(int dev, std::vector<RecoveryQpnEntry> entries) {
    uint8_t* payload = res_->devs[dev].portRecoveryGrhBuf + NCCL_IB_UD_GRH_SIZE;
    std::memcpy(payload, entries.data(), entries.size() * sizeof(RecoveryQpnEntry));
    scene_.PushCqe(ctx_->recoveryCq,
                   P2pResiliencyWc(0, IBV_WC_SUCCESS, IBV_WC_RECV, NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID,
                                   NCCL_IB_UD_GRH_SIZE + entries.size() * sizeof(RecoveryQpnEntry)));
  }
};

TEST_F(P2pResiliencyRecoveryAinicFlowMicrotest, Sender_RecreatesQpsAndConnectsThemToThePeersNewQpns) {
  Make(true, 2, 4);
  SenderToAck(1);
  ClearRecords();
  ibv_qp* old1 = send_->base.qps[1].qp;
  ibv_qp* old3 = send_->base.qps[3].qp;
  AckWithQpns(1, {{1, 0x901}, {3, 0x903}});
  EXPECT_FALSE(Step());
  EXPECT_EQ(scene_.destroyedQps, (std::vector<ibv_qp*>{old1, old3}));
  EXPECT_EQ(scene_.qpCreates.size(), 2u);
  EXPECT_EQ(send_->base.qps[1].rtrAttr.remoteQpNum, 0x901u);
  EXPECT_EQ(send_->base.qps[3].rtrAttr.remoteQpNum, 0x903u);
  // Both QPs are recreated (INIT) and then connected (RTR, RTS) before the ACK goes out.
  ASSERT_EQ(scene_.transitions.size(), 6u);
  EXPECT_EQ(scene_.transitions[0].step, "INIT");
  EXPECT_EQ(scene_.transitions[2].step, "RTR");
  ASSERT_EQ(scene_.sends.size(), 1u);
  ASSERT_EQ(scene_.sends[0].sges.size(), 1u);
  EXPECT_EQ(scene_.sends[0].sges[0].length, 2 * sizeof(RecoveryQpnEntry));
  const RecoveryQpnEntry* sent = (const RecoveryQpnEntry*)res_->portRecoveryQpnBuf;
  EXPECT_EQ(sent[0].qpIndex, 1u);
  EXPECT_EQ(sent[0].newQpn, send_->base.qps[1].qp->qp_num);
  EXPECT_EQ(sent[1].qpIndex, 3u);
  Cqe(IBV_WC_SEND);
  EXPECT_TRUE(Step());
  EXPECT_EQ(State(1), ncclIbResiliencyDevStateRecovered);
}

TEST_F(P2pResiliencyRecoveryAinicFlowMicrotest, Receiver_ConnectsRecreatedQpsOnlyAfterTheFinalAck) {
  Make(false, 1, 2);
  ReceiverToAckPosted(0);
  // Window complete: QPs recreated and announced, but not connected yet.
  EXPECT_EQ(scene_.qpCreates.size(), 2u);
  for (const auto& t : scene_.transitions) EXPECT_NE(t.step, "RTR");
  EXPECT_TRUE(scene_.postedRecvWrQps.empty());
  EXPECT_EQ(scene_.sends.back().sges[0].length, 2 * sizeof(RecoveryQpnEntry));
  Cqe(IBV_WC_SEND);
  ASSERT_FALSE(Step());
  scene_.transitions.clear();
  AckWithQpns(0, {{0, 0x700}, {1, 0x701}});
  EXPECT_TRUE(Step());
  EXPECT_EQ(recv_->base.qps[0].rtrAttr.remoteQpNum, 0x700u);
  EXPECT_EQ(recv_->base.qps[1].rtrAttr.remoteQpNum, 0x701u);
  ASSERT_EQ(scene_.transitions.size(), 4u);
  EXPECT_EQ(scene_.transitions[0].step, "RTR");
  EXPECT_EQ(scene_.postedRecvWrQps.size(), 2u);
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecovered);
}

TEST_F(P2pResiliencyRecoveryAinicFlowMicrotest, Sender_RecreateOrReconnectFails_FailsTheRecovery) {
  for (int failing = 0; failing < 2; failing++) {
    Make(true, 1, 1);
    SenderToAck(0);
    ScopedHook destroy(g_ibvDestroyQp, [&](ibv_qp*) { return failing == 0 ? ncclSystemError : ncclSuccess; });
    AckWithQpns(0, failing == 1 ? std::vector<RecoveryQpnEntry>{} : std::vector<RecoveryQpnEntry>{{0, 0x800}});
    EXPECT_TRUE(Step()) << failing;
    EXPECT_EQ(State(0), ncclIbResiliencyDevStateRecoveryFailed) << failing;
    Reset();
  }
}

class P2pResiliencyRecoveryThreadMicrotest : public P2pResiliencyRecoveryMicrotest {
 protected:
  int started_ = 0;

  void StartThread() {
    ASSERT_EQ(IbCastPortRecoveryThreadStart(), ncclSuccess);
    started_++;
  }
  void StopThread() {
    P2pResiliencyStopRecoveryThread();
    started_--;
  }
  void TearDown() override {
    while (started_ > 0) StopThread();
    P2pResiliencyRecoveryMicrotest::TearDown();
  }

  bool WaitForState(int dev, ncclIbResiliencyDevState want) {
    for (int i = 0; i < 10000 && State(dev) != want; i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return State(dev) == want;
  }
};

TEST_F(P2pResiliencyRecoveryThreadMicrotest, ThreadStart_RecoveryOff_StartsNothing) {
  StartThread();
  EXPECT_FALSE(IbCastPortRecoveryThreadActive.load());
  EXPECT_EQ(IbCastPortRecoveryRefCount.load(), 0);
  StopThread();
  EXPECT_EQ(IbCastPortRecoveryRefCount.load(), 0);
}

TEST_F(P2pResiliencyRecoveryThreadMicrotest, ThreadStartStop_LastUserStopsTheThread) {
  scene_.params["IB_RESILIENCY_PORT_RECOVERY"] = 1;
  StartThread();
  StartThread();
  EXPECT_TRUE(IbCastPortRecoveryThreadActive.load());
  EXPECT_EQ(IbCastPortRecoveryRefCount.load(), 2);
  ncclIbResiliency res = {};
  ncclIbNetCommBase base = {};
  res.baseComm = &base;
  ASSERT_EQ(IbCastPortRecoveryInit(&res), ncclSuccess);
  EXPECT_TRUE(res.recoveryEnabled);
  StopThread();
  EXPECT_TRUE(IbCastPortRecoveryThreadActive.load());
  EXPECT_TRUE(IbCastPortRecoveryAsyncThread.joinable());
  StopThread();
  EXPECT_FALSE(IbCastPortRecoveryThreadActive.load());
  EXPECT_FALSE(IbCastPortRecoveryAsyncThread.joinable());
  ASSERT_EQ(IbCastPortRecoveryInit(&res), ncclSuccess);
  EXPECT_FALSE(res.recoveryEnabled);
}

// Pins AICOMRCCL-3103; a fix flips this pin.
TEST_F(P2pResiliencyRecoveryThreadMicrotest, ThreadStop_KnownBug_ClearsTheFlagWithoutTheMutex) {
  scene_.params["IB_RESILIENCY_PORT_RECOVERY"] = 1;
  StartThread();
  started_--;
  auto stopped = std::make_shared<std::atomic<bool>>(false);
  std::thread stopper;
  bool clearedWhileWeHeldTheMutex = false;
  {
    std::lock_guard<std::mutex> lock(IbCastPortRecoveryMutex);
    stopper = std::thread([stopped] {
      IbCastPortRecoveryThreadStop();
      *stopped = true;
    });
    for (int i = 0; i < 5000 && IbCastPortRecoveryThreadActive.load(); i++) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    clearedWhileWeHeldTheMutex = !IbCastPortRecoveryThreadActive.load();
  }
  EXPECT_TRUE(clearedWhileWeHeldTheMutex);
  // Bounded: a stop that never finishes fails the test instead of hanging it.
  for (int i = 0; i < 10000 && !*stopped; i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (!*stopped) {
    stopper.detach();
    FAIL() << "IbCastPortRecoveryThreadStop did not return";
  }
  stopper.join();
}

TEST_F(P2pResiliencyRecoveryThreadMicrotest, Close_RecoveryOffOrThreadStopped_ReturnsAtOnce) {
  ncclIbResiliency res = {};
  ncclIbNetCommBase base = {};
  res.baseComm = &base;
  EXPECT_EQ(IbCastPortRecoveryClose(&res), ncclSuccess);
  res.recoveryEnabled = true;
  EXPECT_EQ(IbCastPortRecoveryClose(&res), ncclSuccess);
}

TEST_F(P2pResiliencyRecoveryThreadMicrotest, HandleCloseRequests_DropsOnlyTheClosingContextsQueueItems) {
  ncclIbResiliency closing = {}, other = {};
  ncclIbNetCommBase base = {};
  closing.baseComm = other.baseComm = &base;
  auto item = [](ncclIbResiliency* res) {
    ncclIbPortRecoveryContext* ctx = (ncclIbPortRecoveryContext*)calloc(1, sizeof(ncclIbPortRecoveryContext));
    ctx->resCtx = res;
    return ctx;
  };
  ncclIbPortRecoveryContext* kept = item(&other);
  std::list<ncclIbPortRecoveryContext*> queue = {item(&closing), kept, item(&closing)};
  ncclIbPortRecoveryCloseRequest req = {&closing, false};
  std::vector<ncclIbPortRecoveryCloseRequest*> requests = {&req};
  IbCastPortRecoveryHandleCloseRequests(requests, queue);
  EXPECT_EQ(queue, (std::list<ncclIbPortRecoveryContext*>{kept}));
  free(kept);
}

TEST_F(P2pResiliencyRecoveryThreadMicrotest, Thread_DrivesASenderRecoveryToRecovered) {
  scene_.params["IB_RESILIENCY_PORT_RECOVERY"] = 1;
  scene_.params["IB_RESILIENCY_PORT_RECOVERY_START_DELAY"] = 0;
  Make(true, 2, 2);
  ibv_cq* cq = res_->devs[1].portRecoveryCq;
  // The peer: every alive batch completes and is ACKed; the final ACK completes.
  scene_.postSend = [this, cq](ibv_qp*, ibv_send_wr* wr) {
    scene_.PushCqe(cq, P2pResiliencyWc(0, IBV_WC_SUCCESS, IBV_WC_SEND));
    if (wr->imm_data != NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID) {
      scene_.PushCqe(cq, P2pResiliencyWc(0, IBV_WC_SUCCESS, IBV_WC_RECV, NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID));
    }
    return 0;
  };
  StartThread();
  res_->devs[1].state.store(ncclIbResiliencyDevStateRecoveryInProgress);
  ASSERT_EQ(IbCastPortRecoveryHandleFailure(res_, 1), ncclSuccess);
  EXPECT_TRUE(WaitForState(1, ncclIbResiliencyDevStateRecovered));
  EXPECT_EQ(IbCastPortRecoveryClose(res_), ncclSuccess);
  StopThread();
  EXPECT_EQ(State(0), ncclIbResiliencyDevStateOk);
}

TEST_F(P2pResiliencyRecoveryThreadMicrotest, Thread_FailedRecovery_MarksTheDeviceAndMovesOn) {
  scene_.params["IB_RESILIENCY_PORT_RECOVERY"] = 1;
  scene_.params["IB_RESILIENCY_PORT_RECOVERY_START_DELAY"] = 0;
  Make(false, 1, 1);
  ScopedHook rtr(g_ibCastQpRtr, [](ncclIbQp*) { return ncclSystemError; });
  StartThread();
  res_->devs[0].state.store(ncclIbResiliencyDevStateRecoveryInProgress);
  ASSERT_EQ(IbCastPortRecoveryHandleFailure(res_, 0), ncclSuccess);
  EXPECT_TRUE(WaitForState(0, ncclIbResiliencyDevStateRecoveryFailed));
  EXPECT_EQ(IbCastPortRecoveryClose(res_), ncclSuccess);
  StopThread();
}

}  // namespace
