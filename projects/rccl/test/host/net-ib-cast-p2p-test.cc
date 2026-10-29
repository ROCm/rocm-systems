/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/p2p.cc (via NET_IB_CAST_P2P_CC_PATH): isend/irecv/iflush/test
// and the CAST fault-injection API, driven against fake verbs devices (IbVerbsScene.h).

#include <gtest/gtest.h>

#include <arpa/inet.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "IbVerbsScene.h"
#include "ScopedHook.h"
#include "common_cast.h"
#include "fakes/nccl_fakes.h"
#include "fakes/net_ib_cast_common_fakes.h"
#include "fakes/net_ib_cast_init_fakes.h"
#include "fakes/net_ib_cast_p2p_resiliency_fakes.h"
#include "fakes/socket_fakes.h"
#include "ibvsymbols.h"
#include "net_telemetry.h"
#include "qp_sharing.h"

#include "fakes/param_redirect.h"

#include NET_IB_CAST_P2P_CC_PATH

extern struct ncclIbvSymbols ibvSymbols;
extern FILE* IbCastQpSchedLogStream;  // scheduler.cc, whose IbCastLogSched the send path calls

namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

constexpr uint64_t kCtsFifoAddr = 0x7000000;
constexpr uint64_t kCmplsAddr = 0x9000000;
constexpr uint64_t kRemoteAddr = 0x5000000;
constexpr uint32_t kRemoteRkey = 0x400;
constexpr uint32_t kCtsRkey = 0x300;
constexpr uint32_t kCmplsRkey = 0x600;
constexpr uint32_t kSgeLkey = 0x700;
constexpr uint32_t kLkey = 0x100;
constexpr uint32_t kRkey = 0x200;
constexpr char kPeer[] = "10.1.2.3<4242>";
constexpr int kPollBatch = 4;  // IbCastTest's batch off AINIC; test/host/CMakeLists.txt pins it to p2p.cc

class NetIbCastP2pMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    snprintf(verbs_[1].device.name, sizeof(verbs_[1].device.name), "mlx5_1");
    for (int i = 0; i < IbVerbsScene::kQps; i++) verbs_[1].qps[i].qp_num = 0x200 + i;
    for (int d = 0; d < 2; d++) {
      mr_[d].lkey = kLkey + d;
      mr_[d].rkey = kRkey + d;
      mh_.mrs[d] = &mr_[d];
    }
    send_ = std::make_unique<ncclIbSendComm>();
    recv_ = std::make_unique<ncclIbRecvComm>();
  }

  void TearDown() override {
    ResetNetIbCastCommonFakes();
    ResetNetIbCastInitFakes();
    ResetNetIbCastP2pResiliencyFakes();
    ResetSocketFakes();
    rcclTelemetryEnabled = 0;
    rcclTelemetryCfg = {};
    memset(rcclTelemetryDevs, 0, sizeof(rcclTelemetryDevs));
    IbCastQpSharingGlobalEnable = false;
    castGlobalQpSchedParms = {};
    for (auto& e : g_IbCastCommTable) e = {};
    for (int d = 0; d < 2; d++) {
      IbCastDevs[d].stats.fatalErrorCount = 0;
      IbCastDevs[d].speed = 0;
    }
    IbCastArThreshold = arThreshold_;
    if (IbCastQpSchedLogStream) fclose(IbCastQpSchedLogStream);
    IbCastQpSchedLogStream = nullptr;
    free(logText_);
  }

  // nqps QPs spread over ndevs fake devices: activeQps[ndevs * q + d] is the q-th QP of device d, as connect.cc
  // lays them out (IbCastCommBaseGetQpByIndex reads that layout back). With the scheduler off, a request spreads
  // over every QP (NCCL_IB_SPLIT_DATA_ON_QPS=1); tests of the one-QP-per-device default reset splitDataOnQps.
  void Wire(ncclIbNetCommBase& base, int nqps, int ndevs, int scheme) {
    base.ready = 1;
    base.nqps = nqps;
    base.nDataQps = ndevs;
    base.splitDataOnQps = 1;
    base.vProps.ndevs = ndevs;
    base.recvMatchingScheme = scheme;
    for (int d = 0; d < ndevs; d++) {
      base.vProps.devs[d] = d;
      ncclIbNetCommDevBase* dev = IbCastGetNetCommDevBase(&base, d);
      dev->ibDevN = d;
      dev->pd = &verbs_[d].pd;
      dev->cq = &verbs_[d].cq;
      base.remDevs[d].rkey = kCtsRkey + d;
    }
    for (int i = 0; i < nqps; i++) {
      base.qps[i].qp = &verbs_[i % ndevs].qps[i / ndevs];
      base.qps[i].devIndex = i % ndevs;
      base.qps[i].remDevIdx = i % ndevs;
      base.activeQps[i] = &base.qps[i];
    }
  }

  void WireSend(int nqps = 1, int ndevs = 1, int scheme = BY_INDEX) {
    send_->base.isSend = true;
    Wire(send_->base, nqps, ndevs, scheme);
    send_->remCmplsRecords.addr = kCmplsAddr;
    for (int d = 0; d < ndevs; d++) {
      send_->remCmplsRecords.rkeys[d] = kCmplsRkey + d;
      send_->devs[d].sge.lkey = kSgeLkey + d;
    }
  }

  void WireRecv(int nqps = 1, int ndevs = 1, int scheme = BY_INDEX) {
    recv_->base.isSend = false;
    Wire(recv_->base, nqps, ndevs, scheme);
    recv_->remCtsFifo.addr = kCtsFifoAddr;
    recv_->remCtsFifo.flags = IBV_SEND_INLINE;
    recv_->ibRecvWorkRequest.wr_id = NCCL_IB_RECV_WR_ID_DUMMY;
    for (int d = 0; d < ndevs; d++) {
      recv_->devs[d].sge.lkey = kSgeLkey + d;
      recv_->devs[d].gpuFlush.qp.qp = &verbs_[d].qps[IbVerbsScene::kQps - 1];
    }
  }

  // The CTS the receiver RDMA-writes into the sender's FIFO for request r of the next slot.
  ncclIbSendFifo& Cts(int r, int nreqs, int tag, size_t size, uint16_t rxReqIndex = 0) {
    const uint64_t head = send_->base.fifoHead;
    ncclIbSendFifo& e = send_->ctsFifo[head % NET_IB_MAX_REQUESTS][r];
    e.addr = kRemoteAddr + 0x10000 * r;
    e.size = size;
    for (int d = 0; d < NCCL_IB_MAX_DEVS_PER_NIC; d++) e.rkeys[d] = kRemoteRkey + d;
    e.nreqs = nreqs;
    e.tag = tag;
    e.idx = (uint32_t)(head + 1);
    e.rxReqIndex = rxReqIndex;
    return e;
  }

  ncclResult_t Isend(size_t size, int tag, void** req, void* data = nullptr) {
    return IbCastIsend(send_.get(), data ? data : buf_, size, tag, &mh_, nullptr, req);
  }

  ncclResult_t Irecv(int n, void** req, size_t size = 4096) {
    for (int i = 0; i < n; i++) {
      rdata_[i] = buf_ + 0x100 * i;
      rsizes_[i] = size;
      rtags_[i] = 0x10 + i;
      rmh_[i] = &mh_;
    }
    return IbCastIrecv(recv_.get(), n, rdata_, rsizes_, rtags_, rmh_, nullptr, req);
  }

  ncclResult_t Test(void* req, int* done, int* sizes) { return IbCastTest(req, done, sizes); }

  // Any positive bucket interval will do: a zeroed config divides by zero in the latency histogram.
  static void EnableTelemetry() {
    rcclTelemetryCfg.histogram_max_buckets = 4;
    rcclTelemetryCfg.histogram_bucket_interval_ns = 1000;
    rcclTelemetryEnabled = 1;
  }

  const std::vector<IbVerbsScene::Wr>& Chain(int dev, size_t i) { return verbs_[dev].sends.at(i).chain; }
  ncclIbRequest* Req(void* r) { return static_cast<ncclIbRequest*>(r); }

  IbVerbsScene verbs_[2];
  ibv_mr mr_[2]{};
  ncclIbMrHandle mh_{};
  char buf_[0x1000]{};
  // One past the limit, for the n > NCCL_NET_IB_MAX_RECVS rejection.
  void* rdata_[NCCL_NET_IB_MAX_RECVS + 1]{};
  size_t rsizes_[NCCL_NET_IB_MAX_RECVS + 1]{};
  int rtags_[NCCL_NET_IB_MAX_RECVS + 1]{};
  void* rmh_[NCCL_NET_IB_MAX_RECVS + 1]{};
  std::unique_ptr<ncclIbSendComm> send_;
  std::unique_ptr<ncclIbRecvComm> recv_;
  const int64_t arThreshold_ = IbCastArThreshold;
  char* logText_ = nullptr;  // backs an open_memstream log stream
  size_t logLen_ = 0;
};

using NetIbCastP2pRequestMicrotest = NetIbCastP2pMicrotest;

TEST_F(NetIbCastP2pRequestMicrotest, GetRequest_TakesTheFirstUnusedSlotAndClearsIt) {
  WireSend();
  ncclIbNetCommBase& base = send_->base;
  base.reqs[0].type = NCCL_NET_IB_REQ_SEND;
  base.reqs[1].events[0] = 5;
  base.reqs[1].devBases[0] = &send_->devs[0].base;
  base.reqs[1].tel_post_ts = 9;
  base.reqs[1].sock = &base.sock;
  ncclIbRequest* req = nullptr;
  ASSERT_EQ(IbCastGetRequest(&base, &req), ncclSuccess);
  EXPECT_EQ(req, &base.reqs[1]);
  EXPECT_EQ(req->base, &base);
  EXPECT_EQ(req->events[0], 0);
  EXPECT_EQ(req->devBases[0], nullptr);
  EXPECT_EQ(req->tel_post_ts, 0u);
  EXPECT_EQ(req->sock, nullptr);
  EXPECT_EQ(IbCastFreeRequest(&base.reqs[0]), ncclSuccess);
  EXPECT_EQ(base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbCastP2pRequestMicrotest, GetRequest_AllSlotsBusy_Fails) {
  for (auto& r : send_->base.reqs) r.type = NCCL_NET_IB_REQ_RECV;
  ncclIbRequest* req = &send_->base.reqs[0];
  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&] { res = IbCastGetRequest(&send_->base, &req); });
  EXPECT_EQ(res, ncclInternalError);
  EXPECT_EQ(req, nullptr);
  EXPECT_TRUE(LogHas(log, "unable to allocate requests")) << log;
}

TEST_F(NetIbCastP2pRequestMicrotest, AddEvent_CountsPerDeviceAndRecordsItsBase) {
  WireRecv(2, 2);
  ncclIbRequest& req = recv_->base.reqs[0];
  req.base = &recv_->base;
  IbCastAddEvent(&req, 1);
  IbCastAddEvent(&req, 1);
  IbCastAddEventCTS(&req, 0);
  EXPECT_EQ(req.events[1], 2);
  EXPECT_EQ(req.events[0], 1);
  EXPECT_EQ(req.ctsEvents[0], 1);
  EXPECT_EQ(req.ctsEvents[1], 0);
  EXPECT_EQ(req.devBases[1], &recv_->devs[1].base);
  EXPECT_EQ(req.devBases[0], &recv_->devs[0].base);
}

TEST_F(NetIbCastP2pRequestMicrotest, RetrieveAsIndex_RejectsIndexesPastTheTable) {
  ncclIbRequest* req = nullptr;
  EXPECT_EQ(IbCastRequestRetrieveAsIndex(send_->base.reqs, NET_IB_MAX_REQUESTS - 1, &req), ncclSuccess);
  EXPECT_EQ(req, &send_->base.reqs[NET_IB_MAX_REQUESTS - 1]);
  CaptureLog([&] {
    EXPECT_EQ(IbCastRequestRetrieveAsIndex(send_->base.reqs, NET_IB_MAX_REQUESTS, &req), ncclInternalError);
  });
  EXPECT_EQ(req, &send_->base.reqs[NET_IB_MAX_REQUESTS - 1]);
}

TEST_F(NetIbCastP2pRequestMicrotest, CtsFifoAccessors_ReadBothLayouts) {
  ncclIbSendFifo row[2]{};
  row[1] = {0xabc, 77, {1, 2, 3, 4}, 2, 0x55, 9, 3, {}};
  EXPECT_EQ(ctsFifoAddr(row, 1), 0xabcu);
  EXPECT_EQ(ctsFifoSize(row, 1), 77u);
  EXPECT_EQ(ctsFifoRkey(row, 1, 2), 3u);
  EXPECT_EQ(ctsFifoNreqs(row, 1), 2u);
  EXPECT_EQ(ctsFifoTag(row, 1), 0x55u);
  EXPECT_EQ(ctsFifoIdx(row, 1), 9u);
  EXPECT_EQ(ctsFifoRxReqIndex(row, 1), 3u);

  IbCastAinicCtsInlineData = true;
  ncclIbSendFifoCtsInline inl[4]{};
  inl[3].addr = 0xdef;
  inl[3].rkeys[0] = 42;
  inl[3].size = 88;
  inl[3].nreqs = 4;
  inl[3].rxReqIndex = 7;
  inl[3].tag = 0x66;
  inl[3].idx = 12;
  EXPECT_EQ(ctsFifoAddr(inl, 3), 0xdefu);
  EXPECT_EQ(ctsFifoSize(inl, 3), 88u);
  EXPECT_EQ(ctsFifoRkey(inl, 3, 2), 42u);
  EXPECT_EQ(ctsFifoNreqs(inl, 3), 4u);
  EXPECT_EQ(ctsFifoTag(inl, 3), 0x66u);
  EXPECT_EQ(ctsFifoIdx(inl, 3), 12u);
  EXPECT_EQ(ctsFifoRxReqIndex(inl, 3), 7u);
}

// Without CTS offload the CTS rides QP 0 of device (id % ndevs); with it, any active QP may carry one.
TEST_F(NetIbCastP2pRequestMicrotest, CtsQpSelection_FollowsTheOffloadMode) {
  WireRecv(4, 2);
  ncclIbQp* qp = nullptr;
  ASSERT_EQ(IbCastRecvCommGetQpForCts(recv_.get(), 3, &qp), ncclSuccess);
  EXPECT_EQ(qp, &recv_->base.qps[1]);
  EXPECT_TRUE(IbCastRecvCommIsCtsQp(recv_.get(), 0));
  EXPECT_TRUE(IbCastRecvCommIsCtsQp(recv_.get(), 1));
  EXPECT_FALSE(IbCastRecvCommIsCtsQp(recv_.get(), 2));
  EXPECT_FALSE(IbCastRecvCommIsCtsQp(recv_.get(), 3));

  recv_->useCtsOffload = true;
  ASSERT_EQ(IbCastRecvCommGetQpForCts(recv_.get(), 7, &qp), ncclSuccess);
  EXPECT_EQ(qp, &recv_->base.qps[3]);
  for (int i = 0; i < 4; i++) EXPECT_TRUE(IbCastRecvCommIsCtsQp(recv_.get(), i)) << i;
}

TEST_F(NetIbCastP2pRequestMicrotest, Params_KeepTheirDocumentedDefaults) {
  EXPECT_EQ(ncclParamIbCastArThreshold(), -2);
  EXPECT_EQ(ncclParamIbCastReceiverSideMatchingScheme(), -2);
  EXPECT_EQ(rcclParamIbCastGdrFlushGpuMemNoRelaxedOrdering(), 1);
  EXPECT_STREQ(IbCastReqTypeStr[NCCL_NET_IB_REQ_GIN_IPUT], "IPut");
}

using NetIbCastP2pIsendMicrotest = NetIbCastP2pMicrotest;

TEST_F(NetIbCastP2pIsendMicrotest, Isend_CommNotReady_FailsWithoutARequest) {
  WireSend();
  send_->base.ready = 0;
  void* req = buf_;
  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&] { res = Isend(64, 1, &req); });
  EXPECT_EQ(res, ncclInternalError);
  EXPECT_EQ(req, nullptr);
  EXPECT_TRUE(LogHas(log, "comm->base.ready == 0")) << log;
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_FatalErrorSeen_FailsBeforeTouchingTheFifo) {
  WireSend();
  Cts(0, 1, 1, 64);
  send_->base.stats.fatalErrorCount = 1;
  void* req = nullptr;
  EXPECT_EQ(Isend(64, 1, &req), ncclSystemError);
  EXPECT_TRUE(verbs_[0].sends.empty());
  EXPECT_EQ(send_->base.fifoHead, 0u);
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_NoCtsYet_ReturnsNoRequestAndCountsASlotMiss) {
  WireSend();
  RcclQpStats stats{};
  send_->base.qps[0].telQpStats = &stats;
  EnableTelemetry();
  void* req = buf_;
  EXPECT_EQ(Isend(64, 1, &req), ncclSuccess);
  EXPECT_EQ(req, nullptr);
  EXPECT_EQ(stats.num_slot_miss, 1u);
  send_->base.qpIndex = 1;  // past nqps: nothing to charge
  EXPECT_EQ(Isend(64, 1, &req), ncclSuccess);
  EXPECT_EQ(stats.num_slot_miss, 1u);
  EXPECT_TRUE(verbs_[0].sends.empty());
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_TagMismatch_ReturnsNoRequest) {
  WireSend();
  Cts(0, 1, 2, 64);
  void* req = buf_;
  EXPECT_EQ(Isend(64, 1, &req), ncclSuccess);
  EXPECT_EQ(req, nullptr);
  EXPECT_TRUE(verbs_[0].sends.empty());
  EXPECT_EQ(send_->base.fifoHead, 0u);
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_CtsWithoutAddressOrKey_IsRejectedWithThePeer) {
  WireSend();
  ScopedHook toString(g_ncclSocketToString, [](const union ncclSocketAddress*, char* buf, size_t len, int) {
    snprintf(buf, len, "%s", kPeer);
    return buf;
  });
  for (int which = 0; which < 2; which++) {
    ncclIbSendFifo& cts = Cts(0, 1, 1, 64);
    if (which == 0) cts.addr = 0;
    if (which == 1) cts.rkeys[0] = 0;
    void* req = nullptr;
    ncclResult_t res = ncclSuccess;
    const std::string log = CaptureLog([&] { res = Isend(64, 1, &req); });
    EXPECT_EQ(res, ncclInternalError) << which;
    EXPECT_TRUE(LogHas(log, "peer 10.1.2.3<4242> posted incorrect receive info")) << log;
  }
  EXPECT_TRUE(verbs_[0].sends.empty());
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_OneQp_PostsOneWriteWithImmCarryingTheReceiverSlot) {
  WireSend();
  Cts(0, 1, 1, 100, 0x2a);
  void* req = nullptr;
  ASSERT_EQ(Isend(200, 1, &req), ncclSuccess);
  ASSERT_NE(req, nullptr);
  ASSERT_EQ(verbs_[0].sends.size(), 1u);
  EXPECT_EQ(verbs_[0].sends[0].qp, &verbs_[0].qps[0]);
  const auto& chain = Chain(0, 0);
  ASSERT_EQ(chain.size(), 1u);
  const ibv_send_wr& wr = chain[0].wr;
  EXPECT_EQ(wr.opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_EQ(wr.send_flags, IBV_SEND_SIGNALED);
  EXPECT_EQ(be32toh(wr.imm_data), 0x2au << WR_IMM_RX_REQ_IDX_BIT_POS);
  EXPECT_EQ(wr.wr.rdma.remote_addr, kRemoteAddr);
  EXPECT_EQ(wr.wr.rdma.rkey, kRemoteRkey);
  ASSERT_TRUE(chain[0].hasSge);
  EXPECT_EQ(chain[0].sge.addr, (uint64_t)buf_);
  EXPECT_EQ(chain[0].sge.length, 100u);  // clamped to the receive size
  EXPECT_EQ(chain[0].sge.lkey, kLkey);
  // BY_INDEX remaps wr_id to a scheduler slot that remembers the original slot id.
  EXPECT_EQ(wr.wr_id, (uint64_t)&send_->base.remapWrId[0]);
  EXPECT_EQ(send_->base.remapWrId[0].origWrId, 0u);
  EXPECT_EQ(send_->base.remapWrId[0].tx.bytes, 100u);

  ncclIbRequest* r = Req(req);
  EXPECT_EQ(r->type, NCCL_NET_IB_REQ_SEND);
  EXPECT_EQ(r->send.size, 100);
  EXPECT_EQ(r->events[0], 1);
  EXPECT_EQ(r->devBases[0], &send_->devs[0].base);
  EXPECT_EQ(r->send.lkeys[0], kLkey);
  EXPECT_EQ(send_->sendReqs[0][0], r);
  EXPECT_EQ(send_->sendReqsCnt[0], 1);
  EXPECT_EQ(send_->remCmplsRecords.elems[0][0], 100);
  EXPECT_EQ(send_->base.fifoHead, 1u);
}

// 1000 bytes over 3 QPs: chunks are DIVUP(1000 / 3) rounded up to 128 = 384, so 384 + 384 + 232.
TEST_F(NetIbCastP2pIsendMicrotest, Isend_SplitOverQps_ChunksAre128Aligned) {
  WireSend(3);
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(1000, 1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].sends.size(), 3u);
  const uint32_t lens[] = {384, 384, 232};
  uint64_t offset = 0;
  for (int i = 0; i < 3; i++) {
    const auto& c = Chain(0, i)[0];
    EXPECT_EQ(verbs_[0].sends[i].qp, &verbs_[0].qps[i]);
    EXPECT_EQ(c.sge.length, lens[i]) << i;
    EXPECT_EQ(c.sge.addr, (uint64_t)buf_ + offset) << i;
    EXPECT_EQ(c.wr.wr.rdma.remote_addr, kRemoteAddr + offset) << i;
    EXPECT_TRUE(be32toh(c.wr.imm_data) & WR_IMM_SPLIT_DATA_FLAG) << i;
    EXPECT_EQ(c.wr.wr_id, (uint64_t)&send_->base.remapWrId[i]);
    EXPECT_EQ(send_->base.remapWrId[i].qpIndex, i);
    EXPECT_EQ(send_->base.remapWrId[i].tx.bytes, lens[i]);
    offset += lens[i];
  }
  EXPECT_EQ(Req(req)->events[0], 3);
  for (int i = 0; i < 3; i++) EXPECT_TRUE(Req(req)->send.sentData[i]) << i;
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_TooSmallForEveryQp_PostsEmptyWritesOnTheRest) {
  WireSend(3);
  send_->base.fifoHead = 4;  // id 4 starts on qp 4 % 3 = 1
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(100, 1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].sends.size(), 3u);
  EXPECT_EQ(verbs_[0].sends[0].qp, &verbs_[0].qps[1]);
  EXPECT_EQ(Chain(0, 0)[0].sge.length, 100u);
  for (int i = 1; i < 3; i++) {
    EXPECT_FALSE(Chain(0, i)[0].hasSge) << i;
    EXPECT_EQ(Chain(0, i)[0].wr.num_sge, 0) << i;
    EXPECT_EQ(Chain(0, i)[0].wr.opcode, IBV_WR_RDMA_WRITE_WITH_IMM) << i;
  }
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_TwoDevices_UsesEachDevicesKeysAndCountsEventsPerDevice) {
  WireSend(2, 2);
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(256, 1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].sends.size(), 1u);
  ASSERT_EQ(verbs_[1].sends.size(), 1u);
  EXPECT_EQ(Chain(0, 0)[0].sge.lkey, kLkey);
  EXPECT_EQ(Chain(1, 0)[0].sge.lkey, kLkey + 1);
  EXPECT_EQ(Chain(0, 0)[0].wr.wr.rdma.rkey, kRemoteRkey);
  EXPECT_EQ(Chain(1, 0)[0].wr.wr.rdma.rkey, kRemoteRkey + 1);
  EXPECT_EQ(Req(req)->events[0], 1);
  EXPECT_EQ(Req(req)->events[1], 1);
  EXPECT_EQ(Req(req)->devBases[1], &send_->devs[1].base);
}

// NCCL_IB_SPLIT_DATA_ON_QPS=0 (the default) with two QPs per device: a request goes over nDataQps QPs, one per
// device, not over all four.
TEST_F(NetIbCastP2pIsendMicrotest, Isend_NoSplitOnQps_UsesOneQpPerDevice) {
  WireSend(4, 2);
  send_->base.splitDataOnQps = 0;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(1024, 1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].sends.size(), 1u);
  ASSERT_EQ(verbs_[1].sends.size(), 1u);
  EXPECT_EQ(verbs_[0].sends[0].qp, &verbs_[0].qps[0]);
  EXPECT_EQ(verbs_[1].sends[0].qp, &verbs_[1].qps[0]);
  EXPECT_EQ(Chain(0, 0)[0].sge.length, 512u);
  EXPECT_EQ(Chain(1, 0)[0].sge.length, 512u);
  EXPECT_EQ(Chain(1, 0)[0].sge.addr, (uint64_t)buf_ + 512);
  EXPECT_EQ(Chain(1, 0)[0].wr.wr.rdma.remote_addr, kRemoteAddr + 512);
  EXPECT_TRUE(be32toh(Chain(0, 0)[0].wr.imm_data) & WR_IMM_SPLIT_DATA_FLAG);
  EXPECT_EQ(Req(req)->desc.nqps, 2);
  EXPECT_EQ(Req(req)->events[0], 1);
  EXPECT_EQ(Req(req)->events[1], 1);
}

// Below RCCL_IB_SPLIT_DATA_THRESHOLD the whole send rides the first QP and the others carry empty writes.
TEST_F(NetIbCastP2pIsendMicrotest, Isend_BelowSplitThreshold_SendsEverythingOnTheFirstQp) {
  ScopedHook param(g_loadParam, ParamOverride({{"RCCL_IB_SPLIT_DATA_THRESHOLD", 1001}}));
  WireSend(3);
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(1000, 1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].sends.size(), 3u);
  EXPECT_EQ(Chain(0, 0)[0].sge.length, 1000u);
  EXPECT_EQ(Chain(0, 0)[0].sge.addr, (uint64_t)buf_);
  for (int i = 1; i < 3; i++) {
    EXPECT_FALSE(Chain(0, i)[0].hasSge) << i;
    EXPECT_EQ(Chain(0, i)[0].wr.num_sge, 0) << i;
  }
  EXPECT_EQ(Req(req)->events[0], 3);

  // At the threshold the send is split evenly again: 384 + 384 + 233.
  Cts(0, 1, 1, 4096);
  ASSERT_EQ(Isend(1001, 1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].sends.size(), 6u);
  EXPECT_EQ(Chain(0, 3)[0].sge.length, 384u);
  EXPECT_EQ(Chain(0, 4)[0].sge.length, 384u);
  EXPECT_EQ(Chain(0, 5)[0].sge.length, 233u);
}

// The threshold only applies with the scheduler off: a weighted split still splits a send below it.
TEST_F(NetIbCastP2pIsendMicrotest, Isend_BelowSplitThresholdWithScheduler_KeepsTheWeightedSplit) {
  ScopedHook param(g_loadParam, ParamOverride({{"RCCL_IB_SPLIT_DATA_THRESHOLD", 4096}}));
  WireSend(2);
  send_->base.schedParmsInit = true;
  send_->base.schedParms.enable = true;
  send_->base.schedParms.splitData = true;
  send_->base.schedParms.splitDataMin = 1;
  send_->base.schedParms.updateInterval = 1000000000;
  send_->base.qpTxSchedInit = true;
  send_->base.qpTxSched[0].weight = 0.75;
  send_->base.qpTxSched[1].weight = 0.25;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(1000, 1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].sends.size(), 2u);
  EXPECT_EQ(Chain(0, 0)[0].sge.length, 640u);
  EXPECT_EQ(Chain(0, 1)[0].sge.length, 360u);
}

// A two-request multi-recv posts only once both sends have matched, as one chain: a write per request, then a
// write-with-imm that carries both sizes into the receiver's completion record for the slot.
TEST_F(NetIbCastP2pIsendMicrotest, Isend_MultiRecv_PostsOnceWithTheSizesRecord) {
  WireSend();
  send_->base.fifoHead = 5;
  Cts(0, 2, 0x10, 4096, 3);
  Cts(1, 2, 0x11, 4096, 3);
  void* first = nullptr;
  void* second = nullptr;
  ASSERT_EQ(Isend(300, 0x10, &first), ncclSuccess);
  ASSERT_NE(first, nullptr);
  EXPECT_TRUE(verbs_[0].sends.empty());
  EXPECT_EQ(send_->base.fifoHead, 5u);
  ASSERT_EQ(Isend(500, 0x11, &second, buf_ + 0x800), ncclSuccess);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(verbs_[0].sends.size(), 1u);
  const auto& chain = Chain(0, 0);
  ASSERT_EQ(chain.size(), 3u);
  EXPECT_EQ(chain[0].wr.opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(chain[0].wr.send_flags, 0);
  EXPECT_EQ(chain[0].wr.wr_id, 0x05u);
  EXPECT_EQ(chain[0].sge.length, 300u);
  EXPECT_EQ(chain[1].wr.wr_id, 0x0505u);
  EXPECT_EQ(chain[1].sge.length, 500u);
  EXPECT_EQ(chain[1].sge.addr, (uint64_t)(buf_ + 0x800));
  EXPECT_EQ(chain[1].wr.wr.rdma.remote_addr, kRemoteAddr + 0x10000);
  const auto& imm = chain[2];
  EXPECT_EQ(imm.wr.opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_EQ(imm.wr.send_flags, IBV_SEND_SIGNALED);
  EXPECT_EQ(imm.wr.wr.rdma.remote_addr, kCmplsAddr + 5 * sizeof(ncclIbRequestCompletionRecord));
  EXPECT_EQ(imm.wr.wr.rdma.rkey, kCmplsRkey);
  EXPECT_EQ(imm.sge.addr, (uint64_t)send_->remCmplsRecords.elems[5]);
  EXPECT_EQ(imm.sge.length, 2 * sizeof(int));
  EXPECT_EQ(imm.sge.lkey, kSgeLkey);
  EXPECT_EQ(send_->remCmplsRecords.elems[5][0], 300);
  EXPECT_EQ(send_->remCmplsRecords.elems[5][1], 500);
  EXPECT_EQ(imm.wr.wr_id, (uint64_t)&send_->base.remapWrId[0]);
  EXPECT_EQ(send_->base.remapWrId[0].origWrId, 0x0505u);
  EXPECT_EQ(send_->sendReqsCnt[5], 2);
  EXPECT_EQ(send_->base.fifoHead, 6u);

  // Every slot entry already has its send: a third isend on the same slot finds nothing to match.
  send_->base.fifoHead = 5;
  void* third = buf_;
  EXPECT_EQ(Isend(10, 0x11, &third), ncclSuccess);
  EXPECT_EQ(third, nullptr);
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_AdaptiveRouting_SplitsBulkWriteFromTheImmediate) {
  WireSend();
  send_->ar = 1;
  IbCastArThreshold = 1000;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(1001, 1, &req), ncclSuccess);
  const auto& chain = Chain(0, 0);
  ASSERT_EQ(chain.size(), 2u);
  EXPECT_EQ(chain[0].wr.opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(chain[0].wr.send_flags, 0);
  EXPECT_EQ(chain[0].sge.length, 1001u);
  EXPECT_EQ(chain[1].wr.opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_EQ(chain[1].wr.num_sge, 0);
  EXPECT_EQ(chain[1].wr.send_flags, IBV_SEND_SIGNALED);

  // At the threshold, or when both sides have an out-of-order RQ, the data rides the immediate itself.
  for (int variant = 0; variant < 2; variant++) {
    send_->base.remOooRq = send_->base.localOooRq = (variant == 1);
    Cts(0, 1, 1, 4096);
    ASSERT_EQ(Isend(variant == 0 ? 1000 : 1001, 1, &req), ncclSuccess);
    EXPECT_EQ(Chain(0, 1 + variant).size(), 1u) << variant;
  }
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_OptionalRecvCompletion_PostsAPlainWrite) {
  WireSend();
  send_->base.optRecvCompletion = true;
  Cts(0, 1, 1, 4096);
  void* req = (void*)NCCL_NET_OPTIONAL_RECV_COMPLETION;
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  const auto& chain = Chain(0, 0);
  ASSERT_EQ(chain.size(), 1u);
  EXPECT_EQ(chain[0].wr.opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(chain[0].wr.send_flags, IBV_SEND_SIGNALED);
  EXPECT_EQ(chain[0].wr.imm_data, 0u);
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_ById_CarriesTheRequestIdAndAPlainWrId) {
  WireSend(1, 1, BY_ID);
  send_->base.fifoHead = 7;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  const auto& wr = Chain(0, 0)[0].wr;
  EXPECT_EQ(be32toh(wr.imm_data), 7u);
  EXPECT_EQ(wr.wr_id, 7u);
  EXPECT_EQ(send_->base.remapWrId[0].state, NCCL_NET_IB_REMAP_UNUSED);
}

// A sharing sender stamps its commId into wr_id[63:48] and the receiver's commId into the BY_ID immediate.
TEST_F(NetIbCastP2pIsendMicrotest, Isend_QpSharing_EncodesBothCommIds) {
  WireSend(1, 1, BY_ID);
  IbCastQpSharingGlobalEnable = true;
  send_->base.qpSharing.netIbCommId = 9;
  send_->remCommId = 0x33;
  send_->base.fifoHead = 2;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  const auto& wr = Chain(0, 0)[0].wr;
  EXPECT_EQ(wr.wr_id, IbCastEncodeCommId(2, 9));
  EXPECT_EQ(be32toh(wr.imm_data), IbCastEncodeCommIdImmData(2, 0x33));

  send_->remCommId = 0;
  Cts(0, 1, 1, 4096);
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  EXPECT_EQ(be32toh(Chain(0, 1)[0].wr.imm_data), 3u);
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_QpSharingMultiRecv_StampsEveryWrInTheChain) {
  WireSend(1, 1, BY_ID);
  IbCastQpSharingGlobalEnable = true;
  send_->base.qpSharing.netIbCommId = 4;
  send_->base.fifoHead = 1;
  Cts(0, 2, 0x10, 4096);
  Cts(1, 2, 0x11, 4096);
  void* r0 = nullptr;
  void* r1 = nullptr;
  ASSERT_EQ(Isend(64, 0x10, &r0), ncclSuccess);
  ASSERT_EQ(Isend(64, 0x11, &r1), ncclSuccess);
  const auto& chain = Chain(0, 0);
  ASSERT_EQ(chain.size(), 3u);
  EXPECT_EQ(chain[0].wr.wr_id, IbCastEncodeCommId(0x01, 4));
  EXPECT_EQ(chain[1].wr.wr_id, IbCastEncodeCommId(0x0101, 4));
  EXPECT_EQ(chain[2].wr.wr_id, IbCastEncodeCommId(0x0101, 4));
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_CtsOffload_SkipsTheFifoAndRotatesQpsById) {
  WireSend(2);
  send_->useCtsOffload = true;
  send_->base.fifoHead = 3;
  void* req = nullptr;
  ASSERT_EQ(Isend(64, 99, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].sends.size(), 1u);
  EXPECT_EQ(verbs_[0].sends[0].qp, &verbs_[0].qps[1]);
  EXPECT_EQ(Req(req)->events[0], 1);
  EXPECT_EQ(send_->base.fifoHead, 4u);
}

// Weighted split with weights 0.75 / 0.25: the first QP takes floor(750 / 128) * 128 = 640, the last QP the rest.
TEST_F(NetIbCastP2pIsendMicrotest, Isend_WeightedSplit_FollowsTheSchedulerWeights) {
  WireSend(2);
  send_->base.schedParmsInit = true;
  send_->base.schedParms.enable = true;
  send_->base.schedParms.splitData = true;
  send_->base.schedParms.splitDataMin = 1;
  send_->base.schedParms.updateInterval = 1000000000;
  send_->base.qpTxSchedInit = true;
  send_->base.qpTxSched[0].weight = 0.75;
  send_->base.qpTxSched[1].weight = 0.25;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(1000, 1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].sends.size(), 2u);
  EXPECT_EQ(Chain(0, 0)[0].sge.length, 640u);
  EXPECT_EQ(Chain(0, 1)[0].sge.length, 360u);
  EXPECT_EQ(Chain(0, 1)[0].sge.addr, (uint64_t)buf_ + 640);
  EXPECT_EQ(Chain(0, 1)[0].wr.wr.rdma.remote_addr, kRemoteAddr + 640);
  EXPECT_NE(send_->base.remapWrId[0].tx.startTimeNs, 0u);
  EXPECT_NE(send_->base.nextQpTxSchedUpdateNs, 0u);
  EXPECT_FALSE(send_->base.schedParms.logEnable);
  EXPECT_EQ(send_->base.nextSchedLogNs, 0u);
}

// Once the update deadline has passed, a send re-derives the weights from the measured RTTs and logs them.
TEST_F(NetIbCastP2pIsendMicrotest, Isend_PastTheUpdateDeadline_RefreshesWeightsAndLogs) {
  WireSend(2);
  ncclIbNetCommBase& base = send_->base;
  base.schedParmsInit = true;
  base.schedParms.enable = true;
  base.schedParms.doWrr = true;
  base.schedParms.logEnable = true;
  base.schedParms.updateInterval = 1000000000;
  base.schedParms.logInterval = 1000000000;
  base.qpTxStats[0] = {50, 0, 1, 100.0};
  base.qpTxStats[1] = {50, 0, 1, 300.0};
  IbCastQpSchedLogStream = open_memstream(&logText_, &logLen_);
  for (int round = 0; round < 2; round++) {
    base.nextQpTxSchedUpdateNs = round == 0 ? 0 : 1;
    base.nextSchedLogNs = round == 0 ? 0 : 1;
    Cts(0, 1, 1, 4096);
    void* req = nullptr;
    ASSERT_EQ(Isend(64, 1, &req), ncclSuccess) << round;
    EXPECT_GT(base.nextQpTxSchedUpdateNs, 1u) << round;
    EXPECT_GT(base.nextSchedLogNs, 1u) << round;
  }
  EXPECT_TRUE(base.qpTxSchedInit);
  EXPECT_DOUBLE_EQ(base.qpTxSched[0].weight, 5.0 / 6.0);
  fflush(IbCastQpSchedLogStream);
  const std::string text(logText_, logLen_);
  EXPECT_TRUE(LogHas(text, "num qp's 2 ndevs 1")) << text;
}

class NetIbCastP2pRetransmitMicrotest : public NetIbCastP2pMicrotest {
 protected:
  // Sends `size` bytes over nqps QPs with resiliency on, then marks only qp `lost` undelivered and retransmits.
  void SendThenLose(int nqps, int lost, size_t size) {
    WireSend(nqps);
    send_->base.resiliency = res_.get();
    Cts(0, 1, 1, 4096);
    void* req = nullptr;
    ASSERT_EQ(Isend(size, 1, &req), ncclSuccess);
    ASSERT_EQ(verbs_[0].sends.size(), (size_t)nqps);
    verbs_[0].sends.clear();
    Req(req)->send.sentData[lost] = false;
    ncclResult_t r = ncclInternalError;
    CaptureLog([&] { r = IbCastMultiSend(send_.get(), 0, nqps, 0, false, false); });
    ASSERT_EQ(r, ncclSuccess);
    ASSERT_EQ(verbs_[0].sends.size(), 1u);
    EXPECT_EQ(verbs_[0].sends[0].qp, &verbs_[0].qps[lost]);
  }
  void TearDown() override {
    send_->base.resiliency = nullptr;
    NetIbCastP2pMicrotest::TearDown();
  }
  std::unique_ptr<ncclIbResiliency> res_ = std::make_unique<ncclIbResiliency>();
};

TEST_F(NetIbCastP2pRetransmitMicrotest, MultiSend_Retransmit_ResendsOnlyTheUndeliveredQp) {
  SendThenLose(2, 0, 1000);
  EXPECT_EQ(Chain(0, 0)[0].sge.addr, (uint64_t)buf_);
  EXPECT_EQ(Chain(0, 0)[0].wr.wr.rdma.remote_addr, kRemoteAddr);
  EXPECT_EQ(Chain(0, 0)[0].sge.length, 512u);
}

// Pins AICOMRCCL-3080; a fix flips this pin.
// Skipping a delivered QP advances the addresses but not sendOffsets, so the resent chunk keeps the full chunk
// length and runs past the end of the message: of 800 bytes, 512 from offset 512 (should be 288), and of 1000
// bytes over three QPs, 384 from offset 768 (should be 232).
TEST_F(NetIbCastP2pRetransmitMicrotest, MultiSend_KnownBug_RetransmitAfterSkippedQpsOverruns) {
  SendThenLose(2, 1, 800);
  EXPECT_EQ(Chain(0, 0)[0].sge.addr, (uint64_t)buf_ + 512);
  EXPECT_EQ(Chain(0, 0)[0].wr.wr.rdma.remote_addr, kRemoteAddr + 512);
  EXPECT_EQ(Chain(0, 0)[0].sge.length, 512u);
}

TEST_F(NetIbCastP2pRetransmitMicrotest, MultiSend_KnownBug_RetransmitAfterTwoSkippedQpsOverruns) {
  SendThenLose(3, 2, 1000);
  EXPECT_EQ(Chain(0, 0)[0].sge.addr, (uint64_t)buf_ + 768);
  EXPECT_EQ(Chain(0, 0)[0].wr.wr.rdma.remote_addr, kRemoteAddr + 768);
  EXPECT_EQ(Chain(0, 0)[0].sge.length, 384u);
}

TEST_F(NetIbCastP2pIsendMicrotest, MultiSend_MoreRequestsThanAMultiRecvHolds_IsRejected) {
  WireSend();
  send_->ctsFifo[0][0].nreqs = NCCL_NET_IB_MAX_RECVS + 1;
  EXPECT_EQ(IbCastMultiSend(send_.get(), 0, 1, 0, false, false), ncclInternalError);
  EXPECT_TRUE(verbs_[0].sends.empty());
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_RemapPoolExhausted_Fails) {
  WireSend();
  for (auto& r : send_->base.remapWrId) r.state = NCCL_NET_IB_REMAP_USED;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ncclResult_t res = ncclSuccess;
  CaptureLog([&] { res = Isend(64, 1, &req); });
  EXPECT_EQ(res, ncclInternalError);
  EXPECT_TRUE(verbs_[0].sends.empty());
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_NoFreeRequest_Fails) {
  WireSend();
  for (auto& r : send_->base.reqs) r.type = NCCL_NET_IB_REQ_RECV;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ncclResult_t res = ncclSuccess;
  CaptureLog([&] { res = Isend(64, 1, &req); });
  EXPECT_EQ(res, ncclInternalError);
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_PostSendFails_ReturnsSystemErrorAndKeepsTheSlot) {
  WireSend();
  verbs_[0].postSendErr = ENOMEM;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&] { res = Isend(64, 1, &req); });
  EXPECT_EQ(res, ncclSystemError);
  EXPECT_TRUE(LogHas(log, "ibv_post_send() failed")) << log;
  EXPECT_EQ(send_->base.fifoHead, 0u);
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_FaultArmedQp_CountsAFatalErrorInsteadOfPosting) {
  WireSend(2);
  ASSERT_EQ(ncclIbCastFaultSetQpError(send_.get(), 1, true), ncclSuccess);
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  EXPECT_EQ(Isend(1000, 1, &req), ncclSystemError);
  EXPECT_EQ(verbs_[0].sends.size(), 1u);
  int fatal = 0;
  ASSERT_EQ(ncclIbCastFaultGetFatalCount(send_.get(), &fatal), ncclSuccess);
  EXPECT_EQ(fatal, 1);
  CaptureLog([&] { EXPECT_EQ(Isend(64, 1, &req), ncclSystemError); });

  ASSERT_EQ(ncclIbCastFaultClear(send_.get()), ncclSuccess);
  ASSERT_EQ(ncclIbCastFaultGetFatalCount(send_.get(), &fatal), ncclSuccess);
  EXPECT_EQ(fatal, 0);
  EXPECT_FALSE(send_->base.faultQpError[1]);
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_FaultDelay_HoldsThePostForAtLeastTheDelay) {
  WireSend();
  constexpr uint32_t kDelayUs = 2000;
  ASSERT_EQ(ncclIbCastFaultSetQpDelay(send_.get(), 0, kDelayUs), ncclSuccess);
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  const auto start = std::chrono::steady_clock::now();
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::microseconds(kDelayUs));
  EXPECT_EQ(verbs_[0].sends.size(), 1u);
  ASSERT_EQ(ncclIbCastFaultClear(send_.get()), ncclSuccess);
  EXPECT_EQ(send_->base.faultQpDelayUs[0], 0u);
}

TEST_F(NetIbCastP2pIsendMicrotest, Isend_Telemetry_CountsPostsBytesAndWqeSizes) {
  WireSend();
  RcclQpStats stats{};
  send_->base.qps[0].telQpStats = &stats;
  EnableTelemetry();
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(100, 1, &req), ncclSuccess);
  EXPECT_EQ(stats.num_wqe_sent, 1u);
  EXPECT_EQ(stats.num_write_imm_wqe, 1u);
  EXPECT_EQ(rcclTelemetryDevs[0].tx_bytes, 100u);
  EXPECT_EQ(rcclTelemetryDevs[0].wqe_size_histogram[7], 1u);  // 100 bytes: bit length 7
  EXPECT_GT(Req(req)->tel_post_ts, 0u);

  send_->base.optRecvCompletion = true;
  Cts(0, 1, 1, 4096);
  req = (void*)NCCL_NET_OPTIONAL_RECV_COMPLETION;
  ASSERT_EQ(Isend(100, 1, &req), ncclSuccess);
  EXPECT_EQ(stats.num_write_wqe, 1u);
}

using NetIbCastP2pIrecvMicrotest = NetIbCastP2pMicrotest;

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_RejectsAnUnreadyCommTooManyRecvsAndAFatalComm) {
  WireRecv();
  recv_->base.ready = 0;
  void* req = buf_;
  CaptureLog([&] { EXPECT_EQ(Irecv(1, &req), ncclInternalError); });
  EXPECT_EQ(req, nullptr);
  recv_->base.ready = 1;
  EXPECT_EQ(Irecv(NCCL_NET_IB_MAX_RECVS + 1, &req), ncclInternalError);
  recv_->base.stats.fatalErrorCount = 1;
  CaptureLog([&] { EXPECT_EQ(Irecv(1, &req), ncclSystemError); });
  EXPECT_TRUE(verbs_[0].sends.empty());
  EXPECT_TRUE(verbs_[0].recvs.empty());
}

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_OneQp_PostsARecvAndASignalledCts) {
  WireRecv();
  recv_->base.reqs[0].type = NCCL_NET_IB_REQ_SEND;  // the request lands in slot 1
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req, 4096), ncclSuccess);
  ncclIbRequest* r = Req(req);
  ASSERT_EQ(r, &recv_->base.reqs[1]);
  EXPECT_EQ(r->type, NCCL_NET_IB_REQ_RECV);
  EXPECT_EQ(r->nreqs, 1);
  EXPECT_EQ(recv_->recvReqs[0], r);
  EXPECT_EQ(r->recv.cmplsRecords, &recv_->cmplsRecords[0]);
  ASSERT_EQ(verbs_[0].recvs.size(), 1u);
  EXPECT_EQ(verbs_[0].recvs[0].wrId, 0u);
  EXPECT_EQ(recv_->base.rxPosts[0], 1);

  const ncclIbSendFifo& e = recv_->remCtsFifo.elems[0][0];
  EXPECT_EQ(e.addr, (uint64_t)rdata_[0]);
  EXPECT_EQ(e.rkeys[0], kRkey);
  EXPECT_EQ(e.nreqs, 1u);
  EXPECT_EQ(e.size, 4096u);
  EXPECT_EQ(e.tag, 0x10u);
  EXPECT_EQ(e.idx, 1u);
  EXPECT_EQ(e.rxReqIndex, 1u);

  ASSERT_EQ(verbs_[0].sends.size(), 1u);
  const auto& cts = Chain(0, 0)[0];
  EXPECT_EQ(cts.wr.opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(cts.wr.send_flags, IBV_SEND_INLINE | IBV_SEND_SIGNALED);
  EXPECT_EQ(cts.wr.wr_id, 0u);
  EXPECT_EQ(cts.wr.wr.rdma.remote_addr, kCtsFifoAddr);
  EXPECT_EQ(cts.wr.wr.rdma.rkey, kCtsRkey);
  EXPECT_EQ(cts.sge.addr, (uint64_t)&recv_->remCtsFifo.elems[0][0]);
  EXPECT_EQ(cts.sge.length, sizeof(ncclIbSendFifo));
  EXPECT_EQ(cts.sge.lkey, kSgeLkey);
  // One data event plus the signalled CTS.
  EXPECT_EQ(r->events[0], 2);
  EXPECT_EQ(r->ctsEvents[0], 1);
  EXPECT_EQ(recv_->base.fifoHead, 1u);
}

// Only the slot matching the CTS QP's device is signalled; the others stay unsignalled and add no event.
TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_LaterSlot_PostsAnUnsignalledCtsAtItsFifoRow) {
  WireRecv();
  recv_->base.fifoHead = 3;
  void* req = nullptr;
  ASSERT_EQ(Irecv(2, &req), ncclSuccess);
  const auto& cts = Chain(0, 0)[0];
  EXPECT_EQ(cts.wr.send_flags, IBV_SEND_INLINE);
  EXPECT_EQ(cts.wr.wr.rdma.remote_addr, kCtsFifoAddr + 3 * NCCL_NET_IB_MAX_RECVS * sizeof(ncclIbSendFifo));
  EXPECT_EQ(cts.sge.length, 2 * sizeof(ncclIbSendFifo));
  EXPECT_EQ(Req(req)->events[0], 1);
  EXPECT_EQ(Req(req)->ctsEvents[0], 0);
  EXPECT_EQ(recv_->remCtsFifo.elems[3][1].tag, 0x11u);
  EXPECT_EQ(recv_->remCtsFifo.elems[3][1].idx, 4u);
}

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_TwoQpsTwoDevices_PostsARecvPerQpAndTheCtsOnTheIdsDevice) {
  WireRecv(2, 2);
  recv_->base.fifoHead = 1;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].recvs.size(), 1u);
  ASSERT_EQ(verbs_[1].recvs.size(), 1u);
  EXPECT_EQ(verbs_[1].recvs[0].wrId, 1u);
  EXPECT_EQ(verbs_[0].recvs[0].wrId, 0u);
  ASSERT_EQ(verbs_[1].sends.size(), 1u);
  EXPECT_TRUE(verbs_[0].sends.empty());
  const auto& cts = Chain(1, 0)[0];
  EXPECT_EQ(cts.wr.wr.rdma.rkey, kCtsRkey + 1);
  EXPECT_EQ(cts.sge.lkey, kSgeLkey + 1);
  EXPECT_EQ(cts.wr.send_flags, IBV_SEND_INLINE | IBV_SEND_SIGNALED);
  EXPECT_EQ(cts.wr.wr_id, 1u);
  EXPECT_EQ(recv_->remCtsFifo.elems[1][0].rkeys[1], kRkey + 1);
  EXPECT_EQ(Req(req)->events[0], 1);
  EXPECT_EQ(Req(req)->events[1], 2);
}

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_RecvQueueFull_PostsNoMoreRecvs) {
  WireRecv();
  recv_->base.rxPosts[0] = NET_IB_MAX_REQUESTS;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_TRUE(verbs_[0].recvs.empty());
  EXPECT_EQ(recv_->base.rxPosts[0], NET_IB_MAX_REQUESTS);
}

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_PrepostedOrOptionalCompletion_PostsNoRecv) {
  WireRecv();
  recv_->prepostReceiveWorkRequests = true;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_TRUE(verbs_[0].recvs.empty());
  EXPECT_EQ(Req(req)->events[0], 2);

  recv_->prepostReceiveWorkRequests = false;
  recv_->base.optRecvCompletion = true;
  req = (void*)NCCL_NET_OPTIONAL_RECV_COMPLETION;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_TRUE(verbs_[0].recvs.empty());
  EXPECT_EQ(Req(req)->events[0], 0);
  EXPECT_EQ(verbs_[0].sends.size(), 2u);
}

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_ByOrder_TagsRecvsAndTheDrainCtsWithTheRequestIndex) {
  WireRecv(1, 1, BY_ORDER);
  recv_->base.reqs[0].type = NCCL_NET_IB_REQ_SEND;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].recvs.size(), 1u);
  EXPECT_EQ(verbs_[0].recvs[0].wrId, 1u);
  EXPECT_EQ(recv_->base.rxPosts[0], 0);
  const auto& cts = Chain(0, 0)[0];
  EXPECT_EQ(cts.wr.send_flags, IBV_SEND_INLINE | IBV_SEND_SIGNALED);
  EXPECT_EQ(cts.wr.wr_id, 1u);
  EXPECT_EQ(Req(req)->events[0], 2);
  EXPECT_EQ(Req(req)->ctsEvents[0], 0);

  void* next = nullptr;
  ASSERT_EQ(Irecv(1, &next), ncclSuccess);
  EXPECT_EQ(Chain(0, 1)[0].wr.send_flags, IBV_SEND_INLINE);
  EXPECT_EQ(Req(next)->events[0], 1);
}

// A BY_ORDER sender never takes the scheduler path, even when it is globally on, so it sends over the default
// nDataQps QPs; the receiver must arm the same QPs or it waits for completions that never come.
TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_ByOrderWithSchedulerOn_ArmsTheSendersDefaultQps) {
  castGlobalQpSchedParms.enable = true;
  WireSend(4, 2, BY_ORDER);
  send_->base.splitDataOnQps = 0;
  Cts(0, 1, 1, 4096);
  void* sreq = nullptr;
  ASSERT_EQ(Isend(1024, 1, &sreq), ncclSuccess);
  EXPECT_EQ(Req(sreq)->desc.nqps, 2);
  for (int d = 0; d < 2; d++) verbs_[d].sends.clear();

  WireRecv(4, 2, BY_ORDER);
  recv_->base.splitDataOnQps = 0;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  ASSERT_EQ(verbs_[0].recvs.size(), 1u);
  ASSERT_EQ(verbs_[1].recvs.size(), 1u);
  EXPECT_EQ(verbs_[0].recvs[0].qp, &verbs_[0].qps[0]);
  EXPECT_EQ(verbs_[1].recvs[0].qp, &verbs_[1].qps[0]);

  // BY_INDEX comms do take the scheduler path, which spreads over every QP.
  recv_ = std::make_unique<ncclIbRecvComm>();
  for (int d = 0; d < 2; d++) verbs_[d].recvs.clear();
  WireRecv(4, 2, BY_INDEX);
  recv_->base.splitDataOnQps = 0;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_EQ(verbs_[0].recvs.size(), 2u);
  EXPECT_EQ(verbs_[1].recvs.size(), 2u);
}

// With CTS offload the CTS is a fixed inline-sized write on activeQps[id % nqps], signalled on that QP's own slot.
TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_CtsOffload_UsesTheQpsSlotAndOneRecv) {
  WireRecv(2, 1, BY_ORDER);
  recv_->useCtsOffload = true;
  recv_->base.qps[1].ctsQpSlot = 1;
  recv_->base.qps[0].ctsQpSlot = 0;
  recv_->base.fifoHead = 1;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_EQ(verbs_[0].recvs.size(), 1u);
  ASSERT_EQ(verbs_[0].sends.size(), 1u);
  EXPECT_EQ(verbs_[0].sends[0].qp, &verbs_[0].qps[1]);
  EXPECT_EQ(Chain(0, 0)[0].sge.length, (uint32_t)MAX_INLINE_DATA_SIZE);
  EXPECT_TRUE(Chain(0, 0)[0].wr.send_flags & IBV_SEND_SIGNALED);

  recv_->base.qps[0].ctsQpSlot = NCCL_CTS_QP_SLOT_INVALID;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_EQ(verbs_[0].sends[1].qp, &verbs_[0].qps[0]);
  EXPECT_FALSE(Chain(0, 1)[0].wr.send_flags & IBV_SEND_SIGNALED);
}

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_InlineCtsLayout_PacksThirtyTwoByteEntries) {
  WireRecv();
  IbCastAinicCtsInlineData = true;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req, 77), ncclSuccess);
  const auto* inl = reinterpret_cast<const ncclIbSendFifoCtsInline*>(recv_->remCtsFifo.elems[0]);
  EXPECT_EQ(inl[0].addr, (uint64_t)rdata_[0]);
  EXPECT_EQ(inl[0].rkeys[0], kRkey);
  EXPECT_EQ(inl[0].size, 77);
  EXPECT_EQ(inl[0].nreqs, 1);
  EXPECT_EQ(inl[0].tag, 0x10);
  EXPECT_EQ(inl[0].idx, 1u);
  EXPECT_EQ(inl[0].rxReqIndex, 0);
  EXPECT_EQ(Chain(0, 0)[0].sge.length, sizeof(ncclIbSendFifoCtsInline));
  EXPECT_TRUE(Chain(0, 0)[0].wr.send_flags & IBV_SEND_INLINE);

  // Two entries no longer fit the QP's inline limit, so the CTS goes out as a regular write.
  ASSERT_EQ(Irecv(2, &req), ncclSuccess);
  EXPECT_EQ(inl[NCCL_NET_IB_MAX_RECVS * 2 + 1].tag, 0x11);
  EXPECT_EQ(Chain(0, 1)[0].sge.length, 2 * sizeof(ncclIbSendFifoCtsInline));
  EXPECT_FALSE(Chain(0, 1)[0].wr.send_flags & IBV_SEND_INLINE);
}

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_ResiliencyOrSharing_SignalsEveryCts) {
  WireRecv();
  auto res = std::make_unique<ncclIbResiliency>();
  recv_->base.resiliency = res.get();
  recv_->base.fifoHead = 5;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_TRUE(Chain(0, 0)[0].wr.send_flags & IBV_SEND_SIGNALED);
  EXPECT_EQ(Chain(0, 0)[0].wr.wr_id, 5u);
  EXPECT_EQ(Req(req)->ctsEvents[0], 1);
  recv_->base.resiliency = nullptr;

  IbCastQpSharingGlobalEnable = true;
  recv_->base.qpSharing.netIbCommId = 12;
  recv_->base.fifoHead = NET_IB_MAX_REQUESTS;  // slot 0
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_EQ(Chain(0, 1)[0].wr.wr_id, IbCastEncodeCommId(0, 12));
}

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_CtsPostFails_ReleasesTheRequest) {
  WireRecv();
  verbs_[0].postSendErr = EAGAIN;
  void* req = buf_;
  ncclResult_t res = ncclSuccess;
  CaptureLog([&] { res = Irecv(1, &req); });
  EXPECT_EQ(res, ncclSystemError);
  EXPECT_EQ(req, buf_);
  EXPECT_EQ(recv_->base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(recv_->base.fifoHead, 0u);
}

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_RecvPostFailsOrNoFreeRequest_Fails) {
  WireRecv();
  verbs_[0].postRecvErr = ENOMEM;
  void* req = nullptr;
  CaptureLog([&] { EXPECT_EQ(Irecv(1, &req), ncclSystemError); });
  verbs_[0].postRecvErr = 0;
  for (auto& r : recv_->base.reqs) r.type = NCCL_NET_IB_REQ_SEND;
  CaptureLog([&] { EXPECT_EQ(Irecv(1, &req), ncclInternalError); });
  EXPECT_TRUE(verbs_[0].sends.empty());
}

TEST_F(NetIbCastP2pIrecvMicrotest, Irecv_Telemetry_CountsRecvAndCtsPosts) {
  WireRecv();
  RcclQpStats stats{};
  recv_->base.qps[0].telQpStats = &stats;
  EnableTelemetry();
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_EQ(stats.num_recv_wqe, 1u);
  EXPECT_EQ(stats.num_cts_sent_signalled, 1u);
  EXPECT_GT(Req(req)->tel_post_ts, 0u);
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_EQ(stats.num_cts_sent_unsignalled, 1u);
  recv_->base.recvMatchingScheme = BY_ORDER;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  EXPECT_EQ(stats.num_recv_wqe, 3u);
}

class NetIbCastP2pIflushMicrotest : public NetIbCastP2pMicrotest {
 protected:
  void SetUp() override {
    NetIbCastP2pMicrotest::SetUp();
    WireRecv(2, 2);
    recv_->flushEnabled = 1;
    for (int d = 0; d < 2; d++) recv_->devs[d].gpuFlush.sge.lkey = 0x900 + d;
    data_[0] = buf_;
    data_[1] = buf_ + 0x100;
    mhs_[0] = mhs_[1] = &mh_;
  }
  ncclResult_t Flush(void** req) { return IbCastIflush(recv_.get(), 2, data_, sizes_, mhs_, req); }
  void* data_[2]{};
  int sizes_[2] = {64, 0};
  void* mhs_[2]{};
  int gpuMem_ = 0;
  ibv_mr gpuMr_{};
};

TEST_F(NetIbCastP2pIflushMicrotest, Iflush_DisabledOrEmpty_IsANoOp) {
  void* req = buf_;
  recv_->flushEnabled = 0;
  EXPECT_EQ(Flush(&req), ncclSuccess);
  recv_->flushEnabled = 1;
  sizes_[0] = 0;
  EXPECT_EQ(Flush(&req), ncclSuccess);
  EXPECT_EQ(req, buf_);
  EXPECT_TRUE(verbs_[0].sends.empty() && verbs_[1].sends.empty());
}

// The last non-empty receive is read back once per device, since the data could have landed on either.
TEST_F(NetIbCastP2pIflushMicrotest, Iflush_ReadsTheLastNonEmptyBufferOnEveryDevice) {
  sizes_[1] = 8;
  recv_->base.reqs[0].type = NCCL_NET_IB_REQ_RECV;
  void* req = nullptr;
  ASSERT_EQ(Flush(&req), ncclSuccess);
  ncclIbRequest* r = Req(req);
  ASSERT_EQ(r, &recv_->base.reqs[1]);
  EXPECT_EQ(r->type, NCCL_NET_IB_REQ_FLUSH);
  for (int d = 0; d < 2; d++) {
    ASSERT_EQ(verbs_[d].sends.size(), 1u) << d;
    EXPECT_EQ(verbs_[d].sends[0].qp, recv_->devs[d].gpuFlush.qp.qp);
    const auto& wr = Chain(d, 0)[0];
    EXPECT_EQ(wr.wr.opcode, IBV_WR_RDMA_READ);
    EXPECT_EQ(wr.wr.send_flags, IBV_SEND_SIGNALED);
    EXPECT_EQ(wr.wr.wr_id, 1u + NCCL_IB_FLUSH_REQ_WR_ID_OFFSET);
    EXPECT_EQ(wr.wr.wr.rdma.remote_addr, (uint64_t)data_[1]);
    EXPECT_EQ(wr.wr.wr.rdma.rkey, kRkey + d);
    EXPECT_EQ(wr.sge.lkey, 0x900u + d);
    EXPECT_EQ(r->events[d], 1);
  }
}

// With a GPU scratchpad registered, each device first writes it (no completion) and then reads it back.
TEST_F(NetIbCastP2pIflushMicrotest, Iflush_GpuScratchpad_WritesThenReadsIt) {
  for (int d = 0; d < 2; d++) {
    recv_->devs[d].gpuFlush.gpuFlushGpuMem = &gpuMem_;
    recv_->devs[d].gpuFlush.gpuMr = &gpuMr_;
  }
  gpuMr_.rkey = 0xabc;
  void* req = nullptr;
  ASSERT_EQ(Flush(&req), ncclSuccess);
  for (int d = 0; d < 2; d++) {
    ASSERT_EQ(verbs_[d].sends.size(), 2u) << d;
    const auto& write = Chain(d, 0)[0];
    const auto& read = Chain(d, 1)[0];
    EXPECT_EQ(write.wr.opcode, IBV_WR_RDMA_WRITE);
    EXPECT_EQ(write.wr.send_flags, 0);
    EXPECT_EQ(write.wr.wr.rdma.remote_addr, (uint64_t)&gpuMem_);
    EXPECT_EQ(write.wr.wr.rdma.rkey, 0xabcu);
    EXPECT_EQ(write.wr.wr_id, read.wr.wr_id);
    EXPECT_EQ(read.wr.opcode, IBV_WR_RDMA_READ);
    EXPECT_EQ(read.wr.wr.rdma.remote_addr, (uint64_t)&gpuMem_);
    EXPECT_EQ(read.wr.wr.rdma.rkey, 0xabcu);
  }

  // RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 goes back to reading the user buffer.
  ScopedHook param(g_loadParam, [](const char* env, int64_t d) {
    return std::string(env) == "RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING" ? 0 : d;
  });
  ASSERT_EQ(Flush(&req), ncclSuccess);
  EXPECT_EQ(verbs_[0].sends.size(), 3u);
  EXPECT_EQ(Chain(0, 2)[0].wr.wr.rdma.remote_addr, (uint64_t)data_[0]);
}

TEST_F(NetIbCastP2pIflushMicrotest, Iflush_SharingAndFailures) {
  IbCastQpSharingGlobalEnable = true;
  recv_->base.qpSharing.netIbCommId = 5;
  void* req = nullptr;
  ASSERT_EQ(Flush(&req), ncclSuccess);
  EXPECT_EQ(Chain(0, 0)[0].wr.wr_id, IbCastEncodeCommId(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET, 5));

  verbs_[1].postSendErr = EIO;
  CaptureLog([&] { EXPECT_EQ(Flush(&req), ncclSystemError); });
  verbs_[1].postSendErr = 0;
  recv_->devs[0].gpuFlush.gpuFlushGpuMem = &gpuMem_;
  recv_->devs[0].gpuFlush.gpuMr = &gpuMr_;
  verbs_[0].postSendErr = EIO;
  CaptureLog([&] { EXPECT_EQ(Flush(&req), ncclSystemError); });
  verbs_[0].postSendErr = 0;
  for (auto& r : recv_->base.reqs) r.type = NCCL_NET_IB_REQ_RECV;
  CaptureLog([&] { EXPECT_EQ(Flush(&req), ncclInternalError); });
}

using NetIbCastP2pTestMicrotest = NetIbCastP2pMicrotest;

TEST_F(NetIbCastP2pTestMicrotest, Test_SendRoundTrip_CompletesAndFreesTheSlot) {
  WireSend();
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(100, 1, &req), ncclSuccess);
  int done = -1;
  int sizes[1] = {-1};
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(verbs_[0].pollEntries, (std::vector<int>{kPollBatch}));

  verbs_[0].Complete((uint64_t)&send_->base.remapWrId[0], IBV_WC_RDMA_WRITE, verbs_[0].qps[0].qp_num);
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 100);
  EXPECT_EQ(send_->sendReqsCnt[0], 0);
  EXPECT_EQ(send_->sendReqs[0][0], nullptr);
  EXPECT_EQ(send_->base.remapWrId[0].state, NCCL_NET_IB_REMAP_UNUSED);
  EXPECT_EQ(Req(req)->type, NCCL_NET_IB_REQ_UNUSED);
}

// A multi-recv slot is released only when its last send completes.
TEST_F(NetIbCastP2pTestMicrotest, Test_MultiRecvSends_ReleaseTheSlotWithTheLastOne) {
  WireSend(1, 1, BY_ID);
  send_->base.fifoHead = 2;
  Cts(0, 2, 0x10, 4096);
  Cts(1, 2, 0x11, 4096);
  void* r0 = nullptr;
  void* r1 = nullptr;
  ASSERT_EQ(Isend(10, 0x10, &r0), ncclSuccess);
  ASSERT_EQ(Isend(20, 0x11, &r1), ncclSuccess);
  verbs_[0].Complete(0x0202, IBV_WC_RDMA_WRITE, verbs_[0].qps[0].qp_num);
  int done = 0;
  int sizes[1] = {0};
  ASSERT_EQ(Test(r0, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 10);
  EXPECT_EQ(send_->sendReqsCnt[2], 1);
  EXPECT_EQ(send_->sendReqs[2][1], Req(r1));
  ASSERT_EQ(Test(r1, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(send_->sendReqs[2][1], nullptr);
}

TEST_F(NetIbCastP2pTestMicrotest, Test_SendWithSchedulerTiming_FeedsTheRttStats) {
  WireSend(2);
  ncclIbNetCommBase& base = send_->base;
  base.schedParmsInit = true;
  base.schedParms.enable = true;
  base.schedParms.splitData = true;
  base.schedParms.splitDataMin = 1;
  base.schedParms.updateInterval = 1000000000;
  IbCastDevs[0].speed = 100000;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(1000, 1, &req), ncclSuccess);
  for (int i = 0; i < 2; i++) verbs_[0].Complete((uint64_t)&base.remapWrId[i], IBV_WC_RDMA_WRITE, 0);
  int done = 0;
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(base.qpTxStats[0].numMeasurements, 1u);
  EXPECT_EQ(base.qpTxStats[1].numMeasurements, 1u);
}

TEST_F(NetIbCastP2pTestMicrotest, Test_RecvByIndexSingleQp_WaitsForTheCtsThenReportsTheSize) {
  WireRecv();
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  const uint16_t idx = (uint16_t)(Req(req) - recv_->base.reqs);
  ibv_wc& data = verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, verbs_[0].qps[0].qp_num);
  data.imm_data = htobe32((uint32_t)idx << WR_IMM_RX_REQ_IDX_BIT_POS);
  data.byte_len = 777;
  int done = 0;
  int sizes[1] = {0};
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(recv_->base.rxPosts[0], 0);
  EXPECT_EQ(Req(req)->events[0], 1);
  verbs_[0].Complete(0, IBV_WC_RDMA_WRITE, verbs_[0].qps[0].qp_num);
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 777);
}

TEST_F(NetIbCastP2pTestMicrotest, Test_RecvByIndexSplit_CountsEveryQpsImmediate) {
  WireRecv(2);
  recv_->base.fifoHead = 1;  // slot 1: unsignalled CTS, so only data events
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  const uint32_t idx = (uint32_t)(Req(req) - recv_->base.reqs);
  for (int q = 0; q < 2; q++) {
    ibv_wc& wc = verbs_[0].Complete(q, IBV_WC_RECV_RDMA_WITH_IMM, verbs_[0].qps[q].qp_num);
    wc.imm_data = htobe32((idx << WR_IMM_RX_REQ_IDX_BIT_POS) | WR_IMM_SPLIT_DATA_FLAG);
    wc.byte_len = 100 + q;
    int done = 0;
    int sizes[1] = {0};
    ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
    EXPECT_EQ(done, q == 1) << q;
    if (done) EXPECT_EQ(sizes[0], 201);
  }
}

// BY_ID: the request comes from recvReqs[imm]; without a size record the bytes add up in aggSize.
TEST_F(NetIbCastP2pTestMicrotest, Test_RecvById_AggregatesBytesWithoutASizeRecord) {
  WireRecv(1, 1, BY_ID);
  recv_->base.fifoHead = 3;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  ibv_wc& wc = verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, verbs_[0].qps[0].qp_num);
  wc.imm_data = htobe32(3);
  wc.byte_len = 64;
  int done = 0;
  int sizes[1] = {0};
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 64);

  // A size record written by the sender wins, and the immediate's byte count is not added on top.
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  recv_->cmplsRecords[4].sizes[0] = 1234;
  ibv_wc& wc2 = verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, verbs_[0].qps[0].qp_num);
  wc2.imm_data = htobe32(4);
  wc2.byte_len = 64;
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(sizes[0], 1234);
  EXPECT_EQ(Req(req)->recv.aggSize, 0);
}

TEST_F(NetIbCastP2pTestMicrotest, Test_RecvMulti_ReportsEveryRecordedSize) {
  WireRecv(1, 1, BY_ID);
  recv_->base.fifoHead = 1;
  void* req = nullptr;
  ASSERT_EQ(Irecv(3, &req), ncclSuccess);
  recv_->cmplsRecords[1].sizes[0] = 5;
  recv_->cmplsRecords[1].sizes[1] = 6;
  recv_->cmplsRecords[1].sizes[2] = 7;
  ibv_wc& wc = verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, verbs_[0].qps[0].qp_num);
  wc.imm_data = htobe32(1);
  wc.byte_len = 999;
  int done = 0;
  int sizes[3] = {};
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 5);
  EXPECT_EQ(sizes[1], 6);
  EXPECT_EQ(sizes[2], 7);
}

// Preposted receives are replenished on the QP that consumed one, and the QP's delivery is recorded.
TEST_F(NetIbCastP2pTestMicrotest, Test_PrepostedRecv_RepostsOnTheSameQp) {
  WireRecv(2, 1, BY_ID);
  recv_->prepostReceiveWorkRequests = true;
  recv_->base.fifoHead = 1;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  ibv_wc& wc = verbs_[0].Complete(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_RECV_RDMA_WITH_IMM, verbs_[0].qps[1].qp_num);
  wc.imm_data = htobe32(1);
  int done = 0;
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  ASSERT_EQ(verbs_[0].recvs.size(), 1u);
  EXPECT_EQ(verbs_[0].recvs[0].qp, &verbs_[0].qps[1]);
  EXPECT_EQ(verbs_[0].recvs[0].wrId, NCCL_IB_RECV_WR_ID_DUMMY);
  EXPECT_TRUE(recv_->cmplsRecords[1].completions[1]);
  EXPECT_FALSE(recv_->cmplsRecords[1].completions[0]);

  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  ibv_wc& stray = verbs_[0].Complete(NCCL_IB_RECV_WR_ID_DUMMY, IBV_WC_RECV_RDMA_WITH_IMM, 0xbeef);
  stray.imm_data = htobe32(2);
  EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError);
}

TEST_F(NetIbCastP2pTestMicrotest, Test_FlushCompletion_FinishesTheFlush) {
  WireRecv();
  recv_->flushEnabled = 1;
  void* data[1] = {buf_};
  int sizes[1] = {8};
  void* mhs[1] = {&mh_};
  void* req = nullptr;
  ASSERT_EQ(IbCastIflush(recv_.get(), 1, data, sizes, mhs, &req), ncclSuccess);
  verbs_[0].Complete(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET, IBV_WC_RDMA_READ, 0);
  int done = 0;
  int out[1] = {-1};
  ASSERT_EQ(Test(req, &done, out), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(out[0], -1);

  ASSERT_EQ(IbCastIflush(recv_.get(), 1, data, sizes, mhs, &req), ncclSuccess);
  verbs_[0].Complete(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + NET_IB_MAX_REQUESTS, IBV_WC_RDMA_READ, 0);
  CaptureLog([&] { EXPECT_EQ(Test(req, &done, out), ncclInternalError); });
}

TEST_F(NetIbCastP2pTestMicrotest, Test_ErrorCompletionWithoutResiliency_FailsWithThePeerAndGids) {
  WireRecv();
  recv_->devs[0].base.gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
  inet_pton(AF_INET6, "fe80::1", recv_->devs[0].base.gidInfo.localGid.raw);
  inet_pton(AF_INET6, "fe80::2", recv_->base.remDevs[0].remoteGid.raw);
  int peerLookups = 0;
  ScopedHook toString(g_ncclSocketToString, [&](const union ncclSocketAddress*, char* buf, size_t len, int) {
    peerLookups++;
    snprintf(buf, len, "%s", kPeer);
    return buf;
  });
  EnableTelemetry();
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, 0, IBV_WC_RETRY_EXC_ERR);
  int done = 0;
  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&] { res = Test(req, &done, nullptr); });
  EXPECT_EQ(res, ncclRemoteError);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(rcclTelemetryDevs[0].num_cq_errors, 1u);
  EXPECT_EQ(peerLookups, 1);
  EXPECT_TRUE(LogHas(log, "fe80::2")) << log;
}

TEST_F(NetIbCastP2pTestMicrotest, Test_ErrorCompletionOnInfiniband_LogsNoGids) {
  WireRecv();
  EnableTelemetry();
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, 0, IBV_WC_WR_FLUSH_ERR);
  int done = 0;
  const std::string log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclRemoteError); });
  EXPECT_EQ(done, 0);
  EXPECT_EQ(rcclTelemetryDevs[0].num_cq_errors, 1u);
  EXPECT_FALSE(LogHas(log, "localGid")) << log;
}

// With resiliency every device is polled, errors go to the resiliency layer, and its progress runs first.
TEST_F(NetIbCastP2pTestMicrotest, Test_Resiliency_HandsErrorsOverAndDrivesProgress) {
  WireRecv(2, 2);
  auto res = std::make_unique<ncclIbResiliency>();
  recv_->base.resiliency = res.get();
  recv_->base.fifoHead = 1;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  Req(req)->events[0] = 0;
  res->inProgress = true;
  ScopedHook progress(g_ibCastResiliencyProgress, [&](ncclIbResiliency* ctx) {
    EXPECT_EQ(ctx, res.get());
    return ncclSuccess;
  });
  ScopedHook isComplete(g_ibCastResiliencyRequestIsComplete, [](ncclIbRequest*, bool* c) {
    *c = false;
    return ncclSuccess;
  });
  std::vector<int> errorDevs;
  ScopedHook handle(g_ibCastResiliencyHandleCompletionError, [&](ncclIbResiliency*, ibv_wc* wc, int dev) {
    EXPECT_EQ(wc->status, IBV_WC_WR_FLUSH_ERR);
    errorDevs.push_back(dev);
    return ncclSuccess;
  });
  verbs_[0].Complete(0, IBV_WC_SEND, 0, IBV_WC_WR_FLUSH_ERR);
  int done = 0;
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(progress.calls, 1);
  EXPECT_EQ(errorDevs, (std::vector<int>{0}));
  EXPECT_FALSE(verbs_[0].pollEntries.empty());
  EXPECT_FALSE(verbs_[1].pollEntries.empty());

  ScopedHook done2(g_ibCastResiliencyRequestIsComplete, [](ncclIbRequest*, bool* c) {
    *c = true;
    return ncclSuccess;
  });
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);
}

// Pins AICOMRCCL-3078; a fix flips this pin.
TEST_F(NetIbCastP2pTestMicrotest, RequestIsComplete_KnownBug_ResiliencyErrorReadsAsComplete) {
  WireRecv();
  auto res = std::make_unique<ncclIbResiliency>();
  recv_->base.resiliency = res.get();
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  ScopedHook isComplete(g_ibCastResiliencyRequestIsComplete, [](ncclIbRequest*, bool*) { return ncclInternalError; });
  int done = 0;
  EXPECT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(Req(req)->type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbCastP2pTestMicrotest, Test_PollingDetails) {
  WireRecv(2, 2);
  recv_->base.fifoHead = 1;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  Req(req)->events[0] = 0;
  int done = 0;
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_TRUE(verbs_[0].pollEntries.empty());  // nothing expected there
  EXPECT_EQ(verbs_[1].pollEntries, (std::vector<int>{kPollBatch}));

  IbCastAinicRoce = true;
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(verbs_[1].pollEntries.back(), NCCL_CQ_POLL_MAX_EVENT);

  verbs_[1].pollErr = -3;
  CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclSystemError); });
  verbs_[1].pollErr = 0;

  recv_->base.stats.fatalErrorCount = 1;
  CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclSystemError); });
  recv_->base.stats.fatalErrorCount = 0;

  // A device-wide fatal error surfaces after that device has been polled.
  IbCastDevs[1].stats.fatalErrorCount = 1;
  verbs_[1].Complete(1, IBV_WC_RDMA_WRITE, verbs_[1].qps[0].qp_num);
  CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclSystemError); });
  EXPECT_EQ(Req(req)->events[1], 1);
}

TEST_F(NetIbCastP2pTestMicrotest, Test_SenderGetsTheWrongRequestType_Fails) {
  WireSend(1, 1, BY_ID);
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  Req(req)->type = NCCL_NET_IB_REQ_FLUSH;
  verbs_[0].Complete(0, IBV_WC_RDMA_WRITE, 0);
  int done = 0;
  std::string log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "Sender expected a 'send' request but got 'Flush'")) << log;
}

TEST_F(NetIbCastP2pTestMicrotest, Test_SenderEventUnderflow_FailsUnlessResilient) {
  WireSend(1, 1, BY_ID);
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  verbs_[0].Complete(0, IBV_WC_RDMA_WRITE, 0);
  verbs_[0].Complete(0, IBV_WC_RDMA_WRITE, 0);
  Req(req)->events[0] = 2;  // keep it open while both completions arrive
  int done = 0;
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);

  // Already over-drained: non-zero, so the device is still polled, but one more completion is an underflow.
  Cts(0, 1, 1, 4096);
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  Req(req)->events[0] = -1;
  verbs_[0].Complete(1, IBV_WC_RDMA_WRITE, 0);
  std::string log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "devIndex=0, reqIdx=0 <= 0")) << log;

  auto res = std::make_unique<ncclIbResiliency>();
  send_->base.resiliency = res.get();
  verbs_[0].Complete(1, IBV_WC_RDMA_WRITE, 0);
  ScopedHook notDone(g_ibCastResiliencyRequestIsComplete, [](ncclIbRequest*, bool* c) {
    *c = false;
    return ncclSuccess;
  });
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(Req(req)->events[0], -2);
}

TEST_F(NetIbCastP2pTestMicrotest, Test_CompletionForAnEmptySlot_Fails) {
  WireSend(1, 1, BY_ID);
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  verbs_[0].Complete(9, IBV_WC_RDMA_WRITE, 0);
  int done = 0;
  std::string log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "could not retreive a request")) << log;
}

TEST_F(NetIbCastP2pTestMicrotest, Test_ReceiverRequestStates) {
  WireRecv(1, 1, BY_ID);
  recv_->base.fifoHead = 1;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  int done = 0;

  // Data for a slot whose request was already retired: an error without resiliency, ignored with it.
  ibv_wc& wc = verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, 0);
  wc.imm_data = htobe32(7);
  recv_->recvReqs[7] = &recv_->base.reqs[50];
  std::string log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "Receiver expected a 'recv' request but got 'Unused'")) << log;

  auto res = std::make_unique<ncclIbResiliency>();
  recv_->base.resiliency = res.get();
  ScopedHook notDone(g_ibCastResiliencyRequestIsComplete, [](ncclIbRequest*, bool* c) {
    *c = false;
    return ncclSuccess;
  });
  ibv_wc& again = verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, 0);
  again.imm_data = htobe32(7);
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(recv_->base.reqs[50].events[0], 0);

  // A CTS completion for a retired request still drains its counters, then is ignored.
  verbs_[0].Complete(7, IBV_WC_RDMA_WRITE, 0);
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(recv_->base.reqs[50].events[0], -1);
  EXPECT_EQ(recv_->base.reqs[50].ctsEvents[0], -1);

  verbs_[0].Complete(1, IBV_WC_SEND, 0);
  log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "Unknown completion")) << log;
}

TEST_F(NetIbCastP2pTestMicrotest, Test_RecvWithSharing_UsesTheEncodedSlot) {
  WireRecv(1, 1, BY_ID);
  IbCastQpSharingGlobalEnable = true;
  recv_->base.fifoHead = 2;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  ibv_wc& wc = verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, 0);
  wc.imm_data = htobe32(IbCastEncodeCommIdImmData(2, 0x77));
  wc.byte_len = 8;
  int done = 0;
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);
}

// On a shared QP the polling comm may drain another comm's completions; wr_id[63:48] or the immediate routes them.
TEST_F(NetIbCastP2pTestMicrotest, Test_SharedQp_RoutesCompletionsToTheOwningComm) {
  WireSend(1, 1, BY_ID);
  IbCastQpSharingGlobalEnable = true;
  send_->base.qpSharing.netIbCommId = 1;
  auto other = std::make_unique<ncclIbSendComm>();
  other->base.isSend = true;
  other->base.recvMatchingScheme = BY_ID;
  other->base.qpSharing.netIbCommId = 2;
  g_IbCastCommTable[2] = {other.get(), true, true};
  ncclIbRequest& otherReq = other->base.reqs[0];
  otherReq.base = &other->base;
  otherReq.type = NCCL_NET_IB_REQ_SEND;
  otherReq.nreqs = 1;
  otherReq.id = 4;
  otherReq.events[0] = 1;
  other->sendReqs[4][0] = &otherReq;
  other->sendReqsCnt[4] = 1;

  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  verbs_[0].Complete(IbCastEncodeCommId(4, 2), IBV_WC_RDMA_WRITE, 0);
  int done = 0;
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(otherReq.events[0], 0);
  EXPECT_EQ(Req(req)->events[0], 1);

  // Receive side: the immediate's commId routes BY_ID data to the owning recv comm.
  WireRecv(1, 1, BY_ID);
  recv_->base.qpSharing.netIbCommId = 3;
  auto otherRecv = std::make_unique<ncclIbRecvComm>();
  otherRecv->base.recvMatchingScheme = BY_ID;
  g_IbCastCommTable[5] = {otherRecv.get(), false, true};
  ncclIbRequest& rreq = otherRecv->base.reqs[0];
  rreq.base = &otherRecv->base;
  rreq.type = NCCL_NET_IB_REQ_RECV;
  rreq.nreqs = 1;
  rreq.events[0] = 2;
  rreq.recv.cmplsRecords = &otherRecv->cmplsRecords[0];
  otherRecv->recvReqs[6] = &rreq;
  void* mine = nullptr;
  ASSERT_EQ(Irecv(1, &mine), ncclSuccess);
  ibv_wc& wc = verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, 0);
  wc.imm_data = htobe32(IbCastEncodeCommIdImmData(6, 5));
  wc.byte_len = 32;
  ASSERT_EQ(Test(mine, &done, nullptr), ncclSuccess);
  EXPECT_EQ(rreq.events[0], 1);
  EXPECT_EQ(rreq.recv.aggSize, 32);
  EXPECT_EQ(otherRecv->base.rxPosts[0], -1);
}

TEST_F(NetIbCastP2pTestMicrotest, Test_Telemetry_RecordsCompletionsAgainstTheQp) {
  WireSend(1, 1, BY_ID);
  RcclQpStats stats{};
  RcclChannelStats ch{};
  send_->base.qps[0].telQpStats = &stats;
  send_->telChStats = &ch;
  EnableTelemetry();
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  verbs_[0].Complete(0, IBV_WC_RDMA_WRITE, verbs_[0].qps[0].qp_num);
  int done = 0;
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(stats.num_wqe_rcvd, 1u);
  EXPECT_EQ(ch.num_req_completed, 1u);
  EXPECT_EQ(send_->base.telCqPollCount, 1u);

  // A completion whose QP is unknown is not charged to any QP.
  Cts(0, 1, 1, 4096);
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  verbs_[0].Complete(1, IBV_WC_RDMA_WRITE, 0xdead);
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(stats.num_wqe_rcvd, 1u);
}

TEST_F(NetIbCastP2pTestMicrotest, Test_RecvTelemetry_CountsReceivedBytes) {
  WireRecv(1, 1, BY_ID);
  RcclChannelStats ch{};
  recv_->telChStats = &ch;
  EnableTelemetry();
  recv_->base.fifoHead = 1;
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  ibv_wc& wc = verbs_[0].Complete(0, IBV_WC_RECV_RDMA_WITH_IMM, verbs_[0].qps[0].qp_num);
  wc.imm_data = htobe32(1);
  wc.byte_len = 48;
  int done = 0;
  int sizes[1] = {};
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(rcclTelemetryDevs[0].rx_bytes, 48u);
  EXPECT_EQ(ch.num_req_completed, 1u);
}

using NetIbCastP2pByOrderMicrotest = NetIbCastP2pMicrotest;

TEST_F(NetIbCastP2pByOrderMicrotest, ByOrder_SendCompletion_MatchesTheSlot) {
  WireSend(1, 1, BY_ORDER);
  send_->base.fifoHead = 3;
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  EXPECT_EQ(Chain(0, 0)[0].wr.wr_id, 3u);
  verbs_[0].Complete(3, IBV_WC_RDMA_WRITE, verbs_[0].qps[0].qp_num);
  int done = 0;
  int sizes[1] = {};
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 64);
}

TEST_F(NetIbCastP2pByOrderMicrotest, ByOrder_SendFailures) {
  WireSend(1, 1, BY_ORDER);
  Cts(0, 1, 1, 4096);
  void* req = nullptr;
  ASSERT_EQ(Isend(64, 1, &req), ncclSuccess);
  int done = 0;
  verbs_[0].Complete(8, IBV_WC_RDMA_WRITE, 0);
  std::string log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "could not retreive a request")) << log;

  Req(req)->type = NCCL_NET_IB_REQ_RECV;
  verbs_[0].Complete(0, IBV_WC_RDMA_WRITE, 0);
  log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "Sender expected a 'send' request but got 'Recv'")) << log;

  Req(req)->type = NCCL_NET_IB_REQ_SEND;
  Req(req)->events[0] = -1;
  verbs_[0].Complete(0, IBV_WC_RDMA_WRITE, 0);
  log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "devIndex=0, reqIdx=0 <= 0")) << log;

  auto res = std::make_unique<ncclIbResiliency>();
  send_->base.resiliency = res.get();
  ScopedHook notDone(g_ibCastResiliencyRequestIsComplete, [](ncclIbRequest*, bool* c) {
    *c = false;
    return ncclSuccess;
  });
  verbs_[0].Complete(0, IBV_WC_RDMA_WRITE, 0);
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(Req(req)->events[0], -2);
}

TEST_F(NetIbCastP2pByOrderMicrotest, ByOrder_RecvDataAndCts_CompleteTheRequest) {
  WireRecv(1, 1, BY_ORDER);
  void* req = nullptr;
  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  const uint64_t idx = Req(req) - recv_->base.reqs;
  ibv_wc& wc = verbs_[0].Complete(idx, IBV_WC_RECV_RDMA_WITH_IMM, verbs_[0].qps[0].qp_num);
  wc.byte_len = 40;
  int done = 0;
  int sizes[1] = {};
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 0);
  verbs_[0].Complete(idx, IBV_WC_RDMA_WRITE, verbs_[0].qps[0].qp_num);
  ASSERT_EQ(Test(req, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 40);
}

TEST_F(NetIbCastP2pByOrderMicrotest, ByOrder_RecvFlushAndFailures) {
  WireRecv(1, 1, BY_ORDER);
  recv_->flushEnabled = 1;
  void* data[1] = {buf_};
  int fsizes[1] = {8};
  void* mhs[1] = {&mh_};
  void* req = nullptr;
  ASSERT_EQ(IbCastIflush(recv_.get(), 1, data, fsizes, mhs, &req), ncclSuccess);
  verbs_[0].Complete(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET, IBV_WC_RDMA_READ, 0);
  int done = 0;
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);

  ASSERT_EQ(Irecv(1, &req), ncclSuccess);
  const uint64_t idx = Req(req) - recv_->base.reqs;
  verbs_[0].Complete(NET_IB_MAX_REQUESTS, IBV_WC_RECV_RDMA_WITH_IMM, 0);
  std::string log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "Invalid request index")) << log;

  verbs_[0].Complete(idx, IBV_WC_SEND, 0);
  log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "Unknown completion")) << log;

  recv_->base.reqs[100].type = NCCL_NET_IB_REQ_UNUSED;
  verbs_[0].Complete(100, IBV_WC_RECV_RDMA_WITH_IMM, 0);
  log = CaptureLog([&] { EXPECT_EQ(Test(req, &done, nullptr), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "Receiver expected a 'recv' request but got 'Unused'")) << log;

  // Resiliency tolerates a stale data completion: it only drains the counter.
  auto res = std::make_unique<ncclIbResiliency>();
  recv_->base.resiliency = res.get();
  ScopedHook notDone(g_ibCastResiliencyRequestIsComplete, [](ncclIbRequest*, bool* c) {
    *c = false;
    return ncclSuccess;
  });
  recv_->base.reqs[100].nreqs = 2;
  verbs_[0].Complete(100, IBV_WC_RECV_RDMA_WITH_IMM, 0);
  ASSERT_EQ(Test(req, &done, nullptr), ncclSuccess);
  EXPECT_EQ(recv_->base.reqs[100].events[0], -1);
}

TEST_F(NetIbCastP2pByOrderMicrotest, ByOrder_WrongScheme_IsRejected) {
  WireRecv();
  ibv_wc wc{};
  wc.opcode = IBV_WC_RDMA_WRITE;
  const std::string log =
      CaptureLog([&] { EXPECT_EQ(IbCastCompletionEventByOrder(&recv_->base, &wc, 0), ncclInternalError); });
  EXPECT_TRUE(LogHas(log, "recvMatchingScheme not supported")) << log;
}

class NetIbCastP2pFaultMicrotest : public NetIbCastP2pMicrotest {
 protected:
  void TearDown() override {
    for (auto& v : verbs_) ncclIbOpsFaultRemove(&v.ctx);
    ibvSymbols.ibv_internal_modify_qp = savedModifyQp_;
    NetIbCastP2pMicrotest::TearDown();
  }
  decltype(ibvSymbols.ibv_internal_modify_qp) savedModifyQp_ = ibvSymbols.ibv_internal_modify_qp;
};

TEST_F(NetIbCastP2pFaultMicrotest, QpKnobs_ValidateTheCommAndIndex) {
  WireSend(2);
  int out = 0;
  bool fatal = false;
  EXPECT_EQ(ncclIbCastFaultSetQpDelay(nullptr, 0, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultSetQpError(nullptr, 0, true), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultClear(nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultGetFatalCount(nullptr, &out), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultGetFatalCount(send_.get(), nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultDriveQpToError(nullptr, 0), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultDriveRecvQpToError(nullptr, 0), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultCheckErrorFatal(nullptr, 0, &fatal), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultCheckErrorFatal(send_.get(), 0, nullptr), ncclInvalidArgument);
  for (int bad : {-1, 2}) {
    EXPECT_EQ(ncclIbCastFaultSetQpDelay(send_.get(), bad, 1), ncclInvalidArgument) << bad;
    EXPECT_EQ(ncclIbCastFaultSetQpError(send_.get(), bad, true), ncclInvalidArgument) << bad;
    EXPECT_EQ(ncclIbCastFaultDriveQpToError(send_.get(), bad), ncclInvalidArgument) << bad;
  }
  EXPECT_EQ(ncclIbCastFaultSetQpDelay(send_.get(), 1, 7), ncclSuccess);
  EXPECT_EQ(send_->base.faultQpDelayUs[1], 7u);
  EXPECT_EQ(ncclIbCastFaultSetQpError(send_.get(), 1, true), ncclSuccess);
  EXPECT_TRUE(send_->base.faultQpError[1]);
}

TEST_F(NetIbCastP2pFaultMicrotest, DriveToError_MovesTheActiveQpToErr) {
  WireSend(2);
  WireRecv(2);
  std::vector<ibv_qp*> modified;
  static std::vector<ibv_qp*>* sink;
  sink = &modified;
  ibvSymbols.ibv_internal_modify_qp = [](ibv_qp* qp, ibv_qp_attr* attr, int mask) {
    EXPECT_EQ(attr->qp_state, IBV_QPS_ERR);
    EXPECT_EQ(mask, IBV_QP_STATE);
    sink->push_back(qp);
    return 0;
  };
  ASSERT_EQ(ncclIbCastFaultDriveQpToError(send_.get(), 1), ncclSuccess);
  ASSERT_EQ(ncclIbCastFaultDriveRecvQpToError(recv_.get(), 0), ncclSuccess);
  EXPECT_EQ(ncclIbCastFaultDriveRecvQpToError(recv_.get(), 2), ncclInvalidArgument);
  EXPECT_EQ(modified, (std::vector<ibv_qp*>{&verbs_[0].qps[1], &verbs_[0].qps[0]}));
}

TEST_F(NetIbCastP2pFaultMicrotest, CheckErrorFatal_OnlyFlushAndRetryExceededAreRecoverable) {
  WireSend();
  bool fatal = false;
  EXPECT_EQ(ncclIbCastFaultCheckErrorFatal(send_.get(), IBV_WC_WR_FLUSH_ERR, &fatal), ncclInvalidArgument);
  auto res = std::make_unique<ncclIbResiliency>();
  send_->base.resiliency = res.get();
  ASSERT_EQ(ncclIbCastFaultCheckErrorFatal(send_.get(), IBV_WC_WR_FLUSH_ERR, &fatal), ncclSuccess);
  EXPECT_FALSE(fatal);
  ASSERT_EQ(ncclIbCastFaultCheckErrorFatal(send_.get(), IBV_WC_RETRY_EXC_ERR, &fatal), ncclSuccess);
  EXPECT_FALSE(fatal);
  ASSERT_EQ(ncclIbCastFaultCheckErrorFatal(send_.get(), IBV_WC_REM_ACCESS_ERR, &fatal), ncclSuccess);
  EXPECT_TRUE(fatal);
}

// The ops bridges translate (comm, qpIdx) into the QP's context and qp_num and arm the real ops-fault shims.
TEST_F(NetIbCastP2pFaultMicrotest, OpsBridges_ArmTheShimsOnTheQpsContext) {
  WireSend(2, 2);
  WireRecv(2, 2);
  ASSERT_EQ(ncclIbOpsFaultInstall(&verbs_[0].ctx), ncclSuccess);
  ASSERT_EQ(ncclIbOpsFaultInstall(&verbs_[1].ctx), ncclSuccess);
  ASSERT_EQ(ncclIbCastFaultOpsSetPostSendError(send_.get(), 1, EAGAIN), ncclSuccess);
  ASSERT_EQ(ncclIbCastFaultOpsSetPostRecvError(recv_.get(), 0, ENOMEM), ncclSuccess);
  ASSERT_EQ(ncclIbCastFaultOpsSetPollCqError(recv_.get(), 0, IBV_WC_GENERAL_ERR, 1, true), ncclSuccess);

  ibv_send_wr wr{};
  ibv_send_wr* bad = nullptr;
  ibv_recv_wr rwr{};
  ibv_recv_wr* rbad = nullptr;
  EXPECT_EQ(verbs_[1].ctx.ops.post_send(&verbs_[1].qps[0], &wr, &bad), EAGAIN);
  EXPECT_EQ(verbs_[0].ctx.ops.post_send(&verbs_[0].qps[0], &wr, &bad), 0);
  EXPECT_EQ(verbs_[0].ctx.ops.post_recv(&verbs_[0].qps[0], &rwr, &rbad), ENOMEM);
  ibv_wc wc{};
  ASSERT_EQ(verbs_[0].ctx.ops.poll_cq(&verbs_[0].cq, 1, &wc), 1);
  EXPECT_EQ(wc.qp_num, verbs_[0].qps[0].qp_num);

  ASSERT_EQ(ncclIbCastFaultOpsClear(send_.get()), ncclSuccess);
  EXPECT_EQ(verbs_[1].ctx.ops.post_send(&verbs_[1].qps[0], &wr, &bad), 0);
  EXPECT_EQ(verbs_[0].ctx.ops.post_recv(&verbs_[0].qps[0], &rwr, &rbad), 0);
}

TEST_F(NetIbCastP2pFaultMicrotest, OpsBridges_RejectBadArgumentsAndTornDownQps) {
  WireSend(2);
  WireRecv(2);
  EXPECT_EQ(ncclIbCastFaultOpsSetPostSendError(nullptr, 0, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultOpsSetPostRecvError(nullptr, 0, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultOpsSetPollCqError(nullptr, 0, 1, 1, false), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultOpsClear(nullptr), ncclInvalidArgument);
  for (int bad : {-1, 2}) {
    EXPECT_EQ(ncclIbCastFaultOpsSetPostSendError(send_.get(), bad, 1), ncclInvalidArgument) << bad;
    EXPECT_EQ(ncclIbCastFaultOpsSetPostRecvError(recv_.get(), bad, 1), ncclInvalidArgument) << bad;
    EXPECT_EQ(ncclIbCastFaultOpsSetPollCqError(recv_.get(), bad, 1, 1, false), ncclInvalidArgument) << bad;
  }
  // Not installed on the context: the shim registry rejects the arm.
  EXPECT_EQ(ncclIbCastFaultOpsSetPostSendError(send_.get(), 0, 1), ncclInvalidArgument);
  // Resiliency recovery can leave a QP torn down (no ibv_qp) or detached from its context.
  send_->base.qps[1].qp = nullptr;
  recv_->base.qps[1].qp = nullptr;
  EXPECT_EQ(ncclIbCastFaultOpsSetPostSendError(send_.get(), 1, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultOpsSetPostRecvError(recv_.get(), 1, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultOpsSetPollCqError(recv_.get(), 1, 1, 1, false), ncclInvalidArgument);
  ibv_qp detached{};
  send_->base.qps[1].qp = &detached;
  recv_->base.qps[1].qp = &detached;
  EXPECT_EQ(ncclIbCastFaultOpsSetPostSendError(send_.get(), 1, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultOpsSetPostRecvError(recv_.get(), 1, 1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastFaultOpsSetPollCqError(recv_.get(), 1, 1, 1, false), ncclInvalidArgument);
  // Clear walks devices by PD context and skips devices that have none.
  send_->devs[0].base.pd = nullptr;
  EXPECT_EQ(ncclIbCastFaultOpsClear(send_.get()), ncclSuccess);
}

}  // namespace
