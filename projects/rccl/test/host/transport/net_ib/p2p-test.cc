/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib/p2p.cc and p2p.h.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

#include "ScopedHook.h"
#include "fakes/ibv_ops_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/socket_fakes.h"
#include "fakes/transport_net_ib_p2p_fakes.h"
#include "fakes/transport_net_ib_resiliency_fakes.h"
#include "fakes/transport_net_ib_wqe_lat_fakes.h"
#include "transport/net_ib/common.h"

// Keep p2p.cc's generic owner fake available to sibling transport tests while
// this TU compiles the real implementation under unit-qualified plugin names.
#define ncclParamIbArThreshold NetIbP2pParamIbArThreshold
#define ncclIbArThreshold NetIbP2pArThreshold
#define ncclParamIbReceiverSideMatchingScheme NetIbP2pParamIbReceiverSideMatchingScheme
#define ncclIbIsend NetIbP2pIsend
#define ncclIbIrecv NetIbP2pIrecv
#define ncclIbIflush NetIbP2pIflush
#define ncclIbTest NetIbP2pTest
#include NET_IB_P2P_CC_PATH
#undef ncclIbTest
#undef ncclIbIflush
#undef ncclIbIrecv
#undef ncclIbIsend
#undef ncclParamIbReceiverSideMatchingScheme
#undef ncclIbArThreshold
#undef ncclParamIbArThreshold

namespace {

void InitOneQpBase(ncclIbNetCommBase* base, FakeIbvVerbs* verbs, bool isSend) {
  base->isSend = isSend;
  base->ready = 1;
  base->vProps.ndevs = 1;
  base->nqps = 1;
  base->nDataQps = 1;
  base->splitDataOnQps = 1;
  base->qps[0].qp = &verbs->qp;
  base->qps[0].devIndex = 0;
  base->qps[0].remDevIdx = 0;
  base->activeQps[0] = &base->qps[0];
  base->weights[0] = 100;
}

class NetIbP2pMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_ncclSocketGetAddr = [](ncclSocket*, ncclSocketAddress* address) {
      if (address) std::memset(address, 0, sizeof(*address));
      return ncclSuccess;
    };
    g_ncclSocketToString = [](const ncclSocketAddress*, char* buffer, size_t size, int) {
      if (buffer && size > 0) std::snprintf(buffer, size, "micro-peer");
      return buffer;
    };
  }

  void TearDown() override {
    ResetIbvOpsFakes();
    COMPILER_ATOMIC_STORE(&ncclIbSpeedChangeCounter, uint64_t{0}, std::memory_order_release);
    COMPILER_ATOMIC_STORE(&ncclIbDevs[0].currSpeed, uint64_t{0}, std::memory_order_relaxed);
    COMPILER_ATOMIC_STORE(&ncclIbDevs[0].stats.fatalErrorCount, 0, std::memory_order_relaxed);
    ResetTransportNetIbWqeLatFakes();
    ResetTransportNetIbResiliencyFakes();
    ResetTransportNetIbP2pFakes();
    ResetSocketFakes();
    ResetNcclFakes();
  }
};

TEST_F(NetIbP2pMicrotest, RecvCommGetQpForCts_RequestIdSelectsFirstQpOnOwningDevice) {
  ncclIbRecvComm comm{};
  ncclIbQp first{};
  ncclIbQp second{};
  comm.base.vProps.ndevs = 2;
  comm.base.activeQps[0] = &first;
  comm.base.activeQps[1] = &second;
  ncclIbQp* selected = nullptr;

  EXPECT_EQ(ncclIbRecvCommGetQpForCts(&comm, /*id=*/3, &selected), ncclSuccess);
  EXPECT_EQ(selected, &second);
}

TEST_F(NetIbP2pMicrotest, RecvCommGetQpForCts_MissingQpReturnsError) {
  ncclIbRecvComm comm{};
  comm.base.vProps.ndevs = 1;
  ncclIbQp* selected = reinterpret_cast<ncclIbQp*>(0x1);

  EXPECT_EQ(ncclIbRecvCommGetQpForCts(&comm, /*id=*/0, &selected), ncclInternalError);
  EXPECT_EQ(selected, nullptr);
}

TEST_F(NetIbP2pMicrotest, RequestRetrieveAsIndex_ValidIndexReturnsPoolEntryAndCapacityIsRejected) {
  std::array<ncclIbRequest, NET_IB_MAX_REQUESTS> requests{};
  ncclIbRequest* request = nullptr;

  EXPECT_EQ(ncclIbRequestRetrieveAsIndex(requests.data(), NET_IB_MAX_REQUESTS - 1, &request), ncclSuccess);
  EXPECT_EQ(request, &requests.back());
  EXPECT_EQ(ncclIbRequestRetrieveAsIndex(requests.data(), NET_IB_MAX_REQUESTS, &request), ncclInternalError);
}

TEST_F(NetIbP2pMicrotest, GetRequest_FirstUnusedEntryResetsCompletionBookkeeping) {
  ncclIbNetCommBase base{};
  base.reqs[0].type = NCCL_NET_IB_REQ_SEND;
  base.reqs[1].type = NCCL_NET_IB_REQ_UNUSED;
  base.reqs[1].events[0] = 3;
  base.reqs[1].devBases[0] = reinterpret_cast<ncclIbNetCommDevBase*>(0x1);
  ncclIbRequest* request = nullptr;

  ASSERT_EQ(ncclIbGetRequest(&base, &request), ncclSuccess);
  EXPECT_EQ(request, &base.reqs[1]);
  EXPECT_EQ(request->base, &base);
  EXPECT_EQ(request->sock, nullptr);
  EXPECT_EQ(request->events[0], 0);
  EXPECT_EQ(request->devBases[0], nullptr);
}

TEST_F(NetIbP2pMicrotest, GetRequest_ExhaustedPoolReturnsErrorAndNull) {
  ncclIbNetCommBase base{};
  for (auto& request : base.reqs) request.type = NCCL_NET_IB_REQ_SEND;
  ncclIbRequest* request = reinterpret_cast<ncclIbRequest*>(0x1);

  EXPECT_EQ(ncclIbGetRequest(&base, &request), ncclInternalError);
  EXPECT_EQ(request, nullptr);
}

TEST_F(NetIbP2pMicrotest, FreeRequest_LiveEntryBecomesReusable) {
  ncclIbRequest request{};
  request.type = NCCL_NET_IB_REQ_RECV;

  EXPECT_EQ(ncclIbFreeRequest(&request), ncclSuccess);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbP2pMicrotest, AddEvent_TracksOwningCompletionQueue) {
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base.ibDevN = 4;
  ncclIbRequest request{};
  request.base = &comm.base;

  ncclIbAddEvent(&request, 0);

  EXPECT_EQ(request.events[0], 1);
  EXPECT_EQ(request.devBases[0], &comm.devs[0].base);
}

TEST_F(NetIbP2pMicrotest, RequestIsComplete_AllCountersZeroCompletes) {
  ncclIbNetCommBase base{};
  ncclIbRequest request{};
  request.base = &base;
  bool complete = false;

  EXPECT_EQ(ncclIbRequestIsComplete(&request, &complete), ncclSuccess);
  EXPECT_TRUE(complete);
  request.events[2] = 1;
  EXPECT_EQ(ncclIbRequestIsComplete(&request, &complete), ncclSuccess);
  EXPECT_FALSE(complete);
}

TEST_F(NetIbP2pMicrotest, RequestComplete_ReceiveReportsRecordedSizesAndReleasesSlot) {
  ncclIbRecvComm comm{};
  ncclIbRequest request{};
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_RECV;
  request.nreqs = 2;
  request.recv.cmplsRecords = &comm.cmplsRecords[0];
  request.recv.cmplsRecords->sizes[0] = 7;
  request.recv.cmplsRecords->sizes[1] = 9;
  int done = 0;
  int sizes[2] = {};

  EXPECT_EQ(ncclIbRequestComplete(&request, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 7);
  EXPECT_EQ(sizes[1], 9);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbP2pMicrotest, RequestComplete_FinalSendClearsSlotAndReturnsSize) {
  ncclIbSendComm comm{};
  ncclIbRequest request{};
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_SEND;
  request.id = 3;
  request.send.size = 11;
  comm.sendReqsCnt[3] = 1;
  comm.sendReqs[3][0] = &request;
  int done = 0;
  int size = 0;

  EXPECT_EQ(ncclIbRequestComplete(&request, &done, &size), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(size, 11);
  EXPECT_EQ(comm.sendReqsCnt[3], 0);
  EXPECT_EQ(comm.sendReqs[3][0], nullptr);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbP2pMicrotest, RequestRetrieveFromCompletion_InvalidInputsOrErrorCompletionAreRejected) {
  ncclIbNetCommBase base{};
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  ncclIbRequest* request = nullptr;

  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&base, nullptr, &request), ncclInternalError);
  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&base, &completion, nullptr), ncclInternalError);
  completion.status = IBV_WC_REM_ACCESS_ERR;
  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&base, &completion, &request), ncclInternalError);
}

TEST_F(NetIbP2pMicrotest, RequestRetrieveFromCompletion_DecodesRecvFlushAndSendSlots) {
  ncclIbRecvComm recv{};
  recv.base.isSend = false;
  recv.base.recvMatchingScheme = BY_INDEX;
  recv.recvReqs[5] = &recv.base.reqs[5];
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV;
  completion.wr_id = 5;
  ncclIbRequest* request = nullptr;
  ASSERT_EQ(ncclIbRequestRetrieveFromCompletion(&recv.base, &completion, &request), ncclSuccess);
  EXPECT_EQ(request, &recv.base.reqs[5]);

  completion.opcode = IBV_WC_RDMA_READ;
  completion.wr_id = NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + 6;
  ASSERT_EQ(ncclIbRequestRetrieveFromCompletion(&recv.base, &completion, &request), ncclSuccess);
  EXPECT_EQ(request, &recv.base.reqs[6]);

  ncclIbSendComm send{};
  send.base.isSend = true;
  send.sendReqs[7][0] = &send.base.reqs[7];
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.wr_id = 0x107;
  ASSERT_EQ(ncclIbRequestRetrieveFromCompletion(&send.base, &completion, &request), ncclSuccess);
  EXPECT_EQ(request, &send.base.reqs[7]);
}

TEST_F(NetIbP2pMicrotest, RequestRetrieveFromCompletion_ByIdUsesImmediateRequestId) {
  ncclIbRecvComm recv{};
  recv.base.isSend = false;
  recv.base.recvMatchingScheme = BY_ID;
  recv.recvReqs[33] = &recv.base.reqs[33];
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV_RDMA_WITH_IMM;
  completion.imm_data = htobe32(33);
  ncclIbRequest* request = nullptr;

  EXPECT_EQ(ncclIbRequestRetrieveFromCompletion(&recv.base, &completion, &request), ncclSuccess);
  EXPECT_EQ(request, &recv.base.reqs[33]);
}

TEST_F(NetIbP2pMicrotest, CompletionEvent_ByIdAggregatesBytesAndCompletionReportsTotal) {
  ncclIbRecvComm comm{};
  comm.base.isSend = false;
  comm.base.recvMatchingScheme = BY_ID;
  auto& request = comm.base.reqs[33];
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_RECV;
  request.id = 33;
  request.nreqs = 1;
  request.events[0] = 2;
  request.recv.cmplsRecords = &comm.cmplsRecords[33];
  comm.recvReqs[33] = &request;
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV_RDMA_WITH_IMM;
  completion.imm_data = htobe32(33);
  completion.byte_len = 5;

  ASSERT_EQ(ncclIbCompletionEventProcess(&comm.base, &completion, 0), ncclSuccess);
  ASSERT_EQ(ncclIbCompletionEventProcess(&comm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(request.recv.aggSize, 10);
  EXPECT_EQ(request.events[0], 0);
  int done = 0;
  int size = 0;
  EXPECT_EQ(ncclIbRequestComplete(&request, &done, &size), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(size, 10);
}

TEST_F(NetIbP2pMicrotest, IsendAndIrecv_NotReadyRejectAndClearRequest) {
  ncclIbSendComm send{};
  ncclIbRecvComm recv{};
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(NetIbP2pIsend(&send, nullptr, 0, 0, nullptr, nullptr, &request), ncclInternalError);
  EXPECT_EQ(request, nullptr);
  request = reinterpret_cast<void*>(0x1);
  EXPECT_EQ(NetIbP2pIrecv(&recv, 0, nullptr, nullptr, nullptr, nullptr, nullptr, &request), ncclInternalError);
  EXPECT_EQ(request, nullptr);
}

TEST_F(NetIbP2pMicrotest, Irecv_ReceiveCountAboveCapacityIsRejected) {
  ncclIbRecvComm recv{};
  recv.base.ready = 1;
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(NetIbP2pIrecv(&recv, NCCL_NET_IB_MAX_RECVS + 1, nullptr, nullptr, nullptr, nullptr, nullptr, &request),
            ncclInternalError);
}

TEST_F(NetIbP2pMicrotest, Iflush_DisabledOrAllEmptyInputPostsNothing) {
  ncclIbRecvComm recv{};
  int sizes[2] = {1, 0};
  void* request = reinterpret_cast<void*>(0x1);

  recv.flushEnabled = 0;
  EXPECT_EQ(NetIbP2pIflush(&recv, 2, nullptr, sizes, nullptr, &request), ncclSuccess);
  EXPECT_TRUE(g_ibvPostedSends.empty());
  recv.flushEnabled = 1;
  sizes[0] = 0;
  EXPECT_EQ(NetIbP2pIflush(&recv, 2, nullptr, sizes, nullptr, &request), ncclSuccess);
  EXPECT_TRUE(g_ibvPostedSends.empty());
}

TEST_F(NetIbP2pMicrotest, PostFifo_OwningDeviceSlotPostsSignaledCts) {
  FakeIbvVerbs verbs(17);
  ncclIbRecvComm comm{};
  comm.base.vProps.ndevs = 1;
  comm.base.activeQps[0] = &comm.base.qps[0];
  comm.base.qps[0].qp = &verbs.qp;
  comm.base.qps[0].devIndex = 0;
  comm.base.qps[0].remDevIdx = 0;
  comm.base.remDevs[0].rkey = 0x44;
  comm.remCtsFifo.addr = 0x1000;
  comm.remCtsFifo.flags = IBV_SEND_INLINE;
  comm.devs[0].sge.lkey = 0x55;
  ncclIbRequest request{};
  request.base = &comm.base;
  request.id = 0;
  request.nreqs = 1;

  ASSERT_EQ(ncclIbPostFifo(&comm, &request, /*slot=*/0), ncclSuccess);
  ASSERT_EQ(g_ibvPostedSends.size(), 1u);
  EXPECT_EQ(g_ibvPostedSends[0].qpNum, 17u);
  EXPECT_EQ(g_ibvPostedSends[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_NE(g_ibvPostedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_ibvPostedSends[0].wrId, 0u);
  EXPECT_EQ(g_ibvPostedSends[0].remoteAddress, 0x1000u);
  EXPECT_EQ(g_ibvPostedSends[0].rkey, 0x44u);
}

TEST_F(NetIbP2pMicrotest, PostFifo_DifferentSlotIsUnsignaledUnlessResiliencyIsEnabled) {
  FakeIbvVerbs verbs(18);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.base.remDevs[0].rkey = 0x44;
  comm.remCtsFifo.addr = 0x1000;
  ncclIbRequest request{};
  request.base = &comm.base;
  request.id = 0;
  request.nreqs = 1;

  ASSERT_EQ(ncclIbPostFifo(&comm, &request, /*slot=*/1), ncclSuccess);
  ASSERT_EQ(g_ibvPostedSends.size(), 1u);
  EXPECT_EQ(g_ibvPostedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);

  g_ibvPostedSends.clear();
  ncclIbResiliency resiliency{};
  comm.base.resiliency = &resiliency;
  ASSERT_EQ(ncclIbPostFifo(&comm, &request, /*slot=*/1), ncclSuccess);
  ASSERT_EQ(g_ibvPostedSends.size(), 1u);
  EXPECT_NE(g_ibvPostedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbP2pMicrotest, MultiSend_SmallPayloadUsesOnlyRotatedActiveQp) {
  FakeIbvVerbs first(21);
  FakeIbvVerbs second(22);
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.base.nqps = 2;
  comm.base.nDataQps = 2;
  comm.base.splitDataOnQps = 1;
  comm.base.recvMatchingScheme = BY_INDEX;
  comm.base.qps[0].qp = &first.qp;
  comm.base.qps[0].devIndex = 0;
  comm.base.qps[0].remDevIdx = 0;
  comm.base.qps[1].qp = &second.qp;
  comm.base.qps[1].devIndex = 1;
  comm.base.qps[1].remDevIdx = 1;
  comm.base.activeQps[0] = &comm.base.qps[0];
  comm.base.activeQps[1] = &comm.base.qps[1];
  ncclIbRequest request{};
  request.base = &comm.base;
  request.id = 0;
  request.type = NCCL_NET_IB_REQ_SEND;
  std::array<unsigned char, 64> data{};
  request.send.data = data.data();
  request.send.size = data.size();
  request.send.weights[0] = 50;
  request.send.weights[1] = 50;
  request.send.lkeys[0] = 0x10;
  request.send.lkeys[1] = 0x11;
  comm.sendReqs[0][0] = &request;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].addr = 0x2000;
  comm.ctsFifo[0][0].rkeys[0] = 0x20;
  comm.ctsFifo[0][0].rkeys[1] = 0x21;

  ASSERT_EQ(ncclIbMultiSend(&comm, 0), ncclSuccess);
  ASSERT_EQ(g_ibvPostedSends.size(), 2u);
  EXPECT_EQ(g_ibvPostedSends[0].qpNum, 21u);
  EXPECT_EQ(g_ibvPostedSends[0].sge.length, data.size());
  EXPECT_EQ(g_ibvPostedSends[1].qpNum, 22u);
  EXPECT_EQ(g_ibvPostedSends[1].sge.length, 0u);
  EXPECT_EQ(g_ibvPostedSends[0].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_EQ(be32toh(g_ibvPostedSends[0].immData), data.size());
}

TEST_F(NetIbP2pMicrotest, MultiSend_WeightedPayloadAlignsChunksAndClampsFinalBytes) {
  FakeIbvVerbs first(31);
  FakeIbvVerbs second(32);
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.base.nqps = 2;
  comm.base.nDataQps = 2;
  comm.base.splitDataOnQps = 1;
  comm.base.recvMatchingScheme = BY_INDEX;
  for (int i = 0; i < 2; ++i) {
    comm.base.qps[i].qp = i == 0 ? &first.qp : &second.qp;
    comm.base.qps[i].devIndex = i;
    comm.base.qps[i].remDevIdx = i;
    comm.base.activeQps[i] = &comm.base.qps[i];
  }
  ncclIbRequest request{};
  request.base = &comm.base;
  request.id = 0;
  request.type = NCCL_NET_IB_REQ_SEND;
  std::array<unsigned char, 300> data{};
  request.send.data = data.data();
  request.send.size = data.size();
  request.send.weights[0] = 75;
  request.send.weights[1] = 25;
  comm.sendReqs[0][0] = &request;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].addr = 0x3000;
  comm.ctsFifo[0][0].rkeys[0] = 1;
  comm.ctsFifo[0][0].rkeys[1] = 2;

  ASSERT_EQ(ncclIbMultiSend(&comm, 0), ncclSuccess);
  ASSERT_EQ(g_ibvPostedSends.size(), 2u);
  EXPECT_EQ(g_ibvPostedSends[0].sge.length, 256u);
  EXPECT_EQ(g_ibvPostedSends[1].sge.length, 44u);
  EXPECT_EQ(g_ibvPostedSends[0].sge.length + g_ibvPostedSends[1].sge.length, data.size());
  EXPECT_EQ(g_ibvPostedSends[1].remoteAddress, 0x3000u + 256);
}

TEST_F(NetIbP2pMicrotest, MultiSend_PayloadAtSplitThresholdUsesWeightedPath) {
  FakeIbvVerbs first(33);
  FakeIbvVerbs second(34);
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.base.nqps = 2;
  comm.base.nDataQps = 2;
  comm.base.splitDataOnQps = 1;
  for (int i = 0; i < 2; ++i) {
    comm.base.qps[i].qp = i == 0 ? &first.qp : &second.qp;
    comm.base.qps[i].devIndex = i;
    comm.base.qps[i].remDevIdx = i;
    comm.base.activeQps[i] = &comm.base.qps[i];
  }
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "RCCL_IB_SPLIT_DATA_THRESHOLD") == 0 ? int64_t{256} : fallback;
  });
  ncclIbRequest request{};
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_SEND;
  std::array<unsigned char, 256> data{};
  request.send.data = data.data();
  request.send.size = data.size();
  request.send.weights[0] = request.send.weights[1] = 50;
  comm.sendReqs[0][0] = &request;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].addr = 0x3500;
  comm.ctsFifo[0][0].rkeys[0] = 1;
  comm.ctsFifo[0][0].rkeys[1] = 2;

  ASSERT_EQ(ncclIbMultiSend(&comm, 0), ncclSuccess);
  ASSERT_EQ(g_ibvPostedSends.size(), 2u);
  EXPECT_EQ(g_ibvPostedSends[0].sge.length, 128u);
  EXPECT_EQ(g_ibvPostedSends[1].sge.length, 128u);
}

TEST_F(NetIbP2pMicrotest, MultiSend_RequestRotationSelectsOwningActiveQp) {
  FakeIbvVerbs first(35);
  FakeIbvVerbs second(36);
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 2;
  comm.base.nqps = 2;
  comm.base.nDataQps = 1;
  comm.base.splitDataOnQps = 0;
  comm.base.qps[0].qp = &first.qp;
  comm.base.qps[0].devIndex = 0;
  comm.base.qps[0].remDevIdx = 0;
  comm.base.qps[1].qp = &second.qp;
  comm.base.qps[1].devIndex = 1;
  comm.base.qps[1].remDevIdx = 1;
  comm.base.activeQps[0] = &comm.base.qps[0];
  comm.base.activeQps[1] = &comm.base.qps[1];
  ncclIbRequest request{};
  request.base = &comm.base;
  request.id = 1;
  request.type = NCCL_NET_IB_REQ_SEND;
  std::array<unsigned char, 64> data{};
  request.send.data = data.data();
  request.send.size = data.size();
  request.send.weights[0] = request.send.weights[1] = 50;
  comm.sendReqs[0][0] = &request;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].addr = 0x3600;
  comm.ctsFifo[0][0].rkeys[0] = 1;
  comm.ctsFifo[0][0].rkeys[1] = 2;

  ASSERT_EQ(ncclIbMultiSend(&comm, 0), ncclSuccess);
  ASSERT_EQ(g_ibvPostedSends.size(), 1u);
  EXPECT_EQ(g_ibvPostedSends[0].qpNum, 36u);
  EXPECT_EQ(g_ibvPostedSends[0].sge.length, data.size());
}

TEST_F(NetIbP2pMicrotest, Isend_MatchedReceivePublishesClampedRequestAndAdvancesFifo) {
  FakeIbvVerbs verbs(41);
  ncclIbSendComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/true);
  comm.ctsFifo[0][0].idx = 1;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].tag = 7;
  comm.ctsFifo[0][0].size = 32;
  comm.ctsFifo[0][0].addr = 0x4000;
  comm.ctsFifo[0][0].rkeys[0] = 0x42;
  ibv_mr mr{};
  mr.lkey = 0x55;
  ncclIbMrHandle handle{};
  handle.mrs[0] = &mr;
  std::array<unsigned char, 64> data{};
  void* request = nullptr;

  ASSERT_EQ(NetIbP2pIsend(&comm, data.data(), data.size(), 7, &handle, nullptr, &request), ncclSuccess);
  auto* ibRequest = static_cast<ncclIbRequest*>(request);
  ASSERT_NE(ibRequest, nullptr);
  EXPECT_EQ(ibRequest->type, NCCL_NET_IB_REQ_SEND);
  EXPECT_EQ(ibRequest->send.size, 32);
  EXPECT_EQ(ibRequest->events[0], 1);
  EXPECT_EQ(comm.base.fifoHead, 1u);
  EXPECT_EQ(comm.sendReqs[0][0], ibRequest);
  EXPECT_EQ(g_ibvPostedSends.size(), 1u);
}

TEST_F(NetIbP2pMicrotest, Isend_MultiReceiveDefersUntilEveryTagMatchesThenPostsBatch) {
  FakeIbvVerbs verbs(42);
  ncclIbSendComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/true);
  for (int i = 0; i < 2; ++i) {
    comm.ctsFifo[0][i].idx = 1;
    comm.ctsFifo[0][i].nreqs = 2;
    comm.ctsFifo[0][i].tag = 7 + i;
    comm.ctsFifo[0][i].size = 64;
    comm.ctsFifo[0][i].addr = 0x4100 + i * 0x100;
    comm.ctsFifo[0][i].rkeys[0] = 0x43;
  }
  ibv_mr mr{};
  mr.lkey = 0x44;
  ncclIbMrHandle handle{};
  handle.mrs[0] = &mr;
  std::array<unsigned char, 64> firstData{};
  std::array<unsigned char, 32> secondData{};
  void* firstRequest = nullptr;
  void* secondRequest = nullptr;

  ASSERT_EQ(NetIbP2pIsend(&comm, firstData.data(), firstData.size(), 7, &handle, nullptr, &firstRequest), ncclSuccess);
  EXPECT_NE(firstRequest, nullptr);
  EXPECT_EQ(comm.sendReqsCnt[0], 1);
  EXPECT_EQ(comm.base.fifoHead, 0u);
  EXPECT_TRUE(g_ibvPostedSends.empty());

  ASSERT_EQ(NetIbP2pIsend(&comm, secondData.data(), secondData.size(), 8, &handle, nullptr, &secondRequest),
            ncclSuccess);
  EXPECT_NE(secondRequest, nullptr);
  EXPECT_EQ(comm.sendReqsCnt[0], 2);
  EXPECT_EQ(comm.base.fifoHead, 1u);
  ASSERT_EQ(g_ibvPostedSends.size(), 3u);
  EXPECT_EQ(g_ibvPostedSends[0].sge.length, firstData.size());
  EXPECT_EQ(g_ibvPostedSends[1].sge.length, secondData.size());
  EXPECT_EQ(g_ibvPostedSends[2].sge.length, 2 * sizeof(int));
}

TEST_F(NetIbP2pMicrotest, Isend_PostFailureWithEventsReturnsFailedRequestForDrain) {
  FakeIbvVerbs verbs(43);
  ncclIbSendComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/true);
  comm.ctsFifo[0][0].idx = 1;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].tag = 7;
  comm.ctsFifo[0][0].size = 16;
  comm.ctsFifo[0][0].addr = 0x4400;
  comm.ctsFifo[0][0].rkeys[0] = 0x45;
  ibv_mr mr{};
  ncclIbMrHandle handle{};
  handle.mrs[0] = &mr;
  g_ibvPostSendResult = EIO;
  void* request = nullptr;

  EXPECT_EQ(NetIbP2pIsend(&comm, nullptr, 16, 7, &handle, nullptr, &request), ncclSuccess);
  auto* failed = static_cast<ncclIbRequest*>(request);
  ASSERT_NE(failed, nullptr);
  EXPECT_EQ(failed->type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(failed->events[0], 1);
  EXPECT_GT(comm.base.stats.fatalErrorCount, 0);
}

TEST_F(NetIbP2pMicrotest, Isend_ReceiveNotPublishedOrTagMismatchReturnsWithoutRequest) {
  ncclIbSendComm comm{};
  comm.base.ready = 1;
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(NetIbP2pIsend(&comm, nullptr, 0, 5, nullptr, nullptr, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);

  comm.ctsFifo[0][0].idx = 1;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].tag = 6;
  EXPECT_EQ(NetIbP2pIsend(&comm, nullptr, 0, 5, nullptr, nullptr, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);
}

TEST_F(NetIbP2pMicrotest, Irecv_ValidRequestPostsReceiveAndCtsMetadata) {
  FakeIbvVerbs verbs(51);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.base.remDevs[0].rkey = 0x61;
  comm.remCtsFifo.addr = 0x5000;
  ibv_mr mr{};
  mr.rkey = 0x62;
  ncclIbMrHandle handle{};
  handle.mrs[0] = &mr;
  void* data[] = {reinterpret_cast<void*>(0x6000)};
  size_t sizes[] = {128};
  int tags[] = {9};
  void* handles[] = {&handle};
  void* request = nullptr;

  ASSERT_EQ(NetIbP2pIrecv(&comm, 1, data, sizes, tags, handles, nullptr, &request), ncclSuccess);
  auto* ibRequest = static_cast<ncclIbRequest*>(request);
  ASSERT_NE(ibRequest, nullptr);
  EXPECT_EQ(ibRequest->type, NCCL_NET_IB_REQ_RECV);
  EXPECT_EQ(ibRequest->events[0], 1);
  EXPECT_EQ(g_ibvPostRecvQps.size(), 1u);
  EXPECT_EQ(g_ibvPostedSends.size(), 1u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].addr, 0x6000u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].rkeys[0], 0x62u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].nreqs, 1u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].size, 128u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].tag, 9u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].idx, 1u);
  EXPECT_EQ(comm.base.fifoHead, 1u);
}

TEST_F(NetIbP2pMicrotest, Irecv_MultiReceivePublishesEveryDescriptorAndClearsCompletionRecords) {
  FakeIbvVerbs verbs(53);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.base.remDevs[0].rkey = 0x63;
  ibv_mr firstMr{};
  ibv_mr secondMr{};
  firstMr.rkey = 0x64;
  secondMr.rkey = 0x65;
  ncclIbMrHandle firstHandle{};
  ncclIbMrHandle secondHandle{};
  firstHandle.mrs[0] = &firstMr;
  secondHandle.mrs[0] = &secondMr;
  void* data[] = {reinterpret_cast<void*>(0x6100), reinterpret_cast<void*>(0x6200)};
  size_t sizes[] = {11, 13};
  int tags[] = {3, 4};
  void* handles[] = {&firstHandle, &secondHandle};
  std::fill(std::begin(comm.cmplsRecords[0].sizes), std::end(comm.cmplsRecords[0].sizes), -1);
  void* request = nullptr;

  ASSERT_EQ(NetIbP2pIrecv(&comm, 2, data, sizes, tags, handles, nullptr, &request), ncclSuccess);
  auto* ibRequest = static_cast<ncclIbRequest*>(request);
  ASSERT_NE(ibRequest, nullptr);
  EXPECT_EQ(ibRequest->nreqs, 2);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].nreqs, 2u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][1].nreqs, 2u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].size, 11u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][1].size, 13u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].tag, 3u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][1].tag, 4u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][0].idx, 1u);
  EXPECT_EQ(comm.remCtsFifo.elems[0][1].idx, 1u);
  EXPECT_EQ(comm.cmplsRecords[0].sizes[0], 0);
  EXPECT_EQ(comm.cmplsRecords[0].sizes[1], 0);
}

TEST_F(NetIbP2pMicrotest, Irecv_ReceivePostFailurePropagatesError) {
  FakeIbvVerbs verbs(54);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  ibv_mr mr{};
  ncclIbMrHandle handle{};
  handle.mrs[0] = &mr;
  void* data[] = {reinterpret_cast<void*>(0x6300)};
  size_t sizes[] = {8};
  int tags[] = {1};
  void* handles[] = {&handle};
  g_ibvPostRecvResult = EIO;
  void* request = nullptr;

  EXPECT_EQ(NetIbP2pIrecv(&comm, 1, data, sizes, tags, handles, nullptr, &request), ncclSystemError);
  EXPECT_EQ(g_ibvPostRecvQps.size(), 1u);
}

// AICOMRCCL-3126: enable when receive-post failure releases the allocated slot.
TEST_F(NetIbP2pMicrotest, DISABLED_Irecv_ReceivePostFailureReleasesAllocatedSlot) {
  FakeIbvVerbs verbs(55);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  ibv_mr mr{};
  ncclIbMrHandle handle{};
  handle.mrs[0] = &mr;
  void* data[] = {reinterpret_cast<void*>(0x6400)};
  size_t sizes[] = {8};
  int tags[] = {1};
  void* handles[] = {&handle};
  g_ibvPostRecvResult = EIO;
  void* request = nullptr;

  EXPECT_EQ(NetIbP2pIrecv(&comm, 1, data, sizes, tags, handles, nullptr, &request), ncclSystemError);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(comm.base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbP2pMicrotest, Irecv_PrepostedModeSkipsReceivePostButPublishesCts) {
  FakeIbvVerbs verbs(52);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.prepostReceiveWorkRequests = true;
  comm.base.remDevs[0].rkey = 1;
  ibv_mr mr{};
  mr.rkey = 2;
  ncclIbMrHandle handle{};
  handle.mrs[0] = &mr;
  void* data[] = {reinterpret_cast<void*>(0x7000)};
  size_t sizes[] = {16};
  int tags[] = {1};
  void* handles[] = {&handle};
  void* request = nullptr;

  ASSERT_EQ(NetIbP2pIrecv(&comm, 1, data, sizes, tags, handles, nullptr, &request), ncclSuccess);
  EXPECT_TRUE(g_ibvPostRecvQps.empty());
  EXPECT_EQ(g_ibvPostedSends.size(), 1u);
}

TEST_F(NetIbP2pMicrotest, Iflush_FallbackPostsSignaledReadFromLastNonzeroBuffer) {
  FakeIbvVerbs verbs(61);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.flushEnabled = 1;
  comm.flushQpsCreated = true;
  comm.devs[0].gpuFlush.qp.qp = &verbs.qp;
  ibv_mr mr{};
  mr.rkey = 0x72;
  ncclIbMrHandle handle{};
  handle.mrs[0] = &mr;
  void* data[] = {reinterpret_cast<void*>(0x7100), reinterpret_cast<void*>(0x7200)};
  int sizes[] = {0, 4};
  void* handles[] = {&handle, &handle};
  void* request = nullptr;

  ASSERT_EQ(NetIbP2pIflush(&comm, 2, data, sizes, handles, &request), ncclSuccess);
  auto* ibRequest = static_cast<ncclIbRequest*>(request);
  ASSERT_NE(ibRequest, nullptr);
  ASSERT_EQ(g_ibvPostedSends.size(), 1u);
  EXPECT_EQ(g_ibvPostedSends[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(g_ibvPostedSends[0].remoteAddress, 0x7200u);
  EXPECT_EQ(g_ibvPostedSends[0].rkey, 0x72u);
  EXPECT_NE(g_ibvPostedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(ibRequest->events[0], 1);
}

TEST_F(NetIbP2pMicrotest, Iflush_GpuScratchPostsWriteBeforeRead) {
  FakeIbvVerbs verbs(62);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.flushEnabled = 1;
  comm.flushQpsCreated = true;
  comm.devs[0].gpuFlush.qp.qp = &verbs.qp;
  int scratch = 0;
  ibv_mr gpuMr{};
  gpuMr.rkey = 0x82;
  comm.devs[0].gpuFlush.gpuFlushGpuMem = &scratch;
  comm.devs[0].gpuFlush.gpuMr = &gpuMr;
  ibv_mr receiveMr{};
  receiveMr.rkey = 0x83;
  ncclIbMrHandle handle{};
  handle.mrs[0] = &receiveMr;
  void* data[] = {reinterpret_cast<void*>(0x8100)};
  int sizes[] = {4};
  void* handles[] = {&handle};
  void* request = nullptr;

  ASSERT_EQ(NetIbP2pIflush(&comm, 1, data, sizes, handles, &request), ncclSuccess);
  ASSERT_EQ(g_ibvPostedSends.size(), 2u);
  EXPECT_EQ(g_ibvPostedSends[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(g_ibvPostedSends[0].remoteAddress, reinterpret_cast<uint64_t>(&scratch));
  EXPECT_EQ(g_ibvPostedSends[1].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(g_ibvPostedSends[1].remoteAddress, reinterpret_cast<uint64_t>(&scratch));
  EXPECT_NE(g_ibvPostedSends[1].sendFlags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbP2pMicrotest, Iflush_FirstPostFailureReleasesRequestAndReturnsError) {
  FakeIbvVerbs verbs(63);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.flushEnabled = 1;
  comm.flushQpsCreated = true;
  comm.devs[0].gpuFlush.qp.qp = &verbs.qp;
  int scratch = 0;
  ibv_mr gpuMr{};
  comm.devs[0].gpuFlush.gpuFlushGpuMem = &scratch;
  comm.devs[0].gpuFlush.gpuMr = &gpuMr;
  ibv_mr receiveMr{};
  ncclIbMrHandle handle{};
  handle.mrs[0] = &receiveMr;
  void* data[] = {reinterpret_cast<void*>(0x8200)};
  int sizes[] = {4};
  void* handles[] = {&handle};
  g_ibvPostSendResult = EIO;
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(NetIbP2pIflush(&comm, 1, data, sizes, handles, &request), ncclSystemError);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(comm.base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_GT(comm.base.stats.fatalErrorCount, 0);
}

// AICOMRCCL-3126: enable when the READ-post result is returned to the caller.
TEST_F(NetIbP2pMicrotest, DISABLED_Iflush_ReadPostFailurePropagatesError) {
  FakeIbvVerbs verbs(64);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.flushEnabled = 1;
  comm.flushQpsCreated = true;
  comm.devs[0].gpuFlush.qp.qp = &verbs.qp;
  int scratch = 0;
  ibv_mr gpuMr{};
  comm.devs[0].gpuFlush.gpuFlushGpuMem = &scratch;
  comm.devs[0].gpuFlush.gpuMr = &gpuMr;
  ibv_mr receiveMr{};
  ncclIbMrHandle handle{};
  handle.mrs[0] = &receiveMr;
  void* data[] = {reinterpret_cast<void*>(0x8300)};
  int sizes[] = {4};
  void* handles[] = {&handle};
  g_ibvPostSendFailOnCall = 2;
  void* request = nullptr;

  EXPECT_EQ(NetIbP2pIflush(&comm, 1, data, sizes, handles, &request), ncclSystemError);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(comm.base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbP2pMicrotest, Test_NoCompletionsReturnsNotDoneWithoutSpinning) {
  FakeIbvVerbs verbs;
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.devs[0].base.cq = &verbs.cq;
  ncclIbRequest request{};
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_FLUSH;
  request.events[0] = 1;
  request.devBases[0] = &comm.devs[0].base;
  int done = -1;

  EXPECT_EQ(NetIbP2pTest(&request, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_TRUE(g_ibvCompletions.empty());
}

TEST_F(NetIbP2pMicrotest, Test_PollFailureMarksRequestConnectionAndDeviceFatal) {
  FakeIbvVerbs verbs;
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.devs[0].base.cq = &verbs.cq;
  comm.devs[0].base.ibDevN = 0;
  auto& request = comm.base.reqs[0];
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_FLUSH;
  request.events[0] = 1;
  request.devBases[0] = &comm.devs[0].base;
  g_ibvPollResult = -1;
  COMPILER_ATOMIC_STORE(&ncclIbDevs[0].stats.fatalErrorCount, 0, std::memory_order_relaxed);
  int done = 0;

  EXPECT_EQ(NetIbP2pTest(&request, &done, nullptr), ncclSystemError);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_GT(comm.base.stats.fatalErrorCount, 0);
  EXPECT_GT(ncclIbDevs[0].stats.fatalErrorCount, 0);
}

TEST_F(NetIbP2pMicrotest, Test_SuccessfulCompletionDrainsAndFinishesRequest) {
  FakeIbvVerbs verbs;
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.devs[0].base.cq = &verbs.cq;
  auto& request = comm.base.reqs[0];
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_FLUSH;
  request.events[0] = 1;
  request.devBases[0] = &comm.devs[0].base;
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RDMA_READ;
  completion.wr_id = NCCL_IB_FLUSH_REQ_WR_ID_OFFSET;
  g_ibvCompletions.push_back(completion);
  int done = 0;

  EXPECT_EQ(NetIbP2pTest(&request, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbP2pMicrotest, RequestIsComplete_ResiliencyCanCompleteOutstandingRequest) {
  ncclIbNetCommBase base{};
  ncclIbResiliency resiliency{};
  base.resiliency = &resiliency;
  ncclIbRequest request{};
  request.base = &base;
  request.events[0] = 1;
  ScopedHook complete(g_ncclIbResiliencyRequestIsComplete, [](ncclIbRequest*, bool* isComplete) {
    *isComplete = true;
    return ncclSuccess;
  });
  bool isComplete = false;

  EXPECT_EQ(ncclIbRequestIsComplete(&request, &isComplete), ncclSuccess);
  EXPECT_TRUE(isComplete);
  EXPECT_EQ(complete.calls, 1);
}

TEST_F(NetIbP2pMicrotest, CompletionEvent_SendConsumesEverySiblingEvent) {
  ncclIbSendComm comm{};
  comm.base.isSend = true;
  auto& first = comm.base.reqs[0];
  auto& second = comm.base.reqs[1];
  first.base = &comm.base;
  second.base = &comm.base;
  first.type = second.type = NCCL_NET_IB_REQ_SEND;
  first.id = second.id = 5;
  first.nreqs = second.nreqs = 2;
  first.events[0] = second.events[0] = 1;
  comm.sendReqs[5][0] = &first;
  comm.sendReqs[5][1] = &second;
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.wr_id = 5;

  EXPECT_EQ(ncclIbCompletionEventProcess(&comm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(first.events[0], 0);
  EXPECT_EQ(second.events[0], 0);
}

TEST_F(NetIbP2pMicrotest, CompletionEvent_IndexMatchedReceiveRecordsImmediateSize) {
  ncclIbRecvComm comm{};
  comm.base.isSend = false;
  comm.base.recvMatchingScheme = BY_INDEX;
  auto& request = comm.base.reqs[0];
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_RECV;
  request.id = 6;
  request.nreqs = 1;
  request.events[0] = 1;
  request.recv.cmplsRecords = &comm.cmplsRecords[6];
  comm.recvReqs[6] = &request;
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV_RDMA_WITH_IMM;
  completion.wr_id = 6;
  completion.imm_data = htobe32(123);

  EXPECT_EQ(ncclIbCompletionEventProcess(&comm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(request.recv.cmplsRecords->sizes[0], 123);
  EXPECT_EQ(request.events[0], 0);
}

TEST_F(NetIbP2pMicrotest, CompletionEvent_SpeedUpdateClearsPendingState) {
  ncclIbRecvComm comm{};
  comm.postedSpeedUpdate = true;
  ibv_wc completion{};
  completion.wr_id = NCCL_IB_SPEED_UPDATE_WR_ID;

  EXPECT_EQ(ncclIbCompletionEventProcess(&comm.base, &completion, 0), ncclSuccess);
  EXPECT_FALSE(comm.postedSpeedUpdate);
}

TEST_F(NetIbP2pMicrotest, PostSpeedUpdate_ChangedSpeedPostsSignaledSnapshot) {
  FakeIbvVerbs verbs(81);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.base.vProps.devs[0] = 0;
  comm.base.speedChangeCounter = 0;
  comm.remSpeedBufAddr = 0xb000;
  comm.base.remDevs[0].remSpeedBufRkey = 0xb1;
  ibv_mr speedMr{};
  speedMr.lkey = 0xb2;
  comm.devs[0].speedUpdateMr = &speedMr;
  COMPILER_ATOMIC_STORE(&ncclIbDevs[0].currSpeed, uint64_t{200000}, std::memory_order_relaxed);
  COMPILER_ATOMIC_STORE(&ncclIbSpeedChangeCounter, uint64_t{1}, std::memory_order_release);

  ASSERT_EQ(ncclIbPostSpeedUpdateToRemote(&comm), ncclSuccess);
  ASSERT_EQ(g_ibvPostedSends.size(), 1u);
  EXPECT_EQ(g_ibvPostedSends[0].wrId, NCCL_IB_SPEED_UPDATE_WR_ID);
  EXPECT_EQ(g_ibvPostedSends[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_NE(g_ibvPostedSends[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(g_ibvPostedSends[0].remoteAddress, 0xb000u);
  EXPECT_EQ(g_ibvPostedSends[0].rkey, 0xb1u);
  EXPECT_TRUE(comm.postedSpeedUpdate);
  EXPECT_EQ(comm.speedUpdateBuf.speedGbps[0], 200);
}

TEST_F(NetIbP2pMicrotest, PostSpeedUpdate_UnchangedOrPendingStatePostsNothing) {
  FakeIbvVerbs verbs(82);
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.base.vProps.devs[0] = 0;
  comm.remSpeedBufAddr = 0xc000;
  comm.base.remDevs[0].remSpeedBufRkey = 0xc1;
  ibv_mr speedMr{};
  speedMr.lkey = 0xc2;
  comm.devs[0].speedUpdateMr = &speedMr;
  COMPILER_ATOMIC_STORE(&ncclIbDevs[0].currSpeed, uint64_t{200000}, std::memory_order_relaxed);

  comm.base.speedChangeCounter = 7;
  COMPILER_ATOMIC_STORE(&ncclIbSpeedChangeCounter, uint64_t{7}, std::memory_order_release);
  EXPECT_EQ(ncclIbPostSpeedUpdateToRemote(&comm), ncclSuccess);
  EXPECT_TRUE(g_ibvPostedSends.empty());

  comm.base.speedChangeCounter = 0;
  comm.postedSpeedUpdate = true;
  COMPILER_ATOMIC_STORE(&ncclIbSpeedChangeCounter, uint64_t{8}, std::memory_order_release);
  EXPECT_EQ(ncclIbPostSpeedUpdateToRemote(&comm), ncclSuccess);
  EXPECT_TRUE(g_ibvPostedSends.empty());
}

TEST_F(NetIbP2pMicrotest, Test_ErrorCompletionWithoutResiliencyReturnsRemoteError) {
  FakeIbvVerbs verbs;
  ibv_device device{};
  std::snprintf(device.name, sizeof(device.name), "micro0");
  verbs.context.device = &device;
  ibv_pd pd{};
  pd.context = &verbs.context;
  ncclIbRecvComm comm{};
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.devs[0].base.cq = &verbs.cq;
  comm.devs[0].base.pd = &pd;
  auto& request = comm.base.reqs[0];
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_FLUSH;
  request.events[0] = 1;
  request.devBases[0] = &comm.devs[0].base;
  ibv_wc completion{};
  completion.status = IBV_WC_REM_ACCESS_ERR;
  completion.wr_id = NCCL_IB_FLUSH_REQ_WR_ID_OFFSET;
  completion.qp_num = 100;
  g_ibvCompletions.push_back(completion);
  int done = 0;

  EXPECT_EQ(NetIbP2pTest(&request, &done, nullptr), ncclRemoteError);
  EXPECT_EQ(done, 0);
}

}  // namespace
