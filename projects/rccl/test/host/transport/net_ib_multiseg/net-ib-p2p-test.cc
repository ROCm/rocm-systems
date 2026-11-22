/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib_multiseg/p2p.cc and p2p.h.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "IbVerbsScene.h"
#include "ScopedHook.h"
#include "fakes/ibvwrap_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "fakes/socket_fakes.h"
#include "fakes/transport_net_ib_mseg_resiliency_fakes.h"
#include "fakes/transport_net_ib_mseg_wqe_lat_fakes.h"
#include "transport/net_ib_multiseg/NetIbMsegGlobals.h"
#include "transport/net_ib_multiseg/common_mseg.h"

#include NET_IB_MSEG_P2P_CC_PATH
// p2p.cc calls the multi-segment hooks in p2p_multiseg.cc.
#include NET_IB_MSEG_P2P_MULTISEG_CC_PATH
namespace {

void InitOneQpBase(IbMsegNetCommBase* base, IbVerbsScene* verbs, bool isSend) {
  base->isSend = isSend;
  base->ready = 1;
  base->vProps.ndevs = 1;
  base->nqps = 1;
  base->nDataQps = 1;
  base->splitDataOnQps = 1;
  base->qps[0].qp = &verbs->qps[0];
  base->qps[0].devIndex = 0;
  base->qps[0].remDevIdx = 0;
  base->activeQps[0] = &base->qps[0];
  base->weights[0] = 100;
}

// The legacy CTS FIFO (ctsFifo, or remCtsFifo.elems on a receiver) is a separate block; back it with zeroed rows.
struct CtsFifoRows {
  explicit CtsFifoRows(IbMsegSendComm* comm) { comm->ctsFifo = rows; }
  explicit CtsFifoRows(IbMsegRecvComm* comm) { comm->remCtsFifo.elems = rows; }
  CtsFifoRows(const CtsFifoRows&) = delete;
  CtsFifoRows& operator=(const CtsFifoRows&) = delete;
  IbMsegSendFifo rows[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS]{};
};

// A receive comm on QP 0 of its own scene, numbered qpNum.
struct OneQpRecvComm {
  explicit OneQpRecvComm(uint32_t qpNum) : verbs("mlx5_0", qpNum) {
    InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  }

  IbVerbsScene verbs;
  IbMsegRecvComm comm{};
  CtsFifoRows cts{&comm};
};

// OneQpRecvComm with flushing on: the flush QP is the data QP and device 0's CQ is the scene's. With
// `withScratch`, device 0 also has the RO=0 GPU scratchpad a flush fences through.
struct FlushRecvComm : OneQpRecvComm {
  FlushRecvComm(uint32_t qpNum, bool withScratch) : OneQpRecvComm(qpNum) {
    comm.flushEnabled = 1;
    comm.flushQpsCreated = true;
    comm.devs[0].gpuFlush.qp.qp = &verbs.qps[0];
    comm.devs[0].base.cq = &verbs.cq;
    if (withScratch) {
      scratchMr.rkey = 0x82;
      comm.devs[0].gpuFlush.gpuFlushGpuMem = &scratch;
      comm.devs[0].gpuFlush.gpuMr = &scratchMr;
    }
    receiveMr.rkey = 0x83;
    handle.mrs[0] = &receiveMr;
  }

  // A flush waiting on one completion from device 0.
  IbMsegRequest& PendingFlush() {
    auto& request = comm.base.reqs[0];
    request.base = &comm.base;
    request.type = NCCL_NET_IB_REQ_FLUSH;
    request.events[0] = 1;
    request.devBases[0] = &comm.devs[0].base;
    return request;
  }

  int scratch = 0;
  ibv_mr scratchMr{};
  ibv_mr receiveMr{};
  IbMsegMrHandle handle{};
};

// The arguments of a one-receive irecv.
struct IrecvArgs {
  IrecvArgs(uintptr_t addr, size_t size, int tag, uint32_t rkey = 0) {
    mr.rkey = rkey;
    handle.mrs[0] = &mr;
    data[0] = reinterpret_cast<void*>(addr);
    sizes[0] = size;
    tags[0] = tag;
    handles[0] = &handle;
  }
  IrecvArgs(const IrecvArgs&) = delete;
  IrecvArgs& operator=(const IrecvArgs&) = delete;

  ncclResult_t Post(IbMsegRecvComm* comm) {
    return IbMsegIrecv(comm, 1, data, sizes, tags, handles, nullptr, &request);
  }

  ibv_mr mr{};
  IbMsegMrHandle handle{};
  void* data[1];
  size_t sizes[1];
  int tags[1];
  void* handles[1];
  void* request = nullptr;
};

// One receive whose ibv_post_recv fails.
struct FailedReceivePost : OneQpRecvComm {
  explicit FailedReceivePost(uint32_t qpNum) : OneQpRecvComm(qpNum), args(0x6300, 8, 1) { verbs.postRecvErr = EIO; }

  IrecvArgs args;
};

struct TwoQpSendComm {
  static constexpr uint64_t kRemoteAddr = 0x3000;

  TwoQpSendComm(uint32_t firstQpNum, uint32_t secondQpNum, size_t bytes) : data(bytes) {
    verbs.qps[0].qp_num = firstQpNum;
    verbs.qps[1].qp_num = secondQpNum;
    comm.base.isSend = true;
    comm.base.vProps.ndevs = 2;
    comm.base.nqps = 2;
    comm.base.nDataQps = 2;
    comm.base.splitDataOnQps = 1;
    comm.base.recvMatchingScheme = BY_INDEX;
    for (int i = 0; i < 2; ++i) {
      comm.base.qps[i].qp = &verbs.qps[i];
      comm.base.qps[i].devIndex = i;
      comm.base.qps[i].remDevIdx = i;
      comm.base.activeQps[i] = &comm.base.qps[i];
    }
    request.base = &comm.base;
    request.type = NCCL_NET_IB_REQ_SEND;
    request.nreqs = 1; // IbMsegIsend stores the CTS count here before posting
    request.send.data = data.data();
    request.send.size = data.size();
    request.send.weights[0] = request.send.weights[1] = 50;
    request.send.lkeys[0] = 0x10;
    request.send.lkeys[1] = 0x11;
    comm.sendReqs[0][0] = &request;
    comm.ctsFifo[0][0].nreqs = 1;
    comm.ctsFifo[0][0].size = bytes; // the receiver publishes its size; Isend clamps send.size to it
    comm.ctsFifo[0][0].addr = kRemoteAddr;
    comm.ctsFifo[0][0].rkeys[0] = 0x20;
    comm.ctsFifo[0][0].rkeys[1] = 0x21;
  }

  uint64_t DataAddr(size_t offset) const { return reinterpret_cast<uintptr_t>(data.data()) + offset; }

  // Both QPs share one context, so their posts land in one list in post order.
  IbVerbsScene verbs;
  std::vector<unsigned char> data;
  IbMsegSendComm comm{};
  CtsFifoRows cts{&comm};
  IbMsegRequest request{};
};

class NetIbMsegP2pMicrotest : public ::testing::Test {
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
    NetIbMsegResetGlobals();
    ResetTransportNetIbWqeLatFakes();
    ResetTransportNetIbResiliencyFakes();
    ResetSocketFakes();
    ResetNcclFakes();
  }
};

TEST_F(NetIbMsegP2pMicrotest, RecvCommGetQpForCts_RequestIdSelectsFirstQpOnOwningDevice) {
  IbMsegRecvComm comm{};
  IbMsegQp first{};
  IbMsegQp second{};
  comm.base.vProps.ndevs = 2;
  comm.base.activeQps[0] = &first;
  comm.base.activeQps[1] = &second;
  IbMsegQp* selected = nullptr;

  EXPECT_EQ(IbMsegRecvCommGetQpForCts(&comm, /*id=*/3, &selected), ncclSuccess);
  EXPECT_EQ(selected, &second);
}

TEST_F(NetIbMsegP2pMicrotest, RecvCommGetQpForCts_MissingQpReturnsError) {
  IbMsegRecvComm comm{};
  comm.base.vProps.ndevs = 1;
  IbMsegQp* selected = reinterpret_cast<IbMsegQp*>(0x1);

  EXPECT_EQ(IbMsegRecvCommGetQpForCts(&comm, /*id=*/0, &selected), ncclInternalError);
  EXPECT_EQ(selected, nullptr);
}

TEST_F(NetIbMsegP2pMicrotest, RequestRetrieveAsIndex_ValidIndexReturnsPoolEntryAndCapacityIsRejected) {
  std::array<IbMsegRequest, NET_IB_MAX_REQUESTS> requests{};
  IbMsegRequest* request = nullptr;

  EXPECT_EQ(IbMsegRequestRetrieveAsIndex(requests.data(), NET_IB_MAX_REQUESTS - 1, &request), ncclSuccess);
  EXPECT_EQ(request, &requests.back());
  EXPECT_EQ(IbMsegRequestRetrieveAsIndex(requests.data(), NET_IB_MAX_REQUESTS, &request), ncclInternalError);
}

TEST_F(NetIbMsegP2pMicrotest, GetRequest_FirstUnusedEntryResetsCompletionBookkeeping) {
  IbMsegNetCommBase base{};
  base.reqs[0].type = NCCL_NET_IB_REQ_SEND;
  base.reqs[1].type = NCCL_NET_IB_REQ_UNUSED;
  base.reqs[1].events[0] = 3;
  base.reqs[1].devBases[0] = reinterpret_cast<IbMsegNetCommDevBase*>(0x1);
  IbMsegRequest* request = nullptr;

  ASSERT_EQ(IbMsegGetRequest(&base, &request), ncclSuccess);
  EXPECT_EQ(request, &base.reqs[1]);
  EXPECT_EQ(request->base, &base);
  EXPECT_EQ(request->sock, nullptr);
  EXPECT_EQ(request->events[0], 0);
  EXPECT_EQ(request->devBases[0], nullptr);
}

TEST_F(NetIbMsegP2pMicrotest, GetRequest_ExhaustedPoolReturnsErrorAndNull) {
  IbMsegNetCommBase base{};
  for (auto& request : base.reqs) request.type = NCCL_NET_IB_REQ_SEND;
  IbMsegRequest* request = reinterpret_cast<IbMsegRequest*>(0x1);

  EXPECT_EQ(IbMsegGetRequest(&base, &request), ncclInternalError);
  EXPECT_EQ(request, nullptr);
}

TEST_F(NetIbMsegP2pMicrotest, FreeRequest_LiveEntryBecomesReusable) {
  IbMsegRequest request{};
  request.type = NCCL_NET_IB_REQ_RECV;

  EXPECT_EQ(IbMsegFreeRequest(&request), ncclSuccess);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbMsegP2pMicrotest, AddEvent_TracksOwningCompletionQueue) {
  IbMsegSendComm comm{};
  comm.base.isSend = true;
  comm.base.vProps.ndevs = 1;
  comm.devs[0].base.ibDevN = 4;
  IbMsegRequest request{};
  request.base = &comm.base;

  IbMsegAddEvent(&request, 0);

  EXPECT_EQ(request.events[0], 1);
  EXPECT_EQ(request.devBases[0], &comm.devs[0].base);
}

TEST_F(NetIbMsegP2pMicrotest, RequestIsComplete_AnyOfTheFirstFourDeviceCountersKeepsRequestOpen) {
  IbMsegNetCommBase base{};
  IbMsegRequest request{};
  request.base = &base;
  bool complete = false;

  EXPECT_EQ(IbMsegRequestIsComplete(&request, &complete), ncclSuccess);
  EXPECT_TRUE(complete);
  for (int dev = 0; dev < 4; ++dev) {
    SCOPED_TRACE(::testing::Message() << "device " << dev);
    request.events[dev] = 1;
    EXPECT_EQ(IbMsegRequestIsComplete(&request, &complete), ncclSuccess);
    EXPECT_FALSE(complete);
    request.events[dev] = 0;
  }
}

// AICOMRCCL-3441: enable when the completion check covers every device of an 8-device virtual NIC.
TEST_F(NetIbMsegP2pMicrotest, DISABLED_RequestIsComplete_UpperDeviceCountersKeepRequestOpen) {
  IbMsegNetCommBase base{};
  IbMsegRequest request{};
  request.base = &base;
  bool complete = true;

  for (int dev = 4; dev < NCCL_IB_MAX_DEVS_PER_NIC; ++dev) {
    SCOPED_TRACE(::testing::Message() << "device " << dev);
    request.events[dev] = 1;
    EXPECT_EQ(IbMsegRequestIsComplete(&request, &complete), ncclSuccess);
    EXPECT_FALSE(complete);
    request.events[dev] = 0;
  }
}

TEST_F(NetIbMsegP2pMicrotest, RequestComplete_ReceiveReportsRecordedSizesAndReleasesSlot) {
  IbMsegRecvComm comm{};
  IbMsegRequest request{};
  request.base = &comm.base;
  request.type = NCCL_NET_IB_REQ_RECV;
  request.nreqs = 2;
  request.recv.cmplsRecords = &comm.cmplsRecords[0];
  request.recv.cmplsRecords->sizes[0] = 7;
  request.recv.cmplsRecords->sizes[1] = 9;
  int done = 0;
  int sizes[2] = {};

  EXPECT_EQ(IbMsegRequestComplete(&request, &done, sizes), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(sizes[0], 7);
  EXPECT_EQ(sizes[1], 9);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbMsegP2pMicrotest, RequestComplete_SendSlotClearsOnlyAfterLastSendOfMultiReceive) {
  IbMsegSendComm comm{};
  IbMsegRequest& first = comm.base.reqs[0];
  IbMsegRequest& second = comm.base.reqs[1];
  for (IbMsegRequest* request : {&first, &second}) {
    request->base = &comm.base;
    request->type = NCCL_NET_IB_REQ_SEND;
    request->id = 3;
  }
  first.send.size = 11;
  second.send.size = 13;
  comm.sendReqsCnt[3] = 2;
  comm.sendReqs[3][0] = &first;
  comm.sendReqs[3][1] = &second;
  int done = 0;
  int size = 0;

  EXPECT_EQ(IbMsegRequestComplete(&first, &done, &size), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(size, 11);
  EXPECT_EQ(first.type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(comm.sendReqsCnt[3], 1);
  EXPECT_EQ(comm.sendReqs[3][1], &second);

  EXPECT_EQ(IbMsegRequestComplete(&second, &done, &size), ncclSuccess);
  EXPECT_EQ(size, 13);
  EXPECT_EQ(comm.sendReqsCnt[3], 0);
  EXPECT_EQ(comm.sendReqs[3][0], nullptr);
  EXPECT_EQ(comm.sendReqs[3][1], nullptr);
}

TEST_F(NetIbMsegP2pMicrotest, RequestRetrieveFromCompletion_InvalidInputsOrErrorCompletionAreRejected) {
  IbMsegNetCommBase base{};
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  IbMsegRequest* request = nullptr;

  EXPECT_EQ(IbMsegRequestRetrieveFromCompletion(&base, nullptr, &request), ncclInternalError);
  EXPECT_EQ(IbMsegRequestRetrieveFromCompletion(&base, &completion, nullptr), ncclInternalError);
  completion.status = IBV_WC_REM_ACCESS_ERR;
  EXPECT_EQ(IbMsegRequestRetrieveFromCompletion(&base, &completion, &request), ncclInternalError);
}

TEST_F(NetIbMsegP2pMicrotest, RequestRetrieveFromCompletion_DecodesRecvFlushAndSendSlots) {
  IbMsegRecvComm recv{};
  recv.base.isSend = false;
  recv.base.recvMatchingScheme = BY_INDEX;
  recv.recvReqs[5] = &recv.base.reqs[5];
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV;
  completion.wr_id = 5;
  IbMsegRequest* request = nullptr;
  ASSERT_EQ(IbMsegRequestRetrieveFromCompletion(&recv.base, &completion, &request), ncclSuccess);
  EXPECT_EQ(request, &recv.base.reqs[5]);

  completion.opcode = IBV_WC_RDMA_READ;
  completion.wr_id = NCCL_IB_FLUSH_REQ_WR_ID_OFFSET + 6;
  ASSERT_EQ(IbMsegRequestRetrieveFromCompletion(&recv.base, &completion, &request), ncclSuccess);
  EXPECT_EQ(request, &recv.base.reqs[6]);

  IbMsegSendComm send{};
  send.base.isSend = true;
  send.sendReqs[7][0] = &send.base.reqs[7];
  completion.opcode = IBV_WC_RDMA_WRITE;
  completion.wr_id = 0x107;
  ASSERT_EQ(IbMsegRequestRetrieveFromCompletion(&send.base, &completion, &request), ncclSuccess);
  EXPECT_EQ(request, &send.base.reqs[7]);
}

TEST_F(NetIbMsegP2pMicrotest, RequestRetrieveFromCompletion_ByIdUsesImmediateRequestId) {
  IbMsegRecvComm recv{};
  recv.base.isSend = false;
  recv.base.recvMatchingScheme = BY_ID;
  recv.recvReqs[33] = &recv.base.reqs[33];
  ibv_wc completion{};
  completion.status = IBV_WC_SUCCESS;
  completion.opcode = IBV_WC_RECV_RDMA_WITH_IMM;
  completion.imm_data = htobe32(33);
  IbMsegRequest* request = nullptr;

  EXPECT_EQ(IbMsegRequestRetrieveFromCompletion(&recv.base, &completion, &request), ncclSuccess);
  EXPECT_EQ(request, &recv.base.reqs[33]);
}

TEST_F(NetIbMsegP2pMicrotest, CompletionEvent_ByIdAggregatesBytesAndCompletionReportsTotal) {
  IbMsegRecvComm comm{};
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

  ASSERT_EQ(IbMsegCompletionEventProcess(&comm.base, &completion, 0), ncclSuccess);
  ASSERT_EQ(IbMsegCompletionEventProcess(&comm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(request.recv.aggSize, 10);
  EXPECT_EQ(request.events[0], 0);
  int done = 0;
  int size = 0;
  EXPECT_EQ(IbMsegRequestComplete(&request, &done, &size), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(size, 10);
}

TEST_F(NetIbMsegP2pMicrotest, IsendAndIrecv_NotReadyRejectAndClearRequest) {
  IbMsegSendComm send{};
  IbMsegRecvComm recv{};
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegIsend(&send, nullptr, 0, 0, nullptr, nullptr, &request), ncclInternalError);
  EXPECT_EQ(request, nullptr);
  request = reinterpret_cast<void*>(0x1);
  EXPECT_EQ(IbMsegIrecv(&recv, 0, nullptr, nullptr, nullptr, nullptr, nullptr, &request), ncclInternalError);
  EXPECT_EQ(request, nullptr);
}

TEST_F(NetIbMsegP2pMicrotest, Irecv_ReceiveCountAboveCapacityIsRejected) {
  OneQpRecvComm recv(56);
  // Real receives, so a missing guard publishes a request instead of faulting on a null array.
  constexpr int kTooMany = NCCL_NET_IB_MAX_RECVS + 1;
  ibv_mr mr{};
  std::array<IbMsegMrHandle, kTooMany> mrHandles{};
  std::array<void*, kTooMany> data{};
  std::array<size_t, kTooMany> sizes{};
  std::array<int, kTooMany> tags{};
  std::array<void*, kTooMany> handles{};
  for (int i = 0; i < kTooMany; ++i) {
    mrHandles[i].mrs[0] = &mr;
    data[i] = reinterpret_cast<void*>(0x9000 + 0x100 * i);
    sizes[i] = 8;
    tags[i] = i;
    handles[i] = &mrHandles[i];
  }
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegIrecv(&recv.comm, kTooMany, data.data(), sizes.data(), tags.data(), handles.data(), nullptr,
                          &request),
            ncclInternalError);
  EXPECT_EQ(request, reinterpret_cast<void*>(0x1));
  EXPECT_TRUE(recv.verbs.postRecvQps.empty());
  EXPECT_TRUE(recv.verbs.posted.empty());
}

TEST_F(NetIbMsegP2pMicrotest, Iflush_DisabledOrAllEmptyInputPostsNothing) {
  FlushRecvComm flush(57, /*withScratch=*/false);
  void* data[] = {reinterpret_cast<void*>(0x9100), reinterpret_cast<void*>(0x9200)};
  int sizes[] = {1, 0};
  void* handles[] = {&flush.handle, &flush.handle};
  void* request = reinterpret_cast<void*>(0x1);

  flush.comm.flushEnabled = 0;
  EXPECT_EQ(IbMsegIflush(&flush.comm, 2, data, sizes, handles, &request), ncclSuccess);
  EXPECT_EQ(request, reinterpret_cast<void*>(0x1));
  flush.comm.flushEnabled = 1;
  sizes[0] = 0;
  EXPECT_EQ(IbMsegIflush(&flush.comm, 2, data, sizes, handles, &request), ncclSuccess);
  EXPECT_EQ(request, reinterpret_cast<void*>(0x1));
  EXPECT_TRUE(flush.verbs.posted.empty());
}

TEST_F(NetIbMsegP2pMicrotest, PostFifo_OwningDeviceSlotPostsSignaledInlineCts) {
  IbVerbsScene verbs("mlx5_0", 17);
  IbMsegRecvComm comm{};
  CtsFifoRows cts(&comm);
  comm.base.vProps.ndevs = 1;
  comm.base.activeQps[0] = &comm.base.qps[0];
  comm.base.qps[0].qp = &verbs.qps[0];
  comm.base.qps[0].devIndex = 0;
  comm.base.qps[0].remDevIdx = 0;
  comm.base.remDevs[0].rkey = 0x44;
  comm.remCtsFifo.addr = 0x1000;
  comm.remCtsFifo.flags = IBV_SEND_INLINE;
  comm.devs[0].sge.lkey = 0x55;
  IbMsegRequest request{};
  request.base = &comm.base;
  request.id = 0;
  request.nreqs = 1;

  ASSERT_EQ(IbMsegPostFifo(&comm, &request, /*slot=*/0), ncclSuccess);
  ASSERT_EQ(verbs.posted.size(), 1u);
  EXPECT_EQ(verbs.posted[0].qpNum, 17u);
  EXPECT_EQ(verbs.posted[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_NE(verbs.posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_NE(verbs.posted[0].sendFlags & IBV_SEND_INLINE, 0);
  EXPECT_EQ(verbs.posted[0].wrId, 0u);
  EXPECT_EQ(verbs.posted[0].remoteAddress, 0x1000u);
  EXPECT_EQ(verbs.posted[0].rkey, 0x44u);
  EXPECT_EQ(verbs.posted[0].sge.length, sizeof(IbMsegSendFifo));
}

TEST_F(NetIbMsegP2pMicrotest, PostFifo_DifferentSlotIsUnsignaledUnlessResiliencyIsEnabled) {
  IbVerbsScene verbs("mlx5_0", 18);
  IbMsegRecvComm comm{};
  CtsFifoRows cts(&comm);
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/false);
  comm.base.remDevs[0].rkey = 0x44;
  comm.remCtsFifo.addr = 0x1000;
  IbMsegRequest request{};
  request.base = &comm.base;
  request.id = 0;
  request.nreqs = 1;
  const uint64_t slotOneRemote = 0x1000 + NCCL_NET_IB_MAX_RECVS * sizeof(struct IbMsegSendFifo);
  const uint64_t slotOneLocal = reinterpret_cast<uintptr_t>(comm.remCtsFifo.elems[1]);

  ASSERT_EQ(IbMsegPostFifo(&comm, &request, /*slot=*/1), ncclSuccess);
  ASSERT_EQ(verbs.posted.size(), 1u);
  EXPECT_EQ(verbs.posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(verbs.posted[0].remoteAddress, slotOneRemote);
  EXPECT_EQ(verbs.posted[0].sge.addr, slotOneLocal);

  verbs.posted.clear();
  IbMsegResiliency resiliency{};
  comm.base.resiliency = &resiliency;
  ASSERT_EQ(IbMsegPostFifo(&comm, &request, /*slot=*/1), ncclSuccess);
  ASSERT_EQ(verbs.posted.size(), 1u);
  EXPECT_NE(verbs.posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(verbs.posted[0].wrId, 1u);
  EXPECT_EQ(verbs.posted[0].remoteAddress, slotOneRemote);
  EXPECT_EQ(verbs.posted[0].sge.addr, slotOneLocal);
}

TEST_F(NetIbMsegP2pMicrotest, MultiSend_SmallPayloadUsesOnlyRotatedActiveQp) {
  TwoQpSendComm send(21, 22, /*bytes=*/64);

  ASSERT_EQ(IbMsegMultiSend(&send.comm, 0), ncclSuccess);
  ASSERT_EQ(send.verbs.posted.size(), 2u);
  EXPECT_EQ(send.verbs.posted[0].qpNum, 21u);
  EXPECT_EQ(send.verbs.posted[0].sge.length, send.data.size());
  EXPECT_EQ(send.verbs.posted[1].qpNum, 22u);
  EXPECT_EQ(send.verbs.posted[1].sge.length, 0u);
  EXPECT_EQ(send.verbs.posted[0].opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
  EXPECT_EQ(be32toh(send.verbs.posted[0].immData), send.data.size());
}

TEST_F(NetIbMsegP2pMicrotest, MultiSend_WeightedPayloadAlignsChunksAndClampsFinalBytes) {
  TwoQpSendComm send(31, 32, /*bytes=*/300);
  send.request.send.weights[0] = 75;
  send.request.send.weights[1] = 25;
  // Crossed so that a remote-indexed rkey and a local-indexed lkey cannot pass for each other.
  send.comm.base.qps[0].remDevIdx = 1;
  send.comm.base.qps[1].remDevIdx = 0;

  ASSERT_EQ(IbMsegMultiSend(&send.comm, 0), ncclSuccess);
  const auto& posted = send.verbs.posted;
  ASSERT_EQ(posted.size(), 2u);
  EXPECT_EQ(posted[0].sge.length, 256u);
  EXPECT_EQ(posted[1].sge.length, 44u);
  EXPECT_EQ(posted[0].sge.length + posted[1].sge.length, send.data.size());
  EXPECT_EQ(posted[0].sge.addr, send.DataAddr(0));
  EXPECT_EQ(posted[0].sge.lkey, 0x10u);
  EXPECT_EQ(posted[0].rkey, 0x21u);
  EXPECT_EQ(posted[1].sge.addr, send.DataAddr(256));
  EXPECT_EQ(posted[1].sge.lkey, 0x11u);
  EXPECT_EQ(posted[1].rkey, 0x20u);
  EXPECT_EQ(posted[1].remoteAddress, TwoQpSendComm::kRemoteAddr + 256);
}

TEST_F(NetIbMsegP2pMicrotest, MultiSend_PayloadAtSplitThresholdUsesWeightedPath) {
  TwoQpSendComm send(33, 34, /*bytes=*/256);
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "RCCL_IB_SPLIT_DATA_THRESHOLD") == 0 ? int64_t{256} : fallback;
  });

  ASSERT_EQ(IbMsegMultiSend(&send.comm, 0), ncclSuccess);
  ASSERT_EQ(send.verbs.posted.size(), 2u);
  EXPECT_EQ(send.verbs.posted[0].sge.length, 128u);
  EXPECT_EQ(send.verbs.posted[1].sge.length, 128u);
}

TEST_F(NetIbMsegP2pMicrotest, MultiSend_RequestRotationSelectsOwningActiveQp) {
  TwoQpSendComm send(35, 36, /*bytes=*/64);
  send.comm.base.nDataQps = 1;
  send.comm.base.splitDataOnQps = 0;
  send.request.id = 1;

  ASSERT_EQ(IbMsegMultiSend(&send.comm, 0), ncclSuccess);
  ASSERT_EQ(send.verbs.posted.size(), 1u);
  EXPECT_EQ(send.verbs.posted[0].qpNum, 36u);
  EXPECT_EQ(send.verbs.posted[0].sge.length, send.data.size());
}

TEST_F(NetIbMsegP2pMicrotest, Isend_MatchedReceivePublishesClampedRequestAndAdvancesFifo) {
  IbVerbsScene verbs("mlx5_0", 41);
  IbMsegSendComm comm{};
  CtsFifoRows cts(&comm);
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/true);
  comm.ctsFifo[0][0].idx = 1;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].tag = 7;
  comm.ctsFifo[0][0].size = 32;
  comm.ctsFifo[0][0].addr = 0x4000;
  comm.ctsFifo[0][0].rkeys[0] = 0x42;
  ibv_mr mr{};
  mr.lkey = 0x55;
  IbMsegMrHandle handle{};
  handle.mrs[0] = &mr;
  std::array<unsigned char, 64> data{};
  void* request = nullptr;

  ASSERT_EQ(IbMsegIsend(&comm, data.data(), data.size(), 7, &handle, nullptr, &request), ncclSuccess);
  auto* ibRequest = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(ibRequest, nullptr);
  EXPECT_EQ(ibRequest->type, NCCL_NET_IB_REQ_SEND);
  EXPECT_EQ(ibRequest->send.size, 32);
  EXPECT_EQ(ibRequest->events[0], 1);
  EXPECT_EQ(comm.base.fifoHead, 1u);
  EXPECT_EQ(comm.sendReqs[0][0], ibRequest);
  EXPECT_EQ(verbs.posted.size(), 1u);
}

TEST_F(NetIbMsegP2pMicrotest, Isend_MultiReceiveDefersUntilEveryTagMatchesThenPostsBatch) {
  IbVerbsScene verbs("mlx5_0", 42);
  IbMsegSendComm comm{};
  CtsFifoRows cts(&comm);
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
  IbMsegMrHandle handle{};
  handle.mrs[0] = &mr;
  std::array<unsigned char, 64> firstData{};
  std::array<unsigned char, 32> secondData{};
  void* firstRequest = nullptr;
  void* secondRequest = nullptr;

  ASSERT_EQ(IbMsegIsend(&comm, firstData.data(), firstData.size(), 7, &handle, nullptr, &firstRequest), ncclSuccess);
  EXPECT_NE(firstRequest, nullptr);
  EXPECT_EQ(comm.sendReqsCnt[0], 1);
  EXPECT_EQ(comm.base.fifoHead, 0u);
  EXPECT_TRUE(verbs.posted.empty());

  ASSERT_EQ(IbMsegIsend(&comm, secondData.data(), secondData.size(), 8, &handle, nullptr, &secondRequest),
            ncclSuccess);
  EXPECT_NE(secondRequest, nullptr);
  EXPECT_EQ(comm.sendReqsCnt[0], 2);
  EXPECT_EQ(comm.base.fifoHead, 1u);
  ASSERT_EQ(verbs.posted.size(), 3u);
  EXPECT_EQ(verbs.posted[0].sge.length, firstData.size());
  EXPECT_EQ(verbs.posted[1].sge.length, secondData.size());
  EXPECT_EQ(verbs.posted[2].sge.length, 2 * sizeof(int));
}

TEST_F(NetIbMsegP2pMicrotest, Isend_PostFailureWithEventsReturnsFailedRequestForDrain) {
  IbVerbsScene verbs("mlx5_0", 43);
  IbMsegSendComm comm{};
  CtsFifoRows cts(&comm);
  InitOneQpBase(&comm.base, &verbs, /*isSend=*/true);
  comm.ctsFifo[0][0].idx = 1;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].tag = 7;
  comm.ctsFifo[0][0].size = 16;
  comm.ctsFifo[0][0].addr = 0x4400;
  comm.ctsFifo[0][0].rkeys[0] = 0x45;
  ibv_mr mr{};
  IbMsegMrHandle handle{};
  handle.mrs[0] = &mr;
  verbs.postSendErr = EIO;
  void* request = nullptr;

  EXPECT_EQ(IbMsegIsend(&comm, nullptr, 16, 7, &handle, nullptr, &request), ncclSuccess);
  auto* failed = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(failed, nullptr);
  EXPECT_EQ(failed->type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(failed->events[0], 1);
  EXPECT_GT(comm.base.stats.fatalErrorCount, 0);
}

TEST_F(NetIbMsegP2pMicrotest, Isend_ReceiveNotPublishedOrTagMismatchReturnsWithoutRequest) {
  IbMsegSendComm comm{};
  CtsFifoRows cts(&comm);
  comm.base.ready = 1;
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegIsend(&comm, nullptr, 0, 5, nullptr, nullptr, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);

  comm.ctsFifo[0][0].idx = 1;
  comm.ctsFifo[0][0].nreqs = 1;
  comm.ctsFifo[0][0].tag = 6;
  request = reinterpret_cast<void*>(0x1);
  EXPECT_EQ(IbMsegIsend(&comm, nullptr, 0, 5, nullptr, nullptr, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);
}

TEST_F(NetIbMsegP2pMicrotest, Irecv_ValidRequestPostsReceiveAndCtsMetadata) {
  OneQpRecvComm recv(51);
  recv.comm.base.remDevs[0].rkey = 0x61;
  recv.comm.remCtsFifo.addr = 0x5000;
  IrecvArgs args(0x6000, 128, 9, /*rkey=*/0x62);

  ASSERT_EQ(args.Post(&recv.comm), ncclSuccess);
  auto* ibRequest = static_cast<IbMsegRequest*>(args.request);
  ASSERT_NE(ibRequest, nullptr);
  EXPECT_EQ(ibRequest->type, NCCL_NET_IB_REQ_RECV);
  EXPECT_EQ(ibRequest->events[0], 1);
  EXPECT_EQ(recv.verbs.postRecvQps.size(), 1u);
  EXPECT_EQ(recv.verbs.posted.size(), 1u);
  const auto& cts = recv.comm.remCtsFifo.elems[0][0];
  EXPECT_EQ(cts.addr, 0x6000u);
  EXPECT_EQ(cts.rkeys[0], 0x62u);
  EXPECT_EQ(cts.nreqs, 1u);
  EXPECT_EQ(cts.size, 128u);
  EXPECT_EQ(cts.tag, 9u);
  EXPECT_EQ(cts.idx, 1u);
  EXPECT_EQ(recv.comm.base.fifoHead, 1u);
}

TEST_F(NetIbMsegP2pMicrotest, Irecv_MultiReceivePublishesEveryDescriptorInOneNonInlineCts) {
  OneQpRecvComm recv(53);
  auto& comm = recv.comm;
  comm.base.remDevs[0].rkey = 0x63;
  // The QP's max_inline_data fits one CTS entry, so a two-entry CTS must not be inlined.
  comm.remCtsFifo.flags = IBV_SEND_INLINE;
  ibv_mr firstMr{};
  ibv_mr secondMr{};
  firstMr.rkey = 0x64;
  secondMr.rkey = 0x65;
  IbMsegMrHandle firstHandle{};
  IbMsegMrHandle secondHandle{};
  firstHandle.mrs[0] = &firstMr;
  secondHandle.mrs[0] = &secondMr;
  void* data[] = {reinterpret_cast<void*>(0x6100), reinterpret_cast<void*>(0x6200)};
  size_t sizes[] = {11, 13};
  int tags[] = {3, 4};
  void* handles[] = {&firstHandle, &secondHandle};
  std::fill(std::begin(comm.cmplsRecords[0].sizes), std::end(comm.cmplsRecords[0].sizes), -1);
  void* request = nullptr;

  ASSERT_EQ(IbMsegIrecv(&comm, 2, data, sizes, tags, handles, nullptr, &request), ncclSuccess);
  auto* ibRequest = static_cast<IbMsegRequest*>(request);
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
  ASSERT_EQ(recv.verbs.posted.size(), 1u);
  EXPECT_EQ(recv.verbs.posted[0].sendFlags & IBV_SEND_INLINE, 0);
  EXPECT_EQ(recv.verbs.posted[0].sge.length, 2 * sizeof(IbMsegSendFifo));
}

TEST_F(NetIbMsegP2pMicrotest, Irecv_ReceivePostFailurePropagatesError) {
  FailedReceivePost post(54);

  EXPECT_EQ(post.args.Post(&post.comm), ncclSystemError);
  EXPECT_EQ(post.verbs.postRecvQps.size(), 1u);
}

// AICOMRCCL-3126: enable when receive-post failure releases the allocated slot.
TEST_F(NetIbMsegP2pMicrotest, DISABLED_Irecv_ReceivePostFailureReleasesAllocatedSlot) {
  FailedReceivePost post(55);

  ASSERT_EQ(post.args.Post(&post.comm), ncclSystemError);
  EXPECT_EQ(post.args.request, nullptr);
  EXPECT_EQ(post.comm.base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbMsegP2pMicrotest, Irecv_PrepostedModeSkipsReceivePostButPublishesCts) {
  OneQpRecvComm recv(52);
  recv.comm.prepostReceiveWorkRequests = true;
  recv.comm.base.remDevs[0].rkey = 1;
  IrecvArgs args(0x7000, 16, 1, /*rkey=*/2);

  ASSERT_EQ(args.Post(&recv.comm), ncclSuccess);
  EXPECT_TRUE(recv.verbs.postRecvQps.empty());
  EXPECT_EQ(recv.verbs.posted.size(), 1u);
}

TEST_F(NetIbMsegP2pMicrotest, Iflush_FallbackPostsSignaledReadFromLastNonzeroBuffer) {
  FlushRecvComm flush(61, /*withScratch=*/false);
  flush.receiveMr.rkey = 0x72;
  void* data[] = {reinterpret_cast<void*>(0x7100), reinterpret_cast<void*>(0x7200)};
  int sizes[] = {0, 4};
  void* handles[] = {&flush.handle, &flush.handle};
  void* request = nullptr;

  ASSERT_EQ(IbMsegIflush(&flush.comm, 2, data, sizes, handles, &request), ncclSuccess);
  auto* ibRequest = static_cast<IbMsegRequest*>(request);
  ASSERT_NE(ibRequest, nullptr);
  ASSERT_EQ(flush.verbs.posted.size(), 1u);
  EXPECT_EQ(flush.verbs.posted[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(flush.verbs.posted[0].remoteAddress, 0x7200u);
  EXPECT_EQ(flush.verbs.posted[0].rkey, 0x72u);
  EXPECT_NE(flush.verbs.posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(ibRequest->events[0], 1);
}

TEST_F(NetIbMsegP2pMicrotest, Iflush_GpuScratchPostsWriteBeforeRead) {
  FlushRecvComm flush(62, /*withScratch=*/true);
  void* data[] = {reinterpret_cast<void*>(0x8100)};
  int sizes[] = {4};
  void* handles[] = {&flush.handle};
  void* request = nullptr;

  ASSERT_EQ(IbMsegIflush(&flush.comm, 1, data, sizes, handles, &request), ncclSuccess);
  const auto& posted = flush.verbs.posted;
  ASSERT_EQ(posted.size(), 2u);
  EXPECT_EQ(posted[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(posted[0].remoteAddress, reinterpret_cast<uint64_t>(&flush.scratch));
  EXPECT_EQ(posted[1].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(posted[1].remoteAddress, reinterpret_cast<uint64_t>(&flush.scratch));
  EXPECT_EQ(posted[1].rkey, 0x82u);
  EXPECT_NE(posted[1].sendFlags & IBV_SEND_SIGNALED, 0);
}

TEST_F(NetIbMsegP2pMicrotest, Iflush_RelaxedOrderingFenceOffReadsLastReceiveDespiteScratch) {
  FlushRecvComm flush(65, /*withScratch=*/true);
  ScopedHook params(g_loadParam, [](const char* name, int64_t fallback) {
    return std::strcmp(name, "RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING") == 0 ? int64_t{0} : fallback;
  });
  void* data[] = {reinterpret_cast<void*>(0x8400)};
  int sizes[] = {4};
  void* handles[] = {&flush.handle};
  void* request = nullptr;

  ASSERT_EQ(IbMsegIflush(&flush.comm, 1, data, sizes, handles, &request), ncclSuccess);
  ASSERT_EQ(flush.verbs.posted.size(), 1u);
  EXPECT_EQ(flush.verbs.posted[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(flush.verbs.posted[0].remoteAddress, 0x8400u);
  EXPECT_EQ(flush.verbs.posted[0].rkey, 0x83u);
}

TEST_F(NetIbMsegP2pMicrotest, Iflush_FirstPostFailureReleasesRequestAndReturnsError) {
  FlushRecvComm flush(63, /*withScratch=*/true);
  flush.verbs.postSendErr = EIO;
  void* data[] = {reinterpret_cast<void*>(0x8200)};
  int sizes[] = {4};
  void* handles[] = {&flush.handle};
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(IbMsegIflush(&flush.comm, 1, data, sizes, handles, &request), ncclSystemError);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(flush.comm.base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_GT(flush.comm.base.stats.fatalErrorCount, 0);
}

TEST_F(NetIbMsegP2pMicrotest, Iflush_GpuScratchChainsWriteAndReadInOnePost) {
  FlushRecvComm flush(64, /*withScratch=*/true);
  flush.verbs.postSendFailOnCall = 2;
  void* data[] = {reinterpret_cast<void*>(0x8300)};
  int sizes[] = {4};
  void* handles[] = {&flush.handle};
  void* request = nullptr;

  ASSERT_EQ(IbMsegIflush(&flush.comm, 1, data, sizes, handles, &request), ncclSuccess);
  EXPECT_NE(request, nullptr);
  EXPECT_EQ(flush.verbs.postSendQps.size(), 1u);
  ASSERT_EQ(flush.verbs.sends.size(), 1u);
  const auto& chain = flush.verbs.sends[0].chain;
  ASSERT_EQ(chain.size(), 2u);
  EXPECT_EQ(chain[0].wr.opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(chain[0].wr.send_flags & IBV_SEND_SIGNALED, 0u);
  EXPECT_EQ(chain[1].wr.opcode, IBV_WR_RDMA_READ);
  EXPECT_NE(chain[1].wr.send_flags & IBV_SEND_SIGNALED, 0u);
}

TEST_F(NetIbMsegP2pMicrotest, Test_NoCompletionsReturnsNotDoneWithoutSpinning) {
  FlushRecvComm flush(100, /*withScratch=*/false);
  auto& request = flush.PendingFlush();
  int done = -1;

  EXPECT_EQ(IbMsegTest(&request, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(flush.verbs.polledCqs, std::vector<ibv_cq*>{&flush.verbs.cq});
}

TEST_F(NetIbMsegP2pMicrotest, Test_PollFailureMarksRequestConnectionAndDeviceFatal) {
  FlushRecvComm flush(100, /*withScratch=*/false);
  flush.comm.devs[0].base.ibDevN = 0;
  auto& request = flush.PendingFlush();
  flush.verbs.pollErr = -1;
  int done = 0;

  EXPECT_EQ(IbMsegTest(&request, &done, nullptr), ncclSystemError);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_GT(flush.comm.base.stats.fatalErrorCount, 0);
  EXPECT_GT(IbMsegDevs[0].stats.fatalErrorCount, 0);
}

TEST_F(NetIbMsegP2pMicrotest, Test_SuccessfulCompletionDrainsAndFinishesRequest) {
  FlushRecvComm flush(100, /*withScratch=*/false);
  auto& request = flush.PendingFlush();
  flush.verbs.Complete(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET, IBV_WC_RDMA_READ, 100);
  int done = 0;

  EXPECT_EQ(IbMsegTest(&request, &done, nullptr), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(request.type, NCCL_NET_IB_REQ_UNUSED);
}

TEST_F(NetIbMsegP2pMicrotest, Test_ErrorCompletionWithoutResiliencyReturnsRemoteError) {
  FlushRecvComm flush(100, /*withScratch=*/false);
  flush.comm.devs[0].base.pd = &flush.verbs.pd;
  auto& request = flush.PendingFlush();
  flush.verbs.Complete(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET, IBV_WC_RDMA_READ, 100, IBV_WC_REM_ACCESS_ERR);
  int done = 0;

  EXPECT_EQ(IbMsegTest(&request, &done, nullptr), ncclRemoteError);
  EXPECT_EQ(done, 0);
}

TEST_F(NetIbMsegP2pMicrotest, RequestIsComplete_ResiliencyCanCompleteOutstandingRequest) {
  IbMsegNetCommBase base{};
  IbMsegResiliency resiliency{};
  base.resiliency = &resiliency;
  IbMsegRequest request{};
  request.base = &base;
  request.events[0] = 1;
  ScopedHook complete(g_ncclIbResiliencyRequestIsComplete, [](IbMsegRequest*, bool* isComplete) {
    *isComplete = true;
    return ncclSuccess;
  });
  bool isComplete = false;

  EXPECT_EQ(IbMsegRequestIsComplete(&request, &isComplete), ncclSuccess);
  EXPECT_TRUE(isComplete);
  EXPECT_EQ(complete.calls, 1);
}

TEST_F(NetIbMsegP2pMicrotest, CompletionEvent_SendConsumesEverySiblingEvent) {
  IbMsegSendComm comm{};
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

  EXPECT_EQ(IbMsegCompletionEventProcess(&comm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(first.events[0], 0);
  EXPECT_EQ(second.events[0], 0);
}

TEST_F(NetIbMsegP2pMicrotest, CompletionEvent_IndexMatchedReceiveRecordsImmediateSize) {
  IbMsegRecvComm comm{};
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

  EXPECT_EQ(IbMsegCompletionEventProcess(&comm.base, &completion, 0), ncclSuccess);
  EXPECT_EQ(request.recv.cmplsRecords->sizes[0], 123);
  EXPECT_EQ(request.events[0], 0);
}

TEST_F(NetIbMsegP2pMicrotest, CompletionEvent_SpeedUpdateClearsPendingState) {
  IbMsegRecvComm comm{};
  comm.postedSpeedUpdate = true;
  ibv_wc completion{};
  completion.wr_id = NCCL_IB_SPEED_UPDATE_WR_ID;

  EXPECT_EQ(IbMsegCompletionEventProcess(&comm.base, &completion, 0), ncclSuccess);
  EXPECT_FALSE(comm.postedSpeedUpdate);
}

TEST_F(NetIbMsegP2pMicrotest, PostSpeedUpdate_ChangedSpeedPostsSignaledSnapshot) {
  OneQpRecvComm recv(81);
  auto& comm = recv.comm;
  comm.base.vProps.devs[0] = 0;
  comm.base.speedChangeCounter = 0;
  comm.remSpeedBufAddr = 0xb000;
  comm.base.remDevs[0].remSpeedBufRkey = 0xb1;
  ibv_mr speedMr{};
  speedMr.lkey = 0xb2;
  comm.devs[0].speedUpdateMr = &speedMr;
  COMPILER_ATOMIC_STORE(&IbMsegDevs[0].currSpeed, uint64_t{200000}, std::memory_order_relaxed);
  COMPILER_ATOMIC_STORE(&IbMsegSpeedChangeCounter, uint64_t{1}, std::memory_order_release);

  ASSERT_EQ(IbMsegPostSpeedUpdateToRemote(&comm), ncclSuccess);
  ASSERT_EQ(recv.verbs.posted.size(), 1u);
  EXPECT_EQ(recv.verbs.posted[0].wrId, NCCL_IB_SPEED_UPDATE_WR_ID);
  EXPECT_EQ(recv.verbs.posted[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_NE(recv.verbs.posted[0].sendFlags & IBV_SEND_SIGNALED, 0);
  EXPECT_EQ(recv.verbs.posted[0].remoteAddress, 0xb000u);
  EXPECT_EQ(recv.verbs.posted[0].rkey, 0xb1u);
  EXPECT_TRUE(comm.postedSpeedUpdate);
  EXPECT_EQ(comm.speedUpdateBuf.speedGbps[0], 200);
}

TEST_F(NetIbMsegP2pMicrotest, PostSpeedUpdate_UnchangedPendingOrAlreadySentStatePostsNothing) {
  OneQpRecvComm recv(82);
  auto& comm = recv.comm;
  comm.base.vProps.devs[0] = 0;
  comm.remSpeedBufAddr = 0xc000;
  comm.base.remDevs[0].remSpeedBufRkey = 0xc1;
  ibv_mr speedMr{};
  speedMr.lkey = 0xc2;
  comm.devs[0].speedUpdateMr = &speedMr;
  COMPILER_ATOMIC_STORE(&IbMsegDevs[0].currSpeed, uint64_t{200000}, std::memory_order_relaxed);

  comm.base.speedChangeCounter = 7;
  COMPILER_ATOMIC_STORE(&IbMsegSpeedChangeCounter, uint64_t{7}, std::memory_order_release);
  EXPECT_EQ(IbMsegPostSpeedUpdateToRemote(&comm), ncclSuccess);
  EXPECT_TRUE(recv.verbs.posted.empty());

  comm.base.speedChangeCounter = 0;
  comm.postedSpeedUpdate = true;
  COMPILER_ATOMIC_STORE(&IbMsegSpeedChangeCounter, uint64_t{8}, std::memory_order_release);
  EXPECT_EQ(IbMsegPostSpeedUpdateToRemote(&comm), ncclSuccess);
  EXPECT_TRUE(recv.verbs.posted.empty());

  // The counter moved, but the speeds match the last snapshot sent: take the counter, post nothing.
  comm.postedSpeedUpdate = false;
  comm.lastSentSpeeds[0] = 200;
  COMPILER_ATOMIC_STORE(&IbMsegSpeedChangeCounter, uint64_t{9}, std::memory_order_release);
  EXPECT_EQ(IbMsegPostSpeedUpdateToRemote(&comm), ncclSuccess);
  EXPECT_TRUE(recv.verbs.posted.empty());
  EXPECT_EQ(comm.base.speedChangeCounter, 9u);
}

// Closing a send comm releases each read layout's MR and buffer and clears the entry.
TEST_F(NetIbMsegP2pMicrotest, LayoutCacheFree_DeregistersEachReadLayoutAndClearsItsEntry) {
  IbMsegSendComm comm{};
  rcclIbLayoutCache cache{};
  comm.layoutCache = &cache;
  ibv_mr mr{};
  rcclIbLayoutCacheEntry& e = cache.e[3];
  e.buf = std::malloc(64);
  ASSERT_NE(e.buf, nullptr);
  e.cap = 64;
  e.mr = &mr;
  e.addr = 0x1000;
  e.gen = 2;
  e.state = RCCL_IB_LAYOUT_READY;
  ScopedHook dereg(g_ibvDeregMr, [&mr](ibv_mr* value) {
    EXPECT_EQ(value, &mr);
    return ncclSuccess;
  });

  EXPECT_EQ(rcclIbSendLayoutCacheFree(&comm), ncclSuccess);
  EXPECT_EQ(dereg.calls, 1);
  EXPECT_EQ(e.buf, nullptr);
  EXPECT_EQ(e.mr, nullptr);
  EXPECT_EQ(e.addr, 0u);
  EXPECT_EQ(e.state, RCCL_IB_LAYOUT_EMPTY);
}

// The data-path tests read these counters to prove a send took the segmented builder.
TEST_F(NetIbMsegP2pMicrotest, SendCommSegStats_ReadsTheCommCountersAndRejectsNull) {
  IbMsegSendComm comm{};
  rcclIbSegStatsAdd(&comm.segStats, 3, false);
  rcclIbSegStatsAdd(&comm.segStats, 2, true);
  rcclIbSegStats s{};
  ASSERT_EQ(rcclIbSendCommSegStats(&comm, &s), ncclSuccess);
  EXPECT_EQ(s.posts, 2u);
  EXPECT_EQ(s.wrs, 5u);
  EXPECT_EQ(s.laterSegPosts, 1u);
  EXPECT_EQ(rcclIbSendCommSegStats(nullptr, &s), ncclInvalidArgument);
  EXPECT_EQ(rcclIbSendCommSegStats(&comm, nullptr), ncclInvalidArgument);
}

}  // namespace
