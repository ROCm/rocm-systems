/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/net_ib_multiseg/p2p.cc and p2p.h: net_ib/net-ib-p2p-test.cc against the
 * fork (derived_transport_tests.cmake), then the tests the fork changed.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdlib>

#include "fakes/ibvwrap_fakes.h"

#undef NET_IB_P2P_CC_PATH
#define NET_IB_P2P_CC_PATH "transport/net_ib_multiseg/shim/p2p.cc"
#define NetIbP2pMicrotest NetIbMsegP2pMicrotest
#define TwoQpSendComm DevelopTwoQpSendComm
#include "transport/derived_transport_tests_begin.h"
#include "transport/net_ib/net-ib-p2p-test.cc"
#include "transport/derived_transport_tests_end.h"
#undef TwoQpSendComm

namespace {

// The legacy CTS FIFO (ctsFifo, or remCtsFifo.elems on a receiver) is a separate block; back it with zeroed rows.
struct CtsFifoRows {
  explicit CtsFifoRows(IbMsegSendComm* comm) { comm->ctsFifo = rows; }
  explicit CtsFifoRows(IbMsegRecvComm* comm) { comm->remCtsFifo.elems = rows; }
  CtsFifoRows(const CtsFifoRows&) = delete;
  CtsFifoRows& operator=(const CtsFifoRows&) = delete;
  IbMsegSendFifo rows[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS]{};
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
  CtsFifoRows rows(&recv.comm);
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
  CtsFifoRows rows(&recv.comm);
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

TEST_F(NetIbMsegP2pMicrotest, Irecv_PrepostedModeSkipsReceivePostButPublishesCts) {
  OneQpRecvComm recv(52);
  CtsFifoRows rows(&recv.comm);
  recv.comm.prepostReceiveWorkRequests = true;
  recv.comm.base.remDevs[0].rkey = 1;
  IrecvArgs args(0x7000, 16, 1, /*rkey=*/2);

  ASSERT_EQ(args.Post(&recv.comm), ncclSuccess);
  EXPECT_TRUE(recv.verbs.postRecvQps.empty());
  EXPECT_EQ(recv.verbs.posted.size(), 1u);
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
