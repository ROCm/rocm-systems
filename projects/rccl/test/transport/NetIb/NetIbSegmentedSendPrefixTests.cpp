/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Host-only tests for a failed post in the classic NET/IB segmented send
 * (ncclIbMultiSendSegmented). A fake ibv_post_send rejects a chain whole or
 * only its signaled tail; no RDMA device is needed.
 */

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>

#include "net_ib/p2p.h"

namespace {

enum class PostFail { None, Head, Tail };

struct FakePost {
  uint32_t failQpNum = 0;
  PostFail mode = PostFail::None;
  int calls = 0;
  int failedCallWrs = 0;
  int failedCallSignaled = 0;
};
FakePost g_post;

extern "C" int FakePostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr) {
  g_post.calls++;
  if (g_post.mode == PostFail::None || qp->qp_num != g_post.failQpNum) return 0;
  struct ibv_send_wr* tail = wr;
  for (struct ibv_send_wr* w = wr; w != nullptr; w = w->next) {
    g_post.failedCallWrs++;
    if (w->send_flags & IBV_SEND_SIGNALED) g_post.failedCallSignaled++;
    tail = w;
  }
  // Tail mode: the NIC took every WR but the signaled last one.
  *bad_wr = (g_post.mode == PostFail::Tail) ? tail : wr;
  return ENOMEM;
}

constexpr int kNqps = 2;
constexpr int kTag = 7;
constexpr uintptr_t kLocalBase = 0x100000;
constexpr size_t kSize = 0x2000;

// One device, two data QPs, and a three-segment local buffer. Each QP gets a
// 0x1000 chunk that crosses one segment boundary, so it posts two WRs.
class NetIbSegmentedSendPrefix : public ::testing::Test {
protected:
  void SetUp() override {
    g_post = FakePost{};
    const size_t bytes = (sizeof(ncclIbSendComm) + 63) / 64 * 64;
    comm = static_cast<ncclIbSendComm*>(std::aligned_alloc(64, bytes));
    ASSERT_NE(comm, nullptr);
    std::memset(comm, 0, bytes);

    comm->base.isSend = true;
    comm->base.ready = 1;
    comm->base.vProps.ndevs = 1;
    comm->base.nqps = kNqps;
    comm->base.nDataQps = kNqps;
    comm->base.recvMatchingScheme = BY_INDEX;
    comm->base.weights[0] = 100;

    std::memset(&ctx, 0, sizeof(ctx));
    ctx.ops.post_send = FakePostSend;
    for (int q = 0; q < kNqps; q++) {
      std::memset(&qps[q], 0, sizeof(qps[q]));
      qps[q].context = &ctx;
      qps[q].qp_num = 10 + q;
      comm->base.qps[q].qp = &qps[q];
      comm->base.activeQps[q] = &comm->base.qps[q];
    }

    std::memset(cts, 0, sizeof(cts));
    cts[0][0].idx = 1;
    cts[0][0].nreqs = 1;
    cts[0][0].tag = kTag;
    cts[0][0].size = kSize;
    cts[0][0].addr = 0x900000;
    cts[0][0].rkeys[0] = 0x77;
    comm->ctsFifo = cts;

    std::memset(mrs, 0, sizeof(mrs));
    std::memset(&mh, 0, sizeof(mh));
    const size_t segLen[3] = {0x800, 0x1000, 0x800};
    uintptr_t start = kLocalBase;
    mh.nSegments = 3;
    for (int s = 0; s < 3; s++) {
      mrs[s].lkey = 0x10 + s;
      mh.segStart[s] = start;
      mh.segLen[s] = segLen[s];
      mh.segMrs[s][0] = &mrs[s];
      start += segLen[s];
    }
    mh.mrs[0] = &mrs[0];
  }

  void TearDown() override { std::free(comm); }

  ncclResult_t Isend(void** request) {
    *request = nullptr;
    return ncclIbIsend(comm, reinterpret_cast<void*>(kLocalBase), kSize, kTag, &mh, nullptr, request);
  }

  ncclIbSendComm* comm = nullptr;
  ibv_context ctx;
  ibv_qp qps[kNqps];
  alignas(64) ncclIbSendFifo cts[1][NCCL_NET_IB_MAX_RECVS];
  ibv_mr mrs[3];
  ncclIbMrHandle mh;
};

// Nothing posted on either QP, so no completion can come and the request is freed.
TEST_F(NetIbSegmentedSendPrefix, FirstQpRejectsWholeChainFreesRequest) {
  g_post.failQpNum = qps[0].qp_num;
  g_post.mode = PostFail::Head;
  void* request = nullptr;
  Isend(&request);

  EXPECT_EQ(g_post.calls, 1);
  EXPECT_EQ(g_post.failedCallWrs, 2);
  EXPECT_EQ(g_post.failedCallSignaled, 1);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(comm->base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(comm->base.reqs[0].events[0], 0);
  EXPECT_GT(comm->base.stats.fatalErrorCount, 0);
}

// The posted prefix keeps the first QP's event, as iflush does; the second QP never posted.
TEST_F(NetIbSegmentedSendPrefix, FirstQpLosesSignaledTailKeepsOnlyThatEvent) {
  g_post.failQpNum = qps[0].qp_num;
  g_post.mode = PostFail::Tail;
  void* request = nullptr;
  Isend(&request);

  EXPECT_EQ(g_post.calls, 1);
  EXPECT_EQ(request, &comm->base.reqs[0]);
  EXPECT_EQ(comm->base.reqs[0].type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(comm->base.reqs[0].events[0], 1);
  EXPECT_GT(comm->base.stats.fatalErrorCount, 0);
}

// The first QP's signaled tail posted and will complete; only the second QP's event is dropped.
TEST_F(NetIbSegmentedSendPrefix, SecondQpRejectsWholeChainKeepsFirstQpEvent) {
  g_post.failQpNum = qps[1].qp_num;
  g_post.mode = PostFail::Head;
  void* request = nullptr;
  Isend(&request);

  EXPECT_EQ(g_post.calls, 2);
  EXPECT_EQ(g_post.failedCallWrs, 2);
  EXPECT_EQ(request, &comm->base.reqs[0]);
  EXPECT_EQ(comm->base.reqs[0].type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(comm->base.reqs[0].events[0], 1);
  EXPECT_TRUE(comm->base.reqs[0].send.sentData[0]);
  EXPECT_FALSE(comm->base.reqs[0].send.sentData[1]);
}

// Control: with no failure both chains post and both events stay armed.
TEST_F(NetIbSegmentedSendPrefix, BothQpsPostKeepsBothEvents) {
  void* request = nullptr;
  EXPECT_EQ(Isend(&request), ncclSuccess);

  EXPECT_EQ(g_post.calls, 2);
  EXPECT_EQ(request, &comm->base.reqs[0]);
  EXPECT_EQ(comm->base.reqs[0].type, NCCL_NET_IB_REQ_SEND);
  EXPECT_EQ(comm->base.reqs[0].events[0], 2);
  EXPECT_EQ(comm->base.stats.fatalErrorCount, 0);
}

}  // namespace
