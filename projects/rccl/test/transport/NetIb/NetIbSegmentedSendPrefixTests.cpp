/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Host-only tests for the IB-MSEG multi-segment send built by rcclIbFillSendWrs from
 * IbMsegMultiSend, IbMsegIflush on multi-segment handles, and the hooks p2p.cc calls.
 * A fake ibv_post_send records every WR and can reject a chain whole or only its signaled tail;
 * no RDMA device is needed.
 */

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#include "net_ib_multiseg/p2p_mseg.h"
#include "net_ib_multiseg/p2p_multiseg_mseg.h"
#include "net_ib_multiseg/p2p_resiliency_mseg.h"
#include "IbFakePostSend.hpp"

namespace {

using namespace RcclIbFake;

constexpr int kNqps = 2;
constexpr int kTag = 7;
constexpr uintptr_t kLocalBase = 0x100000;
constexpr size_t kSize = 0x2000;

// One device, two data QPs, and a three-segment local buffer. Each QP gets a
// 0x1000 chunk that crosses one segment boundary, so it posts two slices.
class NetIbSegmentedSendPrefix : public ::testing::Test {
protected:
  void SetUp() override {
    g_post = FakePost{};
    const size_t bytes = (sizeof(IbMsegSendComm) + 63) / 64 * 64;
    comm = static_cast<IbMsegSendComm*>(std::aligned_alloc(64, bytes));
    ASSERT_NE(comm, nullptr);
    std::memset(comm, 0, bytes);

    comm->base.isSend = true;
    comm->base.ready = 1;
    comm->base.vProps.ndevs = 1;
    comm->base.nqps = kNqps;
    comm->base.nDataQps = kNqps;
    comm->base.recvMatchingScheme = BY_INDEX;
    comm->base.weights[0] = 100;
    // A multi-segment connect allocates both; LegacyCtsComm* tests clear them.
    std::memset(&cursorStore, 0, sizeof(cursorStore));
    std::memset(&cacheStore, 0, sizeof(cacheStore));
    comm->sendCursor = &cursorStore;
    comm->layoutCache = &cacheStore;

    std::memset(&ctx, 0, sizeof(ctx));
    ctx.ops.post_send = FakePostSend;
    for (int q = 0; q < kNqps; q++) {
      std::memset(&qps[q], 0, sizeof(qps[q]));
      qps[q].context = &ctx;
      qps[q].qp_num = 10 + q;
      comm->base.qps[q].qp = &qps[q];
      comm->base.activeQps[q] = &comm->base.qps[q];
    }

    std::memset(&ctsMem, 0, sizeof(ctsMem));
    ctsMem.cts[0][0].idx = 1;
    ctsMem.cts[0][0].nreqs = 1;
    ctsMem.cts[0][0].tag = kTag;
    ctsMem.cts[0][0].size = kSize;
    ctsMem.cts[0][0].addr = 0x900000;
    ctsMem.cts[0][0].rkeys[0] = 0x77;
    comm->ctsFifo = ctsMem.cts;

    std::memset(mrs, 0, sizeof(mrs));
    std::memset(&mh, 0, sizeof(mh));
    const size_t segLen[3] = {0x800, 0x1000, 0x800};
    uintptr_t start = kLocalBase;
    mh.seg.nSegments = 3;
    std::memset(segMrStore, 0, sizeof(segMrStore));
    mh.seg.segStart = segStartStore;
    mh.seg.segLen = segLenStore;
    mh.seg.segOff = segOffStore;
    mh.seg.segMrs = segMrStore;
    for (int s = 0; s < 3; s++) {
      mrs[s].lkey = 0x10 + s;
      mrs[s].rkey = 0x20 + s;
      mh.seg.segStart[s] = start;
      mh.seg.segLen[s] = segLen[s];
      mh.seg.segOff[s] = start - kLocalBase;
      mh.seg.segMrs[s * NCCL_IB_MAX_DEVS_PER_NIC] = &mrs[s];
      start += segLen[s];
    }
    mh.seg.segOff[3] = start - kLocalBase;
    mh.mrs[0] = &mrs[0];

    std::memset(&mrSingle, 0, sizeof(mrSingle));
    std::memset(&mhSingle, 0, sizeof(mhSingle));
    mrSingle.lkey = 0x30;
    mrSingle.rkey = 0x31;
    mhSingle.seg.nSegments = 1;
    mhSingle.mrs[0] = &mrSingle;
    std::memset(ctsMem.layout, 0, sizeof(ctsMem.layout));
  }

  void TearDown() override {
    if (comm->sendCursor != &cursorStore) std::free(comm->sendCursor);
    std::free(comm);
  }

  ncclResult_t Isend(void** request) {
    *request = nullptr;
    return IbMsegIsend(comm, reinterpret_cast<void*>(kLocalBase), kSize, kTag, &mh, nullptr, request);
  }

  ncclResult_t IsendFrom(void* data, size_t size, IbMsegMrHandle* handle, void** request) {
    *request = nullptr;
    return IbMsegIsend(comm, data, size, kTag, handle, nullptr, request);
  }

  // Publish a receiver layout for CTS entry 0 with these segment starts and device-0 rkeys.
  void UseRemoteLayout(std::initializer_list<uint64_t> starts, std::initializer_list<uint32_t> rkeys) {
    IbMsegSegLayout* layout = rcclIbCtsLayout(ctsMem.cts[0], 1, 0);
    std::memset(layout, 0, sizeof(*layout));
    layout->nSegments = static_cast<uint32_t>(starts.size());
    int s = 0;
    for (uint64_t v : starts) layout->segStart[s++] = v;
    s = 0;
    for (uint32_t k : rkeys) rcclIbLayoutSetRkey(layout, s++, 0, k);
    rcclIbLayoutStamp(layout, ctsMem.cts[0][0].idx);
    comm->useMultiSeg = 1;
    ctsMem.cts[0][0].nreqs |= NCCL_IB_CTS_NREQS_LAYOUT;
  }

  static constexpr uint64_t kRefAddr = 0xc00000;
  static constexpr uint64_t kRefGen = 5;
  static constexpr uint32_t kRefRkey = 0x66;
  static constexpr uint32_t kRefSegs = RCCL_IB_CTS_INLINE_SEGS + 2;
  static constexpr uint64_t kRefVa0 = 0x8ffe00;
  static constexpr uint64_t kRefSeg = 0x600;

  // CTS entry 0 references a kRefSegs-segment layout. Cache entry 0 already holds a buffer, so no MR is registered.
  void UseReferencedLayout(uint64_t addr = kRefAddr) {
    IbMsegSegLayout* layout = rcclIbCtsLayout(ctsMem.cts[0], 1, 0);
    std::memset(layout, 0, sizeof(*layout));
    layout->byRef = 1;
    layout->ref.addr = addr;
    layout->ref.gen = kRefGen;
    layout->ref.nSegments = kRefSegs;
    layout->ref.ndevs = 1;
    layout->ref.rkeys[0] = kRefRkey;
    rcclIbLayoutStamp(layout, ctsMem.cts[0][0].idx);
    comm->useMultiSeg = 1;
    ctsMem.cts[0][0].nreqs |= NCCL_IB_CTS_NREQS_LAYOUT;
    std::memset(refBlob, 0, sizeof(refBlob));
    refMr = {};
    refMr.lkey = 0x3c;
    rcclIbLayoutCacheEntry& e = comm->layoutCache->e[0];
    e.buf = refBlob;
    e.cap = sizeof(refBlob);
    e.mr = &refMr;
    e.dev = 0;
  }

  // What the read returns: kRefSegs segments of kRefSeg from kRefVa0, rkeys 0x90 + s.
  void FillReferencedLayout(void* blob) {
    reinterpret_cast<uint64_t*>(blob)[0] = kRefVa0;
    for (uint32_t s = 0; s <= kRefSegs; s++) rcclIbLayoutBlobOff(blob)[s] = s * kRefSeg;
    for (uint32_t s = 0; s < kRefSegs; s++) rcclIbLayoutBlobRkeys(blob, kRefSegs, 0)[s] = 0x90 + s;
  }

  const PostRec* PostOn(int q) const { return RcclIbFake::PostOn(qps[q].qp_num); }

  // Complete every in-flight slice as the CQ would, then let rcclIbSendProgress post what follows.
  ncclResult_t CompleteSlices() {
    rcclIbSendCursor& c = *comm->sendCursor;
    const int nreqs = IbMsegCtsNreqs(rcclIbSendCtsRow(comm, c.slot)[0].nreqs);
    for (int i = 0; i < c.nqps; i++) {
      if (c.qpState[i] != RCCL_IB_SEND_QP_INFLIGHT) continue;
      EXPECT_TRUE(rcclIbSendSliceDone(comm, c.slot, c.qp[i]->devIndex, c.qp[i]->qp->qp_num));
      for (int r = 0; r < nreqs; r++) comm->sendReqs[c.slot][r]->events[c.qp[i]->devIndex]--;
    }
    return rcclIbSendProgress(comm);
  }

  // Run every slice to its tail.
  void Drain() {
    for (int n = 0; comm->sendCursor->active && n < 16; n++) ASSERT_EQ(CompleteSlices(), ncclSuccess);
    ASSERT_FALSE(comm->sendCursor->active);
  }

  std::vector<WrRec> WrsOn(int q) const { return RcclIbFake::WrsOn(qps[q].qp_num); }

  struct Slice {
    uint64_t laddr;
    uint32_t len;
    uint32_t lkey;
    uint64_t raddr;
    uint32_t rkey;
  };

  // QP q posted exactly these slices; each but the last ends a post and the last carries the imm.
  void ExpectSlicesOn(int q, std::initializer_list<Slice> want) const {
    const std::vector<WrRec> w = WrsOn(q);
    ASSERT_EQ(w.size(), want.size());
    size_t i = 0;
    for (const Slice& s : want) {
      ExpectWrite(w[i], s.laddr, s.len, s.lkey, s.raddr, s.rkey);
      if (++i < want.size()) ExpectSliceEnd(w[i - 1]);
      else ExpectImmTail(w[i - 1], kSize);
    }
  }

  // Nothing posted: the armed request is kept as failed with every QP's event, and the comm goes fatal.
  void ExpectFailedBeforePosting(void* request) const {
    EXPECT_EQ(g_post.calls, 0);
    EXPECT_EQ(request, &comm->base.reqs[0]);
    EXPECT_EQ(comm->base.reqs[0].type, NCCL_NET_IB_REQ_FAILED);
    EXPECT_EQ(comm->base.reqs[0].events[0], kNqps);
    EXPECT_GT(comm->base.stats.fatalErrorCount, 0);
  }

  void ExpectSendFailsBeforePosting(uintptr_t data = kLocalBase) {
    void* request = nullptr;
    EXPECT_EQ(IsendFrom(reinterpret_cast<void*>(data), kSize, &mh, &request), ncclSuccess);
    ExpectFailedBeforePosting(request);
  }

  // Slot 1 matched by two receives: kTag over the three-segment buffer at 0x900000 and kTag+1
  // over a single-segment 0x800 buffer at 0x980000. Only the second isend posts.
  void IsendTwoReceivesOnSlot1(bool flagFlush) {
    std::memset(cts2, 0, sizeof(cts2));
    for (int r = 0; r < 2; r++) {
      cts2[1][r].idx = 2;
      cts2[1][r].nreqs = 2;
      cts2[1][r].tag = kTag + r;
      cts2[1][r].rkeys[0] = 0x77 + r;
    }
    cts2[1][0].size = kSize;
    cts2[1][0].addr = 0x900000;
    cts2[1][0].flagFlush = flagFlush;
    cts2[1][1].size = 0x800;
    cts2[1][1].addr = 0x980000;
    comm->ctsFifo = cts2;
    comm->base.fifoHead = 1;
    comm->remCmplsRecords.addr = 0xa00000;
    comm->remCmplsRecords.rkeys[0] = 0x99;

    void* first = nullptr;
    void* second = nullptr;
    ASSERT_EQ(IbMsegIsend(comm, reinterpret_cast<void*>(kLocalBase), kSize, kTag, &mh, nullptr, &first), ncclSuccess);
    ASSERT_EQ(first, &comm->base.reqs[0]);
    ASSERT_EQ(g_post.calls, 0);
    ASSERT_EQ(IbMsegIsend(comm, reinterpret_cast<void*>(0x300000), 0x800, kTag + 1, &mhSingle, nullptr, &second),
              ncclSuccess);
    ASSERT_EQ(second, &comm->base.reqs[1]);
    ASSERT_EQ(g_post.posts.size(), 2u);
  }

  // The plain or imm sizes record for slot 1: one int per receive.
  void ExpectSlot1SizesRecord(const WrRec& w) const {
    EXPECT_EQ(w.raddr, 0xa00000u + sizeof(IbMsegRequestCompletionRecord));
    EXPECT_EQ(w.rkey, 0x99u);
    EXPECT_EQ(w.laddr, reinterpret_cast<uint64_t>(comm->remCmplsRecords.elems[1]));
    EXPECT_EQ(w.len, 2 * sizeof(int));
  }

  IbMsegSendComm* comm = nullptr;
  ibv_context ctx;
  ibv_qp qps[kNqps];
  // Layout prefix immediately before the CTS row, matching an advertising slot.
  struct CtsWithLayout {
    alignas(64) IbMsegSegLayout layout[NCCL_NET_IB_MAX_RECVS];
    alignas(64) IbMsegSendFifo cts[1][NCCL_NET_IB_MAX_RECVS];
  };
  static_assert(offsetof(CtsWithLayout, cts) == sizeof(IbMsegSegLayout) * NCCL_NET_IB_MAX_RECVS,
                "the test CTS row must follow its layout prefix");
  CtsWithLayout ctsMem;
  alignas(64) IbMsegSendFifo cts2[2][NCCL_NET_IB_MAX_RECVS];
  ibv_mr mrs[3];
  uintptr_t segStartStore[3];
  size_t segLenStore[3];
  uint64_t segOffStore[4];
  void* segMrStore[3 * NCCL_IB_MAX_DEVS_PER_NIC];
  IbMsegMrHandle mh;
  ibv_mr mrSingle;
  IbMsegMrHandle mhSingle;
  uint64_t refBlob[32];
  ibv_mr refMr;
  rcclIbSendCursor cursorStore;
  rcclIbLayoutCache cacheStore;
};

// A legacy-CTS comm has no cursor until a send crosses a segment boundary; then it allocates one and slices.
TEST_F(NetIbSegmentedSendPrefix, LegacyCtsCommAllocatesTheCursorOnItsFirstSegmentedSend) {
  comm->sendCursor = nullptr;
  comm->layoutCache = nullptr;
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  EXPECT_EQ(request, &comm->base.reqs[0]);
  ASSERT_NE(comm->sendCursor, nullptr);
  EXPECT_NE(comm->sendCursor, &cursorStore);
  EXPECT_TRUE(comm->sendCursor->active);
  ASSERT_NO_FATAL_FAILURE(Drain());
  EXPECT_EQ(g_post.posts.size(), 4u);
  EXPECT_EQ(comm->layoutCache, nullptr);
}

// A single-segment send on a legacy-CTS comm posts each QP's chunk whole and allocates no cursor.
TEST_F(NetIbSegmentedSendPrefix, LegacyCtsCommSingleSegmentSendAllocatesNoCursor) {
  comm->sendCursor = nullptr;
  comm->layoutCache = nullptr;
  void* request = nullptr;
  ASSERT_EQ(IsendFrom(reinterpret_cast<void*>(kLocalBase), kSize, &mhSingle, &request), ncclSuccess);
  EXPECT_EQ(request, &comm->base.reqs[0]);
  EXPECT_EQ(g_post.calls, kNqps);
  EXPECT_EQ(comm->sendCursor, nullptr);
  EXPECT_EQ(comm->base.fifoHead, 1u);
  EXPECT_EQ(rcclIbSendProgress(comm), ncclSuccess);
  EXPECT_FALSE(rcclIbSendSliceDone(comm, 0, 0, qps[0].qp_num));
  EXPECT_EQ(rcclIbSendLayoutCacheFree(comm), ncclSuccess);
}

// Nothing posted on either QP, so no completion can come: the request is freed, unlinked
// from its slot, and the post error is returned.
TEST_F(NetIbSegmentedSendPrefix, FirstQpRejectsWholeChainFreesRequest) {
  g_post.failQpNum = qps[0].qp_num;
  g_post.mode = PostFail::Head;
  void* request = nullptr;
  EXPECT_EQ(Isend(&request), ncclSystemError);

  EXPECT_EQ(g_post.calls, 1);
  EXPECT_EQ(g_post.failedCallWrs, 1);
  EXPECT_EQ(g_post.failedCallSignaled, 1);
  EXPECT_EQ(request, nullptr);
  EXPECT_FALSE(comm->sendCursor->active);
  EXPECT_EQ(comm->base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
  EXPECT_EQ(comm->base.reqs[0].events[0], 0);
  EXPECT_EQ(comm->sendReqs[0][0], nullptr);
  EXPECT_EQ(comm->sendReqsCnt[0], 0);
  EXPECT_GT(comm->base.stats.fatalErrorCount, 0);
}

// The posted prefix keeps the first QP's event, as iflush does; the second QP never posted.
// The first QP's chunk sits in one remote segment, so its one post is the data and the BY_ID tail.
TEST_F(NetIbSegmentedSendPrefix, FirstQpLosesSignaledTailKeepsOnlyThatEvent) {
  UseRemoteLayout({0x900000, 0x901800}, {0x81, 0x82});
  comm->base.recvMatchingScheme = BY_ID;
  comm->remCmplsRecords.addr = 0xa00000;
  comm->remCmplsRecords.rkeys[0] = 0x99;
  g_post.failQpNum = qps[0].qp_num;
  g_post.mode = PostFail::Tail;
  void* request = nullptr;
  EXPECT_EQ(IsendFrom(reinterpret_cast<void*>(kLocalBase), kSize, &mhSingle, &request), ncclSuccess);

  EXPECT_EQ(g_post.calls, 1);
  EXPECT_EQ(g_post.failedCallWrs, 2);
  EXPECT_EQ(request, &comm->base.reqs[0]);
  EXPECT_EQ(comm->base.reqs[0].type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(comm->base.reqs[0].events[0], 1);
  EXPECT_EQ(comm->sendReqs[0][0], &comm->base.reqs[0]);
  EXPECT_GT(comm->base.stats.fatalErrorCount, 0);
}

// The first QP's signaled slice posted and will complete; only the second QP's events are dropped.
TEST_F(NetIbSegmentedSendPrefix, SecondQpRejectsWholeChainKeepsFirstQpEvent) {
  g_post.failQpNum = qps[1].qp_num;
  g_post.mode = PostFail::Head;
  void* request = nullptr;
  EXPECT_EQ(Isend(&request), ncclSuccess);

  EXPECT_EQ(g_post.calls, 2);
  EXPECT_EQ(g_post.failedCallWrs, 1);
  EXPECT_EQ(request, &comm->base.reqs[0]);
  EXPECT_EQ(comm->base.reqs[0].type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_EQ(comm->base.reqs[0].events[0], 2);
  EXPECT_FALSE(comm->sendCursor->active);
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
  // Each QP's tail event plus the event its slice holds until that slice completes.
  EXPECT_EQ(comm->base.reqs[0].events[0], 4);
  ASSERT_NO_FATAL_FAILURE(Drain());
  EXPECT_EQ(g_post.calls, 4);
  EXPECT_EQ(comm->base.reqs[0].events[0], 2);
  EXPECT_EQ(comm->base.stats.fatalErrorCount, 0);
}

// An out-of-range layout fails before any post: the armed request is kept
// as failed and the comm goes fatal.
TEST_F(NetIbSegmentedSendPrefix, InvalidRemoteLayoutFailsBeforePosting) {
  IbMsegSegLayout* layout = rcclIbCtsLayout(ctsMem.cts[0], 1, 0);
  std::memset(layout, 0, sizeof(*layout));
  layout->nSegments = RCCL_IB_CTS_INLINE_SEGS + 1;
  rcclIbLayoutStamp(layout, ctsMem.cts[0][0].idx);
  comm->useMultiSeg = 1;
  ctsMem.cts[0][0].nreqs |= NCCL_IB_CTS_NREQS_LAYOUT;
  ExpectSendFailsBeforePosting();
}

// A layout line that has not landed yet: isend returns no request and posts nothing until its stamp arrives.
TEST_F(NetIbSegmentedSendPrefix, LayoutLineNotYetArrivedDefersTheSend) {
  UseRemoteLayout({0x900000, 0x900c00}, {0x81, 0x82});
  IbMsegSegLayout* layout = rcclIbCtsLayout(ctsMem.cts[0], 1, 0);
  layout->idx1 = ctsMem.cts[0][0].idx - 1;
  void* request = nullptr;
  ASSERT_EQ(IsendFrom(reinterpret_cast<void*>(kLocalBase), kSize, &mhSingle, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(g_post.calls, 0);
  EXPECT_EQ(comm->base.fifoHead, 0u);
  EXPECT_EQ(comm->sendReqs[0][0], nullptr);

  layout->idx1 = ctsMem.cts[0][0].idx;
  ASSERT_EQ(IsendFrom(reinterpret_cast<void*>(kLocalBase), kSize, &mhSingle, &request), ncclSuccess);
  EXPECT_EQ(request, &comm->base.reqs[0]);
  EXPECT_EQ(g_post.calls, kNqps);
  EXPECT_EQ(comm->base.fifoHead, 1u);
}

// Each QP's 0x1000 chunk splits at the local 0x800 and 0x1800 boundaries with that
// segment's lkey and the one CTS rkey. The first slice posts alone and signals; the
// second posts after it completes and carries the imm.
TEST_F(NetIbSegmentedSendPrefix, LocalSegmentsSplitAtEachBoundary) {
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  ASSERT_EQ(g_post.posts.size(), 2u);
  EXPECT_TRUE(comm->sendCursor->active);
  ASSERT_NO_FATAL_FAILURE(Drain());
  ASSERT_EQ(g_post.posts.size(), 4u);
  ExpectSlicesOn(0, {{0x100000, 0x800, 0x10, 0x900000, 0x77}, {0x100800, 0x800, 0x11, 0x900800, 0x77}});
  ExpectSlicesOn(1, {{0x101000, 0x800, 0x11, 0x901000, 0x77}, {0x101800, 0x800, 0x12, 0x901800, 0x77}});
  EXPECT_TRUE(comm->base.reqs[0].send.sentData[0]);
  EXPECT_TRUE(comm->base.reqs[0].send.sentData[1]);
  EXPECT_EQ(comm->base.fifoHead, 1u);
}

// A single-segment source still splits at the receiver's 0xc00 boundary, taking each
// remote segment's rkey from the layout.
TEST_F(NetIbSegmentedSendPrefix, RemoteLayoutSplitsAtRemoteBoundaries) {
  UseRemoteLayout({0x900000, 0x900c00}, {0x81, 0x82});
  void* request = nullptr;
  ASSERT_EQ(IsendFrom(reinterpret_cast<void*>(kLocalBase), kSize, &mhSingle, &request), ncclSuccess);
  ASSERT_NO_FATAL_FAILURE(Drain());
  ExpectSlicesOn(0, {{0x100000, 0xc00, 0x30, 0x900000, 0x81}, {0x100c00, 0x400, 0x30, 0x900c00, 0x82}});
  ExpectSlicesOn(1, {{0x101000, 0x1000, 0x30, 0x901000, 0x82}});
}

// Different layouts on the two sides: every slice stays inside one local and one remote segment.
TEST_F(NetIbSegmentedSendPrefix, SplitsAtLocalAndRemoteBoundaries) {
  UseRemoteLayout({0x900000, 0x900c00}, {0x81, 0x82});
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  ASSERT_NO_FATAL_FAILURE(Drain());
  ExpectSlicesOn(0, {{0x100000, 0x800, 0x10, 0x900000, 0x81},
                     {0x100800, 0x400, 0x11, 0x900800, 0x81},
                     {0x100c00, 0x400, 0x11, 0x900c00, 0x82}});
  ExpectSlicesOn(1, {{0x101000, 0x800, 0x11, 0x901000, 0x82}, {0x101800, 0x800, 0x12, 0x901800, 0x82}});
}

// The receive starts 0x1000 into a registration whose second segment begins 0x800 into
// the receive, so the remote request offset is applied before splitting.
TEST_F(NetIbSegmentedSendPrefix, RemoteRequestStartsInsideARegistration) {
  UseRemoteLayout({0x8ff000, 0x900800}, {0x81, 0x82});
  void* request = nullptr;
  ASSERT_EQ(IsendFrom(reinterpret_cast<void*>(kLocalBase), kSize, &mhSingle, &request), ncclSuccess);
  ASSERT_NO_FATAL_FAILURE(Drain());
  ExpectSlicesOn(0, {{0x100000, 0x800, 0x30, 0x900000, 0x81}, {0x100800, 0x800, 0x30, 0x900800, 0x82}});
  ExpectSlicesOn(1, {{0x101000, 0x1000, 0x30, 0x901000, 0x82}});
}

// BY_ID publishes the size record: data WRs stay plain and a dedicated tail writes the
// record with the imm set to the request id.
TEST_F(NetIbSegmentedSendPrefix, ByIdAppendsTheSizesRecordTail) {
  comm->base.recvMatchingScheme = BY_ID;
  comm->remCmplsRecords.addr = 0xa00000;
  comm->remCmplsRecords.rkeys[0] = 0x99;
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  ASSERT_NO_FATAL_FAILURE(Drain());

  for (int q = 0; q < kNqps; q++) {
    const std::vector<WrRec> w = WrsOn(q);
    ASSERT_EQ(w.size(), 3u);
    ExpectSliceEnd(w[0]);
    ExpectPlain(w[1]);
    const WrRec& t = w[2];
    ExpectImmTail(t, 0);
    EXPECT_EQ(t.raddr, 0xa00000u);
    EXPECT_EQ(t.rkey, 0x99u);
    EXPECT_EQ(t.laddr, reinterpret_cast<uint64_t>(comm->remCmplsRecords.elems[0]));
    EXPECT_EQ(t.len, sizeof(int));
  }
  EXPECT_EQ(comm->remCmplsRecords.elems[0][0], static_cast<int>(kSize));
}

// With the GDR flush flag requested, each QP's tail writes its own flag slot with the imm.
TEST_F(NetIbSegmentedSendPrefix, FlagFlushTailWritesThisQpsFlag) {
  comm->flagFlush = 1;
  ctsMem.cts[0][0].flagFlush = 1;
  comm->flagFlushRemAddr = 0xb00000;
  comm->flagFlushRkey = 0x55;
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  ASSERT_NO_FATAL_FAILURE(Drain());

  EXPECT_EQ(comm->flagFlushSrc[0], 1u);
  EXPECT_EQ(comm->flagFlushSge.addr, reinterpret_cast<uint64_t>(&comm->flagFlushSrc[0]));
  for (int q = 0; q < kNqps; q++) {
    const std::vector<WrRec> w = WrsOn(q);
    ASSERT_EQ(w.size(), 3u);
    ExpectSliceEnd(w[0]);
    ExpectPlain(w[1]);
    const WrRec& t = w[2];
    ExpectImmTail(t, kSize);
    EXPECT_EQ(t.sgList, &comm->flagFlushSge);
    EXPECT_EQ(t.rkey, 0x55u);
    EXPECT_EQ(t.raddr, 0xb00000u + q * NCCL_GDR_FLUSH_FLAG_STRIDE);
  }
}

// A zero-byte send still posts one imm WR per QP, with the rkey of the remote segment
// that holds the receive (here the second one).
TEST_F(NetIbSegmentedSendPrefix, ZeroLengthSendUsesTheContainingRemoteSegment) {
  ctsMem.cts[0][0].addr = 0x900c00;
  UseRemoteLayout({0x900000, 0x900c00}, {0x81, 0x82});
  void* request = nullptr;
  ASSERT_EQ(IsendFrom(reinterpret_cast<void*>(kLocalBase), 0, &mh, &request), ncclSuccess);

  ASSERT_EQ(g_post.posts.size(), 2u);
  for (const PostRec& p : g_post.posts) {
    ASSERT_EQ(p.wrs.size(), 1u);
    EXPECT_EQ(p.wrs[0].nsge, 0);
    EXPECT_EQ(p.wrs[0].raddr, 0x900c00u);
    EXPECT_EQ(p.wrs[0].rkey, 0x82u);
    ExpectImmTail(p.wrs[0], 0);
  }
}

// Resiliency resend: the QP that already delivered is skipped and the other QP keeps its
// offset. A resilient comm stages multi-segment buffers, so each QP is one post.
TEST_F(NetIbSegmentedSendPrefix, ResendPostsOnlyTheUndeliveredQp) {
  g_post.failQpNum = qps[1].qp_num;
  g_post.mode = PostFail::Head;
  void* request = nullptr;
  ASSERT_EQ(IsendFrom(reinterpret_cast<void*>(kLocalBase), kSize, &mhSingle, &request), ncclSuccess);
  ASSERT_TRUE(comm->base.reqs[0].send.sentData[0]);
  ASSERT_FALSE(comm->base.reqs[0].send.sentData[1]);

  g_post = FakePost{};
  IbMsegResiliency* resiliency = static_cast<IbMsegResiliency*>(std::calloc(1, sizeof(IbMsegResiliency)));
  ASSERT_NE(resiliency, nullptr);
  comm->base.resiliency = resiliency;
  ncclResult_t ret = IbMsegMultiSend(comm, 0);
  comm->base.resiliency = nullptr;
  std::free(resiliency);
  ASSERT_EQ(ret, ncclSuccess);

  ASSERT_EQ(g_post.posts.size(), 1u);
  ExpectSlicesOn(1, {{0x101000, 0x1000, 0x30, 0x901000, 0x77}});
  EXPECT_TRUE(comm->base.reqs[0].send.sentData[1]);
  EXPECT_FALSE(comm->sendCursor->active);
}

// A receive below the first layout segment cannot be expressed: nothing posts. The wrapped
// offset would also fail the later overflow check, so this pins the outcome, not which check fires.
TEST_F(NetIbSegmentedSendPrefix, RemoteAddrBelowTheFirstSegmentFailsBeforePosting) {
  UseRemoteLayout({0x900100, 0x900c00}, {0x81, 0x82});
  ExpectSendFailsBeforePosting();
}

// Segment starts must strictly increase.
TEST_F(NetIbSegmentedSendPrefix, NonIncreasingRemoteStartsFailBeforePosting) {
  UseRemoteLayout({0x900000, 0x900000}, {0x81, 0x82});
  ExpectSendFailsBeforePosting();
}

// A zero rkey in the layout is rejected.
TEST_F(NetIbSegmentedSendPrefix, ZeroRemoteRkeyFailsBeforePosting) {
  UseRemoteLayout({0x900000, 0x900c00}, {0x81, 0});
  ExpectSendFailsBeforePosting();
}

// A source below its own registration fails on the first QP, before any post. As above, a
// later split check would also reject it, so this pins the outcome.
TEST_F(NetIbSegmentedSendPrefix, LocalAddrBelowTheRegistrationFailsBeforePosting) {
  ExpectSendFailsBeforePosting(kLocalBase - 0x100);
}

// Two receives on slot 1: every WR carries the packed wr_id (slot in each request's byte), each
// request is split at its own offsets, and the imm tail writes a two-int sizes record. The first
// post moves one slice of each receive; the second finishes the first receive and writes the tail.
TEST_F(NetIbSegmentedSendPrefix, TwoReceivesOnSlot1PackTheWrIdAndTheSizesRecord) {
  ASSERT_NO_FATAL_FAILURE(IsendTwoReceivesOnSlot1(false));
  EXPECT_EQ(comm->base.reqs[0].events[0], 2 * kNqps);
  EXPECT_EQ(comm->base.reqs[1].events[0], 2 * kNqps);
  ASSERT_NO_FATAL_FAILURE(Drain());
  const uint64_t wrId = 0x0101;
  for (int q = 0; q < kNqps; q++) {
    const std::vector<WrRec> w = WrsOn(q);
    ASSERT_EQ(w.size(), 4u);
    for (const WrRec& x : w) EXPECT_EQ(x.wrId, wrId);
    const uint64_t off = q * 0x1000;
    ExpectWrite(w[0], 0x100000 + off, 0x800, 0x10 + q, 0x900000 + off, 0x77);
    ExpectPlain(w[0]);
    ExpectWrite(w[1], 0x300000 + q * 0x400, 0x400, 0x30, 0x980000 + q * 0x400, 0x78);
    ExpectSliceEnd(w[1]);
    ExpectWrite(w[2], 0x100800 + off, 0x800, 0x11 + q, 0x900800 + off, 0x77);
    ExpectPlain(w[2]);
    ExpectImmTail(w[3], kSize);
    ExpectSlot1SizesRecord(w[3]);
  }
  EXPECT_EQ(comm->remCmplsRecords.elems[1][0], static_cast<int>(kSize));
  EXPECT_EQ(comm->remCmplsRecords.elems[1][1], 0x800);
  EXPECT_EQ(comm->base.reqs[0].events[0], kNqps);
  EXPECT_EQ(comm->base.reqs[1].events[0], kNqps);
  EXPECT_EQ(comm->base.fifoHead, 2u);
}

// The flag flush with two receives writes the sizes record as a plain WR, then the flag WR
// carries the imm; both keep the packed wr_id.
TEST_F(NetIbSegmentedSendPrefix, FlagFlushWithTwoReceivesWritesTheSizesThenTheFlag) {
  comm->flagFlush = 1;
  comm->flagFlushRemAddr = 0xb00000;
  comm->flagFlushRkey = 0x55;
  ASSERT_NO_FATAL_FAILURE(IsendTwoReceivesOnSlot1(true));
  ASSERT_NO_FATAL_FAILURE(Drain());
  EXPECT_EQ(comm->flagFlushSrc[1], 2u);
  EXPECT_EQ(comm->flagFlushSge.addr, reinterpret_cast<uint64_t>(&comm->flagFlushSrc[1]));
  for (int q = 0; q < kNqps; q++) {
    const std::vector<WrRec> w = WrsOn(q);
    ASSERT_EQ(w.size(), 5u);
    ExpectPlain(w[0]);
    ExpectSliceEnd(w[1]);
    ExpectPlain(w[2]);
    ExpectPlain(w[3]);
    ExpectSlot1SizesRecord(w[3]);
    EXPECT_EQ(w[3].wrId, 0x0101u);
    const WrRec& t = w[4];
    ExpectImmTail(t, kSize);
    EXPECT_EQ(t.wrId, 0x0101u);
    EXPECT_EQ(t.sgList, &comm->flagFlushSge);
    EXPECT_EQ(t.rkey, 0x55u);
    EXPECT_EQ(t.raddr, 0xb00000u + q * NCCL_GDR_FLUSH_FLAG_STRIDE);
  }
}

// Above the AR threshold the imm moves to a dedicated zero-length WR after the data, which stays plain.
TEST_F(NetIbSegmentedSendPrefix, AdaptiveRoutingAddsADedicatedImmWr) {
  const int64_t savedThreshold = IbMsegArThreshold;
  IbMsegArThreshold = kSize / 2;
  comm->ar = 1;
  void* request = nullptr;
  ncclResult_t ret = Isend(&request);
  IbMsegArThreshold = savedThreshold;
  ASSERT_EQ(ret, ncclSuccess);
  // The tail shape is fixed at the first post, so later slices keep the dedicated imm WR.
  ASSERT_NO_FATAL_FAILURE(Drain());

  for (int q = 0; q < kNqps; q++) {
    const std::vector<WrRec> w = WrsOn(q);
    ASSERT_EQ(w.size(), 3u);
    ExpectSliceEnd(w[0]);
    ExpectPlain(w[1]);
    EXPECT_EQ(w[2].nsge, 0);
    ExpectImmTail(w[2], kSize);
  }
}

// Below IB_SPLIT_DATA_THRESHOLD only the first QP carries data; the second posts a zero-length
// imm WR just past it, with the rkey of the remote segment holding that offset.
TEST_F(NetIbSegmentedSendPrefix, BelowTheSplitThresholdOnlyTheFirstQpCarriesData) {
  UseRemoteLayout({0x900000, 0x900c00}, {0x81, 0x82});
  const size_t size = 64;
  ASSERT_LT(static_cast<int64_t>(size), rcclParamIbSplitDataThreshold());
  void* request = nullptr;
  ASSERT_EQ(IsendFrom(reinterpret_cast<void*>(kLocalBase), size, &mhSingle, &request), ncclSuccess);

  const PostRec* p0 = PostOn(0);
  ASSERT_NE(p0, nullptr);
  ASSERT_EQ(p0->wrs.size(), 1u);
  ExpectWrite(p0->wrs[0], 0x100000, size, 0x30, 0x900000, 0x81);
  ExpectImmTail(p0->wrs[0], size);

  const PostRec* p1 = PostOn(1);
  ASSERT_NE(p1, nullptr);
  ASSERT_EQ(p1->wrs.size(), 1u);
  EXPECT_EQ(p1->wrs[0].nsge, 0);
  EXPECT_EQ(p1->wrs[0].raddr, 0x900000u + size);
  EXPECT_EQ(p1->wrs[0].rkey, 0x81u);
  ExpectImmTail(p1->wrs[0], size);
  EXPECT_FALSE(comm->sendCursor->active);
}

// A send behind a slot still posting slices waits, then posts after that slot's tail on each QP,
// so a BY_INDEX receiver consumes its receive WRs in slot order.
TEST_F(NetIbSegmentedSendPrefix, ASendBehindASlicedSlotPostsAfterItsTails) {
  std::memset(cts2, 0, sizeof(cts2));
  for (int s = 0; s < 2; s++) {
    cts2[s][0].idx = s + 1;
    cts2[s][0].nreqs = 1;
    cts2[s][0].tag = kTag;
    cts2[s][0].size = kSize;
    cts2[s][0].addr = 0x900000 + s * 0x10000;
    cts2[s][0].rkeys[0] = 0x77;
  }
  comm->ctsFifo = cts2;
  void* first = nullptr;
  void* second = nullptr;
  ASSERT_EQ(Isend(&first), ncclSuccess);
  ASSERT_EQ(g_post.posts.size(), 2u);
  ASSERT_EQ(IsendFrom(reinterpret_cast<void*>(0x300000), kSize, &mhSingle, &second), ncclSuccess);
  ASSERT_EQ(second, &comm->base.reqs[1]);
  EXPECT_EQ(g_post.posts.size(), 2u);
  EXPECT_EQ(comm->sendCursor->queued, 1);
  EXPECT_EQ(comm->base.fifoHead, 2u);

  ASSERT_EQ(CompleteSlices(), ncclSuccess);
  EXPECT_FALSE(comm->sendCursor->active);
  EXPECT_EQ(comm->sendCursor->queued, 0);
  EXPECT_EQ(comm->sendCursor->slot, 1);
  for (int q = 0; q < kNqps; q++) {
    const std::vector<WrRec> w = WrsOn(q);
    ASSERT_EQ(w.size(), 3u);
    ExpectSliceEnd(w[0]);
    ExpectImmTail(w[1], kSize);
    EXPECT_EQ(w[1].wrId, 0u);
    const uint64_t off = q * 0x1000;
    ExpectWrite(w[2], 0x300000 + off, 0x1000, 0x30, 0x910000 + off, 0x77);
    ExpectImmTail(w[2], kSize);
    EXPECT_EQ(w[2].wrId, 1u);
  }
}

// Only an in-flight slice of the slot still posting frees its QP; another slot, QP or HCA does not.
TEST_F(NetIbSegmentedSendPrefix, SliceDoneMatchesOnlyTheSlotStillPosting) {
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  EXPECT_FALSE(rcclIbSendSliceDone(comm, 1, /*devIndex=*/0, qps[0].qp_num));
  EXPECT_FALSE(rcclIbSendSliceDone(comm, 0, /*devIndex=*/0, 99));
  EXPECT_FALSE(rcclIbSendSliceDone(comm, 0, /*devIndex=*/1, qps[0].qp_num)) << "qp_num is unique only per HCA";
  EXPECT_TRUE(rcclIbSendSliceDone(comm, 0, /*devIndex=*/0, qps[0].qp_num));
  EXPECT_FALSE(rcclIbSendSliceDone(comm, 0, /*devIndex=*/0, qps[0].qp_num));
  EXPECT_EQ(comm->sendCursor->qpState[0], RCCL_IB_SEND_QP_IDLE);
  EXPECT_EQ(comm->sendCursor->qpState[1], RCCL_IB_SEND_QP_INFLIGHT);
}

// A later slice that cannot post returns the post error and makes the comm fatal.
TEST_F(NetIbSegmentedSendPrefix, LaterSliceThatCannotPostIsFatal) {
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  g_post.failQpNum = qps[0].qp_num;
  g_post.mode = PostFail::Head;
  EXPECT_EQ(CompleteSlices(), ncclSystemError);
  EXPECT_GT(comm->base.stats.fatalErrorCount, 0);
  EXPECT_TRUE(comm->sendCursor->active);
}

// A referenced layout is read once on the slot's first QP; the send waits for it, then splits at its boundaries.
TEST_F(NetIbSegmentedSendPrefix, ReferencedLayoutIsReadBeforeTheSendPosts) {
  UseReferencedLayout();
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  EXPECT_EQ(request, &comm->base.reqs[0]);
  ASSERT_EQ(g_post.posts.size(), 1u);
  ASSERT_EQ(g_post.posts[0].qpNum, qps[0].qp_num);
  ASSERT_EQ(g_post.posts[0].wrs.size(), 1u);
  const WrRec& read = g_post.posts[0].wrs[0];
  EXPECT_EQ(read.opcode, IBV_WR_RDMA_READ);
  EXPECT_NE(read.flags & IBV_SEND_SIGNALED, 0u);
  EXPECT_EQ(read.wrId, 0u);
  EXPECT_EQ(read.raddr, kRefAddr);
  EXPECT_EQ(read.rkey, kRefRkey);
  EXPECT_EQ(read.laddr, reinterpret_cast<uint64_t>(refBlob));
  EXPECT_EQ(read.len, rcclIbLayoutBlobBytes(kRefSegs, 1));
  EXPECT_EQ(read.lkey, 0x3cu);
  EXPECT_EQ(comm->layoutCache->e[0].state, RCCL_IB_LAYOUT_READING);
  EXPECT_EQ(comm->sendCursor->queued, 1);

  ASSERT_EQ(rcclIbSendProgress(comm), ncclSuccess);
  EXPECT_EQ(g_post.calls, 1);

  FillReferencedLayout(refBlob);
  ASSERT_EQ(rcclIbSendLayoutReadDone(comm, 0), ncclSuccess);
  EXPECT_EQ(comm->layoutCache->e[0].state, RCCL_IB_LAYOUT_READY);
  ASSERT_EQ(rcclIbSendProgress(comm), ncclSuccess);
  EXPECT_EQ(comm->sendCursor->queued, 0);
  // The request starts 0x200 into the first remote segment, so the first slice ends at its 0x600 end.
  const std::vector<WrRec> q0 = WrsOn(0);
  ASSERT_GE(q0.size(), 2u);
  ExpectWrite(q0[1], kLocalBase, 0x400, 0x10, 0x900000, 0x90);
  Drain();
  EXPECT_EQ(comm->base.stats.fatalErrorCount, 0);
}

TEST_F(NetIbSegmentedSendPrefix, ReadLayoutThatIsInvalidFails) {
  UseReferencedLayout();
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  EXPECT_EQ(rcclIbSendLayoutReadDone(comm, 0), ncclInternalError);
  EXPECT_EQ(comm->layoutCache->e[0].state, RCCL_IB_LAYOUT_EMPTY);
  EXPECT_EQ(rcclIbSendLayoutReadDone(comm, RCCL_IB_LAYOUT_CACHE_ENTRIES), ncclInternalError);
}

// A referenced layout past the segment cap fails before its read is sized or posted.
TEST_F(NetIbSegmentedSendPrefix, ReferencedLayoutPastTheCapFailsBeforeReading) {
  UseReferencedLayout();
  rcclIbCtsLayout(ctsMem.cts[0], 1, 0)->ref.nSegments = RCCL_IB_REG_SEG_DEFAULT + 1;
  void* request = nullptr;
  EXPECT_EQ(Isend(&request), ncclSuccess);

  EXPECT_EQ(g_post.calls, 0);
  EXPECT_EQ(comm->layoutCache->e[0].state, RCCL_IB_LAYOUT_EMPTY);
  EXPECT_EQ(comm->base.reqs[0].type, NCCL_NET_IB_REQ_FAILED);
  EXPECT_GT(comm->base.stats.fatalErrorCount, 0);
}

// A layout already read is used again without another read.
TEST_F(NetIbSegmentedSendPrefix, ReadyLayoutIsNotReadAgain) {
  UseReferencedLayout();
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  FillReferencedLayout(refBlob);
  ASSERT_EQ(rcclIbSendLayoutReadDone(comm, 0), ncclSuccess);
  bool ready = false;
  ASSERT_EQ(rcclIbSendLayoutsReady(comm, 0, &ready), ncclSuccess);
  EXPECT_TRUE(ready);
  EXPECT_EQ(g_post.calls, 1);
}

// A re-registration at the same address publishes a new gen; the cached layout of the old one is not used.
TEST_F(NetIbSegmentedSendPrefix, NewGenerationAtTheSameAddressIsReadAgain) {
  UseReferencedLayout();
  void* request = nullptr;
  ASSERT_EQ(Isend(&request), ncclSuccess);
  FillReferencedLayout(refBlob);
  ASSERT_EQ(rcclIbSendLayoutReadDone(comm, 0), ncclSuccess);
  for (int k = 1; k < RCCL_IB_LAYOUT_CACHE_ENTRIES; k++) comm->layoutCache->e[k].state = RCCL_IB_LAYOUT_READING;
  rcclIbCtsLayout(ctsMem.cts[0], 1, 0)->ref.gen = kRefGen + 1;
  EXPECT_EQ(rcclIbLayoutCacheFind(comm->layoutCache, kRefAddr, kRefGen + 1), nullptr);

  bool ready = true;
  ASSERT_EQ(rcclIbSendLayoutsReady(comm, 0, &ready), ncclSuccess);
  EXPECT_FALSE(ready);
  ASSERT_EQ(g_post.posts.size(), 2u);
  EXPECT_EQ(g_post.posts[1].wrs[0].opcode, IBV_WR_RDMA_READ);
  EXPECT_EQ(g_post.posts[1].wrs[0].raddr, kRefAddr);
  EXPECT_EQ(comm->layoutCache->e[0].gen, kRefGen + 1);
  EXPECT_EQ(comm->layoutCache->e[0].state, RCCL_IB_LAYOUT_READING);
  EXPECT_EQ(rcclIbLayoutCacheFind(comm->layoutCache, kRefAddr, kRefGen), nullptr);
  for (int k = 1; k < RCCL_IB_LAYOUT_CACHE_ENTRIES; k++) comm->layoutCache->e[k].state = RCCL_IB_LAYOUT_EMPTY;
}

// With every entry ready, a new layout replaces the least recently used one.
TEST_F(NetIbSegmentedSendPrefix, NewLayoutEvictsTheLeastRecentlyUsedEntry) {
  UseReferencedLayout();
  for (int k = 0; k < RCCL_IB_LAYOUT_CACHE_ENTRIES; k++) {
    rcclIbLayoutCacheEntry& e = comm->layoutCache->e[k];
    e = comm->layoutCache->e[0];
    e.addr = 0x1000 + k;
    e.gen = 1;
    e.state = RCCL_IB_LAYOUT_READY;
    e.lastUse = (k == 7) ? 50 : 100 + k;
  }
  comm->layoutCache->clock = 200;
  comm->base.reqs[0].id = 0;
  comm->sendReqs[0][0] = &comm->base.reqs[0];
  bool ready = true;
  ASSERT_EQ(rcclIbSendLayoutsReady(comm, 0, &ready), ncclSuccess);
  EXPECT_FALSE(ready);
  ASSERT_EQ(g_post.posts.size(), 1u);
  EXPECT_EQ(g_post.posts[0].wrs[0].wrId, 7u);
  EXPECT_EQ(comm->layoutCache->e[7].addr, kRefAddr);
  EXPECT_EQ(comm->layoutCache->e[7].state, RCCL_IB_LAYOUT_READING);
}

// The flush posts one read, of the last non-zero receive's first byte, on the flush QP.
class NetIbSegmentedFlushPrefix : public NetIbSegmentedSendPrefix {
protected:
  void SetUp() override {
    NetIbSegmentedSendPrefix::SetUp();
    const size_t bytes = (sizeof(IbMsegRecvComm) + 63) / 64 * 64;
    rcomm = static_cast<IbMsegRecvComm*>(std::aligned_alloc(64, bytes));
    ASSERT_NE(rcomm, nullptr);
    std::memset(rcomm, 0, bytes);
    rcomm->base.ready = 1;
    rcomm->base.vProps.ndevs = 1;
    rcomm->flushEnabled = 1;
    rcomm->flushQpsCreated = true;
    rcomm->devs[0].gpuFlush.qp.qp = &qps[0];
  }

  void TearDown() override {
    std::free(rcomm);
    NetIbSegmentedSendPrefix::TearDown();
  }

  ncclResult_t Iflush(void** request) {
    *request = nullptr;
    void* data[1] = {reinterpret_cast<void*>(kLocalBase)};
    int sizes[1] = {static_cast<int>(kSize)};
    void* mhandles[1] = {&mh};
    return IbMsegIflush(rcomm, 1, data, sizes, mhandles, request);
  }

  ncclResult_t IflushN(int n, void** data, int* sizes, void** mhandles, void** request) {
    *request = nullptr;
    return IbMsegIflush(rcomm, n, data, sizes, mhandles, request);
  }

  static void ExpectRead(const WrRec& w, uint64_t raddr, uint32_t rkey) {
    EXPECT_EQ(w.opcode, IBV_WR_RDMA_READ);
    EXPECT_EQ(w.raddr, raddr);
    EXPECT_EQ(w.rkey, rkey);
    EXPECT_EQ(w.nsge, 1);
  }

  // The flush failed with nothing left in flight: the request is freed and the comm goes fatal.
  void ExpectFlushFreed(void* request) const {
    EXPECT_EQ(request, nullptr);
    EXPECT_EQ(rcomm->base.reqs[0].type, NCCL_NET_IB_REQ_UNUSED);
    EXPECT_EQ(rcomm->base.reqs[0].events[0], 0);
    EXPECT_GT(rcomm->base.stats.fatalErrorCount, 0);
  }

  IbMsegRecvComm* rcomm = nullptr;
};

// One read of the receive's first byte, with that segment's rkey, signaled.
TEST_F(NetIbSegmentedFlushPrefix, ReadsTheFirstByteOfTheLastReceive) {
  void* request = nullptr;
  ASSERT_EQ(Iflush(&request), ncclSuccess);
  EXPECT_EQ(request, &rcomm->base.reqs[0]);
  ASSERT_EQ(g_post.posts.size(), 1u);
  const std::vector<WrRec>& w = g_post.posts[0].wrs;
  ASSERT_EQ(w.size(), 1u);
  ExpectRead(w[0], 0x100000, 0x20);
  EXPECT_EQ(w[0].sgList, &rcomm->devs[0].gpuFlush.sge);
  EXPECT_NE(w[0].flags & IBV_SEND_SIGNALED, 0u);
  EXPECT_EQ(w[0].wrId, static_cast<uint64_t>(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET)); // request index 0
  EXPECT_EQ(rcomm->base.reqs[0].type, NCCL_NET_IB_REQ_FLUSH);
  EXPECT_EQ(rcomm->base.reqs[0].events[0], 1);
}

// A receive that starts in segment 0 and crosses into segment 1 still reads segment 0 only.
TEST_F(NetIbSegmentedFlushPrefix, CrossingReadUsesTheFirstBytesSegment) {
  void* data[1] = {reinterpret_cast<void*>(kLocalBase + 0x400)};
  int sizes[1] = {0x800};
  void* mhandles[1] = {&mh};
  void* request = nullptr;
  ASSERT_EQ(IflushN(1, data, sizes, mhandles, &request), ncclSuccess);
  ASSERT_EQ(g_post.posts.size(), 1u);
  const std::vector<WrRec>& w = g_post.posts[0].wrs;
  ASSERT_EQ(w.size(), 1u);
  ExpectRead(w[0], 0x100400, 0x20);
}

// A multi-recv reads only the last non-zero receive. The earlier segmented receive is not read.
TEST_F(NetIbSegmentedFlushPrefix, MultiRecvReadsOnlyTheLastNonZeroReceive) {
  void* data[3] = {reinterpret_cast<void*>(kLocalBase), reinterpret_cast<void*>(0x300000),
                   reinterpret_cast<void*>(0x400000)};
  int sizes[3] = {static_cast<int>(kSize), 0x100, 0};
  void* mhandles[3] = {&mh, &mhSingle, &mhSingle};
  void* request = nullptr;
  ASSERT_EQ(IflushN(3, data, sizes, mhandles, &request), ncclSuccess);
  ASSERT_EQ(g_post.posts.size(), 1u);
  const std::vector<WrRec>& w = g_post.posts[0].wrs;
  ASSERT_EQ(w.size(), 1u);
  ExpectRead(w[0], 0x300000, 0x31);
  EXPECT_NE(w[0].flags & IBV_SEND_SIGNALED, 0u);
}

// The RO=0 scratchpad replaces the direct read with one write and one read, chained in one post.
TEST_F(NetIbSegmentedFlushPrefix, ScratchpadChainsOneWriteAndOneReadInOnePost) {
  alignas(64) int scratch[16] = {};
  ibv_mr scratchMr;
  std::memset(&scratchMr, 0, sizeof(scratchMr));
  scratchMr.rkey = 0x44;
  rcomm->devs[0].gpuFlush.gpuFlushGpuMem = scratch;
  rcomm->devs[0].gpuFlush.gpuMr = &scratchMr;
  void* request = nullptr;
  ASSERT_EQ(Iflush(&request), ncclSuccess);
  rcomm->devs[0].gpuFlush.gpuFlushGpuMem = nullptr;
  rcomm->devs[0].gpuFlush.gpuMr = nullptr;

  ASSERT_EQ(g_post.posts.size(), 1u);
  const std::vector<WrRec>& w = g_post.posts[0].wrs;
  ASSERT_EQ(w.size(), 2u);
  EXPECT_EQ(w[0].opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(w[0].raddr, reinterpret_cast<uint64_t>(scratch));
  EXPECT_EQ(w[0].rkey, 0x44u);
  EXPECT_EQ(w[0].flags, static_cast<unsigned>(IBV_SEND_INLINE));
  ExpectRead(w[1], reinterpret_cast<uint64_t>(scratch), 0x44);
  EXPECT_NE(w[1].flags & IBV_SEND_SIGNALED, 0u);
  for (const WrRec& r : w) {
    EXPECT_EQ(r.wrId, static_cast<uint64_t>(NCCL_IB_FLUSH_REQ_WR_ID_OFFSET));
  }
}

// A receive outside every registered segment is an error and nothing posts.
TEST_F(NetIbSegmentedFlushPrefix, RangeOutsideEverySegmentFails) {
  void* data[1] = {reinterpret_cast<void*>(0x200000)};
  int sizes[1] = {0x100};
  void* mhandles[1] = {&mh};
  void* request = nullptr;
  EXPECT_EQ(IflushN(1, data, sizes, mhandles, &request), ncclInternalError);
  EXPECT_EQ(g_post.calls, 0);
  ExpectFlushFreed(request);
}

// A missing MR for the first byte is an error and nothing posts.
TEST_F(NetIbSegmentedFlushPrefix, MissingSegmentMrFails) {
  mh.seg.segMrs[0] = nullptr;
  mh.mrs[0] = nullptr;
  void* request = nullptr;
  EXPECT_EQ(Iflush(&request), ncclInternalError);
  EXPECT_EQ(g_post.calls, 0);
  ExpectFlushFreed(request);
}

// Nothing posted, so the request is freed and the post error is returned.
TEST_F(NetIbSegmentedFlushPrefix, RejectsWholeChainFreesRequest) {
  g_post.failQpNum = qps[0].qp_num;
  g_post.mode = PostFail::Head;
  void* request = nullptr;
  EXPECT_EQ(Iflush(&request), ncclSystemError);

  EXPECT_EQ(g_post.calls, 1);
  EXPECT_EQ(g_post.failedCallWrs, 1);
  EXPECT_EQ(g_post.failedCallSignaled, 1);
  ExpectFlushFreed(request);
}

// The signaled read is the only WR, so a rejected post frees the request.
TEST_F(NetIbSegmentedFlushPrefix, LostReadFreesRequest) {
  g_post.failQpNum = qps[0].qp_num;
  g_post.mode = PostFail::Tail;
  void* request = nullptr;
  EXPECT_EQ(Iflush(&request), ncclSystemError);

  EXPECT_EQ(g_post.calls, 1);
  EXPECT_EQ(g_post.failedCallWrs, 1);
  ExpectFlushFreed(request);
}

// The multi-segment hooks p2p_multiseg.cc exports to p2p.cc, called directly.
class NetIbSegmentedHooks : public NetIbSegmentedFlushPrefix {
protected:
  static constexpr uint64_t kCtsAddr = 0xb00000;
  static constexpr uint32_t kCtsRkey = 0x55;
  static constexpr int kDevIdx = 0;
  static constexpr int kSideSlot = 1;
  static constexpr int kPostedNreqs = 2;
  static constexpr uint64_t kPostedWrId = 0x1234;

  void SetUp() override {
    NetIbSegmentedFlushPrefix::SetUp();
    localSlots = static_cast<IbMsegCtsSlot*>(std::aligned_alloc(64, 2 * sizeof(IbMsegCtsSlot)));
    ASSERT_NE(localSlots, nullptr);
    std::memset(localSlots, 0, 2 * sizeof(IbMsegCtsSlot));
    rcomm->ctsSlots = localSlots;
    rcomm->ctsSlotStride = sizeof(IbMsegCtsSlot);
    rcomm->remCtsFifo.addr = kCtsAddr;
    rcomm->base.remDevs[kDevIdx].rkey = kCtsRkey;
    comm->sendReqs[0][0] = &comm->base.reqs[0];
  }

  void TearDown() override {
    std::free(localSlots);
    NetIbSegmentedFlushPrefix::TearDown();
  }

  // An inline CTS write of slot as p2p.cc builds it, before rcclIbPrepareCtsWrite adjusts it.
  void PrepareCtsWrite(IbMsegRequest* req, int slot, ibv_send_wr* wr, ibv_sge* sge) {
    std::memset(wr, 0, sizeof(*wr));
    std::memset(sge, 0, sizeof(*sge));
    wr->sg_list = sge;
    wr->num_sge = 1;
    wr->wr_id = kPostedWrId;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = IBV_SEND_INLINE;
    wr->wr.rdma.rkey = kCtsRkey;
    rcclIbPrepareCtsWrite(rcomm, req, slot, wr);
  }

  // Receive i of an nreqs-entry slot 0 over [addr, addr + size).
  void PostReceive(int i, int nreqs, uint64_t addr, uint64_t size) {
    localSlots[0].cts[i].nreqs = nreqs;
    localSlots[0].cts[i].addr = addr;
    localSlots[0].cts[i].size = size;
  }

  static constexpr int kWideSegs = RCCL_IB_CTS_INLINE_SEGS + 2;
  static constexpr uint64_t kWideBase = 0x400000;
  static constexpr uint64_t kWideSeg = 0x1000;

  // A kWideSegs-segment registration whose layout is published, as reg.cc does past the inline limit.
  void UseWideRegistration() {
    std::memset(&wide, 0, sizeof(wide));
    std::memset(wideSegMrs, 0, sizeof(wideSegMrs));
    wide.seg.nSegments = kWideSegs;
    wide.seg.segStart = wideStart;
    wide.seg.segLen = wideLen;
    wide.seg.segOff = wideOff;
    wide.seg.segMrs = wideSegMrs;
    for (int s = 0; s < kWideSegs; s++) {
      wideMrs[s] = {};
      wideMrs[s].rkey = 0x40 + s;
      wideStart[s] = kWideBase + s * kWideSeg;
      wideLen[s] = kWideSeg;
      wideOff[s] = s * kWideSeg;
      wideSegMrs[s * NCCL_IB_MAX_DEVS_PER_NIC] = &wideMrs[s];
    }
    wideOff[kWideSegs] = kWideSegs * kWideSeg;
    wide.mrs[0] = &wideMrs[0];
    wideLayoutMr = {};
    wideLayoutMr.rkey = 0x5a;
    wide.seg.layoutBlob = wideBlob;
    wide.seg.layoutBytes = sizeof(wideBlob);
    wide.seg.layoutGen = 9;
    wide.seg.layoutMrs[0] = &wideLayoutMr;
  }

  IbMsegCtsSlot* localSlots = nullptr;
  IbMsegMrHandle wide;
  ibv_mr wideMrs[kWideSegs];
  ibv_mr wideLayoutMr;
  uintptr_t wideStart[kWideSegs];
  size_t wideLen[kWideSegs];
  uint64_t wideOff[kWideSegs + 1];
  void* wideSegMrs[kWideSegs * NCCL_IB_MAX_DEVS_PER_NIC];
  uint64_t wideBlob[32];
};

TEST_F(NetIbSegmentedHooks, MultiSendNeedsSegmentedForAMultiSegmentSource) {
  comm->base.reqs[0].send.mh = &mh;
  EXPECT_TRUE(rcclIbMultiSendNeedsSegmented(comm, 0, 1));
}

TEST_F(NetIbSegmentedHooks, MultiSendNeedsSegmentedForAMultiSegmentDestination) {
  comm->base.reqs[0].send.mh = &mhSingle;
  UseRemoteLayout({0x900000, 0x900c00}, {0x81, 0x82});
  EXPECT_TRUE(rcclIbMultiSendNeedsSegmented(comm, 0, 1));
}

// One segment on each side, or a remote table that does not apply to this CTS, stays on the legacy builder.
TEST_F(NetIbSegmentedHooks, MultiSendStaysLegacyForSingleSegmentEnds) {
  comm->base.reqs[0].send.mh = &mhSingle;
  EXPECT_FALSE(rcclIbMultiSendNeedsSegmented(comm, 0, 1));
  UseRemoteLayout({0x900000}, {0x81});
  EXPECT_FALSE(rcclIbMultiSendNeedsSegmented(comm, 0, 1));
  UseRemoteLayout({0x900000, 0x900c00}, {0x81, 0x82});
  ctsMem.cts[0][0].nreqs = IbMsegCtsNreqs(ctsMem.cts[0][0].nreqs);
  EXPECT_FALSE(rcclIbMultiSendNeedsSegmented(comm, 0, 1));
  ctsMem.cts[0][0].nreqs |= NCCL_IB_CTS_NREQS_LAYOUT;
  comm->useMultiSeg = 0;
  EXPECT_FALSE(rcclIbMultiSendNeedsSegmented(comm, 0, 1));
}

TEST_F(NetIbSegmentedHooks, PostFifoExtendsOneWriteOverTheLayoutPrefix) {
  rcomm->useMultiSeg = 1;
  localSlots[kSideSlot].cts[kDevIdx].nreqs = kPostedNreqs | NCCL_IB_CTS_NREQS_LAYOUT;
  IbMsegRequest* req = &rcomm->base.reqs[0];
  req->nreqs = kPostedNreqs;
  ibv_send_wr ctsWr;
  ibv_sge sge;
  PrepareCtsWrite(req, kSideSlot, &ctsWr, &sge);
  const uint64_t prefix = (uint64_t)kPostedNreqs * sizeof(IbMsegSegLayout);
  EXPECT_EQ(ctsWr.wr_id, kPostedWrId);
  EXPECT_EQ(ctsWr.opcode, IBV_WR_RDMA_WRITE);
  EXPECT_EQ(ctsWr.send_flags & IBV_SEND_INLINE, 0u);
  EXPECT_EQ(ctsWr.wr.rdma.remote_addr, kCtsAddr + kSideSlot * sizeof(IbMsegCtsSlot) - prefix);
  EXPECT_EQ(ctsWr.wr.rdma.rkey, kCtsRkey);
  EXPECT_EQ(sge.addr, reinterpret_cast<uint64_t>(rcclIbCtsLayout(localSlots[kSideSlot].cts, kPostedNreqs, 0)));
  EXPECT_EQ(sge.length, prefix + kPostedNreqs * sizeof(IbMsegSendFifo));
}

TEST_F(NetIbSegmentedHooks, PostFifoKeepsTheShortWriteWithoutTheBit) {
  rcomm->useMultiSeg = 1;
  localSlots[0].cts[0].nreqs = 1;
  IbMsegRequest* req = &rcomm->base.reqs[0];
  req->nreqs = 1;
  ibv_send_wr ctsWr;
  ibv_sge sge;
  PrepareCtsWrite(req, 0, &ctsWr, &sge);
  EXPECT_EQ(ctsWr.wr.rdma.remote_addr, kCtsAddr);
  EXPECT_EQ(sge.addr, reinterpret_cast<uint64_t>(localSlots[0].cts));
  EXPECT_EQ(sge.length, sizeof(IbMsegSendFifo));
  EXPECT_EQ(ctsWr.send_flags & IBV_SEND_INLINE, IBV_SEND_INLINE);
}

TEST_F(NetIbSegmentedHooks, IrecvAcceptsAHandleWithEveryDeviceMr) {
  EXPECT_EQ(rcclIbIrecvCheckHandle(rcomm, &mhSingle, 0), ncclSuccess);
  EXPECT_EQ(rcclIbIrecvCheckHandle(rcomm, &mh, 0), ncclSuccess);
}

TEST_F(NetIbSegmentedHooks, IrecvRejectsANullHandleOrAMissingDeviceMr) {
  EXPECT_EQ(rcclIbIrecvCheckHandle(rcomm, nullptr, 0), ncclInternalError);
  mhSingle.mrs[0] = nullptr;
  EXPECT_EQ(rcclIbIrecvCheckHandle(rcomm, &mhSingle, 0), ncclInternalError);
}

// The layout prefix carries each overlapped segment's start and rkey. The CTS idx stays the arrival flag.
TEST_F(NetIbSegmentedHooks, IrecvFillsTheLayoutForAMultiSegmentReceive) {
  rcomm->useMultiSeg = 1;
  PostReceive(1, 2, kLocalBase, kSize);
  bool anyMultiSeg = false;
  ASSERT_EQ(rcclIbIrecvFillLayout(rcomm, 0, 1, &mh, &anyMultiSeg), ncclSuccess);
  EXPECT_TRUE(anyMultiSeg);
  const IbMsegSegLayout& e = *rcclIbCtsLayout(localSlots[0].cts, 2, 1);
  ASSERT_EQ(e.nSegments, 3u);
  EXPECT_EQ(e.byRef, 0u);
  for (int s = 0; s < 3; s++) {
    EXPECT_EQ(e.segStart[s], mh.seg.segStart[s]);
    EXPECT_EQ(rcclIbLayoutRkey(&e, s, 0), 0x20u + s);
  }
}

// Only the segments the receive overlaps are carried, starting with the one holding its first byte.
TEST_F(NetIbSegmentedHooks, IrecvCarriesOnlyTheOverlappedSegments) {
  rcomm->useMultiSeg = 1;
  PostReceive(0, 1, kLocalBase + 0x900, 0x1000);
  bool anyMultiSeg = false;
  ASSERT_EQ(rcclIbIrecvFillLayout(rcomm, 0, 0, &mh, &anyMultiSeg), ncclSuccess);
  EXPECT_TRUE(anyMultiSeg);
  const IbMsegSegLayout& e = *rcclIbCtsLayout(localSlots[0].cts, 1, 0);
  ASSERT_EQ(e.nSegments, 2u);
  EXPECT_EQ(e.segStart[0], mh.seg.segStart[1]);
  EXPECT_EQ(e.segStart[1], mh.seg.segStart[2]);
  EXPECT_EQ(rcclIbLayoutRkey(&e, 0, 0), 0x21u);
  EXPECT_EQ(rcclIbLayoutRkey(&e, 1, 0), 0x22u);
}

// A receive inside one segment keeps the short CTS and carries that segment's rkey.
TEST_F(NetIbSegmentedHooks, IrecvInsideOneSegmentSendsTheShortCtsWithItsRkey) {
  rcomm->useMultiSeg = 1;
  PostReceive(0, 1, kLocalBase + 0x900, 0x100);
  localSlots[0].cts[0].rkeys[0] = mrs[0].rkey;
  bool anyMultiSeg = false;
  ASSERT_EQ(rcclIbIrecvFillLayout(rcomm, 0, 0, &mh, &anyMultiSeg), ncclSuccess);
  EXPECT_FALSE(anyMultiSeg);
  EXPECT_EQ(rcclIbCtsLayout(localSlots[0].cts, 1, 0)->nSegments, 0u);
  EXPECT_EQ(localSlots[0].cts[0].rkeys[0], 0x21u);
}

// Past the inline limit the CTS references the registration's published layout.
TEST_F(NetIbSegmentedHooks, IrecvPastTheInlineLimitSendsAReference) {
  rcomm->useMultiSeg = 1;
  UseWideRegistration();
  PostReceive(0, 1, kWideBase + 0x100, kWideSegs * kWideSeg - 0x200);
  bool anyMultiSeg = false;
  ASSERT_EQ(rcclIbIrecvFillLayout(rcomm, 0, 0, &wide, &anyMultiSeg), ncclSuccess);
  EXPECT_TRUE(anyMultiSeg);
  const IbMsegSegLayout& e = *rcclIbCtsLayout(localSlots[0].cts, 1, 0);
  EXPECT_EQ(e.byRef, 1u);
  EXPECT_EQ(e.nSegments, 0u);
  EXPECT_EQ(e.ref.addr, reinterpret_cast<uint64_t>(wide.seg.layoutBlob));
  EXPECT_EQ(e.ref.gen, wide.seg.layoutGen);
  EXPECT_EQ(e.ref.nSegments, static_cast<uint32_t>(kWideSegs));
  EXPECT_EQ(e.ref.ndevs, 1u);
  EXPECT_EQ(e.ref.rkeys[0], wideLayoutMr.rkey);
}

TEST_F(NetIbSegmentedHooks, IrecvPastTheInlineLimitWithoutAPublishedLayoutFails) {
  rcomm->useMultiSeg = 1;
  UseWideRegistration();
  wide.seg.layoutBlob = nullptr;
  PostReceive(0, 1, kWideBase, kWideSegs * kWideSeg);
  bool anyMultiSeg = false;
  EXPECT_EQ(rcclIbIrecvFillLayout(rcomm, 0, 0, &wide, &anyMultiSeg), ncclInternalError);
}

TEST_F(NetIbSegmentedHooks, IrecvRangeOutsideTheRegistrationFails) {
  rcomm->useMultiSeg = 1;
  PostReceive(0, 1, kLocalBase + 0x1f00, 0x200);
  bool anyMultiSeg = false;
  EXPECT_EQ(rcclIbIrecvFillLayout(rcomm, 0, 0, &mh, &anyMultiSeg), ncclInternalError);
}

TEST_F(NetIbSegmentedHooks, IrecvPublishesAnEmptyLayoutForASingleSegmentReceive) {
  rcomm->useMultiSeg = 1;
  localSlots[0].cts[0].nreqs = 1;
  rcclIbCtsLayout(localSlots[0].cts, 1, 0)->nSegments = 7;
  bool anyMultiSeg = false;
  ASSERT_EQ(rcclIbIrecvFillLayout(rcomm, 0, 0, &mhSingle, &anyMultiSeg), ncclSuccess);
  EXPECT_FALSE(anyMultiSeg);
  EXPECT_EQ(rcclIbCtsLayout(localSlots[0].cts, 1, 0)->nSegments, 0u);
}

// A comm on the legacy CTS does not write the layout prefix.
TEST_F(NetIbSegmentedHooks, IrecvSkipsTheLayoutWhenMultiSegIsOff) {
  rcomm->useMultiSeg = 0;
  localSlots[0].cts[0].nreqs = 1;
  rcclIbCtsLayout(localSlots[0].cts, 1, 0)->nSegments = 7;
  bool anyMultiSeg = false;
  EXPECT_EQ(rcclIbIrecvFillLayout(rcomm, 0, 0, &mh, &anyMultiSeg), ncclSuccess);
  EXPECT_FALSE(anyMultiSeg);
  EXPECT_EQ(rcclIbCtsLayout(localSlots[0].cts, 1, 0)->nSegments, 7u);
}

TEST_F(NetIbSegmentedHooks, IrecvMissingSegmentMrFailsBeforeFinishingTheLayout) {
  rcomm->useMultiSeg = 1;
  PostReceive(0, 1, kLocalBase, kSize);
  mh.seg.segMrs[1 * NCCL_IB_MAX_DEVS_PER_NIC] = nullptr;
  bool anyMultiSeg = false;
  EXPECT_EQ(rcclIbIrecvFillLayout(rcomm, 0, 0, &mh, &anyMultiSeg), ncclInternalError);
  EXPECT_EQ(rcclIbCtsLayout(localSlots[0].cts, 1, 0)->nSegments, 0u);
  EXPECT_EQ(rcclIbLayoutRkey(rcclIbCtsLayout(localSlots[0].cts, 1, 0), 0, 0), 0u);
}

TEST_F(NetIbSegmentedHooks, CtsFlagSetsTheLayoutBitOnEachReceiveOnly) {
  IbMsegSendFifo* row = localSlots[0].cts;
  for (int i = 0; i < 3; i++) row[i].nreqs = 2;
  rcclIbCtsFlagLayout(row, 2, /*idx=*/9);
  for (int i = 0; i < 2; i++) {
    EXPECT_TRUE(IbMsegCtsHasLayout(row[i].nreqs));
    EXPECT_EQ(IbMsegCtsNreqs(row[i].nreqs), 2u);
  }
  EXPECT_EQ(row[2].nreqs, 2);
}

// Each 64-byte line of every layout in the write carries idx, so a line that lands late is not read stale.
TEST_F(NetIbSegmentedHooks, CtsFlagStampsEveryLayoutLineAndArrivalNeedsAll) {
  IbMsegSendFifo* row = localSlots[0].cts;
  for (int i = 0; i < 2; i++) row[i].nreqs = 2;
  rcclIbCtsFlagLayout(row, 2, /*idx=*/9);
  for (int r = 0; r < 2; r++) {
    const IbMsegSegLayout* l = rcclIbCtsLayout(row, 2, r);
    EXPECT_EQ(l->idx0, 9u);
    EXPECT_EQ(l->idx1, 9u);
    EXPECT_EQ(l->idx2, 9u);
  }
  EXPECT_TRUE(rcclIbCtsLayoutsArrived(row, 2, 9));
  EXPECT_FALSE(rcclIbCtsLayoutsArrived(row, 2, 10));
  rcclIbCtsLayout(row, 2, 1)->idx1 = 8;
  EXPECT_FALSE(rcclIbCtsLayoutsArrived(row, 2, 9)) << "a stale middle line of the second layout";
  rcclIbCtsLayout(row, 2, 1)->idx1 = 9;
  rcclIbCtsLayout(row, 2, 0)->idx2 = 8;
  EXPECT_FALSE(rcclIbCtsLayoutsArrived(row, 2, 9)) << "a stale last line of the first layout";
}

// Every inline (segment, device) rkey has its own word, and none overlaps a stamp or the header.
TEST_F(NetIbSegmentedHooks, InlineRkeysMapToDistinctWordsOutsideTheStamps) {
  IbMsegSegLayout l;
  std::memset(&l, 0, sizeof(l));
  l.nSegments = RCCL_IB_CTS_INLINE_SEGS;
  for (int s = 0; s < RCCL_IB_CTS_INLINE_SEGS; s++) l.segStart[s] = 0x1000u * (s + 1);
  rcclIbLayoutStamp(&l, 0x55);
  for (int s = 0; s < RCCL_IB_CTS_INLINE_SEGS; s++)
    for (int d = 0; d < RCCL_IB_MSEG_MAX_DEVS; d++)
      rcclIbLayoutSetRkey(&l, s, d, 0x100u + s * RCCL_IB_MSEG_MAX_DEVS + d);
  for (int s = 0; s < RCCL_IB_CTS_INLINE_SEGS; s++) {
    EXPECT_EQ(l.segStart[s], 0x1000u * (s + 1));
    for (int d = 0; d < RCCL_IB_MSEG_MAX_DEVS; d++)
      EXPECT_EQ(rcclIbLayoutRkey(&l, s, d), 0x100u + s * RCCL_IB_MSEG_MAX_DEVS + d)
          << "segment " << s << " device " << d;
  }
  EXPECT_EQ(l.nSegments, static_cast<uint32_t>(RCCL_IB_CTS_INLINE_SEGS));
  EXPECT_EQ(l.byRef, 0u);
  EXPECT_TRUE(rcclIbLayoutArrived(&l, 0x55));
}

}  // namespace
