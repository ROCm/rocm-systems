/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Proxy ops: geteActivationMask, gettaskEventHandle, addProxyOpIfNeeded and ncclAddProxyOpIfNeeded.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// geteActivationMask / gettaskEventHandle (enqueue.cc:1820, :1830)
//
// Identical 3-arm shape. The first arm is a RANGE test over the ncclFunc_t
// enum, so it silently depends on SendRecv/Send/Recv staying adjacent. If they
// are reordered, the wrong union member is read -- a type-confused pointer
// deref. These tests pin the range boundaries specifically.
// ===========================================================================

namespace {
// Distinct sentinels so an arm returning the wrong member is visible.
constexpr int kP2pMask = 0x5A;
constexpr int kCollMask = 0xA5;

struct ActivationOp {
  ncclProxyOp op{};
  ncclTaskP2p p2p{};
  ncclTaskColl coll{};
  explicit ActivationOp(ncclFunc_t f) {
    p2p.eActivationMask = kP2pMask;
    coll.eActivationMask = kCollMask;
    // NCCL 2.32.6: the handle is a per-plugin ncclProfilerEventRef. Self-address the first slot: unique,
    // non-null, comparable.
    p2p.eventHandle.h[0] = &p2p;
    coll.eventHandle.h[0] = &coll;
    op.coll = f;
    op.task.p2p = &p2p;        // union: set p2p last for p2p funcs
  }
  void UseColl() { op.task.coll = &coll; }
};

bool IsNullEventRef(const ncclProfilerEventRef& ref) {
  for (void* h : ref.h)
    if (h != nullptr) return false;
  return true;
}
}  // namespace

TEST_F(EnqueueMicrotest, GeteActivationMask_P2pRange_ReadsP2pTask) {
  // All three of SendRecv/Send/Recv must take the p2p arm.
  for (ncclFunc_t f : {ncclFuncSendRecv, ncclFuncSend, ncclFuncRecv}) {
    ActivationOp a(f);
    EXPECT_EQ(kP2pMask, geteActivationMask(&a.op)) << "func=" << int(f);
    EXPECT_EQ(&a.p2p, gettaskEventHandle(&a.op).h[0]) << "func=" << int(f);
  }
}

TEST_F(EnqueueMicrotest, GeteActivationMask_AllGatherV_IsZeroAndNullHandle) {
  ActivationOp a(ncclFuncAllGatherV);
  a.UseColl();  // the coll member is populated, yet this arm must ignore it
  EXPECT_EQ(0, geteActivationMask(&a.op));
  EXPECT_TRUE(IsNullEventRef(gettaskEventHandle(&a.op)));
}

TEST_F(EnqueueMicrotest, GeteActivationMask_CollFunc_ReadsCollTask) {
  for (ncclFunc_t f : {ncclFuncAllReduce, ncclFuncBroadcast, ncclFuncReduce,
                       ncclFuncAllGather, ncclFuncReduceScatter}) {
    ActivationOp a(f);
    a.UseColl();
    EXPECT_EQ(kCollMask, geteActivationMask(&a.op)) << "func=" << int(f);
    EXPECT_EQ(&a.coll, gettaskEventHandle(&a.op).h[0]) << "func=" << int(f);
  }
}

TEST_F(EnqueueMicrotest, GeteActivationMask_RangeBoundariesAreExact) {
  // The arm is `ncclFuncSendRecv <= coll && coll <= ncclFuncRecv`. Pin that the
  // enum neighbours on each side do NOT take it -- this is what breaks if the
  // ncclFunc_t enum is ever reordered.
  ActivationOp below(static_cast<ncclFunc_t>(int(ncclFuncSendRecv) - 1));
  below.UseColl();
  EXPECT_EQ(kCollMask, geteActivationMask(&below.op)) << "just below the range";

  ActivationOp above(static_cast<ncclFunc_t>(int(ncclFuncRecv) + 1));
  above.UseColl();
  EXPECT_EQ(kCollMask, geteActivationMask(&above.op)) << "just above the range";
}

// ===========================================================================
// addProxyOpIfNeeded (enqueue.cc:198) / ncclAddProxyOpIfNeeded (:771)
// A proxy op is only recorded when the transport says it is needed; the
// "justInquire" protocol means ncclProxySaveOp is asked FIRST whether the op
// matters, and only then is memory allocated for it.
// ===========================================================================

TEST_F(EnqueueMicrotest, AddProxyOpIfNeeded_NotNeeded_DoesNotEnqueue) {
  // justInquire comes back false -> nothing is queued, but the inquiry still
  // happened (the call counter is what distinguishes this from "never asked").
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 0;
  g_proxySaveOpJustInquire = false;
  ASSERT_EQ(ncclSuccess, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_EQ(1, g_proxySaveOpCalls);
  EXPECT_TRUE(ChannelProxyEmpty(bp, 0));
}

TEST_F(EnqueueMicrotest, AddProxyOpIfNeeded_Needed_EnqueuesACopy) {
  // justInquire true -> the op is copied into comm-owned memory and queued. The
  // copy must be a DISTINCT object, since the caller's op is a stack temporary
  // that dies before the plan is launched.
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 0;
  op.opCount = 0xABCD;
  g_proxySaveOpJustInquire = true;
  ASSERT_EQ(ncclSuccess, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_EQ(1, g_proxySaveOpCalls);

  // The INQUIRY must be about this comm and this op. The queued copy below is
  // made from the caller's op regardless, so without these the transport could
  // be handed a different comm/op and the test would still pass.
  EXPECT_EQ(bp.c(), g_proxySaveOpLastComm) << "must inquire with the caller's comm";
  EXPECT_EQ(&op, g_proxySaveOpLastOp) << "must inquire about the caller's op";
  EXPECT_EQ(uint64_t(0xABCD), g_proxySaveOpLastOpCount);

  auto* head = ChannelProxyHead(bp, 0);
  ASSERT_NE(nullptr, head);
  EXPECT_NE(&op, head) << "must copy, not alias the caller's stack object";
  EXPECT_EQ(uint64_t(0xABCD), head->opCount) << "and the copy must carry the contents";
}

TEST_F(EnqueueMicrotest, AddProxyOpIfNeeded_InquiresBeforeAllocating) {
  // The protocol is "ask first, allocate second", and what signals inquiry is
  // passing a NON-NULL justInquire pointer -- not its pointee. Real
  // ncclProxySaveOp overwrites it with false as its first statement (proxy.cc:631),
  // so the `bool needed = true` seed at :199 is never read and asserting on the
  // incoming value would pin our fake rather than production.
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 2;
  op.opCount = 0x77;
  g_proxySaveOpJustInquire = false;
  ASSERT_EQ(ncclSuccess, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_TRUE(g_proxySaveOpSawNonNullJustInquire)
      << "production must ask (non-null justInquire), not tell";
  EXPECT_EQ(2, g_proxySaveOpLastChannelId) << "and must inquire about the op's own channel";
}

TEST_F(EnqueueMicrotest, AddProxyOpIfNeeded_RoutesToTheOpsOwnChannel) {
  // The destination is indexed by op->channelId. A hardcoded channel 0 would
  // pass every other test here but fail this one.
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 3;
  op.opCount = 0x1234;
  g_proxySaveOpJustInquire = true;
  ASSERT_EQ(ncclSuccess, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_EQ(3, g_proxySaveOpLastChannelId) << "the inquiry carries the op's channel";
  EXPECT_TRUE(ChannelProxyEmpty(bp, 0)) << "must not land on channel 0";
  auto* head = ChannelProxyHead(bp, 3);
  ASSERT_NE(nullptr, head);
  EXPECT_EQ(uint64_t(0x1234), head->opCount);
}

TEST_F(EnqueueMicrotest, AddProxyOpIfNeeded_SaveOpFailure_Propagates) {
  // NCCLCHECK on the inquiry: a failure must abandon the add, not queue a
  // half-initialised op.
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 0;
  g_proxySaveOpResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_TRUE(ChannelProxyEmpty(bp, 0));
}
