// ===========================================================================
// waitWorkFifoAvailable (enqueue.cc:1652)
//
// The room test is UNSIGNED WRAPAROUND arithmetic:
//     (desiredProduced - workFifoConsumed) <= workFifoBytes
// Both operands are uint32_t, so it stays correct as the counters roll past
// 2^32 -- which is exactly the case a naive `desired <= consumed + bytes`
// rewrite would break. Several tests below sit deliberately at the rollover.
// ===========================================================================

namespace {
struct FifoComm {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  uint32_t abortStorage = 0;
  FifoComm(uint32_t consumed, uint32_t bytes) {
    comm->workFifoConsumed = consumed;
    comm->workFifoBytes = bytes;
    comm->rank = 0;
    comm->abortFlag = &abortStorage;
  }
  ncclComm* c() { return comm.get(); }
};
}  // namespace

TEST_F(EnqueueMicrotest, WaitWorkFifo_RoomAvailable_ReturnsImmediately) {
  // The fast path. hasRoom is true on entry so the loop never runs -- proven by
  // arming the poll to FAIL: reaching it at all would surface as an error.
  FifoComm f(/*consumed=*/1000, /*bytes=*/4096);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclSuccess, waitWorkFifoAvailable(f.c(), /*desiredProduced=*/2000));
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_ExactlyFull_IsStillRoom) {
  // `<=`, not `<`: a difference of exactly workFifoBytes fits. Same failing-poll
  // trick pins that the boundary is decided WITHOUT entering the loop.
  FifoComm f(/*consumed=*/0, /*bytes=*/4096);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclSuccess, waitWorkFifoAvailable(f.c(), /*desiredProduced=*/4096));
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_OneByteOver_EntersTheWaitLoop) {
  // One byte past the boundary must enter the loop. Observable because the
  // armed poll failure propagates -- the differential against the test above,
  // which differs by exactly one byte.
  FifoComm f(/*consumed=*/0, /*bytes=*/4096);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclUnhandledCudaError, waitWorkFifoAvailable(f.c(), /*desiredProduced=*/4097));
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_AbortFlagSet_BreaksTheDeadlock) {
  // Without this arm a full FIFO plus a dead consumer is an infinite loop. The
  // abort check runs BEFORE the poll, so the armed poll failure must NOT win.
  FifoComm f(/*consumed=*/0, /*bytes=*/16);
  f.abortStorage = 1;
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclInternalError, waitWorkFifoAvailable(f.c(), /*desiredProduced=*/1 << 20))
      << "abort must be checked before the poll";
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_PollFailure_Propagates) {
  // NCCLCHECK on the poll: a failing poll must abandon the wait rather than spin.
  FifoComm f(/*consumed=*/0, /*bytes=*/16);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclUnhandledCudaError, waitWorkFifoAvailable(f.c(), /*desiredProduced=*/1 << 20));
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_CountersWrappedPastUint32_StillComputesRoom) {
  // THE reason the subtraction is written the way it is. consumed sits near 2^32
  // and desired has already wrapped to 0; the unsigned difference is still the
  // true distance, so there IS room. A rewrite to `desired <= consumed + bytes`
  // would overflow and wrongly enter the loop -- which the armed poll failure
  // would then expose as an error return.
  FifoComm f(/*consumed=*/0xFFFFF000u, /*bytes=*/4096);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  const uint32_t desired = 0xFFFFF000u + 4096u;   // wraps to 0x00000000
  EXPECT_EQ(ncclSuccess, waitWorkFifoAvailable(f.c(), desired));
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_WrappedAndOneByteOver_StillBlocks) {
  // Differential with the test above: same wrapped region, one byte too far.
  // Together they pin that wraparound is handled without becoming permissive.
  FifoComm f(/*consumed=*/0xFFFFF000u, /*bytes=*/4096);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  const uint32_t desired = 0xFFFFF000u + 4097u;
  EXPECT_EQ(ncclUnhandledCudaError, waitWorkFifoAvailable(f.c(), desired));
}

// ===========================================================================
// ncclPlanSetDefaultKernel (enqueue.cc:2867) and ncclGetAlgoInfo (:2847)
// Thin out-of-TU shims that exist so upstream-synced sources can link. Small,
// but ncclPlanSetDefaultKernel is the one place ncclKerns[] is read by index --
// worth pinning that the index comes from comm->unroll.
// ===========================================================================

TEST_F(EnqueueMicrotest, PlanSetDefaultKernel_SelectsKernelByCommUnroll) {
  // ncclGetKernelIndex(comm) is comm->unroll. Two different unrolls must select
  // DIFFERENT table entries -- a hardcoded index would give the same pointer.
  auto comm = std::make_unique<ncclComm>();
  auto planA = std::make_unique<ncclKernelPlan>();
  auto planB = std::make_unique<ncclKernelPlan>();

  comm->unroll = 0;
  ncclPlanSetDefaultKernel(comm.get(), planA.get());
  comm->unroll = 1;
  ncclPlanSetDefaultKernel(comm.get(), planB.get());

  EXPECT_NE(nullptr, planA->kernelFn);
  EXPECT_NE(nullptr, planB->kernelFn);
  EXPECT_NE(planA->kernelFn, planB->kernelFn) << "the index must come from comm->unroll";
}

TEST_F(EnqueueMicrotest, PlanSetDefaultKernel_CopiesSpecializedFlagFromTheTable) {
  // Every entry in ncclKerns is {fn, true}; pin that the flag is copied from the
  // table rather than left at the plan's default. Comparing against
  // ncclKerns[2].kernelFn checks assignment consistency WITHIN this TU (the
  // addresses are host surrogates), not real kernel linkage.
  auto comm = std::make_unique<ncclComm>();
  auto plan = std::make_unique<ncclKernelPlan>();
  plan->kernelSpecialized = false;
  comm->unroll = 2;
  ncclPlanSetDefaultKernel(comm.get(), plan.get());
  EXPECT_TRUE(plan->kernelSpecialized);
  EXPECT_EQ(ncclKerns[2].kernelFn, plan->kernelFn);
}

TEST_F(EnqueueMicrotest, KernelTable_HostSurrogate_ShapeIsSixDistinctSpecializedEntries) {
  // SCOPE: in this binary the six kernel addresses are HOST SURROGATES defined at
  // the top of this TU, not the real __global__ symbols. This test therefore
  // proves only that ncclKerns[] is built with one entry per surrogate, all
  // flagged specialized -- it CANNOT catch a missing device kernel, a signature
  // mismatch, or broken device linkage. Those need a device-linked build.
  // The useful part is the shape/arity of the table, which the indexing tests
  // above then rely on.
  constexpr int kCount = int(sizeof(ncclKerns) / sizeof(ncclKerns[0]));
  EXPECT_EQ(6, kCount);
  std::vector<void*> fns;
  for (int i = 0; i < kCount; ++i) {
    EXPECT_NE(nullptr, ncclKerns[i].kernelFn) << "entry " << i;
    EXPECT_TRUE(ncclKerns[i].specialized) << "entry " << i;
    fns.push_back(ncclKerns[i].kernelFn);
  }
  std::sort(fns.begin(), fns.end());
  EXPECT_EQ(fns.end(), std::unique(fns.begin(), fns.end())) << "entries must be distinct";
}

// ===========================================================================
// addProxyOpIfNeeded (enqueue.cc:198) / ncclAddProxyOpIfNeeded (:772)
// A proxy op is only recorded when the transport says it is needed; the
// "justInquire" protocol means ncclProxySaveOp is asked FIRST whether the op
// matters, and only then is memory allocated for it.
// ===========================================================================

namespace {
// addProxyOpIfNeeded enqueues to the op's CHANNEL queue
// (comm->planner.wipPlan.channels[op->channelId].proxyOpQueue) -- NOT to
// plan->proxyOpQueue. finishPlan is what later merges the channel queues into
// the plan. Looking in the wrong queue makes every one of these read "empty".
ncclProxyOp* ChannelProxyHead(BatchPlanComm& bp, int channelId) {
  return ncclIntruQueueHead(&bp.chan(channelId)->proxyOpQueue);
}
bool ChannelProxyEmpty(BatchPlanComm& bp, int channelId) {
  return ncclIntruQueueEmpty(&bp.chan(channelId)->proxyOpQueue);
}
}  // namespace

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
  // The protocol is "ask first, allocate second". Production seeds justInquire
  // to true on the way in (:200), which is what tells ncclProxySaveOp it is
  // being asked rather than told. Pin the incoming value, not just the outgoing.
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 2;
  op.opCount = 0x77;
  g_proxySaveOpJustInquire = false;
  ASSERT_EQ(ncclSuccess, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_TRUE(g_proxySaveOpSawJustInquireIn)
      << "production must seed justInquire=true before asking";
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
