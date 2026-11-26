/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// finishPlan and waitWorkFifoAvailable: packing work batches into kernel arguments or the work FIFO.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// finishPlan (enqueue.cc:294) -- packs the per-channel work batches into the
// kernel-args block and merges the per-channel proxy-op lists into one ordered
// queue. Two self-contained algorithms, both RCCL-specific divergences from
// upstream, and both silent when wrong:
//
//   1. Round-robin batch interleave (:331-353). The FIRST batch of channel c
//      must land at batchZero[c]; later batches are chained with nextJump.
//   2. Proxy-op merge sort (:356-385), ordered by `id >> 1 | id << 63` so
//      collectives sort before p2ps of the same count.
//
// Returns void, so the oracle is entirely the resulting layout.
// ===========================================================================

namespace {
struct FinishComm {
  // Holds a BatchPlanComm rather than repeating the comm/plan pair and the
  // memory-stack lifetime. That also picks up its memPermanent construct, which
  // is inert for these tests but load-bearing for anything reaching
  // addProxyOpIfNeeded -- the weaker of two near-identical fixtures is where a
  // future test lands by accident.
  //
  // (0, 0) reproduces what `new ncclComm{}` gave this fixture before: these tests
  // were written against a zero nNodes and cudaArch, not BatchPlanComm's 1/942.
  BatchPlanComm base{0, 0};
  std::vector<std::unique_ptr<ncclProxyOp>> ops;   // keep proxy ops alive

  FinishComm() {
    c()->workArgsBytes = 1 << 16;   // generous: the args-storage arm is taken
  }

  ncclComm* c() { return base.c(); }
  ncclKernelPlan* p() { return base.p(); }
  ncclKernelPlanner::WipPlan::Channel* chan(int id) { return base.chan(id); }
  void markChannel(int c) { p()->channelMask.masks[c / 64] |= (1ULL << (c % 64)); }

  // Queue n work batches on channel c, each stamped with a recognisable funcId.
  void addBatches(int c, int n, int funcIdBase) {
    for (int i = 0; i < n; ++i) {
      addWorkBatchToPlan(this->c(), p(), c, ncclDevWorkTypeColl,
                         funcIdBase + i, /*progressSlot=*/0, uint32_t(i * ncclDevWorkSize(ncclDevWorkTypeColl)));
    }
    markChannel(c);
  }

  // Queue one proxy op with the given opCount on channel c.
  void addProxyOp(int c, uint64_t opCount) {
    ops.emplace_back(new ncclProxyOp{});
    ops.back()->opCount = opCount;
    ncclIntruQueueEnqueue(&chan(c)->proxyOpQueue, ops.back().get());
  }

  // The packed batch array finishPlan wrote.
  ncclDevWorkBatch* batchZero() {
    return (ncclDevWorkBatch*)(p()->kernelArgs + 1);
  }
  std::vector<uint64_t> mergedOpCounts() {
    std::vector<uint64_t> out;
    // The intrusive link for proxyOpQueue is `enqNext` (comm.h:386), not `next`.
    for (auto* op = ncclIntruQueueHead(&p()->proxyOpQueue); op != nullptr; op = op->enqNext) {
      out.push_back(op->opCount);
    }
    return out;
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, FinishPlan_SymmetricCollReturnsImmediately) {
  // `if (plan->isSymColl) return;` -- the early-out must leave kernelArgs
  // untouched, since a symmetric plan is packed elsewhere.
  FinishComm f;
  f.p()->isSymColl = true;
  f.addBatches(0, 1, 100);
  finishPlan(f.c(), f.p());
  EXPECT_EQ(nullptr, f.p()->kernelArgs) << "symmetric plans must not be packed here";
}

TEST_F(EnqueueMicrotest, FinishPlan_SmallPlan_UsesArgsStorage) {
  // Everything fits in workArgsBytes -> workStorageType becomes Args.
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 1, 100);
  finishPlan(f.c(), f.p());
  EXPECT_EQ(ncclDevWorkStorageTypeArgs, f.p()->workStorageType);
}

TEST_F(EnqueueMicrotest, FinishPlan_OversizedPlan_KeepsFifoStorage) {
  // The other side of the fit test: a plan too large for the args block must NOT
  // be promoted. Differential with the test above -- same shape, bigger workBytes.
  FinishComm f;
  f.c()->workArgsBytes = 256;              // tiny
  f.p()->workBytes = 1 << 20;              // far too big
  f.p()->workStorageType = ncclDevWorkStorageTypeFifo;
  f.addBatches(0, 1, 100);
  finishPlan(f.c(), f.p());
  EXPECT_EQ(ncclDevWorkStorageTypeFifo, f.p()->workStorageType);
  // The kernel reads storage type off kernelArgs, so the copy at :316 must carry
  // it. Non-zero here, so unlike the small-plan case this can see a dropped store.
  ASSERT_NE(nullptr, f.p()->kernelArgs);
  EXPECT_EQ(ncclDevWorkStorageTypeFifo, f.p()->kernelArgs->workStorageType);
}

TEST_F(EnqueueMicrotest, FinishPlan_KernelArgsSizeIsSixteenByteAligned) {
  // :311 alignUp(..., 16). Pins the alignment and the floor at
  // sizeof(ncclDevKernelArgsDefaultStorage) together.
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 3, 100);
  finishPlan(f.c(), f.p());
  EXPECT_EQ(0u, f.p()->kernelArgsSize % 16) << "must be 16B aligned";
  EXPECT_GE(f.p()->kernelArgsSize, sizeof(ncclDevKernelArgsDefaultStorage))
      << "must not fall below the default-storage floor";
  EXPECT_NE(nullptr, f.p()->kernelArgs);
}

TEST_F(EnqueueMicrotest, FinishPlan_StampsChannelMaskAndStorageTypeIntoKernelArgs) {
  // The kernel reads these back on the device, so a dropped assignment is
  // invisible on the host but fatal at run time.
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 1, 100);
  f.addBatches(3, 1, 200);
  finishPlan(f.c(), f.p());
  ASSERT_NE(nullptr, f.p()->kernelArgs);
  EXPECT_EQ(f.p()->channelMask.masks[0], f.p()->kernelArgs->channelMask.masks[0]);
  // HONEST SCOPE: workStorageType is NOT checked here. On this path it is
  // ncclDevWorkStorageTypeArgs, enumerator 0 (device.h:650), and kernelArgs comes
  // from ncclMemoryStackAlloc (:313) which memsets to 0 (utils.h:283), so the
  // comparison is 0 == 0 and deleting the store at :316 kills nothing. The
  // oversized-plan test below stamps a non-zero value and does check it.
}

namespace {
// kernelArgs is zero-filled, so only the cases that set a bit can see a dropped
// devFlags store; the all-off case pins that nothing else sets one.
uint8_t devFlagsAfterFinishPlan(bool profilerLoaded, bool progressCounters) {
  ScopedHook profiler(g_profilerPluginLoaded, [=] { return profilerLoaded; });
  FinishComm f;
  alignas(16) unsigned char countersBlock[16];
  f.c()->deviceCountersBlock =
      progressCounters ? reinterpret_cast<ncclProgressCountersBlock*>(countersBlock) : nullptr;
  f.p()->workBytes = 64;
  f.addBatches(0, 1, 100);
  finishPlan(f.c(), f.p());
  EXPECT_NE(nullptr, f.p()->kernelArgs);
  return f.p()->kernelArgs ? f.p()->kernelArgs->devFlags : 0xff;
}
}  // namespace

TEST_F(EnqueueMicrotest, FinishPlan_DevFlags_BothOff_IsZero) {
  EXPECT_EQ(0, devFlagsAfterFinishPlan(false, false));
}

TEST_F(EnqueueMicrotest, FinishPlan_DevFlags_ProfilerLoaded_SetsProfilerOnly) {
  EXPECT_EQ(RCCL_DEV_FLAG_PROFILER, devFlagsAfterFinishPlan(true, false));
}

TEST_F(EnqueueMicrotest, FinishPlan_DevFlags_CountersAllocated_SetsProgressOnly) {
  EXPECT_EQ(RCCL_DEV_FLAG_PROGRESS, devFlagsAfterFinishPlan(false, true));
}

TEST_F(EnqueueMicrotest, FinishPlan_DevFlags_BothOn_SetsBoth) {
  EXPECT_EQ(RCCL_DEV_FLAG_PROFILER | RCCL_DEV_FLAG_PROGRESS, devFlagsAfterFinishPlan(true, true));
}

TEST_F(EnqueueMicrotest, FinishPlan_OneBatchPerChannel_LandsAtItsOwnSlot) {
  // The core layout contract: with one batch on each of channels 0..2, the first
  // batch of channel c must be at batchZero[c]. funcId identifies which is which.
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 1, /*funcIdBase=*/10);
  f.addBatches(1, 1, /*funcIdBase=*/20);
  f.addBatches(2, 1, /*funcIdBase=*/30);
  finishPlan(f.c(), f.p());
  ASSERT_NE(nullptr, f.p()->kernelArgs);
  EXPECT_EQ(10, f.batchZero()[0].funcId);
  EXPECT_EQ(20, f.batchZero()[1].funcId);
  EXPECT_EQ(30, f.batchZero()[2].funcId);
}

TEST_F(EnqueueMicrotest, FinishPlan_MultipleBatches_InterleavesRoundRobin) {
  // Two batches on each of two channels must interleave c0,c1,c0,c1 -- NOT
  // c0,c0,c1,c1. This is the property the device relies on to find its first
  // batch at batchZero[blockIdx.x].
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 2, /*funcIdBase=*/10);   // funcIds 10, 11
  f.addBatches(1, 2, /*funcIdBase=*/20);   // funcIds 20, 21
  finishPlan(f.c(), f.p());
  ASSERT_NE(nullptr, f.p()->kernelArgs);
  EXPECT_EQ(10, f.batchZero()[0].funcId);
  EXPECT_EQ(20, f.batchZero()[1].funcId) << "channel 1's first batch must be at slot 1";
  EXPECT_EQ(11, f.batchZero()[2].funcId);
  EXPECT_EQ(21, f.batchZero()[3].funcId);
}

TEST_F(EnqueueMicrotest, FinishPlan_ChainsLaterBatchesWithNextJump) {
  // nextJump back-patching: channel 0's batch at slot 0 must point at its NEXT
  // batch (slot 2 when interleaved with channel 1), as an element delta.
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 2, 10);
  f.addBatches(1, 2, 20);
  finishPlan(f.c(), f.p());
  ASSERT_NE(nullptr, f.p()->kernelArgs);
  EXPECT_EQ(2, f.batchZero()[0].nextJump) << "slot 0 -> slot 2 is a jump of 2";
  EXPECT_EQ(2, f.batchZero()[1].nextJump) << "slot 1 -> slot 3";
  EXPECT_EQ(0, f.batchZero()[2].nextJump) << "last batch of the channel: no jump";
  EXPECT_EQ(0, f.batchZero()[3].nextJump);
}

TEST_F(EnqueueMicrotest, FinishPlan_UnevenChannels_StillChainsCorrectly) {
  // Channel 0 has 3 batches, channel 1 has 1. Once channel 1 is exhausted the
  // round robin continues over channel 0 alone, so its jumps shrink from 2 to 1.
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 3, 10);   // 10, 11, 12
  f.addBatches(1, 1, 20);   // 20
  finishPlan(f.c(), f.p());
  ASSERT_NE(nullptr, f.p()->kernelArgs);
  EXPECT_EQ(10, f.batchZero()[0].funcId);
  EXPECT_EQ(20, f.batchZero()[1].funcId);
  EXPECT_EQ(11, f.batchZero()[2].funcId);
  EXPECT_EQ(12, f.batchZero()[3].funcId);
  EXPECT_EQ(2, f.batchZero()[0].nextJump) << "slot 0 -> slot 2";
  EXPECT_EQ(1, f.batchZero()[2].nextJump) << "slot 2 -> slot 3, channel 1 is done";
  EXPECT_EQ(0, f.batchZero()[3].nextJump);
  EXPECT_EQ(0, f.batchZero()[1].nextJump) << "channel 1 had only one batch";
}

TEST_F(EnqueueMicrotest, FinishPlan_HighChannelIndex_IsHandledByTheMaskWordSplit) {
  // The RCCL >64-channel divergence: the mask is an array of 64-bit words and
  // the loop indexes it as masks[c/64] & (1 << c%64). A channel at index >= 64
  // exercises the SECOND word, which a single-uint64 implementation would miss.
  if (MAXCHANNELS <= 64) GTEST_SKIP() << "build has MAXCHANNELS <= 64";
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 1, 10);
  f.addBatches(64, 1, 99);
  finishPlan(f.c(), f.p());
  ASSERT_NE(nullptr, f.p()->kernelArgs);
  EXPECT_EQ(10, f.batchZero()[0].funcId);
  EXPECT_EQ(99, f.batchZero()[1].funcId) << "channel 64 must be packed, not dropped";
}

TEST_F(EnqueueMicrotest, FinishPlan_NoProxyOps_LeavesHasProxyOpsFalse) {
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 1, 10);
  finishPlan(f.c(), f.p());
  EXPECT_FALSE(f.p()->hasProxyOps);
  EXPECT_TRUE(f.mergedOpCounts().empty());
}

TEST_F(EnqueueMicrotest, FinishPlan_MergesProxyOpsInAscendingOpCount) {
  // The merge sort: ops arrive on separate channels, each channel already
  // ordered, and must come out globally ordered.
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 1, 10);
  f.addBatches(1, 1, 20);
  // opCount is shifted left by 1 because bit 0 is the coll/p2p tag.
  f.addProxyOp(0, 2 << 1);
  f.addProxyOp(0, 6 << 1);
  f.addProxyOp(1, 4 << 1);
  f.addProxyOp(1, 8 << 1);
  finishPlan(f.c(), f.p());

  EXPECT_TRUE(f.p()->hasProxyOps);
  const std::vector<uint64_t> want{2 << 1, 4 << 1, 6 << 1, 8 << 1};
  EXPECT_EQ(want, f.mergedOpCounts());
}

TEST_F(EnqueueMicrotest, FinishPlan_CollectivesSortBeforeP2psOfTheSameCount) {
  // THE reason for the `id >> 1 | id << 63` rotate: bit 0 tags p2p, and rotating
  // it to the TOP makes a collective sort before a p2p with the same count.
  // Without the rotate, the raw ids 8 and 9 would already be in this order --
  // so the discriminating case is a p2p with a SMALLER raw id than the coll.
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 1, 10);
  f.addBatches(1, 1, 20);
  f.addProxyOp(0, (5 << 1) | 1);   // p2p, count 5 -> raw id 11
  f.addProxyOp(1, (6 << 1));       // coll, count 6 -> raw id 12
  finishPlan(f.c(), f.p());

  const auto got = f.mergedOpCounts();
  ASSERT_EQ(2u, got.size());
  EXPECT_EQ(uint64_t(6 << 1), got[0]) << "the collective must come first despite the larger raw id";
  EXPECT_EQ(uint64_t((5 << 1) | 1), got[1]);
}

TEST_F(EnqueueMicrotest, FinishPlan_SingleChannelProxyOps_PreserveTheirOrder) {
  // A degenerate merge (one channel) must be order-preserving, not reversing --
  // which a queue implementation that enqueued at the head would get wrong.
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 1, 10);
  for (uint64_t n : {1, 2, 3, 4}) f.addProxyOp(0, n << 1);
  finishPlan(f.c(), f.p());
  const std::vector<uint64_t> want{1 << 1, 2 << 1, 3 << 1, 4 << 1};
  EXPECT_EQ(want, f.mergedOpCounts());
}

TEST_F(EnqueueMicrotest, FinishPlan_DrainsEveryChannelProxyQueue) {
  // No op may be left behind: the per-channel queues must all be empty and the
  // merged queue must hold exactly the ops that were enqueued.
  FinishComm f;
  f.p()->workBytes = 64;
  f.addBatches(0, 1, 10);
  f.addBatches(1, 1, 20);
  f.addBatches(2, 1, 30);
  f.addProxyOp(0, 1 << 1);
  f.addProxyOp(1, 2 << 1);
  f.addProxyOp(2, 3 << 1);
  f.addProxyOp(0, 4 << 1);
  finishPlan(f.c(), f.p());

  EXPECT_EQ(4u, f.mergedOpCounts().size());
  for (int c : {0, 1, 2}) {
    EXPECT_TRUE(ncclIntruQueueEmpty(&f.chan(c)->proxyOpQueue)) << "channel " << c << " not drained";
  }
}

// ===========================================================================
// waitWorkFifoAvailable (enqueue.cc:1651)
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
// Mutation-driven reinforcements.
//
// Each test below exists because a specific mutant SURVIVED the first pass. In
// every case the original test was masked by a different guard firing first, so
// the mutated line was executed but its effect was never observable. These
// isolate the mutated term so that only it can decide the outcome.
// ===========================================================================

// ---------------------------------------------------------------------------
// MUTANT: drop `plan->workStorageType = ncclDevWorkStorageTypeArgs`  (SURVIVED)
// WHY: ncclDevWorkStorageTypeArgs happens to be the zero value, so a
// zero-initialised plan already reads as Args and the assignment is invisible.
// Starting from the OTHER storage type makes the promotion observable.
// ---------------------------------------------------------------------------
TEST_F(EnqueueMicrotest, FinishPlan_PromotesFifoToArgsWhenItFits_Isolated) {
  FinishComm f;
  f.p()->workBytes = 64;
  f.p()->workStorageType = ncclDevWorkStorageTypeFifo;  // start at the NON-default
  f.addBatches(0, 1, 100);
  finishPlan(f.c(), f.p());
  EXPECT_EQ(ncclDevWorkStorageTypeArgs, f.p()->workStorageType)
      << "a plan that fits must be PROMOTED, not left as it arrived";
}

// ---------------------------------------------------------------------------
// MUTANT: `alignUp(kernelArgsSize, 16)` -> `alignUp(..., 8)`        (SURVIVED)
// WHY: with the batch counts used earlier the size was already 16-aligned, so
// both alignments produced the same number. This picks a batch count whose
// unaligned size is 8-aligned but NOT 16-aligned, so the two differ.
// ---------------------------------------------------------------------------
TEST_F(EnqueueMicrotest, FinishPlan_KernelArgsSize_IsSixteenNotEightAligned_Isolated) {
  // Two things make this awkward, and both had to be measured rather than
  // assumed:
  //   1. kernelArgsSize is floored at sizeof(ncclDevKernelArgsDefaultStorage),
  //      which is 5120 here and already 16-aligned -- so any SMALL plan pins
  //      nothing. workBytes must exceed the floor for the alignment to matter.
  //   2. workBytes only reaches kernelArgsSize when storage is Args, which
  //      requires workArgsBytes to be large enough to hold it.
  // Stepping by 8 above the floor guarantees some input is 8- but not
  // 16-aligned, which is exactly where alignUp(...,16) and alignUp(...,8) differ.
  bool sawDiscriminating = false;
  const size_t base = sizeof(ncclDevKernelArgsDefaultStorage);
  for (size_t workBytes = base; workBytes <= base + 8 * 8; workBytes += 8) {
    FinishComm f;
    f.c()->workArgsBytes = 1 << 20;   // big enough that the plan still fits in Args
    f.p()->workBytes = workBytes;
    f.addBatches(0, 1, 100);
    finishPlan(f.c(), f.p());
    ASSERT_EQ(ncclDevWorkStorageTypeArgs, f.p()->workStorageType)
        << "precondition: workBytes must reach kernelArgsSize";

    const size_t raw = sizeof(ncclDevKernelArgs) + sizeof(ncclDevWorkBatch) + workBytes;
    const size_t floored = std::max(raw, sizeof(ncclDevKernelArgsDefaultStorage));
    if (floored % 16 == 8) sawDiscriminating = true;   // 16- and 8-alignment differ here

    EXPECT_EQ(0u, f.p()->kernelArgsSize % 16)
        << "workBytes=" << workBytes << " size=" << f.p()->kernelArgsSize;
  }
  EXPECT_TRUE(sawDiscriminating)
      << "no input produced a size where 16- and 8-alignment differ, so this "
         "test cannot tell them apart -- widen the sweep";
}
