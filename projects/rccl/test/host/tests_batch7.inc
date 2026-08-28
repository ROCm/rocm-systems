/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ===========================================================================
// finishPlan (enqueue.cc:295) -- packs the per-channel work batches into the
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
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  std::unique_ptr<ncclKernelPlan> plan{new ncclKernelPlan{}};
  std::vector<std::unique_ptr<ncclProxyOp>> ops;   // keep proxy ops alive

  FinishComm() {
    ncclMemoryStackConstruct(&comm->memScoped);
    comm->workArgsBytes = 1 << 16;   // generous: the args-storage arm is taken
  }
  ~FinishComm() { ncclMemoryStackDestruct(&comm->memScoped); }

  ncclComm* c() { return comm.get(); }
  ncclKernelPlan* p() { return plan.get(); }
  ncclKernelPlanner::WipPlan::Channel* chan(int id) {
    return &comm->planner.wipPlan.channels[id];
  }
  void markChannel(int c) { plan->channelMask.masks[c / 64] |= (1ULL << (c % 64)); }

  // Queue n work batches on channel c, each stamped with a recognisable funcId.
  void addBatches(int c, int n, int funcIdBase) {
    for (int i = 0; i < n; ++i) {
      addWorkBatchToPlan(comm.get(), plan.get(), c, ncclDevWorkTypeColl,
                         funcIdBase + i, uint32_t(i * ncclDevWorkSize(ncclDevWorkTypeColl)));
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
    return (ncclDevWorkBatch*)(plan->kernelArgs + 1);
  }
  std::vector<uint64_t> mergedOpCounts() {
    std::vector<uint64_t> out;
    // The intrusive link for proxyOpQueue is `enqNext` (comm.h:387), not `next`.
    for (auto* op = ncclIntruQueueHead(&plan->proxyOpQueue); op != nullptr; op = op->enqNext) {
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
  EXPECT_EQ(f.p()->workStorageType, f.p()->kernelArgs->workStorageType);
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
