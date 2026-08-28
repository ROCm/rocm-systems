/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ===========================================================================
// addWorkBatchToPlan (enqueue.cc:209) -- appends one work item to a channel's
// batch queue, opening a NEW batch or an EXTENSION batch when the current one
// cannot absorb it.
//
// High-consequence code: the p2p-round epoch rule exists specifically to keep
// batching uniform across ranks, and getting it wrong hangs the collective
// rather than returning an error. The function returns void, so the ONLY
// oracle is the resulting queue/bitset state.
// ===========================================================================

namespace {
// ncclComm is far too large for the stack (see CostComm), and memScoped needs
// real backing memory for ncclMemoryStackAlloc.
struct BatchPlanComm {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  std::unique_ptr<ncclKernelPlan> plan{new ncclKernelPlan{}};
  BatchPlanComm(int nNodes = 1, int cudaArch = 942) {
    comm->nNodes = nNodes;
    comm->cudaArch = cudaArch;
    ncclMemoryStackConstruct(&comm->memScoped);
    // memPermanent is load-bearing, not decorative: addProxyOpIfNeeded allocates
    // its queued copy from memPool_ncclProxyOp against &comm->memPermanent. A
    // value-initialised stack has bumper == end == 0, so allocate() falls through
    // to allocateSpilled and mallocs a 64 KiB hunk -- and because topFrame.hunk is
    // null it is never linked into the frame chain, so nothing can reclaim it.
    ncclMemoryStackConstruct(&comm->memPermanent);
  }
  ~BatchPlanComm() {
    ncclMemoryStackDestruct(&comm->memScoped);
    ncclMemoryStackDestruct(&comm->memPermanent);
  }
  ncclComm* c() { return comm.get(); }
  ncclKernelPlan* p() { return plan.get(); }
  ncclKernelPlanner::WipPlan::Channel* chan(int id = 0) {
    return &comm->planner.wipPlan.channels[id];
  }
  // Number of batch nodes currently queued on a channel.
  int queueLength(int id = 0) {
    int n = 0;
    for (auto* node = chan(id)->workBatchQueue.head; node != nullptr; node = node->next) ++n;
    return n;
  }
  ncclDevWorkBatch* tailBatch(int id = 0) {
    auto* t = chan(id)->workBatchQueue.tail;
    return t ? &t->batch : nullptr;
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, AddWorkBatch_FirstItem_OpensABatchAndCountsIt) {
  // Empty queue -> newBatch. Pins the initial field stamp, since every later
  // "can we append?" decision reads these back.
  BatchPlanComm bp;
  addWorkBatchToPlan(bp.c(), bp.p(), /*channelId=*/0, ncclDevWorkTypeColl,
                     /*devFuncId=*/7, /*workOffset=*/0);
  ASSERT_EQ(1, bp.queueLength());
  EXPECT_EQ(1, bp.p()->nWorkBatches);
  auto* b = bp.tailBatch();
  ASSERT_NE(nullptr, b);
  EXPECT_EQ(uint32_t(ncclDevWorkTypeColl), b->workType);
  EXPECT_EQ(7, b->funcId);
  EXPECT_EQ(0u, b->offsetBase);
  EXPECT_EQ(0, b->nextExtends);
  EXPECT_EQ(1ull, b->offsetBitset) << "slot 0 must be marked";
}

// MEASURED CAPACITY (this build): NCCL_MAX_DEV_WORK_BATCH_BYTES is 192 and
// sizeof(ncclDevWorkColl) is 160, so exactly ONE coll item fits per batch --
// a second coll always opens a new batch regardless of offset. P2p work is 64
// bytes, so up to three fit by size (further capped to
// NCCL_MAX_DEV_WORK_P2P_PER_BATCH = 2 by the round rules). Tests that need two
// items to SHARE a batch must therefore use p2p, not coll.

TEST_F(EnqueueMicrotest, AddWorkBatch_SecondCollItem_ExceedsByteBudgetAndOpensNewBatch) {
  // Two coll items cannot share: 2*160 > 192. Pins the byte-budget guard with
  // the smallest possible case, and documents the capacity above.
  BatchPlanComm bp;
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeColl);
  ASSERT_GT(2 * ws, size_t(NCCL_MAX_DEV_WORK_BATCH_BYTES))
      << "precondition: two coll items must not fit in one batch";
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeColl, 7, 0);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeColl, 7, uint32_t(ws));
  EXPECT_EQ(2, bp.queueLength());
  EXPECT_EQ(2, bp.p()->nWorkBatches);
}

TEST_F(EnqueueMicrotest, AddWorkBatch_SecondContiguousP2p_AppendsToSameBatch) {
  // p2p work is small enough to share a batch. Same type, same funcId,
  // contiguous offset, same epoch, distinct rounds -> one batch, two bits set.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws),
                     /*p2pRound=*/1, true);
  EXPECT_EQ(1, bp.queueLength()) << "must reuse the open batch";
  EXPECT_EQ(1, bp.p()->nWorkBatches);
  EXPECT_EQ(0b11ull, bp.tailBatch()->offsetBitset) << "slots 0 and 1";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_DifferentWorkType_ForcesNewBatch) {
  // `newBatch |= batch->workType != workType`.
  BatchPlanComm bp;
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeColl, 7, 0);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeCollReg, 7, 0);
  EXPECT_EQ(2, bp.queueLength());
  EXPECT_EQ(2, bp.p()->nWorkBatches);
}

// The funcId guard is covered by
// AddWorkBatch_P2pDifferentFuncId_ForcesNewBatch_Isolated (tests_batch9.inc).
// A coll-based version of this test cannot work: two coll items are 320 B
// against a 192 B budget, so the byte-budget guard splits them regardless of
// funcId, and the mutant `newBatch |= false` survives. Verified by mutation.

TEST_F(EnqueueMicrotest, AddWorkBatch_NonMultipleOffset_CreatesExtensionBatch) {
  // `extendBatch |= 0 != offset % workSize` -- a misaligned offset cannot be
  // expressed in the bitset, so the previous batch is flagged nextExtends and a
  // new node is chained. The distinguishing feature vs a plain new batch is that
  // the PREVIOUS batch's nextExtends becomes 1.
  // Uses p2p: a second COLL item would open a plain new batch on the byte budget
  // before the extension logic could be reached.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, /*p2pRound=*/0, true);
  auto* first = bp.tailBatch();
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws / 2),
                     /*p2pRound=*/1, true);
  EXPECT_EQ(2, bp.queueLength());
  EXPECT_EQ(1, first->nextExtends) << "the previous batch must be marked as extended";
  EXPECT_EQ(0, bp.tailBatch()->nextExtends) << "the new node starts unextended";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_OffsetBeyondBitsetRange_CreatesExtensionBatch) {
  // `extendBatch |= 63 * workSize < offset` -- offsetBitset is 64 bits wide, so
  // slot 64 and beyond cannot be represented. Pins the exact boundary.
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  {
    BatchPlanComm fits(/*nNodes=*/4);
    addWorkBatchToPlan(fits.c(), fits.p(), 0, ncclDevWorkTypeP2p, 7, 0, 0, true);
    addWorkBatchToPlan(fits.c(), fits.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(63 * ws), 1, true);
    EXPECT_EQ(1, fits.queueLength()) << "slot 63 is the last representable one";
    EXPECT_NE(0ull, fits.tailBatch()->offsetBitset & (1ull << 63));
  }
  {
    BatchPlanComm over(/*nNodes=*/4);
    addWorkBatchToPlan(over.c(), over.p(), 0, ncclDevWorkTypeP2p, 7, 0, 0, true);
    addWorkBatchToPlan(over.c(), over.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(64 * ws), 1, true);
    EXPECT_EQ(2, over.queueLength()) << "slot 64 must spill to an extension batch";
  }
}

TEST_F(EnqueueMicrotest, AddWorkBatch_ExtensionBatchDoesNotResetWipCounters) {
  // The comment at :253 is load-bearing: extension batches are FUSED on the
  // device, so wipBatch.workBytes must keep accumulating across them. Only a
  // genuinely new batch resets it.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws / 2),
                     /*p2pRound=*/1, true);  // misaligned -> extension
  EXPECT_EQ(2 * ws, bp.chan()->wipBatch.workBytes) << "extension must ACCUMULATE";

  // A different funcId forces a genuinely new batch, which DOES reset.
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 9, 0, /*p2pRound=*/2, true);
  EXPECT_EQ(ws, bp.chan()->wipBatch.workBytes) << "a new batch must RESET";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_ChannelsAreIndependent) {
  // Every decision reads comm->planner.wipPlan.channels[channelId], so work on
  // one channel must not disturb another. Catches a hardcoded index.
  BatchPlanComm bp;
  addWorkBatchToPlan(bp.c(), bp.p(), /*channelId=*/0, ncclDevWorkTypeColl, 7, 0);
  addWorkBatchToPlan(bp.c(), bp.p(), /*channelId=*/1, ncclDevWorkTypeColl, 7, 0);
  EXPECT_EQ(1, bp.queueLength(0));
  EXPECT_EQ(1, bp.queueLength(1));
  EXPECT_EQ(2, bp.p()->nWorkBatches) << "but the plan-wide count sums both";
}

// The duplicate-round guard is covered by
// AddWorkBatch_P2pDuplicateRoundSameEpoch_ForcesNewBatch_Isolated
// (tests_batch9.inc). A version using the default nNodes == 1 cannot work: for
// nNodes <= 2 production allows only ONE p2p per batch, so the second op splits
// on that cap before the duplicate-round check matters, and the mutant
// `newBatch |= false` survives. Verified by mutation.

TEST_F(EnqueueMicrotest, AddWorkBatch_P2pDifferentEpoch_ForcesNewBatch) {
  // The epoch rule at :233. Rounds 0 and NCCL_MAX_DEV_WORK_P2P_PER_BATCH are in
  // DIFFERENT epochs, so they must not fuse -- this is the rule that keeps
  // batching uniform across ranks and prevents hangs.
  BatchPlanComm bp(/*nNodes=*/4);  // >2 so the per-batch cap is the epoch size
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws),
                     /*p2pRound=*/NCCL_MAX_DEV_WORK_P2P_PER_BATCH, true);
  EXPECT_EQ(2, bp.queueLength()) << "cross-epoch p2ps must not fuse";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_P2pSameEpochDifferentRounds_MayShareABatch) {
  // Differential against both p2p tests above: same epoch, distinct rounds, and
  // nNodes > 2 so the cap is NCCL_MAX_DEV_WORK_P2P_PER_BATCH rather than 1.
  // Without this, "always makes a new batch" would pass the other two.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws),
                     /*p2pRound=*/1, true);
  EXPECT_EQ(1, bp.queueLength()) << "same epoch, different rounds -> one batch";
  EXPECT_EQ(2, bp.chan()->wipBatch.nP2ps);
}

TEST_F(EnqueueMicrotest, AddWorkBatch_P2pTwoNodesOrFewer_CapsAtOnePerBatch) {
  // The nNodes <= 2 arm of the ternary at :229 caps nP2ps at 1, so even distinct
  // rounds in the same epoch cannot share. Differential with the test above,
  // which differs ONLY in nNodes.
  BatchPlanComm bp(/*nNodes=*/2);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws),
                     /*p2pRound=*/1, true);
  EXPECT_EQ(2, bp.queueLength()) << "nNodes<=2 allows only one p2p per batch";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_P2pRecordsRoundAndBatchEligibility) {
  // The bookkeeping at :286-292: the FIRST p2p in a batch fixes batchP2P for the
  // whole batch, and every round is recorded in order.
  // Rounds must share an EPOCH to land in one batch: epoch = round /
  // NCCL_MAX_DEV_WORK_P2P_PER_BATCH (2 here), so 5 and 6 are epochs 2 and 3 and
  // would split. 4 and 5 are both epoch 2.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  const int r0 = 2 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH;      // first round of an epoch
  const int r1 = r0 + 1;                                   // same epoch
  ASSERT_EQ(r0 / NCCL_MAX_DEV_WORK_P2P_PER_BATCH, r1 / NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, r0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws), r1, true);
  EXPECT_TRUE(bp.chan()->wipBatch.batchP2P);
  ASSERT_EQ(2, bp.chan()->wipBatch.nP2ps);
  EXPECT_EQ(r0, bp.chan()->wipBatch.p2pRounds[0]);
  EXPECT_EQ(r1, bp.chan()->wipBatch.p2pRounds[1]);
}

TEST_F(EnqueueMicrotest, AddWorkBatch_P2pIneligibleAfterEligible_ForcesNewBatch) {
  // `newBatch |= !chan->wipBatch.batchP2P` -- an op that is not batch-eligible
  // must not join a batch opened by an eligible one.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, 0, /*batchP2P=*/false);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws), 1,
                     /*batchP2P=*/false);
  EXPECT_EQ(2, bp.queueLength()) << "batchP2P=false caps the batch at one op";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_CountsP2pBatchesSeparately) {
  // nWorkBatchesP2p is incremented only for NEW p2p batches (not extensions),
  // because it derives a proxyOpCount that fused ops must share.
  BatchPlanComm bp(/*nNodes=*/2);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, 0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws), 1, true);
  EXPECT_EQ(2, bp.chan()->nWorkBatchesP2p);

  // A coll batch must NOT bump the p2p counter.
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeColl, 7, 0);
  EXPECT_EQ(2, bp.chan()->nWorkBatchesP2p) << "coll work must not count as p2p";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_EveryItemLandsInExactlyOneBatch) {
  // Whatever the capacity rules decide, no work may be silently dropped: the
  // number of set bits across all batches must equal the number of calls. This
  // is the invariant that survives a change to the byte budget, and it catches
  // an off-by-one that merely shifts items between batches.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  const int kItems = 6;
  for (int i = 0; i < kItems; ++i) {
    addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(i * ws),
                       /*p2pRound=*/i, true);
  }
  int bits = 0;
  for (auto* node = bp.chan()->workBatchQueue.head; node != nullptr; node = node->next) {
    bits += __builtin_popcountll(node->batch.offsetBitset);
  }
  EXPECT_EQ(kItems, bits) << "every call must set exactly one slot bit";
  EXPECT_EQ(bp.queueLength(), bp.p()->nWorkBatches)
      << "plan->nWorkBatches must track the queue length";
}
