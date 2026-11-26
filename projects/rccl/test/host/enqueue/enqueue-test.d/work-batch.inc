/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// addWorkBatchToPlan: appending work items to a channel's batch queue.
// Included into enqueue-test.cc's translation unit; see README.md.

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

TEST_F(EnqueueMicrotest, AddWorkBatch_FirstItem_OpensABatchAndCountsIt) {
  // Empty queue -> newBatch. Pins the initial field stamp, since every later
  // "can we append?" decision reads these back.
  BatchPlanComm bp;
  addWorkBatchToPlan(bp.c(), bp.p(), /*channelId=*/0, ncclDevWorkTypeColl,
                     /*devFuncId=*/7, /*progressSlot=*/0, /*workOffset=*/0);
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
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeColl, 7, /*progressSlot=*/0, 0);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeColl, 7, /*progressSlot=*/0, uint32_t(ws));
  EXPECT_EQ(2, bp.queueLength());
  EXPECT_EQ(2, bp.p()->nWorkBatches);
}

TEST_F(EnqueueMicrotest, AddWorkBatch_SecondContiguousP2p_AppendsToSameBatch) {
  // p2p work is small enough to share a batch. Same type, same funcId,
  // contiguous offset, same epoch, distinct rounds -> one batch, two bits set.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws),
                     /*p2pRound=*/1, true);
  EXPECT_EQ(1, bp.queueLength()) << "must reuse the open batch";
  EXPECT_EQ(1, bp.p()->nWorkBatches);
  EXPECT_EQ(0b11ull, bp.tailBatch()->offsetBitset) << "slots 0 and 1";
}

// The bcast item cap at :240. Mutation-driven: llvm-cov shows :240 executing,
// but changing `nBcasts == maxitem` to `nBcasts > maxitem` left the whole suite
// green, because no test ever queued enough bcast items to reach the cap. With
// `>` the equality never fires, nBcasts runs past maxitem and the batch never
// splits -- so this is a real behaviour difference, not an equivalent mutant.
TEST_F(EnqueueMicrotest, AddWorkBatch_BcastCapSplitsAtExactlyMaxItem) {
  BatchPlanComm bp;
  const size_t bcastSize = ncclDevWorkSize(ncclDevWorkTypeBcast);
  const int maxitem = ncclMaxDevWorkBatchBytes(bp.c()->cudaArch) / int(sizeof(ncclDevWorkBcast));
  ASSERT_GT(maxitem, 1) << "the cap must be reachable by adding items";

  // Contiguous offsets, so the extension rule at :248-249 keeps appending rather
  // than opening a batch of its own: `0 != offset % workSize` is what would
  // otherwise split these and mask the cap.
  for (int i = 0; i < maxitem; ++i) {
    addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeBcast, 7, /*progressSlot=*/0, uint32_t(i * bcastSize));
  }
  const int beforeCap = bp.queueLength();

  // The (maxitem + 1)th item is the first to see nBcasts == maxitem.
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeBcast, 7, /*progressSlot=*/0, uint32_t(maxitem * bcastSize));
  EXPECT_EQ(beforeCap + 1, bp.queueLength())
      << "the cap at :240 must open a new batch on the item that reaches maxitem";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_DifferentWorkType_ForcesNewBatch) {
  // `newBatch |= batch->workType != workType` must be the ONLY splitter here, so
  // every other path to a second node has to be shut off first. The second call
  // is Bcast, so it takes the `workType == ncclDevWorkTypeBcast` arm at :238:
  //   bcast cap (:240)    the ONLY other splitter reachable on this arm.
  //                       nBcasts is 0, not 1 -- the p2p call opened a new batch
  //                       and reset it at :270, and p2p never increments it
  //                       (:289-291). maxitem is 341, so it cannot fire.
  //   byte budget (:242)  lives in the `else` arm and is NEVER evaluated here.
  //                       An assertion on it would pass no matter what the
  //                       budget was, so this test does not make one.
  //   extension (:248-9)  offset must be a MULTIPLE of the bcast work size, or
  //                       `0 != offset % workSize` enqueues a node anyway --
  //                       this is what made the p2p size (64 % 48 = 16) wrong.
  // A Coll->CollReg version cannot work at all: two coll items are 320 B against
  // the 192 B budget, and coll DOES take the else arm, so :242 splits them
  // whatever the workType guard does.
  // Verified by mutation: `newBatch |= false` leaves 1 batch here.
  BatchPlanComm bp;
  const size_t bcastSize = ncclDevWorkSize(ncclDevWorkTypeBcast);
  ASSERT_LT(1, ncclMaxDevWorkBatchBytes(bp.c()->cudaArch) / int(sizeof(ncclDevWorkBcast)))
      << "maxitem must exceed the single bcast item, or the :240 cap is the splitter";
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeBcast, 7, /*progressSlot=*/0, uint32_t(bcastSize));
  EXPECT_EQ(2, bp.queueLength());
  EXPECT_EQ(2, bp.p()->nWorkBatches);
  EXPECT_EQ(0, bp.chan()->workBatchQueue.head->batch.nextExtends)
      << "must be a NEW batch, not an extension of the p2p one";
}

// The funcId guard is covered by
// AddWorkBatch_P2pDifferentFuncId_ForcesNewBatch_Isolated below.
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
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, /*p2pRound=*/0, true);
  auto* first = bp.tailBatch();
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws / 2),
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
    addWorkBatchToPlan(fits.c(), fits.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, 0, true);
    addWorkBatchToPlan(fits.c(), fits.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(63 * ws), 1, true);
    EXPECT_EQ(1, fits.queueLength()) << "slot 63 is the last representable one";
    EXPECT_NE(0ull, fits.tailBatch()->offsetBitset & (1ull << 63));
  }
  {
    BatchPlanComm over(/*nNodes=*/4);
    addWorkBatchToPlan(over.c(), over.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, 0, true);
    addWorkBatchToPlan(over.c(), over.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(64 * ws), 1, true);
    EXPECT_EQ(2, over.queueLength()) << "slot 64 must spill to an extension batch";
  }
}

TEST_F(EnqueueMicrotest, AddWorkBatch_ExtensionBatchDoesNotResetWipCounters) {
  // The comment at :253 is load-bearing: extension batches are FUSED on the
  // device, so wipBatch.workBytes must keep accumulating across them. Only a
  // genuinely new batch resets it.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws / 2),
                     /*p2pRound=*/1, true);  // misaligned -> extension
  EXPECT_EQ(2 * ws, bp.chan()->wipBatch.workBytes) << "extension must ACCUMULATE";

  // A different funcId forces a genuinely new batch, which DOES reset.
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 9, /*progressSlot=*/0, 0, /*p2pRound=*/2, true);
  EXPECT_EQ(ws, bp.chan()->wipBatch.workBytes) << "a new batch must RESET";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_ChannelsAreIndependent) {
  // Every decision reads comm->planner.wipPlan.channels[channelId], so work on
  // one channel must not disturb another. Catches a hardcoded index.
  BatchPlanComm bp;
  addWorkBatchToPlan(bp.c(), bp.p(), /*channelId=*/0, ncclDevWorkTypeColl, 7, /*progressSlot=*/0, 0);
  addWorkBatchToPlan(bp.c(), bp.p(), /*channelId=*/1, ncclDevWorkTypeColl, 7, /*progressSlot=*/0, 0);
  EXPECT_EQ(1, bp.queueLength(0));
  EXPECT_EQ(1, bp.queueLength(1));
  EXPECT_EQ(2, bp.p()->nWorkBatches) << "but the plan-wide count sums both";
}

// The duplicate-round guard is covered by
// AddWorkBatch_P2pDuplicateRoundSameEpoch_ForcesNewBatch_Isolated
// below. A version using the default nNodes == 1 cannot work: for
// nNodes <= 2 production allows only ONE p2p per batch, so the second op splits
// on that cap before the duplicate-round check matters, and the mutant
// `newBatch |= false` survives. Verified by mutation.

TEST_F(EnqueueMicrotest, AddWorkBatch_P2pDifferentEpoch_ForcesNewBatch) {
  // The epoch rule at :233. Rounds 0 and NCCL_MAX_DEV_WORK_P2P_PER_BATCH are in
  // DIFFERENT epochs, so they must not fuse -- this is the rule that keeps
  // batching uniform across ranks and prevents hangs.
  BatchPlanComm bp(/*nNodes=*/4);  // >2 so the per-batch cap is the epoch size
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws),
                     /*p2pRound=*/NCCL_MAX_DEV_WORK_P2P_PER_BATCH, true);
  EXPECT_EQ(2, bp.queueLength()) << "cross-epoch p2ps must not fuse";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_P2pSameEpochDifferentRounds_MayShareABatch) {
  // Differential against both p2p tests above: same epoch, distinct rounds, and
  // nNodes > 2 so the cap is NCCL_MAX_DEV_WORK_P2P_PER_BATCH rather than 1.
  // Without this, "always makes a new batch" would pass the other two.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws),
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
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws),
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
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, r0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws), r1, true);
  EXPECT_TRUE(bp.chan()->wipBatch.batchP2P);
  ASSERT_EQ(2, bp.chan()->wipBatch.nP2ps);
  EXPECT_EQ(r0, bp.chan()->wipBatch.p2pRounds[0]);
  EXPECT_EQ(r1, bp.chan()->wipBatch.p2pRounds[1]);
}

TEST_F(EnqueueMicrotest, AddWorkBatch_P2pEligibleAfterIneligible_ForcesNewBatch) {
  // `newBatch |= !chan->wipBatch.batchP2P` (:227) -- an op that IS batch-eligible
  // must not join a batch opened by an ineligible one. The first call latches
  // wipBatch.batchP2P=false (:283); the second arrives with batchP2P=true.
  //
  // The call order isolates the conjunct: with nNodes>2 and batchP2P=true the
  // next term takes the `nP2ps == NCCL_MAX_DEV_WORK_P2P_PER_BATCH` arm (1 != 2,
  // false), and rounds 0 and 1 neither collide nor cross an epoch. So :227 is the
  // ONLY thing that splits, and deleting it makes this test fail. Passing false
  // twice instead would fall into the `nP2ps == 1` arm, which splits on its own
  // and lets a mutant that drops :227 survive.
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, 0, /*batchP2P=*/false);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws), 1,
                     /*batchP2P=*/true);
  EXPECT_EQ(2, bp.queueLength()) << "an ineligible batch must not absorb an eligible op";
}

TEST_F(EnqueueMicrotest, AddWorkBatch_CountsP2pBatchesSeparately) {
  // nWorkBatchesP2p is incremented only for NEW p2p batches (not extensions),
  // because it derives a proxyOpCount that fused ops must share.
  BatchPlanComm bp(/*nNodes=*/2);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, 0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws), 1, true);
  EXPECT_EQ(2, bp.chan()->nWorkBatchesP2p);

  // A coll batch must NOT bump the p2p counter.
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeColl, 7, /*progressSlot=*/0, 0);
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
    addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(i * ws),
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

// ===========================================================================
// Mutation-driven reinforcements.
//
// Each test below exists because a specific mutant SURVIVED the first pass. In
// every case the original test was masked by a different guard firing first, so
// the mutated line was executed but its effect was never observable. These
// isolate the mutated term so that only it can decide the outcome.
// ===========================================================================

// ---------------------------------------------------------------------------
// MUTANT: `newBatch |= batch->funcId != devFuncId` -> `|= false`   (SURVIVED)
// WHY: coll work is 160 B against a 192 B batch budget, so the byte-budget guard
// already split every two-coll case. p2p work is 64 B, so two p2ps fit -- there
// the funcId term is the ONLY thing that can force a split.
// ---------------------------------------------------------------------------
TEST_F(EnqueueMicrotest, AddWorkBatch_P2pDifferentFuncId_ForcesNewBatch_Isolated) {
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  // Same epoch, distinct rounds, contiguous offsets, within budget: every OTHER
  // guard is satisfied, so only the funcId difference can open a new batch.
  const int r0 = 2 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, /*devFuncId=*/7, /*progressSlot=*/0, 0, r0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, /*devFuncId=*/8, /*progressSlot=*/0,
                     uint32_t(ws), r0 + 1, true);
  EXPECT_EQ(2, bp.queueLength()) << "a different funcId must force a new batch";

  // Control: identical call with the SAME funcId shares a batch. Without this
  // pair the assertion above could also be satisfied by an unconditional split.
  BatchPlanComm same(/*nNodes=*/4);
  addWorkBatchToPlan(same.c(), same.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, r0, true);
  addWorkBatchToPlan(same.c(), same.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws), r0 + 1, true);
  EXPECT_EQ(1, same.queueLength()) << "same funcId must still share";
}

// ---------------------------------------------------------------------------
// MUTANT: `newBatch |= p2pRound == chan->wipBatch.p2pRounds[i]` -> `|= false`
// WHY: the original test ran with the default nNodes == 1, and for nNodes <= 2
// production caps a batch at ONE p2p -- so the second op split on that cap
// before the duplicate-round check was ever consulted. Using nNodes > 2 with
// duplicate rounds in one epoch leaves this guard as the only possible splitter.
// ---------------------------------------------------------------------------
TEST_F(EnqueueMicrotest, AddWorkBatch_P2pDuplicateRoundSameEpoch_ForcesNewBatch_Isolated) {
  BatchPlanComm bp(/*nNodes=*/4);
  const size_t ws = ncclDevWorkSize(ncclDevWorkTypeP2p);
  const int r = 2 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH;  // first round of an epoch
  // Identical round twice: same epoch, so the epoch rule CANNOT fire. Only the
  // duplicate-round guard can split these.
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, 0, r, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, /*progressSlot=*/0, uint32_t(ws), r, true);
  EXPECT_EQ(2, bp.queueLength())
      << "two p2ps of the SAME round use the same connections and must not fuse";
}
