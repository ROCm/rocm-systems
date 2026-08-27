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
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, /*devFuncId=*/7, 0, r0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, /*devFuncId=*/8,
                     uint32_t(ws), r0 + 1, true);
  EXPECT_EQ(2, bp.queueLength()) << "a different funcId must force a new batch";

  // Control: identical call with the SAME funcId shares a batch. Without this
  // pair the assertion above could also be satisfied by an unconditional split.
  BatchPlanComm same(/*nNodes=*/4);
  addWorkBatchToPlan(same.c(), same.p(), 0, ncclDevWorkTypeP2p, 7, 0, r0, true);
  addWorkBatchToPlan(same.c(), same.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws), r0 + 1, true);
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
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, r, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws), r, true);
  EXPECT_EQ(2, bp.queueLength())
      << "two p2ps of the SAME round use the same connections and must not fuse";
}

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

// ---------------------------------------------------------------------------
// EQUIVALENT MUTANT (documented, not fixed): dropping `count == 0` from the
// early-out at enqueue.cc:180.
//
// With count == 0: cells = divUp(0, cellSize) = 0, so
// cellsPerChannel = min(0, ...) = 0 and the SECOND guard at :191
// (`if (cellsPerChannel == 0) return nMaxChannels`) returns the same value by
// the same path. The two guards overlap for this input, so no test can separate
// them. Verified by hand-evaluating the arithmetic; the test below pins the
// OBSERVABLE behaviour either way.
// ---------------------------------------------------------------------------
TEST_F(EnqueueMicrotest, PackedChannels_ZeroCount_ReturnsMaxChannels_EitherGuard) {
  for (int maxCh : {1, 8, 32}) {
    EXPECT_EQ(maxCh, rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, 0,
                                              ncclFloat32, NCCL_PROTO_SIMPLE, maxCh))
        << "maxCh=" << maxCh;
  }
}

// ---------------------------------------------------------------------------
// MUTANT: `if (protocol == NCCL_PROTO_LL) trafficPerByte *= 4` -> `*= 1`
// (SURVIVED)
// WHY: the original test asserted only `ll >= simple`, which a dropped
// multiplier still satisfies (they become equal). Assert INEQUALITY at a size
// where the 4x genuinely changes the channel count.
// ---------------------------------------------------------------------------
TEST_F(EnqueueMicrotest, PackedChannels_LLNeedsStrictlyMoreChannelsThanSimple_Isolated) {
  // Sweep to find a count where LL and SIMPLE must differ, then assert it. The
  // 4x multiplier means LL crosses the per-channel traffic threshold sooner.
  bool sawStrictDifference = false;
  for (size_t count : {size_t(1) << 10, size_t(1) << 11, size_t(1) << 12,
                       size_t(1) << 13, size_t(1) << 14}) {
    const int simple = rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, count,
                                                ncclFloat32, NCCL_PROTO_SIMPLE, 32);
    const int ll = rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, count,
                                            ncclFloat32, NCCL_PROTO_LL, 32);
    EXPECT_GE(ll, simple) << "count=" << count;
    if (ll > simple) sawStrictDifference = true;
  }
  EXPECT_TRUE(sawStrictDifference)
      << "LL must need strictly more channels than SIMPLE somewhere in this "
         "range -- if not, the 4x traffic multiplier is not being applied";
}

// ===========================================================================
// Interaction tests for the RCCL tuning hooks.
//
// These exist because the fakes gained call counters that nothing asserted --
// a counter no test reads cannot catch a dropped call site. topoGetAlgoInfo
// calls each of these exactly once per invocation; deleting any call site is
// invisible to a return-code or cost-table assertion, because the production
// defaults leave the selection unchanged.
// ===========================================================================

namespace {
// topoGetAlgoInfo needs slightly more comm state than updateCollCostTable: it
// reads archName for the arch-specific overrides and the maxThreads table when
// it converts an algorithm choice into a warp count.
struct AlgoInfoComm {
  std::unique_ptr<ncclTopoSystem> topo{new ncclTopoSystem{}};
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  AlgoInfoComm() {
    comm->nRanks = 8;
    comm->nNodes = 1;
    comm->maxLocalRanks = 8;
    comm->localRanks = 8;
    comm->WarpSize = 32;
    topo->type = RCCL_TOPO_XGMI_ALL;
    topo->nodes[GPU].count = 1;
    std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn,
                  sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "%s", "gfx942");
    comm->topo = topo.get();
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
        comm->maxThreads[a][p] = 256;
  }
  ncclComm* get() { return comm.get(); }
};
}  // namespace

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_ConsultsEveryRcclTuningHookExactlyOnce) {
  // Pins that each hook is actually reached. Without these assertions the
  // counters are decorative: production could stop calling rcclSetPipelining
  // entirely and every other test in this suite would still pass, because the
  // no-op default leaves the selection identical either way.
  AlgoInfoComm cc;
  auto task = CostTask(ncclFuncAllReduce);
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  initCollCostTable((float**)table);
  ScriptAllTimes(1.0f);

  float* tablePtr = (float*)table;
  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, 1 << 20, &tablePtr,
                                         /*simInfo=*/nullptr));

  EXPECT_EQ(1, g_rcclUpdateCollectiveProtocolCalls) << "rcclUpdateCollectiveProtocol (:2657)";
  EXPECT_EQ(1, g_rcclSetPipeliningCalls) << "rcclSetPipelining (:2658)";
  EXPECT_EQ(1, g_rcclUpdateThreadThresholdCalls) << "rcclUpdateThreadThreshold";
  EXPECT_EQ(1, g_rcclOptThreadBlockSizeCalls) << "rcclOptThreadBlockSize";
  EXPECT_EQ(1, g_rcclOverrideChannelsCalls) << "rcclOverrideChannels (:2660)";
  // The tuner-table overrides at :2734-2735. Omitting these made the test name
  // a lie: deleting either call site was invisible.
  EXPECT_EQ(1, g_rcclOverrideAlgorithmCalls) << "rcclOverrideAlgorithm (:2734)";
  EXPECT_EQ(1, g_rcclOverrideProtocolCalls) << "rcclOverrideProtocol (:2735)";
}

// LATENT BUG (enqueue.cc:2660): rcclOverrideChannels returns ncclResult_t, but
// the call site discards it -- there is no NCCLCHECK, unlike every other
// ncclResult_t-returning call in topoGetAlgoInfo. A failing channel override is
// silently ignored and tuning proceeds on the unmodified value.
//
// This pins today's behaviour (src/ is untouched): the failure does NOT
// propagate. If a future change adds the missing NCCLCHECK, this test fails and
// should become an assertion that the error IS returned.
TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_OverrideChannelsFailure_IsSilentlyIgnored) {
  AlgoInfoComm cc;
  auto task = CostTask(ncclFuncAllReduce);
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  initCollCostTable((float**)table);
  ScriptAllTimes(1.0f);
  g_rcclOverrideChannels = [](struct ncclComm*, ncclFunc_t, size_t, int&) {
    return ncclInvalidArgument;
  };

  float* tablePtr = (float*)table;
  EXPECT_EQ(ncclSuccess,
            topoGetAlgoInfo(cc.get(), &task, 1 << 20, &tablePtr, /*simInfo=*/nullptr))
      << "enqueue.cc:2660 discards the result, so the failure cannot surface";
  EXPECT_EQ(1, g_rcclOverrideChannelsCalls) << "but the hook WAS consulted";
}

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_OverrideChannelsCanChangeTheChannelCount) {
  // The seam writes through an int& -- pin that the value actually reaches the
  // task, not just that the hook was called.
  AlgoInfoComm cc;
  auto task = CostTask(ncclFuncAllReduce);
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  initCollCostTable((float**)table);
  ScriptAllTimes(1.0f);
  g_rcclOverrideChannels = [](struct ncclComm*, ncclFunc_t, size_t, int& nc) {
    nc = 3;
    return ncclSuccess;
  };

  float* tablePtr = (float*)table;
  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, 1 << 20, &tablePtr,
                                         /*simInfo=*/nullptr));
  EXPECT_EQ(3, task.nMaxChannels) << "the override's channel count must reach the task";
}
