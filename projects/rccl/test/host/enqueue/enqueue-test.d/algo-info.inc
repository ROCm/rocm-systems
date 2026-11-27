/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// topoGetAlgoInfo and getAlgoInfo: algorithm and channel selection, including the RCCL tuning hooks.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// Interaction tests for the RCCL tuning hooks.
//
// These exist because the fakes gained call counters that nothing asserted --
// a counter no test reads cannot catch a dropped call site. topoGetAlgoInfo
// calls each of these exactly once per invocation; deleting any call site is
// invisible to a return-code or cost-table assertion, because the production
// defaults leave the selection unchanged.
// ===========================================================================

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_ConsultsEveryRcclTuningHookExactlyOnce) {
  // Pins that each hook is actually reached. Without these assertions the
  // counters are decorative: production could stop calling rcclSetPipelining
  // entirely and every other test in this suite would still pass, because the
  // no-op default leaves the selection identical either way.
  AlgoInfoComm cc;
  auto task = CostTask(ncclFuncAllReduce);
  // Left empty on purpose: topoGetAlgoInfo never fills the table, so selection
  // takes the RING/SIMPLE fallback and only the hooks are under test.
  CostTable tbl;                       // owns the array + the initCollCostTable call

  // ptr() hands over the ARRAY cast to float**, which is what production passes
  // (:2782). topoGetAlgoInfo casts it straight back to float(*)[NCCL_NUM_PROTOCOLS]
  // and reads all NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS cells -- passing the
  // address of a float* local instead would read far past it.
  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, 1 << 20, tbl.ptr(),
                                         /*simInfo=*/nullptr));

  EXPECT_EQ(1, g_rcclUpdateCollectiveProtocolCalls) << "rcclUpdateCollectiveProtocol (:2608)";
  EXPECT_EQ(1, g_rcclSetPipeliningCalls) << "rcclSetPipelining (:2610)";
  EXPECT_EQ(1, g_rcclUpdateThreadThresholdCalls) << "rcclUpdateThreadThreshold (:2649)";
  EXPECT_EQ(1, g_rcclOptThreadBlockSizeCalls) << "rcclOptThreadBlockSize (:2731)";
  EXPECT_EQ(1, g_rcclOverrideChannelsCalls) << "rcclOverrideChannels (:2659)";
  // The tuner-table overrides at :2733-2734. Omitting these made the test name
  // a lie: deleting either call site was invisible.
  EXPECT_EQ(1, g_rcclOverrideAlgorithmCalls) << "rcclOverrideAlgorithm (:2733)";
  EXPECT_EQ(1, g_rcclOverrideProtocolCalls) << "rcclOverrideProtocol (:2734)";
  EXPECT_EQ(1, g_rcclUpdateCollectiveAlgorithmCalls) << "rcclUpdateCollectiveAlgorithm";
  EXPECT_TRUE(g_rcclUpdateCollectiveAlgorithmBeforeProtocol)
      << "rcclUpdateCollectiveAlgorithm must run before rcclUpdateCollectiveProtocol picks the Tree protocol";
  EXPECT_EQ(1, g_rcclUpdateRemoteFlushAlgoProtoCalls) << "rcclUpdateRemoteFlushAlgoProto";
  EXPECT_TRUE(g_rcclUpdateRemoteFlushAlgoProtoAfterProtocol)
      << "rcclUpdateRemoteFlushAlgoProto must run after rcclUpdateCollectiveProtocol so its choice stands";
  EXPECT_EQ(1, g_rcclUpdateCollectiveChannelsCalls) << "rcclUpdateCollectiveChannels";
}

// LATENT BUG (enqueue.cc:2659): rcclOverrideChannels returns ncclResult_t, but
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
  CostTable tbl;  // left empty: the RING/SIMPLE fallback is taken
  g_rcclOverrideChannels = [](struct ncclComm*, ncclFunc_t, size_t, int&) {
    return ncclInvalidArgument;
  };

  EXPECT_EQ(ncclSuccess,
            topoGetAlgoInfo(cc.get(), &task, 1 << 20, tbl.ptr(), /*simInfo=*/nullptr))
      << "enqueue.cc:2659 discards the result, so the failure cannot surface";
  EXPECT_EQ(1, g_rcclOverrideChannelsCalls) << "but the hook WAS consulted";
}

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_OverrideChannelsCanChangeTheChannelCount) {
  // The seam writes through an int& -- pin that the value actually reaches the
  // task, not just that the hook was called.
  AlgoInfoComm cc;
  auto task = CostTask(ncclFuncAllReduce);
  CostTable tbl;  // left empty: the RING/SIMPLE fallback is taken
  g_rcclOverrideChannels = [](struct ncclComm*, ncclFunc_t, size_t, int& nc) {
    nc = 3;
    return ncclSuccess;
  };

  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, 1 << 20, tbl.ptr(),
                                         /*simInfo=*/nullptr));
  EXPECT_EQ(3, task.nMaxChannels) << "the override's channel count must reach the task";
}

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_RemoteFlushChannelsSetTheCountCappedByComm) {
  // A positive rcclUpdateRemoteFlushAlgoProto return replaces the Ring/Tree channel count, capped at comm->nChannels.
  for (int commChannels : {64, 32}) {
    AlgoInfoComm cc;
    cc.get()->nChannels = commChannels;
    auto task = CostTask(ncclFuncAllReduce);
    CostTable tbl;  // left empty: the RING/SIMPLE fallback is taken
    g_rcclUpdateRemoteFlushAlgoProtoChannels = 48;
    ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, 1 << 20, tbl.ptr(), /*simInfo=*/nullptr));
    EXPECT_EQ(std::min(48, commChannels), task.nMaxChannels) << "comm->nChannels=" << commChannels;
  }
  AlgoInfoComm cc;
  cc.get()->nChannels = 64;
  auto task = CostTask(ncclFuncAllReduce);
  CostTable tbl;
  g_rcclUpdateRemoteFlushAlgoProtoChannels = 0;
  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, 1 << 20, tbl.ptr(), /*simInfo=*/nullptr));
  EXPECT_EQ(64, task.nMaxChannels) << "0 keeps RCCL's own count";
}

// The NCCL_MIN_NCHANNELS floor at :2652-2656. Left undriven, g_paramMinNchannels
// defaults to 0 and the whole clamp degenerates: `nc > minNChannels` becomes
// `nc > 0` and `std::max(minNChannels, X)` becomes `X`, so mutating both
// ncclParamMinNchannels() call sites to a constant left the suite green. Driving
// it is what makes the floor observable.
TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_MinNchannelsIsAFloorOnTheShrink) {
  // The shrink is guarded by `nBytes < nc * nt * threadThreshold && nc > minNChannels`,
  // so BOTH conjuncts have to be armed or the branch never runs: nc starts at
  // comm->nChannels (:2613) and threadThreshold at comm->threadThresholds, and the
  // fixture leaves both zero, which makes the product zero and the comparison false.
  AlgoInfoComm cc;
  cc.get()->nChannels = 8;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
      cc.get()->threadThresholds[a][p] = 64;
    }
  }

  auto task = CostTask(ncclFuncAllReduce);
  CostTable tbl;  // left empty: the RING/SIMPLE fallback is taken
  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, /*nBytes=*/1, tbl.ptr(),
                                         /*simInfo=*/nullptr));
  const int unclamped = task.nMaxChannels;
  ASSERT_LT(unclamped, 4)
      << "the shrink must actually collapse nc, or the floor below proves nothing";

  auto task2 = CostTask(ncclFuncAllReduce);
  CostTable tbl2;
  g_paramMinNchannels = 4;
  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task2, /*nBytes=*/1, tbl2.ptr(),
                                         /*simInfo=*/nullptr));
  EXPECT_EQ(4, task2.nMaxChannels)
      << "NCCL_MIN_NCHANNELS must floor the shrink; unclamped was " << unclamped;
}

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_PatMaxNchannelsClampsTheChannelCount) {
  AlgoInfoComm cc;
  cc.get()->nChannels = 8;
  auto task = CostTask(ncclFuncAllGather);
  CostTable tbl;
  tbl.t[NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 0.5f;
  g_paramMaxNchannels = 3;

  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, /*nBytes=*/2 << 20, tbl.ptr(),
                                         /*simInfo=*/nullptr));
  EXPECT_EQ(NCCL_ALGO_PAT, task.algorithm);
  EXPECT_EQ(3, task.nMaxChannels) << "positive NCCL_MAX_NCHANNELS must clamp PAT's channel count";
}

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_PatUnsetMaxNchannelsDoesNotClamp) {
  AlgoInfoComm cc;
  cc.get()->nChannels = 8;
  auto task = CostTask(ncclFuncAllGather);
  CostTable tbl;
  tbl.t[NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 0.5f;
  g_paramMaxNchannels = -2;

  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, /*nBytes=*/2 << 20, tbl.ptr(),
                                         /*simInfo=*/nullptr));
  EXPECT_EQ(NCCL_ALGO_PAT, task.algorithm);
  EXPECT_EQ(8, task.nMaxChannels) << "the production -2 sentinel must leave PAT's channel count alone";
}

// Guards the fix for the &tablePtr defect: topoGetAlgoInfo must read the cost
// table it was handed. updateCollCostTable is driven first because it owns the
// only ncclTopoGetAlgoTime call site (:2539), so it is what makes the scripted
// costs real; topoGetAlgoInfo alone would leave every cell IGNOREd. The winner
// is TREE/LL precisely because it is NOT the RING/SIMPLE the defaults at
// :2573-2580 install: asserting the fallback would stay green with the argmin
// loop deleted, with the comparison inverted, or with &tablePtr restored.
TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_SelectsTheCheapestScriptedCell) {
  AlgoInfoComm cc;
  auto task = CostTask(ncclFuncAllReduce);
  CostTable tbl;
  // One strictly cheapest cell, deliberately not the default pair.
  g_topoGetAlgoTime = [](struct ncclComm*, int, int a, int p, size_t, int, float* t) {
    if (t) {
      *t = (a == NCCL_ALGO_TREE && p == NCCL_PROTO_LL) ? 0.5f : 1.0f;
    }
    return ncclSuccess;
  };

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0,
                                             /*nvls=*/0, /*numPipeOps=*/1,
                                             /*userAlgoInput=*/0, tbl.ptr()));
  ASSERT_TRUE(tbl.written(NCCL_ALGO_TREE, NCCL_PROTO_LL)) << "the winning cell must be populated";
  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, 1 << 20, tbl.ptr(),
                                         /*simInfo=*/nullptr));
  EXPECT_EQ(NCCL_ALGO_TREE, task.algorithm) << "RING here means the table was not read";
  EXPECT_EQ(NCCL_PROTO_LL, task.protocol) << "SIMPLE here means the table was not read";
}

// The per-call selection has to survive the argmin, not just narrow the table.
// The sibling cases above stop at the cost table, so none of them would notice
// topoGetAlgoInfo overwriting the winner. RING/LL is scripted strictly cheapest
// and is NOT named, so an unfiltered argmin picks it; TREE/LL is the named row
// and differs from the RING/SIMPLE defaults at :2573-2580, which keeps both the
// "selection ignored" and "fell back to the default pair" outcomes red.
TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_HonorsThePerCallAlgSelection) {
  AlgoInfoComm cc;
  auto task = CostTask(ncclFuncAllReduce);
  CostTable tbl;
  task.algMask = GeneralBit(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  g_topoGetAlgoTime = [](struct ncclComm*, int, int a, int p, size_t, int, float* t) {
    if (t) {
      *t = (a == NCCL_ALGO_RING && p == NCCL_PROTO_LL) ? 0.5f : 1.0f;
    }
    return ncclSuccess;
  };

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0,
                                             /*nvls=*/0, /*numPipeOps=*/1,
                                             /*userAlgoInput=*/0, tbl.ptr()));
  // EXPECT, not ASSERT: a regression should report the selection too, not stop at the table.
  EXPECT_TRUE(tbl.written(NCCL_ALGO_TREE, NCCL_PROTO_LL)) << "the named row must survive the filter";
  EXPECT_FALSE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_LL)) << "the cheaper unnamed row must be blanked";

  ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, 1 << 20, tbl.ptr(),
                                         /*simInfo=*/nullptr));
  EXPECT_EQ(NCCL_ALGO_TREE, task.algorithm) << "RING here means the selection did not reach the argmin";
  EXPECT_EQ(NCCL_PROTO_LL, task.protocol) << "SIMPLE here means the default pair won";
}

// ===========================================================================
// Topology, arch, tuner, NVLS and CollNet arms of topoGetAlgoInfo, getAlgoInfo, updateCollCostTable, calcCollChunking.
// ===========================================================================

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_ThreePivotBiRings_PicksTreeLlUpToTheCutoffThenRingSimple) {
#ifdef ENABLE_WARP_SPEED
  GTEST_SKIP() << "Known gap: rcclSetWarpSpeedAuto's final channel writeback "
                  "overwrites the pivot override's 24-channel result";
#endif
  // Each case seeds the opposite pick as cheapest, so the override must replace both fields.
  struct {
    size_t nBytes;
    int algo;
    int proto;
    int nMaxChannels;
  } const cases[] = {{2200008, NCCL_ALGO_TREE, NCCL_PROTO_LL, 24}, {2200009, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 32}};
  for (const auto& c : cases) {
    AlgoInfoComm cc;
    cc.get()->nChannels = 32;
    cc.base.topo().pivotA2ANumBiRings = 3;
    auto task = CostTask(ncclFuncAllReduce);
    CostTable tbl;
    const bool wantTree = c.algo == NCCL_ALGO_TREE;
    tbl.t[wantTree ? NCCL_ALGO_RING : NCCL_ALGO_TREE][wantTree ? NCCL_PROTO_SIMPLE : NCCL_PROTO_LL] = 0.5f;
    ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, c.nBytes, tbl.ptr(), /*simInfo=*/nullptr));
    EXPECT_EQ(c.algo, task.algorithm) << c.nBytes;
    EXPECT_EQ(c.proto, task.protocol) << c.nBytes;
    EXPECT_EQ(c.nMaxChannels, task.nMaxChannels) << c.nBytes;
  }
}

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_TreeDefinedAllReduce_ForcesTreeCappedAtSixtyFourChannels) {
  for (int treeDefined : {0, 1}) {
    AlgoInfoComm cc;
    cc.get()->nChannels = 80;
    cc.base.topo().treeDefined = treeDefined;
    auto task = CostTask(ncclFuncAllReduce);
    CostTable tbl;
    tbl.t[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 0.5f;
    ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, 1 << 20, tbl.ptr(), /*simInfo=*/nullptr));
    EXPECT_EQ(treeDefined ? NCCL_ALGO_TREE : NCCL_ALGO_RING, task.algorithm);
    EXPECT_EQ(treeDefined ? 64 : 80, task.nMaxChannels);
  }
}

namespace {
// Scripts updateCollCostTable so (algo, proto) is the cheapest cell; getAlgoInfo then runs its no-tuner overrides.
void ScriptCheapest(int algo, int proto) {
  g_topoGetAlgoTime = [algo, proto](ncclComm*, int, int a, int p, size_t, int, float* time) {
    *time = (a == algo && p == proto) ? 1.0f : 5.0f;
    return ncclSuccess;
  };
}

ncclTaskColl NoTunerTask(size_t count) {
  ncclTaskColl task = CostTask(ncclFuncAllReduce);
  task.count = count;
  return task;
}
}  // namespace

TEST_F(EnqueueMicrotest, GetAlgoInfo_NoTunerGfx950SingleNodeAllReduce_UsesTreeLlInsideItsWindow) {
  struct {
    size_t bytes;
    int nNodes;
    bool overridden;
  } const cases[] = {{60, 1, false}, {64, 1, true}, {262144, 1, true}, {262148, 1, false}, {1024, 2, false}};
  for (const auto& c : cases) {
    AlgoInfoComm cc;
    cc.get()->nNodes = c.nNodes;
    SetSingleGpuArch(&cc.base.topo(), "gfx950");
    ScriptCheapest(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
    ncclTaskColl task = NoTunerTask(c.bytes / sizeof(float));
    ASSERT_EQ(ncclSuccess, getAlgoInfo(cc.get(), &task, 0, 0, 1, nullptr));
    EXPECT_EQ(c.overridden ? NCCL_ALGO_TREE : NCCL_ALGO_RING, task.algorithm) << c.bytes;
    EXPECT_EQ(c.overridden ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE, task.protocol) << c.bytes;
  }
}

TEST_F(EnqueueMicrotest, GetAlgoInfo_NoTunerNaviSingleNodeAllReduce_UsesRing) {
  for (const char* gcn : {"gfx1200", "gfx1201", "gfx942"}) {
    AlgoInfoComm cc;
    SetSingleGpuArch(&cc.base.topo(), gcn);
    ScriptCheapest(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
    ncclTaskColl task = NoTunerTask(1 << 20);
    ASSERT_EQ(ncclSuccess, getAlgoInfo(cc.get(), &task, 0, 0, 1, nullptr));
    const bool navi = std::strcmp(gcn, "gfx942") != 0;
    EXPECT_EQ(navi ? NCCL_ALGO_RING : NCCL_ALGO_TREE, task.algorithm) << gcn;
  }
}

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_CollnetDirect_ShedsChannelsForSmallMessages) {
  struct {
    size_t nBytes;
    int nMaxChannels;
  } const cases[] = {{1, 1}, {262144, 23}, {size_t(1) << 30, 32}};
  for (const auto& c : cases) {
    AlgoInfoComm cc;
    cc.get()->nChannels = 32;
    cc.get()->channels[0].collnetDirect.nHeads = 1;
    cc.get()->threadThresholds[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 64;
    auto task = CostTask(ncclFuncAllReduce);
    CostTable tbl;
    tbl.t[NCCL_ALGO_COLLNET_DIRECT][NCCL_PROTO_SIMPLE] = 0.5f;
    ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, c.nBytes, tbl.ptr(), /*simInfo=*/nullptr));
    EXPECT_EQ(NCCL_ALGO_COLLNET_DIRECT, task.algorithm);
    EXPECT_EQ(c.nMaxChannels, task.nMaxChannels) << c.nBytes;
  }
}

TEST_F(EnqueueMicrotest, TopoGetAlgoInfo_Nvls_UsesNvlsChannelsCappedByCommOnlyForMultiNodeNvls) {
  struct {
    int algo;
    int nNodes;
    int nMaxChannels;
  } const cases[] = {{NCCL_ALGO_NVLS, 2, 8}, {NCCL_ALGO_NVLS, 1, 12}, {NCCL_ALGO_NVLS_TREE, 2, 12}};
  for (const auto& c : cases) {
    AlgoInfoComm cc;
    cc.get()->nChannels = 8;
    cc.get()->nvlsChannels = 12;
    cc.get()->nNodes = c.nNodes;
    auto task = CostTask(ncclFuncAllReduce);
    CostTable tbl;
    tbl.t[c.algo][NCCL_PROTO_SIMPLE] = 0.5f;
    ASSERT_EQ(ncclSuccess, topoGetAlgoInfo(cc.get(), &task, 1 << 20, tbl.ptr(), /*simInfo=*/nullptr));
    EXPECT_EQ(c.algo, task.algorithm);
    EXPECT_EQ(c.nMaxChannels, task.nMaxChannels) << c.algo << " nNodes=" << c.nNodes;
  }
}

// ===========================================================================
// ncclGroupSimulateEnd estimatedTime: a tuner-forced 0 reports the cost-model
// time of the chosen algorithm, and a direct AllGather is still costed.
// ===========================================================================

namespace {
constexpr float kOtherModelUs = 100.0f;

// The CSV tuner writes 0 into the cell it wants. The 0 selects; it is not a time.
ncclResult_t ZeroTreeLlCell(void*, ncclFunc_t, size_t, int, float** collCostTable, int, int, int, int* nChannels) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  table[NCCL_ALGO_TREE][NCCL_PROTO_LL] = 0.0f;
  if (nChannels) *nChannels = 0;
  return ncclSuccess;
}

void AttachZeroTuner(ncclComm* comm, ncclTuner_t* tuner) {
  *tuner = ncclTuner_t{};
  tuner->getCollInfo = ZeroTreeLlCell;
  comm->tuner = tuner;
  // An empty registration cache makes regBuff false once the graph is invalid,
  // so getAlgoInfo does not consult graph registration.
  comm->planner.capturingGraph = ncclCudaGraphNone(comm->config.graphUsageMode);
}
}  // namespace

TEST_F(EnqueueMicrotest, GetAlgoInfo_TunerZero_ReportsCostModelTimeOfTheChosenAlgo) {
  AlgoInfoComm cc;
  ncclTuner_t tuner{};
  AttachZeroTuner(cc.get(), &tuner);
  ScriptCellTime(NCCL_ALGO_TREE, NCCL_PROTO_LL, kChosenModelUs, kOtherModelUs);
  ncclTaskColl task = CostTask(ncclFuncAllReduce);
  task.count = 1024;
  ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;

  ASSERT_EQ(ncclSuccess, getAlgoInfo(cc.get(), &task, /*collNetSupport=*/0, /*nvlsSupport=*/0,
                                     /*numPipeOps=*/1, &sim));
  EXPECT_EQ(NCCL_ALGO_TREE, task.algorithm) << "the 0 must still win selection";
  EXPECT_EQ(NCCL_PROTO_LL, task.protocol);
  EXPECT_FLOAT_EQ(kChosenModelUs, sim.estimatedTime)
      << "estimatedTime must be the cost-model time, not the tuner's 0";
}

TEST_F(EnqueueMicrotest, GetAlgoInfo_TunerZero_NullSimInfoSucceeds) {
  // A real launch passes a null simulation. The tuner branch must not
  // dereference it, and the 0 must still select TREE/LL.
  AlgoInfoComm cc;
  ncclTuner_t tuner{};
  AttachZeroTuner(cc.get(), &tuner);
  ScriptCellTime(NCCL_ALGO_TREE, NCCL_PROTO_LL, kChosenModelUs, kOtherModelUs);
  ncclTaskColl task = CostTask(ncclFuncAllReduce);
  task.count = 1024;

  ASSERT_EQ(ncclSuccess, getAlgoInfo(cc.get(), &task, 0, 0, 1, nullptr));
  EXPECT_EQ(NCCL_ALGO_TREE, task.algorithm);
  EXPECT_EQ(NCCL_PROTO_LL, task.protocol);
}

TEST_F(EnqueueMicrotest, GetAlgoInfo_TunerZero_KeepsZeroWhenCostModelHasNoPositiveTime) {
  AlgoInfoComm cc;
  ncclTuner_t tuner{};
  AttachZeroTuner(cc.get(), &tuner);
  ScriptCellTime(NCCL_ALGO_TREE, NCCL_PROTO_LL, NCCL_ALGO_PROTO_IGNORE, kOtherModelUs);
  ncclTaskColl task = CostTask(ncclFuncAllReduce);
  task.count = 1024;
  ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;

  ASSERT_EQ(ncclSuccess, getAlgoInfo(cc.get(), &task, 0, 0, 1, &sim));
  EXPECT_EQ(NCCL_ALGO_TREE, task.algorithm);
  EXPECT_EQ(NCCL_PROTO_LL, task.protocol);
  EXPECT_FLOAT_EQ(0.0f, sim.estimatedTime)
      << "a non-positive model time must not be reported in place of the sentinel";
}

TEST_F(EnqueueMicrotest, GetAlgoInfo_TunerZero_ShortCircuitDoesNotQueryTheCostModel) {
  // These 0s come from updateCollCostTable, not from a tuner choice. AlltoAllPivot
  // is past NCCL_NUM_FUNCTIONS, so ncclTopoGetAlgoTime would read off the end of
  // comm->bandwidths. One rank is the other arm of that same short circuit.
  struct {
    int nRanks;
    ncclFunc_t func;
  } const cases[] = {{8, ncclFuncAlltoAllPivot}, {1, ncclFuncAllReduce}};
  for (const auto& c : cases) {
    AlgoInfoComm cc;
    cc.get()->nRanks = c.nRanks;
    ncclTuner_t tuner{};
    AttachZeroTuner(cc.get(), &tuner);
    ScriptCellTime(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, kChosenModelUs, kOtherModelUs);
    g_topoGetAlgoTimeCalls = 0;
    ncclTaskColl task = CostTask(c.func);
    task.count = 1024;
    ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;

    ASSERT_EQ(ncclSuccess, getAlgoInfo(cc.get(), &task, 0, 0, 1, &sim)) << c.func;
    EXPECT_EQ(0, g_topoGetAlgoTimeCalls) << c.func;
    EXPECT_FLOAT_EQ(0.0f, sim.estimatedTime) << c.func;
  }
}

TEST_F(EnqueueMicrotest, GetAlgoInfo_SingleRankZeroCost_StaysZeroWithoutATuner) {
  // nRanks == 1 writes RING/SIMPLE = 0 and never asks the cost model. That 0 is
  // a real cost, and there is no tuner sentinel to replace.
  AlgoInfoComm cc;
  cc.get()->nRanks = 1;
  ScriptCellTime(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, kChosenModelUs, kOtherModelUs);
  ncclTaskColl task = CostTask(ncclFuncAllReduce);
  task.count = 1024;
  ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;

  ASSERT_EQ(ncclSuccess, getAlgoInfo(cc.get(), &task, 0, 0, 1, &sim));
  EXPECT_EQ(0, g_topoGetAlgoTimeCalls) << "the single-rank short circuit must not query the model";
  EXPECT_EQ(NCCL_ALGO_RING, task.algorithm);
  EXPECT_EQ(NCCL_PROTO_SIMPLE, task.protocol);
  EXPECT_FLOAT_EQ(0.0f, sim.estimatedTime);
}
