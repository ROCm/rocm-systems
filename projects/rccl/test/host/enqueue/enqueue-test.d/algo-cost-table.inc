/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ncclGetCollNetSupport, initCollCostTable and updateCollCostTable: the per-(algo, proto) cost table.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// ncclGetCollNetSupport (enqueue.cc:2457)
// Table lookup with an op-translation step. The translation is the subtle part:
// PreMulSum and SumPostDiv both query the table as ncclSum.
// ===========================================================================

namespace {
struct CollNetComm {
  std::unique_ptr<ncclComm> storage{new ncclComm{}};   // ~3.8 MB: not stack-safe
  CollNetComm(int enable, bool sumSupported) {
    ncclComm& comm = *storage;
    comm.config.collnetEnable = enable;
    std::memset(comm.collNetSupportMatrix, 0, sizeof(comm.collNetSupportMatrix));
    if (sumSupported) comm.collNetSupportMatrix[ncclSum][ncclFloat32] = 1;
  }
  ncclComm* get() { return storage.get(); }
};

ncclTaskColl CollNetTask(ncclFunc_t f, ncclRedOp_t hostOp, ncclDevRedOp_t devOp) {
  ncclTaskColl t{};
  t.func = f;
  t.opHost = hostOp;
  t.opDev.op = devOp;
  t.datatype = ncclFloat32;
  return t;
}
}  // namespace

TEST_F(EnqueueMicrotest, GetCollNetSupport_NonReducingFunc_IgnoresMatrix) {
  // AllGather hits the `default: break` arm, so the matrix is never consulted --
  // the answer is purely comm->config.collnetEnable.
  CollNetComm cc(/*enable=*/1, /*sumSupported=*/false);
  auto t = CollNetTask(ncclFuncAllGather, ncclSum, ncclDevSum);
  int out = -1;
  ASSERT_EQ(ncclSuccess, ncclGetCollNetSupport(cc.get(), &t, &out));
  EXPECT_EQ(1, out) << "matrix says unsupported, but AllGather must not consult it";
}

TEST_F(EnqueueMicrotest, GetCollNetSupport_ReducingFuncs_AndWithMatrix) {
  // All three reducing funcs share one arm; assert each so a dropped case label
  // is visible.
  for (auto f : {ncclFuncAllReduce, ncclFuncReduce, ncclFuncReduceScatter}) {
    CollNetComm yes(1, /*sumSupported=*/true);
    auto t = CollNetTask(f, ncclSum, ncclDevSum);
    int out = -1;
    ASSERT_EQ(ncclSuccess, ncclGetCollNetSupport(yes.get(), &t, &out));
    EXPECT_EQ(1, out) << "func=" << int(f);

    CollNetComm no(1, /*sumSupported=*/false);
    int out2 = -1;
    ASSERT_EQ(ncclSuccess, ncclGetCollNetSupport(no.get(), &t, &out2));
    EXPECT_EQ(0, out2) << "func=" << int(f) << " must AND with the matrix";
  }
}

TEST_F(EnqueueMicrotest, GetCollNetSupport_CollnetDisabled_AlwaysZero) {
  // The AND's other operand: enable=0 wins regardless of the matrix.
  CollNetComm cc(/*enable=*/0, /*sumSupported=*/true);
  auto t = CollNetTask(ncclFuncAllReduce, ncclSum, ncclDevSum);
  int out = -1;
  ASSERT_EQ(ncclSuccess, ncclGetCollNetSupport(cc.get(), &t, &out));
  EXPECT_EQ(0, out);
}

TEST_F(EnqueueMicrotest, GetCollNetSupport_PreMulSumAndPostDiv_QueryTableAsSum) {
  // The translation at :2460. opHost is deliberately a DIFFERENT op whose matrix
  // entry is 0; if the translation were dropped, the lookup would use opHost and
  // return 0. Getting 1 proves the query went through ncclSum.
  for (auto devOp : {ncclDevPreMulSum, ncclDevSumPostDiv}) {
    CollNetComm cc(1, /*sumSupported=*/true);   // only [ncclSum][float32] is set
    auto t = CollNetTask(ncclFuncAllReduce, /*opHost=*/ncclMax, devOp);
    int out = -1;
    ASSERT_EQ(ncclSuccess, ncclGetCollNetSupport(cc.get(), &t, &out));
    EXPECT_EQ(1, out) << "devOp=" << int(devOp) << " must be queried as ncclSum";
  }
}

TEST_F(EnqueueMicrotest, GetCollNetSupport_OrdinaryOp_UsesHostOpNotSum) {
  // Differential against the test above: a non-translated op must use opHost, so
  // ncclMax (matrix entry 0) gives 0 even though ncclSum's entry is 1.
  CollNetComm cc(1, /*sumSupported=*/true);
  auto t = CollNetTask(ncclFuncAllReduce, /*opHost=*/ncclMax, ncclDevMinMax);
  int out = -1;
  ASSERT_EQ(ncclSuccess, ncclGetCollNetSupport(cc.get(), &t, &out));
  EXPECT_EQ(0, out);
}

// ===========================================================================
// initCollCostTable (enqueue.cc:2476)
// Fills an ALGORITHMS x PROTOCOLS table with NCCL_ALGO_PROTO_IGNORE through a
// float** -> float(*)[N] reinterpret cast. Worth pinning precisely because the
// cast is fragile: a wrong stride writes out of bounds or leaves holes.
// ===========================================================================

TEST_F(EnqueueMicrotest, InitCollCostTable_MarksEveryCellIgnored) {
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  // Poison every cell first: a partial fill must be detectable.
  for (auto& row : table) {
    for (auto& v : row) v = -1.0f;
  }

  initCollCostTable((float**)table);

  for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
      EXPECT_FLOAT_EQ(NCCL_ALGO_PROTO_IGNORE, table[a][p]) << "a=" << a << " p=" << p;
    }
  }
}

TEST_F(EnqueueMicrotest, InitCollCostTable_DoesNotWritePastTheTable) {
  // Guard rows on both sides catch a wrong stride in the (float(*)[N]) cast.
  struct {
    float before[NCCL_NUM_PROTOCOLS];
    float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
    float after[NCCL_NUM_PROTOCOLS];
  } buf;
  for (auto& v : buf.before) v = 42.0f;
  for (auto& v : buf.after) v = 42.0f;
  for (auto& row : buf.table) {
    for (auto& v : row) v = -1.0f;
  }

  initCollCostTable((float**)buf.table);

  for (auto v : buf.before) EXPECT_FLOAT_EQ(42.0f, v) << "wrote before the table";
  for (auto v : buf.after) EXPECT_FLOAT_EQ(42.0f, v) << "wrote past the table";
}

// ===========================================================================
// updateCollCostTable (enqueue.cc:2486) -- fills the cost table that
// topoGetAlgoInfo then argmins over. Driven entirely through the
// g_topoGetAlgoTime seam; ~14 `continue` guards decide which cells get a time.
//
// The oracle is WHICH CELLS were written, so every test compares against
// NCCL_ALGO_PROTO_IGNORE rather than checking a return code.
// ===========================================================================

TEST_F(EnqueueMicrotest, UpdateCollCostTable_SingleRank_ShortCircuitsToRingSimple) {
  // The nRanks==1 fast path writes exactly ONE cell and returns, without
  // consulting the topology at all.
  CostComm cc(/*nRanks=*/1);
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/1,
                                             /*nvls=*/1, /*numPipeOps=*/1,
                                             /*userAlgoInput=*/0, tbl.ptr()));
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE));
  EXPECT_EQ(1, tbl.countWritten()) << "the fast path must write exactly one cell";
  EXPECT_EQ(0, g_topoGetAlgoTimeCalls) << "and must not query the topology";
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_AlltoAllFuncs_TakeTheSameFastPath) {
  // Three funcs share the nRanks==1 short circuit even at many ranks.
  for (auto f : {ncclFuncAlltoAllPivot, ncclFuncAlltoAllGda, ncclFuncAlltoAllvGda}) {
    ResetEnqueueTestDeps();
    CostComm cc(/*nRanks=*/8);
    CostTable tbl;
    auto task = CostTask(f);
    ScriptAllTimes(1.0f);
    ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, 1, 1, 1, 0, tbl.ptr()))
        << "func=" << int(f);
    EXPECT_EQ(1, tbl.countWritten()) << "func=" << int(f);
    EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE)) << "func=" << int(f);
    EXPECT_EQ(0, g_topoGetAlgoTimeCalls) << "func=" << int(f);
  }
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_CollNetUnsupported_SkipsCollNetAlgorithms) {
  // `collNetSupport != 1` skips both COLLNET_DIRECT and COLLNET_CHAIN.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  ScriptAllTimes(1.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0,
                                             /*nvls=*/0, 1, 0, tbl.ptr()));
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
    EXPECT_FALSE(tbl.written(NCCL_ALGO_COLLNET_DIRECT, p)) << "p=" << p;
    EXPECT_FALSE(tbl.written(NCCL_ALGO_COLLNET_CHAIN, p)) << "p=" << p;
  }
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_TooManyLocalRanks_SkipsCollNetAlgorithms) {
  // The arity guard: maxLocalRanks > NCCL_MAX_DIRECT_ARITY+1 disables both
  // CollNet algorithms even when collNetSupport==1. Differential against the
  // control below, so the guard cannot be confused with the support flag.
  CostComm many;
  many.get()->maxLocalRanks = NCCL_MAX_DIRECT_ARITY + 2;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  ScriptAllTimes(1.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(many.get(), &task, 1 << 20, /*collNet=*/1,
                                             0, 1, 0, tbl.ptr()));
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
    EXPECT_FALSE(tbl.written(NCCL_ALGO_COLLNET_DIRECT, p));
    EXPECT_FALSE(tbl.written(NCCL_ALGO_COLLNET_CHAIN, p));
  }

  // Control: at the arity limit the CollNet rows ARE populated.
  ResetEnqueueTestDeps();
  CostComm ok;
  ok.get()->maxLocalRanks = NCCL_MAX_DIRECT_ARITY + 1;
  CostTable tbl2;
  ScriptAllTimes(1.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(ok.get(), &task, 1 << 20, 1, 0, 1, 0, tbl2.ptr()));
  EXPECT_TRUE(tbl2.written(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE));
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_NvlsUnsupported_SkipsNvlsAlgorithms) {
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  ScriptAllTimes(1.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0,
                                             /*nvls=*/0, 1, 0, tbl.ptr()));
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
    EXPECT_FALSE(tbl.written(NCCL_ALGO_NVLS, p)) << "p=" << p;
    EXPECT_FALSE(tbl.written(NCCL_ALGO_NVLS_TREE, p)) << "p=" << p;
  }

  // Control: the negative above passes just as well if the skip at :2503 is
  // unconditional. Flipping only nvlsSupport must populate the same rows, which
  // is what makes this a test of the guard rather than of NVLS being absent.
  CostTable on;
  auto task2 = CostTask(ncclFuncAllReduce);
  ScriptAllTimes(1.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task2, 1 << 20, /*collNet=*/0,
                                             /*nvls=*/1, 1, 0, on.ptr()));
  bool anyNvls = false;
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
    anyNvls |= on.written(NCCL_ALGO_NVLS, p) || on.written(NCCL_ALGO_NVLS_TREE, p);
  }
  EXPECT_TRUE(anyNvls) << "with nvlsSupport=1 the NVLS rows must be populated";
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_TopoGetAlgoTimeFailure_Propagates) {
  // Every cell goes through NCCLCHECK(ncclTopoGetAlgoTime(...)), so a failure
  // must abort the whole fill rather than leaving a half-populated table.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  g_topoGetAlgoTime = [](struct ncclComm*, int, int, int, size_t, int, float*) {
    return ncclInternalError;
  };
  EXPECT_EQ(ncclInternalError,
            updateCollCostTable(cc.get(), &task, 1 << 20, 0, 0, 1, 0, tbl.ptr()));
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_Fp8Gfx1250SingleNodeAllReduce_RingNotPenalised) {
  // With DDA LL128 on, the ring is exempt at any size, including just below the ceiling.
  constexpr size_t kCeiling = 256ULL << 20;
  for (int nRanks : {kFp8RingPenaltyRanks + 1, 16, 32}) {
    for (auto dt : {ncclFloat8e4m3, ncclFloat8e5m2}) {
      for (size_t nBytes : {kCeiling, kCeiling - 1}) {
        ResetEnqueueTestDeps();
        g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return kCeiling; };
        CostComm big(nRanks);
        big.get()->archName = const_cast<char*>("gfx1250");
        CostTable tbl;
        auto task = CostTask(ncclFuncAllReduce, dt);
        ScriptAllTimes(2.0f);
        ASSERT_EQ(ncclSuccess, updateCollCostTable(big.get(), &task, nBytes, 0, 0, 1, 0, tbl.ptr()));
        EXPECT_FLOAT_EQ(2.0f, tbl.t[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE])
            << "nRanks=" << nRanks << " dtype=" << int(dt) << " nBytes=" << nBytes;
        EXPECT_FLOAT_EQ(2.0f, tbl.t[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE])
            << "nRanks=" << nRanks << " dtype=" << int(dt) << " nBytes=" << nBytes;
      }
    }
  }
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_Fp8AtOrBelowEightRanks_NotPenalised) {
  // The sentinel itself must not be penalised. The penalty starts at the next rank.
  CostComm eight(/*nRanks=*/kFp8RingPenaltyRanks);
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce, ncclFloat8e4m3);
  ScriptAllTimes(2.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(eight.get(), &task, 1 << 20, 0, 0, 1, 0, tbl.ptr()));
  EXPECT_FLOAT_EQ(2.0f, tbl.t[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_Fp8NonAllReduceRingAboveEightRanks_IsPenalised) {
  // Same shape as the exempt allreduce, except the collective.
  constexpr size_t kCeiling = 256ULL << 20;
  ResetEnqueueTestDeps();
  g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return kCeiling; };
  CostComm big(/*nRanks=*/16);
  big.get()->archName = const_cast<char*>("gfx1250");
  CostTable tbl;
  auto task = CostTask(ncclFuncReduceScatter, ncclFloat8e4m3);
  ScriptAllTimes(2.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(big.get(), &task, kCeiling, 0, 0, 1, 0, tbl.ptr()));
  EXPECT_FLOAT_EQ(2.0f * 1024.0f, tbl.t[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
  EXPECT_FLOAT_EQ(2.0f, tbl.t[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_Fp8AllReduceOutsideGfx1250LargeRingCase_IsPenalised) {
  constexpr size_t kCeiling = 256ULL << 20;
  ResetEnqueueTestDeps();
  g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return kCeiling; };
  CostComm big(/*nRanks=*/16);
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce, ncclFloat8e4m3);
  ScriptAllTimes(2.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(big.get(), &task, kCeiling, 0, 0, 1, 0, tbl.ptr()));
  EXPECT_FLOAT_EQ(2.0f * 1024.0f, tbl.t[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
  EXPECT_FLOAT_EQ(2.0f, tbl.t[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]);
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_Fp8Gfx1250ExemptionDropsWhenNodeCountFlips) {
  constexpr size_t kCeiling = 256ULL << 20;
  struct RankNodes { int nRanks; int nNodes; };
  for (RankNodes setup : {RankNodes{kFp8RingPenaltyRanks + 1, 2}, RankNodes{16, 2}}) {
    ResetEnqueueTestDeps();
    g_rcclDdaLL128Threshold = [](const ncclComm*, ncclFunc_t) { return kCeiling; };
    CostComm big(setup.nRanks, setup.nNodes);
    big.get()->archName = const_cast<char*>("gfx1250");
    CostTable tbl;
    auto task = CostTask(ncclFuncAllReduce, ncclFloat8e4m3);
    ScriptAllTimes(2.0f);
    ASSERT_EQ(ncclSuccess, updateCollCostTable(big.get(), &task, kCeiling, 0, 0, 1, 0, tbl.ptr()));
    EXPECT_FLOAT_EQ(2.0f * 1024.0f, tbl.t[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE])
        << "nRanks=" << setup.nRanks << " nNodes=" << setup.nNodes;
    EXPECT_FLOAT_EQ(2.0f, tbl.t[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE])
        << "nRanks=" << setup.nRanks << " nNodes=" << setup.nNodes;
  }
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_NonFp8_NeverPenalised) {
  // The datatype half of the guard: float32 at many ranks stays unpenalised.
  CostComm big(/*nRanks=*/64);
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce, ncclFloat32);
  ScriptAllTimes(2.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(big.get(), &task, 1 << 20, 0, 0, 1, 0, tbl.ptr()));
  EXPECT_FLOAT_EQ(2.0f, tbl.t[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_NonXgmiTopology_IgnoresLL128) {
  // :2535 -- LL128 is gated on the topology being all-XGMI (unless the user
  // explicitly set NCCL_PROTO). Without XGMI the LL128 column stays IGNOREd
  // while SIMPLE and LL are still filled, so this is not just "nothing ran".
  CostComm cc;
  cc.topo().type = 0;  // not XGMI_ALL
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  ScriptAllTimes(1.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, 0, 0, 1, 0, tbl.ptr()));

  EXPECT_FALSE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_LL128)) << "LL128 must be gated off";
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE)) << "but SIMPLE must still fill";
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_LL)) << "and LL must still fill";
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_XgmiTopology_FillsLL128) {
  // The other side of the same gate -- differential with the test above.
  CostComm cc;  // constructor sets XGMI_ALL
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  ScriptAllTimes(1.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, 0, 0, 1, 0, tbl.ptr()));
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_LL128));
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_ForwardsFuncAndBytesToTheTimeQuery) {
  // The seam is the only place the collective identity reaches the cost model;
  // pin that func and nBytes are passed through rather than defaulted.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncReduceScatter);
  int seenFunc = -1;
  size_t seenBytes = 0;
  g_topoGetAlgoTime = [&](struct ncclComm*, int coll, int, int, size_t nBytes, int,
                          float* time) {
    seenFunc = coll;
    seenBytes = nBytes;
    if (time) *time = 1.0f;
    return ncclSuccess;
  };
  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 4096, 0, 0, 1,
                                             /*userAlgoInput=*/1, tbl.ptr()));
  EXPECT_EQ(int(ncclFuncReduceScatter), seenFunc);
  EXPECT_EQ(size_t(4096), seenBytes);
}

// --- per-call algSelection narrowing (AICOMRCCL-2728) -----------------------
// A registry mask bit IS its tuning id, so a general row sits at (a * NCCL_NUM_PROTOCOLS + p).
// The mask arrives as task->algMask, already parsed and validated upstream.

TEST_F(EnqueueMicrotest, UpdateCollCostTable_AlgSelectionKeepsOnlyTheNamedGeneralRows) {
  // Before the filter the mask was parsed, validated and discarded, so naming RING_LL left
  // every eligible cell a candidate.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  task.algMask = GeneralBit(NCCL_ALGO_RING, NCCL_PROTO_LL);
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/1,
                                             /*nvls=*/1, 1, 0, tbl.ptr()));
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_LL));
  EXPECT_EQ(1, tbl.countWritten()) << "only the named row may survive the filter";
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_AlgSelectionAcceptsAFamilyOfRows) {
  // Prefix matching yields several bits ("RING" -> every Ring protocol); all must survive. The NaN
  // cell is only filled under NCCL_PROTO=NaN, so the family here is the other three.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  task.algMask = GeneralBit(NCCL_ALGO_RING, NCCL_PROTO_LL) | GeneralBit(NCCL_ALGO_RING, NCCL_PROTO_LL128) |
                 GeneralBit(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, 1, 1, 1, 0, tbl.ptr()));
  for (int p : {NCCL_PROTO_LL, NCCL_PROTO_LL128, NCCL_PROTO_SIMPLE}) {
    EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, p)) << "p=" << p;
  }
  EXPECT_EQ(3, tbl.countWritten());
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_NoAlgSelection_LeavesEveryEligibleCell) {
  // An empty algMask is "automatic": the filter must not narrow the default path.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  task.algMask = rcclTuningMask();
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, 1, 1, 1, 0, tbl.ptr()));
  EXPECT_LT(1, tbl.countWritten()) << "automatic selection must leave the table wide";
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE));
  EXPECT_TRUE(tbl.written(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE));
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_SymmetricOnlySelection_LeavesGeneralTableIntact) {
  // A SYMK_*-only selection has no general bits, and the general table is what the symmetric
  // scheduler falls back TO. forceAlgSelection is 1, the nccl.h.in default, so the error must
  // stay inside the generalMask != 0 guard instead of firing on every SYMK_* selection.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  task.algMask = rcclTuningMask::bit(NCCL_TUNING_SYM_KERNEL_ID_OFFSET);
  task.forceAlgSelection = 1;
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, 1, 1, 1, 0, tbl.ptr()));
  EXPECT_LT(1, tbl.countWritten()) << "symmetric-only selection must not narrow the general table";
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE));
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_EnvForcedFunction_OverridesPerCallSelection) {
  // NCCL_ALGO/NCCL_PROTO/NCCL_SYM_KERNEL win over a per-call selection. The named row is
  // ineligible here (nvls=0) and forceAlgSelection is 1, so dropping the forced[] term would
  // blank the table and error; the return code is what proves the override.
  CostComm cc;
  cc.get()->tuningContext.forced[ncclFuncAllReduce] = 1;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  task.algMask = GeneralBit(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE);
  task.forceAlgSelection = 1;
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0,
                                             /*nvls=*/0, 1, 0, tbl.ptr()));
  EXPECT_LT(1, tbl.countWritten()) << "env-forced function must ignore algMask";
  EXPECT_TRUE(tbl.written(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE));
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_MixedGeneralAndSymmetricSelection_NarrowsOnlyTheGeneralHalf) {
  // The shape a real selection string makes: "RING_LL,SYMK_LL" on AllGather is one general bit
  // plus two symmetric ones, and the general half must narrow exactly as if they were absent.
  // The mask-AND itself is pinned by the symmetric-only case above, not here.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllGather);
  task.algMask = GeneralBit(NCCL_ALGO_RING, NCCL_PROTO_LL) |
                 rcclTuningMask::bit(NCCL_TUNING_SYM_KERNEL_ID_OFFSET + ncclSymkKernelId_AllGather_LL) |
                 rcclTuningMask::bit(NCCL_TUNING_SYM_KERNEL_ID_OFFSET + ncclSymkKernelId_AllGather_LLMC);
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, 1, 1, 1, 0, tbl.ptr()));
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_LL));
  EXPECT_EQ(1, tbl.countWritten()) << "the symmetric bits must neither widen nor blank the general table";
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_AlgSelectionCannotResurrectAnIneligibleRow) {
  // The filter only removes: a row the eligibility guards already skipped stays IGNOREd.
  // forceAlgSelection is 0 explicitly, not by CostTask default: production defaults it to 1,
  // and this is the one case whose outcome turns on it. comm.h documents 0 as "fall back to
  // automatic", so the table must be left wide rather than blanked down to nothing.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  task.algMask = GeneralBit(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE);
  task.forceAlgSelection = 0;
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0,
                                             /*nvls=*/0, 1, 0, tbl.ptr()));
  EXPECT_FALSE(tbl.written(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE));
  EXPECT_LT(1, tbl.countWritten()) << "an unsatisfiable selection with force off is automatic";
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE));
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_ZeroCostNamedRowCountsAsEligible) {
  // The eligibility rule must be `>= 0.0`, matching topoGetAlgoInfo's argmin. A zero cost is a
  // real candidate, so tightening this to `> 0.0` would error on a perfectly selectable row.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  task.algMask = GeneralBit(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  task.forceAlgSelection = 1;
  ScriptAllTimes(0.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, 1, 1, 1, 0, tbl.ptr()));
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE));
  EXPECT_EQ(1, tbl.countWritten());
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_PatSizeWindowExcludesPatWhenNothingNamesIt) {
  // The negative control for the case below: on the automatic path the window really does keep
  // PAT out at 64 MiB, so the lift that follows is a lift and not a no-op. nNodes=8 selects the
  // nNodes <= 8 branch, whose window is [32 KiB, 4 MiB], and 64 MiB sits outside it.
  CostComm cc(/*nRanks=*/8, /*nNodes=*/8);
  CostTable tbl;
  auto task = CostTask(ncclFuncAllGather);
  task.algMask = rcclTuningMask();
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 64u << 20, 1, 1, 1,
                                             /*userAlgoInput=*/0, tbl.ptr()));
  EXPECT_FALSE(tbl.written(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE));
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_NamingPatLiftsItsHeuristicSizeWindow) {
  // The window keeps the AUTO selector off PAT outside a good range, and NCCL_ALGO already lifts
  // it. A per-call selection is just as explicit, so it must lift it too, or one string succeeds
  // at 1 MiB and is dropped at 64 MiB on the same comm. forceAlgSelection is 0 so the EXPECT below
  // is what fails if the lift goes away, rather than the call erroring first.
  CostComm cc(/*nRanks=*/8, /*nNodes=*/8);
  CostTable tbl;
  auto task = CostTask(ncclFuncAllGather);
  task.algMask = GeneralBit(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE);
  task.forceAlgSelection = 0;
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 64u << 20, 1, 1, 1,
                                             /*userAlgoInput=*/0, tbl.ptr()));
  EXPECT_TRUE(tbl.written(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE));
  EXPECT_EQ(1, tbl.countWritten());
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_NamingLl128LiftsTheXgmiGate) {
  // The sibling of the PAT case, and the reason the lift is written per row rather than per
  // algorithm: NCCL_PROTO lifts the XGMI-only LL128 gate, so naming RING_LL128 per call must too.
  // Without it the only named row is blanked and the default forceAlgSelection fails the launch.
  CostComm cc;
  cc.topo().type = 0;  // not all-XGMI, so the LL128 gate would otherwise fire
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  task.algMask = GeneralBit(NCCL_ALGO_RING, NCCL_PROTO_LL128);
  task.forceAlgSelection = 1;
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, 1, 1, 1, 0, tbl.ptr()));
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_LL128));
  EXPECT_EQ(1, tbl.countWritten());
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_ScaledSentinelCellDoesNotCountAsEligible) {
  // The fp8 relegation scales RING cells by 1024, so a cell ncclTopoGetAlgoTime left at the -1.0
  // sentinel becomes -1024.0. Testing `!= NCCL_ALGO_PROTO_IGNORE` would read that as eligible and
  // swallow the error, while topoGetAlgoInfo's argmin still rejects it for being negative.
  CostComm cc(/*nRanks=*/16);
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce, ncclFloat8e4m3);
  task.algMask = GeneralBit(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  task.forceAlgSelection = 1;
  ScriptAllTimes(NCCL_ALGO_PROTO_IGNORE);

  EXPECT_EQ(ncclInvalidArgument, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0,
                                                     /*nvls=*/0, 1, 0, tbl.ptr()));
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_ForceAlgSelection_ErrorsWhenNothingNamedSurvives) {
  // nccl.h.in makes forceAlgSelection=1 an error on an unsatisfiable selection, but the
  // parse-time check only sees the registry; eligibility on THIS comm is known only here.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  task.algMask = GeneralBit(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE);
  task.forceAlgSelection = 1;
  ScriptAllTimes(1.0f);

  EXPECT_EQ(ncclInvalidArgument, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0,
                                                     /*nvls=*/0, 1, 0, tbl.ptr()));
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_ForceAlgSelection_StaysSilentWhenOneNamedRowSurvives) {
  // The error is only for "nothing left": two rows with one eligible is still satisfiable.
  CostComm cc;
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce);
  task.algMask = GeneralBit(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE) | GeneralBit(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  task.forceAlgSelection = 1;
  ScriptAllTimes(1.0f);

  ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0,
                                             /*nvls=*/0, 1, 0, tbl.ptr()));
  EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE));
  EXPECT_EQ(1, tbl.countWritten());
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_PatSizeWindowWidensWithNodeCount) {
  struct {
    int nNodes;
    size_t nBytes;
    bool written;
  } const cases[] = {{6, (32 << 10) - 1, false}, {6, 32 << 10, true},         {6, 4 << 20, true},
                     {6, (4 << 20) + 1, false},  {16, 32 << 20, true},        {16, (32 << 20) + 1, false}};
  for (const auto& c : cases) {
    CostComm cc(/*nRanks=*/8 * c.nNodes, c.nNodes);
    CostTable tbl;
    auto task = CostTask(ncclFuncAllGather);
    ScriptAllTimes(1.0f);
    ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, c.nBytes, 0, 0, 1, /*userAlgoInput=*/0, tbl.ptr()));
    EXPECT_EQ(c.written, tbl.written(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE)) << c.nNodes << " nodes, " << c.nBytes;
  }
}

// Under NCCL_PROTO=NaN a float collective with a NaN kernel is priced on its NaN cells alone, and one without
// keeps the default set.
TEST_F(EnqueueMicrotest, UpdateCollCostTable_NanForced_KeepsOnlyNanCells) {
  struct Case {
    ncclFunc_t func;
    ncclDataType_t datatype;
    bool ringNan, treeNan;
  };
  for (const Case& c : {Case{ncclFuncAllReduce, ncclFloat32, true, true}, Case{ncclFuncAllGather, ncclFloat32, true, false},
                        Case{ncclFuncReduceScatter, ncclFloat32, true, false},
                        Case{ncclFuncAllReduce, ncclInt32, false, false}}) {
    UnderNanProto([c] {
      CostComm cc;
      cc.get()->buffSizes[NCCL_PROTO_NAN] = 1 << 20;
      CostTable tbl;
      auto task = CostTask(c.func, c.datatype);
      ScriptAllTimes(1.0f);

      ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0, /*nvls=*/0, 1, 0, tbl.ptr()));
      EXPECT_EQ(c.ringNan, tbl.written(NCCL_ALGO_RING, NCCL_PROTO_NAN));
      EXPECT_EQ(c.treeNan, tbl.written(NCCL_ALGO_TREE, NCCL_PROTO_NAN));
      if (c.ringNan) {
        EXPECT_EQ(c.treeNan ? 2 : 1, tbl.countWritten());
      } else {
        EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE));
      }
    });
  }
}

namespace {
template <typename Body>
void UnderNanAuto(Body body) {
  const std::string style = ::testing::GTEST_FLAG(death_test_style);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  EXPECT_EXIT(
      {
        SetMicroEnv("RCCL_NAN_PROTO_AUTO", "1");
        SetMicroEnvAbsent("NCCL_PROTO");
        body();
        std::exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::GTEST_FLAG(death_test_style) = style;
}

// Two gfx950 nodes of 8, where RCCL_NAN_PROTO_AUTO picks NaN for a 1 MiB float AllReduce.
struct NanAutoCostComm : CostComm {
  char arch[8] = "gfx950";
  NanAutoCostComm() : CostComm(/*nRanks=*/16, /*nNodes=*/2) {
    get()->archName = arch;
    get()->buffSizes[NCCL_PROTO_NAN] = 1 << 20;
#if ENABLE_TDM_NAN
    get()->nanTdmMinBytes = -1;
#endif
  }
};
}  // namespace

// Under RCCL_NAN_PROTO_AUTO a per-call selection naming no NaN row is priced on exactly the rows it names, and
// with no selection AUTO still prices NaN alone.
TEST_F(EnqueueMicrotest, UpdateCollCostTable_NanAuto_PerCallSelectionWithoutNanKeepsItsRows) {
  UnderNanAuto([] {
    NanAutoCostComm cc;
    CostTable tbl;
    auto task = CostTask(ncclFuncAllReduce, ncclFloat32);
    task.algMask = GeneralBit(NCCL_ALGO_RING, NCCL_PROTO_LL128);
    task.forceAlgSelection = 1;
    ScriptAllTimes(1.0f);

    ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0, /*nvls=*/0, 1, 0, tbl.ptr()));
    EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_LL128));
    EXPECT_EQ(1, tbl.countWritten());
  });
  UnderNanAuto([] {
    NanAutoCostComm cc;
    CostTable tbl;
    auto task = CostTask(ncclFuncAllReduce, ncclFloat32);
    task.algMask = rcclTuningMask();
    ScriptAllTimes(1.0f);

    ASSERT_EQ(ncclSuccess, updateCollCostTable(cc.get(), &task, 1 << 20, /*collNet=*/0, /*nvls=*/0, 1, 0, tbl.ptr()));
    EXPECT_TRUE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_NAN));
    EXPECT_FALSE(tbl.written(NCCL_ALGO_RING, NCCL_PROTO_LL128));
  });
}
