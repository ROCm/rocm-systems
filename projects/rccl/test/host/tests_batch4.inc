// ===========================================================================
// rcclKernelPackedChannels (enqueue.cc:177) -- how many channels a single
// collective's traffic actually packs into. Reported to users via
// rccl_wrap.cc, so a wrong answer is user-visible, not just internal.
//
// Pure arithmetic over divUp/min/max with a 16 KiB floor. The oracle is the
// channel count; a return-code test would see none of it.
// ===========================================================================

namespace {
// MEASURED: sizeof(ncclComm) is ~3.8 MB. Two stack instances in one test is
// 7.6 MB against a default 8 MB stack, so these are heap-allocated -- the same
// reason CostComm/BatchPlanComm/FinishComm/RedOpComm do. RankComm owns the
// storage for the lifetime of the test.
class RankComm {
 public:
  explicit RankComm(int nRanks) : storage_(new ncclComm{}) { storage_->nRanks = nRanks; }
  ncclComm* get() { return storage_.get(); }
 private:
  std::unique_ptr<ncclComm> storage_;
};
}  // namespace

TEST_F(EnqueueMicrotest, PackedChannels_NonPositiveMaxChannels_ReturnedUnchanged) {
  // First early-out: `nMaxChannels <= 0`. Pass both 0 and negative -- the guard
  // is <=, and a mutant using < would still pass 0 without this.
  EXPECT_EQ(0, rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, 1024,
                                        ncclFloat32, NCCL_PROTO_SIMPLE, 0));
  EXPECT_EQ(-4, rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, 1024,
                                         ncclFloat32, NCCL_PROTO_SIMPLE, -4));
}

TEST_F(EnqueueMicrotest, PackedChannels_ZeroCount_ReturnsMaxChannels) {
  // Second half of the same early-out: `count == 0`.
  EXPECT_EQ(16, rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, 0,
                                         ncclFloat32, NCCL_PROTO_SIMPLE, 16));
}

TEST_F(EnqueueMicrotest, PackedChannels_TinyTransfer_PacksIntoOneChannel) {
  // Below the 16 KiB MinTrafficPerChannel floor everything fits in one channel,
  // however many are offered.
  EXPECT_EQ(1, rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, 4,
                                        ncclFloat32, NCCL_PROTO_SIMPLE, 32));
}

TEST_F(EnqueueMicrotest, PackedChannels_LargeTransfer_SaturatesToMaxChannels) {
  // Far above the floor, the divUp(cells, cellsPerChannel) term exceeds
  // nMaxChannels and the final min() clamps it.
  EXPECT_EQ(8, rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce,
                                        size_t(1) << 24, ncclFloat32,
                                        NCCL_PROTO_SIMPLE, 8));
}

TEST_F(EnqueueMicrotest, PackedChannels_NeverExceedsMaxChannels) {
  // Invariant sweep: the return is min()-clamped, so no input may break it.
  for (size_t count : {size_t(1), size_t(1) << 10, size_t(1) << 20, size_t(1) << 28}) {
    for (int maxCh : {1, 2, 8, 32}) {
      int got = rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, count,
                                         ncclFloat32, NCCL_PROTO_SIMPLE, maxCh);
      EXPECT_GE(got, 1) << "count=" << count << " max=" << maxCh;
      EXPECT_LE(got, maxCh) << "count=" << count << " max=" << maxCh;
    }
  }
}

TEST_F(EnqueueMicrotest, PackedChannels_IsMonotonicInCount) {
  // More data must never need FEWER channels. Catches an inverted divUp.
  int prev = 0;
  for (size_t count : {size_t(1) << 8, size_t(1) << 12, size_t(1) << 16,
                       size_t(1) << 20, size_t(1) << 24}) {
    int got = rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, count,
                                       ncclFloat32, NCCL_PROTO_SIMPLE, 32);
    EXPECT_GE(got, prev) << "count=" << count;
    prev = got;
  }
}

TEST_F(EnqueueMicrotest, PackedChannels_LLProtocolQuadruplesTraffic) {
  // `if (protocol == NCCL_PROTO_LL) trafficPerByte *= 4` -- LL moves 4x the
  // bytes, so at a size where SIMPLE has not yet saturated, LL needs >= as many
  // channels. Differential, since the absolute value is not the interesting part.
  const size_t count = 1 << 12;
  int simple = rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, count,
                                        ncclFloat32, NCCL_PROTO_SIMPLE, 32);
  int ll = rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, count,
                                    ncclFloat32, NCCL_PROTO_LL, 32);
  EXPECT_GE(ll, simple);
}

TEST_F(EnqueueMicrotest, PackedChannels_AllGatherScalesWithRankCount) {
  // trafficPerByte for AllGather is nRanks, so more ranks means more traffic and
  // (below saturation) more channels. Pins that comm->nRanks is actually read.
  const size_t count = 1 << 10;
  int few = rcclKernelPackedChannels(RankComm(2).get(), ncclFuncAllGather, count,
                                     ncclFloat32, NCCL_PROTO_SIMPLE, 32);
  int many = rcclKernelPackedChannels(RankComm(64).get(), ncclFuncAllGather, count,
                                      ncclFloat32, NCCL_PROTO_SIMPLE, 32);
  // STRICTLY greater: with >=, an implementation that ignored comm->nRanks
  // altogether would return the same count for both and still pass.
  EXPECT_GT(many, few) << "few=" << few << " many=" << many;
}

TEST_F(EnqueueMicrotest, PackedChannels_ElementSizeScalesTraffic) {
  // count is in ELEMENTS; bytes = count * ncclTypeSize(datatype). Same element
  // count in float64 is twice the bytes of float32.
  const size_t count = 1 << 11;
  int f32 = rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, count,
                                     ncclFloat32, NCCL_PROTO_SIMPLE, 32);
  int f64 = rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, count,
                                     ncclFloat64, NCCL_PROTO_SIMPLE, 32);
  // STRICTLY greater: with >=, ignoring ncclTypeSize(datatype) would pass.
  EXPECT_GT(f64, f32) << "f32=" << f32 << " f64=" << f64;
}

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
  for (auto& row : table) for (auto& v : row) v = -1.0f;

  initCollCostTable((float**)table);

  for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
      EXPECT_FLOAT_EQ(NCCL_ALGO_PROTO_IGNORE, table[a][p]) << "a=" << a << " p=" << p;
    }
  }
}

TEST_F(EnqueueMicrotest, InitCollCostTable_DoesNotWritePastTheTable) {
  // Guard rows on both sides catch a wrong stride in the (float(*)[N]) cast.
  struct { float before[NCCL_NUM_PROTOCOLS];
           float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
           float after[NCCL_NUM_PROTOCOLS]; } buf;
  for (auto& v : buf.before) v = 42.0f;
  for (auto& v : buf.after) v = 42.0f;
  for (auto& row : buf.table) for (auto& v : row) v = -1.0f;

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

namespace {
// LOAD-BEARING, twice over:
//  1. The protocol loop at :2534 reads `comm->topo->type` for the LL128 XGMI
//     gate, so comm->topo may not be null -- a zero-initialised ncclComm
//     segfaults there rather than failing an assertion.
//  2. ncclTopoSystem is far too large to hold by value in a test fixture: a
//     stack instance overflows and crashes inside the CONSTRUCTOR, before any
//     production code runs. Both it and ncclComm are heap-allocated here.
struct CostComm {
  std::unique_ptr<ncclTopoSystem> topoStorage{new ncclTopoSystem{}};
  std::unique_ptr<ncclComm> commStorage{new ncclComm{}};
  CostComm(int nRanks = 8, int nNodes = 1) {
    ncclComm& comm = *commStorage;
    comm.nRanks = nRanks;
    comm.nNodes = nNodes;
    comm.maxLocalRanks = 8;
    comm.localRanks = 8;
    // XGMI_ALL set: the LL128 gate at :2535 passes, so LL128 cells are filled
    // like every other protocol. Tests that care about the gate clear it.
    topoStorage->type = RCCL_TOPO_XGMI_ALL;
    comm.topo = topoStorage.get();
  }
  ncclComm* get() { return commStorage.get(); }
  ncclTopoSystem& topo() { return *topoStorage; }
};

// A table wrapper that starts fully IGNOREd, like production's initCollCostTable.
struct CostTable {
  float t[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  CostTable() { initCollCostTable((float**)t); }
  float** ptr() { return (float**)t; }
  bool written(int a, int p) const { return t[a][p] != NCCL_ALGO_PROTO_IGNORE; }
  int countWritten() const {
    int n = 0;
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a)
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p)
        if (t[a][p] != NCCL_ALGO_PROTO_IGNORE) ++n;
    return n;
  }
};

ncclTaskColl CostTask(ncclFunc_t f, ncclDataType_t dt = ncclFloat32,
                      ncclDevRedOp_t devOp = ncclDevSum) {
  ncclTaskColl t{};
  t.func = f;
  t.datatype = dt;
  t.opDev.op = devOp;
  return t;
}

// Scripts every (algo, proto) pair to a fixed cost, so "was this cell reached?"
// is answerable.
void ScriptAllTimes(float cost) {
  g_topoGetAlgoTime = [cost](struct ncclComm*, int, int, int, size_t, int, float* time) {
    if (time) *time = cost;
    return ncclSuccess;
  };
}
}  // namespace

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
    ResetEnqueueFakes();
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
  ResetEnqueueFakes();
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

TEST_F(EnqueueMicrotest, UpdateCollCostTable_Fp8RingAboveEightRanks_IsPenalised) {
  // The fp8 precision-loss penalty: RING costs are multiplied by 1024 when
  // nRanks > 8 and the datatype is fp8. Assert the RATIO, so the penalty factor
  // itself is pinned rather than just "bigger".
  for (auto dt : {ncclFloat8e4m3, ncclFloat8e5m2}) {
    ResetEnqueueFakes();
    CostComm big(/*nRanks=*/16);
    CostTable tbl;
    auto task = CostTask(ncclFuncAllReduce, dt);
    ScriptAllTimes(2.0f);
    ASSERT_EQ(ncclSuccess, updateCollCostTable(big.get(), &task, 1 << 20, 0, 0, 1, 0, tbl.ptr()));
    EXPECT_FLOAT_EQ(2.0f * 1024.0f, tbl.t[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE])
        << "dtype=" << int(dt);
    // TREE is not penalised -- that asymmetry is the point of the penalty.
    EXPECT_FLOAT_EQ(2.0f, tbl.t[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE]) << "dtype=" << int(dt);
  }
}

TEST_F(EnqueueMicrotest, UpdateCollCostTable_Fp8AtOrBelowEightRanks_NotPenalised) {
  // The `nRanks > 8` half of the guard. 8 exactly must NOT be penalised.
  CostComm eight(/*nRanks=*/8);
  CostTable tbl;
  auto task = CostTask(ncclFuncAllReduce, ncclFloat8e4m3);
  ScriptAllTimes(2.0f);
  ASSERT_EQ(ncclSuccess, updateCollCostTable(eight.get(), &task, 1 << 20, 0, 0, 1, 0, tbl.ptr()));
  EXPECT_FLOAT_EQ(2.0f, tbl.t[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
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
