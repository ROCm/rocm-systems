// ===========================================================================
// calcCollChunking (enqueue.cc:2871) -- the largest host-reachable block in the
// file (278 lines, ~89 branch points) and entirely HIP-free.
//
// Two separable concerns:
//   1. Pattern selection   (:2876-2916) -- a 9-arm switch over func x algorithm.
//   2. Chunk-size ladders  (:2930-3060) -- nine halving loops with 262144 /
//      131072 / 65536 / 32768 floors, gated by algorithm+protocol.
//
// The oracle is the COMPUTED chunk size and the proxyOp's pattern -- never a
// return code. A test that only checked ncclSuccess would survive almost any
// mutation in here.
// ===========================================================================

namespace {
// Minimal comm the chunking maths reads. Everything here is a plain field read;
// no HIP, no allocation, no topology.
struct ChunkComm {
  // MEASURED: sizeof(ncclComm) is ~3.8 MB -- far too large for a stack fixture.
  std::unique_ptr<ncclComm> storage{new ncclComm{}};
  ncclComm& comm{*storage};
  explicit ChunkComm(int protoSimpleBuf = 1 << 22) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) comm.buffSizes[p] = protoSimpleBuf;
    // LOAD-BEARING: rcclProtoGrainSize(LL128) (scheduler.h:22) is
    //   WarpSize * ELEMS_PER_THREAD * ll128DataElems * 8 / ll128LineElems
    // so a zero WarpSize makes grainSize 0, and :3029's
    // `chunkSize / grainSize * grainSize` then SIGFPEs. A zero-initialised
    // ncclComm is not a usable fixture for any LL128 path.
    comm.WarpSize = 32;
    comm.nRanks = 8;
    comm.nNodes = 1;
    comm.nvlsChunkSize = 128 * 1024;
    comm.nvlsTreeMaxChunkSize = 128 * 1024;
    comm.ll128LineElems = 120;
    comm.ll128DataElems = 112;
    comm.channels[0].tree.depth = 4;
    comm.channels[0].collnetDirect.depth = 4;
    comm.channels[0].collnetDirect.nHeads = 1;
    comm.channels[0].collnetChain.depth = 4;
    comm.channels[0].nvls.nHeads = 1;
  }
  ncclComm* get() { return storage.get(); }
};

ncclTaskColl MakeTask(ncclFunc_t func, int algo, int proto) {
  ncclTaskColl t{};
  t.func = func;
  t.algorithm = algo;
  t.protocol = proto;
  t.chunkSteps = 1;
  t.sliceSteps = 1;
  t.datatype = ncclFloat32;
  return t;
}

// Runs calcCollChunking and hands back the chunk size + the proxyOp it filled.
struct ChunkResult {
  ncclResult_t rc;
  uint32_t chunkSize;
  uint32_t directFlags;
  ncclProxyOp proxyOp;
};

ChunkResult RunChunking(ChunkComm& cc, ncclTaskColl& task, int nChannels, size_t nBytes) {
  ChunkResult r{};
  r.chunkSize = 0xFFFFFFFFu;  // poison: "not written" must be distinguishable
  r.rc = calcCollChunking(cc.get(), &task, nChannels, nBytes,
                          &r.chunkSize, &r.directFlags, &r.proxyOp);
  return r;
}
}  // namespace

// ---------------------------------------------------------------------------
// 1. Pattern selection. Each arm maps (func, algorithm) -> ncclPattern_t, and
//    the pattern lands in proxyOp.pattern. Asserting the pattern (not the return
//    code) is what makes a swapped ternary arm visible.
// ---------------------------------------------------------------------------

TEST_F(EnqueueMicrotest, CalcCollChunking_Broadcast_TreeVsRing_SelectsDistinctPatterns) {
  ChunkComm cc;
  auto tree = MakeTask(ncclFuncBroadcast, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  auto ring = MakeTask(ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  auto rt = RunChunking(cc, tree, 4, 1 << 20);
  auto rr = RunChunking(cc, ring, 4, 1 << 20);
  ASSERT_EQ(ncclSuccess, rt.rc);
  ASSERT_EQ(ncclSuccess, rr.rc);
  EXPECT_EQ(ncclPatternTreeDown, rt.proxyOp.pattern);
  EXPECT_EQ(ncclPatternPipelineFrom, rr.proxyOp.pattern);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_Reduce_TreeVsRing_SelectsDistinctPatterns) {
  ChunkComm cc;
  auto tree = MakeTask(ncclFuncReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  auto ring = MakeTask(ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  auto rt = RunChunking(cc, tree, 4, 1 << 20);
  auto rr = RunChunking(cc, ring, 4, 1 << 20);
  EXPECT_EQ(ncclPatternTreeUp, rt.proxyOp.pattern);
  EXPECT_EQ(ncclPatternPipelineTo, rr.proxyOp.pattern);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_AllGather_EachAlgorithmSelectsItsPattern) {
  // A 4-way ternary chain: PAT / NVLS / COLLNET_DIRECT / else RING. Sweeping all
  // four pins the ORDER as well as the mapping.
  ChunkComm cc;
  const struct { int algo; ncclPattern_t want; } kCases[] = {
      {NCCL_ALGO_PAT, ncclPatternPatDown},
      {NCCL_ALGO_NVLS, ncclPatternNvls},
      {NCCL_ALGO_COLLNET_DIRECT, ncclPatternCollnetDirect},
      {NCCL_ALGO_RING, ncclPatternRing}};
  for (auto c : kCases) {
    auto t = MakeTask(ncclFuncAllGather, c.algo, NCCL_PROTO_SIMPLE);
    auto r = RunChunking(cc, t, 4, 1 << 20);
    ASSERT_EQ(ncclSuccess, r.rc) << "algo=" << c.algo;
    EXPECT_EQ(c.want, r.proxyOp.pattern) << "algo=" << c.algo;
  }
}

TEST_F(EnqueueMicrotest, CalcCollChunking_ReduceScatter_EachAlgorithmSelectsItsPattern) {
  ChunkComm cc;
  const struct { int algo; ncclPattern_t want; } kCases[] = {
      {NCCL_ALGO_PAT, ncclPatternPatUp},
      {NCCL_ALGO_NVLS, ncclPatternNvls},
      {NCCL_ALGO_COLLNET_DIRECT, ncclPatternCollnetDirect},
      {NCCL_ALGO_RING, ncclPatternRing}};
  for (auto c : kCases) {
    auto t = MakeTask(ncclFuncReduceScatter, c.algo, NCCL_PROTO_SIMPLE);
    auto r = RunChunking(cc, t, 4, 1 << 20);
    ASSERT_EQ(ncclSuccess, r.rc) << "algo=" << c.algo;
    EXPECT_EQ(c.want, r.proxyOp.pattern) << "algo=" << c.algo;
  }
}

TEST_F(EnqueueMicrotest, CalcCollChunking_AllGatherAndReduceScatterPatDiffer) {
  // PatUp vs PatDown is the one place these two funcs must NOT agree; a
  // copy-paste between the two arms is otherwise invisible.
  ChunkComm cc;
  auto ag = MakeTask(ncclFuncAllGather, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE);
  auto rs = MakeTask(ncclFuncReduceScatter, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE);
  EXPECT_NE(RunChunking(cc, ag, 4, 1 << 20).proxyOp.pattern,
            RunChunking(cc, rs, 4, 1 << 20).proxyOp.pattern);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_AllReduce_EachAlgorithmSelectsItsPattern) {
  // The longest chain: 6 arms.
  ChunkComm cc;
  const struct { int algo; ncclPattern_t want; } kCases[] = {
      {NCCL_ALGO_NVLS, ncclPatternNvls},
      {NCCL_ALGO_NVLS_TREE, ncclPatternNvlsTree},
      {NCCL_ALGO_COLLNET_DIRECT, ncclPatternCollnetDirect},
      {NCCL_ALGO_COLLNET_CHAIN, ncclPatternCollnetChain},
      {NCCL_ALGO_TREE, ncclPatternTreeUpDown},
      {NCCL_ALGO_RING, ncclPatternRingTwice}};
  for (auto c : kCases) {
    auto t = MakeTask(ncclFuncAllReduce, c.algo, NCCL_PROTO_SIMPLE);
    auto r = RunChunking(cc, t, 4, 1 << 20);
    ASSERT_EQ(ncclSuccess, r.rc) << "algo=" << c.algo;
    EXPECT_EQ(c.want, r.proxyOp.pattern) << "algo=" << c.algo;
  }
}

TEST_F(EnqueueMicrotest, CalcCollChunking_AlltoAllVariants_AlwaysRingPattern) {
  // Three funcs share an unconditional ncclPatternRing, independent of algorithm.
  ChunkComm cc;
  for (auto f : {ncclFuncAlltoAllPivot, ncclFuncAlltoAllGda, ncclFuncAlltoAllvGda}) {
    for (int algo : {NCCL_ALGO_RING, NCCL_ALGO_TREE, NCCL_ALGO_NVLS}) {
      auto t = MakeTask(f, algo, NCCL_PROTO_SIMPLE);
      auto r = RunChunking(cc, t, 4, 1 << 20);
      ASSERT_EQ(ncclSuccess, r.rc) << "func=" << int(f) << " algo=" << algo;
      EXPECT_EQ(ncclPatternRing, r.proxyOp.pattern) << "func=" << int(f) << " algo=" << algo;
    }
  }
}

TEST_F(EnqueueMicrotest, CalcCollChunking_UnknownFunc_ReturnsInternalErrorAndWarns) {
  // The switch's `default:` arm -- an error return AND a WARN. Assert both: the
  // return code alone would not notice a deleted log line.
  ChunkComm cc;
  auto t = MakeTask(static_cast<ncclFunc_t>(0x5F), NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  ncclResult_t rc = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    uint32_t cs = 0, df = 0;
    ncclProxyOp op{};
    rc = calcCollChunking(cc.get(), &t, 4, 1 << 20, &cs, &df, &op);
  });
  EXPECT_EQ(ncclInternalError, rc);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "Unknown pattern for collective"))
      << "actual log:\n" << log;
}

// ---------------------------------------------------------------------------
// 2. Chunk-size computation. stepSize = buffSizes[proto]/NCCL_STEPS, then
//    protocol adjustments, then algorithm-specific halving ladders.
// ---------------------------------------------------------------------------

TEST_F(EnqueueMicrotest, CalcCollChunking_SimpleProtocol_ChunkIsStepSizeTimesChunkSteps) {
  // Baseline with no ladder: AllGather+RING+SIMPLE has no halving loop, so the
  // result is exactly stepSize*chunkSteps. Pins the base formula.
  ChunkComm cc(/*protoSimpleBuf=*/1 << 22);
  auto t = MakeTask(ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  auto r = RunChunking(cc, t, 4, 1 << 24);
  ASSERT_EQ(ncclSuccess, r.rc);
  EXPECT_EQ(uint32_t((1 << 22) / NCCL_STEPS), r.chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_LLProtocol_HalvesTheChunk) {
  // :2925 `if (protocol == NCCL_PROTO_LL) chunkSize /= 2`.
  ChunkComm cc(1 << 22);
  auto simple = MakeTask(ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  auto ll = MakeTask(ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL);
  auto rs = RunChunking(cc, simple, 4, 1 << 24);
  auto rl = RunChunking(cc, ll, 4, 1 << 24);
  EXPECT_EQ(rs.chunkSize / 2, rl.chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_LL128Protocol_ScalesByLineAndDataElemsThenAlignsToGrain) {
  // TWO steps, and the second is easy to forget:
  //   :2926 chunkSize = (chunkSize / ll128LineElems) * ll128DataElems
  //   :3029 chunkSize = chunkSize / grainSize * grainSize   (align DOWN)
  // Modelling only the first gives 489328; the real answer is 489216. Both are
  // asserted so dropping either step fails.
  ChunkComm cc(1 << 22);
  auto t = MakeTask(ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL128);
  auto r = RunChunking(cc, t, 4, 1 << 24);
  ASSERT_EQ(ncclSuccess, r.rc);

  const int base = (1 << 22) / NCCL_STEPS;
  const uint32_t scaled = uint32_t((base / 120) * 112);
  const size_t grain = rcclProtoGrainSize(NCCL_PROTO_LL128, cc.get());
  ASSERT_GT(grain, 0u);
  EXPECT_EQ(uint32_t(scaled / grain * grain), r.chunkSize);
  EXPECT_LE(r.chunkSize, scaled) << "grain alignment only rounds DOWN";
}

TEST_F(EnqueueMicrotest, CalcCollChunking_RingSimple_ChunkStepsMultiplyTheChunk) {
  // chunkSteps is honoured ONLY for (SIMPLE, RING) -- :2923. Two tasks that
  // differ solely in chunkSteps must produce different chunk sizes here.
  ChunkComm cc(1 << 22);
  auto one = MakeTask(ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  auto two = MakeTask(ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  two.chunkSteps = 2;
  EXPECT_EQ(RunChunking(cc, one, 4, 1 << 24).chunkSize * 2,
            RunChunking(cc, two, 4, 1 << 24).chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_NonRingSimple_IgnoresChunkSteps) {
  // The other half of the same gate: with TREE, chunkSteps is forced to 1, so
  // raising it must change nothing. Differential against the test above.
  ChunkComm cc(1 << 22);
  auto one = MakeTask(ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  auto two = MakeTask(ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  two.chunkSteps = 8;
  EXPECT_EQ(RunChunking(cc, one, 4, 1 << 30).chunkSize,
            RunChunking(cc, two, 4, 1 << 30).chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_TreeSimple_SmallBytesHalveDownToFloor) {
  // The TreeUpDown ladder (:2932-2936). With tiny nBytes every guard fires, so
  // the chunk walks down to the 32768 floor and stops -- never below it.
  ChunkComm cc(1 << 22);
  auto t = MakeTask(ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  auto r = RunChunking(cc, t, /*nChannels=*/4, /*nBytes=*/1024);
  ASSERT_EQ(ncclSuccess, r.rc);
  EXPECT_EQ(32768u, r.chunkSize) << "the last ladder floor is 32768";
}

TEST_F(EnqueueMicrotest, CalcCollChunking_TreeSimple_LargeBytesSkipTheLadder) {
  // Differential: with nBytes huge, no guard fires and the chunk stays at the
  // unhalved base. Together with the test above this pins the ladder CONDITION,
  // not just its floor.
  ChunkComm cc(1 << 22);
  auto t = MakeTask(ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  auto r = RunChunking(cc, t, /*nChannels=*/4, /*nBytes=*/size_t(1) << 40);
  EXPECT_EQ(uint32_t((1 << 22) / NCCL_STEPS), r.chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_TreeSimple_ChunkIsMonotonicInBytes) {
  // The ladder must never grow the chunk as nBytes shrinks. Sweeping sizes
  // catches an inverted comparison that a single-point test would miss.
  ChunkComm cc(1 << 22);
  auto t = MakeTask(ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  uint32_t prev = 0;
  for (size_t n : {size_t(1) << 12, size_t(1) << 18, size_t(1) << 24,
                   size_t(1) << 30, size_t(1) << 36}) {
    uint32_t cur = RunChunking(cc, t, 4, n).chunkSize;
    EXPECT_GE(cur, prev) << "nBytes=" << n << " must not shrink the chunk";
    prev = cur;
  }
}

TEST_F(EnqueueMicrotest, CalcCollChunking_TreeLadderRespondsToTreeDepth) {
  // Every guard in the TreeUpDown ladder multiplies by tree.depth, so a deeper
  // tree halves more aggressively at the same nBytes.
  ChunkComm shallow(1 << 22);
  shallow.comm.channels[0].tree.depth = 1;
  ChunkComm deep(1 << 22);
  deep.comm.channels[0].tree.depth = 16;
  auto t = MakeTask(ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  EXPECT_GE(RunChunking(shallow, t, 4, 1 << 22).chunkSize,
            RunChunking(deep, t, 4, 1 << 22).chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_PipelineRingSimple_LadderFloorsAt32768) {
  // The Broadcast/Reduce pipeline ladder (:2939-2943) -- distinct from the tree
  // one, with constant (not depth-scaled) thresholds 64/32/16/8.
  ChunkComm cc(1 << 22);
  auto t = MakeTask(ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  auto r = RunChunking(cc, t, 4, 1024);
  ASSERT_EQ(ncclSuccess, r.rc);
  EXPECT_EQ(ncclPatternPipelineFrom, r.proxyOp.pattern);
  EXPECT_EQ(32768u, r.chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_PipelineLadder_LargeBytesStayUnhalved) {
  ChunkComm cc(1 << 22);
  auto t = MakeTask(ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  auto r = RunChunking(cc, t, 4, size_t(1) << 40);
  EXPECT_EQ(uint32_t((1 << 22) / NCCL_STEPS), r.chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_CollnetChain_ClampsToTwoFiftySixK) {
  // :2960 chunkSize = min(256*1024, stepSize*chunkSteps). With a large buffer
  // the 256K clamp is what decides, so this pins the literal.
  ChunkComm cc(/*protoSimpleBuf=*/1 << 26);  // stepSize alone far exceeds 256K
  auto t = MakeTask(ncclFuncAllReduce, NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_SIMPLE);
  auto r = RunChunking(cc, t, 4, size_t(1) << 40);  // huge: no ladder halving
  ASSERT_EQ(ncclSuccess, r.rc);
  EXPECT_EQ(256u * 1024u, r.chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_NvlsTree_ClampedByNvlsChunkSizes) {
  // :2988 chunkSize = min(nvlsChunkSize, nvlsTreeMaxChunkSize). Make them differ
  // so the min() direction is pinned rather than assumed.
  ChunkComm cc(1 << 22);
  cc.comm.nvlsChunkSize = 128 * 1024;
  cc.comm.nvlsTreeMaxChunkSize = 64 * 1024;
  auto t = MakeTask(ncclFuncAllReduce, NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE);
  auto r = RunChunking(cc, t, 4, size_t(1) << 40);
  ASSERT_EQ(ncclSuccess, r.rc);
  EXPECT_EQ(64u * 1024u, r.chunkSize) << "the SMALLER of the two must win";
}

TEST_F(EnqueueMicrotest, CalcCollChunking_Nvls_ClampedByNvlsChunkSize) {
  // The non-registered NVLS arm caps at comm->nvlsChunkSize.
  ChunkComm cc(1 << 26);
  cc.comm.nvlsChunkSize = 32 * 1024;
  auto t = MakeTask(ncclFuncAllReduce, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE);
  auto r = RunChunking(cc, t, 4, size_t(1) << 40);
  ASSERT_EQ(ncclSuccess, r.rc);
  EXPECT_EQ(32u * 1024u, r.chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_NvlsMultiNodeLowBandwidth_CapsAt32768) {
  // :2974 -- with nNodes>1 AND NVLS allreduce bandwidth < 150, the cap drops to
  // 32768 regardless of nvlsChunkSize. Both conditions must hold, so the
  // single-node control below is what makes this test meaningful.
  ChunkComm cc(1 << 26);
  cc.comm.nvlsChunkSize = 1 << 20;
  cc.comm.nNodes = 2;
  cc.comm.bandwidths[ncclFuncAllReduce][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = 100.0f;
  auto t = MakeTask(ncclFuncAllReduce, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE);
  EXPECT_EQ(32768u, RunChunking(cc, t, 4, size_t(1) << 40).chunkSize);

  // Control: same bandwidth, single node -> the cap does NOT apply.
  ChunkComm one(1 << 26);
  one.comm.nvlsChunkSize = 1 << 20;
  one.comm.nNodes = 1;
  one.comm.bandwidths[ncclFuncAllReduce][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = 100.0f;
  EXPECT_EQ(uint32_t(1 << 20), RunChunking(one, t, 4, size_t(1) << 40).chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_NvlsHighBandwidthMultiNode_KeepsFullChunk) {
  // The other side of the && : bandwidth >= 150 leaves the cap alone even on
  // multiple nodes. Pins that 150 is a threshold, not a formality.
  ChunkComm cc(1 << 26);
  cc.comm.nvlsChunkSize = 1 << 20;
  cc.comm.nNodes = 2;
  cc.comm.bandwidths[ncclFuncAllReduce][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = 200.0f;
  auto t = MakeTask(ncclFuncAllReduce, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE);
  EXPECT_EQ(uint32_t(1 << 20), RunChunking(cc, t, 4, size_t(1) << 40).chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_ChunkIsAlignedDownToProtocolGrain) {
  // :3029 `chunkSize = chunkSize / grainSize * grainSize` -- an align-DOWN to the
  // protocol's grain (512 for SIMPLE, 16 for LL). Assert divisibility rather than
  // a literal, so the test survives a grain change but not a dropped alignment.
  ChunkComm cc(1 << 22);
  for (int proto : {NCCL_PROTO_SIMPLE, NCCL_PROTO_LL, NCCL_PROTO_LL128}) {
    const size_t grain = rcclProtoGrainSize(proto, cc.get());
    ASSERT_GT(grain, 0u) << "proto=" << proto << " grain must be positive";
    auto t = MakeTask(ncclFuncAllGather, NCCL_ALGO_RING, proto);
    auto r = RunChunking(cc, t, 4, 1 << 24);
    ASSERT_EQ(ncclSuccess, r.rc) << "proto=" << proto;
    EXPECT_EQ(0u, r.chunkSize % grain)
        << "proto=" << proto << " chunk=" << r.chunkSize << " grain=" << grain;
  }
}

TEST_F(EnqueueMicrotest, CalcCollChunking_Ll128GrainDependsOnWarpSizeAndLineElems) {
  // Guards the fixture invariant above as an executable assertion: if WarpSize or
  // the ll128 elem counts are ever left unset, grainSize goes to 0 and :3029
  // divides by zero. This fails loudly instead of core-dumping the suite.
  ChunkComm cc;
  EXPECT_GT(rcclProtoGrainSize(NCCL_PROTO_LL128, cc.get()), 0)
      << "LL128 grain is WarpSize*8*ll128DataElems*8/ll128LineElems -- a zero "
         "WarpSize or ll128DataElems makes calcCollChunking:3029 SIGFPE";
}

TEST_F(EnqueueMicrotest, CalcCollChunking_ChunkSizeIsAlwaysWritten) {
  // The out-param is poisoned to 0xFFFFFFFF before every call; a path that
  // returns success without writing it would show up as the poison value.
  ChunkComm cc;
  const struct { ncclFunc_t f; int algo; int proto; } kCases[] = {
      {ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE},
      {ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL},
      {ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL128},
      {ncclFuncReduceScatter, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE},
      {ncclFuncBroadcast, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE},
      {ncclFuncReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE}};
  for (auto c : kCases) {
    auto t = MakeTask(c.f, c.algo, c.proto);
    auto r = RunChunking(cc, t, 4, 1 << 20);
    ASSERT_EQ(ncclSuccess, r.rc) << "func=" << int(c.f) << " algo=" << c.algo;
    EXPECT_NE(0xFFFFFFFFu, r.chunkSize) << "func=" << int(c.f) << " algo=" << c.algo;
    EXPECT_GT(r.chunkSize, 0u) << "func=" << int(c.f) << " algo=" << c.algo;
  }
}
