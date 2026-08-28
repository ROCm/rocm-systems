/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ===========================================================================
// ncclFuncTrafficPerByte (enqueue.cc:158) -- traffic multiplier per collective.
// Pure: no externals. The three nRanks-returning arms are indistinguishable
// unless each is asserted separately, so a swapped case label would survive a
// test that only checked "some arm returns nRanks".
// ===========================================================================

TEST_F(EnqueueMicrotest, FuncTrafficPerByte_AllReduce_IsTwoRegardlessOfRanks) {
  // 2 is a literal, not a function of nRanks: sweep ranks to pin that.
  for (int n : {1, 2, 8, 64}) {
    EXPECT_EQ(2, ncclFuncTrafficPerByte(ncclFuncAllReduce, n)) << "nRanks=" << n;
  }
}

TEST_F(EnqueueMicrotest, FuncTrafficPerByte_AllGather_IsNRanks) {
  EXPECT_EQ(8, ncclFuncTrafficPerByte(ncclFuncAllGather, 8));
  EXPECT_EQ(3, ncclFuncTrafficPerByte(ncclFuncAllGather, 3));  // distinct from 8: catches a pinned constant
}

TEST_F(EnqueueMicrotest, FuncTrafficPerByte_ReduceScatter_IsNRanks) {
  EXPECT_EQ(8, ncclFuncTrafficPerByte(ncclFuncReduceScatter, 8));
  EXPECT_EQ(3, ncclFuncTrafficPerByte(ncclFuncReduceScatter, 3));
}

TEST_F(EnqueueMicrotest, FuncTrafficPerByte_EveryFuncOutsideTheNamedArmsIsOne) {
  // Sweep the WHOLE enum rather than a hand-picked few. A short list left
  // ncclFuncAlltoAllvGda untested, so deleting `case ncclFuncAlltoAllvGda:
  // return nRanks;` kept every test green. Sweeping also stays closed when a new
  // ncclFunc_t is added: the new value lands in the default arm and is checked.
  const int kRanks = 8;
  for (int i = 0; i < int(ncclNumFuncs); ++i) {
    const auto f = static_cast<ncclFunc_t>(i);
    const bool namedArm = (f == ncclFuncAllReduce) || (f == ncclFuncAllGather) ||
                          (f == ncclFuncReduceScatter) || (f == ncclFuncAlltoAllvGda);
    if (namedArm) continue;
    EXPECT_EQ(1, ncclFuncTrafficPerByte(f, kRanks)) << "func=" << i;
  }
}

TEST_F(EnqueueMicrotest, FuncTrafficPerByte_AlltoAllvGda_IsNRanks) {
  // The arm the sweep above deliberately excludes -- and the one a short default
  // list missed entirely. Two rank counts so a pinned constant cannot pass.
  EXPECT_EQ(8, ncclFuncTrafficPerByte(ncclFuncAlltoAllvGda, 8));
  EXPECT_EQ(3, ncclFuncTrafficPerByte(ncclFuncAlltoAllvGda, 3));
}

// ===========================================================================
// ncclTestBudget (enqueue.cc:406) -- does a plan still fit its arg budget?
// Pure. Two OR'd predicates:
//   A: batchBytes + workBytes <= inArgsBytes            (everything inline)
//   B: batchBytes <= inArgsBytes && workBytes <= outArgs (work spilled out)
// The full truth table is 4 cases; assert each so neither disjunct can be
// deleted without a failure.
// ===========================================================================

namespace {
constexpr ssize_t kBatch = sizeof(struct ncclDevWorkBatch);
ncclKernelPlanBudget MakeBudget(ssize_t in, ssize_t out) {
  ncclKernelPlanBudget b{};
  b.inArgsBytes = in;
  b.outArgsBytes = out;
  return b;
}
}  // namespace

TEST_F(EnqueueMicrotest, TestBudget_BothFitInline_ArmAOnly) {
  auto b = MakeBudget(/*in=*/kBatch + 64, /*out=*/0);  // out=0 so arm B cannot fire
  EXPECT_TRUE(ncclTestBudget(&b, 1, 64));
}

TEST_F(EnqueueMicrotest, TestBudget_WorkSpillsToOutArgs_ArmBOnly) {
  // in holds the batch but NOT batch+work, so arm A is false; out holds work.
  auto b = MakeBudget(/*in=*/kBatch, /*out=*/64);
  EXPECT_TRUE(ncclTestBudget(&b, 1, 64));
}

TEST_F(EnqueueMicrotest, TestBudget_BatchAloneOverflowsInArgs_False) {
  auto b = MakeBudget(/*in=*/kBatch - 1, /*out=*/1 << 20);  // huge out cannot rescue it
  EXPECT_FALSE(ncclTestBudget(&b, 1, 0));
}

TEST_F(EnqueueMicrotest, TestBudget_WorkExceedsBothBudgets_False) {
  auto b = MakeBudget(/*in=*/kBatch, /*out=*/63);
  EXPECT_FALSE(ncclTestBudget(&b, 1, 64));
}

TEST_F(EnqueueMicrotest, TestBudget_ExactFitInline_IsInclusive) {
  // <= not <: the boundary is the whole point. batch+work == in exactly.
  auto b = MakeBudget(/*in=*/kBatch + 64, /*out=*/0);
  EXPECT_TRUE(ncclTestBudget(&b, 1, 64));
  auto tight = MakeBudget(/*in=*/kBatch + 63, /*out=*/0);
  EXPECT_FALSE(ncclTestBudget(&tight, 1, 64)) << "one byte over must not fit";
}

TEST_F(EnqueueMicrotest, TestBudget_ExactFitSpilled_IsInclusive) {
  auto b = MakeBudget(/*in=*/kBatch, /*out=*/64);  // both == exactly
  EXPECT_TRUE(ncclTestBudget(&b, 1, 64));
  auto tight = MakeBudget(/*in=*/kBatch, /*out=*/63);
  EXPECT_FALSE(ncclTestBudget(&tight, 1, 64));
}

TEST_F(EnqueueMicrotest, TestBudget_BatchBytesScalesWithBatchCount) {
  // Pins the nWorkBatches * sizeof(ncclDevWorkBatch) product: a dropped
  // multiply would make 4 batches look the same as 1.
  auto b = MakeBudget(/*in=*/4 * kBatch, /*out=*/0);
  EXPECT_TRUE(ncclTestBudget(&b, 4, 0));
  EXPECT_FALSE(ncclTestBudget(&b, 5, 0));
}

TEST_F(EnqueueMicrotest, TestBudget_ZeroWorkZeroBatches_Fits) {
  auto b = MakeBudget(0, 0);
  EXPECT_TRUE(ncclTestBudget(&b, 0, 0));  // 0 + 0 <= 0
}

// ===========================================================================
// rcclShmemScratchWarpSize / rcclShmemDynamicSize (enqueue.cc:60, :72)
// constexpr -- assert at COMPILE time. No fakes, no runtime.
// ===========================================================================

TEST_F(EnqueueMicrotest, ShmemScratchWarpSize_IsSixteenByteAligned) {
  // The `+15 & -16` pad is the last operation; dropping it changes these.
  static_assert(rcclShmemScratchWarpSize(942, 32) % 16 == 0, "must be 16B aligned");
  static_assert(rcclShmemScratchWarpSize(950, 32) % 16 == 0, "must be 16B aligned");
  EXPECT_EQ(0, rcclShmemScratchWarpSize(942, 32) % 16);
}

TEST_F(EnqueueMicrotest, ShmemScratchWarpSize_GrowsWithWarpSize) {
  // Every term in the max() scales with WarpSize, so 64 must exceed 32.
  EXPECT_GT(rcclShmemScratchWarpSize(942, 64), rcclShmemScratchWarpSize(942, 32));
}

TEST_F(EnqueueMicrotest, ShmemScratchWarpSize_SimpleTermDominatesAtWarp32) {
  // Pins the actual computed value AND which term won. Measured: the SIMPLE
  // term (ncclCollUnroll*WarpSize + 1)*16 = 4112 at gfx942, which beats LL128's
  // 8*32*8 = 2048. Asserting the winner by name means a change to ncclCollUnroll
  // -- or a reordered max() argument -- fails here instead of silently shifting
  // the LDS budget.
  constexpr int kSimple = (ncclCollUnroll(942) * 32 + 1) * 16;
  constexpr int kLL128 = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD * 32) * int(sizeof(uint64_t));
  static_assert(kSimple > kLL128, "SIMPLE is expected to dominate at gfx942/warp32");
  EXPECT_EQ((kSimple + 15) & -16, rcclShmemScratchWarpSize(942, 32));
  EXPECT_EQ(4112, rcclShmemScratchWarpSize(942, 32)) << "measured constant";
}

TEST_F(EnqueueMicrotest, ShmemScratchWarpSize_NvlsTermOnlyAppliesAtArch900Plus) {
  // :66 gates the NVLS term on `cudaArch >= 900`, then adds a flat +16 either
  // way. Below 900 the term is 0, so the result is driven purely by SIMPLE.
  constexpr int kSimple899 = (ncclCollUnroll(899) * 32 + 1) * 16;
  EXPECT_EQ((kSimple899 + 15) & -16, rcclShmemScratchWarpSize(899, 32));
}

#ifndef RCCL_DEVICE_LINKER
TEST_F(EnqueueMicrotest, ShmemDynamicSize_BelowArch700_IsZero) {
  // The `cudaArch < 700 ? 0` gate.
  EXPECT_EQ(0, rcclShmemDynamicSize(600, 32));
  EXPECT_NE(0, rcclShmemDynamicSize(700, 32)) << "700 is the inclusive boundary";
}

TEST_F(EnqueueMicrotest, ShmemDynamicSize_IsScratchTimesWarpCount) {
  // Pins the product, so a changed maxNthreads or a dropped divide shows up.
  constexpr int kWarps = RCCL_DEFAULT_MAX_NTHREADS / 32;
  EXPECT_EQ(rcclShmemScratchWarpSize(942, 32) * kWarps, rcclShmemDynamicSize(942, 32));
}

TEST_F(EnqueueMicrotest, ShmemDynamicSize_Gfx950_UsesItsOwnMaxNthreads) {
  // HONEST SCOPE: RCCL_GFX950_MAX_NTHREADS and RCCL_DEFAULT_MAX_NTHREADS are
  // both 256 today, so the `(cudaArch == 950) ? ... : ...` ternary is an
  // EQUIVALENT MUTANT and the assertion below is a tautology -- there is nothing
  // here for a test to catch. The static_assert is the real guard: the moment
  // the two constants diverge it fails, and whoever changes them is told to make
  // this test differential (compare 950 against a non-950 arch) at exactly the
  // point the branch regains power.
  static_assert(RCCL_GFX950_MAX_NTHREADS == RCCL_DEFAULT_MAX_NTHREADS,
                "The gfx950 nthreads constant now differs from the default: make "
                "ShmemDynamicSize_Gfx950_UsesItsOwnMaxNthreads differential.");
  constexpr int kWarps950 = RCCL_GFX950_MAX_NTHREADS / 32;
  EXPECT_EQ(rcclShmemScratchWarpSize(950, 32) * kWarps950, rcclShmemDynamicSize(950, 32));
}
#endif

// ===========================================================================
// calcP2pChannelCount (enqueue.cc:1550)
//
// DEAD CODE: repo-wide grep finds exactly one occurrence -- its own definition.
// Nothing calls it. Tested anyway to pin current behaviour, because the
// alternative is deleting it, and that is a maintainer's call not a test's.
// Reported in the PR body rather than silently covered.
// ===========================================================================

TEST_F(EnqueueMicrotest, CalcP2pChannelCount_FitsAtMinChannels_NoDoubling) {
  // size <= maxSize immediately, so the loop never runs.
  EXPECT_EQ(1, calcP2pChannelCount(/*total=*/1024, /*min=*/1, /*max=*/8,
                                   /*minSize=*/0, /*maxSize=*/4096));
}

TEST_F(EnqueueMicrotest, CalcP2pChannelCount_DoublesUntilUnderMaxSize) {
  // total/1 = 8192 > 2048 -> 2 (4096) -> 4 (2048) stops: 2048 > 2048 is false.
  EXPECT_EQ(4, calcP2pChannelCount(/*total=*/8192, /*min=*/1, /*max=*/8,
                                   /*minSize=*/0, /*maxSize=*/2048));
}

TEST_F(EnqueueMicrotest, CalcP2pChannelCount_StopsAtHalfMaxChannels) {
  // The guard is `nChannels <= maxChannels/2`, so with max=8 the last legal
  // doubling is 4->8. Even a still-too-large size cannot push past 8.
  EXPECT_EQ(8, calcP2pChannelCount(/*total=*/1 << 20, /*min=*/1, /*max=*/8,
                                   /*minSize=*/0, /*maxSize=*/1));
}

TEST_F(EnqueueMicrotest, CalcP2pChannelCount_MinSizeFloorsInitialSizeOnly) {
  // LATENT INCONSISTENCY (enqueue.cc:1551 vs :1554): the initial size is
  // max(minSize, divUp(total,min)) but the in-loop recompute drops the minSize
  // floor. Here minSize alone forces entry to the loop even though the true
  // per-channel size is already tiny; the first recompute then falls below
  // maxSize and stops. Pins that asymmetry.
  EXPECT_EQ(2, calcP2pChannelCount(/*total=*/16, /*min=*/1, /*max=*/8,
                                   /*minSize=*/4096, /*maxSize=*/2048));
}

// ===========================================================================
// geteActivationMask / gettaskEventHandle (enqueue.cc:1820, :1830)
//
// Identical 3-arm shape. The first arm is a RANGE test over the ncclFunc_t
// enum, so it silently depends on SendRecv/Send/Recv staying adjacent. If they
// are reordered, the wrong union member is read -- a type-confused pointer
// deref. These tests pin the range boundaries specifically.
// ===========================================================================

namespace {
// Distinct sentinels so an arm returning the wrong member is visible.
constexpr int kP2pMask = 0x5A;
constexpr int kCollMask = 0xA5;

struct ActivationOp {
  ncclProxyOp op{};
  ncclTaskP2p p2p{};
  ncclTaskColl coll{};
  explicit ActivationOp(ncclFunc_t f) {
    p2p.eActivationMask = kP2pMask;
    coll.eActivationMask = kCollMask;
    p2p.eventHandle = &p2p;    // self-address: unique, non-null, comparable
    coll.eventHandle = &coll;
    op.coll = f;
    op.task.p2p = &p2p;        // union: set p2p last for p2p funcs
  }
  void UseColl() { op.task.coll = &coll; }
};
}  // namespace

TEST_F(EnqueueMicrotest, GeteActivationMask_P2pRange_ReadsP2pTask) {
  // All three of SendRecv/Send/Recv must take the p2p arm.
  for (ncclFunc_t f : {ncclFuncSendRecv, ncclFuncSend, ncclFuncRecv}) {
    ActivationOp a(f);
    EXPECT_EQ(kP2pMask, geteActivationMask(&a.op)) << "func=" << int(f);
    EXPECT_EQ(&a.p2p, gettaskEventHandle(&a.op)) << "func=" << int(f);
  }
}

TEST_F(EnqueueMicrotest, GeteActivationMask_AllGatherV_IsZeroAndNullHandle) {
  ActivationOp a(ncclFuncAllGatherV);
  a.UseColl();  // the coll member is populated, yet this arm must ignore it
  EXPECT_EQ(0, geteActivationMask(&a.op));
  EXPECT_EQ(nullptr, gettaskEventHandle(&a.op));
}

TEST_F(EnqueueMicrotest, GeteActivationMask_CollFunc_ReadsCollTask) {
  for (ncclFunc_t f : {ncclFuncAllReduce, ncclFuncBroadcast, ncclFuncReduce,
                       ncclFuncAllGather, ncclFuncReduceScatter}) {
    ActivationOp a(f);
    a.UseColl();
    EXPECT_EQ(kCollMask, geteActivationMask(&a.op)) << "func=" << int(f);
    EXPECT_EQ(&a.coll, gettaskEventHandle(&a.op)) << "func=" << int(f);
  }
}

TEST_F(EnqueueMicrotest, GeteActivationMask_RangeBoundariesAreExact) {
  // The arm is `ncclFuncSendRecv <= coll && coll <= ncclFuncRecv`. Pin that the
  // enum neighbours on each side do NOT take it -- this is what breaks if the
  // ncclFunc_t enum is ever reordered.
  ActivationOp below(static_cast<ncclFunc_t>(int(ncclFuncSendRecv) - 1));
  below.UseColl();
  EXPECT_EQ(kCollMask, geteActivationMask(&below.op)) << "just below the range";

  ActivationOp above(static_cast<ncclFunc_t>(int(ncclFuncRecv) + 1));
  above.UseColl();
  EXPECT_EQ(kCollMask, geteActivationMask(&above.op)) << "just above the range";
}
