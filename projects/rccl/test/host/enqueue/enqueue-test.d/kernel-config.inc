/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Kernel launch configuration: traffic per byte, argument budgets, shared memory, packed channels, default kernel.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// ncclFuncTrafficPerByte -- traffic multiplier per collective. NCCL 2.32.6 moved it from enqueue.cc to
// src/channel.cc; fakes/channel_fakes.cc compiles that real body. Pure: no externals. The nRanks-returning arms
// are indistinguishable
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
    const bool namedArm = (f == ncclFuncAllReduce) || (f == ncclFuncAllGather) || (f == ncclFuncAlltoAll) ||
                          (f == ncclFuncReduceScatter) || (f == ncclFuncAlltoAllvGda);
    if (namedArm) continue;
    EXPECT_EQ(1, ncclFuncTrafficPerByte(f, kRanks)) << "func=" << i;
  }
}

// NCCL 2.32.6 added AlltoAll to the nRanks arm.
TEST_F(EnqueueMicrotest, FuncTrafficPerByte_AlltoAll_IsNRanks) {
  EXPECT_EQ(8, ncclFuncTrafficPerByte(ncclFuncAlltoAll, 8));
  EXPECT_EQ(3, ncclFuncTrafficPerByte(ncclFuncAlltoAll, 3));
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

TEST_F(EnqueueMicrotest, KernelArgs_DefaultStorageFitsOneBatchPerChannelPlusOneP2pWork) {
  // ROCM-33800: a 1-rank gfx1250 comm schedules a P2P round over MAXCHANNELS parts, one batch
  // each, so that round plus its one ncclDevWorkP2p must fit an empty plan's kernel args.
#ifdef ENABLE_WARP_SPEED
  GTEST_SKIP() << "WARP_SPEED raises MAXCHANNELS to 512 over the same 5K args; such a round is rejected";
#endif
  const ssize_t inArgs = ncclMaxKernelArgsSize() - sizeof(struct ncclDevKernelArgs);
  EXPECT_LE(ssize_t(MAXCHANNELS * kBatch + sizeof(struct ncclDevWorkP2p)), inArgs);
}

// ===========================================================================
// rcclShmemScratchWarpSize / rcclShmemDynamicSize (enqueue.cc:60, :72)
// constexpr -- assert at COMPILE time. No fakes, no runtime.
// ===========================================================================

TEST_F(EnqueueMicrotest, ShmemScratchWarpSize_IsSixteenByteAligned) {
  // HONEST SCOPE: the `+15 & -16` pad at :67-68 is an EQUIVALENT MUTANT -- every
  // term entering the max is already a multiple of 16 (LL 0; LL128 64*WarpSize;
  // SIMPLE (unroll*WarpSize + 1)*16; NVLS 64*WarpSize + 16), so dropping the pad
  // changes nothing and this cannot fail on it. It pins the alignment contract
  // the callers rely on, not the pad.
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

TEST_F(EnqueueMicrotest, ShmemScratchWarpSize_Ll128TermDominatesAtGfx1250) {
  // gfx1250 is the only arch where LL128 wins the max(): 32 elems/thread, and
  // ncclCollUnroll(1250) is 6 rather than 8.
  constexpr int kLL128 = (32 * 32) * int(sizeof(uint64_t));
  constexpr int kSimple = (ncclCollUnroll(1250) * 32 + 1) * 16;
  static_assert(kLL128 > kSimple, "LL128 is expected to dominate at gfx1250/warp32");
  EXPECT_EQ((kLL128 + 15) & -16, rcclShmemScratchWarpSize(1250, 32));
  EXPECT_EQ(8192, rcclShmemScratchWarpSize(1250, 32)) << "measured constant";
  EXPECT_EQ(8, rcclLL128ShmemElemsPerThread(942)) << "non-gfx1250 unchanged";
  EXPECT_EQ(32, rcclLL128ShmemElemsPerThread(1250));
}

TEST_F(EnqueueMicrotest, ShmemScratchWarpSize_NvlsTermNeverWins) {
  // HONEST SCOPE: :66 gates the NVLS term on `cudaArch >= 900`, but that gate is
  // an EQUIVALENT MUTANT today. ncclNvlsUnrollBytes is the constant 4*16 for
  // every arch, so the gated term is at most WarpSize*64 + 16, while SIMPLE is
  // ncclCollUnroll(arch)*WarpSize*16 + 16 and ncclCollUnroll is never below 4.
  // SIMPLE therefore wins (or ties) at every arch and WarpSize, and deleting the
  // gate changes nothing. The static_assert is the real guard: the moment the
  // two constants move apart it fails, and whoever changes them is told to make
  // this test differential (compare 899 against 900) at exactly the point the
  // branch regains power.
  static_assert(ncclCollUnroll(700) * 16 >= ncclNvlsUnrollBytes(900),
                "The NVLS unroll can now exceed the collective unroll: make "
                "ShmemScratchWarpSize_NvlsTermNeverWins differential.");
  constexpr int kSimple899 = (ncclCollUnroll(899) * 32 + 1) * 16;
  EXPECT_EQ((kSimple899 + 15) & -16, rcclShmemScratchWarpSize(899, 32));
  EXPECT_EQ(rcclShmemScratchWarpSize(899, 32), rcclShmemScratchWarpSize(900, 32))
      << "the gate is inert while ncclCollUnroll(899) == ncclCollUnroll(900)";
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
#else
TEST_F(EnqueueMicrotest, ShmemDynamicSize_DeviceLinker_IsAlwaysZero) {
  // The arm this build actually SHIPS. ENABLE_DEVICE_LINKER defaults ON
  // (projects/rccl/CMakeLists.txt:92) and reaches this binary through the rccl
  // target's COMPILE_DEFINITIONS (src/CMakeLists.txt:1065 ->
  // test/CMakeLists.txt:208 -> RCCL_COMMON_COMPILE_DEFS), so the three tests
  // above are compiled out of the default configuration and this is the only
  // ShmemDynamicSize coverage there.
  //
  // Under RCCL_DEVICE_LINKER the scratch and ncclShmem are static __shared__
  // (device/common.h), so asking for the same bytes as DYNAMIC shmem would
  // double-count and overflow the per-block LDS budget. enqueue.cc:73-78
  // therefore returns 0 for every input, with both parameters cast to void.
  // HONEST SCOPE: on this arm the preprocessor has DELETED the arch gate and the
  // gfx950 ternary, so the inputs below do not exercise them -- they are not
  // evaluated at all. Varying the inputs proves only that the arm ignores them,
  // which is exactly what `(void)cudaArch; (void)WarpSize;` promises. What this
  // test actually guards is REMOVAL of the guard: mutating the arm's body to the
  // non-linker formula fails here and nowhere else in the suite.
  EXPECT_EQ(0, rcclShmemDynamicSize(600, 32));
  EXPECT_EQ(0, rcclShmemDynamicSize(700, 32));
  EXPECT_EQ(0, rcclShmemDynamicSize(942, 32));
  EXPECT_EQ(0, rcclShmemDynamicSize(950, 32));
  EXPECT_EQ(0, rcclShmemDynamicSize(950, 64));
}
#endif

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
  // HONEST SCOPE: this pins the behaviour at count == 0 but CANNOT fail on its
  // own. Dropping `count == 0` from the early-out at :179 is an EQUIVALENT
  // MUTANT: with count == 0, cells = divUp(0, cellSize) is 0, so cellsPerChannel
  // is 0 and the guard at :190 returns the same nMaxChannels by the long route.
  // The value is documentary -- it records the contract, not a caught mutant.
  EXPECT_EQ(16, rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, 0,
                                         ncclFloat32, NCCL_PROTO_SIMPLE, 16));
}

TEST_F(EnqueueMicrotest, PackedChannels_TinyTransfer_PacksIntoOneChannel) {
  // Below the 16 KiB MinTrafficPerChannel floor everything fits in one channel,
  // however many are offered.
  EXPECT_EQ(1, rcclKernelPackedChannels(RankComm(8).get(), ncclFuncAllReduce, 4,
                                        ncclFloat32, NCCL_PROTO_SIMPLE, 32));
}

TEST_F(EnqueueMicrotest, PackedChannels_CapsToCommNChannels) {
  // nChannels == 0 skips the cap. Count 10240 / max 32 packs to 5.
  {
    RankComm rc(8);
    EXPECT_EQ(0, rc.get()->nChannels);
    EXPECT_EQ(5, rcclKernelPackedChannels(rc.get(), ncclFuncAllReduce, 10240,
                                          ncclFloat32, NCCL_PROTO_SIMPLE, 32));
  }
  // nMaxChannels > nChannels: the cap is applied before the packer arithmetic.
  // 3 with the cap, 5 without it, 4 if the cap were applied after the arithmetic.
  {
    RankComm rc(8);
    rc.get()->nChannels = 4;
    EXPECT_EQ(3, rcclKernelPackedChannels(rc.get(), ncclFuncAllReduce, 10240,
                                          ncclFloat32, NCCL_PROTO_SIMPLE, 32));
  }
  // nMaxChannels <= nChannels: the cap does not change the answer.
  {
    RankComm uncapped(8);
    RankComm capped(8);
    capped.get()->nChannels = 16;
    const size_t count = size_t(1) << 24;
    int plain = rcclKernelPackedChannels(uncapped.get(), ncclFuncAllReduce, count,
                                         ncclFloat32, NCCL_PROTO_SIMPLE, 8);
    int limited = rcclKernelPackedChannels(capped.get(), ncclFuncAllReduce, count,
                                           ncclFloat32, NCCL_PROTO_SIMPLE, 8);
    EXPECT_EQ(8, plain);
    EXPECT_EQ(plain, limited);
  }
  // nChannels == 1 is the first value where the cap bites; a > 1 guard would return 5.
  {
    RankComm rc(8);
    rc.get()->nChannels = 1;
    EXPECT_EQ(1, rcclKernelPackedChannels(rc.get(), ncclFuncAllReduce, 10240,
                                          ncclFloat32, NCCL_PROTO_SIMPLE, 32));
  }
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

TEST_F(EnqueueMicrotest, PackedChannels_LLNeverNeedsFewerChannelsThanSimple) {
  // `if (protocol == NCCL_PROTO_LL) trafficPerByte *= 4` -- LL moves 4x the
  // bytes, so at a size where SIMPLE has not yet saturated, LL needs >= as many
  // channels. This pins the ORDERING only: >= is satisfied with equality by an
  // implementation that dropped the 4x, so it does not on its own prove the
  // multiplier is applied. PackedChannels_LLNeedsStrictlyMoreChannelsThanSimple_Isolated
  // carries the strict check.
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
// ncclPlanSetDefaultKernel (enqueue.cc:2866) and ncclGetAlgoInfo (:2846)
// Thin out-of-TU shims that exist so upstream-synced sources can link. Small,
// but ncclPlanSetDefaultKernel is the one place ncclKerns[] is read by index --
// worth pinning that the index comes from comm->unroll.
// ===========================================================================

TEST_F(EnqueueMicrotest, PlanSetDefaultKernel_SelectsKernelByCommUnroll) {
  // ncclGetKernelIndex(comm) is comm->unroll. Two different unrolls must select
  // DIFFERENT table entries -- a hardcoded index would give the same pointer.
  auto comm = std::make_unique<ncclComm>();
  auto planA = std::make_unique<ncclKernelPlan>();
  auto planB = std::make_unique<ncclKernelPlan>();

  comm->unroll = 0;
  ncclPlanSetDefaultKernel(comm.get(), planA.get());
  comm->unroll = 1;
  ncclPlanSetDefaultKernel(comm.get(), planB.get());

  // Assert the INDEX ARITHMETIC against the table, the same shape the sibling
  // test below uses. Comparing the two plans to each other instead would rest on
  // the surrogates having distinct addresses, which is a property of the test
  // shim rather than of the code under test.
  EXPECT_EQ(ncclKerns[0].kernelFn, planA->kernelFn) << "unroll 0 must select entry 0";
  EXPECT_EQ(ncclKerns[1].kernelFn, planB->kernelFn) << "unroll 1 must select entry 1";
}

TEST_F(EnqueueMicrotest, PlanSetDefaultKernel_CopiesSpecializedFlagFromTheTable) {
  // Every entry in ncclKerns is {fn, true}; pin that the flag is copied from the
  // table rather than left at the plan's default. Comparing against
  // ncclKerns[2].kernelFn checks assignment consistency WITHIN this TU (the
  // addresses are host surrogates), not real kernel linkage.
  auto comm = std::make_unique<ncclComm>();
  auto plan = std::make_unique<ncclKernelPlan>();
  plan->kernelSpecialized = false;
  comm->unroll = 2;
  ncclPlanSetDefaultKernel(comm.get(), plan.get());
  EXPECT_TRUE(plan->kernelSpecialized);
  EXPECT_EQ(ncclKerns[2].kernelFn, plan->kernelFn);
}

TEST_F(EnqueueMicrotest, KernelTable_HostSurrogate_ShapeIsSixDistinctSpecializedEntries) {
  // SCOPE: in this binary the six kernel addresses are HOST SURROGATES defined at
  // the top of this TU, not the real __global__ symbols. This test therefore
  // proves only that ncclKerns[] is built with one entry per surrogate, all
  // flagged specialized -- it CANNOT catch a missing device kernel, a signature
  // mismatch, or broken device linkage. Those need a device-linked build.
  // The useful part is the shape/arity of the table, which the indexing tests
  // above then rely on.
  constexpr int kCount = int(sizeof(ncclKerns) / sizeof(ncclKerns[0]));
  EXPECT_EQ(6, kCount);
  std::vector<void*> fns;
  for (int i = 0; i < kCount; ++i) {
    EXPECT_NE(nullptr, ncclKerns[i].kernelFn) << "entry " << i;
    EXPECT_TRUE(ncclKerns[i].specialized) << "entry " << i;
    fns.push_back(ncclKerns[i].kernelFn);
  }
  std::sort(fns.begin(), fns.end());
  EXPECT_EQ(fns.end(), std::unique(fns.begin(), fns.end())) << "entries must be distinct";
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
// EQUIVALENT MUTANT (documented, not fixed): dropping `count == 0` from the
// early-out at enqueue.cc:179.
//
// With count == 0: cells = divUp(0, cellSize) = 0, so
// cellsPerChannel = min(0, ...) = 0 and the SECOND guard at :190
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
