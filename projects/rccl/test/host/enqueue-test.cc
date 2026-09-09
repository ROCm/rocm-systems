/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/enqueue.cc: the UUT is #include'd, so static
// helpers are directly callable.
//
// LINE-NUMBER BASE: every `enqueue.cc:NNNN` citation in these tests refers to
// src/enqueue.cc as committed, NOT to the hipified copy this TU compiles. Hipify
// inserts one line near the top, so add 1 when navigating build/hipify/src/enqueue.cc.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "fakes/enqueue_fakes.h"

// alloc.h first, so its macros are visible to be #undef'd before enqueue.cc's
// transitive includes see them.
#include "alloc.h"

#include "fakes/param_redirect.h"  // redirects NCCL_PARAM and both RCCL_PARAM spellings

#include "fakes/nvtx_redirect.h"  // neuter / block nvtx.h before enqueue.cc includes it

// ---------------------------------------------------------------------------
// enqueue.cc:28 includes "common.h", which resolves to src/device/common.h -- a
// DEVICE header. Under --offload-host-only it cannot compile on its own terms
// (extern __shared__ variables; undeclared insert_random_delay_per_warp).
//
// As of this change, the only declaration enqueue.cc uses from it is the six
// ncclDevKernel_Generic_N kernels whose ADDRESSES populate ncclKerns[] at
// enqueue.cc:53. On every path this binary covers, the only use is
// `plan->kernelFn = ncclKerns[i].kernelFn` -- an opaque void* that is stored and
// nothing more. That is a statement about COVERAGE, not about enqueue.cc, which
// does dereference these addresses elsewhere: ncclInitKernelsForDevice (:110)
// passes them to cudaFuncGetAttributes (:115) and cudaFuncSetAttribute (:121),
// and ncclLaunchKernel (:2218) reads plan->kernelFn back to launch it. No
// covered path reaches any of those three, which is what makes the substitution
// safe today; a test that reaches one invalidates it. So we pre-set the device
// header's own include guard to neuter it and supply those six symbols as
// ordinary host functions.
//
// LIMITS OF THIS SUBSTITUTION -- read before relying on it:
//   * These are host functions, NOT __global__ declarations with the production
//     grid-constant annotation. This binary therefore CANNOT validate the real
//     kernel declarations, their symbols, or the host/device ABI. It exercises
//     host-side table indexing only; kernel-table integrity belongs to a
//     device-linked build.
//   * Neutering the whole header changes the include environment for the rest of
//     the TU. If enqueue.cc later needs another declaration from device/common.h,
//     the build may keep working only because something arrives transitively.
//     A compile error here is the expected signal to revisit this shim -- do not
//     paper over it by adding another local definition.
//
// Pre-setting another header's guard is the same technique init-test.cc uses for
// the nvtx.h / nvtx_stub.h nccl_domain collision.
// ---------------------------------------------------------------------------
#define NCCL_DEVICE_COMMON_H_
// ncclDevKernelArgsDefaultStorage comes from src/include/device.h, already in
// scope via fakes/enqueue_fakes.h -> sym_kernels.h (enqueue.cc itself is only
// included later). The generated device_table.h is NOT needed here.
void ncclDevKernel_Generic_1(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_2(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_4(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_8(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_16(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_32(ncclDevKernelArgsDefaultStorage) {}

// ENQUEUE_CC_PATH is ${PROJECT_BINARY_DIR}/hipify/src/enqueue.cc -- enqueue.cc is
// basename-unique in the tree, so hipify keeps its name (no _tmp suffix).
#include ENQUEUE_CC_PATH

class EnqueueMicrotest : public ::testing::Test {
 protected:
  // Reset in BOTH, not just TearDown. TearDown alone leaves the first test
  // executed running against static-initialiser state rather than reset state --
  // which test that is depends on --gtest_shuffle. Those two states happen to be
  // identical for every global in the reset closure today, but nothing enforces
  // that, and SetUp costs one line to stop relying on it. It also means a future
  // second fixture in this binary cannot inherit a dirty process.
  void SetUp() override { ResetEnqueueFakes(); }
  void TearDown() override { ResetEnqueueFakes(); }
};

// ---------------------------------------------------------------------------
// The test bodies follow, grouped by unit under test. Each group opens with a
// `// ===` banner naming its unit and the enqueue.cc line it starts at.
//
// Order is not arbitrary: several fixtures are reused later in the file.
//   CostTask / ScriptAllTimes / CostTable  defined with updateCollCostTable
//   RedOpComm                              defined with ncclRedOpCreatePreMulSum
//   BatchPlanComm                          defined with addWorkBatchToPlan
// ---------------------------------------------------------------------------

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
  // LATENT INCONSISTENCY (enqueue.cc:1551 vs :1555): the initial size is
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

// ===========================================================================
// hostToDevRedOp (enqueue.cc:3186) -- maps a host ncclRedOp_t + datatype onto
// the device-side op and its scalar argument.
//
// The scalarArg for Min/Max is a XOR MASK, not a value, computed from the
// datatype's bit width:
//     allBits = ~0 >> (64 - nbits)
//     signBit = allBits ^ (allBits >> 1)
//     mask    = (signed ? signBit : 0) ^ (op==Max ? allBits : 0)
// A return-code assertion cannot see any of that, so every test below asserts
// the exact mask. This is the function where mutation earns its keep.
// ===========================================================================

namespace {
// Poison, not zero: scalarArg is written by every arm, so a zero-initialised
// out-param cannot distinguish "written 0" from "not written".
constexpr uint64_t kRedOpPoison = 0xDEADBEEFDEADBEEFull;

ncclDevRedOpFull MakeRedOpOut() {
  ncclDevRedOpFull o{};
  o.scalarArg = kRedOpPoison;
  o.scalarArgIsPtr = true;   // every path must clear this
  return o;
}

// Expected mask, recomputed independently of the production expression so a
// mutated production formula does not also mutate the oracle.
uint64_t ExpectMinMaxMask(int nbits, bool isSigned, bool isMax) {
  uint64_t allBits = (nbits >= 64) ? ~0ull : ((1ull << nbits) - 1);
  uint64_t signBit = allBits ^ (allBits >> 1);
  uint64_t m = 0;
  if (isSigned) m ^= signBit;
  if (isMax) m ^= allBits;
  return m;
}
// MEASURED: sizeof(ncclComm) is ~3.8 MB. These tests only need nRanks set, but a
// stack instance is still 3.8 MB of frame -- and AvgScalesWithRankCount needs two
// at once, which alone exceeds a default 8 MB stack. Heap-allocate, as
// CostComm/ChunkComm/RankComm/BatchPlanComm/FinishComm/RedOpComm all do.
class AvgComm {
 public:
  explicit AvgComm(int nRanks) : storage_(new ncclComm{}) { storage_->nRanks = nRanks; }
  ncclComm* get() { return storage_.get(); }
 private:
  std::unique_ptr<ncclComm> storage_;
};
}  // namespace

TEST_F(EnqueueMicrotest, HostToDevRedOp_Sum_MapsToDevSumAndClearsPtrFlag) {
  auto out = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclSum, ncclFloat32, nullptr));
  EXPECT_EQ(ncclDevSum, out.op);
  EXPECT_FALSE(out.scalarArgIsPtr);
  EXPECT_EQ(ncclSum, out.proxyOp) << "proxyOp must carry the ORIGINAL host op";
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_Prod_MapsToDevProd) {
  auto out = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclProd, ncclFloat32, nullptr));
  EXPECT_EQ(ncclDevProd, out.op);
  EXPECT_EQ(ncclProd, out.proxyOp);
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_MinUnsigned_MaskIsZero) {
  // Unsigned + Min: neither XOR term applies, so the mask is exactly 0. This is
  // the one case where poisoning the out-param matters -- without it, "wrote 0"
  // and "wrote nothing" look identical.
  for (auto dt : {ncclUint8, ncclUint32, ncclUint64}) {
    auto out = MakeRedOpOut();
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclMin, dt, nullptr));
    EXPECT_EQ(ncclDevMinMax, out.op);
    EXPECT_EQ(0u, out.scalarArg) << "dtype=" << int(dt);
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_MaxUnsigned_MaskIsAllBitsOfThatWidth) {
  // Width-dependent: 8/32/64-bit types must give DIFFERENT masks. A hardcoded
  // 64-bit allBits would pass ncclUint64 and fail the other two.
  const struct { ncclDataType_t dt; int nbits; } kCases[] = {
      {ncclUint8, 8}, {ncclUint32, 32}, {ncclUint64, 64}};
  for (auto c : kCases) {
    auto out = MakeRedOpOut();
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclMax, c.dt, nullptr));
    EXPECT_EQ(ExpectMinMaxMask(c.nbits, /*signed=*/false, /*max=*/true), out.scalarArg)
        << "dtype=" << int(c.dt) << " nbits=" << c.nbits;
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_MinSigned_MaskIsSignBitOnly) {
  const struct { ncclDataType_t dt; int nbits; } kCases[] = {
      {ncclInt8, 8}, {ncclInt32, 32}, {ncclInt64, 64}};
  for (auto c : kCases) {
    auto out = MakeRedOpOut();
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclMin, c.dt, nullptr));
    EXPECT_EQ(ExpectMinMaxMask(c.nbits, /*signed=*/true, /*max=*/false), out.scalarArg)
        << "dtype=" << int(c.dt);
    // Differential vs the unsigned case: the signed mask must NOT be zero.
    EXPECT_NE(0u, out.scalarArg) << "signed Min must set the sign bit";
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_MaxSigned_MaskIsSignBitXorAllBits) {
  const struct { ncclDataType_t dt; int nbits; } kCases[] = {
      {ncclInt8, 8}, {ncclInt32, 32}, {ncclInt64, 64}};
  for (auto c : kCases) {
    auto out = MakeRedOpOut();
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclMax, c.dt, nullptr));
    EXPECT_EQ(ExpectMinMaxMask(c.nbits, /*signed=*/true, /*max=*/true), out.scalarArg)
        << "dtype=" << int(c.dt);
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_SignedMaxDiffersFromUnsignedMax) {
  // Pins that the `datatype == ncclInt8|Int32|Int64` guard actually gates the
  // signBit XOR: deleting it makes these two identical.
  auto s = MakeRedOpOut();
  auto u = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&s, ncclMax, ncclInt32, nullptr));
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&u, ncclMax, ncclUint32, nullptr));
  EXPECT_NE(s.scalarArg, u.scalarArg);
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_MinAndMaxDifferByAllBits) {
  // Pins the `(op == ncclMax) ? allBits : 0` term specifically.
  auto mn = MakeRedOpOut();
  auto mx = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&mn, ncclMin, ncclUint32, nullptr));
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&mx, ncclMax, ncclUint32, nullptr));
  EXPECT_EQ(0xFFFFFFFFull, mn.scalarArg ^ mx.scalarArg);
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_ProxyOpPreservesMinVsMax) {
  // Min and Max share one device op (ncclDevMinMax); only proxyOp keeps them
  // apart downstream. A mutant that assigns proxyOp = opFull->op survives every
  // scalarArg assertion but dies here.
  auto mn = MakeRedOpOut();
  auto mx = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&mn, ncclMin, ncclInt32, nullptr));
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&mx, ncclMax, ncclInt32, nullptr));
  EXPECT_EQ(mn.op, mx.op) << "both map to ncclDevMinMax";
  EXPECT_NE(mn.proxyOp, mx.proxyOp) << "but proxyOp must still distinguish them";
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgIntegral_UsesSumPostDivWithRankCountAndSignFlag) {
  // The Avg integral arm packs TWO things into scalarArg:
  //     u64 = comm->nRanks << 1 | datatype_signed
  // so the low bit is the signedness and the rest is the rank count.
  AvgComm comm(8);

  auto si = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&si, ncclAvg, ncclInt32, comm.get()));
  EXPECT_EQ(ncclDevSumPostDiv, si.op);
  EXPECT_EQ((8ull << 1) | 1ull, si.scalarArg) << "signed -> low bit set";

  auto ui = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&ui, ncclAvg, ncclUint32, comm.get()));
  EXPECT_EQ(ncclDevSumPostDiv, ui.op);
  EXPECT_EQ((8ull << 1) | 0ull, ui.scalarArg) << "unsigned -> low bit clear";
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgIntegral_SignedFallthroughCoversAllIntWidths) {
  // The signed cases deliberately FALL THROUGH into the unsigned ones (:3240
  // "no break, we want to fall through"). If a `break` were added, the signed
  // types would leave op unassigned. Sweep all six integral types.
  AvgComm comm(4);
  const struct { ncclDataType_t dt; bool sgn; } kCases[] = {
      {ncclInt8, true},  {ncclInt32, true},  {ncclInt64, true},
      {ncclUint8, false}, {ncclUint32, false}, {ncclUint64, false}};
  for (auto c : kCases) {
    auto out = MakeRedOpOut();
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclAvg, c.dt, comm.get())) << "dtype=" << int(c.dt);
    EXPECT_EQ(ncclDevSumPostDiv, out.op) << "dtype=" << int(c.dt);
    EXPECT_EQ((4ull << 1) | (c.sgn ? 1ull : 0ull), out.scalarArg) << "dtype=" << int(c.dt);
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgFloat32_UsesPreMulSumWithReciprocal) {
  // Float Avg becomes PreMulSum with scalar 1/nRanks, bit-pattern compared.
  AvgComm comm(4);
  auto out = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclAvg, ncclFloat32, comm.get()));
  EXPECT_EQ(ncclDevPreMulSum, out.op);
  float want = float(1.0 / 4);
  uint32_t wantBits;
  std::memcpy(&wantBits, &want, sizeof(wantBits));
  EXPECT_EQ(wantBits, uint32_t(out.scalarArg & 0xFFFFFFFFull));
  EXPECT_FALSE(out.scalarArgIsPtr) << "the scalar is inline, not a pointer";
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgFloat64_UsesPreMulSumWithReciprocal) {
  AvgComm comm(8);
  auto out = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclAvg, ncclFloat64, comm.get()));
  EXPECT_EQ(ncclDevPreMulSum, out.op);
  double want = 1.0 / 8;
  uint64_t wantBits;
  std::memcpy(&wantBits, &want, sizeof(wantBits));
  EXPECT_EQ(wantBits, out.scalarArg);
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgScalesWithRankCount) {
  // Differential: a mutant hardcoding a divisor passes one rank count only.
  AvgComm c4(4);
  AvgComm c8(8);
  auto a = MakeRedOpOut();
  auto b = MakeRedOpOut();
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&a, ncclAvg, ncclFloat32, c4.get()));
  ASSERT_EQ(ncclSuccess, hostToDevRedOp(&b, ncclAvg, ncclFloat32, c8.get()));
  EXPECT_NE(a.scalarArg, b.scalarArg);
}

// ---------------------------------------------------------------------------
// NOT A BUG -- investigated and cleared. The ncclAvg inner datatype switch
// (:3235-3275) has no `default:` arm, which reads like an uninitialised-`op`
// hazard. It is not reachable:
//
//   ncclTypeSize (collectives.h:61-82) returns -1 for any type outside
//   {Int8,Uint8,Float8e4m3,Float8e5m2,Float16,Bfloat16,Int32,Uint32,Float32,
//    Int64,Uint64,Float64} -- and the Avg switch names EXACTLY that same set.
//   So for any type the switch would miss, `nbits = 8*ncclTypeSize(dt)` is
//   negative and the guard at :3211 returns ncclInvalidArgument first.
//
// The guard is therefore load-bearing for two reasons, and the test below pins
// it: it prevents UB in the `>> (64 - nbits)` shift AND it is what makes the
// missing default unreachable. If ncclTypeSize ever gains a type the Avg switch
// does not handle, this coupling breaks silently -- which is worth knowing, and
// is reported in the PR body rather than pinned as a fake bug.
// ---------------------------------------------------------------------------
TEST_F(EnqueueMicrotest, HostToDevRedOp_UnknownDatatype_RejectedByWidthGuard) {
  // :3211 `if (nbits <= 0) return ncclInvalidArgument`. Without it, :3212's
  // `uint64_t(-1) >> (64 - nbits)` is UB for nbits <= 0.
  const auto kBogus = static_cast<ncclDataType_t>(ncclNumTypes + 50);
  ASSERT_LE(ncclTypeSize(kBogus), 0) << "precondition: this id must be unknown";

  for (auto op : {ncclSum, ncclProd, ncclMin, ncclMax, ncclAvg}) {
    auto out = MakeRedOpOut();
    EXPECT_EQ(ncclInvalidArgument, hostToDevRedOp(&out, op, kBogus, nullptr))
        << "op=" << int(op);
  }
}

TEST_F(EnqueueMicrotest, HostToDevRedOp_AvgSwitchCoversEveryTypeNcclTypeSizeAccepts) {
  // The coupling described above, as an executable assertion: for EVERY datatype
  // that ncclTypeSize accepts, the Avg arm must assign a real op. If someone adds
  // a datatype to ncclTypeSize but forgets the Avg switch, this fails here rather
  // than shipping an uninitialised op.
  AvgComm comm(4);
  const auto kSentinel = static_cast<ncclDevRedOp_t>(0x7E);
  for (int i = 0; i < int(ncclNumTypes); ++i) {
    const auto dt = static_cast<ncclDataType_t>(i);
    if (ncclTypeSize(dt) <= 0) continue;  // rejected by the width guard anyway
    auto out = MakeRedOpOut();
    out.op = kSentinel;
    ASSERT_EQ(ncclSuccess, hostToDevRedOp(&out, ncclAvg, dt, comm.get())) << "dtype=" << i;
    EXPECT_NE(kSentinel, out.op)
        << "dtype=" << i << " is accepted by ncclTypeSize but not handled by the "
           "ncclAvg switch at enqueue.cc:3235-3275, leaving op uninitialised";
  }
}

// ===========================================================================
// calcCollChunking (enqueue.cc:2871) -- the largest host-reachable block in the
// file (278 lines, ~89 branch points) and entirely HIP-free.
//
// Two separable concerns:
//   1. Pattern selection   (:2877-2916) -- a 9-arm switch over func x algorithm.
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
    // so a zero WarpSize makes grainSize 0, and :3028's
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
  ASSERT_EQ(ncclSuccess, rt.rc);
  ASSERT_EQ(ncclSuccess, rr.rc);
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
  // :2924 `if (protocol == NCCL_PROTO_LL) chunkSize /= 2`.
  ChunkComm cc(1 << 22);
  auto simple = MakeTask(ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  auto ll = MakeTask(ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL);
  auto rs = RunChunking(cc, simple, 4, 1 << 24);
  auto rl = RunChunking(cc, ll, 4, 1 << 24);
  EXPECT_EQ(rs.chunkSize / 2, rl.chunkSize);
}

TEST_F(EnqueueMicrotest, CalcCollChunking_LL128Protocol_ScalesByLineAndDataElemsThenAlignsToGrain) {
  // TWO steps, and the second is easy to forget:
  //   :2925 chunkSize = (chunkSize / ll128LineElems) * ll128DataElems
  //   :3028 chunkSize = chunkSize / grainSize * grainSize   (align DOWN)
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
  // chunkSteps is honoured ONLY for (SIMPLE, RING) -- :2921. Two tasks that
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
  // STRICTLY greater: every rung is `nBytes / (nChannels*chunkSize) < depth*K`,
  // so substituting 1 for tree.depth makes shallow and deep identical and a >=
  // assertion still passes. Measured values are 131072 vs 65536.
  EXPECT_GT(RunChunking(shallow, t, 4, 1 << 22).chunkSize,
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
  // :2973 -- with nNodes>1 AND NVLS allreduce bandwidth < 150, the cap drops to
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
  // :3028 `chunkSize = chunkSize / grainSize * grainSize` -- an align-DOWN to the
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

TEST_F(EnqueueMicrotest, CalcCollChunking_Ll128GrainIsNonZeroForTheFixture) {
  // HONEST SCOPE: ChunkComm hardcodes WarpSize=32, ll128LineElems=120 and
  // ll128DataElems=112, so this does not vary any of them -- it only asserts the
  // grain is non-zero for that one fixture.
  // Guards the fixture invariant above as an executable assertion: if WarpSize or
  // the ll128 elem counts are ever left unset, grainSize goes to 0 and :3028
  // divides by zero. This fails loudly instead of core-dumping the suite.
  ChunkComm cc;
  EXPECT_GT(rcclProtoGrainSize(NCCL_PROTO_LL128, cc.get()), 0)
      << "LL128 grain is WarpSize*8*ll128DataElems*8/ll128LineElems -- a zero "
         "WarpSize or ll128DataElems makes calcCollChunking:3028 SIGFPE";
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
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a) {
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
        if (t[a][p] != NCCL_ALGO_PROTO_IGNORE) {
          ++n;
        }
      }
    }
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

// ===========================================================================
// ncclRedOpCreatePreMulSum_impl (enqueue.cc:4103) / ncclRedOpDestroy_impl (:4148)
//
// A create/destroy PAIR over a free list embedded in comm->userRedOps, with
// capacity doubling. Testing them together is what makes the free list
// observable: create -> destroy -> create must REUSE the slot, which no
// single-call test can show.
//
// Public API surface (ncclRedOpCreatePreMulSum / ncclRedOpDestroy).
// ===========================================================================

namespace {
// A comm that passes CommCheck + ncclCommEnsureReady. The real argcheck.cc is
// compiled in as an oracle, so these fields must genuinely satisfy it.
struct RedOpComm {
  std::unique_ptr<ncclComm> storage{new ncclComm{}};
  RedOpComm() {
    ncclComm& c = *storage;
    // CommCheck (argcheck.cc:40) validates startMagic AND endMagic -- the
    // sentinels bracketing the struct, not a single `magic` field. The real
    // argcheck.cc is compiled in as an oracle, so both must be set or every
    // create is rejected with "corrupted comm object detected".
    c.startMagic = NCCL_MAGIC;
    c.endMagic = NCCL_MAGIC;
    c.rank = 0;
    c.nRanks = 8;
    c.userRedOps = nullptr;
    c.userRedOpCapacity = 0;
    c.userRedOpFreeHead = 0;
    c.initState = ncclSuccess;   // ncclCommEnsureReady: already initialised
    c.finalizeCalled = 0;
  }
  ~RedOpComm() { delete[] storage->userRedOps; }
  ncclComm* get() { return storage.get(); }
};
}  // namespace

TEST_F(EnqueueMicrotest, RedOpCreate_FirstCall_GrowsCapacityToFour) {
  // `if (cap < 4) cap = 4` -- the first allocation jumps straight to 4, it does
  // not double from 0.
  RedOpComm rc;
  ncclRedOp_t op = ncclSum;
  float scalar = 2.0f;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &scalar, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  EXPECT_EQ(4, rc.get()->userRedOpCapacity);
  EXPECT_NE(nullptr, rc.get()->userRedOps);
}

TEST_F(EnqueueMicrotest, RedOpCreate_ReturnsOpAboveBuiltinRange) {
  // The handle is ncclNumOps + slot index, so it must never collide with a
  // builtin op -- that separation is what ncclRedOpDestroy's first guard relies on.
  RedOpComm rc;
  ncclRedOp_t op = ncclSum;
  float scalar = 2.0f;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &scalar, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  EXPECT_GE(int(op), int(ncclNumOps));
}

TEST_F(EnqueueMicrotest, RedOpCreate_HostImmediate_CopiesScalarBytesInline) {
  // residence == HostImmediate memcpy's ncclTypeSize(datatype) bytes INTO
  // scalarArg. Compare the bit pattern, and assert the pointer flag is clear --
  // a mutant that stored the pointer instead would otherwise look similar.
  RedOpComm rc;
  ncclRedOp_t op = ncclSum;
  float scalar = 0.25f;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &scalar, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  const int ix = int(ncclUserRedOpMangle(rc.get(), op)) - int(ncclNumOps);
  const ncclUserRedOp& u = rc.get()->userRedOps[ix];
  EXPECT_EQ(ncclDevPreMulSum, u.opFull.op);
  EXPECT_FALSE(u.opFull.scalarArgIsPtr);
  uint32_t want;
  std::memcpy(&want, &scalar, sizeof(want));
  EXPECT_EQ(want, uint32_t(u.opFull.scalarArg & 0xFFFFFFFFull));
  EXPECT_EQ(ncclFloat32, u.datatype);
}

TEST_F(EnqueueMicrotest, RedOpCreate_DeviceResidence_StoresPointerNotBytes) {
  // The other residence arm: scalarArg holds the POINTER and the flag is set.
  RedOpComm rc;
  ncclRedOp_t op = ncclSum;
  double scalar = 3.5;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &scalar, ncclFloat64,
                                                       ncclScalarDevice, rc.get()));
  const int ix = int(ncclUserRedOpMangle(rc.get(), op)) - int(ncclNumOps);
  const ncclUserRedOp& u = rc.get()->userRedOps[ix];
  EXPECT_TRUE(u.opFull.scalarArgIsPtr);
  EXPECT_EQ(reinterpret_cast<uint64_t>(&scalar), u.opFull.scalarArg);
}

TEST_F(EnqueueMicrotest, RedOpCreate_MarksSlotAllocatedViaFreeNext) {
  // freeNext == -1 is the "allocated" marker Destroy checks. Pin it explicitly.
  RedOpComm rc;
  ncclRedOp_t op = ncclSum;
  float s = 1.0f;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  const int ix = int(ncclUserRedOpMangle(rc.get(), op)) - int(ncclNumOps);
  EXPECT_EQ(-1, rc.get()->userRedOps[ix].freeNext);
}

TEST_F(EnqueueMicrotest, RedOpCreate_FourOpsFitThenCapacityDoubles) {
  // Exhaust the initial capacity of 4, then the 5th create must double to 8 and
  // PRESERVE the existing four (the memcpy at :4115). Asserting the handles stay
  // distinct is what catches a dropped or short memcpy.
  RedOpComm rc;
  float s = 1.0f;
  std::vector<ncclRedOp_t> ops;
  for (int i = 0; i < 4; ++i) {
    ncclRedOp_t op = ncclSum;
    ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                                         ncclScalarHostImmediate, rc.get()))
        << "i=" << i;
    ops.push_back(op);
  }
  EXPECT_EQ(4, rc.get()->userRedOpCapacity);

  ncclRedOp_t fifth = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&fifth, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  EXPECT_EQ(8, rc.get()->userRedOpCapacity) << "capacity must DOUBLE, not reset";
  ops.push_back(fifth);

  // Every handle distinct, and every slot still marked allocated.
  std::vector<int> seen;
  for (auto op : ops) {
    int ix = int(ncclUserRedOpMangle(rc.get(), op)) - int(ncclNumOps);
    EXPECT_EQ(-1, rc.get()->userRedOps[ix].freeNext) << "slot " << ix << " lost its marker";
    seen.push_back(ix);
  }
  std::sort(seen.begin(), seen.end());
  EXPECT_EQ(seen.end(), std::unique(seen.begin(), seen.end())) << "handles must be distinct";
}

TEST_F(EnqueueMicrotest, RedOpCreateDestroy_RoundTrip_ReusesTheFreedSlot) {
  // THE point of the pair: destroy pushes the slot onto the free list, so the
  // next create pops the SAME slot. A destroy that forgot to update
  // userRedOpFreeHead would leak the slot and hand back a different index.
  RedOpComm rc;
  float s = 1.0f;
  ncclRedOp_t first = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&first, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  const int ixFirst = int(ncclUserRedOpMangle(rc.get(), first)) - int(ncclNumOps);

  ASSERT_EQ(ncclSuccess, ncclRedOpDestroy_impl(first, rc.get()));
  EXPECT_NE(-1, rc.get()->userRedOps[ixFirst].freeNext) << "destroy must clear the marker";

  ncclRedOp_t second = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&second, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  const int ixSecond = int(ncclUserRedOpMangle(rc.get(), second)) - int(ncclNumOps);
  EXPECT_EQ(ixFirst, ixSecond) << "the freed slot must be reused";
  EXPECT_EQ(4, rc.get()->userRedOpCapacity) << "and no growth should have been needed";
}

TEST_F(EnqueueMicrotest, RedOpDestroy_BuiltinOp_RejectedWithWarning) {
  // Guard 1: builtin ops are not destroyable. Assert the WARN too -- the return
  // code alone cannot distinguish this guard from the three below it.
  RedOpComm rc;
  for (auto op : {ncclSum, ncclProd, ncclMin, ncclMax, ncclAvg}) {
    ncclResult_t r = ncclSuccess;
    const std::string log = RcclUnitTesting::CaptureLog([&] {
      r = ncclRedOpDestroy_impl(op, rc.get());
    });
    EXPECT_EQ(ncclInvalidArgument, r) << "op=" << int(op);
    EXPECT_TRUE(RcclUnitTesting::LogHas(log, "operator is a NCCL builtin"))
        << "op=" << int(op) << " actual log:\n" << log;
  }
}

TEST_F(EnqueueMicrotest, RedOpDestroy_NullComm_RejectedWithWarning) {
  // Guard 3. Reached only by an op ABOVE the builtin range, so this also proves
  // guard 1 did not swallow it.
  RedOpComm rc;
  float s = 1.0f;
  ncclRedOp_t op = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  ncclResult_t r = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    r = ncclRedOpDestroy_impl(op, nullptr);
  });
  EXPECT_EQ(ncclInvalidArgument, r);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "invalid communicator")) << "actual log:\n" << log;
}

TEST_F(EnqueueMicrotest, RedOpDestroy_UnknownToThisComm_RejectedWithWarning) {
  // Guard 4, first half: an index past this comm's capacity.
  RedOpComm rc;
  float s = 1.0f;
  ncclRedOp_t op = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  // Fabricate a handle far beyond the 4 allocated slots.
  auto bogus = static_cast<ncclRedOp_t>(int(op) + 1000);
  ncclResult_t r = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    r = ncclRedOpDestroy_impl(bogus, rc.get());
  });
  EXPECT_EQ(ncclInvalidArgument, r);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "operator unknown to this communicator"))
      << "actual log:\n" << log;
}

TEST_F(EnqueueMicrotest, RedOpDestroy_DoubleFree_RejectedWithWarning) {
  // Guard 4, second half: freeNext != -1 means the slot is already on the free
  // list. This is the double-free rejection and it shares a message with the
  // out-of-range case, so the two tests together pin both halves of the ||.
  RedOpComm rc;
  float s = 1.0f;
  ncclRedOp_t op = ncclSum;
  ASSERT_EQ(ncclSuccess, ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                                       ncclScalarHostImmediate, rc.get()));
  ASSERT_EQ(ncclSuccess, ncclRedOpDestroy_impl(op, rc.get()));

  ncclResult_t r = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    r = ncclRedOpDestroy_impl(op, rc.get());
  });
  EXPECT_EQ(ncclInvalidArgument, r) << "second destroy must be rejected";
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "operator unknown to this communicator"))
      << "actual log:\n" << log;
}

TEST_F(EnqueueMicrotest, RedOpCreate_NullComm_RejectedByCommCheck) {
  // The real argcheck.cc oracle is compiled in, so this exercises production's
  // own validation rather than a fake's.
  ncclRedOp_t op = ncclSum;
  float s = 1.0f;
  EXPECT_EQ(ncclInvalidArgument,
            ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                          ncclScalarHostImmediate, nullptr));
}

// ===========================================================================
// rcclEffectiveP2pBatchEnable (enqueue.cc:1169)
// Four arms gating P2P batching for the whole file. The user override is
// checked FIRST, so it must win over both the node count and the arch.
// ===========================================================================

namespace {
// One place that stamps a topology's single GPU node with an arch string. Both
// fixtures below set the same pair of fields, and a copy that forgets the count
// leaves the gcn string unreachable rather than failing.
void SetSingleGpuArch(ncclTopoSystem* topo, const char* gcn) {
  topo->nodes[GPU].count = 1;
  std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn,
                sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "%s", gcn);
}

struct BatchComm {
  std::unique_ptr<ncclTopoSystem> topo{new ncclTopoSystem{}};
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  BatchComm(int nNodes, const char* gcn) {
    comm->nNodes = nNodes;
    comm->topo = topo.get();
    SetSingleGpuArch(topo.get(), gcn);
  }
  ncclComm* get() { return comm.get(); }
};

// One param setter for the whole file: three GetImplicitOrder tests below used
// to re-roll this same lambda with a different key, which risks the copies
// drifting on the `deft` fallback.
void SetParam(const char* key, int64_t v) {
  std::string k(key);
  g_loadParam = [k, v](const char* env, int64_t deft) {
    return std::strcmp(env, k.c_str()) == 0 ? v : deft;
  };
}
void SetBatchParam(int64_t v) { SetParam("RCCL_P2P_BATCH_ENABLE", v); }
}  // namespace

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_UserOverrideWinsOverEverything) {
  // userInput >= 0 returns immediately -- single node and a non-gfx950 arch,
  // both of which would otherwise force 0.
  BatchComm bc(/*nNodes=*/1, "gfx942");
  SetBatchParam(1);
  EXPECT_EQ(1, rcclEffectiveP2pBatchEnable(bc.get()));
  SetBatchParam(0);
  EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get()));
  SetBatchParam(7);
  EXPECT_EQ(7, rcclEffectiveP2pBatchEnable(bc.get())) << "returned verbatim, not clamped";
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_SingleNode_IsDisabled) {
  // With the override off (-1), nNodes <= 1 short-circuits to 0 before the arch
  // check -- so even gfx950 gives 0 here.
  BatchComm bc(/*nNodes=*/1, "gfx950");
  SetBatchParam(-1);
  EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get()));
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_MultiNodeGfx950_IsEnabled) {
  // The real gfx950 arm. IsArchMatch comes from the archinfo.cc oracle, so this
  // is production's own matching, not a fake's.
  BatchComm bc(/*nNodes=*/2, "gfx950");
  SetBatchParam(-1);
  EXPECT_EQ(1, rcclEffectiveP2pBatchEnable(bc.get()));
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_Gfx950WithAinic_IsDisabled) {
  // The `!rcclUseAinic()` conjunct -- the reason g_rcclUseAinic is a seam in
  // fakes/transport_stubs.cc rather than the fail-loud stub it used to be.
  // Without it the flag stays false in every test and dropping the conjunct
  // survives. Differential with MultiNodeGfx950_IsEnabled, which is identical
  // but for the AINIC flag.
  BatchComm bc(/*nNodes=*/2, "gfx950");
  SetBatchParam(-1);
  g_rcclUseAinic = true;
  EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get()))
      << "gfx950 with AINIC must not enable p2p batching";
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_MultiNodeOtherArch_IsDisabled) {
  // Differential against the test above: same node count, different arch.
  for (const char* gcn : {"gfx942", "gfx90a", "gfx1100"}) {
    BatchComm bc(/*nNodes=*/2, gcn);
    SetBatchParam(-1);
    EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get())) << "gcn=" << gcn;
  }
}

// ===========================================================================
// getImplicitOrder (enqueue.cc:1996)
// On AMD the CUDA driver-version arm is #if'd out entirely, so only two arms
// are reachable: param-on -> Serial, param-off -> None. Pinning that the AMD
// build cannot return ncclImplicitOrderLaunch is the useful assertion.
// ===========================================================================

TEST_F(EnqueueMicrotest, GetImplicitOrder_ParamDisabled_IsNone) {
  SetParam("LAUNCH_ORDER_IMPLICIT", 0);
  auto mode = ncclImplicitOrderLaunch;  // poison
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&mode, /*capturing=*/false));
  EXPECT_EQ(ncclImplicitOrderNone, mode);
}

TEST_F(EnqueueMicrotest, GetImplicitOrder_ParamEnabled_IsSerialOnAmd) {
  SetParam("LAUNCH_ORDER_IMPLICIT", 1);
  auto mode = ncclImplicitOrderNone;  // poison
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&mode, /*capturing=*/false));
  EXPECT_EQ(ncclImplicitOrderSerial, mode);
}

TEST_F(EnqueueMicrotest, GetImplicitOrder_CapturingIsIrrelevantOnAmd) {
  // HONEST SCOPE: this pins that `capturing` does not change the answer; it does
  // NOT prove the AMD arm is what produced it. Under the seam's driver 12000 the
  // CUDA arm returns Serial for both values too (:2002 12000 < 12090; :2006
  // 12030 <= min(CUDART, 12000) is false), so an #if change would not fail here.
  // getImplicitOrder's third parameter (driver, :1996) is what separates the arms.
  SetParam("LAUNCH_ORDER_IMPLICIT", 1);
  auto a = ncclImplicitOrderNone;
  auto b = ncclImplicitOrderNone;
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&a, /*capturing=*/true));
  ASSERT_EQ(ncclSuccess, getImplicitOrder(&b, /*capturing=*/false));
  EXPECT_EQ(a, b);
  EXPECT_EQ(ncclImplicitOrderSerial, a);
}

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
    addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeBcast, 7, uint32_t(i * bcastSize));
  }
  const int beforeCap = bp.queueLength();

  // The (maxitem + 1)th item is the first to see nBcasts == maxitem.
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeBcast, 7, uint32_t(maxitem * bcastSize));
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
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, /*p2pRound=*/0, true);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeBcast, 7, uint32_t(bcastSize));
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
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, 0, 0, /*batchP2P=*/false);
  addWorkBatchToPlan(bp.c(), bp.p(), 0, ncclDevWorkTypeP2p, 7, uint32_t(ws), 1,
                     /*batchP2P=*/true);
  EXPECT_EQ(2, bp.queueLength()) << "an ineligible batch must not absorb an eligible op";
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

// ===========================================================================
// finishPlan (enqueue.cc:294) -- packs the per-channel work batches into the
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
  // Holds a BatchPlanComm rather than repeating the comm/plan pair and the
  // memory-stack lifetime. That also picks up its memPermanent construct, which
  // is inert for these tests but load-bearing for anything reaching
  // addProxyOpIfNeeded -- the weaker of two near-identical fixtures is where a
  // future test lands by accident.
  //
  // (0, 0) reproduces what `new ncclComm{}` gave this fixture before: these tests
  // were written against a zero nNodes and cudaArch, not BatchPlanComm's 1/942.
  BatchPlanComm base{0, 0};
  std::vector<std::unique_ptr<ncclProxyOp>> ops;   // keep proxy ops alive

  FinishComm() {
    c()->workArgsBytes = 1 << 16;   // generous: the args-storage arm is taken
  }

  ncclComm* c() { return base.c(); }
  ncclKernelPlan* p() { return base.p(); }
  ncclKernelPlanner::WipPlan::Channel* chan(int id) { return base.chan(id); }
  void markChannel(int c) { p()->channelMask.masks[c / 64] |= (1ULL << (c % 64)); }

  // Queue n work batches on channel c, each stamped with a recognisable funcId.
  void addBatches(int c, int n, int funcIdBase) {
    for (int i = 0; i < n; ++i) {
      addWorkBatchToPlan(this->c(), p(), c, ncclDevWorkTypeColl,
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
    return (ncclDevWorkBatch*)(p()->kernelArgs + 1);
  }
  std::vector<uint64_t> mergedOpCounts() {
    std::vector<uint64_t> out;
    // The intrusive link for proxyOpQueue is `enqNext` (comm.h:386), not `next`.
    for (auto* op = ncclIntruQueueHead(&p()->proxyOpQueue); op != nullptr; op = op->enqNext) {
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
  // The kernel reads storage type off kernelArgs, so the copy at :316 must carry
  // it. Non-zero here, so unlike the small-plan case this can see a dropped store.
  ASSERT_NE(nullptr, f.p()->kernelArgs);
  EXPECT_EQ(ncclDevWorkStorageTypeFifo, f.p()->kernelArgs->workStorageType);
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
  // HONEST SCOPE: workStorageType is NOT checked here. On this path it is
  // ncclDevWorkStorageTypeArgs, enumerator 0 (device.h:650), and kernelArgs comes
  // from ncclMemoryStackAlloc (:313) which memsets to 0 (utils.h:283), so the
  // comparison is 0 == 0 and deleting the store at :316 kills nothing. The
  // oversized-plan test below stamps a non-zero value and does check it.
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

// ===========================================================================
// waitWorkFifoAvailable (enqueue.cc:1651)
//
// The room test is UNSIGNED WRAPAROUND arithmetic:
//     (desiredProduced - workFifoConsumed) <= workFifoBytes
// Both operands are uint32_t, so it stays correct as the counters roll past
// 2^32 -- which is exactly the case a naive `desired <= consumed + bytes`
// rewrite would break. Several tests below sit deliberately at the rollover.
// ===========================================================================

namespace {
struct FifoComm {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  uint32_t abortStorage = 0;
  FifoComm(uint32_t consumed, uint32_t bytes) {
    comm->workFifoConsumed = consumed;
    comm->workFifoBytes = bytes;
    comm->rank = 0;
    comm->abortFlag = &abortStorage;
  }
  ncclComm* c() { return comm.get(); }
};
}  // namespace

TEST_F(EnqueueMicrotest, WaitWorkFifo_RoomAvailable_ReturnsImmediately) {
  // The fast path. hasRoom is true on entry so the loop never runs -- proven by
  // arming the poll to FAIL: reaching it at all would surface as an error.
  FifoComm f(/*consumed=*/1000, /*bytes=*/4096);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclSuccess, waitWorkFifoAvailable(f.c(), /*desiredProduced=*/2000));
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_ExactlyFull_IsStillRoom) {
  // `<=`, not `<`: a difference of exactly workFifoBytes fits. Same failing-poll
  // trick pins that the boundary is decided WITHOUT entering the loop.
  FifoComm f(/*consumed=*/0, /*bytes=*/4096);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclSuccess, waitWorkFifoAvailable(f.c(), /*desiredProduced=*/4096));
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_OneByteOver_EntersTheWaitLoop) {
  // One byte past the boundary must enter the loop. Observable because the
  // armed poll failure propagates -- the differential against the test above,
  // which differs by exactly one byte.
  FifoComm f(/*consumed=*/0, /*bytes=*/4096);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclUnhandledCudaError, waitWorkFifoAvailable(f.c(), /*desiredProduced=*/4097));
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_AbortFlagSet_BreaksTheDeadlock) {
  // Without this arm a full FIFO plus a dead consumer is an infinite loop. The
  // abort check runs BEFORE the poll, so the armed poll failure must NOT win.
  FifoComm f(/*consumed=*/0, /*bytes=*/16);
  f.abortStorage = 1;
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclInternalError, waitWorkFifoAvailable(f.c(), /*desiredProduced=*/1 << 20))
      << "abort must be checked before the poll";
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_PollFailure_Propagates) {
  // NCCLCHECK on the poll: a failing poll must abandon the wait rather than spin.
  FifoComm f(/*consumed=*/0, /*bytes=*/16);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclUnhandledCudaError, waitWorkFifoAvailable(f.c(), /*desiredProduced=*/1 << 20));
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_CountersWrappedPastUint32_StillComputesRoom) {
  // THE reason the subtraction is written the way it is. consumed sits near 2^32
  // and desired has already wrapped to 0; the unsigned difference is still the
  // true distance, so there IS room. A rewrite to `desired <= consumed + bytes`
  // would overflow and wrongly enter the loop -- which the armed poll failure
  // would then expose as an error return.
  FifoComm f(/*consumed=*/0xFFFFF000u, /*bytes=*/4096);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  const uint32_t desired = 0xFFFFF000u + 4096u;   // wraps to 0x00000000
  EXPECT_EQ(ncclSuccess, waitWorkFifoAvailable(f.c(), desired));
}

TEST_F(EnqueueMicrotest, WaitWorkFifo_WrappedAndOneByteOver_StillBlocks) {
  // Differential with the test above: same wrapped region, one byte too far.
  // Together they pin that wraparound is handled without becoming permissive.
  FifoComm f(/*consumed=*/0xFFFFF000u, /*bytes=*/4096);
  g_hipAsyncOpsResult = hipErrorInvalidValue;
  const uint32_t desired = 0xFFFFF000u + 4097u;
  EXPECT_EQ(ncclUnhandledCudaError, waitWorkFifoAvailable(f.c(), desired));
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
// addProxyOpIfNeeded (enqueue.cc:198) / ncclAddProxyOpIfNeeded (:771)
// A proxy op is only recorded when the transport says it is needed; the
// "justInquire" protocol means ncclProxySaveOp is asked FIRST whether the op
// matters, and only then is memory allocated for it.
// ===========================================================================

namespace {
// addProxyOpIfNeeded enqueues to the op's CHANNEL queue
// (comm->planner.wipPlan.channels[op->channelId].proxyOpQueue) -- NOT to
// plan->proxyOpQueue. finishPlan is what later merges the channel queues into
// the plan. Looking in the wrong queue makes every one of these read "empty".
ncclProxyOp* ChannelProxyHead(BatchPlanComm& bp, int channelId) {
  return ncclIntruQueueHead(&bp.chan(channelId)->proxyOpQueue);
}
bool ChannelProxyEmpty(BatchPlanComm& bp, int channelId) {
  return ncclIntruQueueEmpty(&bp.chan(channelId)->proxyOpQueue);
}
}  // namespace

TEST_F(EnqueueMicrotest, AddProxyOpIfNeeded_NotNeeded_DoesNotEnqueue) {
  // justInquire comes back false -> nothing is queued, but the inquiry still
  // happened (the call counter is what distinguishes this from "never asked").
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 0;
  g_proxySaveOpJustInquire = false;
  ASSERT_EQ(ncclSuccess, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_EQ(1, g_proxySaveOpCalls);
  EXPECT_TRUE(ChannelProxyEmpty(bp, 0));
}

TEST_F(EnqueueMicrotest, AddProxyOpIfNeeded_Needed_EnqueuesACopy) {
  // justInquire true -> the op is copied into comm-owned memory and queued. The
  // copy must be a DISTINCT object, since the caller's op is a stack temporary
  // that dies before the plan is launched.
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 0;
  op.opCount = 0xABCD;
  g_proxySaveOpJustInquire = true;
  ASSERT_EQ(ncclSuccess, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_EQ(1, g_proxySaveOpCalls);

  // The INQUIRY must be about this comm and this op. The queued copy below is
  // made from the caller's op regardless, so without these the transport could
  // be handed a different comm/op and the test would still pass.
  EXPECT_EQ(bp.c(), g_proxySaveOpLastComm) << "must inquire with the caller's comm";
  EXPECT_EQ(&op, g_proxySaveOpLastOp) << "must inquire about the caller's op";
  EXPECT_EQ(uint64_t(0xABCD), g_proxySaveOpLastOpCount);

  auto* head = ChannelProxyHead(bp, 0);
  ASSERT_NE(nullptr, head);
  EXPECT_NE(&op, head) << "must copy, not alias the caller's stack object";
  EXPECT_EQ(uint64_t(0xABCD), head->opCount) << "and the copy must carry the contents";
}

TEST_F(EnqueueMicrotest, AddProxyOpIfNeeded_InquiresBeforeAllocating) {
  // The protocol is "ask first, allocate second", and what signals inquiry is
  // passing a NON-NULL justInquire pointer -- not its pointee. Real
  // ncclProxySaveOp overwrites it with false as its first statement (proxy.cc:631),
  // so the `bool needed = true` seed at :199 is never read and asserting on the
  // incoming value would pin our fake rather than production.
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 2;
  op.opCount = 0x77;
  g_proxySaveOpJustInquire = false;
  ASSERT_EQ(ncclSuccess, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_TRUE(g_proxySaveOpSawNonNullJustInquire)
      << "production must ask (non-null justInquire), not tell";
  EXPECT_EQ(2, g_proxySaveOpLastChannelId) << "and must inquire about the op's own channel";
}

TEST_F(EnqueueMicrotest, AddProxyOpIfNeeded_RoutesToTheOpsOwnChannel) {
  // The destination is indexed by op->channelId. A hardcoded channel 0 would
  // pass every other test here but fail this one.
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 3;
  op.opCount = 0x1234;
  g_proxySaveOpJustInquire = true;
  ASSERT_EQ(ncclSuccess, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_EQ(3, g_proxySaveOpLastChannelId) << "the inquiry carries the op's channel";
  EXPECT_TRUE(ChannelProxyEmpty(bp, 0)) << "must not land on channel 0";
  auto* head = ChannelProxyHead(bp, 3);
  ASSERT_NE(nullptr, head);
  EXPECT_EQ(uint64_t(0x1234), head->opCount);
}

TEST_F(EnqueueMicrotest, AddProxyOpIfNeeded_SaveOpFailure_Propagates) {
  // NCCLCHECK on the inquiry: a failure must abandon the add, not queue a
  // half-initialised op.
  BatchPlanComm bp;
  ncclProxyOp op{};
  op.channelId = 0;
  g_proxySaveOpResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, ncclAddProxyOpIfNeeded(bp.c(), bp.p(), &op));
  EXPECT_TRUE(ChannelProxyEmpty(bp, 0));
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
  // Holds a CostComm for the comm/topo pair, the rank counts and the XGMI_ALL
  // stamp; adds only what topoGetAlgoInfo needs beyond updateCollCostTable.
  //
  // NOT parameterised, deliberately, even though topoGetAlgoInfo reads nRanks and
  // nNodes. CostComm hardcodes maxLocalRanks and localRanks to 8 regardless of
  // nRanks (:1387-1392), so CostComm(64, 8) would describe 64 ranks over 8 nodes
  // with 8 local ranks and one GPU node -- a comm that cannot exist. Exposing a
  // pair of parameters that produce an incoherent fixture is worse than the fixed
  // 8/1 here. Varying either one needs CostComm to derive the local counts first,
  // and that changes what the existing CostComm tests cover (e.g. :1582).
  CostComm base{8, 1};
  AlgoInfoComm() {
    ncclComm* comm = base.get();
    comm->WarpSize = 32;
    SetSingleGpuArch(&base.topo(), "gfx942");
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; ++a) {
      for (int p = 0; p < NCCL_NUM_PROTOCOLS; ++p) {
        comm->maxThreads[a][p] = 256;
      }
    }
  }
  ncclComm* get() { return base.get(); }
};
}  // namespace

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

// ===========================================================================
// Seams whose comments promised a tested rejection path. Driving them here so
// the promise holds; the remaining declared-but-undriven seams are marked in
// enqueue_fakes.h as link-floor-only rather than left to imply coverage.
// ===========================================================================

TEST_F(EnqueueMicrotest, RedOpCreate_CommNotReady_IsRejected) {
  // g_commEnsureReadyResult is documented as "controllable so a test can
  // exercise the not-ready rejection" -- this is that test. RedOpCreate joins
  // the init thread through ncclCommEnsureReady before touching the free list,
  // so a not-ready comm must be rejected with nothing allocated.
  // The code is ncclInternalError, not the ncclInvalidArgument the preceding
  // CommCheck also returns, so a regressed RedOpComm magic cannot fake a pass.
  RedOpComm rc;
  g_commEnsureReadyResult = ncclInternalError;
  ncclRedOp_t op = ncclSum;
  float s = 1.0f;
  EXPECT_EQ(ncclInternalError,
            ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                          ncclScalarHostImmediate, rc.get()));
  EXPECT_EQ(0, rc.get()->userRedOpCapacity) << "must reject before growing the free list";
}

TEST_F(EnqueueMicrotest, RedOpCreate_RecorderFailure_Propagates) {
  // Production NCCLCHECKs the recorder call at the end of RedOpCreate, which is
  // why the seam exists. Note the contrast with init.cc:369, where the identical
  // call is NOT checked -- pinning this one keeps that asymmetry visible.
  RedOpComm rc;
  g_recorderResult = ncclInternalError;
  ncclRedOp_t op = ncclSum;
  float s = 1.0f;
  EXPECT_EQ(ncclInternalError,
            ncclRedOpCreatePreMulSum_impl(&op, &s, ncclFloat32,
                                          ncclScalarHostImmediate, rc.get()));
}
