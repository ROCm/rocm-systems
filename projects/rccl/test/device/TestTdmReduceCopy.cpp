/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for the TDM reduce and multi-destination copy paths of reduceCopy (reduceCopyTdm in tdm/tdmReduce.h).
//
// Each kernel drives reduceCopy from one block the way genericOp does: the pointer
// lists live in shared memory and EnableTdmReduce is set. Every case runs twice, with
// the runtime switch on and off, and both outputs must match a host reference exactly,
// so the TDM tiles and the vector passes that finish each slice are checked together.
// The CopyTdmOnly cases call reduceCopyTdm alone, so only the mover can have written their output.
// Inputs are small multiples of 1/4, so float sums are exact in any order.

#include "DeviceTestBase.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// Gives this TU its own ncclShmem definition, as each shard of librccl's device-linker build has.
#ifndef RCCL_DEVICE_LINKER
#define RCCL_DEVICE_LINKER
#endif
// Counts the copies reduceCopyTdm takes (rcclTdmReduceTestCopies), so a case can tell it from tryTdmCopy.
#define RCCL_TDM_REDUCE_TEST_HOOKS
#include "common_kernel.h"

namespace RcclUnitTesting
{

// NCCL_MAX_NTHREADS: eight 32-lane warps on gfx1250.
constexpr int kBlockThreads = 256;
// reduceCopyTdm's tile on gfx1250, half of the 16 KiB per-warp stage. The host pass cannot see the device constant.
constexpr size_t kTdmTileBytes = 8192;
// Elements of sentinel padding on both sides of every destination window; keeps 256-byte alignment for 4-byte types.
constexpr size_t kGuardElts = 64;

template<typename RedOp, typename T, int NSrcs, int NDsts, int PreOpSrcs>
__global__ void __launch_bounds__(kBlockThreads)
  kReduceCopyTdm(T* const* srcs, T* const* dsts, ssize_t nElts, uint64_t redArg, bool postOp,
                 [[maybe_unused]] int tdmCopy, [[maybe_unused]] int tdmReduce) {
  __shared__ void* srcPtrs[NSrcs];
  __shared__ void* dstPtrs[NDsts];
  if (threadIdx.x == 0) {
#if ENABLE_TDM_SIMPLE
    ncclShmem.comm.tdmSimpleEnable = tdmCopy;
    ncclShmem.comm.tdmSimpleReduceEnable = tdmReduce;
#endif
    for (int s = 0; s < NSrcs; s++) srcPtrs[s] = srcs[s];
    for (int d = 0; d < NDsts; d++) dstPtrs[d] = dsts[d];
  }
  __syncthreads();
  reduceCopy<4, /*useAcc=*/0, RedOp, T, /*MultimemSrcs=*/0, NSrcs, NSrcs, /*MultimemDsts=*/0, NDsts, NDsts, PreOpSrcs,
             /*Pipeline=*/0, /*EnableTdmReduce=*/1>(threadIdx.x, blockDim.x, redArg, postOp, NSrcs, srcPtrs, NDsts,
                                                    dstPtrs, nElts);
}

// Writes canUseTdmReduce's answer for the same block and pointers as kReduceCopyTdm, or -1 when the device code has no
// TDM reduce path. Equal outputs alone cannot show that the TDM tiles ran, since the vector path computes the same.
template<typename RedOp, typename T, int NSrcs, int NDsts, int PreOpSrcs>
__global__ void __launch_bounds__(kBlockThreads)
  kTdmReduceGate(T* const* srcs, T* const* dsts, ssize_t nElts, bool postOp, [[maybe_unused]] int tdmCopy,
                 [[maybe_unused]] int tdmReduce, int* gate) {
  __shared__ void* srcPtrs[NSrcs];
  __shared__ void* dstPtrs[NDsts];
  if (threadIdx.x == 0) {
#if ENABLE_TDM_SIMPLE
    ncclShmem.comm.tdmSimpleEnable = tdmCopy;
    ncclShmem.comm.tdmSimpleReduceEnable = tdmReduce;
#endif
    for (int s = 0; s < NSrcs; s++) srcPtrs[s] = srcs[s];
    for (int d = 0; d < NDsts; d++) dstPtrs[d] = dsts[d];
  }
  __syncthreads();
#if TDM_SUPPORTED && ENABLE_TDM_SIMPLE
  void** srcList = srcPtrs;
  void** dstList = dstPtrs;
  const bool admitted = canUseTdmReduce<RedOp, PreOpSrcs>(
    threadIdx.x, blockDim.x, postOp, NSrcs, [=] __device__(int i) { return srcList[i]; }, NDsts,
    [=] __device__(int i) { return dstList[i]; }, nElts * static_cast<ssize_t>(sizeof(T)));
  if (threadIdx.x == 0) *gate = admitted ? 1 : 0;
#else
  if (threadIdx.x == 0) *gate = -1;
#endif
}

// Runs reduceCopyTdm on a copy without reduceCopy's vector passes, so only the mover writes the output. Writes the
// bytes left for the vector path, or -1 when the device code has no TDM path.
template<typename T, int NDsts>
__global__ void __launch_bounds__(kBlockThreads)
  kCopyTdmOnly(T* src, T* const* dsts, ssize_t nElts, ssize_t* left) {
#if TDM_SUPPORTED && ENABLE_TDM_SIMPLE
  __shared__ void* dstPtrs[NDsts];
  if (threadIdx.x == 0)
    for (int d = 0; d < NDsts; d++) dstPtrs[d] = dsts[d];
  __syncthreads();
  void** dstList = dstPtrs;
  int thread = threadIdx.x;
  ssize_t behind = 0, ahead = nElts * static_cast<ssize_t>(sizeof(T));
  reduceCopyTdm<FuncSum<T>, T, /*MaxSrcs=*/1, NDsts, /*PreOpSrcs=*/0>(
    blockDim.x, thread, /*redArg=*/0, /*postOp=*/false, /*nSrcs=*/1,
    [=] __device__(int) { return static_cast<void*>(src); }, NDsts, [=] __device__(int i) { return dstList[i]; },
    behind, ahead);
  if (threadIdx.x == 0) *left = ahead;
#else
  if (threadIdx.x == 0) *left = -1;
#endif
}

class TdmReduceCopyTest : public DeviceTestBase {
protected:
  bool supported_ = false;

  void SetUp() override {
    DeviceTestBase::SetUp();
    supported_ = tdm::IsTdmCopySupported(0);
  }

  // Elements in `tilesPerWarp` whole TDM tiles for every warp of the block.
  template<typename T>
  static size_t tileElts(size_t tilesPerWarp) {
    return tilesPerWarp * (kBlockThreads / 32) * kTdmTileBytes / sizeof(T);
  }

  template<typename T>
  static T value(size_t i, int s);

  // Checks that `out` holds `expected` from `start` on and the sentinel everywhere else.
  template<typename T>
  static void checkDst(const std::vector<T>& out, size_t start, const std::vector<T>& expected, const T& sentinel,
                       const std::string& where) {
    for (size_t i = 0; i < expected.size(); i++) {
      ASSERT_EQ(0, std::memcmp(&out[start + i], &expected[i], sizeof(T)))
        << where << ": mismatch at element " << i << " of " << expected.size();
    }
    for (size_t i = 0; i < out.size(); i++) {
      if (i >= start && i < start + expected.size()) continue;
      ASSERT_EQ(0, std::memcmp(&out[i], &sentinel, sizeof(T))) << where << ": guard overwritten at index " << i;
    }
  }

  // Runs one reduceCopy shape with a TDM switch on and off against `ref`, and checks the TDM gate and, for copies, that
  // reduceCopyTdm took the slice exactly when `expectTdm`. The switch is RCCL_TDM_SIMPLE_REDUCE_ENABLE, or
  // RCCL_TDM_SIMPLE_ENABLE with `copySwitch`. `offElts` offsets every pointer by that many elements.
  template<typename RedOp, typename T, int NSrcs, int NDsts, int PreOpSrcs>
  void run(size_t nElts, size_t offElts, uint64_t redArg, bool postOp, bool expectTdm,
           const std::function<T(const std::vector<std::vector<T>>&, size_t)>& ref, const std::string& tag,
           bool copySwitch = false) {
#if !ENABLE_TDM_SIMPLE
    GTEST_SKIP() << "Built without ENABLE_TDM_SIMPLE";
#endif
    if (!supported_) GTEST_SKIP() << "TDM not supported on this device";

    const size_t srcTotal = offElts + nElts;
    const size_t dstTotal = kGuardElts + offElts + nElts + kGuardElts;
    const size_t dstStart = kGuardElts + offElts;

    std::vector<std::vector<T>> hSrc(NSrcs, std::vector<T>(srcTotal));
    for (int s = 0; s < NSrcs; s++)
      for (size_t i = 0; i < srcTotal; i++) hSrc[s][i] = value<T>(i, s);

    std::vector<std::vector<T>> windows(NSrcs, std::vector<T>(nElts));
    for (int s = 0; s < NSrcs; s++)
      for (size_t i = 0; i < nElts; i++) windows[s][i] = hSrc[s][offElts + i];
    std::vector<T> expected(nElts);
    for (size_t i = 0; i < nElts; i++) expected[i] = ref(windows, i);

    T sentinel;
    std::memset(&sentinel, 0xAB, sizeof(T));

    for (int tdm = 1; tdm >= 0; tdm--) {
      const int tdmCopy = copySwitch ? tdm : 0;
      const int tdmReduce = copySwitch ? 0 : tdm;
      std::vector<DeviceBuffer<T>> dSrc, dDst;
      std::vector<T*> srcPtrs, dstPtrs;
      for (int s = 0; s < NSrcs; s++) {
        dSrc.emplace_back(srcTotal);
        dSrc.back().copyFrom(hSrc[s]);
        srcPtrs.push_back(dSrc.back().ptr + offElts);
      }
      std::vector<T> hDstInit(dstTotal, sentinel);
      for (int d = 0; d < NDsts; d++) {
        dDst.emplace_back(dstTotal);
        dDst.back().copyFrom(hDstInit);
        dstPtrs.push_back(dDst.back().ptr + dstStart);
      }
      DeviceBuffer<T*> dSrcPtrs(NSrcs), dDstPtrs(NDsts);
      dSrcPtrs.copyFrom(srcPtrs);
      dDstPtrs.copyFrom(dstPtrs);

      const unsigned int zero = 0;
      HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(rcclTdmReduceTestCopies), &zero, sizeof(zero)));
      kReduceCopyTdm<RedOp, T, NSrcs, NDsts, PreOpSrcs><<<1, kBlockThreads>>>(
        dSrcPtrs.ptr, dDstPtrs.ptr, static_cast<ssize_t>(nElts), redArg, postOp, tdmCopy, tdmReduce);
      syncAndCheck();
      unsigned int copies = 0;
      HIP_CHECK(hipMemcpyFromSymbol(&copies, HIP_SYMBOL(rcclTdmReduceTestCopies), sizeof(copies)));

      DeviceBuffer<int> dGate(1);
      kTdmReduceGate<RedOp, T, NSrcs, NDsts, PreOpSrcs><<<1, kBlockThreads>>>(
        dSrcPtrs.ptr, dDstPtrs.ptr, static_cast<ssize_t>(nElts), postOp, tdmCopy, tdmReduce, dGate.ptr);
      syncAndCheck();
      const int gate = dGate.copyTo()[0];
      if (gate < 0) GTEST_SKIP() << "Device code was built without the TDM reduce path";
      ASSERT_EQ(tdm && expectTdm ? 1 : 0, gate) << tag << " tdm=" << tdm << ": TDM gate";
      ASSERT_EQ(copySwitch && tdm && expectTdm ? 1u : 0u, copies)
        << tag << " tdm=" << tdm << ": copies reduceCopyTdm took";

      for (int d = 0; d < NDsts; d++) {
        ASSERT_NO_FATAL_FAILURE(checkDst(dDst[d].copyTo(), dstStart, expected, sentinel,
                                         tag + " tdm=" + std::to_string(tdm) + " dst=" + std::to_string(d)));
      }
    }
  }

  // Copies nElts through reduceCopyTdm alone and checks that it consumed the whole slice and that every destination
  // matches the source with its guards intact.
  template<typename T, int NDsts>
  void runCopyTdmOnly(size_t nElts, const std::string& tag) {
#if !ENABLE_TDM_SIMPLE
    GTEST_SKIP() << "Built without ENABLE_TDM_SIMPLE";
#endif
    if (!supported_) GTEST_SKIP() << "TDM not supported on this device";

    // 256 bytes of sentinel on both sides keep every destination on the alignment the mover assumes.
    const size_t guard = 256 / sizeof(T);
    const size_t dstTotal = guard + nElts + guard;
    std::vector<T> hSrc(nElts);
    for (size_t i = 0; i < nElts; i++) hSrc[i] = value<T>(i, 0);
    T sentinel;
    std::memset(&sentinel, 0xAB, sizeof(T));

    DeviceBuffer<T> dSrc(nElts);
    dSrc.copyFrom(hSrc);
    std::vector<DeviceBuffer<T>> dDst;
    std::vector<T*> dstPtrs;
    for (int d = 0; d < NDsts; d++) {
      dDst.emplace_back(dstTotal);
      dDst.back().copyFrom(std::vector<T>(dstTotal, sentinel));
      dstPtrs.push_back(dDst.back().ptr + guard);
    }
    DeviceBuffer<T*> dDstPtrs(NDsts);
    dDstPtrs.copyFrom(dstPtrs);
    DeviceBuffer<ssize_t> dLeft(1);

    kCopyTdmOnly<T, NDsts><<<1, kBlockThreads>>>(dSrc.ptr, dDstPtrs.ptr, static_cast<ssize_t>(nElts), dLeft.ptr);
    syncAndCheck();
    const ssize_t left = dLeft.copyTo()[0];
    if (left < 0) GTEST_SKIP() << "Device code was built without the TDM path";
    ASSERT_EQ(0, left) << tag << ": reduceCopyTdm left bytes for the vector path";

    for (int d = 0; d < NDsts; d++) {
      ASSERT_NO_FATAL_FAILURE(checkDst(dDst[d].copyTo(), guard, hSrc, sentinel, tag + " dst=" + std::to_string(d)));
    }
  }
};

template<>
float TdmReduceCopyTest::value<float>(size_t i, int s) {
  return static_cast<float>(static_cast<int>((i * 7 + s * 13) % 101) - 50) * 0.25f;
}

template<>
uint32_t TdmReduceCopyTest::value<uint32_t>(size_t i, int s) {
  return static_cast<uint32_t>((i * 2654435761u) ^ (s * 40503u)) & 0x00FFFFFFu;
}

template<>
uint8_t TdmReduceCopyTest::value<uint8_t>(size_t i, int s) {
  return static_cast<uint8_t>(i * 131 + s * 7 + 1);
}

static float sumRef(const std::vector<std::vector<float>>& src, size_t i) {
  float acc = src[0][i];
  for (size_t s = 1; s < src.size(); s++) acc += src[s][i];
  return acc;
}

static uint64_t floatArg(float x) {
  uint64_t arg = 0;
  std::memcpy(&arg, &x, sizeof(x));
  return arg;
}

// Two sources, as in a ring's recv-reduce steps: whole tiles plus a vector remainder.
TEST_F(TdmReduceCopyTest, SumF32TwoSrcsOneDst) {
  run<FuncSum<float>, float, 2, 1, 0>(tileElts<float>(2) + 333, 0, 0, false, true, sumRef, "sum2x1");
}

// recvReduceCopySend: the reduced tile goes to the local output and the next peer.
TEST_F(TdmReduceCopyTest, SumF32TwoSrcsTwoDsts) {
  run<FuncSum<float>, float, 2, 2, 0>(tileElts<float>(2) + 77, 0, 0, false, true, sumRef, "sum2x2");
}

// Tree-shaped fan-in: the third source goes through the second tile after the first two are reduced.
TEST_F(TdmReduceCopyTest, SumF32ThreeSrcs) {
  run<FuncSum<float>, float, 3, 1, 0>(tileElts<float>(1) + 5, 0, 0, false, true, sumRef, "sum3x1");
}

TEST_F(TdmReduceCopyTest, SumF32FourSrcsTwoDsts) {
  run<FuncSum<float>, float, 4, 2, 0>(tileElts<float>(1), 0, 0, false, true, sumRef, "sum4x2");
}

// A tile count that is not a multiple of the warp count exercises the warp rotation before the vector passes.
TEST_F(TdmReduceCopyTest, SumF32UnevenTiles) {
  run<FuncSum<float>, float, 2, 1, 0>(tileElts<float>(1) * 13 / 8 + 1000, 0, 0, false, true, sumRef, "sumUneven");
}

// One source with a scaling pre-op is not copy-shaped, so the reduce engine takes it.
TEST_F(TdmReduceCopyTest, PreMulSumF32OneSrc) {
  run<FuncPreMulSum<float>, float, 1, 1, 1>(
    tileElts<float>(2) + 9, 0, floatArg(0.5f), false, true,
    [](const std::vector<std::vector<float>>& src, size_t i) { return src[0][i] * 0.5f; }, "premul1x1");
}

// With PreOpSrcs = 1 the pre-op scales source 0 only.
TEST_F(TdmReduceCopyTest, PreMulSumF32TwoSrcs) {
  run<FuncPreMulSum<float>, float, 2, 1, 1>(
    tileElts<float>(2) + 9, 0, floatArg(0.5f), false, true,
    [](const std::vector<std::vector<float>>& src, size_t i) { return src[0][i] * 0.5f + src[1][i]; },
    "premul2x1");
}

// PreOpSrcs above 1, as in a SIMPLE direct fan-in: the pre-op scales every source below it, including the ones
// reduced after the first tile pair.
TEST_F(TdmReduceCopyTest, PreMulSumF32ThreeSrcsAllPreOp) {
  run<FuncPreMulSum<float>, float, 3, 1, 3>(
    tileElts<float>(2) + 9, 0, floatArg(0.5f), false, true,
    [](const std::vector<std::vector<float>>& src, size_t i) {
      return src[0][i] * 0.5f + src[1][i] * 0.5f + src[2][i] * 0.5f;
    },
    "premul3x1pre3");
}

// The boundary: sources 0 and 1 are scaled, source 2 is not.
TEST_F(TdmReduceCopyTest, PreMulSumF32ThreeSrcsTwoPreOp) {
  run<FuncPreMulSum<float>, float, 3, 1, 2>(
    tileElts<float>(2) + 9, 0, floatArg(0.5f), false, true,
    [](const std::vector<std::vector<float>>& src, size_t i) { return src[0][i] * 0.5f + src[1][i] * 0.5f + src[2][i]; },
    "premul3x1pre2");
}

// Integer ncclAvg: the post-op divides the final sum.
TEST_F(TdmReduceCopyTest, SumPostDivU32TwoSrcs) {
  constexpr uint32_t Divisor = 3;
  run<FuncSumPostDiv<uint32_t>, uint32_t, 2, 1, 0>(
    tileElts<uint32_t>(2) + 21, 0, uint64_t(Divisor) << 1, true, true,
    [](const std::vector<std::vector<uint32_t>>& src, size_t i) { return (src[0][i] + src[1][i]) / Divisor; },
    "postdiv2x1");
}

// Pointers that are only 4-byte aligned fail the TDM gate and take the vector path.
TEST_F(TdmReduceCopyTest, SumF32MisalignedFallsBack) {
  run<FuncSum<float>, float, 2, 1, 0>(tileElts<float>(2), 1, 0, false, false, sumRef, "sumMisaligned");
}

// Less than one tile also takes the vector path.
TEST_F(TdmReduceCopyTest, SumF32BelowOneTileFallsBack) {
  run<FuncSum<float>, float, 2, 1, 0>(1000, 0, 0, false, false, sumRef, "sumSmall");
}

static float copyRef(const std::vector<std::vector<float>>& src, size_t i) {
  return src[0][i];
}

// recvCopySend: one FIFO source to the output and the next peer, in four rounds of 6400-byte tiles, the last shorter.
TEST_F(TdmReduceCopyTest, CopyF32OneSrcTwoDsts) {
  run<FuncSum<float>, float, 1, 2, 0>(tileElts<float>(3) + 333, 0, 0, false, true, copyRef, "copy1x2",
                                      /*copySwitch=*/true);
}

// 1.7 tiles' worth per warp: two rounds of 6912-byte tiles over all eight warps instead of 8 KiB tiles over five.
TEST_F(TdmReduceCopyTest, CopyF32UnevenTilesThreeDsts) {
  run<FuncSum<float>, float, 1, 3, 0>(tileElts<float>(1) * 13 / 8 + 1000, 0, 0, false, true, copyRef, "copyUneven1x3",
                                      /*copySwitch=*/true);
}

// A copy shorter than one tile still goes to the mover, in 512-byte tiles.
TEST_F(TdmReduceCopyTest, CopyF32BelowOneTileTwoDsts) {
  run<FuncSum<float>, float, 1, 2, 0>(1000, 0, 0, false, true, copyRef, "copySmall1x2", /*copySwitch=*/true);
}

// An aborted genericOp passes an empty slice; the pointers stay aligned, so only the size refuses it.
TEST_F(TdmReduceCopyTest, CopyF32EmptySliceFallsBack) {
  run<FuncSum<float>, float, 1, 2, 0>(0, 64, 0, false, false, copyRef, "copyEmpty1x2", /*copySwitch=*/true);
}

// Pointers off the 256-byte alignment still fail the gate for copies.
TEST_F(TdmReduceCopyTest, CopyF32MisalignedFallsBack) {
  run<FuncSum<float>, float, 1, 2, 0>(tileElts<float>(2), 1, 0, false, false, copyRef, "copyMisaligned1x2",
                                      /*copySwitch=*/true);
}

// The cases below call reduceCopyTdm without the vector passes, so they pass only if the mover copies every byte.
TEST_F(TdmReduceCopyTest, CopyTdmOnlyF32TwoDsts) {
  runCopyTdmOnly<float, 2>(tileElts<float>(3) + 333, "copyOnly1x2");
}

TEST_F(TdmReduceCopyTest, CopyTdmOnlyF32ThreeDsts) {
  runCopyTdmOnly<float, 3>(tileElts<float>(1) * 13 / 8 + 1000, "copyOnly1x3");
}

// Three rounds of 5632-byte tiles ending in a 1613-byte one: an odd-length tail through the mover's byte path.
TEST_F(TdmReduceCopyTest, CopyTdmOnlyU8OddTail) {
  runCopyTdmOnly<uint8_t, 2>(tileElts<uint8_t>(2) + 77, "copyOnlyOddTail");
}

// Four 256-byte tiles, the last 9 bytes, and four warps with nothing to do.
TEST_F(TdmReduceCopyTest, CopyTdmOnlyU8FewerTilesThanWarps) {
  runCopyTdmOnly<uint8_t, 2>(777, "copyOnlyFewTiles");
}

TEST_F(TdmReduceCopyTest, CopyTdmOnlyU8OneByte) {
  runCopyTdmOnly<uint8_t, 2>(1, "copyOnlyOneByte");
}

// A single-destination copy stays with tryTdmCopy.
TEST_F(TdmReduceCopyTest, CopyF32OneDstStaysOnCopyPath) {
  run<FuncSum<float>, float, 1, 1, 0>(tileElts<float>(2), 0, 0, false, false, copyRef, "copy1x1", /*copySwitch=*/true);
}

// The reduce switch alone does not admit a copy.
TEST_F(TdmReduceCopyTest, CopyF32TwoDstsNeedsCopySwitch) {
  run<FuncSum<float>, float, 1, 2, 0>(tileElts<float>(2), 0, 0, false, false, copyRef, "copy1x2reduceSwitch");
}

// Every case above skips when the TDM reduce path is not built, so a TDM test config would pass while covering nothing.
// With RCCL_TEST_EXPECT_TDM_SIMPLE set, this case fails instead when the build flag or the device pass disagrees.
TEST_F(TdmReduceCopyTest, BuildFlagMatchesExpectation) {
  const char* want = getenv("RCCL_TEST_EXPECT_TDM_SIMPLE");
  if (want == nullptr) GTEST_SKIP() << "set RCCL_TEST_EXPECT_TDM_SIMPLE=1 to assert the TDM SIMPLE build flag";
  const int expect = atoi(want);
#if ENABLE_TDM_SIMPLE
  constexpr int built = 1;
#else
  constexpr int built = 0;
#endif
  ASSERT_EQ(built, expect) << "ENABLE_TDM_SIMPLE is " << built << " but RCCL_TEST_EXPECT_TDM_SIMPLE asks for " << expect
                           << ". Either the build is missing --enable-tdm-simple (in which case every TDM reduce suite "
                           << "runs the vector path and covers nothing), or the config sets the wrong expectation.";
  if (expect == 0) return;
  if (!supported_) GTEST_SKIP() << "TDM not supported on this device";

  const size_t nElts = tileElts<float>(1);
  DeviceBuffer<float> src0(nElts), src1(nElts), dst(nElts);
  DeviceBuffer<float*> dSrcPtrs(2), dDstPtrs(1);
  dSrcPtrs.copyFrom(std::vector<float*>{src0.ptr, src1.ptr});
  dDstPtrs.copyFrom(std::vector<float*>{dst.ptr});
  DeviceBuffer<int> dGate(1);
  kTdmReduceGate<FuncSum<float>, float, 2, 1, 0><<<1, kBlockThreads>>>(
    dSrcPtrs.ptr, dDstPtrs.ptr, static_cast<ssize_t>(nElts), /*postOp=*/false, /*tdmCopy=*/0, /*tdmReduce=*/1,
    dGate.ptr);
  syncAndCheck();
  EXPECT_EQ(1, dGate.copyTo()[0]) << "The device pass has no TDM reduce path (TDM_SUPPORTED is 0 there), so every TDM "
                                  << "reduce suite runs the vector path.";
}

} // namespace RcclUnitTesting
