/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for the TDM paths of reduceCopy: the reduction (reduceCopyTdm in tdm/tdmReduce.h)
// and the one-source copy (tryTdmCopy in common_kernel.h).
//
// Each kernel drives reduceCopy from one block the way genericOp does, with the pointer
// lists in shared memory; the reduce kernels also set EnableTdmReduce. Every case runs
// twice, with the runtime switch on and off, and both outputs must match a host reference
// exactly, so the TDM tiles and the vector passes that finish each slice are checked together.
// Inputs are small multiples of 1/4, so float sums are exact in any order.

#include "DeviceTestBase.hpp"

#include <array>
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
                 [[maybe_unused]] int tdmReduce) {
  __shared__ void* srcPtrs[NSrcs];
  __shared__ void* dstPtrs[NDsts];
  if (threadIdx.x == 0) {
#if ENABLE_TDM_SIMPLE
    ncclShmem.comm.tdmSimpleEnable = 0;
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
  kTdmReduceGate(T* const* srcs, T* const* dsts, ssize_t nElts, bool postOp, [[maybe_unused]] int tdmReduce,
                 int* gate) {
  __shared__ void* srcPtrs[NSrcs];
  __shared__ void* dstPtrs[NDsts];
  if (threadIdx.x == 0) {
#if ENABLE_TDM_SIMPLE
    ncclShmem.comm.tdmSimpleEnable = 0;
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

  // Runs one reduceCopy shape with the TDM reduce switch on and off and checks both against `ref`, and checks that the
  // TDM gate admits the shape only with the switch on, and only when `expectTdm`.
  // `offElts` shifts every pointer off the 256-byte alignment the TDM gate requires.
  template<typename RedOp, typename T, int NSrcs, int NDsts, int PreOpSrcs>
  void run(size_t nElts, size_t offElts, uint64_t redArg, bool postOp, bool expectTdm,
           const std::function<T(const std::vector<std::vector<T>>&, size_t)>& ref, const std::string& tag) {
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

    for (int tdmReduce = 1; tdmReduce >= 0; tdmReduce--) {
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

      kReduceCopyTdm<RedOp, T, NSrcs, NDsts, PreOpSrcs><<<1, kBlockThreads>>>(
        dSrcPtrs.ptr, dDstPtrs.ptr, static_cast<ssize_t>(nElts), redArg, postOp, tdmReduce);
      syncAndCheck();

      DeviceBuffer<int> dGate(1);
      kTdmReduceGate<RedOp, T, NSrcs, NDsts, PreOpSrcs><<<1, kBlockThreads>>>(
        dSrcPtrs.ptr, dDstPtrs.ptr, static_cast<ssize_t>(nElts), postOp, tdmReduce, dGate.ptr);
      syncAndCheck();
      const int gate = dGate.copyTo()[0];
      if (gate < 0) GTEST_SKIP() << "Device code was built without the TDM reduce path";
      ASSERT_EQ(tdmReduce && expectTdm ? 1 : 0, gate) << tag << " tdmReduce=" << tdmReduce << ": TDM gate";

      for (int d = 0; d < NDsts; d++) {
        std::vector<T> out = dDst[d].copyTo();
        for (size_t i = 0; i < nElts; i++) {
          ASSERT_EQ(0, std::memcmp(&out[dstStart + i], &expected[i], sizeof(T)))
            << tag << " tdmReduce=" << tdmReduce << " dst=" << d << ": mismatch at element " << i << " of " << nElts;
        }
        for (size_t i = 0; i < dstTotal; i++) {
          if (i >= dstStart && i < dstStart + nElts) continue;
          ASSERT_EQ(0, std::memcmp(&out[i], &sentinel, sizeof(T)))
            << tag << " tdmReduce=" << tdmReduce << " dst=" << d << ": guard overwritten at index " << i;
        }
      }
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
    dSrcPtrs.ptr, dDstPtrs.ptr, static_cast<ssize_t>(nElts), /*postOp=*/false, /*tdmReduce=*/1, dGate.ptr);
  syncAndCheck();
  EXPECT_EQ(1, dGate.copyTo()[0]) << "The device pass has no TDM reduce path (TDM_SUPPORTED is 0 there), so every TDM "
                                  << "reduce suite runs the vector path.";
}

// Runs reduceCopy's copy shape, one source and no reduction, from the team of warps [warp0, warp1). Writes canUseTdm's
// answer for the team and pointers to *gate, or -1 when the device code has no TDM SIMPLE path.
template<typename T, int NDsts>
__device__ void tdmSimpleCopyByTeam(void** srcPtrs, void** dstPtrs, ssize_t nElts, int warp0, int warp1, int* gate) {
  const int thread = threadIdx.x - warp0 * WARP_SIZE;
  const int nThreads = (warp1 - warp0) * WARP_SIZE;
#if TDM_SUPPORTED && ENABLE_TDM_SIMPLE
  const bool admitted = canUseTdm</*useAcc=*/0, FuncSum<T>, T, /*MultimemSrcs=*/0, /*MultimemDsts=*/0,
                                  /*PreOpSrcs=*/0, /*Pipeline=*/0>(thread, nThreads, /*postOp=*/false, 1, srcPtrs,
                                                                   NDsts, dstPtrs, nElts);
  if (thread == 0) *gate = admitted ? 1 : 0;
#else
  if (thread == 0) *gate = -1;
#endif
  reduceCopy<4, /*useAcc=*/0, FuncSum<T>, T, /*MultimemSrcs=*/0, 1, 1, /*MultimemDsts=*/0, NDsts, NDsts,
             /*PreOpSrcs=*/0>(thread, nThreads, 0, /*postOp=*/false, 1, srcPtrs, NDsts, dstPtrs, nElts);
}

// Copies src to dsts from the team of warps [warp0, 8) with the TDM SIMPLE switch set to `tdmSimple`. When warp0 > 0,
// warps [0, warp0) copy lowSrc to lowDst at the same time, so a team that stages outside its own warps' slices
// corrupts the other's data. gate[0] and gate[1] take canUseTdm's answers for the upper and lower teams.
template<typename T, int NDsts>
__global__ void __launch_bounds__(kBlockThreads)
  kTdmSimpleCopy(T* src, T* const* dsts, T* lowSrc, T* lowDst, ssize_t nElts, [[maybe_unused]] int tdmSimple,
                 int warp0, int* gate) {
  __shared__ void* srcPtrs[1];
  __shared__ void* dstPtrs[NDsts];
  __shared__ void* lowSrcPtrs[1];
  __shared__ void* lowDstPtrs[1];
  if (threadIdx.x == 0) {
#if ENABLE_TDM_SIMPLE
    ncclShmem.comm.tdmSimpleEnable = tdmSimple;
    ncclShmem.comm.tdmSimpleReduceEnable = 0;
#endif
    srcPtrs[0] = src;
    for (int d = 0; d < NDsts; d++) dstPtrs[d] = dsts[d];
    lowSrcPtrs[0] = lowSrc;
    lowDstPtrs[0] = lowDst;
  }
  __syncthreads();
  const int warp = threadIdx.x / WARP_SIZE;
  if (warp >= warp0) {
    tdmSimpleCopyByTeam<T, NDsts>(srcPtrs, dstPtrs, nElts, warp0, blockDim.x / WARP_SIZE, gate);
  } else {
    tdmSimpleCopyByTeam<T, 1>(lowSrcPtrs, lowDstPtrs, nElts, 0, warp0, gate + 1);
  }
}

class TdmSimpleCopyTest : public DeviceTestBase {
protected:
  // Sentinel padding on both sides of every destination window, a whole number of 256-byte lines so offset 0 is
  // line-aligned.
  static constexpr size_t kGuardBytes = 256;

  bool supported_ = false;

  void SetUp() override {
    DeviceTestBase::SetUp();
    supported_ = tdm::IsTdmCopySupported(0);
  }

  // Copies nElts from src + srcOff to each dst + dstOffs[d] (offsets in elements) with the TDM SIMPLE switch on and
  // off, from warps [warp0, 8). When warp0 > 0, warps [0, warp0) copy a different source at the same time. Every
  // destination must match its source exactly and leave the guards alone, and the gate must admit each copy only with
  // the switch on: the copy takes any alignment.
  template<typename T, int NDsts>
  void run(size_t nElts, size_t srcOff, const std::array<size_t, NDsts>& dstOffs, int warp0, const std::string& tag) {
#if !ENABLE_TDM_SIMPLE
    GTEST_SKIP() << "Built without ENABLE_TDM_SIMPLE";
#endif
    if (!supported_) GTEST_SKIP() << "TDM not supported on this device";

    const size_t guardElts = kGuardBytes / sizeof(T);
    std::vector<T> hSrc(srcOff + nElts);
    for (size_t i = 0; i < hSrc.size(); i++) hSrc[i] = static_cast<T>((i * 2654435761u) >> 7);
    std::vector<T> hLowSrc(nElts);
    for (size_t i = 0; i < nElts; i++) hLowSrc[i] = static_cast<T>((i * 2246822519u) >> 5);

    T sentinel;
    std::memset(&sentinel, 0xAB, sizeof(T));

    for (int tdmSimple = 1; tdmSimple >= 0; tdmSimple--) {
      DeviceBuffer<T> dSrc(hSrc.size());
      dSrc.copyFrom(hSrc);
      std::vector<DeviceBuffer<T>> dDst;
      std::vector<T*> dstPtrs;
      for (int d = 0; d < NDsts; d++) {
        dDst.emplace_back(guardElts + dstOffs[d] + nElts + guardElts);
        dDst.back().copyFrom(std::vector<T>(dDst.back().count, sentinel));
        dstPtrs.push_back(dDst.back().ptr + guardElts + dstOffs[d]);
      }
      DeviceBuffer<T*> dDstPtrs(NDsts);
      dDstPtrs.copyFrom(dstPtrs);
      DeviceBuffer<T> dLowSrc(nElts), dLowDst(guardElts + nElts + guardElts);
      dLowSrc.copyFrom(hLowSrc);
      dLowDst.copyFrom(std::vector<T>(dLowDst.count, sentinel));
      DeviceBuffer<int> dGate(2);

      kTdmSimpleCopy<T, NDsts><<<1, kBlockThreads>>>(dSrc.ptr + srcOff, dDstPtrs.ptr, dLowSrc.ptr,
                                                     dLowDst.ptr + guardElts, static_cast<ssize_t>(nElts), tdmSimple,
                                                     warp0, dGate.ptr);
      syncAndCheck();
      const std::vector<int> gate = dGate.copyTo();
      if (gate[0] < 0) GTEST_SKIP() << "Device code was built without the TDM SIMPLE path";
      ASSERT_EQ(tdmSimple, gate[0]) << tag << " tdmSimple=" << tdmSimple << ": TDM gate";
      if (warp0 > 0) ASSERT_EQ(tdmSimple, gate[1]) << tag << " tdmSimple=" << tdmSimple << ": TDM gate, warps below";

      auto check = [&](const std::vector<T>& out, size_t start, const T* want, const std::string& what) {
        for (size_t i = 0; i < out.size(); i++) {
          const bool inside = i >= start && i < start + nElts;
          ASSERT_EQ(0, std::memcmp(&out[i], inside ? &want[i - start] : &sentinel, sizeof(T)))
            << tag << " tdmSimple=" << tdmSimple << " " << what << ": "
            << (inside ? "mismatch at element " : "guard overwritten at index ") << (inside ? i - start : i)
            << " of " << nElts;
        }
      };
      for (int d = 0; d < NDsts; d++) {
        check(dDst[d].copyTo(), guardElts + dstOffs[d], hSrc.data() + srcOff, "dst=" + std::to_string(d));
        if (HasFatalFailure()) return;
      }
      if (warp0 > 0) check(dLowDst.copyTo(), guardElts, hLowSrc.data(), "warps below");
      if (HasFatalFailure()) return;
    }
  }
};

// Enough for every warp of the block to stream many staging buffers.
constexpr size_t kCopyBytes = (size_t(1) << 20) + 77;

TEST_F(TdmSimpleCopyTest, Aligned) {
  run<uint8_t, 1>(kCopyBytes, 0, {0}, 0, "aligned");
}

// Source and destination off their lines by different amounts, including one off and one on.
TEST_F(TdmSimpleCopyTest, MisalignedOffsets) {
  for (size_t srcOff : {0, 1, 2, 3, 64, 255}) {
    for (size_t dstOff : {0, 1, 3, 128, 200}) {
      if (srcOff == 0 && dstOff == 0) continue;
      run<uint8_t, 1>(kCopyBytes, srcOff, {dstOff}, 0,
                      "src+" + std::to_string(srcOff) + " dst+" + std::to_string(dstOff));
    }
  }
}

// The fp16 rccl-tests shape at offset 1: both legs 2 bytes off.
TEST_F(TdmSimpleCopyTest, HalfOffsetOne) {
  run<uint16_t, 1>(size_t(1) << 20, 1, {1}, 0, "half+1");
}

// directRecvCopySend's two destinations, each off its line by a different amount and neither matching the source.
TEST_F(TdmSimpleCopyTest, TwoDstsDifferentOffsets) {
  run<uint8_t, 2>(kCopyBytes, 5, {1, 130}, 0, "two dsts src+5");
  run<uint16_t, 2>(kCopyBytes / 2, 0, {3, 0}, 0, "two dsts half");
}

// Teams that do not start at warp 0 take their staging from their own warps' slices. The warps below run their own
// copy at the same time, so a team staging in another team's slice shows up as corruption in one of the outputs.
TEST_F(TdmSimpleCopyTest, TeamNotAtWarpZero) {
  run<uint8_t, 1>(kCopyBytes, 1, {2}, 3, "warps 3-7");
  run<uint8_t, 1>(kCopyBytes, 3, {0}, 7, "warp 7 alone");
}

// Sizes below the 256-byte head, around one staging buffer and one block's worth of them, and across many.
TEST_F(TdmSimpleCopyTest, SizeSweep) {
  for (size_t n : {size_t(1), size_t(3), size_t(127), size_t(255), size_t(256), size_t(257), size_t(511),
                   size_t(8191), size_t(8193), size_t(16383), size_t(16385), size_t(65537), size_t(131071),
                   size_t(131073), size_t(3 * 131072 + 100)}) {
    for (size_t off : {0, 1})
      run<uint8_t, 1>(n, off, {3 * off}, 0, std::to_string(n) + " bytes off " + std::to_string(off));
  }
}

} // namespace RcclUnitTesting
