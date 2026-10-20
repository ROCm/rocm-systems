/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// On-GPU correctness test for the GIN Anvil-SDMA ReduceScatter (-D 3,
// GinReduceScatterKernel). Exercises on real hardware:
//
//   (A) the size-adaptive CTA ladder (gin_sdma_reducescatter::reduceScatterCtas)
//       evaluates identically on device and host, and
//
//   (B) the source fold both kernel tiers call (gin_sdma_reduce::foldSources)
//       over the read-reduce addressing -- rank R reads pack p of its owned
//       slice from every peer's send buffer at R*count and folds in ascending
//       source-rank order -- for rank counts that reach every load batch, and
//
//   (C) the gin_sdma_reduce.h element ops, which must round back to T after
//       every pairwise step to bit-match verifiable.cu.
//
// The simulation kernel below replaces ncclGetLsaPointer with plain peer send
// buffers laid out as send[peer*(nRanks*count) + rank*count + i]. It needs
// neither librccl nor a GIN communicator, so the barriers, the SDMA puts and
// the source-0 prefetch of the real kernel are not exercised here; the
// reduce_scatter_perf -D 3 datacheck covers those end to end.
//
// Requires a visible GPU at run time; skips cleanly (GTEST_SKIP) otherwise.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <hip/hip_runtime.h>

// gin_sdma_reduce.h documents itself as "include after common.h", but common.h
// is rccl-tests' MPI harness and this target has no business linking it.
// Reproduce only what the header actually reads: rccl.h for NCCL_VERSION_CODE,
// ncclRedOp_t and __half; the HAVE_BF16 / HAVE_FP8 guards common.h derives from
// RCCL_BFLOAT16 / RCCL_FLOAT8; and rccl_float8.h for the two fp8 types.
#include <rccl/rccl.h>

#if NCCL_MAJOR >= 2 && RCCL_BFLOAT16 == 1
#define HAVE_BF16 1
#else
#define HAVE_BF16 0
#endif
#if NCCL_MAJOR >= 2 && RCCL_FLOAT8 == 1
#define HAVE_FP8 1
#else
#define HAVE_FP8 0
#endif
#if HAVE_FP8
#include "rccl_float8.h"
#endif

// The reduce helpers are pure arithmetic: no device API, no symmetric memory,
// no communicator. Open the gate here rather than let this coverage blink out
// depending on how librccl happened to be configured.
#ifndef ENABLE_DEVICE_API
#define ENABLE_DEVICE_API 1
#endif

#include "gin_sdma_reduce.h"
#include "gin_sdma_reducescatter_policy.h"

namespace {

#define HIP_OK(cmd)                                                       \
  do {                                                                    \
    hipError_t _e = (cmd);                                                \
    ASSERT_EQ(_e, hipSuccess) << "HIP error " << (int)_e << " ("          \
                              << hipGetErrorString(_e) << ") at "         \
                              << __FILE__ << ":" << __LINE__;             \
  } while (0)

bool gpuAvailable() {
  int n = 0;
  hipError_t e = hipGetDeviceCount(&n);
  return e == hipSuccess && n > 0;
}

// ---- (A) CTA ladder: device must match host ----------------------------------

__global__ void ctasKernel(const uint64_t* totalBytes, const uint64_t* envCtas, int* out, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    out[i] = gin_sdma_reducescatter::reduceScatterCtas((size_t)totalBytes[i], (size_t)envCtas[i]);
  }
}

TEST(ReduceScatterGpu, CtaLadderMatchesHost) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";

  std::vector<uint64_t> totals, envs;
  const uint64_t envUnset = gin_sdma_reducescatter::kThresholdUnset;
  for (uint64_t t : {0ull, 1ull * 1024 * 1024, 8ull * 1024 * 1024, 16ull * 1024 * 1024,
                     32ull * 1024 * 1024, 48ull * 1024 * 1024, 64ull * 1024 * 1024,
                     2ull * 1024 * 1024 * 1024}) {
    totals.push_back(t);
    envs.push_back(envUnset);
  }
  totals.push_back(16ull * 1024 * 1024);
  envs.push_back(64);  // env override
  const int n = (int)totals.size();

  uint64_t *dTotals = nullptr, *dEnvs = nullptr;
  int* dOut = nullptr;
  HIP_OK(hipMalloc(&dTotals, n * sizeof(uint64_t)));
  HIP_OK(hipMalloc(&dEnvs, n * sizeof(uint64_t)));
  HIP_OK(hipMalloc(&dOut, n * sizeof(int)));
  HIP_OK(hipMemcpy(dTotals, totals.data(), n * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dEnvs, envs.data(), n * sizeof(uint64_t), hipMemcpyHostToDevice));

  ctasKernel<<<(n + 63) / 64, 64>>>(dTotals, dEnvs, dOut, n);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());

  std::vector<int> out(n, -1);
  HIP_OK(hipMemcpy(out.data(), dOut, n * sizeof(int), hipMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    const int host = gin_sdma_reducescatter::reduceScatterCtas((size_t)totals[i], (size_t)envs[i]);
    EXPECT_EQ(out[i], host) << "totalBytes=" << totals[i] << " envCtas=" << envs[i];
  }

  HIP_OK(hipFree(dTotals));
  HIP_OK(hipFree(dEnvs));
  HIP_OK(hipFree(dOut));
}

__global__ void sdmaTierKernel(const int* grids, int* out, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) out[i] = gin_sdma_reducescatter::usesSdmaTier(grids[i]) ? 1 : 0;
}

TEST(ReduceScatterGpu, SdmaTierMatchesHost) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  const int grids[] = {0, 1, 4, 15, 16, 32, 48};
  const int n = (int)(sizeof(grids) / sizeof(grids[0]));
  int *dGrids = nullptr, *dOut = nullptr;
  HIP_OK(hipMalloc(&dGrids, n * sizeof(int)));
  HIP_OK(hipMalloc(&dOut, n * sizeof(int)));
  HIP_OK(hipMemcpy(dGrids, grids, n * sizeof(int), hipMemcpyHostToDevice));
  sdmaTierKernel<<<1, 64>>>(dGrids, dOut, n);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());
  int out[8];
  HIP_OK(hipMemcpy(out, dOut, n * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(out[i] != 0, gin_sdma_reducescatter::usesSdmaTier(grids[i])) << "grid=" << grids[i];
  }
  HIP_OK(hipFree(dGrids));
  HIP_OK(hipFree(dOut));
}

// ---- (B) read-reduce addressing + foldSources --------------------------------
//
// Layout mirrors GinReduceScatterKernel: send is [nRanks, nRanks, count] flattened
// as send[peer * (nRanks*count) + ownerRank * count + i]. Rank R reduces over
// source s in ascending order: out[R,i] = fold_s send[s][R][i].

template <typename T>
__global__ void rsFoldSimKernel(const T* send, T* recv, int nRanks, size_t count, int op) {
  using Pack = gin_sdma_reduce::Pack<T>;
  constexpr int VEC = Pack::kVec;
  const int rank = blockIdx.x;  // one CTA per rank (simple smoke layout)
  if (rank >= nRanks) return;
  const size_t perPeerPacks = (size_t)nRanks * count / VEC;
  const size_t myBaseP = gin_sdma_reducescatter::reduceScatterSliceOffset(rank, count) / VEC;
  const Pack* sendP = (const Pack*)send;
  Pack* recvP = (Pack*)recv;
  for (size_t pk = threadIdx.x; pk < count / VEC; pk += blockDim.x) {
    auto load = [&](int s) { return sendP[(size_t)s * perPeerPacks + myBaseP + pk]; };
    T acc[VEC];
    gin_sdma_reduce::foldSources<T>(op, nRanks, load(0), load, acc);
    Pack o;
    for (int e = 0; e < VEC; e++) o.e[e] = gin_sdma_reduce::postOp<T>(op, acc[e], nRanks);
    recvP[myBaseP + pk] = o;
  }
}

// value(peer, owner, i) fills the send buffers; the expected sum folds them in
// ascending peer order in T on the host.
template <typename T, typename ValueFn>
void runFoldCase(int nRanks, size_t count, ValueFn value) {
  ASSERT_EQ(count % gin_sdma_reduce::Pack<T>::kVec, 0u) << "count must tile 16-byte packs";
  const size_t perPeer = (size_t)nRanks * count;
  const size_t sendElts = (size_t)nRanks * perPeer;

  std::vector<T> send(sendElts);
  for (int peer = 0; peer < nRanks; ++peer)
    for (int owner = 0; owner < nRanks; ++owner)
      for (size_t i = 0; i < count; ++i)
        send[(size_t)peer * perPeer + (size_t)owner * count + i] = value(peer, owner, i);

  std::vector<T> expected((size_t)nRanks * count, (T)0);
  for (int owner = 0; owner < nRanks; ++owner)
    for (size_t i = 0; i < count; ++i) {
      T acc = send[(size_t)owner * count + i];
      for (int s = 1; s < nRanks; ++s)
        acc = (T)(acc + send[(size_t)s * perPeer + (size_t)owner * count + i]);
      expected[(size_t)owner * count + i] = acc;
    }

  T *dSend = nullptr, *dRecv = nullptr;
  HIP_OK(hipMalloc(&dSend, send.size() * sizeof(T)));
  HIP_OK(hipMalloc(&dRecv, expected.size() * sizeof(T)));
  HIP_OK(hipMemset(dRecv, 0, expected.size() * sizeof(T)));
  HIP_OK(hipMemcpy(dSend, send.data(), send.size() * sizeof(T), hipMemcpyHostToDevice));

  rsFoldSimKernel<T><<<nRanks, 256>>>(dSend, dRecv, nRanks, count, (int)ncclSum);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());

  std::vector<T> recv(expected.size());
  HIP_OK(hipMemcpy(recv.data(), dRecv, recv.size() * sizeof(T), hipMemcpyDeviceToHost));

  size_t wrong = 0;
  for (size_t k = 0; k < expected.size(); ++k)
    if (recv[k] != expected[k]) ++wrong;
  EXPECT_EQ(wrong, 0u) << "nRanks=" << nRanks << " count=" << count << " eltSize=" << sizeof(T);

  HIP_OK(hipFree(dSend));
  HIP_OK(hipFree(dRecv));
}

// 2 and 3 reach only the one-source tail, 4 the two-source step, 7 the
// four-source step plus the two-source step, 8 the eight-source batch alone,
// and 12 and 16 the batch followed by four-source steps.
constexpr int kFoldRankCounts[] = {2, 3, 4, 7, 8, 12, 16};

template <typename T>
T smallPattern(int peer, int owner, size_t i) {
  return (T)((peer * 9973 + owner * 131 + (uint32_t)i) & 0x7f);
}

TEST(ReduceScatterGpu, FoldAddressingInt32) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  for (int nr : kFoldRankCounts)
    for (size_t c : {(size_t)4, (size_t)20, (size_t)256, (size_t)1024})
      runFoldCase<int32_t>(nr, c, smallPattern<int32_t>);
}

TEST(ReduceScatterGpu, FoldAddressingInt8) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  for (int nr : kFoldRankCounts)
    for (size_t c : {(size_t)16, (size_t)144, (size_t)1024})
      runFoldCase<int8_t>(nr, c, smallPattern<int8_t>);
}

// 2^24 + 1 rounds back to 2^24 in float, so folding source 0's 2^24 first
// absorbs every later 1.0f and the result stays 2^24. Any fold that adds some
// of the 1.0f sources together before meeting 2^24 lands higher.
TEST(ReduceScatterGpu, FoldRunsInAscendingSourceOrder) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  auto value = [](int peer, int, size_t) { return peer == 0 ? 16777216.0f : 1.0f; };
  for (int nr : kFoldRankCounts) runFoldCase<float>(nr, 64, value);
}

// ---- (C) reduce element ops: narrowing happens on every pairwise step -------
//
// gin_sdma_reduce.h exists only to bit-match verifiable.cu, which rounds each
// low-precision fold back to T after every pair instead of accumulating in
// float. reduce_scatter.cu is its sole caller, so replacing those overloads
// with a float accumulator would leave every gtest green and only show up as a
// datacheck mismatch in the non-gating -D 3 pytest job.
//
// Every case folds {B, 1, 1, 1, 1, 1, 1, 1} over eight ranks. B is chosen so
// that one ulp of B exceeds 2, which puts B+1 strictly inside the ulp -- never
// on a tie, so the answer does not depend on ties-to-even -- and pins the
// step-wise sum at B. A float accumulator would reach B+7 and narrow once,
// landing an ulp or two higher (bf16 520, half 4104, both fp8 40); avg splits
// the same way. float and int32 are the controls: neither narrows, so they see
// the full B+7.

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

template <class T> __device__ double toDouble(T x) { return (double)(float)x; }
__device__ double toDouble(double x) { return x; }

template <class T>
__global__ void reduceFoldKernel(const float* in, int n, int op, int rankN, double* out) {
  T acc = gin_sdma_reduce::preOp<T>(op, (T)in[0], rankN);
  for (int i = 1; i < n; ++i) {
    acc = gin_sdma_reduce::combine<T>(op, acc, gin_sdma_reduce::preOp<T>(op, (T)in[i], rankN));
  }
  *out = toDouble(gin_sdma_reduce::postOp<T>(op, acc, rankN));
}

// rankN is in.size().
template <class T>
void runReduceFoldCase(const char* tname, const std::vector<float>& in, ncclRedOp_t op,
                       const char* opName, double expected) {
  float* dIn = nullptr;
  double* dOut = nullptr;
  HIP_OK(hipMalloc(&dIn, in.size() * sizeof(float)));
  HIP_OK(hipMalloc(&dOut, sizeof(double)));
  HIP_OK(hipMemcpy(dIn, in.data(), in.size() * sizeof(float), hipMemcpyHostToDevice));

  reduceFoldKernel<T><<<1, 1>>>(dIn, (int)in.size(), (int)op, (int)in.size(), dOut);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());

  double got = 0.0;
  HIP_OK(hipMemcpy(&got, dOut, sizeof(double), hipMemcpyDeviceToHost));
  // Exact compare: every expected value is representable in the element type
  // and converts back without loss, so a tolerance would only hide the
  // one-ulp difference this test is looking for.
  EXPECT_EQ(got, expected) << tname << " " << opName;

  HIP_OK(hipFree(dIn));
  HIP_OK(hipFree(dOut));
}

template <class T>
void runReduceFoldCase(const char* tname, float b, ncclRedOp_t op, const char* opName,
                       double expected) {
  runReduceFoldCase<T>(tname, {b, 1, 1, 1, 1, 1, 1, 1}, op, opName, expected);
}

#if HAVE_FP8
// fp8: 3 and 2 mantissa bits, ulp(32) = 4 and 8. prod is left out because
// SPECIALIZE_REDUCE_KERNEL never dispatches it for the fp8 types. 32, 4 and
// 1 are exact in both the fnuz and the IEEE encodings, so gfx942 and gfx950
// expect the same values.
template <class E4m3, class E5m2>
void runFp8FoldCases() {
  runReduceFoldCase<E4m3>("fp8_e4m3", 32, ncclSum, "sum", 32.0);
  runReduceFoldCase<E4m3>("fp8_e4m3", 32, ncclMax, "max", 32.0);
  runReduceFoldCase<E4m3>("fp8_e4m3", 32, ncclMin, "min", 1.0);
  runReduceFoldCase<E4m3>("fp8_e4m3", 32, ncclAvg, "avg", 4.0);

  runReduceFoldCase<E5m2>("fp8_e5m2", 32, ncclSum, "sum", 32.0);
  runReduceFoldCase<E5m2>("fp8_e5m2", 32, ncclMax, "max", 32.0);
  runReduceFoldCase<E5m2>("fp8_e5m2", 32, ncclMin, "min", 1.0);
  runReduceFoldCase<E5m2>("fp8_e5m2", 32, ncclAvg, "avg", 4.0);
}

[[maybe_unused]] bool gpuIsGfx942() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return false;
  return strncmp(prop.gcnArchName, "gfx942", 6) == 0;
}
#endif

TEST(ReduceScatterGpu, ReduceOpsNarrowEveryStep) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";

  // combine() switches on the raw int, and SPECIALIZE_REDUCE_KERNEL passes an
  // ncclRedOp_t straight through, so the two have to keep agreeing.
  ASSERT_EQ((int)ncclSum, 0);
  ASSERT_EQ((int)ncclProd, 1);
  ASSERT_EQ((int)ncclMax, 2);
  ASSERT_EQ((int)ncclMin, 3);
  ASSERT_EQ((int)ncclAvg, 4);

  // Controls. float avg premultiplies by 1/8 (exact here) and int32 avg sums
  // then integer-divides, which truncates 519/8 to 64.
  runReduceFoldCase<float>("float", 512, ncclSum, "sum", 519.0);
  runReduceFoldCase<float>("float", 512, ncclProd, "prod", 512.0);
  runReduceFoldCase<float>("float", 512, ncclMax, "max", 512.0);
  runReduceFoldCase<float>("float", 512, ncclMin, "min", 1.0);
  runReduceFoldCase<float>("float", 512, ncclAvg, "avg", 64.875);

  runReduceFoldCase<int32_t>("int32", 512, ncclSum, "sum", 519.0);
  runReduceFoldCase<int32_t>("int32", 512, ncclProd, "prod", 512.0);
  runReduceFoldCase<int32_t>("int32", 512, ncclMax, "max", 512.0);
  runReduceFoldCase<int32_t>("int32", 512, ncclMin, "min", 1.0);
  runReduceFoldCase<int32_t>("int32", 512, ncclAvg, "avg", 64.0);

  runReduceFoldCase<double>("double", 512, ncclSum, "sum", 519.0);
  runReduceFoldCase<double>("double", 512, ncclProd, "prod", 512.0);
  runReduceFoldCase<double>("double", 512, ncclMax, "max", 512.0);
  runReduceFoldCase<double>("double", 512, ncclMin, "min", 1.0);

  // half: 10 mantissa bits, ulp(4096) = 4, so 4096+1 stays 4096 seven times.
  runReduceFoldCase<__half>("half", 4096, ncclSum, "sum", 4096.0);
  runReduceFoldCase<__half>("half", 4096, ncclProd, "prod", 4096.0);
  runReduceFoldCase<__half>("half", 4096, ncclMax, "max", 4096.0);
  runReduceFoldCase<__half>("half", 4096, ncclMin, "min", 1.0);
  runReduceFoldCase<__half>("half", 4096, ncclAvg, "avg", 512.0);

#if HAVE_BF16
  // bf16: 7 mantissa bits, ulp(512) = 4.
  runReduceFoldCase<hip_bfloat16>("bf16", 512, ncclSum, "sum", 512.0);
  runReduceFoldCase<hip_bfloat16>("bf16", 512, ncclProd, "prod", 512.0);
  runReduceFoldCase<hip_bfloat16>("bf16", 512, ncclMax, "max", 512.0);
  runReduceFoldCase<hip_bfloat16>("bf16", 512, ncclMin, "min", 1.0);
  runReduceFoldCase<hip_bfloat16>("bf16", 512, ncclAvg, "avg", 64.0);
#endif

#if HAVE_FP8
  // rccl_float8.h makes rccl_float8 the fnuz type only in the gfx942 device
  // pass, so the host pass names both encodings and runs the one the device
  // image defines, as reduce_scatter.cu's dispatch does.
#if __HIP_DEVICE_COMPILE__ || HIP_VERSION < 60300000
  runFp8FoldCases<rccl_float8, rccl_bfloat8>();
#else
  if (gpuIsGfx942())
    runFp8FoldCases<__hip_fp8_e4m3_fnuz, __hip_fp8_e5m2_fnuz>();
  else
    runFp8FoldCases<__hip_fp8_e4m3, __hip_fp8_e5m2>();
#endif
#endif
}

// verifiable.cu builds the double avg factor as 1.0 / rankN in double. 1/7 is
// inexact, so a factor built in float, as avgFactor's primary template does
// for the narrower types, sums to a different double.
TEST(ReduceScatterGpu, DoubleAvgFactorIsBuiltInDouble) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  const int rankN = 7;
  const double f = 1.0 / (double)rankN;
  const double fFloat = (double)(1.0f / (float)rankN);
  double expected = f, viaFloat = fFloat;
  for (int i = 1; i < rankN; ++i) {
    expected += f;
    viaFloat += fFloat;
  }
  ASSERT_NE(expected, viaFloat);
  runReduceFoldCase<double>("double", std::vector<float>(rankN, 1.0f), ncclAvg, "avg", expected);
}

#endif  // NCCL_VERSION_CODE >= 2.28

}  // namespace
