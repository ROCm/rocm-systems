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
//   (B) the direct-LSA read-reduce addressing the production kernel uses --
//       rank R reads element i of its owned slice from every peer's send buffer
//       at offset R*count + i and folds in ascending source-rank order, and
//
//   (C) the gin_sdma_reduce.h element ops, which must round back to T after
//       every pairwise step to bit-match verifiable.cu.
//
// The simulation kernel below replaces ncclGetLsaPointer with plain peer send
// buffers laid out as send[peer*(nRanks*count) + rank*count + i], matching the
// slice offset implied by myBaseP = rank*count in the device kernel. It needs
// neither librccl nor a GIN communicator. End-to-end datacheck with the real
// GinReduceScatterKernel is covered by reduce_scatter_perf -D 3 in
// test_ReduceScatterGinSdma.py.
//
// Requires a visible GPU at run time; skips cleanly (GTEST_SKIP) otherwise.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

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

// ---- (B) read-reduce addressing + ascending-rank sum -------------------------
//
// Layout mirrors GinReduceScatterKernel: send is [nRanks, nRanks, count] flattened
// as send[peer * (nRanks*count) + ownerRank * count + i]. Rank R reduces over
// source s in ascending order: out[R,i] = sum_s send[s][R][i].

template <typename T>
__global__ void rsReadReduceSimKernel(const T* send, T* recv, int nRanks, size_t count) {
  const int rank = blockIdx.x;  // one CTA per rank (simple smoke layout)
  if (rank >= nRanks) return;
  const size_t totalPerPeer = (size_t)nRanks * count;
  const size_t myOff =
      gin_sdma_reducescatter::reduceScatterSliceOffset(rank, count);
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  for (size_t i = tid; i < count; i += nthreads) {
    T acc = (T)0;
    for (int s = 0; s < nRanks; ++s) {
      const T* src = send + (size_t)s * totalPerPeer + myOff;
      acc = acc + src[i];
    }
    recv[myOff + i] = acc;
  }
}

template <typename T>
void runReadReduceCase(int nRanks, size_t count) {
  const size_t perPeer = (size_t)nRanks * count;
  const size_t sendElts = (size_t)nRanks * perPeer;

  std::vector<T> send(sendElts);
  for (int peer = 0; peer < nRanks; ++peer)
    for (int owner = 0; owner < nRanks; ++owner)
      for (size_t i = 0; i < count; ++i)
        send[(size_t)peer * perPeer + (size_t)owner * count + i] =
            (T)((peer * 9973 + owner * 131 + (uint32_t)i) & 0x7f);

  std::vector<T> expected((size_t)nRanks * count, (T)0);
  for (int owner = 0; owner < nRanks; ++owner)
    for (size_t i = 0; i < count; ++i) {
      T acc = (T)0;
      for (int s = 0; s < nRanks; ++s)
        acc = acc + send[(size_t)s * perPeer + (size_t)owner * count + i];
      expected[(size_t)owner * count + i] = acc;
    }

  T *dSend = nullptr, *dRecv = nullptr;
  HIP_OK(hipMalloc(&dSend, send.size() * sizeof(T)));
  HIP_OK(hipMalloc(&dRecv, expected.size() * sizeof(T)));
  HIP_OK(hipMemset(dRecv, 0, expected.size() * sizeof(T)));
  HIP_OK(hipMemcpy(dSend, send.data(), send.size() * sizeof(T), hipMemcpyHostToDevice));

  rsReadReduceSimKernel<T><<<nRanks, 256>>>(dSend, dRecv, nRanks, count);
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

TEST(ReduceScatterGpu, ReadReduceAddressingInt32) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  for (int nr : {2, 4, 8})
    for (size_t c : {(size_t)1, (size_t)17, (size_t)256, (size_t)1024})
      runReadReduceCase<int32_t>(nr, c);
}

TEST(ReduceScatterGpu, ReadReduceAddressingInt8) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  for (int nr : {2, 4, 8})
    for (size_t c : {(size_t)1, (size_t)129, (size_t)1024})
      runReadReduceCase<int8_t>(nr, c);
}

TEST(ReduceScatterGpu, SliceBaseCountMatchesPackLayout) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  // sliceBaseCount alignment must divide count evenly for VEC=4 (int32 packs).
  for (size_t total : {4096u, 4095u, 100u, 104u}) {
    const size_t base = gin_sdma_reducescatter::sliceBaseCount(total, 4, 8);
    EXPECT_EQ(base % 4, 0u) << "total=" << total;
    EXPECT_LE(base * 8, total) << "total=" << total;
  }
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

template <class T>
__global__ void reduceFoldKernel(const float* in, int n, int op, int rankN, float* out) {
  T acc = gin_sdma_reduce::preOp<T>(op, (T)in[0], rankN);
  for (int i = 1; i < n; ++i) {
    acc = gin_sdma_reduce::combine<T>(op, acc, gin_sdma_reduce::preOp<T>(op, (T)in[i], rankN));
  }
  *out = (float)gin_sdma_reduce::postOp<T>(op, acc, rankN);
}

template <class T>
void runReduceFoldCase(const char* tname, float b, ncclRedOp_t op, const char* opName,
                       float expected) {
  const std::vector<float> in{b, 1, 1, 1, 1, 1, 1, 1};
  float* dIn = nullptr;
  float* dOut = nullptr;
  HIP_OK(hipMalloc(&dIn, in.size() * sizeof(float)));
  HIP_OK(hipMalloc(&dOut, sizeof(float)));
  HIP_OK(hipMemcpy(dIn, in.data(), in.size() * sizeof(float), hipMemcpyHostToDevice));

  reduceFoldKernel<T><<<1, 1>>>(dIn, (int)in.size(), (int)op, (int)in.size(), dOut);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());

  float got = 0.0f;
  HIP_OK(hipMemcpy(&got, dOut, sizeof(float), hipMemcpyDeviceToHost));
  // Exact compare: every expected value is representable in the element type
  // and converts back to float without loss, so a tolerance would only hide
  // the one-ulp difference this test is looking for.
  EXPECT_EQ(got, expected) << tname << " " << opName;

  HIP_OK(hipFree(dIn));
  HIP_OK(hipFree(dOut));
}

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
  runReduceFoldCase<float>("float", 512, ncclSum, "sum", 519.0f);
  runReduceFoldCase<float>("float", 512, ncclProd, "prod", 512.0f);
  runReduceFoldCase<float>("float", 512, ncclMax, "max", 512.0f);
  runReduceFoldCase<float>("float", 512, ncclMin, "min", 1.0f);
  runReduceFoldCase<float>("float", 512, ncclAvg, "avg", 64.875f);

  runReduceFoldCase<int32_t>("int32", 512, ncclSum, "sum", 519.0f);
  runReduceFoldCase<int32_t>("int32", 512, ncclProd, "prod", 512.0f);
  runReduceFoldCase<int32_t>("int32", 512, ncclMax, "max", 512.0f);
  runReduceFoldCase<int32_t>("int32", 512, ncclMin, "min", 1.0f);
  runReduceFoldCase<int32_t>("int32", 512, ncclAvg, "avg", 64.0f);

  // half: 10 mantissa bits, ulp(4096) = 4, so 4096+1 stays 4096 seven times.
  runReduceFoldCase<__half>("half", 4096, ncclSum, "sum", 4096.0f);
  runReduceFoldCase<__half>("half", 4096, ncclProd, "prod", 4096.0f);
  runReduceFoldCase<__half>("half", 4096, ncclMax, "max", 4096.0f);
  runReduceFoldCase<__half>("half", 4096, ncclMin, "min", 1.0f);
  runReduceFoldCase<__half>("half", 4096, ncclAvg, "avg", 512.0f);

#if HAVE_BF16
  // bf16: 7 mantissa bits, ulp(512) = 4.
  runReduceFoldCase<hip_bfloat16>("bf16", 512, ncclSum, "sum", 512.0f);
  runReduceFoldCase<hip_bfloat16>("bf16", 512, ncclProd, "prod", 512.0f);
  runReduceFoldCase<hip_bfloat16>("bf16", 512, ncclMax, "max", 512.0f);
  runReduceFoldCase<hip_bfloat16>("bf16", 512, ncclMin, "min", 1.0f);
  runReduceFoldCase<hip_bfloat16>("bf16", 512, ncclAvg, "avg", 64.0f);
#endif

#if HAVE_FP8
  // fp8: 3 and 2 mantissa bits, ulp(32) = 4 and 8. prod is left out because
  // SPECIALIZE_REDUCE_KERNEL never dispatches it for the fp8 types. 32, 4 and
  // 1 are exact in both the fnuz and the IEEE encodings, so gfx942 and gfx950
  // agree even though rccl_float8.h typedefs different types for them.
  runReduceFoldCase<rccl_float8>("fp8_e4m3", 32, ncclSum, "sum", 32.0f);
  runReduceFoldCase<rccl_float8>("fp8_e4m3", 32, ncclMax, "max", 32.0f);
  runReduceFoldCase<rccl_float8>("fp8_e4m3", 32, ncclMin, "min", 1.0f);
  runReduceFoldCase<rccl_float8>("fp8_e4m3", 32, ncclAvg, "avg", 4.0f);

  runReduceFoldCase<rccl_bfloat8>("fp8_e5m2", 32, ncclSum, "sum", 32.0f);
  runReduceFoldCase<rccl_bfloat8>("fp8_e5m2", 32, ncclMax, "max", 32.0f);
  runReduceFoldCase<rccl_bfloat8>("fp8_e5m2", 32, ncclMin, "min", 1.0f);
  runReduceFoldCase<rccl_bfloat8>("fp8_e5m2", 32, ncclAvg, "avg", 4.0f);
#endif
}

#endif  // NCCL_VERSION_CODE >= 2.28

}  // namespace
