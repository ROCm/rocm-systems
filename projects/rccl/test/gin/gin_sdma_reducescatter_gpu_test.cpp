/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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
//       at offset R*count + i and folds in ascending source-rank order.
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
  const size_t myOff = (size_t)rank * count;
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

}  // namespace
