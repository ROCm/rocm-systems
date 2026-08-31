/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// On-GPU correctness test for the GIN Anvil-SDMA hybrid AllGather (-D 3,
// GinHybridAllGatherKernel). AllGather performs no reduction, so unlike the
// AllReduce reduce-ops GPU test there is no low-precision rounding to validate;
// the two things worth exercising on real hardware are:
//
//   (A) the LSA<->SDMA tier predicate (gin_sdma_allgather::chunkUsesLsaTier)
//       evaluates identically on the device (where GinHybridAllGatherKernel runs
//       it) and on the host, and
//
//   (B) the direct-LSA gather addressing that AllGatherLsaDirect implements --
//       each rank r reads src[i] and stores it into every peer's recv buffer at
//       slot (r*count + i) -- reproduces the AllGather golden layout on hardware.
//
// The addressing kernel below is the pointer-arithmetic core of AllGatherLsaDirect
// with the GIN window peer pointers replaced by plain device buffers, so it needs
// neither librccl nor a GIN communicator and runs on any single GPU. Crucially it
// derives the per-rank recv-slice offset from the SAME shared helper the production
// kernel uses (gin_sdma_allgather::allGatherRecvSliceOffset), so a wrong slice
// formula now breaks this test rather than only being self-consistent. What it does
// NOT cover is the GIN window resolution itself (ncclGetLsaPointer / recvoffset),
// which needs a live communicator; the full GIN put/signal path and end-to-end
// datacheck are covered separately by all_gather_perf -D 3 in gin-sdma-ag-test.bash.
//
// Requires a visible GPU at run time; skips cleanly (GTEST_SKIP) otherwise, so it
// carries a "gpu" label alongside "unit".

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include <hip/hip_runtime.h>

#include "gin_sdma_allgather_policy.h"

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

// ---- (A) tier predicate: device must match host --------------------------------

__global__ void tierKernel(const uint64_t* chunk, const uint64_t* thr, uint8_t* out, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    out[i] = gin_sdma_allgather::chunkUsesLsaTier((size_t)chunk[i], (size_t)thr[i]) ? 1u : 0u;
  }
}

TEST(AllGatherGpu, TierPredicateMatchesHost) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";

  // Absolute semantics at the compiled default crossover (not just host/device parity).
  EXPECT_TRUE(gin_sdma_allgather::chunkUsesLsaTier(
      32768, gin_sdma_allgather::kAllGatherSdmaThresholdDefault));
  EXPECT_FALSE(gin_sdma_allgather::chunkUsesLsaTier(
      32769, gin_sdma_allgather::kAllGatherSdmaThresholdDefault));

  std::vector<uint64_t> chunk, thr;
  const uint64_t thresholds[] = {0, 128, 2097152, 16777216};
  for (uint64_t t : thresholds) {
    for (uint64_t d = 0; d <= 4; ++d) {
      // probe below/at/above each threshold and a couple absolute sizes
      chunk.push_back(t == 0 ? d : (t - 2 + d));  // t-2..t+2 (clamped for t=0)
      thr.push_back(t);
    }
    chunk.push_back(1);         thr.push_back(t);
    chunk.push_back(1u << 20);  thr.push_back(t);
  }
  const int n = (int)chunk.size();

  uint64_t *dChunk = nullptr, *dThr = nullptr;
  uint8_t* dOut = nullptr;
  HIP_OK(hipMalloc(&dChunk, n * sizeof(uint64_t)));
  HIP_OK(hipMalloc(&dThr, n * sizeof(uint64_t)));
  HIP_OK(hipMalloc(&dOut, n * sizeof(uint8_t)));
  HIP_OK(hipMemcpy(dChunk, chunk.data(), n * sizeof(uint64_t), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dThr, thr.data(), n * sizeof(uint64_t), hipMemcpyHostToDevice));

  tierKernel<<<(n + 63) / 64, 64>>>(dChunk, dThr, dOut, n);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());

  std::vector<uint8_t> out(n, 0xff);
  HIP_OK(hipMemcpy(out.data(), dOut, n * sizeof(uint8_t), hipMemcpyDeviceToHost));

  for (int i = 0; i < n; ++i) {
    const uint8_t host = gin_sdma_allgather::chunkUsesLsaTier((size_t)chunk[i], (size_t)thr[i]) ? 1u : 0u;
    EXPECT_EQ(out[i], host) << "chunk=" << chunk[i] << " thr=" << thr[i];
  }

  HIP_OK(hipFree(dChunk));
  HIP_OK(hipFree(dThr));
  HIP_OK(hipFree(dOut));
}

// ---- (B) direct-LSA gather addressing: reproduces the AllGather layout ---------
//
// Mirrors AllGatherLsaDirect (all_gather.cu): rank r's element i is stored into
// every peer's recv buffer at dstOff = r*count. recv is laid out as
// recv[peer*(nRanks*count) + r*count + i]; send as send[r*count + i]. After every
// (rank, element) is written to every peer, each peer's slice must equal the
// gather of all ranks' send data.

template <typename T>
__global__ void agLsaDirectSimKernel(const T* send, T* recv, int nRanks, size_t count) {
  const size_t total = (size_t)nRanks * count;              // elements per peer recv
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nthreads = blockDim.x * gridDim.x;
  // Flatten (rank, element) work items; each writes to all peers.
  for (size_t w = tid; w < total; w += nthreads) {
    const int r = (int)(w / count);
    const size_t i = w % count;
    // Shared with production AllGatherLsaDirect: rank r's slice within each peer.
    const size_t dstOff = gin_sdma_allgather::allGatherRecvSliceOffset(r, count);
    const T value = send[dstOff + i];
    for (int lp = 0; lp < nRanks; ++lp) {
      T* dst = recv + (size_t)lp * total + dstOff;
      dst[i] = value;
    }
  }
}

template <typename T>
void runGatherAddressingCase(int nRanks, size_t count) {
  const size_t perPeer = (size_t)nRanks * count;
  const size_t recvElts = (size_t)nRanks * perPeer;

  std::vector<T> send((size_t)nRanks * count);
  for (int r = 0; r < nRanks; ++r)
    for (size_t i = 0; i < count; ++i)
      send[r * count + i] = (T)((r * 1315423911u + (uint32_t)i) & 0x7f);  // fits int8..double

  T *dSend = nullptr, *dRecv = nullptr;
  HIP_OK(hipMalloc(&dSend, send.size() * sizeof(T)));
  HIP_OK(hipMalloc(&dRecv, recvElts * sizeof(T)));
  HIP_OK(hipMemset(dRecv, 0, recvElts * sizeof(T)));
  HIP_OK(hipMemcpy(dSend, send.data(), send.size() * sizeof(T), hipMemcpyHostToDevice));

  agLsaDirectSimKernel<T><<<8, 256>>>(dSend, dRecv, nRanks, count);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());

  std::vector<T> recv(recvElts);
  HIP_OK(hipMemcpy(recv.data(), dRecv, recvElts * sizeof(T), hipMemcpyDeviceToHost));

  size_t wrong = 0;
  for (int lp = 0; lp < nRanks; ++lp)
    for (int r = 0; r < nRanks; ++r)
      for (size_t i = 0; i < count; ++i) {
        const T got = recv[(size_t)lp * perPeer + (size_t)r * count + i];
        const T exp = send[r * count + i];
        if (got != exp) ++wrong;
      }
  EXPECT_EQ(wrong, 0u) << "nRanks=" << nRanks << " count=" << count
                       << " eltSize=" << sizeof(T);

  HIP_OK(hipFree(dSend));
  HIP_OK(hipFree(dRecv));
}

TEST(AllGatherGpu, LsaDirectAddressingInt32) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  for (int nr : {2, 3, 4, 8})
    for (size_t c : {(size_t)1, (size_t)17, (size_t)256, (size_t)1024})
      runGatherAddressingCase<int32_t>(nr, c);
}

TEST(AllGatherGpu, LsaDirectAddressingInt8) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  for (int nr : {2, 4, 8})
    for (size_t c : {(size_t)1, (size_t)129, (size_t)1024})
      runGatherAddressingCase<int8_t>(nr, c);
}

TEST(AllGatherGpu, LsaDirectAddressingDouble) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  for (int nr : {2, 4, 8})
    for (size_t c : {(size_t)1, (size_t)64, (size_t)512})
      runGatherAddressingCase<double>(nr, c);
}

}  // namespace
