/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Single-GPU check of ginScatterReduceChunk. Fills a fake incoming scratch
// with known per-peer patterns and compares the reduced slice to a host sum.
// Covers staged and full-column layouts, including a chunk large enough that
// the 56x512 unroll used to reduce the tail twice, and multi-rank sums at the
// 64 MiB and 1 GiB GinScatter sizes. No librccl and no GIN communicator.
// Skips when no GPU is visible.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <hip/hip_runtime.h>

#include "algorithms/gin/gin_all_reduce_policy.h"
#include "algorithms/gin/sdma/gin_scatter_reduce.h"

namespace {

#define HIP_OK(cmd)                                                                              \
  do {                                                                                           \
    hipError_t _e = (cmd);                                                                       \
    ASSERT_EQ(_e, hipSuccess) << "HIP error " << (int)_e << " (" << hipGetErrorString(_e)       \
                              << ") at " << __FILE__ << ":" << __LINE__;                         \
  } while (0)

bool gpuAvailable() {
  int n = 0;
  hipError_t e = hipGetDeviceCount(&n);
  return e == hipSuccess && n > 0;
}

__global__ void ginScatterReduceChunkKernel(const float* selfSend, const float* incoming, float* reducedOut, int rank,
                                            int nRanks, size_t incomingStride, size_t elemStart, size_t elemEnd,
                                            bool staged) {
  gin::sdma::ginScatterReduceChunk<float>(reinterpret_cast<const char*>(selfSend),
                                          reinterpret_cast<const char*>(incoming),
                                          reinterpret_cast<char*>(reducedOut), rank, nRanks, incomingStride, elemStart,
                                          elemEnd, staged);
}

float hostSum(const std::vector<float>& selfSend, const std::vector<float>& incoming, int rank, int nRanks,
              size_t strideElems, size_t elem, size_t elemStart, bool staged) {
  float sum = 0.f;
  for (int peer = 0; peer < nRanks; ++peer) {
    if (peer == rank) {
      sum += selfSend[elem];
    } else if (staged) {
      sum += incoming[static_cast<size_t>(peer) * strideElems + (elem - elemStart)];
    } else {
      sum += incoming[static_cast<size_t>(peer) * strideElems + elem];
    }
  }
  return sum;
}

void runReduceCase(bool staged, size_t elemStart, size_t nVec, int rank, int nRanks, size_t extraStrideElems = 0) {
  constexpr int W = static_cast<int>(sizeof(uint4) / sizeof(float));
  const size_t chunkElems = nVec * static_cast<size_t>(W);
  const size_t elemEnd = elemStart + chunkElems;
  // Production passes slotBytes or perRankBytes, which exceed the last chunk.
  const size_t strideElems = (staged ? chunkElems : elemEnd) + extraStrideElems;
  const size_t incomingStride = strideElems * sizeof(float);

  std::vector<float> selfSend(elemEnd, 0.f);
  std::vector<float> incoming(static_cast<size_t>(nRanks) * strideElems, -1.f);
  for (size_t elem = elemStart; elem < elemEnd; ++elem) {
    selfSend[elem] = static_cast<float>(rank * 1000 + static_cast<int>(elem % 64));
    for (int peer = 0; peer < nRanks; ++peer) {
      if (peer == rank) continue;
      const size_t src = staged ? static_cast<size_t>(peer) * strideElems + (elem - elemStart)
                                : static_cast<size_t>(peer) * strideElems + elem;
      incoming[src] = static_cast<float>(peer * 1000 + static_cast<int>(elem % 64));
    }
  }

  float *dSelf = nullptr, *dIncoming = nullptr, *dOut = nullptr;
  HIP_OK(hipMalloc(&dSelf, selfSend.size() * sizeof(float)));
  HIP_OK(hipMalloc(&dIncoming, incoming.size() * sizeof(float)));
  HIP_OK(hipMalloc(&dOut, elemEnd * sizeof(float)));
  HIP_OK(hipMemcpy(dSelf, selfSend.data(), selfSend.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(dIncoming, incoming.data(), incoming.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_OK(hipMemset(dOut, 0, elemEnd * sizeof(float)));

  ginScatterReduceChunkKernel<<<kGinAllReduceLsaCtas, kGinAllReduceLsaThreadsPerCta>>>(
    dSelf, dIncoming, dOut, rank, nRanks, incomingStride, elemStart, elemEnd, staged);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());

  std::vector<float> got(elemEnd, 1.f);
  HIP_OK(hipMemcpy(got.data(), dOut, elemEnd * sizeof(float), hipMemcpyDeviceToHost));

  size_t wrong = 0;
  for (size_t elem = 0; elem < elemStart; ++elem) {
    if (got[elem] != 0.f) ++wrong;
  }
  for (size_t elem = elemStart; elem < elemEnd; ++elem) {
    const float exp = hostSum(selfSend, incoming, rank, nRanks, strideElems, elem, elemStart, staged);
    if (got[elem] != exp) ++wrong;
  }
  EXPECT_EQ(wrong, 0u) << "staged=" << staged << " elemStart=" << elemStart << " nVec=" << nVec << " rank=" << rank
                       << " nRanks=" << nRanks << " strideElems=" << strideElems;

  HIP_OK(hipFree(dSelf));
  HIP_OK(hipFree(dIncoming));
  HIP_OK(hipFree(dOut));
}

struct DeviceFree {
  void* ptr = nullptr;
  ~DeviceFree() {
    if (ptr != nullptr) hipFree(ptr);
  }
};

float peerPattern(int peer, size_t elem) {
  return static_cast<float>((peer + 1) * 17 + static_cast<int>(elem & 63));
}

__global__ void fillRankPattern(float* dst, size_t n, int peer, size_t elemBase) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = static_cast<float>((peer + 1) * 17 + static_cast<int>((elemBase + i) & 63));
}

void launchFill(float* dst, size_t n, int peer, size_t elemBase) {
  if (n == 0 || dst == nullptr) return;
  constexpr int kThreads = 256;
  const unsigned blocks = static_cast<unsigned>((n + static_cast<size_t>(kThreads) - 1) / kThreads);
  fillRankPattern<<<blocks, kThreads>>>(dst, n, peer, elemBase);
}

// One ginScatterReduceChunk at a production size. strideBytes is the incoming
// column (slotBytes when staged, per-rank bytes when not). The chunk itself
// may be shorter than that column.
void runProductionSum(int nRanks, size_t elemStart, size_t chunkBytes, size_t strideBytes, bool staged) {
  constexpr int W = static_cast<int>(sizeof(uint4) / sizeof(float));
  ASSERT_EQ(chunkBytes % sizeof(uint4), 0u);
  ASSERT_EQ(strideBytes % sizeof(float), 0u);
  ASSERT_EQ(elemStart % static_cast<size_t>(W), 0u);
  const size_t chunkElems = chunkBytes / sizeof(float);
  const size_t strideElems = strideBytes / sizeof(float);
  const size_t elemEnd = elemStart + chunkElems;
  ASSERT_GE(strideElems, staged ? chunkElems : elemEnd);

  DeviceFree self;
  DeviceFree incoming;
  DeviceFree out;
  hipError_t err = hipMalloc(&self.ptr, elemEnd * sizeof(float));
  if (err != hipSuccess) {
    ADD_FAILURE() << "hipMalloc self: " << hipGetErrorString(err);
    (void)hipGetLastError();
    return;
  }
  err = hipMalloc(&incoming.ptr, static_cast<size_t>(nRanks) * strideElems * sizeof(float));
  if (err != hipSuccess) {
    ADD_FAILURE() << "hipMalloc incoming: " << hipGetErrorString(err);
    (void)hipGetLastError();
    return;
  }
  err = hipMalloc(&out.ptr, elemEnd * sizeof(float));
  if (err != hipSuccess) {
    ADD_FAILURE() << "hipMalloc out: " << hipGetErrorString(err);
    (void)hipGetLastError();
    return;
  }

  float* dSelf = static_cast<float*>(self.ptr);
  float* dIncoming = static_cast<float*>(incoming.ptr);
  float* dOut = static_cast<float*>(out.ptr);
  // Poison the incoming columns. The self column must not be read.
  HIP_OK(hipMemset(dIncoming, 0x7f, static_cast<size_t>(nRanks) * strideElems * sizeof(float)));
  // The kernel indexes self by absolute element, so the buffer covers
  // [0, elemEnd) and only the chunk range holds this rank's pattern.
  const int rank = nRanks > 1 ? 1 : 0;
  launchFill(dSelf + elemStart, chunkElems, rank, elemStart);
  for (int peer = 0; peer < nRanks; ++peer) {
    if (peer == rank) continue;
    launchFill(dIncoming + static_cast<size_t>(peer) * strideElems, strideElems, peer, staged ? elemStart : 0);
  }
  HIP_OK(hipMemset(dOut, 0, elemEnd * sizeof(float)));
  HIP_OK(hipGetLastError());

  ginScatterReduceChunkKernel<<<kGinAllReduceLsaCtas, kGinAllReduceLsaThreadsPerCta>>>(
    dSelf, dIncoming, dOut, rank, nRanks, strideBytes, elemStart, elemEnd, staged);
  HIP_OK(hipGetLastError());
  HIP_OK(hipDeviceSynchronize());

  if (elemStart > 0) {
    float head = 1.f;
    float before = 1.f;
    HIP_OK(hipMemcpy(&head, dOut, sizeof(float), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(&before, dOut + elemStart - 1, sizeof(float), hipMemcpyDeviceToHost));
    EXPECT_EQ(head, 0.f);
    EXPECT_EQ(before, 0.f);
  }

  constexpr size_t kTile = 1u << 20;
  std::vector<float> got(std::min(kTile, chunkElems));
  size_t wrong = 0;
  for (size_t base = elemStart; base < elemEnd; base += kTile) {
    const size_t n = std::min(kTile, elemEnd - base);
    HIP_OK(hipMemcpy(got.data(), dOut + base, n * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
      const size_t elem = base + i;
      float exp = 0.f;
      for (int peer = 0; peer < nRanks; ++peer) exp += peerPattern(peer, elem);
      if (got[i] != exp) ++wrong;
    }
  }
  EXPECT_EQ(wrong, 0u) << "staged=" << staged << " elemStart=" << elemStart << " chunkBytes=" << chunkBytes
                       << " strideBytes=" << strideBytes << " nRanks=" << nRanks;
}

} // namespace

TEST(GinScatterReduceGpu, ChunkMatchesHostSum) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";

  const size_t stride = static_cast<size_t>(kGinAllReduceLsaCtas) * static_cast<size_t>(kGinAllReduceLsaThreadsPerCta);
  const size_t span = stride * 4u;
  // One full unroll span plus a tail. The old tail restarted inside that span.
  const size_t overlapping = span + 1024u;

  runReduceCase(/*staged=*/false, /*elemStart=*/0, overlapping, /*rank=*/1, /*nRanks=*/4);
  runReduceCase(/*staged=*/true, /*elemStart=*/256, overlapping, /*rank=*/0, /*nRanks=*/4);
  runReduceCase(/*staged=*/false, /*elemStart=*/0, /*nVec=*/100, /*rank=*/0, /*nRanks=*/2);
  runReduceCase(/*staged=*/true, /*elemStart=*/64, /*nVec=*/100, /*rank=*/1, /*nRanks=*/3);
  // Stride strictly larger than the chunk, matching a last-chunk slot or column.
  runReduceCase(/*staged=*/true, /*elemStart=*/256, overlapping, /*rank=*/0, /*nRanks=*/4, /*extraStrideElems=*/64);
  runReduceCase(/*staged=*/false, /*elemStart=*/0, /*nVec=*/100, /*rank=*/1, /*nRanks=*/4, /*extraStrideElems=*/32);
}

TEST(GinScatterReduceGpu, MultiRankSumAt64MiBAnd1GiB) {
  if (!gpuAvailable()) GTEST_SKIP() << "no visible GPU";
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  // Last 1 GiB slot keeps a 128 MiB output plus 256 MiB of incoming columns.
  if (hipMemGetInfo(&freeBytes, &totalBytes) != hipSuccess || freeBytes < (640ull << 20)) {
    GTEST_SKIP() << "need about 640 MiB free GPU memory for the 64 MiB and 1 GiB sums";
  }

  constexpr int kRanks = 8;
  constexpr size_t kScratch = kGinAllReduceTwoShotScratchBytes;
  constexpr size_t k64MiB = 64ull * 1024ull * 1024ull;
  constexpr size_t k1GiB = 1024ull * 1024ull * 1024ull;

  // 64 MiB fits in one incoming column per rank. 4 and 8 cover the rank loop.
  const size_t perRank4 = k64MiB / 4u;
  ASSERT_FALSE(ginAllReduceGinScatterStaged(perRank4, 4, kScratch));
  runProductionSum(/*nRanks=*/4, /*elemStart=*/0, perRank4, perRank4, /*staged=*/false);
  const size_t perRank8 = k64MiB / static_cast<size_t>(kRanks);
  ASSERT_FALSE(ginAllReduceGinScatterStaged(perRank8, kRanks, kScratch));
  runProductionSum(kRanks, /*elemStart=*/0, perRank8, perRank8, /*staged=*/false);

  // 1 GiB is staged ping-pong. Reduce the first slot and the last slot; the
  // last slot is the reuse offset the kernel writes on a later chunk.
  const size_t perRank = k1GiB / static_cast<size_t>(kRanks);
  int nChunks = 0;
  int nPhases = 0;
  size_t uniformBytes = 0;
  size_t lastBytes = 0;
  ginAllReduceGinScatterChunkPlan(perRank, kRanks, kScratch, &nChunks, &uniformBytes, &lastBytes, &nPhases);
  ASSERT_GT(nChunks, 1);
  ASSERT_EQ(nPhases, kGinAllReduceGinScatterNumSlots);
  runProductionSum(kRanks, /*elemStart=*/0, uniformBytes, uniformBytes, /*staged=*/true);
  const size_t lastElem = ginAllReduceGinScatterChunkOff(nChunks - 1, uniformBytes) / sizeof(float);
  runProductionSum(kRanks, lastElem, lastBytes, uniformBytes, /*staged=*/true);
}
