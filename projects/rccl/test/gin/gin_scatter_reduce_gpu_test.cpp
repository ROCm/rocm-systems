/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Single-GPU check of ginScatterReduceChunk. Fills a fake incoming scratch
// with known per-peer patterns and compares the reduced slice to a host sum.
// Covers staged and full-column layouts, including a chunk large enough that
// the 56x512 unroll used to reduce the tail twice. No librccl and no GIN
// communicator. Skips when no GPU is visible.

#include <gtest/gtest.h>

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

void runReduceCase(bool staged, size_t elemStart, size_t nVec, int rank, int nRanks) {
  constexpr int W = static_cast<int>(sizeof(uint4) / sizeof(float));
  const size_t chunkElems = nVec * static_cast<size_t>(W);
  const size_t elemEnd = elemStart + chunkElems;
  const size_t strideElems = staged ? chunkElems : elemEnd;
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
                       << " nRanks=" << nRanks;

  HIP_OK(hipFree(dSelf));
  HIP_OK(hipFree(dIncoming));
  HIP_OK(hipFree(dOut));
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
}
