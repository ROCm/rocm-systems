/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Policy of the LSA AllToAll lane (alltoall.cu, -D 3): when it runs instead of
// gin.put, and its launch geometry. Free of ncclDevComm, GPU intrinsics and the
// environment, so the lane kernel and the host unit test
// (alltoall_lsa_lane_host_test.cpp) run the same code.

#ifndef RCCL_TESTS_ALLTOALL_LSA_LANE_POLICY_H_
#define RCCL_TESTS_ALLTOALL_LSA_LANE_POLICY_H_

#include <cstddef>
#include <cstdint>

#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(ALLTOALL_LSA_LANE_HOST_ONLY)
#define ALLTOALL_LSA_LANE_HD __host__ __device__
#else
#define ALLTOALL_LSA_LANE_HD
#endif

namespace alltoall_lsa {

// CTA (peer, chunk) pairs share one pool, and each needs its own LL flag block.
constexpr int maxCtas = 64;
constexpr int maxChunksPerPeer = 8;
constexpr size_t vectorBytes = 16;

struct Band {
  size_t maxBytesPerPeer;
  int chunksPerPeer;
  int threadsPerCta;
};

// Every band uses 256 threads, the block size defaultThreshold was measured with.
constexpr Band bands[] = {
    {16 * 1024, 2, 256},
    {64 * 1024, 4, 256},
    {512 * 1024, 8, 256},
    {~(size_t)0, 8, 256},
};

ALLTOALL_LSA_LANE_HD inline size_t divUp(size_t a, size_t b) { return (a + b - 1) / b; }
ALLTOALL_LSA_LANE_HD inline size_t alignUp(size_t a, size_t b) { return divUp(a, b) * b; }

ALLTOALL_LSA_LANE_HD inline void launchConfig(int nRanks, size_t bytesPerPeer, int* chunksPerPeer, int* threads,
                                              size_t* chunkBytes) {
  const Band* band = bands;
  while (bytesPerPeer >= band->maxBytesPerPeer) band++;

  int maxChunks = maxCtas / nRanks;
  if (maxChunks < 1) maxChunks = 1;
  const int chunks = band->chunksPerPeer > maxChunks ? maxChunks : band->chunksPerPeer;

  *chunksPerPeer = chunks;
  *threads = band->threadsPerCta;
  *chunkBytes = alignUp(divUp(bytesPerPeer, (size_t)chunks), vectorBytes);
}

// LL flag blocks the lane needs, one per CTA index (ncclLLA2ACreateRequirement's nBlocks).
ALLTOALL_LSA_LANE_HD inline int flagBlocks(int nRanks) {
  const int pool = nRanks * maxChunksPerPeer;
  return pool > maxCtas ? maxCtas : pool;
}

// The lane stores straight into every peer's receive window, so the LSA team
// must cover the whole communicator.
ALLTOALL_LSA_LANE_HD inline bool capable(int nRanks, int lsaSize) {
  return nRanks >= 2 && nRanks <= maxCtas && lsaSize == nRanks;
}

// Total-size gate (nRanks * per-peer bytes). Per-peer sizes that are not a
// multiple of 16 bytes go to gin.put: the lane was tuned on vector copies only.
ALLTOALL_LSA_LANE_HD inline bool sizeOk(int nRanks, size_t bytesPerPeer, size_t threshold) {
  if (threshold == 0) return false;
  if (nRanks < 2 || nRanks > maxCtas) return false;
  if (bytesPerPeer == 0 || bytesPerPeer % vectorBytes != 0) return false;
  return (size_t)nRanks * bytesPerPeer <= threshold;
}

enum class Algo : int { Put = 0, Lsa = 1 };

// laneOn: every rank opted in with the same threshold and is capable.
ALLTOALL_LSA_LANE_HD inline Algo select(bool laneOn, int nRanks, size_t bytesPerPeer, size_t threshold) {
  return laneOn && sizeOk(nRanks, bytesPerPeer, threshold) ? Algo::Lsa : Algo::Put;
}

// Default total-size threshold. On MI450 the lane's crossover with gin.put
// grows with the rank count: between 4 and 8 MiB total at 2 ranks, 16 MiB at 4
// (1P4G) and 64 MiB at 8 (2P8G), which nRanks^2 MiB follows. At 16 ranks
// (4P16G) the lane beats gin.put at every size measured, up to 4 GiB, so there
// is no cap at uncappedRanks. Above it each peer gets fewer of the lane's 64
// CTAs (64 / nRanks), and no larger clique has been measured, so those keep
// nRanks^2 MiB.
constexpr size_t defaultThresholdUnit = (size_t)1024 * 1024;
constexpr int uncappedRanks = 16;

ALLTOALL_LSA_LANE_HD inline size_t defaultThreshold(int nRanks) {
  if (nRanks == uncappedRanks) return ~(size_t)0;
  return (size_t)nRanks * (size_t)nRanks * defaultThresholdUnit;
}

// This rank's threshold: envThreshold (RCCL_TESTS_A2A_LSA_THRESHOLD, negative
// when unset) or the default, and 0 when the rank did not opt in.
ALLTOALL_LSA_LANE_HD inline size_t threshold(bool optIn, int64_t envThreshold, int nRanks) {
  if (!optIn || nRanks < 2) return 0;
  return envThreshold >= 0 ? (size_t)envThreshold : defaultThreshold(nRanks);
}

}  // namespace alltoall_lsa

#undef ALLTOALL_LSA_LANE_HD

#endif  // RCCL_TESTS_ALLTOALL_LSA_LANE_POLICY_H_
