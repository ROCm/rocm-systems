/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Size and alignment policy for GIN-SDMA AllReduce. Pure host header so unit
 * tests can exercise the same gates ncclAllReduceGinSdmaEligible() uses after
 * the comm / buffer / gfx950 checks pass.
 * See LICENSE.txt for license information.
 ******************************************************************************/

#ifndef GIN_ALL_REDUCE_POLICY_H_
#define GIN_ALL_REDUCE_POLICY_H_

#include <cstddef>

constexpr int kGinAllReduceLsaCtas = 56;
constexpr int kGinAllReduceLsaTwoShotCtasPerPeer = 8;
constexpr int kGinAllReduceLsaTwoShotMaxCtas = kGinAllReduceLsaTwoShotCtasPerPeer * 16;
constexpr int kGinAllReduceMaxRanks = 8;

constexpr int kGinAllReduceMinBytes = 512ULL * 1024;
constexpr int kGinAllReduceLsaThreadsPerCta = 512;
constexpr size_t kGinAllReduceLsaOneShotMaxBytes = 8ULL * 1024 * 1024;
constexpr size_t kGinAllReduceLsaTwoShotMidBytes = 32ULL * 1024 * 1024;
constexpr size_t kGinAllReduceGinTwoShotMinBytes = 256ULL * 1024 * 1024;

constexpr size_t kGinAllReduceMinPutBytes = 128;

// Two-shot kernels require a whole number of elements per rank and a 16-byte
// per-rank slice so vector loads stay aligned.
inline bool ginAllReduceTwoShotEligible(size_t count, size_t typeSize, int nRanks) {
  if (nRanks <= 0 || typeSize == 0) return false;
  if (count % static_cast<size_t>(nRanks) != 0) return false;
  const size_t countPerRank = count / static_cast<size_t>(nRanks);
  return (countPerRank * typeSize) % 16 == 0;
}

inline bool ginAllReduceGinTwoShotEligible(size_t count, size_t typeSize, int nRanks) {
  if (!ginAllReduceTwoShotEligible(count, typeSize, nRanks)) return false;
  const size_t chunkBytes = (count / static_cast<size_t>(nRanks)) * typeSize;
  return chunkBytes >= kGinAllReduceMinPutBytes;
}

// Size-only GIN AllReduce policy. forceEnable is RCCL_GIN_ALLREDUCE_FORCE_ENABLE==1.
// Default: true only for messages >= 256 MiB that pass GIN two-shot alignment.
// Force: LSA one-shot <= 8 MiB (and >= 512 KiB), LSA two-shot in between, GIN two-shot at 256 MiB+.
inline bool ginAllReduceSizePolicyEligible(size_t count, size_t typeSize, int nRanks, bool forceEnable) {
  const size_t bytes = count * typeSize;
  if (bytes < kGinAllReduceGinTwoShotMinBytes && !forceEnable) return false;
  if (bytes < static_cast<size_t>(kGinAllReduceMinBytes)) return false;
  if (bytes <= kGinAllReduceLsaOneShotMaxBytes) return true;
  if (bytes >= kGinAllReduceGinTwoShotMinBytes) {
    return ginAllReduceGinTwoShotEligible(count, typeSize, nRanks);
  }
  return ginAllReduceTwoShotEligible(count, typeSize, nRanks);
}

// Smaller-than-256 MiB GIN candidates yield to DDA unless FORCE_ENABLE=1.
inline bool ginAllReduceYieldToDdaBySize(size_t count, size_t typeSize, bool forceEnable) {
  if (forceEnable) return false;
  return count * typeSize < kGinAllReduceGinTwoShotMinBytes;
}

#endif
