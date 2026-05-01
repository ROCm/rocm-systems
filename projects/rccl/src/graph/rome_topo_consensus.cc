/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "graph/rome_topo_consensus.h"
#include "debug.h"
#include <unordered_map>

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
ncclResult_t rcclCheckRomeTopoModelIdxConsensus(
  int nranks,
  std::function<int(int)> getRomeTopoModelIdx,
  std::function<const char*(int)> getHostname,
  std::function<uint64_t(int)> getHostHash) {
  if (nranks <= 0) return ncclSuccess;

  std::unordered_map<int, std::pair<int, int>> tallies; // modelIdx -> (vote count, first rank)
  tallies.reserve(nranks);
  for (int r = 0; r < nranks; r++) {
    int v = getRomeTopoModelIdx(r);
    auto it = tallies.find(v);
    if (it == tallies.end()) {
      tallies.emplace(v, std::make_pair(1, r));
    } else {
      it->second.first++;
    }
  }
  int refIdx = getRomeTopoModelIdx(0);
  int refVotes = -1;
  int refFirstRank = nranks;
  for (const auto& e : tallies) {
    int cnt = e.second.first;
    int firstRank = e.second.second;
    if (cnt > refVotes || (cnt == refVotes && firstRank < refFirstRank)) {
      refVotes = cnt;
      refIdx = e.first;
      refFirstRank = firstRank;
    }
  }
  if (tallies.size() == 1) return ncclSuccess;
  int nDisagree = 0;
  for (int r = 0; r < nranks; r++) {
    if (getRomeTopoModelIdx(r) != refIdx) nDisagree++;
  }
  if (nDisagree > 0) {
    WARN("RCCL FATAL: mismatched Rome preset topology model index across ranks; all ranks must agree for precomputed graphs (voted refIdx %d from %d of %d ranks).", refIdx, refVotes, nranks);
    std::unordered_map<uint64_t, int> lowestMismatchRankByHost;
    lowestMismatchRankByHost.reserve(nranks);
    for (int r = 0; r < nranks; r++) {
      if (getRomeTopoModelIdx(r) == refIdx) continue;
      uint64_t h = getHostHash(r);
      if (lowestMismatchRankByHost.find(h) == lowestMismatchRankByHost.end()) {
        lowestMismatchRankByHost.emplace(h, r);
      }
    }
    for (int r = 0; r < nranks; r++) {
      if (getRomeTopoModelIdx(r) == refIdx) continue;
      uint64_t h = getHostHash(r);
      if (lowestMismatchRankByHost[h] != r) continue;
      WARN("  rank %d host %s romeTopoModelIdx=%d", r, getHostname(r), getRomeTopoModelIdx(r));
    }
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}
