/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "hier_host.h"

#include "algorithms/dda/fabric/fabric_init.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

// Split children skip DDA fabric init because they have a parent. Without it the
// intra-host and inter-host collectives miss the DDA fast path.
ncclResult_t initChildFabric(struct ncclComm* child) {
  if (child == nullptr || child->nRanks < 2) return ncclSuccess;
  if (child->ddaFabricMemHandler != nullptr) return ncclSuccess;
  if (!ncclDdaUseFabricPath(child)) return ncclSuccess;
  return ncclDdaFabricCommInit(child);
}

} // namespace

bool rcclHostLayoutGet(const struct ncclComm* comm, struct rcclHostLayout* layout) {
  if (comm == nullptr || !comm->peerInfoValid || comm->peerInfo == nullptr || comm->nRanks < 2) return false;
  std::vector<uint64_t> hosts;
  std::vector<int> counts;
  int myHost = -1;
  int myLocal = -1;
  for (int r = 0; r < comm->nRanks; r++) {
    const uint64_t hash = comm->peerInfo[r].hostHash;
    size_t idx = 0;
    while (idx < hosts.size() && hosts[idx] != hash) idx++;
    if (idx == hosts.size()) {
      hosts.push_back(hash);
      counts.push_back(0);
    }
    if (r == comm->rank) {
      myHost = static_cast<int>(idx);
      myLocal = counts[idx];
    }
    counts[idx]++;
  }
  const int ranksPerHost = counts[0];
  if (ranksPerHost < 2 || myHost < 0) return false;
  for (int c : counts) {
    if (c != ranksPerHost) return false;
  }
  layout->nHosts = static_cast<int>(hosts.size());
  layout->ranksPerHost = ranksPerHost;
  layout->hostIndex = myHost;
  layout->localRank = myLocal;
  return true;
}

const struct rcclHostLayout* rcclHostHierGetLayout(struct ncclComm* comm) {
  if (comm == nullptr) return nullptr;
  switch (comm->hostHierState) {
  case rcclHostHierLayout:
  case rcclHostHierReady:
    return &comm->hostHier->layout;
  case rcclHostHierUnprobed:
    break;
  default:
    return nullptr;
  }
  struct rcclHostLayout layout;
  if (!rcclHostLayoutGet(comm, &layout)) {
    comm->hostHierState = rcclHostHierOff;
    return nullptr;
  }
  struct rcclHostHier* h = nullptr;
  if (ncclCalloc(&h, 1) != ncclSuccess) {
    comm->hostHierState = rcclHostHierOff;
    return nullptr;
  }
  h->layout = layout;
  comm->hostHier = h;
  comm->hostHierState = rcclHostHierLayout;
  return &h->layout;
}

ncclResult_t rcclHostHierEnsure(struct ncclComm* comm) {
  if (comm->hostHierState == rcclHostHierReady) return ncclSuccess;
  const struct rcclHostLayout* layout = rcclHostHierGetLayout(comm);
  if (layout == nullptr) return ncclInvalidUsage;
  struct rcclHostHier* h = comm->hostHier;

  // A non-blocking split can return before the child exists, so split synchronously.
  const int parentBlocking = comm->config.blocking;
  comm->config.blocking = 1;
  ncclResult_t res = ncclSuccess;

  NCCLCHECKGOTO(ncclCommSplit(comm, layout->hostIndex, layout->localRank, &h->intra, nullptr), res, fail);
  NCCLCHECKGOTO(ncclCommEnsureReady(h->intra), res, fail);
  h->intra->hostHierState = rcclHostHierOff;
  if (layout->nHosts > 1) {
    NCCLCHECKGOTO(ncclCommSplit(comm, layout->localRank, layout->hostIndex, &h->inter, nullptr), res, fail);
    NCCLCHECKGOTO(ncclCommEnsureReady(h->inter), res, fail);
    h->inter->hostHierState = rcclHostHierOff;
  }
  NCCLCHECKGOTO(initChildFabric(h->intra), res, fail);
  NCCLCHECKGOTO(initChildFabric(h->inter), res, fail);

  comm->config.blocking = parentBlocking;
  comm->hostHierState = rcclHostHierReady;
  INFO(NCCL_INIT, "Host-hierarchical comms: rank %d host %d/%d localRank %d/%d intraDda %d interDda %d", comm->rank,
       layout->hostIndex, layout->nHosts, layout->localRank, layout->ranksPerHost,
       h->intra->ddaFabricMemHandler != nullptr, h->inter != nullptr && h->inter->ddaFabricMemHandler != nullptr);
  return ncclSuccess;

fail:
  comm->config.blocking = parentBlocking;
  // A failed split can leave the parent's bootstrap mid-exchange, so never retry.
  comm->hostHierState = rcclHostHierFailed;
  if (h->inter != nullptr) (void)ncclCommDestroy(h->inter);
  if (h->intra != nullptr) (void)ncclCommDestroy(h->intra);
  h->inter = nullptr;
  h->intra = nullptr;
  WARN("Host-hierarchical comms: split failed on rank %d (%s)", comm->rank, ncclGetErrorString(res));
  return res;
}

ncclResult_t rcclHostHierTeardown(struct ncclComm* comm) {
  struct rcclHostHier* h = comm->hostHier;
  comm->hostHier = nullptr;
  comm->hostHierState = rcclHostHierOff;
  if (h == nullptr) return ncclSuccess;
  ncclResult_t res = ncclSuccess;
  if (h->inter != nullptr) NCCLCHECKGOTO(ncclCommDestroy(h->inter), res, exit);
  if (h->intra != nullptr) NCCLCHECKGOTO(ncclCommDestroy(h->intra), res, exit);
exit:
  free(h);
  return res;
}
