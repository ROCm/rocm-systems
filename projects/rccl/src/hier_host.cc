/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "hier_host.h"

#include "algorithms/dda/fabric/fabric_init.h"
#include "alloc.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "param.h"
#include "recorder.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

RCCL_PARAM(HostHierAllReduce, "HOST_HIER_ALLREDUCE", -1);

namespace {

// Split children skip DDA fabric init because they have a parent. Without it the
// intra-host and inter-host collectives miss the DDA fast path.
ncclResult_t initChildFabric(struct ncclComm* child) {
  if (child == nullptr || child->nRanks < 2) return ncclSuccess;
  if (child->ddaFabricMemHandler != nullptr) return ncclSuccess;
  if (!ncclDdaUseFabricPath(child)) return ncclSuccess;
  return ncclDdaFabricCommInit(child);
}

// A destroy syncs the child's streams and waits at its intra-node barrier, which hangs when a peer is gone, so
// abort instead when that can be the case.
ncclResult_t releaseChild(ncclComm_t child, bool abort) {
  if (child == nullptr) return ncclSuccess;
  return abort ? ncclCommAbort(child) : ncclCommDestroy(child);
}

// Each rank runs a setup step on its own, and the next step is collective over the parent or a child. Every rank
// waits for all of them to finish the step, so ranks that failed never wait in a different collective than the
// rest. On failure *res holds this rank's error, or ncclRemoteError when only a peer failed.
bool allRanksReady(struct ncclComm* comm, ncclResult_t* res) {
  std::vector<uint8_t> ready(comm->nRanks);
  ready[comm->rank] = *res == ncclSuccess;
  const ncclResult_t voteRes = bootstrapAllGather(comm->bootstrap, ready.data(), sizeof(uint8_t));
  if (voteRes == ncclSuccess && std::find(ready.begin(), ready.end(), 0) == ready.end()) return true;
  if (*res == ncclSuccess) *res = voteRes != ncclSuccess ? voteRes : ncclRemoteError;
  return false;
}

ncclResult_t reserveScratch(struct ncclComm* comm, struct rcclHostHier* h, size_t bytes) {
  if (h->scratchUsed == nullptr) CUDACHECK(cudaEventCreateWithFlags(&h->scratchUsed, cudaEventDisableTiming));
  if (h->scratchBytes >= bytes) return ncclSuccess;
  if (h->scratch != nullptr) {
    // Earlier collectives, on any stream, may still read the old buffer.
    CUDACHECK(cudaEventSynchronize(h->scratchUsed));
    NCCLCHECK(ncclCudaFree(h->scratch, comm->memManager));
    h->scratch = nullptr;
    h->scratchBytes = 0;
  }
  char* buf = nullptr;
  NCCLCHECK(ncclCudaMalloc(&buf, bytes, comm->memManager));
  h->scratch = buf;
  h->scratchBytes = bytes;
  return ncclSuccess;
}

// The child comms cannot use a PreMulSum op created on the parent, and integer
// averages would round once per stage.
bool opSplitsAcrossStages(ncclRedOp_t op, ncclDataType_t datatype) {
  switch (op) {
  case ncclSum:
  case ncclProd:
  case ncclMax:
  case ncclMin:
    return true;
  case ncclAvg:
    return datatype == ncclFloat16 || datatype == ncclFloat32 || datatype == ncclFloat64 || datatype == ncclBfloat16 ||
           datatype == ncclFloat8e4m3 || datatype == ncclFloat8e5m2;
  default:
    return false;
  }
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

  // A non-blocking split can return before the child exists, so split synchronously. Children sharing the parent's
  // resources would share its abort flag, and aborting them after a failed setup would abort the parent.
  const int parentBlocking = comm->config.blocking;
  const int parentSplitShare = comm->config.splitShare;
  comm->config.blocking = 1;
  comm->config.splitShare = 0;
  ncclResult_t res = ncclCommSplit(comm, layout->hostIndex, layout->localRank, &h->intra, nullptr);
  if (res == ncclSuccess) res = ncclCommEnsureReady(h->intra);
  if (h->intra != nullptr) h->intra->hostHierState = rcclHostHierOff;
  if (!allRanksReady(comm, &res)) goto fail;
  if (layout->nHosts > 1) {
    res = ncclCommSplit(comm, layout->localRank, layout->hostIndex, &h->inter, nullptr);
    if (res == ncclSuccess) res = ncclCommEnsureReady(h->inter);
    if (h->inter != nullptr) h->inter->hostHierState = rcclHostHierOff;
    if (!allRanksReady(comm, &res)) goto fail;
  }
  res = initChildFabric(h->intra);
  if (!allRanksReady(comm, &res)) goto fail;
  res = initChildFabric(h->inter);
  if (!allRanksReady(comm, &res)) goto fail;

  comm->config.blocking = parentBlocking;
  comm->config.splitShare = parentSplitShare;
  comm->hostHierState = rcclHostHierReady;
  INFO(NCCL_INIT, "Host-hierarchical comms: rank %d host %d/%d localRank %d/%d intraDda %d interDda %d", comm->rank,
       layout->hostIndex, layout->nHosts, layout->localRank, layout->ranksPerHost,
       h->intra->ddaFabricMemHandler != nullptr, h->inter != nullptr && h->inter->ddaFabricMemHandler != nullptr);
  return ncclSuccess;

fail:
  comm->config.blocking = parentBlocking;
  comm->config.splitShare = parentSplitShare;
  // A failed split can leave the parent's bootstrap mid-exchange, so never retry.
  comm->hostHierState = rcclHostHierFailed;
  (void)releaseChild(h->inter, /*abort=*/true);
  (void)releaseChild(h->intra, /*abort=*/true);
  h->inter = nullptr;
  h->intra = nullptr;
  ATTN("Host-hierarchical comms: setup failed on rank %d (%s), using the default path", comm->rank,
       ncclGetErrorString(res));
  return res;
}

ncclResult_t rcclHostHierTeardown(struct ncclComm* comm) {
  struct rcclHostHier* h = comm->hostHier;
  comm->hostHier = nullptr;
  comm->hostHierState = rcclHostHierOff;
  if (h == nullptr) return ncclSuccess;
  // A revoke clears abortFlag again but leaves revokedFlag set, and a peer can be gone either way.
  const bool aborting =
    (comm->abortFlag != nullptr && COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire) != 0) ||
    COMPILER_ATOMIC_LOAD(&comm->revokedFlag, std::memory_order_acquire) != 0;
  ncclResult_t res = ncclSuccess;
  NCCLCHECKGOTO(releaseChild(h->inter, aborting), res, exit);
  NCCLCHECKGOTO(releaseChild(h->intra, aborting), res, exit);
  // The children's collectives read the scratch buffer.
  if (h->scratch != nullptr) NCCLCHECKGOTO(ncclCudaFree(h->scratch, comm->memManager), res, exit);
exit:
  if (h->scratchUsed != nullptr) CUDACHECKIGNORE(cudaEventDestroy(h->scratchUsed));
  free(h);
  return res;
}

bool rcclHostHierAllReduceAutoWindow(int nHosts, size_t bytes) {
  constexpr size_t MiB = size_t{1} << 20;
  switch (nHosts) {
  case 2:
    return (bytes > 1 * MiB && bytes <= 2 * MiB) || (bytes >= 8 * MiB && bytes <= 16 * MiB) ||
           (bytes >= 64 * MiB && bytes <= 128 * MiB);
  case 4:
    return bytes >= 2 * MiB && bytes <= 256 * MiB;
  case 8:
    return bytes >= 2 * MiB && bytes <= 512 * MiB;
  default:
    return false;
  }
}

bool rcclHostHierAllReduceEligible(struct ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                                   bool registeredBuffers) {
  const int64_t mode = rcclParamHostHierAllReduce();
  if (mode == 0) return false;
  if (count == 0 || !opSplitsAcrossStages(op, datatype)) return false;
  const bool forced = mode == 1;
  // Registered windows are left to the symmetric and CE paths: the child comms do not see the registrations.
  // The windows were measured on one MNNVL clique; a job spanning several stays on the default path.
  const bool oneClique = ncclDdaUseFabricPath(comm) && comm->clique.size == comm->nRanks;
  if (!forced && (registeredBuffers || !(oneClique && datatype == ncclFloat32 && op == ncclSum))) {
    return false;
  }
  const struct rcclHostLayout* layout = rcclHostHierGetLayout(comm);
  if (layout == nullptr || count % layout->ranksPerHost != 0) return false;
  if (forced) return true;
  return layout->ranksPerHost == 8 && rcclHostHierAllReduceAutoWindow(layout->nHosts, count * ncclTypeSize(datatype));
}

ncclResult_t rcclHostHierAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclRedOp_t op, struct ncclComm* comm, cudaStream_t stream) {
  if (comm->hostHierState != rcclHostHierReady) return ncclInternalError;
  struct rcclHostHier* h = comm->hostHier;
  const size_t chunk = count / h->layout.ranksPerHost;
  const size_t chunkBytes = chunk * ncclTypeSize(datatype);
  const bool doInter = h->inter != nullptr;
  // The inter-host AllReduce runs out of place: in place it is several times slower.
  const size_t stride = (chunkBytes + 255) & ~size_t{255};
  const size_t scratchBytes = doInter ? stride + chunkBytes : chunkBytes;
  if (h->scratchBytes < scratchBytes) {
    // Every rank grows on the same call, and one that cannot allocate would skip the ReduceScatter its peers wait in.
    ncclResult_t res = reserveScratch(comm, h, scratchBytes);
    if (!allRanksReady(comm, &res)) {
      // The ranks may now hold scratch of different sizes and would disagree on the next grow, so every rank drops its
      // scratch and stops using host-hierarchical AllReduce, as after a failed setup.
      if (h->scratch != nullptr) {
        if (h->scratchUsed != nullptr) (void)cudaEventSynchronize(h->scratchUsed);
        (void)ncclCudaFree(h->scratch, comm->memManager);
        h->scratch = nullptr;
        h->scratchBytes = 0;
      }
      comm->hostHierState = rcclHostHierFailed;
      ATTN("Host-hierarchical AllReduce: scratch allocation failed on rank %d (%s), using the default path", comm->rank,
           ncclGetErrorString(res));
      return res;
    }
  }
  char* rsOut = static_cast<char*>(h->scratch);
  char* agIn = doInter ? rsOut + stride : rsOut;

  // The parent AllReduce is already recorded.
  rccl::Recorder::instance().skip(true);
  ncclResult_t res = ncclSuccess;
  NCCLCHECKGOTO(ncclReduceScatter(sendbuff, rsOut, chunk, datatype, op, h->intra, stream), res, exit);
  if (doInter) NCCLCHECKGOTO(ncclAllReduce(rsOut, agIn, chunk, datatype, op, h->inter, stream), res, exit);
  NCCLCHECKGOTO(ncclAllGather(agIn, recvbuff, chunk, datatype, h->intra, stream), res, exit);
  CUDACHECKGOTO(cudaEventRecord(h->scratchUsed, stream), res, exit);
exit:
  rccl::Recorder::instance().skip(false);
  return res;
}
