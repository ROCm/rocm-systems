/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "hier_host.h"

#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/fabric/fabric_init.h"
#include "alloc.h"
#include "archinfo.h"
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
// -1 runs host-hierarchical AllReduce in the fused DDA kernel inside rcclHostHierAllReduceFusedWindow, 1 wherever the
// kernel applies, 0 never.
RCCL_PARAM(HostHierAllReduceFused, "HOST_HIER_ALLREDUCE_FUSED", -1);

namespace {

// Split children skip DDA fabric init because they have a parent. Without it the
// intra-host and inter-host collectives miss the DDA fast path.
ncclResult_t initChildFabric(struct ncclComm* child) {
  if (child == nullptr || child->nRanks < 2) return ncclSuccess;
  if (child->ddaFabricMemHandler != nullptr) return ncclSuccess;
  if (!ncclDdaUseFabricPath(child)) return ncclSuccess;
  return ncclDdaFabricCommInit(child);
}

// Never frees a buffer while the comm lives: freeing one that peer GPUs of the same process have just accessed
// through the child comms intermittently fails in HIP, so a buffer that is outgrown is kept until teardown. Growing
// at least twofold keeps the kept buffers under the size of the current one.
ncclResult_t reserveScratch(struct ncclComm* comm, struct rcclHostHier* h, size_t bytes) {
  if (h->scratchBytes >= bytes) return ncclSuccess;
  if (h->nRetired == kHostHierMaxRetired) return ncclInternalError;
  const size_t grown = std::max(bytes, 2 * h->scratchBytes);
  char* buf = nullptr;
  NCCLCHECK(ncclCudaMalloc(&buf, grown, comm->memManager));
  if (h->scratch != nullptr) h->retired[h->nRetired++] = h->scratch;
  h->scratch = buf;
  h->scratchBytes = grown;
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
    return datatype == ncclFloat16 || datatype == ncclFloat32 || datatype == ncclFloat64 ||
           datatype == ncclBfloat16 || datatype == ncclFloat8e4m3 || datatype == ncclFloat8e5m2;
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

namespace {

// Same host numbering as rcclHostLayoutGet.
void fillHostRanks(const struct ncclComm* comm, const struct rcclHostLayout& layout, int* hostRanks) {
  std::vector<uint64_t> hosts;
  std::vector<int> counts;
  for (int r = 0; r < comm->nRanks; r++) {
    const uint64_t hash = comm->peerInfo[r].hostHash;
    size_t idx = 0;
    while (idx < hosts.size() && hosts[idx] != hash) idx++;
    if (idx == hosts.size()) {
      hosts.push_back(hash);
      counts.push_back(0);
    }
    hostRanks[idx * layout.ranksPerHost + counts[idx]++] = r;
  }
}

} // namespace

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
  if (ncclCalloc(&h->hostRanks, comm->nRanks) != ncclSuccess) {
    free(h);
    comm->hostHierState = rcclHostHierOff;
    return nullptr;
  }
  fillHostRanks(comm, layout, h->hostRanks);
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
  for (int i = 0; i < h->nRetired; i++) NCCLCHECKGOTO(ncclCudaFree(h->retired[i], comm->memManager), res, exit);
  if (h->scratch != nullptr) NCCLCHECKGOTO(ncclCudaFree(h->scratch, comm->memManager), res, exit);
exit:
  free(h->hostRanks);
  free(h);
  return res;
}

bool rcclHostHierAllReduceAutoWindow(int nHosts, size_t bytes) {
  constexpr size_t MiB = size_t{1} << 20;
  switch (nHosts) {
  case 2:
    return bytes >= 2 * MiB && bytes <= 512 * MiB;
  case 4:
    return bytes >= 1 * MiB && bytes <= 1024 * MiB;
  case 8:
    return bytes >= 1 * MiB && bytes <= 2048 * MiB;
  default:
    return false;
  }
}

bool rcclHostHierAllReduceFusedWindow(int nHosts, size_t bytes) {
  constexpr size_t MiB = size_t{1} << 20;
  return nHosts <= 2 && bytes >= 4 * MiB && bytes <= 128 * MiB;
}

bool rcclHostHierAllReduceEligible(struct ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op) {
  const int64_t mode = rcclParamHostHierAllReduce();
  if (mode == 0) return false;
  if (count == 0 || !opSplitsAcrossStages(op, datatype)) return false;
  const bool forced = mode == 1;
  const bool halfType = datatype == ncclFloat16 || datatype == ncclBfloat16;
  if (!forced && !(IsArchMatch(comm->archName, "gfx1250") && (datatype == ncclFloat32 || halfType) && op == ncclSum)) {
    return false;
  }
  const struct rcclHostLayout* layout = rcclHostHierGetLayout(comm);
  if (layout == nullptr || count % layout->ranksPerHost != 0) return false;
  if (forced) return true;
  // FP16 and BF16 were measured on two hosts only.
  if (halfType && layout->nHosts != 2) return false;
  return layout->ranksPerHost == 8 && rcclHostHierAllReduceAutoWindow(layout->nHosts, count * ncclTypeSize(datatype));
}

ncclResult_t rcclHostHierAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                   ncclRedOp_t op, struct ncclComm* comm, cudaStream_t stream) {
  if (comm->hostHierState != rcclHostHierReady) return ncclInternalError;
  struct rcclHostHier* h = comm->hostHier;
  const struct rcclHostLayout& layout = h->layout;
  const int64_t fused = rcclParamHostHierAllReduceFused();
  if ((fused == 1 || (fused == -1 && rcclHostHierAllReduceFusedWindow(layout.nHosts, count * ncclTypeSize(datatype)))) &&
      ncclAllReduceDdaFabricHierEligible(comm, sendbuff, recvbuff, count, datatype, op, layout.nHosts,
                                         layout.ranksPerHost)) {
    std::vector<int> inter(layout.nHosts);
    for (int host = 0; host < layout.nHosts; host++) {
      inter[host] = h->hostRanks[host * layout.ranksPerHost + layout.localRank];
    }
    return ncclAllReduceDdaFabricHier(sendbuff, recvbuff, count, datatype, comm, stream, layout.nHosts,
                                      layout.ranksPerHost, layout.hostIndex, layout.localRank,
                                      h->hostRanks + layout.hostIndex * layout.ranksPerHost, inter.data());
  }
  const size_t chunk = count / h->layout.ranksPerHost;
  const size_t chunkBytes = chunk * ncclTypeSize(datatype);
  const bool doInter = h->inter != nullptr;
  // The inter-host AllReduce runs out of place: in place it is several times slower.
  const size_t stride = (chunkBytes + 255) & ~size_t{255};
  NCCLCHECK(reserveScratch(comm, h, doInter ? stride + chunkBytes : chunkBytes));
  char* rsOut = static_cast<char*>(h->scratch);
  char* agIn = doInter ? rsOut + stride : rsOut;

  // The parent AllReduce is already recorded.
  rccl::Recorder::instance().skip(true);
  ncclResult_t res = ncclSuccess;
  NCCLCHECKGOTO(ncclReduceScatter(sendbuff, rsOut, chunk, datatype, op, h->intra, stream), res, exit);
  if (doInter) NCCLCHECKGOTO(ncclAllReduce(rsOut, agIn, chunk, datatype, op, h->inter, stream), res, exit);
  NCCLCHECKGOTO(ncclAllGather(agIn, recvbuff, chunk, datatype, h->intra, stream), res, exit);
exit:
  rccl::Recorder::instance().skip(false);
  return res;
}
