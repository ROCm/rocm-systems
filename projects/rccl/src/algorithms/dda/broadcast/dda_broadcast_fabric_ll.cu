/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA broadcast.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/broadcast/dda_broadcast.h"

#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>

// Lowers the per-rank block budget (CU-derived); 0 keeps it.
RCCL_PARAM(DdaBcMaxBlocks, "DDA_BC_MAX_BLOCKS", 0);

// Tuned block size, and the fallback when the env override is out of range.
constexpr unsigned kDdaLLBcDefaultThreads = 256;

RCCL_PARAM(DdaLLBcThreads, "DDA_LL_BC_THREADS", kDdaLLBcDefaultThreads);

namespace {

using dda::common::ddaBankSize;
using dda::common::ddaLLBcSlotPkts;

constexpr size_t kDdaLLBcMaxChunks = 8;

// RCCL_DDA_LL_BC_THREADS when it is a multiple of the wave size and at most kDdaBcMaxThreads.
unsigned ddaLLBcThreads(const ncclComm* comm) {
  const int64_t userInput = rcclParamDdaLLBcThreads();
  const int64_t warp = comm->WarpSize > 0 ? comm->WarpSize : 64;
  if (userInput >= warp && userInput <= kDdaBcMaxThreads && userInput % warp == 0) {
    return (unsigned)userInput;
  }
  return kDdaLLBcDefaultThreads;
}

// One packet per thread per chunk, at most kDdaLLBcMaxChunks chunks; above that each thread loops.
size_t ddaLLBcChunks(const ncclComm* comm, size_t bytes) {
  const size_t pkts = ddaLLBcThreads(comm);
  const size_t c = ((bytes >> 3) + pkts - 1) / pkts;
  return c < 1 ? 1 : (c > kDdaLLBcMaxChunks ? kDdaLLBcMaxChunks : c);
}

// Root: one column per peer (at least one block each, as AllGather LL128), the
// rest of the budget split across chunks. Non-root: chunks within the budget.
dim3 ddaLLBcGrid(const ncclComm* comm, bool isRoot, size_t bytes) {
  const size_t budget = ncclBroadcastDdaBlockBudget(comm);
  const size_t chunks = ddaLLBcChunks(comm, bytes);
  if (isRoot) {
    const size_t peers = (size_t)(comm->nRanks - 1);
    const size_t perPeer = budget / peers < 1 ? 1 : budget / peers;
    return dim3((unsigned)peers, (unsigned)(chunks < perPeer ? chunks : perPeer));
  }
  return dim3((unsigned)(chunks < budget ? chunks : budget));
}

} // namespace

size_t ncclBroadcastDdaBlockBudget(const ncclComm* comm) {
  int64_t cap = comm->ddaFabricMaxBlocks > 0 ? comm->ddaFabricMaxBlocks : comm->cuCount;
  const int64_t env = rcclParamDdaBcMaxBlocks();
  if (env > 0 && env < cap) cap = env;
  return cap < 1 ? 1 : (size_t)cap;
}

bool ncclBroadcastDdaFabricLLEligible(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  if (comm == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) return false;
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) return false;
  if (count == 0) return false;
  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes % 16 != 0) return false;
  if (bytes > rcclDdaLLThreshold(comm, ncclFuncBroadcast)) return false;
  if ((bytes >> 3) > ddaLLBcSlotPkts(ddaBankSize(comm->ddaScratchBytes), comm->nRanks)) return false;
  // Both the root and the non-root grid must fit the epoch cells.
  const dim3 r = ddaLLBcGrid(comm, /*isRoot=*/true, bytes);
  const dim3 n = ddaLLBcGrid(comm, /*isRoot=*/false, bytes);
  const long most = (long)r.x * r.y > (long)n.x ? (long)r.x * r.y : (long)n.x;
  return comm->ddaLLEpochDev != nullptr && most <= (long)comm->ddaLLEpochLen;
}

uint32_t ncclBroadcastDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const dim3 g = ddaLLBcGrid(comm, /*isRoot=*/true, count * ncclTypeSize(datatype));
  return g.x * g.y;
}

ncclResult_t ncclBroadcastDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      int root, ncclComm* comm, cudaStream_t stream) {
  const size_t bytes = count * ncclTypeSize(datatype);
  const bool isRoot = comm->rank == root;
  const dim3 grid = ddaLLBcGrid(comm, isRoot, bytes);
  const unsigned threads = ddaLLBcThreads(comm);
  // Only this rank's own buffers; the wire format does not depend on it. The
  // aligned path moves user data in 4-byte words.
  const uintptr_t addrs = reinterpret_cast<uintptr_t>(recvbuff) | (isRoot ? reinterpret_cast<uintptr_t>(sendbuff) : 0);
  const bool aligned = addrs % sizeof(uint32_t) == 0;

  char* const* peers = reinterpret_cast<char* const*>(comm->ddaPeerPtrsDev);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);
  const dda::common::DdaKernelParams params{comm->abortFlagDev};

  INFO(NCCL_COLL, "DDA Broadcast LL: nRanks=%d root=%d bytes=%zu grid=%ux%u block=%u aligned=%d", comm->nRanks, root,
       bytes, grid.x, grid.y, threads, (int)aligned);

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (aligned) {
    hipExtLaunchKernelGGL((dda::common::ddaBroadcastFabricLL<true>), grid, dim3(threads), 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<char*>(recvbuff),
                          static_cast<const char*>(sendbuff), bytes, comm->rank, comm->nRanks, root,
                          comm->ddaLLEpochDev, comm->ddaLLEpochLen, bankSize, params);
  } else {
    hipExtLaunchKernelGGL((dda::common::ddaBroadcastFabricLL<false>), grid, dim3(threads), 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<char*>(recvbuff),
                          static_cast<const char*>(sendbuff), bytes, comm->rank, comm->nRanks, root,
                          comm->ddaLLEpochDev, comm->ddaLLEpochLen, bankSize, params);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
