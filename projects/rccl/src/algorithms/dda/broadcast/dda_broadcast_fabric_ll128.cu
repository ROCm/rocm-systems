/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA broadcast.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/broadcast/dda_broadcast.h"

#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll128.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>

// Tuned block size, and the fallback when the env override is out of range.
constexpr unsigned kDdaLL128BcDefaultThreads = 512;

RCCL_PARAM(DdaLL128BcThreads, "DDA_LL128_BC_THREADS", kDdaLL128BcDefaultThreads);

namespace {

using dda::common::ddaBankSize;
using dda::common::ddaLL128BcSlotSlices;
using dda::common::kDdaLL128DataBytesPerSliceL;
using dda::common::kDdaLL128WireWordsPerSliceW;

// Host mirror of kDdaLL128ArchLineElems / kDdaLL128ArchWarp.
struct DdaLL128BcArch {
  int lineBytes;
  size_t warp;
  size_t sliceWords;
  size_t sliceBytes; // payload
};

DdaLL128BcArch ddaLL128BcArch(const ncclComm* comm) {
  if (IsArchMatch(comm->archName, "gfx1250")) {
    return {128, 32, kDdaLL128WireWordsPerSliceW<32>, kDdaLL128DataBytesPerSliceL<16, 32>};
  }
  return {64, 64, kDdaLL128WireWordsPerSliceW<64>, kDdaLL128DataBytesPerSliceL<8, 64>};
}

// RCCL_DDA_LL128_BC_THREADS when it is a multiple of the arch warp and at most kDdaBcMaxThreads.
unsigned ddaLL128BcThreads(const ncclComm* comm) {
  const int64_t userInput = rcclParamDdaLL128BcThreads();
  const int64_t warp = (int64_t)ddaLL128BcArch(comm).warp;
  if (userInput >= warp && userInput <= kDdaBcMaxThreads && userInput % warp == 0) {
    return (unsigned)userInput;
  }
  return kDdaLL128BcDefaultThreads;
}

size_t ddaLL128BcSlices(const ncclComm* comm, size_t bytes) {
  const size_t per = ddaLL128BcArch(comm).sliceBytes;
  return (bytes + per - 1) / per;
}

// Root: one column per peer (at least one block each, as AllGather LL128), the
// rest of the budget split across slice groups. Non-root: groups within the budget.
dim3 ddaLL128BcGrid(const ncclComm* comm, bool isRoot, size_t bytes) {
  const size_t warps = ddaLL128BcThreads(comm) / ddaLL128BcArch(comm).warp;
  const size_t groups = (ddaLL128BcSlices(comm, bytes) + warps - 1) / warps;
  const size_t budget = ncclBroadcastDdaBlockBudget(comm);
  if (isRoot) {
    const size_t peers = (size_t)(comm->nRanks - 1);
    const size_t perPeer = budget / peers < 1 ? 1 : budget / peers;
    return dim3((unsigned)peers, (unsigned)(groups < perPeer ? groups : perPeer));
  }
  return dim3((unsigned)(groups < budget ? groups : budget));
}

} // namespace

bool ncclBroadcastDdaFabricLL128Eligible(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  if (comm == nullptr || comm->ddaFabricMemHandler == nullptr) return false;
  if (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) return false;
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1) return false;
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) return false;
  // gfx1250 only until a cross-GPU gfx9 data test exists.
  if (!IsArchMatch(comm->archName, "gfx1250")) return false;
  if (count == 0) return false;
  const size_t bytes = count * ncclTypeSize(datatype);
  if (bytes % 16 != 0) return false;
  if (bytes > rcclDdaLL128Threshold(comm, ncclFuncBroadcast)) return false;
  if (ddaLL128BcSlices(comm, bytes) >
      ddaLL128BcSlotSlices(ddaBankSize(comm->ddaScratchBytes), comm->nRanks, ddaLL128BcArch(comm).sliceWords)) {
    return false;
  }
  // Both the root and the non-root grid must fit the epoch cells.
  const dim3 r = ddaLL128BcGrid(comm, /*isRoot=*/true, bytes);
  const dim3 n = ddaLL128BcGrid(comm, /*isRoot=*/false, bytes);
  const long most = (long)r.x * r.y > (long)n.x ? (long)r.x * r.y : (long)n.x;
  return most <= (long)comm->ddaLLEpochLen;
}

uint32_t ncclBroadcastDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const dim3 g = ddaLL128BcGrid(comm, /*isRoot=*/true, count * ncclTypeSize(datatype));
  return g.x * g.y;
}

ncclResult_t ncclBroadcastDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         int root, ncclComm* comm, cudaStream_t stream) {
  const size_t bytes = count * ncclTypeSize(datatype);
  const bool isRoot = comm->rank == root;
  const dim3 grid = ddaLL128BcGrid(comm, isRoot, bytes);
  const uintptr_t addrs =
    reinterpret_cast<uintptr_t>(recvbuff) | (isRoot ? reinterpret_cast<uintptr_t>(sendbuff) : 0);
  const bool aligned = addrs % 16 == 0;

  char* const* peers = reinterpret_cast<char* const*>(comm->ddaPeerPtrsDev);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);
  const dda::common::DdaKernelParams params{comm->abortFlagDev};

  const DdaLL128BcArch arch = ddaLL128BcArch(comm);
  const unsigned threads = ddaLL128BcThreads(comm);
  INFO(NCCL_COLL, "DDA Broadcast LL128: nRanks=%d root=%d bytes=%zu grid=%ux%u block=%u line=%dB warp=%zu aligned=%d",
       comm->nRanks, root, bytes, grid.x, grid.y, threads, arch.lineBytes, arch.warp, (int)aligned);

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (aligned) {
    hipExtLaunchKernelGGL((dda::common::ddaBroadcastFabricLL128<true>), grid, dim3(threads), 0,
                          stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<char*>(recvbuff),
                          static_cast<const char*>(sendbuff), bytes, comm->rank, comm->nRanks, root,
                          comm->ddaLLEpochDev, comm->ddaLLEpochLen, bankSize, params);
  } else {
    hipExtLaunchKernelGGL((dda::common::ddaBroadcastFabricLL128<false>), grid, dim3(threads), 0,
                          stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<char*>(recvbuff),
                          static_cast<const char*>(sendbuff), bytes, comm->rank, comm->nRanks, root,
                          comm->ddaLLEpochDev, comm->ddaLLEpochLen, bankSize, params);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
