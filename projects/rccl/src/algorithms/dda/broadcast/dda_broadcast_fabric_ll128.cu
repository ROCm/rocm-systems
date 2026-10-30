/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA broadcast.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/broadcast/dda_broadcast.h"

#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll128.h"
#include "algorithms/dda/broadcast/broadcast_dda_fabric_ll128_twoshot.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "graph/topo.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <cstddef>
#include <cstdint>

// Block size; 0 (or out of range) picks it by size, see ddaLL128BcThreads.
RCCL_PARAM(DdaLL128BcThreads, "DDA_LL128_BC_THREADS", 0);

// One-shot up to this many bytes, scatter-allgather above it (up to the LL128 tier cap).
RCCL_PARAM(DdaLL128BcOneShotThreshold, "DDA_LL128_BC_ONESHOT_THRESHOLD", (int64_t)256 * 1024);

// Two-shot slices per warp task: 1, 2 or 4; 0 picks it by size, see ddaLL128BcUnroll.
RCCL_PARAM(DdaLL128BcUnroll, "DDA_LL128_BC_UNROLL", 0);

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

bool ddaLL128BcTwoShot(size_t bytes) {
  return (int64_t)bytes > rcclParamDdaLL128BcOneShotThreshold();
}

// Two-shot chunk size in bytes.
size_t ddaLL128BcChunkBytes(const ncclComm* comm, size_t bytes) {
  return bytes / (size_t)(comm->nRanks - 1);
}

// RCCL_DDA_LL128_BC_THREADS when it is a multiple of the arch warp and at most kDdaBcMaxThreads.
// Otherwise tuned on gfx1250: smaller blocks spread small chunks over more CUs.
unsigned ddaLL128BcThreads(const ncclComm* comm, size_t bytes) {
  const int64_t userInput = rcclParamDdaLL128BcThreads();
  const int64_t warp = (int64_t)ddaLL128BcArch(comm).warp;
  if (userInput >= warp && userInput <= kDdaBcMaxThreads && userInput % warp == 0) {
    return (unsigned)userInput;
  }
  if (!ddaLL128BcTwoShot(bytes)) return 256;
  const size_t chunk = ddaLL128BcChunkBytes(comm, bytes);
  return chunk <= ((size_t)512 << 10) ? 128 : chunk <= ((size_t)4 << 20) ? 256 : 512;
}

// Slices per two-shot warp task. Tuned on gfx1250: from 8 MiB chunks 4 up to 4 ranks, 2 above; else 1.
int ddaLL128BcUnroll(const ncclComm* comm, size_t bytes) {
  const int64_t u = rcclParamDdaLL128BcUnroll();
  if (u == 1 || u == 2 || u == 4) return (int)u;
  if (ddaLL128BcChunkBytes(comm, bytes) < ((size_t)8 << 20)) return 1;
  return comm->nRanks <= 4 ? 4 : 2;
}

size_t ddaLL128BcSlices(const ncclComm* comm, size_t bytes) {
  const size_t per = ddaLL128BcArch(comm).sliceBytes;
  return (bytes + per - 1) / per;
}

// Root: one column per peer (at least one block each, as AllGather LL128), the
// rest of the budget split across slice groups. Non-root: groups within the budget.
// Two-shot: a root column covers one chunk; a non-root covers nRanks-1 chunks.
dim3 ddaLL128BcGrid(const ncclComm* comm, bool isRoot, size_t bytes) {
  const size_t warps = ddaLL128BcThreads(comm, bytes) / ddaLL128BcArch(comm).warp;
  const size_t peers = (size_t)(comm->nRanks - 1);
  const size_t slices = ddaLL128BcSlices(comm, bytes);
  const bool twoShot = ddaLL128BcTwoShot(bytes);
  // Unaligned ranks launch kU=1 on this grid (eligibility is rank-uniform); grid-stride loops cover the rest.
  const size_t ng = dda::common::ddaLL128BcChunkTasks(dda::common::ddaLL128BcChunkSlices(slices, comm->nRanks),
                                                      ddaLL128BcUnroll(comm, bytes));
  const size_t rootTasks = twoShot ? ng : slices;
  const size_t budget = ncclBroadcastDdaBlockBudget(comm);
  if (isRoot) {
    const size_t groups = (rootTasks + warps - 1) / warps;
    const size_t perPeer = budget / peers < 1 ? 1 : budget / peers;
    return dim3((unsigned)peers, (unsigned)(groups < perPeer ? groups : perPeer));
  }
  const size_t groups = ((twoShot ? peers * ng : slices) + warps - 1) / warps;
  return dim3((unsigned)(groups < budget ? groups : budget));
}

} // namespace

bool ncclBroadcastDdaFabricLL128Eligible(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  if (comm == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) return false;
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1) return false;
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) return false;
  // gfx9 relies on the same 64B store atomicity as native LL128.
  if (!IsArchMatch(comm->archName, "gfx1250") && (comm->topo == nullptr || !comm->topo->ll128Enabled)) return false;
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
  const unsigned threads = ddaLL128BcThreads(comm, bytes);
  const bool twoShot = ddaLL128BcTwoShot(bytes);
  const int u = twoShot && aligned ? ddaLL128BcUnroll(comm, bytes) : 1;
  INFO(NCCL_COLL,
       "DDA Broadcast LL128 %s: nRanks=%d root=%d bytes=%zu grid=%ux%u block=%u line=%dB warp=%zu aligned=%d "
       "unroll=%d",
       twoShot ? "two-shot" : "one-shot", comm->nRanks, root, bytes, grid.x, grid.y, threads, arch.lineBytes,
       arch.warp, (int)aligned, u);

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  auto launch = [&](auto kernel) {
    hipExtLaunchKernelGGL(kernel, grid, dim3(threads), 0, stream, /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers,
                          static_cast<char*>(recvbuff), static_cast<const char*>(sendbuff), bytes, comm->rank,
                          comm->nRanks, root, comm->ddaLLEpochDev, comm->ddaLLEpochLen, bankSize, params);
  };
  if (!twoShot) {
    if (aligned) {
      launch(dda::common::ddaBroadcastFabricLL128<true>);
    } else {
      launch(dda::common::ddaBroadcastFabricLL128<false>);
    }
  } else if (!aligned) {
    launch(dda::common::ddaBroadcastFabricLL128TwoShot<false, 1>);
  } else if (u == 4) {
    launch(dda::common::ddaBroadcastFabricLL128TwoShot<true, 4>);
  } else if (u == 2) {
    launch(dda::common::ddaBroadcastFabricLL128TwoShot<true, 2>);
  } else {
    launch(dda::common::ddaBroadcastFabricLL128TwoShot<true, 1>);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
