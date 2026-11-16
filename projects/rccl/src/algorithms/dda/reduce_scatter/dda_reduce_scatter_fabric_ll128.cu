/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric reduce-scatter.
 * Reduce-scatter analogue of the LL128 one-shot launcher in
 * dda_all_reduce_fabric_ll.cu. Launches the peer-parallel
 * ddaReduceScatterFabricLL128PeerPar kernel from
 * reduce_scatter_dda_fabric_ll128_peerpar.h.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"

#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128_peerpar.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "param.h"
#include "rccl_common.h"

#include <cuda_runtime.h>
#include <hip/hip_ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

using dda::common::ddaBankSize;
using dda::common::ddaLL128RsSlotWords;
using dda::common::ddaLL128Slices;
using dda::common::kDdaLL128RsPeerMaxThreads;
using dda::common::kDdaLL128RsPeerMaxWarps;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;

// A block starts as one warp group, a warp per rank up to kDdaLL128RsPeerMaxWarps, so slices spread over the most CUs.
// It doubles its groups up to kDdaLL128RsPeerMaxThreads only while one group per block would pass the block cap.
static inline std::pair<dim3, dim3> ddaReduceScatterFabricLL128Geom(ncclComm* comm, size_t shardBytes) {
  const size_t slices = ddaLL128Slices(shardBytes);
  const unsigned groupThreads =
    (unsigned)dda::common::ddaLL128RsPeerWarps(comm->nRanks, kDdaLL128RsPeerMaxWarps) * (unsigned)kDdaLL128Warp;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned threads = groupThreads;
  while (threads * 2 <= (unsigned)kDdaLL128RsPeerMaxThreads) {
    const size_t groups = threads / groupThreads;
    if ((slices + groups - 1) / groups <= (size_t)nBlocksMax) {
      break;
    }
    threads *= 2;
  }
  const size_t groups = threads / groupThreads;
  unsigned blocks = (unsigned)std::min<size_t>((slices + groups - 1) / groups, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  // flatBlockId (blockIdx.x) must stay within the device epoch array.
  if ((int)blocks > comm->ddaLLEpochLen) {
    blocks = (unsigned)comm->ddaLLEpochLen;
    if (blocks == 0) {
      blocks = 1;
    }
  }
  return std::make_pair(dim3(blocks), dim3(threads));
}

template <typename T>
static ncclResult_t ncclReduceScatterDdaFabricLL128Typed(const void* sendbuff, void* recvbuff, size_t recvcount,
                                                         ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t shardBytes = recvcount * sizeof(T);
  const size_t slices = ddaLL128Slices(shardBytes);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  auto gridBlock = ddaReduceScatterFabricLL128Geom(comm, shardBytes);
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter every other LL/LL128 tier uses: they share a scratch
  // layout, so one monotonic flag is what keeps either from accepting a line
  // the other left.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL,
       "DDA fabric ReduceScatter LL128: nRanks=%d shardBytes=%zu slices=%zu grid=%u block=%u "
       "(peer-parallel, bankSize=%zu)",
       nRanks, shardBytes, slices, grid.x, block.x, bankSize);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // NRANKS_CT 4/8 unroll the fold rounds. 0 is the runtime fallback.
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL128PeerPar<T, 4>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev, epochLen, slices,
                          bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL128PeerPar<T, 8>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev, epochLen, slices,
                          bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL128PeerPar<T, 0>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev, epochLen, slices,
                          bankSize, params);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclReduceScatterDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                             ncclDataType_t datatype, ncclRedOp_t op) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->ddaLLEpochDev == nullptr || comm->ddaLLEpochLen < 1) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (recvcount == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  const size_t shardBytes = recvcount * ncclTypeSize(datatype);

  // The LL128 line format packs 16B-aligned data with no chunk straddling a
  // line, so a partial 16B chunk has nowhere to go, and the 16B wire accesses
  // need both user buffers aligned to match.
  if (shardBytes % 16 != 0) {
    return false;
  }
  if (!dda::common::ddaUserBuffers16ByteAligned(sendbuff, recvbuff)) {
    return false;
  }

  // Wrap compares the full message (nRanks shards) against the arch table; keep
  // that same total here so direct Eligible() callers agree with the selector.
  if (shardBytes * (size_t)comm->nRanks > rcclDdaLL128Threshold(comm, ncclFuncReduceScatter)) {
    return false;
  }

  // The shard has to fit the slices one rank's slot holds.
  const size_t slotWords = ddaLL128RsSlotWords(ddaBankSize(comm->ddaScratchBytes), comm->nRanks);
  if (ddaLL128Slices(shardBytes) * (size_t)kDdaLL128WireWordsPerSlice > slotWords) {
    return false;
  }

  return true;
}

ncclResult_t ncclReduceScatterDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t recvcount,
                                             ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm,
                                             cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclReduceScatterDdaFabricLL128Typed<float>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat16:
    return ncclReduceScatterDdaFabricLL128Typed<half>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclBfloat16:
    return ncclReduceScatterDdaFabricLL128Typed<bf16>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat8e4m3:
    return ncclReduceScatterDdaFabricLL128Typed<dda::common::fp8e4m3>(sendbuff, recvbuff, recvcount, comm, stream);
  case ncclFloat8e5m2:
    return ncclReduceScatterDdaFabricLL128Typed<dda::common::fp8e5m2>(sendbuff, recvbuff, recvcount, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
