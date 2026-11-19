/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric reduce-scatter.
 * Reduce-scatter analogue of the LL128 one-shot launcher in
 * dda_all_reduce_fabric_ll.cu; reuses the codepath-agnostic
 * ddaReduceScatterFabricLL128 kernel from reduce_scatter_dda_fabric_ll128.h.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"

#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "param.h"
#include "rccl_arch_thresholds.h"
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
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;

constexpr unsigned kDdaLL128RsThreads = 512;
constexpr unsigned kDdaLL128RsNarrowThreads = 128;
constexpr unsigned kDdaLL128RsWideShardThreads = 256;

// The arch table's launch-shape entries when they apply to this comm (rankScaledMinRanks ranks and up), else null.
static inline const rcclArchThresholds* ddaLL128RsShapeTable(const ncclComm* comm) {
  const rcclArchThresholds* t = comm->archThresholds;
  if (t == nullptr || t->rankScaledMinRanks <= 0 || comm->nRanks < t->rankScaledMinRanks) {
    return nullptr;
  }
  return t;
}

// Grid budget for a shard Ring would run: where Ring Simple runs on capped channels (one node, the table's band above
// the Ring LL128 bound for this type size), at most a quarter more blocks than those channels.
static inline int ddaLL128RsMaxBlocks(const ncclComm* comm, const rcclArchThresholds* t, size_t shardBytes,
                                      int typeSize) {
  int nBlocksMax = std::max(comm->ddaFabricMaxBlocks, 1);
  if (t == nullptr || comm->nNodes != 1 || t->rsRingSimpleCappedChannels <= 0) {
    return nBlocksMax;
  }
  const size_t ringLL128Max =
    typeSize == 1 && t->rsRingLL128MaxPerRank1B != 0 ? t->rsRingLL128MaxPerRank1B : t->rsRingLL128MaxPerRank;
  if (shardBytes > ringLL128Max && shardBytes <= t->rsRingSimpleCappedMaxPerRank) {
    nBlocksMax = std::min(nBlocksMax, t->rsRingSimpleCappedChannels + t->rsRingSimpleCappedChannels / 4);
  }
  return nBlocksMax;
}

// Whether LL ran the shards LL128 now takes narrow: LL on at no less than the table's cap, its bank in the scratch.
// Otherwise those shards ran LL128 at full width before.
static inline bool ddaLL128RsTakesLLShards(const ncclComm* comm, const rcclArchThresholds* t) {
  const size_t llMax = t->ddaLLMax[ncclFuncReduceScatter];
  if (llMax == 0 || rcclDdaLLThreshold(comm, ncclFuncReduceScatter) < llMax) {
    return false;
  }
  const size_t bankBytes = dda::common::ddaLLRsBankBytes(comm->nRanks);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);
  return bankBytes <= bankSize && bankSize + bankBytes <= comm->ddaScratchBytes;
}

// One warp carries one slice, so the grid is sized in slices rather than in
// lines: blocks = ceil(slices / warpsPerBlock), capped by the grid budget. A
// shard with more slices than the capped grid has warps just means each warp
// takes several passes through the slice loop. From rankScaledMinRanks ranks the
// arch table also shapes the launch: shards up to rsDdaLL128NarrowMaxPerRank that
// LL ran before start at narrow blocks and widen once the budget is full, so they
// run on no more blocks than LL did; larger shards keep the blocks 512 threads
// take but run 256, each warp taking more slices.
static inline std::pair<dim3, dim3> ddaReduceScatterFabricLL128Geom(ncclComm* comm, size_t shardBytes, int typeSize) {
  const size_t slices = ddaLL128Slices(shardBytes);
  const rcclArchThresholds* t = ddaLL128RsShapeTable(comm);
  const int nBlocksMax = ddaLL128RsMaxBlocks(comm, t, shardBytes, typeSize);
  const size_t narrowMax = t != nullptr ? t->rsDdaLL128NarrowMaxPerRank : 0;
  // LL never ran shards past kDdaLLRsMaxBytes, whatever the table says.
  const bool narrow =
    shardBytes <= std::min(narrowMax, dda::common::kDdaLLRsMaxBytes) && ddaLL128RsTakesLLShards(comm, t);
  unsigned gridThreads = narrow ? kDdaLL128RsNarrowThreads : kDdaLL128RsThreads;
  while (gridThreads < kDdaLL128RsThreads &&
         slices > (size_t)nBlocksMax * (gridThreads / (unsigned)kDdaLL128Warp)) {
    gridThreads *= 2;
  }
  const size_t warps = gridThreads / (unsigned)kDdaLL128Warp;
  unsigned blocks = (unsigned)std::min<size_t>((slices + warps - 1) / warps, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  const unsigned threads = narrowMax != 0 && shardBytes > narrowMax ? kDdaLL128RsWideShardThreads : gridThreads;
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

  auto gridBlock = ddaReduceScatterFabricLL128Geom(comm, shardBytes, (int)sizeof(T));
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
       "(warp-per-slice, bankSize=%zu)",
       nRanks, shardBytes, slices, grid.x, block.x, bankSize);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // NRANKS_CT 4/8: unrolled peer loops; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL128<T, 4>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev, epochLen, slices,
                          bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL128<T, 8>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev, epochLen, slices,
                          bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaReduceScatterFabricLL128<T, 0>), grid, block, 0, stream,
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
