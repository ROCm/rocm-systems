/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL-protocol DDA fabric all-reduce.
 * All-reduce analogue of dda_all_gather_fabric_ll.cu; reuses the codepath-
 * agnostic ddaAllReduceFlatLL kernel from all_reduce_dda_ll.h.
 *
 * Carries both LL all-reduce tiers: the one-shot kernel above and the two-shot
 * kernel from all_reduce_dda_ll_twoshot.h, which transports one shard per rank
 * instead of the whole message, plus the LL128 one-shot tier from
 * all_reduce_dda_ll128.h. They share a scratch layout and epoch counter, so
 * keeping the launchers in one translation unit also keeps the invariants that
 * tie them (the static_assert below) next to the code it constrains. LL128
 * LL128 one-shot and two-shot are gated by RCCL_DDA_LL128 (not RCCL_DDA_LL); the two protocols are independently controlled.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/all_reduce/all_reduce_dda_ll.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll_twoshot.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128.h"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_twoshot.h"
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

RCCL_PARAM_DECLARE(DdaLL);
RCCL_PARAM_DECLARE(DdaLLOneShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128OneShotThreshold);

namespace {

using dda::common::ddaLLArSlotPkts;
using dda::common::ddaLLArTwoShotSlotPkts;

using dda::common::ddaBankSize;
using dda::common::ddaLL128ArSlotWords;
using dda::common::ddaLL128ArTwoShotSlotWords;
using dda::common::ddaLL128Slices;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;

// Single source of the LL and LL128 launch geometry: a 1-D grid over `units` of work, `unitThreads` threads each,
// in blocks of `threads`, capped low (LL serves small messages where latency, not occupancy, dominates). `cap` lowers
// the blocks and the threads (whole 64-thread granules; every kernel strides over the grid); {0, 0} keeps the tier's.
static inline std::pair<dim3, dim3> ddaArFabricLLGrid(const ncclComm* comm, size_t units, unsigned threads,
                                                      unsigned unitThreads, rcclDdaLaunchCap cap) {
  if (cap.threads > 0 && threads > cap.threads) {
    threads = std::max(cap.threads / 64 * 64, 64u);
  }
  size_t nBlocksMax = (size_t)std::max(comm->ddaFabricMaxBlocks, 1);
  if (cap.blocks > 0 && cap.blocks < nBlocksMax) {
    nBlocksMax = cap.blocks;
  }
  const size_t unitsPerBlock = threads / unitThreads;
  const size_t blocks = std::min((units + unitsPerBlock - 1) / unitsPerBlock, nBlocksMax);
  return std::make_pair(dim3((unsigned)std::max<size_t>(blocks, 1)), dim3(threads));
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLLTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                                  cudaStream_t stream, rcclDdaLaunchCap cap) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet

  const auto geom = ddaArFabricLLGrid(comm, nPk, 512, 1, cap);
  const dim3 grid = geom.first;
  const dim3 block = geom.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Shared epoch counter (same as AG/RS) so bank = flag & 1 is consistent
  // across all LL operation types and cannot alias scratch banks.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  INFO(NCCL_COLL, "DDA fabric AllReduce LL: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u", nRanks, bytes, nPk, grid.x,
       block.x);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback (up to
  // kDdaMaxNranks).
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// One warp carries one slice, so the grid is sized in slices rather than in
// packets: blocks = ceil(slices / warpsPerBlock), capped by the grid budget. A
// message with more slices than the capped grid has warps just means each warp
// takes several passes through the slice loop.
template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLL128OneShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                            ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t slices = ddaLL128Slices(bytes);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  const auto geom = ddaArFabricLLGrid(comm, slices, 512, kDdaLL128Warp, cap);
  const dim3 grid = geom.first;
  const dim3 block = geom.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter the LL tiers use: they share a scratch layout, so one
  // monotonic flag is what keeps either from accepting a line the other left.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL,
       "DDA fabric AllReduce LL128 one-shot: nRanks=%d bytes=%zu slices=%zu grid=%u block=%u "
       "(warp-per-slice, bankSize=%zu)",
       nRanks, bytes, slices, grid.x, block.x, bankSize);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL128<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          bytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL128<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          bytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceFlatLL128<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          bytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// The LL128 two-shot kernel transports one shard per rank, so the grid is sized
// on the shard's slices rather than the whole message's.
template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLL128TwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                            ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t shardBytes = bytes / (size_t)nRanks;
  const size_t slices = ddaLL128Slices(shardBytes);
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  const auto geom = ddaArFabricLLGrid(comm, slices, 512, kDdaLL128Warp, cap);
  const dim3 grid = geom.first;
  const dim3 block = geom.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter every other LL/LL128 tier uses.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL,
       "DDA fabric AllReduce LL128 two-shot: nRanks=%d bytes=%zu shardBytes=%zu slices=%zu grid=%u block=%u "
       "(warp-per-slice, bankSize=%zu)",
       nRanks, bytes, shardBytes, slices, grid.x, block.x, bankSize);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // NRANKS_CT 4/8: unrolled peer loops; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, 4>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev, epochLen, slices,
                          bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, 8>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev, epochLen, slices,
                          bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, 0>), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), shardBytes, comm->rank, nRanks, epochDev, epochLen, slices,
                          bankSize, params);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

// Every phase of the two-shot kernel loops over one shard, with the peer fan-out
// inside it, so the grid covers count/nRanks packets rather than the whole
// message; sizing it on the message would only add blocks whose gtid starts past
// the loop bound.
template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLLTwoShotTyped(const void* sendbuff, void* recvbuff, size_t count,
                                                         ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = (bytes >> 3 ) / (size_t)nRanks; // 8 payload bytes per packet

  const auto geom = ddaArFabricLLGrid(comm, nPk, 1024, 1, cap);
  const dim3 grid = geom.first;
  const dim3 block = geom.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter the one-shot tier uses: the two share a scratch layout, so
  // one monotonic flag is what keeps either from accepting a line the other left.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;
  const size_t bankSize = ddaBankSize(comm->ddaScratchBytes);

  INFO(NCCL_COLL, "DDA fabric AllReduce LL two-shot: nRanks=%d bytes=%zu nPk=%zu grid=%u block=%u", nRanks, bytes, nPk,
       grid.x, block.x);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback (up to
  // kDdaMaxNranks).
  switch (nRanks) {
  case 4:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL<T, 4>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  case 8:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL<T, 8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  default:
    hipExtLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL<T, 0>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          count, comm->rank, nRanks, epochDev, epochLen, bankSize, params);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

// Shape/resource eligibility for the one-shot variant, independent of whether
// the tier is switched on for this size.
bool ddaLLArOneShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (count * ncclTypeSize(datatype) > (size_t)rcclParamDdaLLOneShotThreshold()) {
      return false;
  }

  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // Fabric path: requires the fabric handler + scratch + peer table.
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);

  if (bytes % 16 != 0) {
    return false;
  }
  // One packet carries 8B of payload, so the whole message has to fit the slot
  // the bank gives each rank. A bank too small for one packet per rank takes the
  // tier out entirely.
  if ((bytes >> 3) > ddaLLArSlotPkts(ddaBankSize(comm->ddaScratchBytes), comm->nRanks)) {
    return false;
  }

  return true;
}

// Shape/resource eligibility for the two-shot variant.
bool ddaLLArTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                            ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (count * ncclTypeSize(datatype) > rcclDdaLLTwoShotThreshold(comm)) {
      return false;
  }

  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  // Fabric path: requires the fabric handler + scratch + peer table.
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);

  if (bytes % (size_t)comm->nRanks != 0) {
    return false;
  }

  const size_t bytesPerRank = bytes / (size_t)comm->nRanks;

  if (bytesPerRank % 16 != 0) {
    return false;
  }

  // A two-shot slot is half the one-shot's because a bank carries two staging
  // areas, so only the shard has to fit that halved slot.
  if ((bytesPerRank >> 3) > ddaLLArTwoShotSlotPkts(ddaBankSize(comm->ddaScratchBytes), comm->nRanks)) {
    return false;
  }

  return true;
}

// Shape/resource eligibility for the LL128 one-shot variant.
bool ddaLL128ArOneShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op) {
  if (count * ncclTypeSize(datatype) > (size_t)rcclParamDdaLL128OneShotThreshold()) {
    return false;
  }

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
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);

  // The LL128 line format packs 16B-aligned data with no chunk straddling a
  // line, so a partial 16B chunk has nowhere to go, and the 16B wire accesses
  // need both user buffers aligned to match.
  if (bytes % 16 != 0) {
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(sendbuff) % 16) != 0 || (reinterpret_cast<uintptr_t>(recvbuff) % 16) != 0) {
    return false;
  }
  // can the bank hold the data in the slot
  const size_t slotWords = ddaLL128ArSlotWords(ddaBankSize(comm->ddaScratchBytes), comm->nRanks);
  if (ddaLL128Slices(bytes) * (size_t)kDdaLL128WireWordsPerSlice > slotWords) {
    return false;
  }

  return true;
}

// Shape/resource eligibility for the LL128 two-shot variant.
bool ddaLL128ArTwoShotEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op) {
  if (count * ncclTypeSize(datatype) > rcclDdaLL128TwoShotThreshold(comm)) {
    return false;
  }

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
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (!rcclDdaDatatypeSupported(comm, datatype)) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);

  // Every rank owns exactly one shard, and the LL128 line format packs 16B-aligned
  // data with no chunk straddling a line, so the shard itself must be a whole
  // number of 16B chunks.
  if (bytes % (size_t)comm->nRanks != 0) {
    return false;
  }

  const size_t shardBytes = bytes / (size_t)comm->nRanks;

  if (shardBytes % 16 != 0) {
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(sendbuff) % 16) != 0 || (reinterpret_cast<uintptr_t>(recvbuff) % 16) != 0) {
    return false;
  }

  // A two-shot slot is half the one-shot's because a bank carries two staging
  // areas, so the shard has to fit the slices that halved stride holds.
  const size_t slotWords = ddaLL128ArTwoShotSlotWords(ddaBankSize(comm->ddaScratchBytes), comm->nRanks);
  if (ddaLL128Slices(shardBytes) * (size_t)kDdaLL128WireWordsPerSlice > slotWords) {
    return false;
  }

  return true;
}

// Tier selection: enabled, within this tier's threshold, and shape-eligible.
// Tested cheapest-first, so a message that qualifies for several takes the
// earliest; a run hands sizes over to a later tier by lowering the earlier
// tier's threshold and raising the later one's.
bool ncclAllReduceDdaFabricLLEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op) {
  return ddaLLArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op) ||
         ddaLLArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op);
}

bool ncclAllReduceDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype, ncclRedOp_t op) {
  return ddaLL128ArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op) ||
         ddaLL128ArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op);
}

ncclResult_t ncclAllReduceDdaFabricLL(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  const size_t bytes = count * ncclTypeSize(datatype);

  if (ddaLLArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL, "AllReduce: taking DDA fabric LL one-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    (void)op;
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLLTyped<float>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLLTyped<half>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLLTyped<bf16>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e4m3:
      return ncclAllReduceDdaFabricLLTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e5m2:
      return ncclAllReduceDdaFabricLLTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, count, comm, stream, cap);
    default:
      return ncclInvalidArgument;
    }
  }

  if (ddaLLArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL, "AllReduce: taking DDA fabric LL two-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLLTwoShotTyped<float>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLLTwoShotTyped<half>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLLTwoShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e4m3:
      return ncclAllReduceDdaFabricLLTwoShotTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e5m2:
      return ncclAllReduceDdaFabricLLTwoShotTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, count, comm, stream, cap);
    default:
      return ncclInvalidArgument;
    }
  }

  // Callers gate on ncclAllReduceDdaFabricLLEligible, which is the disjunction of
  // the two selectors above, so neither matching means the caller skipped it.
  WARN("ncclAllReduceDdaFabricLL called for a message no LL tier claims: count=%zu datatype=%d bytes=%zu", count,
       (int)datatype, bytes);
  return ncclInternalError;
}

ncclResult_t ncclAllReduceDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, rcclDdaLaunchCap cap) {
  const size_t bytes = count * ncclTypeSize(datatype);

  if (ddaLL128ArOneShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL,
         "AllReduce: taking DDA fabric LL128 one-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLL128OneShotTyped<float>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLL128OneShotTyped<half>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLL128OneShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e4m3:
      return ncclAllReduceDdaFabricLL128OneShotTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, count, comm, stream,
                                                                           cap);
    case ncclFloat8e5m2:
      return ncclAllReduceDdaFabricLL128OneShotTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, count, comm, stream,
                                                                           cap);
    default:
      return ncclInvalidArgument;
    }
  }

  if (ddaLL128ArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
    INFO(NCCL_COLL,
         "AllReduce: taking DDA fabric LL128 two-shot path: nRanks=%d nNodes=%d count=%zu datatype=%d bytes=%zu",
         comm->nRanks, comm->nNodes, count, (int)datatype, bytes);
    switch (datatype) {
    case ncclFloat32:
      return ncclAllReduceDdaFabricLL128TwoShotTyped<float>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat16:
      return ncclAllReduceDdaFabricLL128TwoShotTyped<half>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclBfloat16:
      return ncclAllReduceDdaFabricLL128TwoShotTyped<bf16>(sendbuff, recvbuff, count, comm, stream, cap);
    case ncclFloat8e4m3:
      return ncclAllReduceDdaFabricLL128TwoShotTyped<dda::common::fp8e4m3>(sendbuff, recvbuff, count, comm, stream,
                                                                           cap);
    case ncclFloat8e5m2:
      return ncclAllReduceDdaFabricLL128TwoShotTyped<dda::common::fp8e5m2>(sendbuff, recvbuff, count, comm, stream,
                                                                           cap);
    default:
      return ncclInvalidArgument;
    }
  }

  WARN("ncclAllReduceDdaFabricLL128 called for a message no LL128 tier claims: count=%zu datatype=%d bytes=%zu", count,
       (int)datatype, bytes);
  return ncclInternalError;
}

uint32_t ncclAllReduceDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype, rcclDdaLaunchCap cap) {
  const size_t nPk = (count * ncclTypeSize(datatype)) >> 3;
  // Uncapped reports the one-shot grid, a bound on the two-shot's; a capped call reports the tier that launches.
  if ((cap.blocks > 0 || cap.threads > 0) &&
      !ddaLLArOneShotEligible(comm, nullptr, nullptr, count, datatype, ncclSum)) {
    const dim3 grid = ddaArFabricLLGrid(comm, nPk / (size_t)std::max(comm->nRanks, 1), 1024, 1, cap).first;
    return grid.x * grid.y;
  }
  const dim3 grid = ddaArFabricLLGrid(comm, nPk, 512, 1, cap).first;
  return grid.x * grid.y;
}

// Reports the one-shot tier's grid, as the LL counterpart above reports the LL
// one-shot's: the two-shot tier sizes its grid on a shard rather than the whole
// message, so it runs on no more blocks than this. A capped call reports the tier that launches.
uint32_t ncclAllReduceDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype,
                                           rcclDdaLaunchCap cap) {
  const size_t bytes = count * ncclTypeSize(datatype);
  if ((cap.blocks > 0 || cap.threads > 0) &&
      !ddaLL128ArOneShotEligible(comm, nullptr, nullptr, count, datatype, ncclSum)) {
    const size_t shardBytes = bytes / (size_t)std::max(comm->nRanks, 1);
    const dim3 grid = ddaArFabricLLGrid(comm, ddaLL128Slices(shardBytes), 512, kDdaLL128Warp, cap).first;
    return grid.x * grid.y;
  }
  const dim3 grid = ddaArFabricLLGrid(comm, ddaLL128Slices(bytes), 512, kDdaLL128Warp, cap).first;
  return grid.x * grid.y;
}
