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
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_peerpar.h"
#include "archinfo.h"
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
#include <type_traits>
#include <utility>

RCCL_PARAM_DECLARE(DdaLL);
RCCL_PARAM_DECLARE(DdaLLOneShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128OneShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128AsyncWire);
RCCL_PARAM_DECLARE(DdaLL128ArPeerParallel);
RCCL_PARAM_DECLARE(DdaLL128ArSplitSend);

namespace {

using dda::common::ddaLLArSlotPkts;
using dda::common::ddaLLArTwoShotSlotPkts;

using dda::common::ddaBankSize;
using dda::common::ddaLL128ArSlotWords;
using dda::common::ddaLL128ArTwoShotSlotWords;
using dda::common::ddaLL128Slices;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;

// Block width of each tier when uncapped.
constexpr unsigned kDdaArLLOneShotThreads = 512;
constexpr unsigned kDdaArLLTwoShotThreads = 1024;
constexpr unsigned kDdaArLL128Threads = 512;

// Single source of the LL and LL128 launch geometry: a 1-D grid over `units` of work, `unitThreads` threads each,
// in blocks of `threads`, capped low (LL serves small messages where latency, not occupancy, dominates). `cap` lowers
// the blocks and the threads (dda::common::ddaCapThreads); {0, 0} keeps the tier's.
static inline std::pair<dim3, dim3> ddaArFabricLLGrid(const ncclComm* comm, size_t units, unsigned threads,
                                                      unsigned unitThreads, rcclDdaLaunchCap cap) {
  threads = dda::common::ddaCapThreads(threads, cap.threads);
  const size_t nBlocksMax = dda::common::ddaCapBlocks((size_t)std::max(comm->ddaFabricMaxBlocks, 1), cap.blocks);
  const size_t unitsPerBlock = threads / unitThreads;
  const size_t blocks = std::min((units + unitsPerBlock - 1) / unitsPerBlock, nBlocksMax);
  return std::make_pair(dim3((unsigned)std::max<size_t>(blocks, 1)), dim3(threads));
}

// Per-tier geometry for `bytes` of message, shared by each launcher and the *Blocks reporters.
// LL one-shot: 8-byte packets of the whole message, one thread each.
static inline std::pair<dim3, dim3> ddaArLLOneShotGeom(const ncclComm* comm, size_t bytes, rcclDdaLaunchCap cap) {
  return ddaArFabricLLGrid(comm, bytes >> 3, kDdaArLLOneShotThreads, 1, cap);
}

// LL two-shot: 8-byte packets of one shard, one thread each.
static inline std::pair<dim3, dim3> ddaArLLTwoShotGeom(const ncclComm* comm, size_t bytes, rcclDdaLaunchCap cap) {
  return ddaArFabricLLGrid(comm, (bytes >> 3) / (size_t)std::max(comm->nRanks, 1), kDdaArLLTwoShotThreads, 1, cap);
}

static_assert(kDdaArLL128Threads <= (unsigned)dda::common::kDdaLL128ArPeerMaxThreads,
              "the peer-parallel LL128 kernels size their LDS for at most kDdaLL128ArPeerMaxThreads");

// Threads that carry one LL128 slice: a warp, or a group of up to one warp per rank when peer-parallel.
static inline unsigned ddaArLL128SliceThreads(const ncclComm* comm, bool peerPar, rcclDdaLaunchCap cap) {
  if (!peerPar) {
    return kDdaLL128Warp;
  }
  const unsigned warps = dda::common::ddaCapThreads(kDdaArLL128Threads, cap.threads) / kDdaLL128Warp;
  return (unsigned)dda::common::ddaLL128ArPeerWarps(comm->nRanks, (int)warps) * kDdaLL128Warp;
}

// Block width of an LL128 launch over `slices`. Peer-parallel blocks start at one warp group (one slice) each, so the
// slices spread over the most CUs and a barrier waits on one slice only; a block takes more groups, doubling up to
// kDdaArLL128Threads, only while the group-per-block grid would pass the block cap.
static inline unsigned ddaArLL128Threads(const ncclComm* comm, size_t slices, bool peerPar, rcclDdaLaunchCap cap) {
  if (!peerPar) {
    return kDdaArLL128Threads;
  }
  const unsigned maxThreads = dda::common::ddaCapThreads(kDdaArLL128Threads, cap.threads);
  const unsigned groupThreads = ddaArLL128SliceThreads(comm, peerPar, cap);
  const size_t nBlocksMax = dda::common::ddaCapBlocks((size_t)std::max(comm->ddaFabricMaxBlocks, 1), cap.blocks);
  unsigned threads = groupThreads;
  while (threads * 2 <= maxThreads) {
    const size_t groups = threads / groupThreads;
    if ((slices + groups - 1) / groups <= nBlocksMax) {
      break;
    }
    threads *= 2;
  }
  return threads;
}

// LL128 one-shot: slices of the whole message.
static inline std::pair<dim3, dim3> ddaArLL128OneShotGeom(const ncclComm* comm, size_t bytes, bool peerPar,
                                                          rcclDdaLaunchCap cap) {
  const size_t slices = ddaLL128Slices(bytes);
  return ddaArFabricLLGrid(comm, slices, ddaArLL128Threads(comm, slices, peerPar, cap),
                           ddaArLL128SliceThreads(comm, peerPar, cap), cap);
}

// LL128 two-shot: slices of one shard.
static inline std::pair<dim3, dim3> ddaArLL128TwoShotGeom(const ncclComm* comm, size_t bytes, bool peerPar,
                                                          rcclDdaLaunchCap cap) {
  const size_t slices = ddaLL128Slices(bytes / (size_t)std::max(comm->nRanks, 1));
  return ddaArFabricLLGrid(comm, slices, ddaArLL128Threads(comm, slices, peerPar, cap),
                           ddaArLL128SliceThreads(comm, peerPar, cap), cap);
}

// The LDS-staged writers need gfx1250's async LDS stores (ASYNC_COPY_SUPPORTED) and one LL128 warp per wave.
static bool ddaArLL128StagedWireOk(const ncclComm* comm) {
  return comm->archName != nullptr && IsArchMatch(comm->archName, "gfx1250") && comm->WarpSize == kDdaLL128Warp;
}

// Calls pick(std::integral_constant<int, N>{}): NRANKS_CT 4/8 unroll the peer loops, 0 is the runtime fallback.
template <typename Pick>
static auto ddaArPickRanksCt(int nRanks, Pick pick) {
  switch (nRanks) {
  case 4:
    return pick(std::integral_constant<int, 4>{});
  case 8:
    return pick(std::integral_constant<int, 8>{});
  default:
    return pick(std::integral_constant<int, 0>{});
  }
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLLTyped(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                                  cudaStream_t stream, rcclDdaLaunchCap cap) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nPk = bytes >> 3; // 8 payload bytes per packet (for logging)

  const auto geom = ddaArLLOneShotGeom(comm, bytes, cap);
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

// Smallest per-rank shard whose LL128 two-shot takes the staged writer: enough slices per warp to pay off.
constexpr size_t kDdaArLL128AsyncMinShardBytes = 2 * 1024 * 1024;

// Writer of an LL128 two-shot launch: staged from kDdaArLL128AsyncMinShardBytes where ddaArLL128StagedWireOk.
// Opt-in with RCCL_DDA_LL128_ASYNC_WIRE=1; at 0 (default) every launch takes the cooperative writer and no dynamic LDS.
static int ddaArLL128Wire(const ncclComm* comm, size_t bytes) {
  if (rcclParamDdaLL128AsyncWire() == 0) {
    return dda::common::kDdaLL128WireCoop;
  }
  const size_t shard = bytes / (size_t)std::max(comm->nRanks, 1);
  return ddaArLL128StagedWireOk(comm) && shard >= kDdaArLL128AsyncMinShardBytes ? dda::common::kDdaLL128WireAsync
                                                                                  : dda::common::kDdaLL128WireCoop;
}

// Dynamic LDS of an LL128 launch: the staged writer's two windows per warp.
static size_t ddaArLL128SharedBytes(int wire, unsigned threads) {
  return wire == dda::common::kDdaLL128WireCoop ? 0 : dda::common::ddaLL128StageBytes(threads);
}

// Signature of the LL128 one-shot and two-shot kernels.
template <typename T>
using DdaArLL128Kernel = void (*)(T* const*, T*, const T*, size_t, int, int, uint32_t*, int, size_t, size_t,
                                  dda::common::DdaKernelParams);

// Whether the one-shot takes the peer-parallel kernel.
static bool ddaArLL128OneShotPeerPar() {
  return rcclParamDdaLL128ArPeerParallel() != 0;
}

// Whether the two-shot takes the peer-parallel kernel: cooperative writer only (the staged writer keeps its kernel).
// At every shard size: on gfx1250 h13 it beats the flat two-shot past 8 MiB shards too (2-4% at 4 and 8 ranks).
static bool ddaArLL128TwoShotPeerPar(int wire) {
  return rcclParamDdaLL128ArPeerParallel() != 0 && wire == dda::common::kDdaLL128WireCoop;
}

template <typename T>
static DdaArLL128Kernel<T> ddaArLL128OneShotKernel(int nRanks) {
  return ddaArPickRanksCt(nRanks, [](auto n) -> DdaArLL128Kernel<T> {
    constexpr int kN = decltype(n)::value;
    return &dda::common::ddaAllReduceFlatLL128<T, kN>;
  });
}

// Signature of the peer-parallel one-shot: the LL128 kernel's, plus the team block count.
template <typename T>
using DdaArLL128PeerParOneShotKernel = void (*)(T* const*, T*, const T*, size_t, int, int, uint32_t*, int, size_t,
                                                size_t, dda::common::DdaKernelParams, int);

template <typename T>
static DdaArLL128PeerParOneShotKernel<T> ddaArLL128OneShotPeerParKernel(int nRanks, bool split) {
  return ddaArPickRanksCt(nRanks, [split](auto n) -> DdaArLL128PeerParOneShotKernel<T> {
    constexpr int kN = decltype(n)::value;
    return split ? &dda::common::ddaAllReduceFlatLL128PeerPar<T, kN, true>
                 : &dda::common::ddaAllReduceFlatLL128PeerPar<T, kN, false>;
  });
}

// An LL128 one-shot launch: grid, block, and for a split send the team blocks before the sender columns.
struct DdaArLL128OneShotLaunch {
  dim3 grid;
  dim3 block;
  int teamBlocks;
  bool split;
};

// Split send (RCCL_DDA_LL128_AR_SPLIT_SEND) adds nRanks - 1 sender columns after the team blocks of a peer-parallel,
// out-of-place one-shot, `height` blocks each so a sender warp carries about one slice. -1 (default): only while
// team and sender blocks fit the block cap as sized; past it the team blocks would shrink, which costs more than
// the split saves. 1: always, shrinking both to fit the cap. 0: never.
static DdaArLL128OneShotLaunch ddaArLL128OneShotLaunch(const ncclComm* comm, size_t bytes, bool peerPar,
                                                       bool outOfPlace, rcclDdaLaunchCap cap) {
  const auto geom = ddaArLL128OneShotGeom(comm, bytes, peerPar, cap);
  DdaArLL128OneShotLaunch launch{geom.first, geom.second, (int)geom.first.x, false};
  const int64_t mode = rcclParamDdaLL128ArSplitSend();
  if (!peerPar || !outOfPlace || mode == 0 || comm->nRanks < 2) {
    return launch;
  }
  const long maxBlocks = (long)dda::common::ddaCapBlocks((size_t)std::max(comm->ddaFabricMaxBlocks, 1), cap.blocks);
  const long cols = comm->nRanks - 1;
  const long nwarps = std::max<long>((long)launch.block.x / kDdaLL128Warp, 1);
  long team = (long)launch.grid.x;
  long height = std::max<long>(((long)ddaLL128Slices(bytes) + nwarps - 1) / nwarps, 1);
  if (team + cols * height > maxBlocks) {
    if (mode != 1) {
      return launch;
    }
    const double f = (double)maxBlocks / (double)(team + cols * height);
    team = std::max<long>((long)((double)team * f), 1);
    height = std::max<long>((maxBlocks - team) / cols, 1);
  }
  if (team + cols * height > maxBlocks) {
    return launch; // not even one sender block per peer fits
  }
  launch.split = true;
  launch.teamBlocks = (int)team;
  launch.grid.x = (unsigned)(team + cols * height);
  return launch;
}

template <typename T>
static DdaArLL128Kernel<T> ddaArLL128TwoShotKernel(int nRanks, int wire, bool peerPar) {
  return ddaArPickRanksCt(nRanks, [wire, peerPar](auto n) -> DdaArLL128Kernel<T> {
    constexpr int kN = decltype(n)::value;
    if (peerPar) {
      return &dda::common::ddaAllReduceTwoShotLL128PeerPar<T, kN>;
    }
    return wire == dda::common::kDdaLL128WireAsync
             ? &dda::common::ddaAllReduceTwoShotLL128<T, kN, dda::common::kDdaLL128WireAsync>
             : &dda::common::ddaAllReduceTwoShotLL128<T, kN, dda::common::kDdaLL128WireCoop>;
  });
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

  const bool peerPar = ddaArLL128OneShotPeerPar();
  const DdaArLL128OneShotLaunch launch = ddaArLL128OneShotLaunch(comm, bytes, peerPar, sendbuff != recvbuff, cap);
  const dim3 grid = launch.grid;
  const dim3 block = launch.block;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter the LL tiers use: they share a scratch layout, so one
  // monotonic flag is what keeps either from accepting a line the other left.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL,
       "DDA fabric AllReduce LL128 one-shot: nRanks=%d bytes=%zu slices=%zu grid=%u block=%u peerPar=%d "
       "split=%d teamBlocks=%d (bankSize=%zu)",
       nRanks, bytes, slices, grid.x, block.x, (int)peerPar, (int)launch.split, launch.teamBlocks, bankSize);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (peerPar) {
    hipExtLaunchKernelGGL(ddaArLL128OneShotPeerParKernel<T>(nRanks, launch.split), grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff),
                          static_cast<const T*>(sendbuff), bytes, comm->rank, nRanks, epochDev, epochLen, slices,
                          bankSize, params, launch.teamBlocks);
  } else {
    hipExtLaunchKernelGGL(ddaArLL128OneShotKernel<T>(nRanks), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                          bytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);
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

  const int wire = ddaArLL128Wire(comm, bytes);
  const bool peerPar = ddaArLL128TwoShotPeerPar(wire);
  const auto geom = ddaArLL128TwoShotGeom(comm, bytes, peerPar, cap);
  const dim3 grid = geom.first;
  const dim3 block = geom.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  // Same epoch counter every other LL/LL128 tier uses.
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  const size_t lds = ddaArLL128SharedBytes(wire, block.x);

  INFO(NCCL_COLL,
       "DDA fabric AllReduce LL128 two-shot: nRanks=%d bytes=%zu shardBytes=%zu slices=%zu grid=%u block=%u wire=%d "
       "lds=%zu peerPar=%d (bankSize=%zu)",
       nRanks, bytes, shardBytes, slices, grid.x, block.x, wire, lds, (int)peerPar, bankSize);

  const dda::common::DdaKernelParams params{comm->abortFlagDev};
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  hipExtLaunchKernelGGL(ddaArLL128TwoShotKernel<T>(nRanks, wire, peerPar), grid, block, lds, stream, /*startEvent=*/nullptr,
                        stopEvent, /*flags=*/0, peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
                        shardBytes, comm->rank, nRanks, epochDev, epochLen, slices, bankSize, params);

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

  const auto geom = ddaArLLTwoShotGeom(comm, bytes, cap);
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
  // Past the rank-scaled crossover the two-shot is faster, but only on a message it
  // splits into whole shards; anything else stays here up to the unscaled cap
  // rather than leaving DDA.
  const size_t msgBytes = count * ncclTypeSize(datatype);
  if (msgBytes > rcclDdaLL128OneShotThreshold(comm) &&
      (msgBytes > (size_t)rcclParamDdaLL128OneShotThreshold() ||
       ddaLL128ArTwoShotEligible(comm, sendbuff, recvbuff, count, datatype, op))) {
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

// Reports the grid of the tier ncclAllReduceDdaFabricLL launches: one-shot when eligible, else two-shot.
uint32_t ncclAllReduceDdaFabricLLBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype, rcclDdaLaunchCap cap) {
  const size_t bytes = count * ncclTypeSize(datatype);
  const bool oneShot = ddaLLArOneShotEligible(comm, nullptr, nullptr, count, datatype, ncclSum);
  const dim3 grid = (oneShot ? ddaArLLOneShotGeom(comm, bytes, cap) : ddaArLLTwoShotGeom(comm, bytes, cap)).first;
  return grid.x * grid.y;
}

// Reports the grid of the tier ncclAllReduceDdaFabricLL128 launches: one-shot when eligible, else two-shot.
uint32_t ncclAllReduceDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype,
                                           rcclDdaLaunchCap cap) {
  const size_t bytes = count * ncclTypeSize(datatype);
  const bool oneShot = ddaLL128ArOneShotEligible(comm, nullptr, nullptr, count, datatype, ncclSum);
  // One-shot: the out-of-place launch, sender columns included when it splits.
  const dim3 grid =
      oneShot ? ddaArLL128OneShotLaunch(comm, bytes, ddaArLL128OneShotPeerPar(), /*outOfPlace=*/true, cap).grid
              : ddaArLL128TwoShotGeom(comm, bytes, ddaArLL128TwoShotPeerPar(ddaArLL128Wire(comm, bytes)),
                                      cap)
                  .first;
  return grid.x * grid.y;
}
