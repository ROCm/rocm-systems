/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric all-gather.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_gather/dda_all_gather.h"

#include "algorithms/dda/all_gather/all_gather_dda_fabric_ll128.h"
#include "checks.h"
#include "comm.h"
#include "algorithms/dda/dda_init_detail.h" // nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer
#include "debug.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib> // getenv (grid-shape tuning overrides)
#include <utility>

// Runtime-adjustable LL128 AllGather block size (threads/block). Must be a
// multiple of 16 (lanes per 128B line) in [16, 1024]; invalid values fall back
// to the tuned default. Env: RCCL_DDA_LL128_AG_THREADS.
RCCL_PARAM(DdaLL128AgThreads, "DDA_LL128_AG_THREADS", 1024);

namespace {

// Grid-shape tuning knobs (env-overridable) for the LL128 AllGather mapping.
// Defaults are tuned for gfx1250; env overrides are kept for per-platform
// retuning: RCCL_DDA_LL128_AG_LPB (lines/block), _MAXBPP (blocks/peer cap,
// clamped to the device epoch array).
static inline size_t ddaLL128AgLinesPerBlockEnv(size_t dflt) {
  const char* s = getenv("RCCL_DDA_LL128_AG_LPB");
  if (s && *s) {
    long v = strtol(s, nullptr, 10);
    if (v > 0) return (size_t)v;
  }
  return dflt;
}
static inline int ddaLL128AgMaxBppEnv(int dflt) {
  const char* s = getenv("RCCL_DDA_LL128_AG_MAXBPP");
  if (s && *s) {
    long v = strtol(s, nullptr, 10);
    if (v > 0) return (int)v;
  }
  return dflt;
}
// Validated block size from the runtime flag; falls back to `dflt` if the
// configured value is not a multiple of 16 in [16, 1024].
static inline unsigned ddaLL128AgThreads(unsigned dflt) {
  const int64_t v = rcclParamDdaLL128AgThreads();
  if (v >= 16 && v <= 1024 && (v % 16) == 0) {
    return (unsigned)v;
  }
  return dflt;
}

using dda::common::kDdaLL128AgMaxPerRankBytes;
using dda::common::kDdaLL128AgSlotStrideLines;
using dda::common::kDdaLL128DataElems;
using dda::common::LLLine128;
using nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer;

// LL128 scratch: 2 banks * nRanks slots * kDdaLL128AgSlotStrideLines * 128B.
static inline size_t ddaLL128AgScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLL128AgSlotStrideLines * sizeof(LLLine128);
}

// Adaptive block-per-peer fan-out over 128B lines. One block per peer for small
// messages; larger ones split a peer's line range across blocksPerPeer blocks.
// Tuned on gfx1250 (4-GPU): a small lines-per-block split saturates the
// blocks-per-peer cap early, and 1024-thread blocks (64 line-groups) maximize
// per-block memory-level parallelism. See ddaLL128AgThreadsEnv default below.
constexpr size_t kDdaLL128AgLinesPerBlock = 4;

static inline int ddaLL128AgBlocksPerPeer(size_t perRankBytes) {
  const size_t nWords = perRankBytes >> 3;
  const size_t numLines = (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;
  const size_t linesPerBlock = ddaLL128AgLinesPerBlockEnv(kDdaLL128AgLinesPerBlock);
  const int maxBpp = ddaLL128AgMaxBppEnv(kDdaLLAgMaxBlocksPerPeer);
  if (numLines <= linesPerBlock) {
    return 1;
  }
  size_t bpp = (numLines + linesPerBlock - 1) / linesPerBlock;
  if (bpp > (size_t)maxBpp) {
    bpp = (size_t)maxBpp;
  }
  return (int)bpp;
}

// Single source of the launch geometry: grid.x = peer (nRanks), grid.y = the
// per-peer line split, clamped so flatBlockId (nRanks*bpp-1) stays within the
// device epoch array (sized for nRanks*kDdaLLAgMaxBlocksPerPeer cells).
static inline std::pair<dim3, dim3> ddaAllGatherFabricLL128Geom(ncclComm* comm, size_t perRankBytes) {
  const unsigned threads = ddaLL128AgThreads(1024); // multiple of 16 (lanes/line)
  int blocksPerPeer = ddaLL128AgBlocksPerPeer(perRankBytes);
  if (comm->nRanks * blocksPerPeer > comm->ddaLLEpochLen) {
    blocksPerPeer = comm->ddaLLEpochLen / comm->nRanks;
    if (blocksPerPeer < 1) blocksPerPeer = 1;
  }
  return std::make_pair(dim3((unsigned)comm->nRanks, (unsigned)blocksPerPeer), dim3(threads));
}

template <typename T>
static ncclResult_t ncclAllGatherDdaFabricLL128Typed(
  const void* sendbuff, void* recvbuff,
  size_t sendcount, // per-rank element count of T (== bytes when T == int8_t)
  ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t perRankBytes = sendcount * sizeof(T);

  auto gridBlock = ddaAllGatherFabricLL128Geom(comm, perRankBytes);
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;
  const int blocksPerPeer = (int)grid.y;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL, "DDA fabric AllGather LL128: nRanks=%d perRankBytes=%zu grid=%ux%u block=%u (block-per-peer, bpp=%d)",
       nRanks, perRankBytes, grid.x, grid.y, block.x, blocksPerPeer);

  // NRANKS_CT 4/8: unrolled; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    dda::common::ddaAllGatherFabricLL128<T, 4>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perRankBytes,
                                   comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    dda::common::ddaAllGatherFabricLL128<T, 8>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perRankBytes,
                                   comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    dda::common::ddaAllGatherFabricLL128<T, 0>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perRankBytes,
                                   comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllGatherDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                         ncclDataType_t datatype) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (sendcount == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t perRankBytes = sendcount * ncclTypeSize(datatype);
  if (perRankBytes % 16 != 0) {
    return false;
  }
  if (perRankBytes > kDdaLL128AgMaxPerRankBytes) {
    return false;
  }
  if (ddaLL128AgScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

uint32_t ncclAllGatherDdaFabricLL128Blocks(ncclComm* comm, size_t sendcount, ncclDataType_t datatype) {
  const auto grid = ddaAllGatherFabricLL128Geom(comm, sendcount * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllGatherDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t sendcount,
                                         ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  // AllGather is a pure copy, so the payload moves as raw bytes: instantiate the
  // kernel once for int8_t and scale the count, like ncclAllGatherDdaFabricLL.
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllGatherDdaFabricLL128Typed<int8_t>(sendbuff, recvbuff, sendcount * typeSize, comm, stream);
}
