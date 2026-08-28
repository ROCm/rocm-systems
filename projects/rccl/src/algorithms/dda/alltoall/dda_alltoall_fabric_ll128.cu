/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric all-to-all.
 * Personalized analogue of dda_all_gather_fabric_ll128.cu; reuses the codepath-
 * agnostic ddaAllToAllFabricLL128 kernel from alltoall_dda_fabric_ll128.h.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/dda_alltoall.h"

#include "algorithms/dda/alltoall/alltoall_dda_fabric_ll128.h"
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

// Runtime-adjustable LL128 AllToAll block size (threads/block). Must be a
// multiple of 16 (lanes per 128B line) in [16, 1024]; invalid values fall back
// to the tuned default. Env: RCCL_DDA_LL128_A2A_THREADS.
RCCL_PARAM(DdaLL128A2AThreads, "DDA_LL128_A2A_THREADS", 1024);

namespace {

using dda::common::kDdaLL128A2AMaxPerChunkBytes;
using dda::common::kDdaLL128A2ASlotStrideLines;
using dda::common::kDdaLL128DataElems;
using dda::common::LLLine128;
using nccl_dda_detail::kDdaLLAgMaxBlocksPerPeer;

// LL128 scratch: 2 banks * nRanks slots * kDdaLL128A2ASlotStrideLines * 128B.
static inline size_t ddaLL128A2AScratchSize(int nRanks) {
  return (size_t)2 * (size_t)nRanks * kDdaLL128A2ASlotStrideLines * sizeof(LLLine128);
}

// Grid-shape tuning knobs (env-overridable); defaults tuned for gfx1250.
constexpr size_t kDdaLL128A2ALinesPerBlock = 4;

static inline size_t ddaLL128A2ALinesPerBlockEnv(size_t dflt) {
  const char* s = getenv("RCCL_DDA_LL128_A2A_LPB");
  if (s && *s) {
    long v = strtol(s, nullptr, 10);
    if (v > 0) return (size_t)v;
  }
  return dflt;
}
static inline int ddaLL128A2AMaxBppEnv(int dflt) {
  const char* s = getenv("RCCL_DDA_LL128_A2A_MAXBPP");
  if (s && *s) {
    long v = strtol(s, nullptr, 10);
    if (v > 0) return (int)v;
  }
  return dflt;
}
// Validated block size from the runtime flag; falls back to `dflt` if the
// configured value is not a multiple of 16 in [16, 1024].
static inline unsigned ddaLL128A2AThreads(unsigned dflt) {
  const int64_t v = rcclParamDdaLL128A2AThreads();
  if (v >= 16 && v <= 1024 && (v % 16) == 0) {
    return (unsigned)v;
  }
  return dflt;
}

static inline int ddaLL128A2ABlocksPerPeer(size_t perChunkBytes) {
  const size_t nWords = perChunkBytes >> 3;
  const size_t numLines = (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;
  const size_t linesPerBlock = ddaLL128A2ALinesPerBlockEnv(kDdaLL128A2ALinesPerBlock);
  const int maxBpp = ddaLL128A2AMaxBppEnv(kDdaLLAgMaxBlocksPerPeer);
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
// device epoch array.
static inline std::pair<dim3, dim3> ddaAllToAllFabricLL128Geom(ncclComm* comm, size_t perChunkBytes) {
  const unsigned threads = ddaLL128A2AThreads(1024); // multiple of 16 (lanes/line)
  int blocksPerPeer = ddaLL128A2ABlocksPerPeer(perChunkBytes);
  if (comm->nRanks * blocksPerPeer > comm->ddaLLEpochLen) {
    blocksPerPeer = comm->ddaLLEpochLen / comm->nRanks;
    if (blocksPerPeer < 1) blocksPerPeer = 1;
  }
  return std::make_pair(dim3((unsigned)comm->nRanks, (unsigned)blocksPerPeer), dim3(threads));
}

template <typename T>
static ncclResult_t ncclAllToAllDdaFabricLL128Typed(
  const void* sendbuff, void* recvbuff,
  size_t count, // per-peer element count of T (== bytes when T == int8_t)
  ncclComm* comm, cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t perChunkBytes = count * sizeof(T);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  auto gridBlock = ddaAllToAllFabricLL128Geom(comm, perChunkBytes);
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;
  const int blocksPerPeer = (int)grid.y;

  INFO(NCCL_COLL, "DDA fabric AllToAll LL128: nRanks=%d perChunkBytes=%zu grid=%ux%u block=%u (block-per-peer, bpp=%d)",
       nRanks, perChunkBytes, grid.x, grid.y, block.x, blocksPerPeer);

  switch (nRanks) {
  case 4:
    dda::common::ddaAllToAllFabricLL128<T, 4>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perChunkBytes,
                                   comm->rank, nRanks, epochDev, epochLen);
    break;
  case 8:
    dda::common::ddaAllToAllFabricLL128<T, 8>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perChunkBytes,
                                   comm->rank, nRanks, epochDev, epochLen);
    break;
  default:
    dda::common::ddaAllToAllFabricLL128<T, 0>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), perChunkBytes,
                                   comm->rank, nRanks, epochDev, epochLen);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllToAllDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                        ncclDataType_t datatype) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > dda::common::kDdaMaxNranks) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t perChunkBytes = count * ncclTypeSize(datatype);
  if (perChunkBytes % 16 != 0) {
    return false;
  }
  if (perChunkBytes > kDdaLL128A2AMaxPerChunkBytes) {
    return false;
  }
  if (ddaLL128A2AScratchSize(comm->nRanks) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

uint32_t ncclAllToAllDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const auto grid = ddaAllToAllFabricLL128Geom(comm, count * ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllToAllDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                        ncclComm* comm, cudaStream_t stream) {
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return ncclInvalidArgument;
  }
  const int typeSize = ncclTypeSize(datatype);
  return ncclAllToAllDdaFabricLL128Typed<int8_t>(sendbuff, recvbuff, count * typeSize, comm, stream);
}
