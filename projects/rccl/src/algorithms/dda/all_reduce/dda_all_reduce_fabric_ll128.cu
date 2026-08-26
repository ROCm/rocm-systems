/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128-protocol DDA fabric all-reduce.
 * All-reduce analogue of dda_all_gather_fabric_ll128.cu; reuses the codepath-
 * agnostic ddaAllReduceFlatLL128 kernel from all_reduce_dda_fabric_ll128.h.
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/all_reduce/dda_all_reduce.h"

#include "algorithms/dda/all_reduce/all_reduce_dda_fabric_ll128.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h" // dda::common::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

// Runtime-adjustable LL128 AllReduce block size (threads/block). Must be a
// multiple of 16 (lanes per 128B line) in [16, 1024]; invalid values fall back
// to the tuned default. Env: RCCL_DDA_LL128_AR_THREADS.
RCCL_PARAM(DdaLL128ArThreads, "DDA_LL128_AR_THREADS", 1024);

namespace {

using dda::common::kDdaLL128DataElems;
using dda::common::kDdaLL128Lanes;
using dda::common::LLLine128;

// Per-call slot stride in 128B lines for a message of `numLines` lines. The
// stride matches the message exactly (compact layout) so small all-reduces keep
// their scratch slots close together for good L2/TLB locality.
static inline size_t ddaLL128ArSlotLines(size_t numLines) {
  return numLines;
}

// LL128 scratch for this call: 2 banks * nRanks slots * slotLines * 128B.
static inline size_t ddaLL128ArScratchSize(int nRanks, size_t numLines) {
  return (size_t)2 * (size_t)nRanks * ddaLL128ArSlotLines(numLines) * sizeof(LLLine128);
}

// Validated block size from the runtime flag; falls back to `dflt` if the
// configured value is not a multiple of 16 in [16, 1024].
static inline unsigned ddaLL128ArThreads(unsigned dflt) {
  const int64_t v = rcclParamDdaLL128ArThreads();
  if (v >= 16 && v <= 1024 && (v % 16) == 0) {
    return (unsigned)v;
  }
  return dflt;
}

// Single source of the launch geometry: 1-D grid over 128B line-groups (each
// block has threads/16 groups), capped by the block count and clamped so
// flatBlockId (blockIdx.x) stays within the device epoch array.
static inline std::pair<dim3, dim3> ddaAllReduceFabricLL128Geom(ncclComm* comm, size_t count, int typeSize) {
  const size_t nWords = ((size_t)count * (size_t)typeSize) >> 3;
  const size_t numLines = (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;
  const unsigned threads = ddaLL128ArThreads(1024); // multiple of 16 (lanes/line)
  const size_t groups = threads / (unsigned)kDdaLL128Lanes;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>((numLines + groups - 1) / groups, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  if ((int)blocks > comm->ddaLLEpochLen) {
    blocks = (unsigned)comm->ddaLLEpochLen;
    if (blocks == 0) blocks = 1;
  }
  return std::make_pair(dim3(blocks), dim3(threads));
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLL128Typed(const void* sendbuff, void* recvbuff, size_t count, ncclComm* comm,
                                                     cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nWords = bytes >> 3;
  const size_t numLines = (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;
  const size_t slotStrideLines = ddaLL128ArSlotLines(numLines);

  auto gridBlock = ddaAllReduceFabricLL128Geom(comm, count, sizeof(T));
  const dim3 grid = gridBlock.first;
  const dim3 block = gridBlock.second;

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(NCCL_COLL, "DDA fabric AllReduce LL128: nRanks=%d bytes=%zu numLines=%zu grid=%u block=%u", nRanks, bytes,
       numLines, grid.x, block.x);

  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    dda::common::ddaAllReduceFlatLL128<T, 4>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank,
                                   nRanks, epochDev, epochLen, slotStrideLines);
    break;
  case 8:
    dda::common::ddaAllReduceFlatLL128<T, 8>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank,
                                   nRanks, epochDev, epochLen, slotStrideLines);
    break;
  default:
    dda::common::ddaAllReduceFlatLL128<T, 0>
      <<<grid, block, 0, stream>>>(peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff), count, comm->rank,
                                   nRanks, epochDev, epochLen, slotStrideLines);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaFabricLL128Eligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype, ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
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
  if (datatype != ncclFloat32 && datatype != ncclFloat16 && datatype != ncclBfloat16) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);
  // Payload is staged as 8-byte words, so it must be a whole number of words.
  if (bytes % 8 != 0) {
    return false;
  }
  // Use the runtime LL128 threshold (RCCL_DDA_LL128_THRESHOLD) as the cap.
  const int64_t ll128Thresh = rcclParamDdaLL128Threshold();
  if (ll128Thresh <= 0 || bytes > (size_t)ll128Thresh) {
    return false;
  }
  // Scratch is sized from the actual message (compact per-call slot stride), so
  // eligibility is bounded by the runtime scratch capacity for this size.
  const size_t nWords = bytes >> 3;
  const size_t numLines = (nWords + (size_t)kDdaLL128DataElems - 1) / (size_t)kDdaLL128DataElems;
  if (ddaLL128ArScratchSize(comm->nRanks, numLines) > comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

uint32_t ncclAllReduceDdaFabricLL128Blocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const auto grid = ddaAllReduceFabricLL128Geom(comm, count, ncclTypeSize(datatype)).first;
  return grid.x * grid.y;
}

ncclResult_t ncclAllReduceDdaFabricLL128(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaFabricLL128Typed<float>(sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaFabricLL128Typed<half>(sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaFabricLL128Typed<bf16>(sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
