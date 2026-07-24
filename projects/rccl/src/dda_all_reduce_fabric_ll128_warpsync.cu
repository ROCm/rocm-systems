/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host launcher + eligibility for the LL128 warpsync (full-line + barrier) DDA
 * fabric all-reduce. Larger-message analogue of dda_all_reduce_fabric_ll128.cu:
 * it drives the ddaAllReduceFlatLL128_warpsync kernel, which carries payload in
 * all 16 words of each 128B line (no in-line flag word) and synchronizes its two
 * phases with a per-warp arrival barrier in the reserved front region of scratch
 * (see all_reduce_dda_fabric_ll128_warpsync.h).
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "dda_all_reduce.h"

#include "algorithms/all_reduce/all_reduce_dda_fabric_ll128.h" // kDdaLL128ArMaxBytes
#include "algorithms/all_reduce/all_reduce_dda_fabric_ll128_warpsync.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "fabric_gpu_barrier.h" // meta::comms::kDdaMaxNranks
#include "param.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Block size knob shared with the flagged LL128 AllReduce launcher (defined in
// dda_all_reduce_fabric_ll128.cu). Env: RCCL_DDA_LL128_AR_THREADS.
RCCL_PARAM_DECLARE(DdaLL128ArThreads);

namespace {

using meta::comms::kDdaLL128ArMaxBytes;
using meta::comms::kDdaLL128ArReserveBytes;
using meta::comms::kDdaLL128LineElems;
using meta::comms::kDdaLL128Lanes;
using meta::comms::LLLine128;

// Payload words per 128B line for the warpsync kernel: all kDdaLL128LineElems
// (16) words carry payload (no in-line flag word; sync uses the reserved front
// barrier region).
static inline int ddaLL128ArWarpsyncElemsPerLine() {
  return kDdaLL128LineElems;
}

// Per-call slot stride in 128B lines for a message of `numLines` lines. The
// stride matches the message exactly (compact layout) for good L2/TLB locality.
static inline size_t ddaLL128ArWarpsyncSlotLines(size_t numLines) {
  return numLines;
}

// LL128 warpsync scratch for this call: the reserved front region (barrier flag
// exchange) plus 2 banks * nRanks slots * slotLines * 128B for the payload.
static inline size_t ddaLL128ArWarpsyncScratchSize(int nRanks, size_t numLines) {
  return kDdaLL128ArReserveBytes +
         (size_t)2 * (size_t)nRanks * ddaLL128ArWarpsyncSlotLines(numLines) *
             sizeof(LLLine128);
}

// Validated block size from the runtime flag; falls back to `dflt` if the
// configured value is not a multiple of 16 in [16, 1024].
static inline unsigned ddaLL128ArWarpsyncThreads(unsigned dflt) {
  const int64_t v = rcclParamDdaLL128ArThreads();
  if (v >= 16 && v <= 1024 && (v % 16) == 0) {
    return (unsigned)v;
  }
  return dflt;
}

template <typename T>
static ncclResult_t ncclAllReduceDdaFabricLL128WarpsyncTyped(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclComm* comm,
    cudaStream_t stream) {
  const int nRanks = comm->nRanks;
  const size_t bytes = count * sizeof(T);
  const size_t nWords = bytes >> 3;
  const size_t lineElems = (size_t)ddaLL128ArWarpsyncElemsPerLine();
  const size_t numLines = (nWords + lineElems - 1) / lineElems;
  const size_t slotStrideLines = ddaLL128ArWarpsyncSlotLines(numLines);

  // 1D grid over line-groups; each block has threads/16 groups.
  const unsigned threads = ddaLL128ArWarpsyncThreads(1024); // multiple of 16
  const size_t groups = threads / (unsigned)kDdaLL128Lanes;
  int nBlocksMax = comm->ddaFabricMaxBlocks;
  if (nBlocksMax < 1) {
    nBlocksMax = 1;
  }
  unsigned blocks = (unsigned)std::min<size_t>(
      (numLines + groups - 1) / groups, (size_t)nBlocksMax);
  if (blocks == 0) {
    blocks = 1;
  }
  // flatBlockId (blockIdx.x) must stay within the device epoch array.
  if ((int)blocks > comm->ddaLLEpochLen) {
    blocks = (unsigned)comm->ddaLLEpochLen;
    if (blocks == 0) blocks = 1;
  }
  dim3 block(threads);
  dim3 grid(blocks);

  T** peers = reinterpret_cast<T**>(comm->ddaPeerPtrsDev);
  uint32_t* epochDev = comm->ddaLLEpochDev;
  const int epochLen = comm->ddaLLEpochLen;

  INFO(
      NCCL_COLL,
      "DDA fabric AllReduce LL128 warpsync: nRanks=%d bytes=%zu numLines=%zu grid=%u block=%u",
      nRanks, bytes, numLines, grid.x, block.x);

  // NRANKS_CT 4/8: unrolled reduce loop; 0: runtime fallback.
  switch (nRanks) {
  case 4:
    meta::comms::ddaAllReduceFlatLL128_warpsync<T, 4><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen, slotStrideLines);
    break;
  case 8:
    meta::comms::ddaAllReduceFlatLL128_warpsync<T, 8><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen, slotStrideLines);
    break;
  default:
    meta::comms::ddaAllReduceFlatLL128_warpsync<T, 0><<<grid, block, 0, stream>>>(
        peers, static_cast<T*>(recvbuff), static_cast<const T*>(sendbuff),
        count, comm->rank, nRanks, epochDev, epochLen, slotStrideLines);
    break;
  }

  CUDACHECK(cudaGetLastError());

  return ncclSuccess;
}

} // namespace

bool ncclAllReduceDdaFabricLL128WarpsyncEligible(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
  (void)sendbuff;
  (void)recvbuff;
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaFabricMemHandler == nullptr || comm->ddaScratch == nullptr ||
      comm->ddaPeerPtrsDev == nullptr) {
    return false;
  }
  if (comm->nRanks < 2 || comm->nRanks > meta::comms::kDdaMaxNranks) {
    return false;
  }
  if (count == 0) {
    return false;
  }
  if (op != ncclSum) {
    return false;
  }
  if (datatype != ncclFloat32 && datatype != ncclFloat16 &&
      datatype != ncclBfloat16) {
    return false;
  }

  const size_t bytes = count * ncclTypeSize(datatype);
  // Payload is staged as 8-byte words, so it must be a whole number of words.
  if (bytes % 8 != 0) {
    return false;
  }
  if (bytes > kDdaLL128ArMaxBytes) {
    return false;
  }
  // Scratch is sized from the actual message (compact per-call slot stride), so
  // eligibility is bounded by the runtime scratch capacity for this size.
  const size_t nWords = bytes >> 3;
  const size_t lineElems = (size_t)ddaLL128ArWarpsyncElemsPerLine();
  const size_t numLines = (nWords + lineElems - 1) / lineElems;
  if (ddaLL128ArWarpsyncScratchSize(comm->nRanks, numLines) >
      comm->ddaScratchBytes) {
    return false;
  }

  return true;
}

ncclResult_t ncclAllReduceDdaFabricLL128Warpsync(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  (void)op;
  switch (datatype) {
  case ncclFloat32:
    return ncclAllReduceDdaFabricLL128WarpsyncTyped<float>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclFloat16:
    return ncclAllReduceDdaFabricLL128WarpsyncTyped<half>(
        sendbuff, recvbuff, count, comm, stream);
  case ncclBfloat16:
    return ncclAllReduceDdaFabricLL128WarpsyncTyped<bf16>(
        sendbuff, recvbuff, count, comm, stream);
  default:
    return ncclInvalidArgument;
  }
}
