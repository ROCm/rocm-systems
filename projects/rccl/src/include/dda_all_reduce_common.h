/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "collectives.h"
#include "comm.h"
#include "nccl.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace nccl_dda_detail {

// Flat all-reduce kernel below this size; tree kernel at or above it.
constexpr size_t kDdaFlatTreeThresholdBytes = 1ULL << 18;

inline uint32_t divRoundUp(size_t a, size_t b) {
  uint32_t y = static_cast<uint32_t>((a + b - 1) / b);
  if (y == 0) {
    y = 1;
  }
  return y;
}

constexpr uint32_t
calcBlockCount(size_t numThreads, size_t threadsPerBlock, size_t maxBlocks) {
  const auto uNumThreads = static_cast<uint64_t>(numThreads);
  const auto uThreadsPerBlock = static_cast<uint64_t>(threadsPerBlock);
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks =
      uNumThreads / uThreadsPerBlock + (uNumThreads % uThreadsPerBlock != 0);
  uint32_t y = static_cast<uint32_t>(std::min(blocks, maxBlocks));
  if (y == 0) {
    y = 1;
  }
  return y;
}

inline std::pair<dim3, dim3>
getGridAndBlockDims(size_t count, int typeSize, size_t maxBlocks) {
  constexpr uint32_t kThreadsPerWarp = 64;
  constexpr uint32_t kThreadsPerBlock = 512;

  const uint32_t elementsPerThread = 16 / typeSize; // we do 16 Byte load in kernel
  const uint32_t elementsPerWarp = elementsPerThread * kThreadsPerWarp;

  dim3 threads(0, 1, 1);
  dim3 blocks(0, 1, 1);
  if (count < elementsPerThread * kThreadsPerBlock) {
    threads.x = divRoundUp(count, elementsPerWarp) * kThreadsPerWarp;
    blocks.x = 1;
  } else {
    auto warpsRequired = divRoundUp(count, elementsPerWarp);
    blocks.x = calcBlockCount(
        divRoundUp(count, elementsPerThread), kThreadsPerBlock, maxBlocks);
    auto warpsPerBlock = divRoundUp(warpsRequired, blocks.x);
    auto threadsPerBlock =
        std::min<uint32_t>(kThreadsPerBlock, warpsPerBlock * kThreadsPerWarp);
    threads.x = threadsPerBlock;
  }

  return std::make_pair(blocks, threads);
}

// Eligibility checks common to both DDA all-reduce paths
inline bool ddaAllReduceCommonEligible(
    ncclComm* comm,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op) {
  if (comm == nullptr || comm->bootstrap == nullptr) {
    return false;
  }
  if (comm->ddaScratch == nullptr || comm->ddaPeerPtrsDev == nullptr) {
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
  if (bytes > comm->ddaScratchBytes) {
    return false;
  }
  if (bytes % 16) {
    // 16-byte alignment: the DDA kernels do 16-byte vectorized loads.
    return false;
  }
  if (bytes > kDdaFlatTreeThresholdBytes) {
    if (count % comm->nRanks ||
        ((count / comm->nRanks) * ncclTypeSize(datatype)) % 16) {
      // Two-shot/tree path: each rank reduces count/nRanks elements, so that
      // per-rank slice must also be 16-byte aligned.
      return false;
    }
  }
  return true;
}

} // namespace nccl_dda_detail
