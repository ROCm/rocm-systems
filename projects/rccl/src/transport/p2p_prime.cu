/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Owner-side cache-line priming of legacy IPC P2P buffers (see
 * p2pPrimeLegacyIpcEnabled and p2pAllocateShareableBuffer in p2p.cc).
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include <stdint.h>
#include <algorithm>
#include "checks.h"
#include "bitops.h"
#include "p2p.h"

// One load per 64 B covers every cache line on all supported architectures
// (the gfx9 L2 line is 128 B).
static constexpr size_t kP2pPrimeStrideBytes = 64;
static constexpr int kP2pPrimeThreads = 256;
static constexpr size_t kP2pPrimeMaxBlocks = 256;

__global__ void p2pPrimeBufferKernel(const uint32_t* buff, size_t nLines, size_t strideWords, size_t lastWord) {
  uint32_t acc = 0;
  for (size_t line = blockIdx.x * (size_t)blockDim.x + threadIdx.x; line < nLines;
       line += (size_t)blockDim.x * gridDim.x) {
    // Plain (cacheable) load: non-temporal loads and DMA reads do not have the effect we need.
    // A partial last line is read through its last whole word inside the buffer.
    size_t word = line * strideWords;
    acc ^= buff[word < lastWord ? word : lastWord];
  }
  // Keep the loads without storing anything.
  asm volatile("" : : "v"(acc));
}

ncclResult_t ncclP2pPrimeBuffer(const void* buff, size_t size, cudaStream_t stream) {
  if (buff == nullptr || size < sizeof(uint32_t)) return ncclSuccess;
  size_t nLines = DIVUP(size, kP2pPrimeStrideBytes);
  int blocks = (int)std::min(kP2pPrimeMaxBlocks, (size_t)DIVUP(nLines, (size_t)kP2pPrimeThreads));
  p2pPrimeBufferKernel<<<blocks, kP2pPrimeThreads, 0, stream>>>((const uint32_t*)buff, nLines,
                                                                kP2pPrimeStrideBytes / sizeof(uint32_t),
                                                                size / sizeof(uint32_t) - 1);
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}
