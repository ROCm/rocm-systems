#ifndef NCCL_HIERARCHICAL_SHUFFLE_H
#define NCCL_HIERARCHICAL_SHUFFLE_H

#include <cstdint>
#include <hip/hip_runtime.h>

static constexpr int HIERARCHICAL_SHUFFLE_THREADS = 1024;

/*
 * Shuffle (transpose) kernel for hierarchical AllGather and ReduceScatter.
 * Transposes between local-rank-major and node-major tile layouts.
 *
 * AllGather (after inter and intra AG), the data in the temp buffer:
 *   src: [LR0:{N0..Nn}, LR1:{N0..Nn}, ..., LRk:{N0..Nn}]   (local-rank-major)
 * this kernel shuffles the data to the following layout:
 *   dst: [N0:{LR0..LRk}, N1:{LR0..LRk}, ..., Nn:{LR0..LRk}] (node-major)
 *
 * ReduceScatter (before intra and inter RS): the reverse of AG (node-major -> local-rank-major)
 *   src: [N0:{LR0..LRk}, N1:{LR0..LRk}, ..., Nn:{LR0..LRk}] (node-major)
 * this kernel shuffles the data to the following layout:
 *   dst: [LR0:{N0..Nn}, LR1:{N0..Nn}, ..., LRk:{N0..Nn}]   (local-rank-major)
 *
 * Each tile is `rankOffset` bytes. Call with (cols, rows):
 *   AllGather:     hierarchicalShuffle(..., nNodes, localRanks)
 *   ReduceScatter: hierarchicalShuffle(..., localRanks, nNodes)
 *
 * Uses 16-byte vectorized copies when aligned, with 4-byte and byte fallbacks;
 * work is block-strided.
 */
static __global__ __launch_bounds__(HIERARCHICAL_SHUFFLE_THREADS) void hierarchicalShuffle(
  const char* __restrict__ src, char* __restrict__ dst, size_t rankOffset, int cols, int rows) {
  int totalPairs = rows * cols;
  constexpr size_t VecBytes = sizeof(int4);
  constexpr size_t WordBytes = sizeof(uint32_t);

  for (int pair = blockIdx.x; pair < totalPairs; pair += gridDim.x) {
    int i = pair / cols;
    int j = pair % cols;
    int dstIdx = j * rows + i;

    const char* srcTile = src + (size_t)pair * rankOffset;
    char* dstTile = dst + (size_t)dstIdx * rankOffset;
    size_t copied = 0;

    if (reinterpret_cast<uintptr_t>(srcTile) % VecBytes == 0 && reinterpret_cast<uintptr_t>(dstTile) % VecBytes == 0) {
      const int4* src4 = reinterpret_cast<const int4*>(srcTile);
      int4* dst4 = reinterpret_cast<int4*>(dstTile);
      size_t numInt4 = rankOffset / VecBytes;
      for (size_t k = threadIdx.x; k < numInt4; k += blockDim.x) {
        dst4[k] = src4[k];
      }
      copied = numInt4 * VecBytes;
    }

    if (reinterpret_cast<uintptr_t>(srcTile + copied) % WordBytes == 0 &&
        reinterpret_cast<uintptr_t>(dstTile + copied) % WordBytes == 0) {
      const uint32_t* srcWords = reinterpret_cast<const uint32_t*>(srcTile + copied);
      uint32_t* dstWords = reinterpret_cast<uint32_t*>(dstTile + copied);
      size_t numWords = (rankOffset - copied) / WordBytes;
      for (size_t w = threadIdx.x; w < numWords; w += blockDim.x) {
        dstWords[w] = srcWords[w];
      }
      copied += numWords * WordBytes;
    }

    for (size_t b = copied + threadIdx.x; b < rankOffset; b += blockDim.x) {
      dstTile[b] = srcTile[b];
    }
  }
}

#endif
