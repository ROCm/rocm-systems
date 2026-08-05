#ifndef NCCL_HIERARCHICAL_SHUFFLE_H
#define NCCL_HIERARCHICAL_SHUFFLE_H

#include <hip/hip_runtime.h>
#include "nccl_device/rccl_ptr.h"

static constexpr int HIERARCHICAL_SHUFFLE_THREADS = 256;

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
 * Uses 16B (v4u) non-temporal vectorized copies when aligned, with 4B and 1B
 * fallbacks for valid unaligned buffers; work is block-strided.
 */
static __global__ __launch_bounds__(HIERARCHICAL_SHUFFLE_THREADS) void hierarchicalShuffle(
  const char* __restrict__ src, char* __restrict__ dst, size_t rankOffset, int cols, int rows) {
  int totalPairs = rows * cols;
  constexpr size_t VecAlign = alignof(v4u);
  constexpr size_t WordAlign = alignof(uint32_t);

  for (int pair = blockIdx.x; pair < totalPairs; pair += gridDim.x) {
    int i = pair / cols;
    int j = pair - i * cols;
    int dstIdx = j * rows + i;

    const char* srcTile = src + (size_t)pair * rankOffset;
    char* dstTile = dst + (size_t)dstIdx * rankOffset;
    size_t stride = blockDim.x;
    size_t copied = 0;

    // rankOffset can change alignment from one tile to the next, so select the
    // widest legal copy size independently for each source/destination tile pair.
    const bool useVec =
      reinterpret_cast<uintptr_t>(srcTile) % VecAlign == 0 && reinterpret_cast<uintptr_t>(dstTile) % VecAlign == 0;
    const bool useWords =
      reinterpret_cast<uintptr_t>(srcTile) % WordAlign == 0 && reinterpret_cast<uintptr_t>(dstTile) % WordAlign == 0;

    if (useVec) {
      v4u_gptr srcV = (v4u_gptr)srcTile;
      v4u_gptr dstV = (v4u_gptr)dstTile;
      size_t numVec = rankOffset / sizeof(v4u);
      size_t k = threadIdx.x;
      for (; k + 3 * stride < numVec; k += 4 * stride) {
        v4u a = __builtin_nontemporal_load(srcV + k + 0 * stride);
        v4u b = __builtin_nontemporal_load(srcV + k + 1 * stride);
        v4u c = __builtin_nontemporal_load(srcV + k + 2 * stride);
        v4u d = __builtin_nontemporal_load(srcV + k + 3 * stride);
        __builtin_nontemporal_store(a, dstV + k + 0 * stride);
        __builtin_nontemporal_store(b, dstV + k + 1 * stride);
        __builtin_nontemporal_store(c, dstV + k + 2 * stride);
        __builtin_nontemporal_store(d, dstV + k + 3 * stride);
      }
      for (; k < numVec; k += stride) {
        __builtin_nontemporal_store(__builtin_nontemporal_load(srcV + k), dstV + k);
      }
      copied = numVec * sizeof(v4u);
    }

    if (useWords) {
      u32_gptr srcW = (u32_gptr)(srcTile + copied);
      u32_gptr dstW = (u32_gptr)(dstTile + copied);
      size_t numWords = (rankOffset - copied) / sizeof(uint32_t);
      for (size_t w = threadIdx.x; w < numWords; w += stride) {
        __builtin_nontemporal_store(__builtin_nontemporal_load(srcW + w), dstW + w);
      }
      copied += numWords * sizeof(uint32_t);
    }

    u8_gptr srcB = (u8_gptr)(srcTile + copied);
    u8_gptr dstB = (u8_gptr)(dstTile + copied);
    size_t numBytes = rankOffset - copied;
    for (size_t b = threadIdx.x; b < numBytes; b += stride) {
      __builtin_nontemporal_store(__builtin_nontemporal_load(srcB + b), dstB + b);
    }
  }
}

#endif
