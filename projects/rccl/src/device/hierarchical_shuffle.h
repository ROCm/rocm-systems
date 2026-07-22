#ifndef NCCL_HIERARCHICAL_SHUFFLE_H
#define NCCL_HIERARCHICAL_SHUFFLE_H

#include <hip/hip_runtime.h>
#include "nccl_device/rccl_ptr.h"

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
 * Uses 16B (v4u) non-temporal vectorized copies; work is block-strided.
 */
static __global__ __launch_bounds__(256) void hierarchicalShuffle(const char* __restrict__ src, char* __restrict__ dst,
                                                                  size_t rankOffset, int cols, int rows) {
  int totalPairs = rows * cols;
  size_t numVec = rankOffset / sizeof(v4u);

  for (int pair = blockIdx.x; pair < totalPairs; pair += gridDim.x) {
    int i = pair / cols;
    int j = pair - i * cols;
    int dstIdx = j * rows + i;

    v4u_gptr srcV = (v4u_gptr)(src + (size_t)pair * rankOffset);
    v4u_gptr dstV = (v4u_gptr)(dst + (size_t)dstIdx * rankOffset);

    size_t stride = blockDim.x;
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

    size_t copied = numVec * sizeof(v4u);
    if (copied < rankOffset && threadIdx.x == 0) {
      for (size_t b = copied; b < rankOffset; b++) {
        dst[(size_t)dstIdx * rankOffset + b] = src[(size_t)pair * rankOffset + b];
      }
    }
  }
}

#endif
