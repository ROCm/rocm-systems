#ifndef NCCL_HIERARCHICAL_AG_SHUFFLE_H
#define NCCL_HIERARCHICAL_AG_SHUFFLE_H

#include <hip/hip_runtime.h>

/* 
* Shuffle kernel for hierarchical allgather.
* After inter-node AG and intra-node AG, the data in the temp buffer:
*   src: [LR0:{N0,N1, ..., Nn}, LR1:{N0,N1, ..., Nn}, ..., LRk:{N0,N1, ..., Nn}]
* this kernel shuffles the data to the following layout:
*   dst: [N0:{LR0, LR1, ..., LRk}, N1:{LR0, LR1, ..., LRk}, ..., Nn:{LR0, LR1, ..., LRk}]
*
* Uses int4 vectorized copies
* Work is distributed across blocks via block-strided loop.
*/
static __global__ __launch_bounds__(1024)
void hierarchicalAGShuffle(const char* __restrict__ src, char* __restrict__ dst,
    size_t rankOffset, int nNodes, int localRanks) {
  int totalPairs = nNodes * localRanks;
  size_t numInt4 = rankOffset / sizeof(int4);

  for (int pair = blockIdx.x; pair < totalPairs; pair += gridDim.x) {
    int srcLr = pair / nNodes;
    int srcN = pair % nNodes;
    int dstIdx = srcN * localRanks + srcLr;

    const int4* src4 = reinterpret_cast<const int4*>(src + (size_t)pair * rankOffset);
    int4* dst4 = reinterpret_cast<int4*>(dst + (size_t)dstIdx * rankOffset);

    for (size_t i = threadIdx.x; i < numInt4; i += blockDim.x) {
      dst4[i] = src4[i];
    }

    size_t copied = numInt4 * sizeof(int4);
    if (threadIdx.x == 0) {
      for (size_t i = copied; i < rankOffset; i++) {
        dst[(size_t)dstIdx * rankOffset + i] = src[(size_t)pair * rankOffset + i];
      }
    }
  }
}

#endif
