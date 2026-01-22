/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatterDirect, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nThreads, struct ncclDevWorkColl* work) {

    size_t msgSize = work->count * sizeof(T) * ncclShmem.comm.nRanks;
    if (work->enableDirectReduceScatter && msgSize <= (size_t)work->directReduceScatterLimitBytes) {
      const int nRanks = ncclShmem.comm.nRanks; 
      const ssize_t numElements = work->count;

      // Calculate Offset to utilize multiple channels
      ssize_t elementsPerBlock = numElements / gridDim.x;
      ssize_t remainderElements = numElements % gridDim.x;
      // Calculate the number of elements per block for each block
      // The first n blocks get 1 extra element to account for the remainder (n = remainderElements)
      ssize_t numElementsPerBlock = elementsPerBlock + (blockIdx.x < remainderElements ? 1 : 0);
      ssize_t channelOffset = blockIdx.x * elementsPerBlock + min((ssize_t)blockIdx.x, remainderElements);

      // Array of src pointers pointing to rank offsets in tempBuff
      void** srcPtrs = (void**)ncclScratchForWarp(0); 
      if (tid == 0) {
        for (int i = 0; i < nRanks; i++) {
          // Define offset into tempbuff for each rank's data
          const ssize_t srcOffset = i * numElements + channelOffset;
          srcPtrs[i] = (void*)((T*)work->tempBuff + srcOffset);
        }
      }
      // Sync threads to ensure all srcPtrs are set before reduction
      __syncthreads();

      T* recvbuff = (T*)work->recvbuff;
      // Array for destination pointer to recvbuff
      void* dstPtrs[1];
      dstPtrs[0] = (void*)(recvbuff + channelOffset);
      if (tid < nThreads) {
        // Call reduction across all rank offsets in tempbuff and store in recvbuff
        reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0, 1, 64, 0, 1, 1, 0>
          (tid, nThreads, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, false, nRanks, srcPtrs, 1, dstPtrs, numElementsPerBlock);
      }
    }
  }
};
