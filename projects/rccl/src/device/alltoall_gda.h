/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

#ifdef ENABLE_ROCSHMEM
#include <rocshmem/rocshmem.hpp>

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllToAllGda, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nThreads, struct ncclDevWorkColl* work) {
    if (blockIdx.x == 0) {
        int num_pes = rocshmem::rocshmem_n_pes();

        reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&work->sendbuff, 1, (void **)&work->sndbuff, 
            (work->size*num_pes));

        rocshmem::rocshmem_char_alltoall_wg(work->team, ((char*)work->tempbuff), ((char*)work->sndbuff), work->size);

        reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            tid, nThreads, 0, nullptr, false, 1, (void **)&work->tempbuff, 1, (void **)&work->recvbuff, 
            (work->size*num_pes));
        }
  }
};
#endif

