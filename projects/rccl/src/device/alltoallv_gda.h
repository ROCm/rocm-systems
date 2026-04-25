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
struct RunWorkColl<ncclFuncAlltoAllvGda, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nThreads, struct ncclDevWorkColl* work) {
        int num_pes = rocshmem::rocshmem_n_pes();

	if (blockIdx.x == 0) {
        if (work->size <= 131072) {
           void *src = (T*)work->sendbuff;
           void *dst = (T*)work->sndbuff;
           reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            	tid, nThreads, 0, nullptr, false, 1, (void **)&src, 1, (void **)&dst,
            	work->size);
        }		   
	   work->sendSizes = (size_t*)work->sizes;
	   work->sendDispls = (size_t*)work->sizes + num_pes;
       work->recvSizes = (size_t*)work->sizes + 2 * num_pes;
       work->recvDispls = (size_t*)work->sizes + 3 * num_pes;

       rocshmem::rocshmem_char_alltoallv_wg(work->team, (char*)work->tempbuff, work->recvSizes, work->recvDispls, 
			(char*)work->sndbuff, work->sendSizes, work->sendDispls);

	   if (work->size <= 131072) {
           void *srcR = (T*)work->tempbuff;
           void *dstR = (T*)work->recvbuff;
           reduceCopy<COLL_UNROLL, USE_ACC, RedOp, T, 0,1, 1, 0, 1, 1, 0>(
            	tid, nThreads, 0, nullptr, false, 1, (void **)&srcR, 1, (void **)&dstR,
            	work->size);
       }

	}
 }
};
#endif

