/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"

__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

struct RunWorkNop {
  __device__ void run() {}
};

__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_1(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/1>(&args4K.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_2(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/2>(&args4K.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_4(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/4>(&args4K.args);
}
#ifdef ENABLE_COLLTRACE
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_1(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/1>(&args4K.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_2(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/2>(&args4K.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_4(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/4>(&args4K.args);
}
#endif

#ifdef USE_INDIRECT_FUNCTION_CALL
__device__ void ncclDevFunc_Nop();
#else
__device__ __attribute__((noinline)) void ncclDevFunc_Nop();
#endif
