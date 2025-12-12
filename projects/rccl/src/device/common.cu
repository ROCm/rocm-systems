/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"

#ifndef RCCL_DEVICE_LINKER
__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif
#endif

struct RunWorkNop {
  __device__ void run() {}
};

__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*Unroll*/1>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*Unroll*/2>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*Unroll*/4>(&argsStorage.args);
}

#if defined(USE_INDIRECT_FUNCTION_CALL) || defined(RCCL_DEVICE_LINKER)
__device__ void ncclDevFunc_Nop();
#else
__device__ __attribute__((noinline)) void ncclDevFunc_Nop();
#endif

// [RCCL] Body for the no-op device func. RCCL's common.cu declares ncclDevFunc_Nop
// above (mode-aware attributes); generate.py excludes "Nop" from the generated
// per-impl/specialized files, so the definition must live here (as in upstream and
// the v2.29.7-1 base). The generated ncclDevFuncTable references this symbol.
__device__ void ncclDevFunc_Nop() {}
