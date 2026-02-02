/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"

// ncclShmem is defined in common.h via NCCL_SHMEM_DECL when DEVICE_LINKER is set
// Only define here for non-DEVICE_LINKER builds (RDC mode with extern __shared__)
#ifndef DEVICE_LINKER
__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif
#endif

#ifdef DEVICE_LINKER
// Device linker mode: define function tables here (device linker overwrites with actual pointers)
// const at namespace scope gives internal linkage like IFC
// Tables in .data.rel.ro - device linker creates relocations
#define FUNC_COUNT 859
typedef void(*ncclDevFuncPtr_t)();
__device__ ncclDevFuncPtr_t const ncclDevFuncTable_1[FUNC_COUNT] = {};
__device__ ncclDevFuncPtr_t const ncclDevFuncTable_2[FUNC_COUNT] = {};
__device__ ncclDevFuncPtr_t const ncclDevFuncTable_4[FUNC_COUNT] = {};
#endif

struct RunWorkNop {
  __device__ void run() {}
};

// DEVICE_LINKER mode: No scratch reservation code - device linker patches .note metadata

__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/1>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/2>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/4>(&argsStorage.args);
}
#ifdef ENABLE_COLLTRACE
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/1>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/2>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/4>(&argsStorage.args);
}
#endif

#ifdef USE_INDIRECT_FUNCTION_CALL
__device__ void ncclDevFunc_Nop();
#else
__device__ __attribute__((noinline)) void ncclDevFunc_Nop();
#endif
