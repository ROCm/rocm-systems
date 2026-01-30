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
#define FUNC_COUNT 859
typedef void(*ncclDevFuncPtr_t)();
__device__ ncclDevFuncPtr_t ncclDevFuncTable_1[FUNC_COUNT];
__device__ ncclDevFuncPtr_t ncclDevFuncTable_2[FUNC_COUNT];
__device__ ncclDevFuncPtr_t ncclDevFuncTable_4[FUNC_COUNT];
#endif

struct RunWorkNop {
  __device__ void run() {}
};

#ifdef DEVICE_LINKER
// Force compiler to allocate scratch memory for the dispatcher kernel.
// When compiling the dispatcher alone, the compiler doesn't see the specialized
// functions and would set PRIVATE_SEGMENT_FIXED_SIZE=0. But specialized functions
// need scratch (up to ~1092 bytes). By using scratch here, we force the compiler
// to generate proper scratch initialization code in the dispatcher.
__device__ volatile int ncclDeviceScratchTrigger = 0;
__device__ volatile int ncclDeviceScratchSink;

// Use recursion to force a deep call stack that compiler can't optimize away.
// Each frame needs ~8 bytes (saved registers), so 160 recursive calls gives ~1280 bytes.
__device__ __noinline__ int ncclDeviceScratchRecurse(int depth, int val) {
  if (depth <= 0) return val;
  volatile int local = val + depth;  // Force stack frame
  return ncclDeviceScratchRecurse(depth - 1, local);
}

__device__ __noinline__ void ncclDeviceScratchReserve() {
  // Recurse deeply to allocate stack frames totaling >1092 bytes
  ncclDeviceScratchSink = ncclDeviceScratchRecurse(160, threadIdx.x);
}
#endif

__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
#ifdef DEVICE_LINKER
  // Force scratch allocation by referencing the scratch reserve function.
  // Use a device variable the compiler can't prove is zero.
  if (ncclDeviceScratchTrigger) ncclDeviceScratchReserve();
#endif
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/1>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
#ifdef DEVICE_LINKER
  if (ncclDeviceScratchTrigger) ncclDeviceScratchReserve();
#endif
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/2>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
#ifdef DEVICE_LINKER
  if (ncclDeviceScratchTrigger) ncclDeviceScratchReserve();
#endif
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/4>(&argsStorage.args);
}
#ifdef ENABLE_COLLTRACE
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
#ifdef DEVICE_LINKER
  if (ncclDeviceScratchTrigger) ncclDeviceScratchReserve();
#endif
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/1>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
#ifdef DEVICE_LINKER
  if (ncclDeviceScratchTrigger) ncclDeviceScratchReserve();
#endif
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/2>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
#ifdef DEVICE_LINKER
  if (ncclDeviceScratchTrigger) ncclDeviceScratchReserve();
#endif
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/4>(&argsStorage.args);
}
#endif

#ifdef USE_INDIRECT_FUNCTION_CALL
__device__ void ncclDevFunc_Nop();
#else
__device__ __attribute__((noinline)) void ncclDevFunc_Nop();
#endif
