/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"

struct RunWorkNop {
  __device__ void run(LDSPtr<ncclShmemData> /*ncclShmem*/, ncclShmemPerWarpPtr /*ncclShmemPerWarp*/) {}
};

#ifdef RCCL_ARGS_IN_SCRATCH
#define STORE_KERNARG_PTR() \
  if (threadIdx.x == 0) { \
    const void* _kptr = (const void*)__builtin_amdgcn_kernarg_segment_ptr(); \
    LDSPtr<ncclShmemData>(&ncclShmem)->kernargPtr = _kptr; \
  } \
  __syncthreads(); \
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\tbuffer_inv sc0 sc1" ::: "memory");
#else
#define STORE_KERNARG_PTR()
#endif

__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  STORE_KERNARG_PTR()
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/1>(&argsStorage.args, LDSPtr<ncclShmemData>(&ncclShmem), ncclShmemPerWarpPtr(ncclShmemPerWarp));
}

__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  STORE_KERNARG_PTR()
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/2>(&argsStorage.args, LDSPtr<ncclShmemData>(&ncclShmem), ncclShmemPerWarpPtr(ncclShmemPerWarp));
}

__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  STORE_KERNARG_PTR()
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/false, /*Unroll*/4>(&argsStorage.args, LDSPtr<ncclShmemData>(&ncclShmem), ncclShmemPerWarpPtr(ncclShmemPerWarp));
}
#ifdef ENABLE_COLLTRACE
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  STORE_KERNARG_PTR()
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/1>(&argsStorage.args, LDSPtr<ncclShmemData>(&ncclShmem), ncclShmemPerWarpPtr(ncclShmemPerWarp));
}

__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  STORE_KERNARG_PTR()
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/2>(&argsStorage.args, LDSPtr<ncclShmemData>(&ncclShmem), ncclShmemPerWarpPtr(ncclShmemPerWarp));
}

__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  STORE_KERNARG_PTR()
  ncclKernelMain<-1, RunWorkNop, /*COLLTRACE*/true, /*Unroll*/4>(&argsStorage.args, LDSPtr<ncclShmemData>(&ncclShmem), ncclShmemPerWarpPtr(ncclShmemPerWarp));
}
#endif

#ifdef USE_INDIRECT_FUNCTION_CALL
__device__ void ncclDevFunc_Nop(LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp);
#else
__device__ __attribute__((noinline)) void ncclDevFunc_Nop(LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp);
#endif
