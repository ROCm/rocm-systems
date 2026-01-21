/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collectives.h"
#include "common.h"
#include "device.h"
#include "device_table.h"

__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

struct RunWorkNop {
  __device__ void run() {}
};

// Forward declarations for type-specific function tables and callers
// These are defined in the generated kernels_<type>.cu files
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx950__)
// Declare type-specific function tables
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_i8_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_i8_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_i8_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_u8_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_u8_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_u8_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_i32_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_i32_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_i32_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_u32_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_u32_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_u32_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_i64_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_i64_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_i64_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_u64_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_u64_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_u64_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f16_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f16_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f16_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f32_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f32_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f32_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f64_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f64_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f64_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_bf16_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_bf16_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_bf16_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f8e4m3_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f8e4m3_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f8e4m3_4[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f8e5m2_1[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f8e5m2_2[];
extern __device__ ncclDevFuncPtr_t const ncclDevFuncTable_f8e5m2_4[];
#endif

// Type-specific caller functions are defined inline in device_table.h

#ifdef BUILD_GENERIC_KERNELS
// Generic kernel that dispatches to type-specific device functions
// This avoids the need for a 1000+ entry generic function table
// by dispatching to smaller type-specific tables at runtime

// Helper to call type-specific function tables based on funcId
// The funcId encodes the type information that we decode here
// Unroll-specific dispatch functions - no runtime branching on unroll factor
// Each function only calls the specific unroll variant for each type

__device__ __forceinline__ void callTypeSpecificFunction_1(int funcId)
{
    extern __device__ __constant__ unsigned char ncclDevFuncIdToType[];
    int                                          typeIdx = ncclDevFuncIdToType[funcId];

    switch(typeIdx)
    {
        case 0: NCCL_CALL_FUNCTIONS_i8_1(funcId); break;
        case 1: NCCL_CALL_FUNCTIONS_u8_1(funcId); break;
        case 2: NCCL_CALL_FUNCTIONS_i32_1(funcId); break;
        case 3: NCCL_CALL_FUNCTIONS_u32_1(funcId); break;
        case 4: NCCL_CALL_FUNCTIONS_i64_1(funcId); break;
        case 5: NCCL_CALL_FUNCTIONS_u64_1(funcId); break;
        case 6: NCCL_CALL_FUNCTIONS_f16_1(funcId); break;
        case 7: NCCL_CALL_FUNCTIONS_f32_1(funcId); break;
        case 8: NCCL_CALL_FUNCTIONS_f64_1(funcId); break;
        case 9: NCCL_CALL_FUNCTIONS_bf16_1(funcId); break;
        case 10: NCCL_CALL_FUNCTIONS_f8e4m3_1(funcId); break;
        case 11: NCCL_CALL_FUNCTIONS_f8e5m2_1(funcId); break;
        default: NCCL_CALL_FUNCTIONS_i8_1(funcId); break;
    }
}

__device__ __forceinline__ void callTypeSpecificFunction_2(int funcId)
{
    extern __device__ __constant__ unsigned char ncclDevFuncIdToType[];
    int                                          typeIdx = ncclDevFuncIdToType[funcId];

    switch(typeIdx)
    {
        case 0: NCCL_CALL_FUNCTIONS_i8_2(funcId); break;
        case 1: NCCL_CALL_FUNCTIONS_u8_2(funcId); break;
        case 2: NCCL_CALL_FUNCTIONS_i32_2(funcId); break;
        case 3: NCCL_CALL_FUNCTIONS_u32_2(funcId); break;
        case 4: NCCL_CALL_FUNCTIONS_i64_2(funcId); break;
        case 5: NCCL_CALL_FUNCTIONS_u64_2(funcId); break;
        case 6: NCCL_CALL_FUNCTIONS_f16_2(funcId); break;
        case 7: NCCL_CALL_FUNCTIONS_f32_2(funcId); break;
        case 8: NCCL_CALL_FUNCTIONS_f64_2(funcId); break;
        case 9: NCCL_CALL_FUNCTIONS_bf16_2(funcId); break;
        case 10: NCCL_CALL_FUNCTIONS_f8e4m3_2(funcId); break;
        case 11: NCCL_CALL_FUNCTIONS_f8e5m2_2(funcId); break;
        default: NCCL_CALL_FUNCTIONS_i8_2(funcId); break;
    }
}

__device__ __forceinline__ void callTypeSpecificFunction_4(int funcId)
{
    extern __device__ __constant__ unsigned char ncclDevFuncIdToType[];
    int                                          typeIdx = ncclDevFuncIdToType[funcId];

    switch(typeIdx)
    {
        case 0: NCCL_CALL_FUNCTIONS_i8_4(funcId); break;
        case 1: NCCL_CALL_FUNCTIONS_u8_4(funcId); break;
        case 2: NCCL_CALL_FUNCTIONS_i32_4(funcId); break;
        case 3: NCCL_CALL_FUNCTIONS_u32_4(funcId); break;
        case 4: NCCL_CALL_FUNCTIONS_i64_4(funcId); break;
        case 5: NCCL_CALL_FUNCTIONS_u64_4(funcId); break;
        case 6: NCCL_CALL_FUNCTIONS_f16_4(funcId); break;
        case 7: NCCL_CALL_FUNCTIONS_f32_4(funcId); break;
        case 8: NCCL_CALL_FUNCTIONS_f64_4(funcId); break;
        case 9: NCCL_CALL_FUNCTIONS_bf16_4(funcId); break;
        case 10: NCCL_CALL_FUNCTIONS_f8e4m3_4(funcId); break;
        case 11: NCCL_CALL_FUNCTIONS_f8e5m2_4(funcId); break;
        default: NCCL_CALL_FUNCTIONS_i8_4(funcId); break;
    }
}

// Generic dispatchers - one per unroll factor for compile-time optimization
// Each dispatcher calls its specific unroll-specialized function with zero runtime branching

struct GenericDispatcher_1
{
    static __device__ __forceinline__ void dispatch(int funcId, int unroll)
    {
        // callTypeSpecificFunction handles both direct and indirect paths internally via NCCL_CALL_FUNCTIONS
        callTypeSpecificFunction_1(funcId);
    }
};

struct GenericDispatcher_2
{
    static __device__ __forceinline__ void dispatch(int funcId, int unroll)
    {
        // callTypeSpecificFunction handles both direct and indirect paths internally via NCCL_CALL_FUNCTIONS
        callTypeSpecificFunction_2(funcId);
    }
};

struct GenericDispatcher_4
{
    static __device__ __forceinline__ void dispatch(int funcId, int unroll)
    {
        // callTypeSpecificFunction handles both direct and indirect paths internally via NCCL_CALL_FUNCTIONS
        callTypeSpecificFunction_4(funcId);
    }
};

// Generic kernels - dispatch to type-specific tables at runtime
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_1, /*COLLTRACE*/ false, /*Unroll*/ 1>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_2, /*COLLTRACE*/ false, /*Unroll*/ 2>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernel_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_4, /*COLLTRACE*/ false, /*Unroll*/ 4>(&argsStorage.args);
}

#ifdef ENABLE_COLLTRACE
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_1, /*COLLTRACE*/ true, /*Unroll*/ 1>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_2, /*COLLTRACE*/ true, /*Unroll*/ 2>(&argsStorage.args);
}
__launch_bounds__(NCCL_MAX_NTHREADS, 1) __global__ void ncclDevKernelDebug_Generic_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
  ncclKernelMain<-1, RunWorkNop, GenericDispatcher_4, /*COLLTRACE*/ true, /*Unroll*/ 4>(&argsStorage.args);
}
#endif // ENABLE_COLLTRACE
#endif // BUILD_GENERIC_KERNELS

__device__ void ncclDevFunc_Nop();
