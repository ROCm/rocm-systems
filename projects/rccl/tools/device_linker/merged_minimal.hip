/**
 * Minimal merged dispatcher for device linker prototype.
 * 
 * This defines just enough types to match RCCL's kernel signature without
 * pulling in the full header chain.
 */

#include <hip/hip_runtime.h>
#include <cstdint>

// ============================================================================
// Type definitions matching RCCL's layout
// ============================================================================

#define NCCL_MAX_NTHREADS 512
#define MAXCHANNELS 32
#define NCCL_MAX_GROUPS 4
#define NCCL_MAX_WORK_ELEMENTS 1

#ifndef NCCL_GRID_CONSTANT
#define NCCL_GRID_CONSTANT
#endif

// Match ncclDevKernelArgsStorage layout
template<int MaxChannels>
struct ncclDevKernelArgsStorage {
    char data[4096];
};

using ncclDevKernelArgsDefaultStorage = ncclDevKernelArgsStorage<MAXCHANNELS>;

// Minimal ncclShmemData - we only need funcId and basic control fields
struct MinimalShmem {
    uint64_t ll128warp[NCCL_MAX_NTHREADS/64][64];  // 512 threads, 64 warp size
    int aborted;
    int nextBatchIx;  
    unsigned int funcId;   // This is what we dispatch on
    int channelId;
    // ... rest of fields we don't need
    char padding[16384];   // Make sure we have enough space
};

__shared__ MinimalShmem ncclShmem;

// ============================================================================
// Function tables - populated by device linker
// ============================================================================

#define FUNC_COUNT 859  // Max funcId is 858

typedef void (*DevFuncPtr)();

// Tables for each unroll factor
__device__ DevFuncPtr ncclDevFuncTable_Merged_1[FUNC_COUNT];
__device__ DevFuncPtr ncclDevFuncTable_Merged_2[FUNC_COUNT];
__device__ DevFuncPtr ncclDevFuncTable_Merged_4[FUNC_COUNT];

// ============================================================================
// Merged dispatcher kernels
// These have EXACTLY the same signature as ncclDevKernel_Generic_*
// ============================================================================

__launch_bounds__(NCCL_MAX_NTHREADS, 1)
__global__ void ncclDevKernel_Merged_1(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
    // In production, ncclKernelMain loads work batch to shared memory
    // For this prototype, we assume the host has prepared shared memory
    // The real implementation needs loadWorkBatchToShmem logic
    
    unsigned int funcId = ncclShmem.funcId;
    if (funcId < FUNC_COUNT && ncclDevFuncTable_Merged_1[funcId] != nullptr) {
        ncclDevFuncTable_Merged_1[funcId]();
    }
}

__launch_bounds__(NCCL_MAX_NTHREADS, 1)
__global__ void ncclDevKernel_Merged_2(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
    unsigned int funcId = ncclShmem.funcId;
    if (funcId < FUNC_COUNT && ncclDevFuncTable_Merged_2[funcId] != nullptr) {
        ncclDevFuncTable_Merged_2[funcId]();
    }
}

__launch_bounds__(NCCL_MAX_NTHREADS, 1)
__global__ void ncclDevKernel_Merged_4(ncclDevKernelArgsDefaultStorage NCCL_GRID_CONSTANT const argsStorage) {
    unsigned int funcId = ncclShmem.funcId;
    if (funcId < FUNC_COUNT && ncclDevFuncTable_Merged_4[funcId] != nullptr) {
        ncclDevFuncTable_Merged_4[funcId]();
    }
}

// Host-side accessors
extern "C" void* ncclDevKernel_Merged_1_getPtr() { return (void*)ncclDevKernel_Merged_1; }
extern "C" void* ncclDevKernel_Merged_2_getPtr() { return (void*)ncclDevKernel_Merged_2; }
extern "C" void* ncclDevKernel_Merged_4_getPtr() { return (void*)ncclDevKernel_Merged_4; }
