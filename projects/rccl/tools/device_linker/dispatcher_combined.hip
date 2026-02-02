/*************************************************************************
 * Combined dispatcher source for device linker mode.
 * 
 * This file includes both:
 *   - common.cu (generic dispatcher kernels + function tables)
 *   - onerank.cu (oneRankReduce helper kernels)
 * 
 * Compiling them as a single unit is required for non-RDC mode (-fno-gpu-rdc)
 * and ensures the compiler generates __clang_gpu_used_external correctly.
 * 
 * The device linker then merges this dispatcher with specialized kernels.
 ************************************************************************/

// Include common.cu first - it defines:
//   - ncclShmem (shared memory)
//   - ncclDevFuncTable_1/2/4 (function tables)
//   - ncclDevKernel_Generic_1/2/4 (dispatcher kernels)
#include "common.cu.cpp"

// Include onerank.cu - it defines:
//   - oneRankReduce<T> kernels for each data type (12 total)
//   - ncclLaunchOneRank() host function
// The oneRankReduce kernels are in anonymous namespace, so no symbol conflicts.
#include "onerank.cu.cpp"
