// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef NCCL_DEVICE_SYMMETRIC_KERNEL_H_
#define NCCL_DEVICE_SYMMETRIC_KERNEL_H_

#if defined(__HIP_PLATFORM_AMD__)
#include "kernel_profiler.h"
#else
#include "../kernel_profiler.cuh"
#endif

#if __CUDA_ARCH__ >= 700
// __grid_constant__ appears to break cuda-gdb
#define NCCL_GRID_CONSTANT __grid_constant__
#else
#define NCCL_GRID_CONSTANT
#endif

// ncclSymkRun_* entrypoints take a leading compile-time `bool EnableProfiler`: false
// emits no instrumentation (byte-identical to the base kernels), true emits the phase
// stamps. The host picks the variant (ncclSymkKernelList vs ncclSymkKernelListProfile).
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLLMC_R(struct ncclSymkDevWorkArgs const* args);

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxTmaLD_AGxTmaST(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LL(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LLMC(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_ST(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_STMC(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_TmaST(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_TmaSTMC(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AlltoAll_LL(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_HierLsaST(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_HierLsaTmaST(struct ncclSymkDevWorkArgs const* args);

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LD(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LDMC(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_TmaLD(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_RailA2A_LsaLD(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_RailA2A_LsaLDMC(struct ncclSymkDevWorkArgs const* args);

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_RailRing_LsaST(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_RailRing_LsaSTMC(struct ncclSymkDevWorkArgs const* args);
#endif
