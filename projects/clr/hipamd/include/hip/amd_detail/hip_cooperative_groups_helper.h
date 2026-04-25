/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 *  @file  amd_detail/hip_cooperative_groups_helper.h
 *
 *  @brief Device side implementation of cooperative group feature.
 *
 *  Defines helper constructs and APIs which aid the types and device API
 *  wrappers defined within `amd_detail/hip_cooperative_groups.h`.
 */
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H

#if __cplusplus
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_runtime.h>  // threadId, blockId
#include <hip/amd_detail/amd_device_functions.h>
#endif
#if !defined(__align__)
#define __align__(x) __attribute__((aligned(x)))
#endif

#if !defined(__CG_QUALIFIER__)
#define __CG_QUALIFIER__ __device__ __forceinline__
#endif

#if !defined(__CG_STATIC_QUALIFIER__)
#define __CG_STATIC_QUALIFIER__ __device__ static __forceinline__
#endif

#if !defined(_CG_STATIC_CONST_DECL_)
#define _CG_STATIC_CONST_DECL_ static constexpr
#endif

using lane_mask = unsigned long long int;
namespace cooperative_groups {

/* Global scope */
template <unsigned int size> using is_power_of_2 =
    __hip_internal::integral_constant<bool, (size & (size - 1)) == 0>;

template <unsigned int size> using is_valid_wavefront =
    __hip_internal::integral_constant<bool, size <= 64>;

template <unsigned int size> using is_valid_tile_size =
    __hip_internal::integral_constant<bool, is_power_of_2<size>::value &&
                                                is_valid_wavefront<size>::value>;

template <typename T> using is_valid_type =
    __hip_internal::integral_constant<bool, __hip_internal::is_integral<T>::value ||
                                                __hip_internal::is_floating_point<T>::value>;

namespace internal {

/**
 * @brief Enums representing different cooperative group types
 * @note  This enum is only applicable on Linux.
 *
 */
typedef enum {
  cg_invalid,
  cg_multi_grid,
  cg_grid,
  cg_workgroup,
  cg_tiled_group,
  cg_coalesced_group
} group_type;
/**
 *  @ingroup CooperativeG
 *  @{
 *  This section describes the cooperative groups functions of HIP runtime API.
 *
 *  The cooperative groups provides flexible thread parallel programming algorithms, threads
 *  cooperate and share data to perform collective computations.
 *
 *  @note  Cooperative groups feature is implemented on Linux, under developement
 *  on Windows.
 *
 */
namespace helper {
/**
 * @brief Create output mask from input_mask at places where base_mask is set
 *
 * Example: base_mask = 0101'0101, input_mask = 1111'0000
 * Output mask: 1100
 * Explaination:
 *           | | | |  | | | |
 * base:    0|1|0|1|'0|1|0|1|   // Which bits are set
 * input:   1|1|1|1|'0|0|0|0|   // Which values are picked
 *           | | | |  | | | |
 * output:    1   1    0   0
 */
__CG_STATIC_QUALIFIER__ unsigned long long adjust_mask(unsigned long long base_mask,
                                                       unsigned long long input_mask) {
  unsigned long long out = 0;
  for (unsigned int i = 0, index = 0; i < warpSize; i++) {
    auto lane_active = base_mask & (1ull << i);
    if (lane_active) {
      auto result = input_mask & (1ull << i);
      out |= ((result ? 1ull : 0ull) << index);
      index++;
    }
  }
  return out;
}
}  // namespace helper
/**
 *
 * @brief  Functionalities related to multi-grid cooperative group type
 * @note  The following cooperative groups functions are only applicable on Linux.
 *
 */
namespace multi_grid {

__CG_STATIC_QUALIFIER__ __hip_uint32_t num_grids() {
  return static_cast<__hip_uint32_t>(__ockl_multi_grid_num_grids());
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t grid_rank() {
  return static_cast<__hip_uint32_t>(__ockl_multi_grid_grid_rank());
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t num_threads() {
  return static_cast<__hip_uint32_t>(__ockl_multi_grid_size());
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t thread_rank() {
  return static_cast<__hip_uint32_t>(__ockl_multi_grid_thread_rank());
}

__CG_STATIC_QUALIFIER__ bool is_valid() { return static_cast<bool>(__ockl_multi_grid_is_valid()); }

__CG_STATIC_QUALIFIER__ void sync() { __ockl_multi_grid_sync(); }

}  // namespace multi_grid

/**
 *  @brief Functionalities related to grid cooperative group type
 *  @note  The following cooperative groups functions are only applicable on Linux.
 */
namespace grid {

__CG_STATIC_QUALIFIER__ __hip_uint32_t num_threads() {
  return static_cast<__hip_uint32_t>((blockDim.z * gridDim.z) * (blockDim.y * gridDim.y) *
                                     (blockDim.x * gridDim.x));
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t thread_rank() {
  // Compute global id of the workgroup to which the current thread belongs to
  __hip_uint32_t blkIdx = static_cast<__hip_uint32_t>((blockIdx.z * gridDim.y * gridDim.x) +
                                                      (blockIdx.y * gridDim.x) + (blockIdx.x));

  // Compute total number of threads being passed to reach current workgroup
  // within grid
  __hip_uint32_t num_threads_till_current_workgroup =
      static_cast<__hip_uint32_t>(blkIdx * (blockDim.x * blockDim.y * blockDim.z));

  // Compute thread local rank within current workgroup
  __hip_uint32_t local_thread_rank = static_cast<__hip_uint32_t>(
      (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + (threadIdx.x));

  return (num_threads_till_current_workgroup + local_thread_rank);
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t block_rank() {
  return static_cast<__hip_uint32_t>((blockIdx.z * gridDim.y * gridDim.x) +
                                     (blockIdx.y * gridDim.x) + (blockIdx.x));
}

__CG_STATIC_QUALIFIER__ bool is_valid() { return static_cast<bool>(__ockl_grid_is_valid()); }

__CG_STATIC_QUALIFIER__ void sync() { __ockl_grid_sync(); }

__CG_STATIC_QUALIFIER__ dim3 grid_dim() {
  return (dim3(static_cast<__hip_uint32_t>(gridDim.x), static_cast<__hip_uint32_t>(gridDim.y),
               static_cast<__hip_uint32_t>(gridDim.z)));
}

__CG_STATIC_QUALIFIER__ unsigned int barrier_arrive() { return __ockl_grid_bar_arrive(); }

__CG_STATIC_QUALIFIER__ unsigned int barrier_signal() { return __ockl_grid_bar_arrive(); }

__CG_STATIC_QUALIFIER__ void barrier_wait(unsigned int s) { __ockl_grid_bar_wait(s); }
}  // namespace grid

/**
 *  @brief Functionalities related to `workgroup` (thread_block in CUDA terminology)
 *  cooperative group type
 *  @note  The following cooperative groups functions are only applicable on Linux.
 */
namespace workgroup {

__CG_STATIC_QUALIFIER__ dim3 group_index() {
  return (dim3(static_cast<__hip_uint32_t>(blockIdx.x), static_cast<__hip_uint32_t>(blockIdx.y),
               static_cast<__hip_uint32_t>(blockIdx.z)));
}

__CG_STATIC_QUALIFIER__ dim3 thread_index() {
  return (dim3(static_cast<__hip_uint32_t>(threadIdx.x), static_cast<__hip_uint32_t>(threadIdx.y),
               static_cast<__hip_uint32_t>(threadIdx.z)));
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t num_threads() {
  return (static_cast<__hip_uint32_t>(blockDim.x * blockDim.y * blockDim.z));
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t thread_rank() {
  return (static_cast<__hip_uint32_t>((threadIdx.z * blockDim.y * blockDim.x) +
                                      (threadIdx.y * blockDim.x) + (threadIdx.x)));
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t block_rank() {
  return (static_cast<__hip_uint32_t>((blockIdx.z * gridDim.x * gridDim.y) +
                                      (blockIdx.y * gridDim.x) + (blockIdx.x)));
}

__CG_STATIC_QUALIFIER__ bool is_valid() { return true; }

__CG_STATIC_QUALIFIER__ void sync() { __syncthreads(); }

__CG_STATIC_QUALIFIER__ dim3 block_dim() {
  return (dim3(static_cast<__hip_uint32_t>(blockDim.x), static_cast<__hip_uint32_t>(blockDim.y),
               static_cast<__hip_uint32_t>(blockDim.z)));
}

__CG_STATIC_QUALIFIER__ void barrier_arrive() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
#if __has_builtin(__builtin_amdgcn_s_barrier_signal) &&                                            \
    __has_builtin(__builtin_amdgcn_s_barrier_wait)
  __builtin_amdgcn_s_barrier_signal(-1);  // -1 is workgroup barriers
#endif  // __builtin_amdgcn_s_barrier_signal && __builtin_amdgcn_s_barrier_wait
}

__CG_STATIC_QUALIFIER__ void barrier_wait() {
#if __has_builtin(__builtin_amdgcn_s_barrier_signal) &&                                            \
    __has_builtin(__builtin_amdgcn_s_barrier_wait)
  __builtin_amdgcn_s_barrier_wait(-1);
#else
  __builtin_amdgcn_s_barrier();
#endif  // __builtin_amdgcn_s_barrier_signal && __builtin_amdgcn_s_barrier_wait
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
}  // namespace workgroup

namespace tiled_group {

// enforce ordering for memory intructions
__CG_STATIC_QUALIFIER__ void sync() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent"); }

}  // namespace tiled_group

namespace coalesced_group {

// enforce ordering for memory intructions
__CG_STATIC_QUALIFIER__ void sync() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent"); }

// Masked bit count
//
// For each thread, this function returns the number of active threads which
// have i-th bit of x set and come before the current thread.
__CG_STATIC_QUALIFIER__ unsigned int masked_bit_count(lane_mask x, unsigned int add = 0) {
  unsigned int counter = 0;
  if (static_cast<int>(warpSize) == 32) {
    counter = __builtin_amdgcn_mbcnt_lo(static_cast<unsigned int>(x), add);
  } else {
    unsigned int lo = static_cast<unsigned int>(x & 0xFFFFFFFF);
    unsigned int hi = static_cast<unsigned int>((x >> 32) & 0xFFFFFFFF);
    counter = __builtin_amdgcn_mbcnt_lo(lo, add);
    counter = __builtin_amdgcn_mbcnt_hi(hi, counter);
  }

  return counter;
}

}  // namespace coalesced_group

namespace cluster {
__CG_STATIC_QUALIFIER__ void sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "cluster");
#if __has_builtin(__builtin_amdgcn_s_cluster_barrier)
  // Generates a signal + wait combination for cluster barrier
  __builtin_amdgcn_s_cluster_barrier();
#else
  __builtin_amdgcn_s_barrier();  // fallback to s_barrier if device does not support clusters
#endif
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "cluster");
}

__CG_STATIC_QUALIFIER__ void barrier_arrive() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "cluster");
#if __has_builtin(__builtin_amdgcn_s_barrier_signal) and                                           \
    __has_builtin(__builtin_amdgcn_s_barrier_wait)
  bool isfirst = __builtin_amdgcn_s_barrier_signal_isfirst(-1);  // -1 is workgroup barrier
  __builtin_amdgcn_s_barrier_wait(-1);

  if (isfirst) {
    // Signal the cluster barrier, -3 means user cluster barrier
    __builtin_amdgcn_s_barrier_signal(-3);
  }
#endif
}

__CG_STATIC_QUALIFIER__ void barrier_wait() {
#if __has_builtin(__builtin_amdgcn_s_barrier_wait)
  // wait on the cluster barrier, -3 means user cluster barrier
  __builtin_amdgcn_s_barrier_wait(-3);
#else
  __builtin_amdgcn_s_barrier();  // Fall back to s_barrier
#endif
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "cluster");
}

__CG_STATIC_QUALIFIER__ dim3 block_index() {
#if __has_builtin(__builtin_amdgcn_cluster_workgroup_id_x)
  return dim3(__builtin_amdgcn_cluster_workgroup_id_x(), __builtin_amdgcn_cluster_workgroup_id_y(),
              __builtin_amdgcn_cluster_workgroup_id_z());
#else
  return dim3{0, 0, 0};
#endif
}

__CG_STATIC_QUALIFIER__ dim3 dim_blocks() {
#if __has_builtin(__builtin_amdgcn_cluster_workgroup_max_id_x)
  return dim3(__builtin_amdgcn_cluster_workgroup_max_id_x() + 1,
              __builtin_amdgcn_cluster_workgroup_max_id_y() + 1,
              __builtin_amdgcn_cluster_workgroup_max_id_z() + 1);
#else
  return dim3{1, 1, 1};
#endif
}

__CG_STATIC_QUALIFIER__ unsigned int block_rank() {
  auto idx = block_index();
  auto dim = dim_blocks();
  return idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
}

__CG_STATIC_QUALIFIER__ dim3 thread_index() {
  const dim3 blockIndex = block_index();
  return dim3(blockIndex.x * blockDim.x + threadIdx.x, blockIndex.y * blockDim.y + threadIdx.y,
              blockIndex.z * blockDim.z + threadIdx.z);
}

__CG_STATIC_QUALIFIER__ unsigned int num_blocks() {
#if __has_builtin(__builtin_amdgcn_cluster_workgroup_max_flat_id)
  return __builtin_amdgcn_cluster_workgroup_max_flat_id() + 1;
#else
  return 1;
#endif
}

__CG_STATIC_QUALIFIER__ dim3 dim_threads() {
  const dim3 dimBlocks = dim_blocks();
  const unsigned int x = dimBlocks.x * blockDim.x;
  const unsigned int y = dimBlocks.y * blockDim.y;
  const unsigned int z = dimBlocks.z * blockDim.z;
  return dim3(x, y, z);
}

__CG_STATIC_QUALIFIER__ unsigned int num_threads() {
  auto d = dim_threads();
  return d.x * d.y * d.z;
}

__CG_STATIC_QUALIFIER__ unsigned int thread_rank() {
  return block_rank() * (blockDim.x * blockDim.y * blockDim.z) +
      ((threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x);
}

template <typename T> __CG_STATIC_QUALIFIER__ T* map_shared_rank(T* in, int rank) {
#if __has_builtin(__builtin_amdgcn_map_shared_rank)
  return (T*)(__builtin_amdgcn_map_shared_rank((void*)in, rank));
#else
  return nullptr;
#endif
}

__CG_STATIC_QUALIFIER__ unsigned int query_shared_rank(const void* in) {
#if __has_builtin(__builtin_amdgcn_query_shared_rank)
  return static_cast<unsigned int>(
      __builtin_amdgcn_query_shared_rank((__attribute__((address_space(11))) const void*)in));
#else
  return 0;
#endif
}
}  // namespace cluster
}  // namespace internal
}  // namespace cooperative_groups
/**
 *  @}
 */

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H
