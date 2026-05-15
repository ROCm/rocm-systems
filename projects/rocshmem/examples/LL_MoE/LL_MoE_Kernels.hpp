/******************************************************************************
 * MIT License
 * 
 * Copyright (c) 2025 DeepSeek
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 * 
 * SPDX-License-Identifier: MIT
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 * 
 *****************************************************************************/

#include <rocshmem/rocshmem.hpp>
#include "../util.h"

#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define FINISHED_SUM_TAG 1024
static constexpr int32_t kWaveSize = 64;

using namespace rocshmem;

namespace ll_kernels {

template <typename T>
__host__ __device__ T cell_div(T a, T b) {
  return (a + b - 1) / b;
}

// Warp synchronization function
__forceinline__ __device__ void warp_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

/**
 * Grid barrier implementation using a global counter.
 * All the work-groups must be co-resident on the GPU for this to work
 * correctly.
 */
__forceinline__ __device__ void grid_barrier(int* global_counter,
    int num_blocks) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(&global_counter[0], 1,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(global_counter,
                             __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT) != num_blocks);
  }
  __syncthreads();
}

// Warp reduction sum function
__forceinline__ __device__ int warp_reduce_sum(int val) {
  for (int offset = kWaveSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
  return val;
}

// device warp function to copy the data from source to destination
template <typename T>
__device__ void warp_copy(T* dst, const T* src, size_t num_elems) {
  const int lane_id = threadIdx.x % kWaveSize;
  for (size_t i = lane_id; i < num_elems; i += kWaveSize) {
    dst[i] = src[i];
  }
  __threadfence();
}

/* ================================= DISPATCH =================================
 * Dispatch moves token payloads from the *origin rank* to the *destination
 * expert/rank*, based on routing (topk / expert assignment).
 *
 * In DeepEP LL mode, the key idea is that the communication is structured as:
 *   (A) GPU-side "send" work: write payload into remote-visible buffers +
 *       publish lightweight signals/counters so receivers know what's ready.
 *   (B) GPU-side "recv" work: receivers observe signals, then read/consume the
 *       payload from their receive buffers.
 *
 *****************************************************************************/

// Dispatch kernel for low-latency deepEP
template <int kNumWavesPerGroup, int kNumWaveGroups, typename T>
__global__ __launch_bounds__(kNumWavesPerGroup * kNumWaveGroups * kWaveSize, 1)
void dispatch_kernel(void *packed_recv_x, int *packed_recv_src_info,
    int64_t *packed_recv_layout_range, int *packed_recv_count,
    int *global_atomic_counter, void *rdma_recv_x, int64_t *rdma_recv_count,
    void *rdma_x, const void *x, const int64_t *topk_idx,
    int *atomic_counter_per_expert, int *atomic_finish_counter_per_expert,
    int64_t *next_clean, int num_next_clean_int, int num_tokens, int hidden,
    int num_topk, int num_experts, int rank, int num_ranks) {
  const int wg_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int wave_id = thread_id / kWaveSize;
  const int num_wgs = static_cast<int>(gridDim.x);
  const int lane_id = thread_id % kWaveSize;
  constexpr int num_waves = kNumWavesPerGroup * kNumWaveGroups;
  const int num_local_experts = num_experts / num_ranks;
  const int wave_group_id = wave_id / kNumWavesPerGroup;
  const int sub_wave_id = wave_id % kNumWavesPerGroup;
  const int responsible_expert_id = wg_id * kNumWaveGroups + wave_group_id;

  // size of each token in bytes
  const size_t hidden_bytes = static_cast<size_t>(hidden) * sizeof(T);
  // size of each message in bytes
  const size_t num_bytes_per_msg = sizeof(int) + hidden_bytes;

  DEVICE_ASSERT(num_bytes_per_msg % sizeof(int) == 0);
  const size_t num_int_per_msg = num_bytes_per_msg / sizeof(int);
  
  // Expert counts
  __shared__ int shared_num_tokens_sent_per_expert[kNumWaveGroups];
  if (wave_id < num_waves) {
    constexpr int num_threads = kNumWaveGroups * kNumWavesPerGroup * kWaveSize;
    for (int token_idx = wg_id; token_idx < num_tokens; token_idx += num_wgs) {
      // Pointer to the token data
      // Dimensions: [num_tokens][hidden]
      const T* x_ptr = reinterpret_cast<const T*>(x) + token_idx * hidden;
      // Source symmetric heap buffer for RDMA write
      int* const rdma_x_src_idx = reinterpret_cast<int*>(
          reinterpret_cast<uint8_t*>(rdma_x) + token_idx *
          num_bytes_per_msg);
      // Source data pointer to store the token data after the int header
      T* const rdma_x_vec = reinterpret_cast<T*>(
          reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int));

      // Each warp processes different top-k experts for the same token
      const int64_t dst_expert_idx = wave_id < num_topk ?
          static_cast<int64_t>(topk_idx[token_idx * num_topk + wave_id]) : -1;

      // thread 0 in the warp writes the source token index
      thread_id == 0 ? (*(rdma_x_src_idx) = token_idx) : 0;
      
      // #pragma unroll
      for (int i = thread_id; i < hidden; i += num_threads) {
        // Each thread in the thread block copies a portion of the token data
        rdma_x_vec[i] = x_ptr[i];
      }
      // Synchronize to ensure all threads have completed copying
      __syncthreads();
      // Only warps assigned to valid experts proceed
      if (dst_expert_idx >= 0) {
        // Calculate the destination offset for RDMA write
        int slot_idx = lane_id == 0 ?
                       atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1)
                       : 0;
        // Broadcast the slot index to all threads in the warp
        slot_idx = __shfl(slot_idx, 0);
        const int dst_rank = static_cast<int>(dst_expert_idx / num_local_experts);
        const int dst_expert_local_idx =
            static_cast<int>(dst_expert_idx % num_local_experts);
        // Source ptr for rocSHMEM put
        const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
        // Destination ptr for rocSHMEM put
        // Dimensions: [num_experts][num_ranks][num_tokens]
        const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                             dst_expert_local_idx * num_ranks * num_tokens *
                             num_bytes_per_msg + rank * num_tokens *
                             num_bytes_per_msg + slot_idx * num_bytes_per_msg;

        if (dst_rank != rank) {
          // Remote RDMA write using rocSHMEM
          rocshmem_putmem_nbi_wave(reinterpret_cast<void*>(dst_ptr),
              reinterpret_cast<void*>(src_ptr), num_bytes_per_msg, dst_rank);
        } else {
          // Local copy for same-rank communication
          warp_copy<T>(reinterpret_cast<T*>(dst_ptr),
              reinterpret_cast<T*>(src_ptr), hidden + sizeof(int)/sizeof(T));
        }

        warp_sync();
        // Increment local counter after ensuring PUTs are issued
        lane_id == 0 ?
          __hip_atomic_fetch_add(atomic_finish_counter_per_expert + dst_expert_idx, 1,
                                 __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT) : 0;
      }
    }
  }
  if (wave_id == num_waves - 1) {
    if (wg_id == 0) {
      // The first WG is also responsible for cleaning the next buffer
      for (int i = lane_id; i < num_next_clean_int; i += kWaveSize)
        next_clean[i] = 0;
      for (int i = lane_id; i < num_experts; i += kWaveSize) {
        __hip_atomic_fetch_add(atomic_finish_counter_per_expert + i,
                               FINISHED_SUM_TAG, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    /**
     * Each work group is responsible for some destination experts,
     * read `topk_idx` for them and count the number of tokens sent
     * to those experts
     */
    int expert_count[kNumWaveGroups] = {0};
    const int expert_begin_idx = wg_id * kNumWaveGroups;
    const int expert_end_idx = min(expert_begin_idx + kNumWaveGroups, num_experts);

    // Per lane count
    for (int i = lane_id; i < num_tokens * num_topk; i += kWaveSize) {
      const int64_t idx = static_cast<int64_t>(topk_idx[i]);
      if (idx >= expert_begin_idx && idx < expert_end_idx) {
        expert_count[idx - expert_begin_idx]++;
      }
    }

    // Warp reduce
    for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
      int sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
      if (lane_id == 0) {
        shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
        __hip_atomic_fetch_add(atomic_finish_counter_per_expert + i,
                               FINISHED_SUM_TAG - sum, __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
  // Synchronize work-group
  __syncthreads();

  // Notify each expert about the number of tokens sent to it
  if (responsible_expert_id < num_experts && sub_wave_id == 0 && lane_id == 0) {
    const int dst_rank = responsible_expert_id / num_local_experts;
    const int dst_expert_local_idx = responsible_expert_id % num_local_experts;
    const int num_tokens_sent =
        shared_num_tokens_sent_per_expert[responsible_expert_id -
                                          wg_id * kNumWaveGroups];

    // Wait until all tokens have been sent and counted
    while(__hip_atomic_load(atomic_finish_counter_per_expert + responsible_expert_id,
                             __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) !=
          FINISHED_SUM_TAG * 2);

    if (dst_rank != rank) {
      rocshmem_long_atomic_add(
          rdma_recv_count + dst_expert_local_idx * num_ranks + rank,
          -num_tokens_sent - 1, dst_rank);
    } else {
      /**
       * Local store for same-rank communication
       * TODO: Does it require atomic store? each store is to a unique location
       */
      __hip_atomic_store(rdma_recv_count + dst_expert_local_idx * num_ranks + rank,
                         -num_tokens_sent - 1, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
    }

    // Clean workspace for next use
    atomic_counter_per_expert[responsible_expert_id] = 0;
    atomic_finish_counter_per_expert[responsible_expert_id] = 0;

    // Clean packed_recv_count
    if (dst_rank == 0)
      packed_recv_count[dst_expert_local_idx] = 0;
  }
  warp_sync();

  /**
   * Grid barrier to ensure all WGs have completed sending
   * All WGs must be co-resident for this to work correctly
   */
  grid_barrier(global_atomic_counter, num_wgs);

  // Pack the data from rdma_recv_x to packed_recv_x
  if (responsible_expert_id < num_experts) {
    const int src_rank = responsible_expert_id / num_local_experts;
    const int local_expert_idx = responsible_expert_id % num_local_experts;
    /**
     * Pointer to the starting location of the local expert's data of it's
     * source rank in rdma_recv_x
     * Dimensions: [num_local_experts][num_ranks][num_tokens]
     */
    T* const rdma_recv_x_ptr = reinterpret_cast<T*>(
        reinterpret_cast<uint8_t*>(rdma_recv_x) +
        local_expert_idx * num_ranks * num_tokens * num_bytes_per_msg +
        src_rank * num_tokens * num_bytes_per_msg);
    /**
     * Pointer to the starting location of the local expert's data in
     * packed_recv_x
     */
    T* const packed_recv_x_ptr = reinterpret_cast<T*>(packed_recv_x) +
        local_expert_idx * num_ranks * num_tokens * hidden;
    /**
     * Pointer to the starting location of the local expert's data in
     * packed_recv_src_info
     */
    int* const packed_recv_src_info_ptr = packed_recv_src_info +
        local_expert_idx * num_ranks * num_tokens;
    /**
     * Pointer to the starting location of the local expert's data in
     * packed_recv_layout_range
     */
    int64_t* const packed_recv_layout_range_ptr =
        packed_recv_layout_range + local_expert_idx * num_ranks;

    // Shared between sub-warps in warp groups
    __shared__ int shared_num_recv_tokens[kNumWaveGroups],
                   shared_recv_token_begin_idx[kNumWaveGroups];

    /**
     * Wait until tokens are received for the assigned local expert
     * from its source rank
     */
    int num_recv_tokens, recv_token_begin_idx;
    if (sub_wave_id == 0 && lane_id == 0) {
      while ((num_recv_tokens = __hip_atomic_load(
                  reinterpret_cast<int*>(rdma_recv_count + local_expert_idx *
                                         num_ranks + src_rank),
                  __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) == 0);
      num_recv_tokens = -num_recv_tokens - 1;
      /**
       * Once the number of received tokens is known, pack the data from
       * rdma_recv_x to packed_recv_x
       *
       * each local expert's data is stored contiguously for each rank in
       * packed_recv_x, but the each rank's starting location is not known until
       * the number of received tokens is known
       */
      recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx,
                                       num_recv_tokens);
      // Store the info in the shared buffer for other sub-warps in the warp group
      shared_num_recv_tokens[wave_group_id] = num_recv_tokens;
      shared_recv_token_begin_idx[wave_group_id] = recv_token_begin_idx;
      // Pack the recv range info
      packed_recv_layout_range_ptr[src_rank] =
          (static_cast<int64_t>(recv_token_begin_idx) << 32) |
          static_cast<int64_t>(num_recv_tokens);
    }
    // Synchronize sub-warps in the warp group
    __syncthreads();
    num_recv_tokens = shared_num_recv_tokens[wave_group_id];
    recv_token_begin_idx = shared_recv_token_begin_idx[wave_group_id];
    // Pack the received data from rdma_recv_x to packed_recv_x
    for (int i = sub_wave_id; i < num_recv_tokens; i += kNumWavesPerGroup) {
      // Source token index in rdma_recv_x
      const int* const src_token_idx = reinterpret_cast<int*>(
          reinterpret_cast<uint8_t*>(rdma_recv_x_ptr) +
          i * num_bytes_per_msg);
      // Write the source token index to packed_recv_src_info from lane 0
      if (lane_id == 0) {
        packed_recv_src_info_ptr[recv_token_begin_idx + i] = *src_token_idx;
      }
      warp_sync();
      // Source token data pointer in rdma_recv_x
      const T* const src_token_data = reinterpret_cast<T*>(
          reinterpret_cast<uint8_t*>(rdma_recv_x_ptr) +
          i * num_bytes_per_msg + sizeof(int));
      // Destination token data pointer in packed_recv_x
      T* const dst_token_data = packed_recv_x_ptr +
          (recv_token_begin_idx + i) * hidden;
      // Copy the token data
      warp_copy<T>(dst_token_data, src_token_data, hidden);
    }
  }
}

// Dispatch function to launch the dispatch kernel
template <typename T>
void dispatch(void *packed_recv_x, int* packed_recv_src_info,
    int64_t* packed_recv_layout_range, int* packed_recv_count,
    int* global_atomic_counter, void* rdma_recv_x, int64_t* rdma_recv_count,
    void* rdma_x, const void* x, const int64_t* topk_idx,
    int64_t* next_clean, int num_next_clean_int, int num_tokens, int hidden,
    int num_topk, int num_experts, int rank, int num_ranks,
    void* workspace, hipStream_t stream) {
  constexpr int kNumWavesPerGroup = 4;
  constexpr int kNumWaveGroups = 4;

  constexpr int kNumMaxTopK = 9;
  ASSERT(kNumMaxTopK + 1 <= kNumWavesPerGroup * kNumWaveGroups);

  const auto num_waves   = kNumWaveGroups * kNumWavesPerGroup;
  const auto num_wgs     = cell_div(num_experts, kNumWaveGroups);
  const auto num_threads = num_waves * kWaveSize;

  ASSERT(num_topk <= kNumMaxTopK);

  // Workspace checks
  int* atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
  int* atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;

  ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

  dim3 grid(num_wgs);
  dim3 block(num_threads);

  /**
   * Calculate the maximum number of co-resident work-groups per compute unit
   * based on the resource usage of the kernel
   */
  int max_co_resident_wgs_per_cu = 0;
  CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_co_resident_wgs_per_cu,
      dispatch_kernel<kNumWavesPerGroup, kNumWaveGroups, T>,
      num_threads,
      0));
  // Get the number of compute units
  hipDeviceProp_t device_prop;
  CHECK_HIP(hipGetDeviceProperties(&device_prop, 0));
  const int num_cus = device_prop.multiProcessorCount;
  const int max_sustainable_wgs = max_co_resident_wgs_per_cu * num_cus;

  // Print warning if num_wgs exceeds max co-resident work-groups
  if (num_wgs > max_sustainable_wgs) {
    std::cout << "Warning: Number of work-groups (" << num_wgs
              << ") exceeds max sustainable work-groups ("
              << max_sustainable_wgs << ")." << std::endl;
  }

  dispatch_kernel<kNumWavesPerGroup, kNumWaveGroups, T>
    <<<grid, block, 0, stream>>>(packed_recv_x, packed_recv_src_info,
      packed_recv_layout_range, packed_recv_count, global_atomic_counter,
      rdma_recv_x, rdma_recv_count, rdma_x, x, topk_idx, atomic_counter_per_expert,
      atomic_finish_counter_per_expert, next_clean, num_next_clean_int,
      num_tokens, hidden, num_topk, num_experts, rank, num_ranks);
}

/*
 * ================================= COMBINE ==================================
 * Combine is the reverse direction of dispatch: it returns expert outputs
 * back to the originating ranks/tokens
 *
 * As with dispatch, we reuse the same pre-allocated RDMA/signal buffers
 *****************************************************************************/
template <int kNumWavesPerGroup, int kNumWaveGroups, int kNumMaxTopK, typename T>
__global__ __launch_bounds__(kNumWavesPerGroup * kNumWaveGroups * kWaveSize, 1)
void combine_kernel(T* combined_x, void* rdma_recv_x, int64_t* rdma_recv_flag,
    void* rdma_send_x, const void* x, const int64_t* topk_idx,
    const int* src_info, const int64_t* layout_range,
    int* global_atomic_counter, int64_t* next_clean, int num_next_clean_int,
    int* atomic_clean_flag, int num_tokens, int num_topk, int hidden,
    int num_experts, int rank, int num_ranks) {
  const int wg_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int wave_id = thread_id / kWaveSize;
  const int num_wgs = static_cast<int>(gridDim.x);
  const int num_threads = static_cast<int>(blockDim.x);
  const int lane_id = thread_id % kWaveSize;
  constexpr int num_waves = kNumWavesPerGroup * kNumWaveGroups;
  const int num_local_experts = num_experts / num_ranks;
  const int wave_group_id = wave_id / kNumWavesPerGroup;
  const int sub_wave_id = wave_id % kNumWavesPerGroup;
  const int responsible_expert_id = wg_id * kNumWaveGroups + wave_group_id;

  // size of each slot in bytes
  const size_t num_bytes_per_slot = sizeof(int) + static_cast<size_t>(hidden) *
                                    sizeof(T);
  const size_t num_T_per_slot = num_bytes_per_slot / sizeof(T);

  // Shared memory to synchronize sub-warps in a warp group
  __syncthreads();
  constexpr int max_num_warps = 16;
  __shared__ volatile int sync_large_warp_counters[max_num_warps];
  // initialize the shared memory to zero
  if (thread_id < max_num_warps) {
    sync_large_warp_counters[thread_id] = 0;
  }
  __syncthreads();

  DEVICE_ASSERT(sizeof(int) % sizeof(T) == 0);

  // Clean up next buffer
  if (wg_id == 0 && wave_group_id == 0 && sub_wave_id == 0) {
    for (int i = lane_id; i < num_next_clean_int; i += kWaveSize)
      next_clean[i] = 0;

    warp_sync();
    if (lane_id == 0)
      __hip_atomic_fetch_add(atomic_clean_flag, num_experts,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }

  // Issue rocSHMEM puts
  if (responsible_expert_id < num_experts) {
    const int dst_rank = responsible_expert_id / num_local_experts;
    const int local_expert_idx = responsible_expert_id % num_local_experts;
    const int global_expert_idx = rank * num_local_experts + local_expert_idx;
    const int64_t *layout_info = &layout_range[local_expert_idx * num_ranks +
                                               dst_rank];
    const T* const local_x = reinterpret_cast<const T*>(x) + local_expert_idx *
                             num_ranks * num_tokens * hidden;
    const int* const local_src_info = &src_info[local_expert_idx *
                                                num_ranks * num_tokens];
    T* const rdma_send_x_ptr = reinterpret_cast<T*>(rdma_send_x) +
                               local_expert_idx * num_ranks * num_tokens *
                               num_T_per_slot;
    // Unpack layout info
    const int num_tokens_to_send = reinterpret_cast<const int*>(layout_info)[0];
    const int offset             = reinterpret_cast<const int*>(layout_info)[1];

    // Issue rocSHMEM puts
    for (int token_idx = offset + sub_wave_id; token_idx < offset + num_tokens_to_send;
         token_idx += kNumWaveGroups) {
      const T* const x_ptr = local_x + token_idx * hidden;
      int* const rdma_send_x_tkn_idx = reinterpret_cast<int*>(
          rdma_send_x_ptr + token_idx * num_T_per_slot);
      T* const rdma_send_x_tkn_data = reinterpret_cast<T*>(
          rdma_send_x_tkn_idx + 1);

      /**
       * Copy token data to local buffer for local sends or copy token data to
       * symmetric heap buffer to issue rocSHMEM put for remote sends
       */
      // Token index
      const int src_token_idx = local_src_info[token_idx];
      T* const buf_ptr = rdma_send_x_tkn_data;
      T* const dst_ptr = reinterpret_cast<T*>(rdma_recv_x) +
                         (global_expert_idx * num_tokens + src_token_idx) *
                         num_T_per_slot + sizeof(int)/sizeof(T);

      if (dst_rank == rank) {
        // Local copy for same-rank communication
        // Write the token index
        warp_copy<T>(dst_ptr, x_ptr, hidden);
      } else {
        // Copy to symmetric heap buffer for remote RDMA write
        // Write the token index
        warp_copy<T>(buf_ptr, x_ptr, hidden);
        // Issue RDMA write using rocSHMEM
        rocshmem_putmem_nbi_wave(dst_ptr, buf_ptr, hidden * sizeof(T), dst_rank);
      }
    }

    // Synchronize sub-warps in the warp group
    if (lane_id == 0) {
      volatile int ret = __hip_atomic_fetch_add(
          &sync_large_warp_counters[wave_group_id], 1,
          __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      warp_sync();
      while (sync_large_warp_counters[wave_group_id] < kNumWavesPerGroup);
    }

    if (sub_wave_id == 0 && lane_id == 0) {
      //
      while (__hip_atomic_load(atomic_clean_flag, __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_AGENT) == 0);

      // Issue atomic add to notify expert about completed sends
      if (dst_rank != rank) {
        rocshmem_long_atomic_add(rdma_recv_flag + global_expert_idx, 1,
                                 dst_rank);
      } else {
        // Local store for same-rank communication
        __hip_atomic_store(rdma_recv_flag + global_expert_idx, 1,
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      }
      __hip_atomic_fetch_add(atomic_clean_flag, -1,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
  }

  // Wait until data is received for the assigned expert
  if (responsible_expert_id < num_experts && sub_wave_id == 0 && lane_id == 0) {
    while (__hip_atomic_load(rdma_recv_flag + responsible_expert_id,
                             __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0);
  }

  /**
   * Grid barrier to ensure all WGs have completed receiving
   * All WGs must be co-resident for this to work correctly
   */
  grid_barrier(global_atomic_counter, num_wgs);

  // Pack the data from rdma_recv_x to combined_x
  // Each wg is assigned a token to process
  for (int token_idx = wg_id; token_idx < num_tokens; token_idx += num_wgs) {
    // Read top-k expert indices for the token
    int reg_topk_idx[kNumMaxTopK];
    for (int k = 0; k < num_topk; ++k) {
      reg_topk_idx[k] = static_cast<int>(topk_idx[token_idx * num_topk + k]);
    }

    // Combine the data from the top-k experts
    for (int i = thread_id; i < hidden; i += num_threads) {
      // Iterate over top-k experts
      for (int k = 0; k < num_topk; ++k) if (reg_topk_idx[k] >= 0) {
        int* const rdma_recv_x_tkn_idx = reinterpret_cast<int*>(
            reinterpret_cast<uint8_t*>(rdma_recv_x) +
            (reg_topk_idx[k] * num_tokens + token_idx) * num_bytes_per_slot);
        T* const rdma_recv_x_tkn_data = reinterpret_cast<T*>(
            rdma_recv_x_tkn_idx + 1);
        // Reduce the token data
        combined_x[token_idx * hidden + i] += rdma_recv_x_tkn_data[i];
      }
    }
  }
}

// Combine function to launch the combine kernel
template <typename T>
void combine(T* combined_x, void* rdma_recv_x, int64_t* rdma_recv_flag,
    void* rdma_send_x, const void* x, const int64_t* topk_idx,
    const int* src_info, const int64_t* layout_range,
    int* global_atomic_counter, int64_t* next_clean, int num_next_clean_int,
    int num_tokens, int num_topk, int hidden, int num_experts, int rank,
    int num_ranks, void* workspace, hipStream_t stream) {

  constexpr int kNumWavesPerGroup = 4;
  constexpr int kNumWaveGroups = 4;

  constexpr int kNumMaxTopK = 9;
  ASSERT(num_topk <= kNumMaxTopK);

  const auto num_waves   = kNumWaveGroups * kNumWavesPerGroup;
  const auto num_wgs     = cell_div(num_experts, kNumWaveGroups);
  const auto num_threads = num_waves * kWaveSize;

  int* atomic_clean_flag = reinterpret_cast<int*>(workspace);
  ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);

  dim3 grid(num_wgs);
  dim3 block(num_threads);

  /**
   * Calculate the maximum number of co-resident work-groups per compute unit
   * based on the resource usage of the kernel
   */
  int max_co_resident_wgs_per_cu = 0;
  CHECK_HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_co_resident_wgs_per_cu,
      combine_kernel<kNumWavesPerGroup, kNumWaveGroups, kNumMaxTopK, T>,
      num_threads,
      0));
  // Get the number of compute units
  hipDeviceProp_t device_prop;
  CHECK_HIP(hipGetDeviceProperties(&device_prop, 0));
  const int num_cus = device_prop.multiProcessorCount;
  const int max_sustainable_wgs = max_co_resident_wgs_per_cu * num_cus;

  // Print warning if num_wgs exceeds max co-resident work-groups
  if (num_wgs > max_sustainable_wgs) {
    std::cout << "Warning: Number of work-groups (" << num_wgs
              << ") exceeds max sustainable work-groups ("
              << max_sustainable_wgs << ")." << std::endl;
  }

  combine_kernel<kNumWavesPerGroup, kNumWaveGroups, kNumMaxTopK, T>
    <<<grid, block, 0, stream>>>(combined_x, rdma_recv_x, rdma_recv_flag,
      rdma_send_x, x, topk_idx, src_info, layout_range, global_atomic_counter,
      next_clean, num_next_clean_int, atomic_clean_flag, num_tokens, num_topk,
      hidden, num_experts, rank, num_ranks);

}
}  // namespace ll_kernels