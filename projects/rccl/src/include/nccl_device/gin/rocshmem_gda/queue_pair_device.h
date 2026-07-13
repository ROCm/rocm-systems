/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Lightweight device-only header for QueuePair.
 * Provides only the device-side declarations needed by external consumers,
 * without pulling in IB verbs types or host-only dependencies.
 * Link against rocshmem's device bitcode for the definitions.
 */

#ifndef NCCL_DEVICE_GIN_ROCSHMEM_QUEUE_PAIR_DEVICE_H_
#define NCCL_DEVICE_GIN_ROCSHMEM_QUEUE_PAIR_DEVICE_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace rocshmem {

static __device__ __forceinline__ uint64_t get_active_lane_mask() {
  return __ballot(true);
}

static __device__ __forceinline__ int popcount(uint64_t val) {
  return __builtin_popcountll(val);
}

static __device__ __forceinline__ uint64_t lanemask_eq() {
  return 1ULL << (__builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0)));
}

static __device__ __forceinline__ int get_active_lane_count(uint64_t mask) {
  return popcount(mask);
}

static __device__ __forceinline__ int get_active_lane_num(uint64_t mask) {
  return popcount(mask & (lanemask_eq() - 1));
}

static __device__ __forceinline__ int get_first_active_lane_id(uint64_t mask) {
  return __builtin_ctzll(mask);
}

static __device__ __forceinline__ int get_last_active_lane_id(uint64_t mask) {
  return 63 - __builtin_clzll(mask);
}

enum class ThreadScope : int {
  thread,
  wave,
  wg,
  exclusive
};

class ActiveWFInfo {
 public:
  uint64_t    activemask{0};
  uint64_t    pe_group_mask{0};
  int         pe{-1};
  int         num_pe_group_lanes{0};
  int         pe_group_logical_lane_id{0};
  int         pe_group_first_phys_lane_id{0};
  int         pe_group_last_phys_lane_id{0};
  ThreadScope scope{ThreadScope::thread};
  bool        is_pe_group_first{false};
  bool        is_pe_group_last{false};

  __device__ explicit ActiveWFInfo(int pe, ThreadScope scope = ThreadScope::thread)
      : pe(pe), scope(scope) {
    activemask = get_active_lane_mask();
    switch (scope) {
      case ThreadScope::thread: {
        pe_group_mask       = __match_any_sync(activemask, pe);
        num_pe_group_lanes  = get_active_lane_count(pe_group_mask);
        pe_group_logical_lane_id = get_active_lane_num(pe_group_mask);
        pe_group_first_phys_lane_id = get_first_active_lane_id(pe_group_mask);
        pe_group_last_phys_lane_id  = get_last_active_lane_id(pe_group_mask);
        break;
      }
      case ThreadScope::exclusive: {
        pe_group_mask       = 1;
        num_pe_group_lanes  = 1;
        pe_group_logical_lane_id = 0;
        pe_group_first_phys_lane_id = 0;
        pe_group_last_phys_lane_id  = 0;
        break;
      }
      case ThreadScope::wave:
      case ThreadScope::wg: {
        pe_group_mask       = 1;
        num_pe_group_lanes  = 1;
        pe_group_logical_lane_id = get_active_lane_num(activemask);
        pe_group_first_phys_lane_id = 0;
        pe_group_last_phys_lane_id  = 0;
      }
    }
    is_pe_group_first = (pe_group_logical_lane_id == 0);
    is_pe_group_last  = (pe_group_logical_lane_id == num_pe_group_lanes - 1);
  }
};

class QueuePair {
 public:
  __device__ void put_nbi(void *raddr, uint32_t rkey,
      const void *laddr, uint32_t lkey,
      size_t length, ActiveWFInfo &wf_info, bool ring_db = true);

  __device__ void put_nbi_single(void *raddr, uint32_t rkey,
      const void *laddr, uint32_t lkey,
      size_t length, bool ring_db = true);

  __device__ void atomic_add(void *raddr, uint32_t rkey,
      int64_t value, ActiveWFInfo &wf_info, bool fence = false);

  __device__ void atomic_add_single(void *raddr, uint32_t rkey,
      int64_t value, bool fence = false);

  __device__ void quiet(ActiveWFInfo &wf_info);

  __device__ void quiet_single();
};

}  // namespace rocshmem

#endif  // NCCL_DEVICE_GIN_ROCSHMEM_QUEUE_PAIR_DEVICE_H_
