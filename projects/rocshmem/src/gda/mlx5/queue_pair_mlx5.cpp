/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
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
 *****************************************************************************/

#include "gda/queue_pair.hpp"
#include "util.hpp"
#include "containers/free_list_impl.hpp"
#include "gda/endian.hpp"
#include "segment_builder.hpp"

namespace rocshmem {

__device__ void QueuePair::mlx5_ring_doorbell(uint64_t db_val, uint64_t my_sq_counter) {
  *dbrec = byteswap<uint32_t>(my_sq_counter);
  __atomic_signal_fence(__ATOMIC_SEQ_CST);

  __hip_atomic_store(db.ptr, db_val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  uint64_t db_uint = __hip_atomic_load(&db.uint, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  db_uint ^= 0x100;
  __hip_atomic_store(&db.uint, db_uint, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ void QueuePair::mlx5_quiet() {
  uint64_t activemask = get_active_lane_mask();
  uint8_t num_active_lanes = get_active_lane_count(activemask);
  uint8_t my_logical_lane_id = get_active_lane_num(activemask);
  bool is_leader{my_logical_lane_id == 0};

  if (!is_leader) {
    return;
  }

  while (true) {
    volatile mlx5_cqe64 *cqe_entry = cq_buf;
    uint8_t op_own = *((volatile uint8_t*)&cqe_entry->op_own);
    bool opcode_valid = (op_own >> 4) != MLX5_CQE_INVALID;
    if (!opcode_valid) {
//      continue;
      return;
    }

    uint16_t be_wqe_counter = *((volatile uint16_t*)&cqe_entry->wqe_counter);
    uint16_t wqe_counter = byteswap<uint16_t>(be_wqe_counter);

    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    uint64_t sunk_u64 = __hip_atomic_load(&sq_sunk, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint16_t sunk_u16 = (uint16_t)sunk_u64;
    uint16_t delta = (uint16_t)(wqe_counter - sunk_u16);

    if (delta == 0) {
      return;
    }
    if (delta > 32768) {
      continue;
    }

    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    bool done = __hip_atomic_compare_exchange_strong(&sq_sunk, &sunk_u64, sunk_u64 + delta, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (done) {
      return;
    }
  }
}

__device__ __forceinline__ void QueuePair::mlx5_wait_for_free_sq_slots(
    uint64_t wave_sq_counter, uint8_t num_active_lanes) {
  while (true) {
    uint64_t db_touched = __hip_atomic_load(&sq_db_touched,
                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    uint64_t sunk       = __hip_atomic_load(&sq_sunk,
                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    int64_t num_active_sq_entries =
        static_cast<int64_t>(db_touched) -
        static_cast<int64_t>(sunk);

    if (num_active_sq_entries < 0) {
      continue;
    }

    uint64_t num_free_entries =
        sq_wqe_cnt -
        static_cast<uint64_t>(num_active_sq_entries);

    uint64_t num_entries_until_wave_last_entry =
        wave_sq_counter + num_active_lanes - db_touched;

    if (num_free_entries > num_entries_until_wave_last_entry) {
      break;
    }

    mlx5_quiet();
  }
}

__device__ __forceinline__ void QueuePair::mlx5_build_rma_wqe(
    uint64_t my_sq_counter, uint64_t my_sq_index, uintptr_t laddr,
    uintptr_t raddr, int32_t size, uint8_t opcode) {
  SegmentBuilder seg_build(my_sq_index, sq_buf);

  seg_build.update_ctrl_seg(my_sq_counter, opcode, 0, qp_num,
                            MLX5_WQE_CTRL_CQ_UPDATE, 3, 0, 0);
  seg_build.update_raddr_seg(raddr, rkey);

  if (size <= inline_threshold && opcode == gda_op_rdma_write) {
    seg_build.update_inl_data_seg(reinterpret_cast<const void*>(laddr), size);
  } else {
    seg_build.update_data_seg(laddr, size, lkey);
  }
}

__device__ __forceinline__ void QueuePair::mlx5_wait_for_db_touched_eq(
    uint64_t target_sq_counter) {
  uint64_t db_touched {0};
  do {
    db_touched = __hip_atomic_load(&sq_db_touched,
                 __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } while (db_touched != target_sq_counter);
}

__device__ __forceinline__ void QueuePair::mlx5_ring_doorbell(
    uint64_t wave_sq_counter, uint8_t num_wqes) {
  mlx5_wait_for_db_touched_eq(wave_sq_counter);

  uint8_t *base_ptr = reinterpret_cast<uint8_t*>(sq_buf);
  uint64_t* ctrl_wqe_8B_for_db =
      reinterpret_cast<uint64_t*>(&base_ptr[64 *
        ((wave_sq_counter + num_wqes - 1) % sq_wqe_cnt)]);

  mlx5_ring_doorbell(*ctrl_wqe_8B_for_db, wave_sq_counter + num_wqes);

  __hip_atomic_store(&sq_db_touched, wave_sq_counter + num_wqes,
    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ void QueuePair::mlx5_post_wqe_rma(int32_t size, uintptr_t laddr,
    uintptr_t raddr, uint8_t opcode) {
  uint64_t activemask          = get_active_lane_mask();
  uint8_t  num_active_lanes    = get_active_lane_count(activemask);
  uint8_t  my_logical_lane_id  = get_active_lane_num(activemask);
  bool     is_leader           = {my_logical_lane_id == 0};
  uint64_t leader_phys_lane_id = get_first_active_lane_id(activemask);

  uint8_t  num_wqes        = num_active_lanes;
  uint64_t wave_sq_counter = 0;
  uint64_t my_sq_counter   = 0;
  uint64_t my_sq_index     = 0;

  // 1. Leader allocates SQ entries for the whole wave
  if (is_leader) {
    wave_sq_counter = __hip_atomic_fetch_add(&sq_posted, num_wqes,
                      __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
  }
  wave_sq_counter = __shfl(wave_sq_counter, leader_phys_lane_id);
  my_sq_counter   = wave_sq_counter + my_logical_lane_id;
  my_sq_index     = my_sq_counter % sq_wqe_cnt;

  // 2. Wait for SQ space for the whole wave
  mlx5_wait_for_free_sq_slots(wave_sq_counter, num_active_lanes);

  // 3. Build the WQE for this lane
  mlx5_build_rma_wqe(my_sq_counter, my_sq_index, laddr, raddr, size, opcode);

  __atomic_signal_fence(__ATOMIC_SEQ_CST);

  // 4. Leader rings doorbell for the wave
  if (is_leader) {
    mlx5_ring_doorbell(wave_sq_counter, num_wqes);
  }
}

__device__ __forceinline__ uint64_t*
QueuePair::mlx5_allocate_wave_fetching_atomic_buffer(
    uint64_t wave_sq_counter, bool is_leader,
    uint64_t leader_phys_lane_id) {
  uint64_t* wave_fetch_atomic{nullptr};
  if (is_leader) {
    mlx5_wait_for_db_touched_eq(wave_sq_counter);

    auto res = fetching_atomic_freelist->pop_front();
    while (!res.success) {
      res = fetching_atomic_freelist->pop_front();
    }
    wave_fetch_atomic = res.value;
  }
  wave_fetch_atomic = (uint64_t*)__shfl((uint64_t)wave_fetch_atomic,
                        leader_phys_lane_id);
  return wave_fetch_atomic;
}

__device__ __forceinline__ void QueuePair::mlx5_build_amo_wqe(
    uint64_t my_sq_counter, uint64_t my_sq_index, uintptr_t raddr,
    uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp, bool fetching,
    uint64_t *wave_fetch_atomic) {
  SegmentBuilder seg_build(my_sq_index, sq_buf);
  seg_build.update_ctrl_seg(my_sq_counter, opcode, 0, qp_num,
                            MLX5_WQE_CTRL_CQ_UPDATE, 4, 0, 0);
  seg_build.update_raddr_seg(raddr, rkey);
  seg_build.update_atomic_seg(atomic_data, atomic_cmp);

  if (fetching) {
    seg_build.update_data_seg(reinterpret_cast<uintptr_t>(wave_fetch_atomic), 8, fetching_atomic_lkey);
  } else {
    seg_build.update_data_seg(reinterpret_cast<uintptr_t>(nonfetching_atomic), 8, nonfetching_atomic_lkey);
  }
}

__device__ uint64_t QueuePair::mlx5_post_wqe_amo(int32_t size,
    uintptr_t raddr, uint8_t opcode, int64_t atomic_data,
    int64_t atomic_cmp, bool fetching) {
  uint64_t activemask          = get_active_lane_mask();
  uint8_t  num_active_lanes    = get_active_lane_count(activemask);
  uint8_t  my_logical_lane_id  = get_active_lane_num(activemask);
  bool     is_leader           = {my_logical_lane_id == 0};
  uint64_t leader_phys_lane_id = get_first_active_lane_id(activemask);

  uint8_t  num_wqes        = num_active_lanes;
  uint64_t wave_sq_counter = 0;
  uint64_t my_sq_counter   = 0;
  uint64_t my_sq_index     = 0;

  // 1. Leader allocates SQ entries for the whole wave
  if (is_leader) {
    wave_sq_counter = __hip_atomic_fetch_add(&sq_posted, num_wqes,
                      __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  wave_sq_counter = __shfl(wave_sq_counter, leader_phys_lane_id);
  my_sq_counter   = wave_sq_counter + my_logical_lane_id;
  my_sq_index     = my_sq_counter % sq_wqe_cnt;

  // 2. Wait for SQ space for the whole wave
  mlx5_wait_for_free_sq_slots(wave_sq_counter, num_active_lanes);

  uint64_t* wave_fetch_atomic{nullptr};
  if (fetching) {
    wave_fetch_atomic = mlx5_allocate_wave_fetching_atomic_buffer(
        wave_sq_counter,
        is_leader,
        leader_phys_lane_id);
  }

  // 3. Build the WQE for this lane
  mlx5_build_amo_wqe(my_sq_counter, my_sq_index, raddr, opcode,
                     atomic_data, atomic_cmp, fetching,
                     wave_fetch_atomic + my_logical_lane_id);

  __atomic_signal_fence(__ATOMIC_SEQ_CST);

  // 4. Leader rings doorbell for the wave
  if (is_leader) {
    mlx5_ring_doorbell(wave_sq_counter, num_wqes);
  }

  // 5. Fetch result if requested
  uint64_t ret{0};
  if (fetching) {
    mlx5_quiet();
    ret = wave_fetch_atomic[my_logical_lane_id];

    __atomic_signal_fence(__ATOMIC_SEQ_CST);

    if (is_leader) {
      fetching_atomic_freelist->push_back(wave_fetch_atomic);
    }
  }
  return ret;
}

}  // namespace rocshmem
