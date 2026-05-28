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
#include "gda/endian.hpp"
#include "util.hpp"
#include "containers/free_list_impl.hpp"

namespace rocshmem {

__device__ uint64_t QueuePair::get_same_qp_lane_mask() {
  uint64_t lane_mask = get_active_lane_mask();
  uintptr_t this_val = reinterpret_cast<uintptr_t>(this);

  // exclude threads operating on a different qp from this thread lane mask
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    uint64_t bit_i = 1ull << i;
    if ((lane_mask & bit_i) && __shfl(this_val, i) != this_val) {
      lane_mask &= ~bit_i;
    }
  }

  return lane_mask;
}

__device__ uint32_t QueuePair::reserve_sq(uint64_t activemask, uint32_t num_wqes) {
  uint32_t my_sq_prod = 0;

  // reserve space for wqes in sq
  if (is_first_active_lane(activemask)) {
    my_sq_prod = __hip_atomic_fetch_add(&sq_prod, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  my_sq_prod = __shfl(my_sq_prod, get_first_active_lane_id(activemask));

  // wait for that space to be available
  ionic_quiet_internal(activemask, my_sq_prod + num_wqes - sq_mask);

  return my_sq_prod;
}

__device__ uint32_t QueuePair::commit_sq(uint64_t activemask, uint32_t my_sq_prod, uint32_t my_sq_pos, uint32_t num_wqes) {
  uint32_t dbprod = my_sq_prod + num_wqes;

  spin_lock_acquire_shared(&sq_lock, activemask);

  if (is_first_active_lane(activemask) && ((sq_dbprod - dbprod) & (1u << 31))) {
    sq_dbprod = dbprod;

    ionic_ring_doorbell(dbprod);
  }

  spin_lock_release_shared(&sq_lock, activemask);

  return dbprod;
}

__device__ void QueuePair::poll_wave_cqes(uint64_t activemask) {
  uint32_t my_logical_lane_id = get_active_lane_num(activemask);
  uint32_t my_cq_pos = cq_pos + my_logical_lane_id;

  /* Look at the cqe at the current position in the cq buffer */
  struct ionic_v1_cqe *cqe = &ionic_cq_buf[my_cq_pos & cq_mask];

  /* Determine expected color based on cq wrap count */
  uint32_t qtf_color_bit = swap_endian_val<uint32_t>(IONIC_V1_CQE_COLOR);
  uint32_t qtf_color_exp = qtf_color_bit;
  if (my_cq_pos & (cq_mask + 1)) {
    qtf_color_exp = 0;
  }

  /* Check if my cqe color == expected color */
  uint32_t qtf_be = *(volatile uint32_t *)(&cqe->qid_type_flags);
  if ((qtf_be & qtf_color_bit) != qtf_color_exp) {
    return;
  }

  uint32_t msn = swap_endian_val<uint32_t>(cqe->send.msg_msn);

  /* Report if the completion indicates an error. */
  if (!!(qtf_be & swap_endian_val<uint32_t>(IONIC_V1_CQE_ERROR))) {
#if defined(DEBUG)
    uint32_t qtf = swap_endian_val<uint32_t>(qtf_be);
    uint32_t qid = qtf >> IONIC_V1_CQE_QID_SHIFT;
    uint32_t type = (qtf >> IONIC_V1_CQE_TYPE_SHIFT) & IONIC_V1_CQE_TYPE_MASK;
    uint32_t flag = qtf & 0xf;
    uint32_t status = swap_endian_val<uint32_t>(cqe->status_length);
    uint64_t npg = cqe->send.npg_wqe_idx_timestamp & IONIC_V1_CQE_WQE_IDX_MASK;

    printf("QUIET ERROR: %s qid %u type %u flag %#x status %u msn %u npg %lu\n",
           dev_name, qid, type, flag, status, msn, npg);
#endif
    /* No other way to signal an error, so just crash. */
    abort();
  }

  /* Only proceed with the furthest ahead cqe to update the sq state */
  uint64_t my_lane_mask = 1ull << __lane_id();
  uint64_t lesser_lane_mask = my_lane_mask - 1;
  if (my_lane_mask != (__ballot(true) & activemask & ~lesser_lane_mask)) {
    return;
  }

  /* update position in the cq */
  cq_pos = my_cq_pos + 1;

  /*
   * Ring cq doorbell frequently enough to avoid cq full.
   *
   * NB: IONIC_CQ_GRACE is 100
   */
  if (((cq_pos - cq_dbpos) & cq_mask) >= 100) {
    cq_dbpos = cq_pos;
    __atomic_store_n(cq_dbreg, cq_dbval | (cq_mask & cq_dbpos), __ATOMIC_SEQ_CST); //TODO:maybe relaxed?
  }

  sq_msn = msn;
}

__device__ void QueuePair::ionic_quiet_internal(uint64_t activemask, uint32_t cons) {
  uint32_t greed = 10;

  /* wait for sq_msn to catch up or pass cons. */
  /* 0x800000 - sign bit for 24-bit fields     */
  while ((sq_msn - cons) & 0x800000) {
    if (!spin_lock_try_acquire_shared(&cq_lock, activemask)) {
      continue;
    }

    /* with lock acquired, this wave polls cqes until caught up */
    while ((sq_msn - cons) & 0x800000) {
      uint32_t old_sq_msn = sq_msn;

      poll_wave_cqes(activemask);

      if (!((sq_msn - cons) & 0x800000)) {
        if (sq_msn == old_sq_msn) {
          break;
        }
        if (!greed) {
          break;
        }
        --greed;
      }
    }

    spin_lock_release_shared(&cq_lock, activemask);
    break;
  }
}

__device__ void QueuePair::ionic_ring_doorbell(uint32_t pos) {
  // When threads write at once to the same address, not all writes reach the bus.
  // Take turns and insert a thread fence between writes to the same address.
  for (int i = 0; i < 64; ++i) {
    if (__lane_id() == i) {
      __threadfence();
      __atomic_store_n(sq_dbreg, sq_dbval | (sq_mask & pos), __ATOMIC_SEQ_CST);
    }
  }
  __threadfence();
}

__device__ void QueuePair::ionic_quiet() {
  ionic_quiet_internal(get_same_qp_lane_mask(), sq_prod);
}

__device__ void QueuePair::ionic_post_wqe_rma(int pe, int32_t size, uintptr_t *laddr, uintptr_t *raddr, uint8_t opcode, Collectivity cy) {
  uint64_t activemask = get_same_qp_lane_mask();
  uint32_t my_logical_lane_id = get_active_lane_num(activemask);
  uint32_t num_wqes = 1;
  if (cy == THREAD) {
    num_wqes = get_active_lane_count(activemask);
  }

  uint32_t my_sq_prod = reserve_sq(activemask, num_wqes);
  if (cy == WAVE) {
    if (!is_first_active_lane(activemask)) {
      return;
    }
    activemask &= activemask ^ (activemask - 1);
  }

  uint32_t my_sq_pos = my_sq_prod + my_logical_lane_id;
  struct ionic_v1_wqe *wqe = &ionic_sq_buf[my_sq_pos & sq_mask];
  uint16_t wqe_flags = 0;

  if (!(my_sq_pos & (sq_mask + 1))) {
    wqe_flags |= swap_endian_val<uint16_t>(IONIC_V1_FLAG_COLOR);
  }

  if (is_last_active_lane(activemask)) {
    wqe_flags |= swap_endian_val<uint16_t>(IONIC_V1_FLAG_SIG);
  }

  // TODO why is this needed?
  if (size && !laddr && opcode == IONIC_V2_OP_RDMA_WRITE) {
    size = 1;
  }

  wqe->base.wqe_idx = my_sq_pos;
  wqe->base.op = opcode;
  wqe->base.num_sge_key = size ? 1 : 0;
  wqe->base.imm_data_key = swap_endian_val<uint32_t>(0);

  wqe->common.rdma.remote_va_high = swap_endian_val<uint32_t>(reinterpret_cast<uint64_t>(raddr) >> 32);
  wqe->common.rdma.remote_va_low = swap_endian_val<uint32_t>(reinterpret_cast<uint64_t>(raddr));
  wqe->common.rdma.remote_rkey = swap_endian_val<uint32_t>(rkey);
  wqe->common.length = swap_endian_val<uint32_t>(size);

  if (size) {
    if (opcode == IONIC_V2_OP_RDMA_WRITE && size <= inline_threshold) {
      wqe_flags |= swap_endian_val<uint16_t>(IONIC_V1_FLAG_INL);
      wqe->base.num_sge_key = 0;
      if (!laddr) {
        // TODO why is this needed?
        wqe->common.pld.data[0] = 1;
      } else {
        memcpy(wqe->common.pld.data, laddr, size);
      }
    } else {
      wqe->common.pld.sgl[0].va = swap_endian_val<uint64_t>(reinterpret_cast<uint64_t>(laddr));
      wqe->common.pld.sgl[0].len = swap_endian_val<uint32_t>(size);
      wqe->common.pld.sgl[0].lkey = swap_endian_val<uint32_t>(lkey);
    }
  }

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

  commit_sq(activemask, my_sq_prod, my_sq_pos, num_wqes);
}

__device__ uint64_t QueuePair::ionic_post_wqe_amo(int pe, int32_t size, uintptr_t *raddr, uint8_t opcode,
                                                  int64_t atomic_data, int64_t atomic_cmp, bool fetching) {
  uint64_t activemask = get_same_qp_lane_mask();
  uint32_t num_wqes = get_active_lane_count(activemask);
  uint32_t my_logical_lane_id = get_active_lane_num(activemask);
  bool is_leader{my_logical_lane_id == 0};
  const uint64_t leader_phys_lane_id = get_first_active_lane_id(activemask);
  uint32_t my_sq_prod = reserve_sq(activemask, num_wqes);
  uint32_t my_sq_pos = my_sq_prod + my_logical_lane_id;
  struct ionic_v1_wqe *wqe = &ionic_sq_buf[my_sq_pos & sq_mask];
  uint16_t wqe_flags = 0;
  uint32_t cons;

  uint64_t* wave_fetch_atomic{nullptr};
  if (fetching) {
    if (is_leader) {
      auto res = fetching_atomic_freelist->pop_front();
      while (!res.success) {
        res = fetching_atomic_freelist->pop_front();
      }
      wave_fetch_atomic = res.value;
    }
    wave_fetch_atomic = (uint64_t*)__shfl((uint64_t)wave_fetch_atomic, leader_phys_lane_id);
  }

  if (!(my_sq_pos & (sq_mask + 1))) {
    wqe_flags |= swap_endian_val<uint16_t>(IONIC_V1_FLAG_COLOR);
  }

  if (is_last_active_lane(activemask)) {
    wqe_flags |= swap_endian_val<uint16_t>(IONIC_V1_FLAG_SIG);
  }

  wqe->base.wqe_idx = my_sq_pos;
  wqe->base.op = opcode;
  wqe->base.num_sge_key = 1;
  wqe->base.imm_data_key = swap_endian_val<uint32_t>(0);

  wqe->atomic_v2.remote_va_high = swap_endian_val<uint32_t>(reinterpret_cast<uint64_t>(raddr) >> 32);
  wqe->atomic_v2.remote_va_low = swap_endian_val<uint32_t>(reinterpret_cast<uint64_t>(raddr));
  wqe->atomic_v2.remote_rkey = swap_endian_val<uint32_t>(rkey);
  wqe->atomic_v2.swap_add_high = swap_endian_val<uint32_t>(atomic_data >> 32);
  wqe->atomic_v2.swap_add_low = swap_endian_val<uint32_t>(atomic_data);
  wqe->atomic_v2.compare_high = swap_endian_val<uint32_t>(atomic_cmp >> 32);
  wqe->atomic_v2.compare_low = swap_endian_val<uint32_t>(atomic_cmp);

  if (fetching) {
    wqe->atomic_v2.local_va = swap_endian_val<uint64_t>(reinterpret_cast<uint64_t>(wave_fetch_atomic + my_logical_lane_id));
    wqe->atomic_v2.lkey = swap_endian_val<uint32_t>(fetching_atomic_lkey);
  } else {
    wqe->atomic_v2.local_va = swap_endian_val<uint64_t>(reinterpret_cast<uint64_t>(nonfetching_atomic));
    wqe->atomic_v2.lkey = swap_endian_val<uint32_t>(nonfetching_atomic_lkey);
  }

  __hip_atomic_store(&wqe->base.flags, wqe_flags, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);

  cons = commit_sq(activemask, my_sq_prod, my_sq_pos, num_wqes);

  uint64_t ret{0};
  if (fetching) {
    ionic_quiet_internal(activemask, cons);
    ret = wave_fetch_atomic[my_logical_lane_id];
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    if (is_leader) {
      fetching_atomic_freelist->push_back(wave_fetch_atomic);
    }
  }
  return ret;
}

}  // namespace rocshmem
