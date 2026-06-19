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

#include "log.hpp"
#include "bit.hpp"
#include "util.hpp"

#include "gda/endian.hpp"
#include "gda/queue_pair.hpp"

namespace rocshmem {

#define MLX5_LOCK_USE_S_SLEEP  1
#define MLX5_LOCK_USE_S_WAKEUP (0 && MLX5_LOCK_USE_S_SLEEP)
// sleep for up to 64 * MLX5_LOCK_S_SLEEP_DELAY clock cycles
static constexpr int MLX5_LOCK_S_SLEEP_DELAY = 2;

#if MLX5_LOCK_USE_S_WAKEUP
__device__ static inline void amdgcn_s_wakeup() {
  /* why doesn't __builtin_amdgcn_s_wakeup() exist?
   * signals other wavefronts in the same workgroup to exit early from s_sleep */
  asm volatile("s_wakeup");
}
#endif

__device__ static inline void acquire_lock(uint32_t *lock) {
  /* acquire lock when new value 1 (locked) is exchanged with prior value 0 (unlocked)
   *
   * the __ATOMIC_ACQUIRE load synchronizes with the __ATOMIC_RELEASE store in release_lock(),
   * but not with the (implicit) __ATOMIC_RELAXED store part of the exchange
   * this is fine, since we only need to ensure happens-before between the threads
   * that released and acquired the lock, not between the different threads contending on the lock
   * when they (eventually) acquire the lock, *then* they will synchronize */
  while (__hip_atomic_exchange(lock, 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) {
#if MLX5_LOCK_USE_S_SLEEP
    // sleep so we don't hammer the memory
    __builtin_amdgcn_s_sleep(MLX5_LOCK_S_SLEEP_DELAY);
#endif
  }
}

__device__ static inline void release_lock(uint32_t *lock) {
  // release lock by storing 0 (unlocked)
  __hip_atomic_store(lock, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#if MLX5_LOCK_USE_S_WAKEUP
  // wake up any other sleeping waves (in the same workgroup)
  amdgcn_s_wakeup();
#endif
}

__device__ static inline uint16_t mlx5_wqe_idx(const gda_mlx5_device_sq& sq, uint8_t lane_id) {
  return static_cast<uint16_t>(sq.post + lane_id);
}

__device__ static inline uint16_t mlx5_sq_idx(const gda_mlx5_device_sq& sq, uint16_t wqe_idx) {
  // sq.depth is a power of 2, so just mask off everything above that
  return wqe_idx & sq.depth_mask;
}

__device__ void QueuePair::mlx5_ring_doorbell(uint64_t sq_post, const gda_mlx5_wqe& wqe) {
  // sq_wqebb_counter is the least significant bits of the post counter
  uint16_t sq_wqebb_counter = static_cast<uint16_t>(sq_post);
  // gda_mlx5_db_register constructor extracts first 8 bytes of WQE
  gda_mlx5_db_register db_val{wqe};
  __be32 be_sq_wqebb_counter = endian::to_be<uint32_t>(sq_wqebb_counter);

  // get BlueFlame buffer from SQ
  gda_mlx5_bf_buffer* bf = mlx5_sq.bf_buffer();

  // store sq_wqebb_counter to doorbell record
  __hip_atomic_store(mlx5_sq.dbrec, be_sq_wqebb_counter, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  // ring doorbell by storing first 8B of WQE to the doorbell register
  __hip_atomic_store(&bf->db_reg, db_val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

  LOGD_TRACE("SQ: posted WQEs with dbrec(%p)=%x (%hu), dbreg(%p)=%lx (%x, %x)",
             mlx5_sq.dbrec, be_sq_wqebb_counter, sq_wqebb_counter,
             &bf->db_reg, db_val.val, db_val.wqe_header.opmod_idx_opcode, db_val.wqe_header.qpn_ds);
}

[[maybe_unused]] __attribute__((noinline))
__device__ void QueuePair::mlx5_print_cqe_error(const mlx5_cqe64* cqe, uint8_t opcode) {
  const mlx5_err_cqe* err_cqe = reinterpret_cast<const mlx5_err_cqe*>(cqe);
  uint8_t syndrome = 0x0;

  switch (opcode) {
  case MLX5_CQE_RESP_WR_IMM:
  case MLX5_CQE_RESP_SEND:
  case MLX5_CQE_RESP_SEND_IMM:
  case MLX5_CQE_RESP_SEND_INV:
    // (valid) responder completion?!
    LOGD_ERROR("CQ: unexpected responder completion (%x)", opcode);
    break;
  case MLX5_CQE_RESIZE_CQ:
  case MLX5_CQE_NO_PACKET:
    LOGD_ERROR("CQ: unexpected completion type (%x)", opcode);
    break;
  case MLX5_CQE_SIG_ERR:
    LOGD_ERROR("CQ: unexpected signature error (%x)", opcode);
    break;
  case MLX5_CQE_REQ_ERR:
    syndrome = __hip_atomic_load(&err_cqe->syndrome, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    switch (syndrome) {
    case MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR:
      LOGD_ERROR("CQ requester error LOCAL_LENGTH_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR:
      LOGD_ERROR("CQ requester error LOCAL_QP_OP_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_LOCAL_PROT_ERR:
      LOGD_ERROR("CQ requester error LOCAL_PROT_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_WR_FLUSH_ERR:
      LOGD_ERROR("CQ requester error WR_FLUSH_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_MW_BIND_ERR:
      LOGD_ERROR("CQ requester error MW_BIND_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_BAD_RESP_ERR:
      LOGD_ERROR("CQ requester error BAD_RESP_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_LOCAL_ACCESS_ERR:
      LOGD_ERROR("CQ requester error LOCAL_ACCESS_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_REMOTE_INVAL_REQ_ERR:
      LOGD_ERROR("CQ requester error REMOTE_INVAL_REQ_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR:
      LOGD_ERROR("CQ requester error REMOTE_ACCESS_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_REMOTE_OP_ERR:
      LOGD_ERROR("CQ requester error REMOTE_OP_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR:
      LOGD_ERROR("CQ requester error TRANSPORT_RETRY_EXC_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR:
      LOGD_ERROR("CQ requester error RNR_RETRY_EXC_ERR (%x)", syndrome);
      break;
    case MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR:
      LOGD_ERROR("CQ requester error REMOTE_ABORTED_ERR (%x)", syndrome);
      break;
    default:
      LOGD_ERROR("CQ requester error unknown syndrome type (%x)", syndrome);
      break;
    }
    break;
  case MLX5_CQE_RESP_ERR:
    LOGD_ERROR("CQ: unexpected responder error (%x)", opcode);
    break;
  case MLX5_CQE_INVALID: {
    uint8_t owner = __hip_atomic_load(&cqe->op_own, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                    & MLX5_CQE_OWNER_MASK;
    LOGD_ERROR("CQ: invalid completion (%x), check owner bit = %u?", opcode, owner);
    break;
  }
  default:
    LOGD_ERROR("CQ: unknown completion type (%x)", opcode);
    break;
  }
  abort();
}

// precondition: called with all active lanes using different QPs
__device__ void QueuePair::mlx5_poll_cq_until(uint16_t requested_available_slots) {
  uint16_t sq_depth = mlx5_sq.depth;

  uint64_t sq_post = __hip_atomic_load(&mlx5_sq.post, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  // don't need to check CQ if we haven't ever filled SQ and there's enough space left
  if (sq_post + requested_available_slots <= sq_depth) {
    return;
  }

  while (true) {
    struct mlx5_cqe64* cqe = mlx5_cq.buf;

    /* Update the SQ head
     * This param provides us the sq_wqebb_counter; all our WQEs are exactly one WQEBB (64B) */
    // 32-bit load: big-endian 16-bit field, then two 8-bit fields
    uint32_t wqecnt_sig_op_own = __hip_atomic_load(reinterpret_cast<uint32_t*>(&cqe->wqe_counter),
                                                   __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    // GPU is little-endian, so wqe_counter is loaded into the low half of wqecnt_sig_op_own
    __be16 be_wqe_counter = static_cast<__be16>(wqecnt_sig_op_own);
    /* GPU is little-endian, so op_own is loaded into the top half of wqecnt_sig_op_own;
     * opcode is the top 4 bits of op_own */
    uint8_t opcode = static_cast<uint8_t>(wqecnt_sig_op_own >> 28);
    uint16_t sq_head = endian::from_be(be_wqe_counter);

    // sq_tail is the least significant bits of the post counter
    uint16_t sq_tail = static_cast<uint16_t>(sq_post);

    // CQEs are initially invalid, retry until we see a valid CQE
    if (opcode == MLX5_CQE_INVALID) {
      LOGD_TRACE("CQ: invalid completion (%x)", opcode);
      continue;
    }

#if defined(BUILD_DEBUG_DEVICE)
    if (opcode != MLX5_CQE_REQ)
      mlx5_print_cqe_error(cqe, opcode);
#endif

    /* sq_tail is an index to the next free WQE i.e. counts number of posted WQEs
     * sq_head is an index to the *last* completed WQE - need to add one to get *count* of completed WQEs */
    uint16_t posted    = sq_tail;
    uint16_t completed = sq_head + 1;

    /* posted >= completed, except when posted has wrapped around 0xFFFF and completed hasn't
     * but posted - completed is correct even when it wraps around
     * in some marginal cases it's maybe possible to see consumed_slots > sq_depth,
     * but in that case available_slots will be very large, > requested_available_slots,
     * and the loop will continue for another iteration */
    uint16_t consumed_slots  = posted   - completed;
    uint16_t available_slots = sq_depth - consumed_slots;

    /* continue until both:
     *   - no additional WQEs have been posted
     *   - the number of requested SQ slots are available */
    uint64_t prior_sq_post = sq_post;
    sq_post = __hip_atomic_load(&mlx5_sq.post, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    if (sq_post == prior_sq_post && available_slots >= requested_available_slots) {
      return;
    }
  }
}

// precondition: called with all active lanes using different QPs
__device__ void QueuePair::mlx5_quiet() {
  mlx5_poll_cq_until(mlx5_sq.depth);
}

/**
 * TODO: This function is redundant but kept because ionic has a different
 * quiet_single implementation. Remove once ionic's quiet is unified.
 */
__device__ void QueuePair::mlx5_quiet_single() {
  mlx5_poll_cq_until(mlx5_sq.depth);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
__device__ void QueuePair::mlx5_post_wqe_rma(int32_t length, uintptr_t raddr,
    uint32_t rkey, uintptr_t laddr, uint32_t lkey,
    uint8_t opcode, ActiveWFInfo &wf_info, bool ring_db) {
  if (wf_info.is_pe_group_last) {
    // get SQ lock
    acquire_lock(&mlx5_sq.lock);
    // poll until we have enough WQEBB for all lanes using this QP
    mlx5_poll_cq_until(wf_info.num_pe_group_lanes);
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_wqe_idx(mlx5_sq, wf_info.pe_group_logical_lane_id);
  uint16_t sq_idx = mlx5_sq_idx(mlx5_sq, wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = gda_mlx5_wqe_rma::can_inline(opcode, length, inline_threshold);

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, byteswap<uint32_t>(rkey), laddr, byteswap<uint32_t>(lkey),
                   static_cast<uint32_t>(length), send_inline};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    // increment post counter
    mlx5_sq.post += wf_info.num_pe_group_lanes;
    // we are the last thread in the wavefront, so we have the last WQE posted
    if (ring_db) {
      mlx5_ring_doorbell(mlx5_sq.post, wqe);
    }
    // release SQ lock
    release_lock(&mlx5_sq.lock);
  }
}

// precondition: called with all active lanes using different QPs
__device__ void QueuePair::mlx5_post_wqe_rma_single(int32_t length, uintptr_t laddr,
                                                    uint32_t lkey, uintptr_t raddr,
                                                    uint32_t rkey, uint8_t opcode,
                                                    bool ring_db) {
  bool send_inline = gda_mlx5_wqe_rma::can_inline(opcode, length, inline_threshold);

  // get SQ lock
  acquire_lock(&mlx5_sq.lock);
  // poll until we have enough space for at least one WQE
  mlx5_poll_cq_until(1);

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_wqe_idx(mlx5_sq, 0);
  uint16_t sq_idx = mlx5_sq_idx(mlx5_sq, wqe_idx);

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, byteswap<uint32_t>(rkey), laddr, byteswap<uint32_t>(lkey),
                   static_cast<uint32_t>(length), send_inline};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  // increment post counter
  mlx5_sq.post += 1;

  if (ring_db) {
    // ring doorbell for this WQE
    mlx5_ring_doorbell(mlx5_sq.post, wqe);
  }

  // release SQ lock
  release_lock(&mlx5_sq.lock);
}

/* can be called with all active lanes using any number of different QPs, don't assume anything
 * assumes that `fetching' is constant across all lanes using the same QP
 * TODO: make `fetching' a template parameter */
__device__ uint64_t QueuePair::mlx5_post_wqe_amo(uintptr_t raddr, uint32_t rkey,
    uint8_t opcode, int64_t atomic_data, int64_t atomic_cmp,
    ActiveWFInfo &wf_info, bool fetching, bool fence) {
  if (wf_info.is_pe_group_last) {
    // get SQ lock
    acquire_lock(&mlx5_sq.lock);
    // poll until we have enough WQEBB for all lanes using this QP
    mlx5_poll_cq_until(wf_info.num_pe_group_lanes);
  }

  uint64_t* atomic_laddr = nonfetching_atomic;
  uint32_t atomic_lkey = nonfetching_atomic_lkey;
  if (fetching) {
    uint32_t atomic_idx = (fetching_atomic_idx + wf_info.pe_group_logical_lane_id) % FETCHING_ATOMIC_CNT;
    atomic_laddr = &fetching_atomic[atomic_idx];
    atomic_lkey = fetching_atomic_lkey;
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_wqe_idx(mlx5_sq, wf_info.pe_group_logical_lane_id);
  uint16_t sq_idx = mlx5_sq_idx(mlx5_sq, wqe_idx);

  uint8_t fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
  if (fence) fm_ce_se |= MLX5_WQE_CTRL_FENCE;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, fm_ce_se,
                   raddr, byteswap<uint32_t>(rkey),
                   static_cast<uint64_t>(atomic_data), static_cast<uint64_t>(atomic_cmp),
                   reinterpret_cast<uintptr_t>(atomic_laddr), byteswap<uint32_t>(atomic_lkey)};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    // increment post and fetching atomic counters
    mlx5_sq.post += wf_info.num_pe_group_lanes;
    if (fetching) {
      fetching_atomic_idx += wf_info.num_pe_group_lanes;
    }
    // we are the last thread in the wavefront, so we have the last WQE posted
    mlx5_ring_doorbell(mlx5_sq.post, wqe);
    // release SQ lock
    release_lock(&mlx5_sq.lock);
    // wait until fetch completes
    if (fetching) {
      mlx5_quiet_single();
    }
  }

  return fetching ? *atomic_laddr : 0;
}

// precondition: called with all active lanes using different QPs
__device__ uint64_t QueuePair::mlx5_post_wqe_amo_single(uintptr_t raddr,
                                                        uint32_t rkey, uint8_t opcode,
                                                        int64_t atomic_data, int64_t atomic_cmp,
                                                        bool fetching, bool fence) {
  // get SQ lock
  acquire_lock(&mlx5_sq.lock);
  // poll until we have enough space for at least one WQE
  mlx5_poll_cq_until(1);

  uint64_t* atomic_laddr = nonfetching_atomic;
  uint32_t atomic_lkey = nonfetching_atomic_lkey;
  if (fetching) {
    uint32_t atomic_idx = fetching_atomic_idx % FETCHING_ATOMIC_CNT;
    atomic_laddr = &fetching_atomic[atomic_idx];
    atomic_lkey = fetching_atomic_lkey;
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_wqe_idx(mlx5_sq, 0);
  uint16_t sq_idx = mlx5_sq_idx(mlx5_sq, wqe_idx);

  uint8_t fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;
  if (fence) fm_ce_se |= MLX5_WQE_CTRL_FENCE;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, fm_ce_se,
                   raddr, byteswap<uint32_t>(rkey),
                   static_cast<uint64_t>(atomic_data), static_cast<uint64_t>(atomic_cmp),
                   reinterpret_cast<uintptr_t>(atomic_laddr), byteswap<uint32_t>(atomic_lkey)};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  // increment post counter and fetching-atomic counters
  mlx5_sq.post += 1;
  if (fetching) {
    fetching_atomic_idx += 1;
  }
  // ring doorbell for this WQE
  mlx5_ring_doorbell(mlx5_sq.post, wqe);
  // release SQ lock
  release_lock(&mlx5_sq.lock);
  // wait until fetch completes
  if (fetching) {
    mlx5_quiet_single();
  }

  return fetching ? *atomic_laddr : 0;
}

}  // namespace rocshmem
