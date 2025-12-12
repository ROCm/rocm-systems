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

__device__ static inline void amdgcn_s_wakeup() {
  /* why doesn't __builtin_amdgcn_s_wakeup() exist?
   * signals other wavefronts in the same workgroup to exit early from s_sleep */
  asm volatile("s_wakeup");
}

__device__ static inline void acquire_lock(uint32_t *lock) {
  /* acquire lock when 1 (locked) is exchanged with 0 (unlocked)
   *
   * the __ATOMIC_ACQUIRE load synchronizes with the __ATOMIC_RELEASE store in release_lock(),
   * but not with the (implicit) __ATOMIC_RELAXED store part of the exchange
   * this is fine, since we only need to ensure happens-before between the threads
   * that released and acquired the lock, not between the different threads contending on the lock
   * when they (eventually) acquire the lock, *then* they will synchronize */
  while (__hip_atomic_exchange(lock, 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) {
    // sleep for 65-128 cycles so we don't hammer the memory
    __builtin_amdgcn_s_sleep(2);
  }
}

__device__ static inline void release_lock(uint32_t *lock) {
  // release lock by storing 0 (unlocked)
  __hip_atomic_store(lock, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#if 0
  // wake up any other sleeping waves (in the same workgroup)
  amdgcn_s_wakeup();
#endif
}

__device__ static inline uint16_t mlx5_wqe_idx(const gda_mlx5_device_sq& sq, uint8_t lane_id) {
  return sq.tail + lane_id;
}

__device__ void QueuePair::mlx5_ring_doorbell(uint16_t sq_wqebb_counter, const gda_mlx5_wqe& wqe) {
#if 0
  /* yes, the HCA is big-endian, so this looks wrong
   * however, these are technically two 32-bit fields and they are already stored as BE
   * so storing db_val from as LE will store the (BE swapped) opmod_idx_opcode field first
   * and then the (BE swapped) qpn_ds field second, which is what we want */
  uint64_t db_val = (static_cast<uint64_t>(wqe.ctrl.qpn_ds) << 32) |
                     static_cast<uint64_t>(wqe.ctrl.opmod_idx_opcode);
#endif
  //gda_mlx5_db_register db_val{wqe.ctrl.opmod_idx_opcode, wqe.ctrl.qpn_ds};
  gda_mlx5_db_register db_val{wqe};
  __be32 be_sq_wqebb_counter = endian::to_be<uint32_t>(sq_wqebb_counter);

#if 0
  // get pointer to doorbell register in current BlueFlame buffer
  uint64_t* db = __hip_atomic_load(&mlx5_sq.db, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  do {
    // swap BlueFlame buffers: each buffer is 256B, so the pointers are offset by 0x100
    uintptr_t db_uintptr = reinterpret_cast<uintptr_t>(db);
    db_uintptr ^= 0x100UL;
    uint64_t* db_next = reinterpret_cast<uint64_t*>(db_uintptr);
    // I don't think this cmpxch can ever fail since we hold the SQ lock, but let's be safe
  } while (!__hip_atomic_compare_exchange_weak(&mlx5_db, &mlx5_sq.db, db_next,
                                               __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
                                               __HIP_MEMORY_SCOPE_AGENT));
#endif
#if 0
  // get pointer to doorbell register in current BlueFlame buffer
  uint64_t* db = mlx5_sq.db;
  // swap BlueFlame buffers: each buffer is 256B, so the pointers are offset by 0x100
  mlx5_sq.db = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(db) ^ 0x100UL);
#endif
  gda_mlx5_bf_buffer* bf = mlx5_sq.swap_bf_buffer();

  // store sq_wqebb_counter to doorbell record
  __hip_atomic_store(mlx5_sq.dbrec, be_sq_wqebb_counter, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  // ring doorbell by storing first 8B of WQE to the doorbell register
#if 0
  __hip_atomic_store(db, db_val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
  __hip_atomic_store(&bf->db_reg, db_val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ void QueuePair::mlx5_check_cqe_error(const mlx5_cqe64* cqe) {
  const mlx5_err_cqe* err_cqe = reinterpret_cast<const mlx5_err_cqe*>(cqe);
  const char* cqe_syndrome_string = "";
  uint8_t syndrome = 0x0;

  uint8_t op_own = __hip_atomic_load(&cqe->op_own, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  uint8_t owner = op_own & MLX5_CQE_OWNER_MASK;
  uint8_t opcode = op_own >> 4;

  switch (opcode) {
  case MLX5_CQE_REQ:
    // everything okay
    return;
  case MLX5_CQE_RESP_WR_IMM:
  case MLX5_CQE_RESP_SEND:
  case MLX5_CQE_RESP_SEND_IMM:
  case MLX5_CQE_RESP_SEND_INV:
    // (valid) responder completion?!
    printf("CQ: unexpected responder completion (%x)\n", opcode);
    break;
  case MLX5_CQE_RESIZE_CQ:
  case MLX5_CQE_NO_PACKET:
    printf("CQ: unexpected completion type (%x)\n", opcode);
    break;
  case MLX5_CQE_SIG_ERR:
    printf("CQ: unexpected signature error (%x)\n", opcode);
    break;
  case MLX5_CQE_REQ_ERR:
    syndrome = __hip_atomic_load(&err_cqe->syndrome, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    switch (syndrome) {
    case MLX5_CQE_SYNDROME_LOCAL_LENGTH_ERR:
      cqe_syndrome_string = "LOCAL_LENGTH_ERR";
      break;
    case MLX5_CQE_SYNDROME_LOCAL_QP_OP_ERR:
      cqe_syndrome_string = "LOCAL_QP_OP_ERR";
      break;
    case MLX5_CQE_SYNDROME_LOCAL_PROT_ERR:
      cqe_syndrome_string = "LOCAL_PROT_ERR";
      break;
    case MLX5_CQE_SYNDROME_WR_FLUSH_ERR:
      cqe_syndrome_string = "WR_FLUSH_ERR";
      break;
    case MLX5_CQE_SYNDROME_MW_BIND_ERR:
      cqe_syndrome_string = "MW_BIND_ERR";
      break;
    case MLX5_CQE_SYNDROME_BAD_RESP_ERR:
      cqe_syndrome_string = "BAD_RESP_ERR";
      break;
    case MLX5_CQE_SYNDROME_LOCAL_ACCESS_ERR:
      cqe_syndrome_string = "LOCAL_ACCESS_ERR";
      break;
    case MLX5_CQE_SYNDROME_REMOTE_INVAL_REQ_ERR:
      cqe_syndrome_string = "REMOTE_INVAL_REQ_ERR";
      break;
    case MLX5_CQE_SYNDROME_REMOTE_ACCESS_ERR:
      cqe_syndrome_string = "REMOTE_ACCESS_ERR";
      break;
    case MLX5_CQE_SYNDROME_REMOTE_OP_ERR:
      cqe_syndrome_string = "REMOTE_OP_ERR";
      break;
    case MLX5_CQE_SYNDROME_TRANSPORT_RETRY_EXC_ERR:
      cqe_syndrome_string = "TRANSPORT_RETRY_EXC_ERR";
      break;
    case MLX5_CQE_SYNDROME_RNR_RETRY_EXC_ERR:
      cqe_syndrome_string = "RNR_RETRY_EXC_ERR";
      break;
    case MLX5_CQE_SYNDROME_REMOTE_ABORTED_ERR:
      cqe_syndrome_string = "REMOTE_ABORTED_ERR";
      break;
    default:
      cqe_syndrome_string = "unknown syndrome type";
      break;
    }
    printf("CQ requester error: %s (%x)\n", cqe_syndrome_string, syndrome);
    break;
  case MLX5_CQE_RESP_ERR:
    printf("CQ: unexpected responder error (%x)\n", opcode);
    break;
  case MLX5_CQE_INVALID:
    printf("CQ: invalid completion (%x), check owner bit = %u?\n", opcode, owner);
    break;
  default:
    printf("CQ: unknown completion type (%x)\n", opcode);
    break;
  }
  abort();
}

__device__ void QueuePair::mlx5_poll_cq_until(uint16_t requested_available_slots) {
  uint16_t consumed_slots;
  uint16_t available_slots;

  uint16_t sq_depth = mlx5_sq.depth;

  // CQ lock (possibly) not needed with CQ collapsing
  //acquire_lock(&mlx5_cq.lock);

  do {
    struct mlx5_cqe64* cqe = mlx5_cq.buf;

#ifdef DEBUG
    mlx5_check_cqe_error(cqe);
#endif

    /* Update the SQ head
     * This param provides us the sq_wqebb_counter; all our WQEs are exactly one WQEBB (64B) */
    __be16 be_wqe_counter = __hip_atomic_load(&cqe->wqe_counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    uint16_t sq_head = endian::from_be(be_wqe_counter);
    mlx5_sq.head = sq_head;

    /* do we not need to hold the SQ lock here?
     * what happens if one wavefront is in quiet() and another is is post_wqe_rma()?
     * sq.tail can only get updated by another wavefront while this one is holding cq.lock if:
     *   1. other thread was already holding sq.lock, and
     *   2. other thread had already drained CQ until it had enough WQEs for the active lanes, and
     *   3. other thread had not yet finished writing to the SQ and ringing the doorbell
     * I think this is fine as long as post_wqe_rma stores to sq.tail before the doorbell ring
     * maybe needs SEQ_CST?
     *
     * wavefront0       | wavefront1                   | HCA
     * -----------------|------------------------------|-----------------------------
     * store(sq.tail)   |                              |
     * atomic_store(db) |                              |
     *                  |                              | atomic_load(db)
     *                  |                              | process WQ
     *                  |                              | atomic_store(cqe.wqe_counter)
     *                  | atomic_load(cqe.wqe_counter) |
     *                  | load(sq.tail) [atomic?]      |
     */
    uint16_t sq_tail = __hip_atomic_load(&mlx5_sq.tail, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);

    /* sq_tail > sq_head, except when sq_tail has wrapped around 0xFFFF and sq_head hasn't
     * but sq_tail - sq_head is correct even when it wraps around
     * in some marginal cases it's maybe possible to see consumed_slots > sq_depth,
     * but in that case available_slots will be very large, > requested_available_slots,
     * and the loop will continue for another iteration */
    consumed_slots  = sq_tail - sq_head;
    available_slots = sq_depth - consumed_slots;
    // can put in __builtin_amdgcn_s_sleep(1) when fail?
  } while (available_slots < requested_available_slots);

  //release_lock(&mlx5_cq.lock);
}

__device__ void QueuePair::mlx5_quiet() {
  if (is_first_active_lane()) {
    mlx5_poll_cq_until(mlx5_sq.depth);
  }
}

__device__ void QueuePair::mlx5_quiet_single() {
  mlx5_poll_cq_until(mlx5_sq.depth);
}

// called with all active lanes communicating with the same PE, i.e. using the same SQ
__device__ void QueuePair::mlx5_post_wqe_rma(int pe, int32_t length, uintptr_t laddr, uintptr_t raddr, uint8_t opcode) {
  uint64_t active_lane_mask;
  uint8_t active_lane_count;
  uint8_t active_lane_id;

  active_lane_mask  = get_active_lane_mask();
  active_lane_count = get_active_lane_count(active_lane_mask);
  active_lane_id    = get_active_lane_num(active_lane_mask);

  /* since the leader needs to write the first 8 bytes of the LAST WQE to the doorbell register,
   * it's easier if the LAST thread is the leader; does this have any performance implications? */
  bool is_leader = (active_lane_id == active_lane_count - 1);

  if (is_leader) {
    // get SQ lock
    acquire_lock(&mlx5_sq.lock);
    // poll until we have enough WQEBB for all active lanes
    mlx5_poll_cq_until(active_lane_count);
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_wqe_idx(mlx5_sq, active_lane_id);
  uint16_t sq_idx = wqe_idx % mlx5_sq.depth;

  // construct the WQE on the stack
  bool send_inline = gda_mlx5_wqe_rma::can_inline(opcode, length, inline_threshold);

  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey, laddr, lkey, static_cast<uint32_t>(length), send_inline};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  if (is_leader) {
    // increment tail counter
    mlx5_sq.tail += active_lane_count;
    // we are the last thread in the wavefront, so we have the last WQE posted
    mlx5_ring_doorbell(mlx5_sq.tail, wqe);
    // release SQ lock
    release_lock(&mlx5_sq.lock);
  }
}

// called with all active lanes communicating with different PEs, i.e. using different SQs
__device__ void QueuePair::mlx5_post_wqe_rma_single(int pe, int32_t length, uintptr_t laddr,
                                                    uintptr_t raddr, uint8_t opcode) {
  // get SQ lock
  acquire_lock(&mlx5_sq.lock);
  // poll until we have enough space for at least one WQE
  mlx5_poll_cq_until(1);

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_sq.tail;
  uint16_t sq_idx = wqe_idx % mlx5_sq.depth;

  // construct the WQE on the stack
  bool send_inline = gda_mlx5_wqe_rma::can_inline(opcode, length, inline_threshold);
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey, laddr, lkey, static_cast<uint32_t>(length), send_inline};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  // increment tail counter
  mlx5_sq.tail += 1;
  // ring doorbell for this WQE (note: need to check this for correctness)
  mlx5_ring_doorbell(mlx5_sq.tail, wqe);
  // release SQ lock
  release_lock(&mlx5_sq.lock);
}

// called with all active lanes communicating with the same PE, i.e. using the same SQ
__device__ uint64_t QueuePair::mlx5_post_wqe_amo(int pe, int32_t length, uintptr_t raddr, uint8_t opcode,
                                                 int64_t atomic_data, int64_t atomic_cmp, bool fetching) {
  uint64_t active_lane_mask;
  uint8_t active_lane_count;
  uint8_t active_lane_id;

  active_lane_mask  = get_active_lane_mask();
  active_lane_count = get_active_lane_count(active_lane_mask);
  active_lane_id    = get_active_lane_num(active_lane_mask);

  /* since the leader needs to write the first 8 bytes of the LAST WQE to the doorbell register,
   * it's easier if the LAST thread is the leader; does this have any performance implications? */
  bool is_leader = (active_lane_id == active_lane_count - 1);

  if (is_leader) {
    // get SQ lock
    acquire_lock(&mlx5_sq.lock);
    // poll until we have enough WQEBB for all active lanes
    mlx5_poll_cq_until(active_lane_count);
  }

  uint64_t* atomic_laddr = nonfetching_atomic;
  uint32_t atomic_lkey = nonfetching_atomic_lkey;
  if (fetching) {
    uint32_t atomic_idx = (fetching_atomic_idx + active_lane_id) % FETCHING_ATOMIC_CNT;
    atomic_laddr = &fetching_atomic[atomic_idx];
    atomic_lkey = fetching_atomic_lkey;
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_wqe_idx(mlx5_sq, active_lane_id);
  uint16_t sq_idx = wqe_idx % mlx5_sq.depth;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey,
                   static_cast<uint64_t>(atomic_data), static_cast<uint64_t>(atomic_cmp),
                   reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  if (is_leader) {
    // increment tail and fetching atomic counters
    mlx5_sq.tail += active_lane_count;
    if (fetching) {
      fetching_atomic_idx += active_lane_count;
    }
    // we are the last thread in the wavefront, so we have the last WQE posted
    mlx5_ring_doorbell(mlx5_sq.tail, wqe);
    // release SQ lock
    release_lock(&mlx5_sq.lock);
  }

  if (fetching) {
    mlx5_quiet();
  }

  return fetching ? *atomic_laddr : 0;
}

}  // namespace rocshmem
