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
//  __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#ifdef DEBUG
  printf("SQ: posted WQEs with dbrec(%p)=%x (%hu), dbreg(%p)=%lx (%x, %x)\n",
         mlx5_sq.dbrec, be_sq_wqebb_counter, sq_wqebb_counter,
	 &bf->db_reg, db_val.val, db_val.wqe_header.opmod_idx_opcode, db_val.wqe_header.qpn_ds);
#endif
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
    // possibly OK
    return;
//    break;
  default:
    printf("CQ: unknown completion type (%x)\n", opcode);
    break;
  }
  abort();
}

template <typename T>
__device__ static inline bool hip_atomic_compare(T* obj, T* expected, int order, int scope) {
  T exp = *expected;
  T val = __hip_atomic_load(obj, order, scope);
  *expected = val;
  return val == exp;
}

__device__ void QueuePair::mlx5_poll_cq_until(uint16_t requested_available_slots) {
  uint16_t consumed_slots;
  uint16_t available_slots;

  uint16_t sq_depth = mlx5_sq.depth;

  // CQ lock (possibly) not needed with CQ collapsing
  //acquire_lock(&mlx5_cq.lock);

  uint64_t sq_post = __hip_atomic_load(&mlx5_sq.post, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  // don't need to check CQEs if we haven't ever filled SQ and there's enough space left
  if (sq_post <= static_cast<uint64_t>((sq_depth - requested_available_slots))) {
    return;
  }

  uint64_t sq_sig = __hip_atomic_load(&mlx5_sq.sig, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);

  do {
    /* sq_sig counts the number of posted signaled WQEs
     *  odd count -> CQ[0]
     * even count -> CQ[1]
     * so use CQ[1 - (sq_sig % 2)]
     * or equivalently CQ[(sq_sig + 1) % 2], etc.
     */
    struct mlx5_cqe64* cqe = &mlx5_cq.buf[1 - (sq_sig  % 2)];

    /* Update the SQ head
     * This param provides us the sq_wqebb_counter; all our WQEs are exactly one WQEBB (64B) */
//    __be16 be_wqe_counter = __hip_atomic_load(&cqe->wqe_counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    // 32-bit load: big-endian 16-bit field, then two 8-bit fields
    uint32_t wqecnt_sig_op_own = __hip_atomic_load(reinterpret_cast<uint32_t*>(&cqe->wqe_counter), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    // GPU is little-endian, so wqe_counter is loaded into the low half of wqecnt_sig_op_own
    __be16 be_wqe_counter = static_cast<__be16>(wqecnt_sig_op_own);
    // GPU is little-endian, so op_own is loaded into the top bits of wqecnt_sig_op_own; and opcode is the top 4 bits of that
    uint8_t opcode = wqecnt_sig_op_own >> 28;
    uint16_t sq_head = endian::from_be(be_wqe_counter);

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

    // this isn't quite right, but try for now
    if (opcode == MLX5_CQE_INVALID) {
#ifdef DEBUG
      printf("CQ: invalid completion (%x)\n", opcode);
#endif
      // reload sq_sig, we might need to look at the other CQE
      sq_sig = __hip_atomic_load(&mlx5_sq.sig, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
      continue;
    }

    mlx5_sq.head = sq_head;

#ifdef DEBUG
    mlx5_check_cqe_error(cqe);
#endif

    /* sq_tail is an index to the next free WQE i.e. counts number of posted WQEs
     * sq_head is an index to the *last* completed WQE - need to add one to get *count* of completed WQEs */
    uint16_t posted = sq_tail;
    uint16_t completed = sq_head + 1;

    /* posted >= completed, except when posted has wrapped around 0xFFFF and completed hasn't
     * but posted - completed is correct even when it wraps around
     * in some marginal cases it's maybe possible to see consumed_slots > sq_depth,
     * but in that case available_slots will be very large, > requested_available_slots,
     * and the loop will continue for another iteration */
    consumed_slots  = posted - completed;
    available_slots = sq_depth - consumed_slots;
    // can put in __builtin_amdgcn_s_sleep(1) when fail?
  } while (available_slots < requested_available_slots ||
           !hip_atomic_compare(&mlx5_sq.sig, &sq_sig, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));

  //release_lock(&mlx5_cq.lock);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
__device__ void QueuePair::mlx5_quiet() {
  uint64_t qp_lane_mask = get_same_qp_lane_mask();
  if (is_first_active_lane(qp_lane_mask)) {
    mlx5_poll_cq_until(mlx5_sq.depth);
  }
}

// called with all active lanes using different QPs
__device__ void QueuePair::mlx5_quiet_single() {
  mlx5_poll_cq_until(mlx5_sq.depth);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
__device__ void QueuePair::mlx5_post_wqe_rma(int pe, int32_t length, uintptr_t laddr, uintptr_t raddr, uint8_t opcode) {
  uint64_t qp_lane_mask;
  uint8_t qp_lane_count;
  uint8_t qp_lane_id;

  qp_lane_mask  = get_same_qp_lane_mask();
  qp_lane_count = get_active_lane_count(qp_lane_mask);
  qp_lane_id    = get_active_lane_num(qp_lane_mask);

  /* since the leader needs to write the first 8 bytes of the LAST WQE to the doorbell register,
   * it's easier if the LAST thread is the leader; does this have any performance implications? */
  bool is_leader = (qp_lane_id == qp_lane_count - 1);

  if (is_leader) {
    // get SQ lock
    acquire_lock(&mlx5_sq.lock);
    // poll until we have enough WQEBB for all lanes using this QP
    mlx5_poll_cq_until(qp_lane_count);
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_wqe_idx(mlx5_sq, qp_lane_id);
  uint16_t sq_idx = wqe_idx % mlx5_sq.depth;

  // can we inline the data into the WQE?
  bool send_inline = gda_mlx5_wqe_rma::can_inline(opcode, length, inline_threshold);
  // only generate CQEs for last WQE
  //uint8_t fm_ce_se = is_leader ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
  uint8_t fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, fm_ce_se,
                   raddr, rkey, laddr, lkey, static_cast<uint32_t>(length), send_inline};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  if (is_leader) {
    // increment tail counter
    mlx5_sq.tail += qp_lane_count;
    mlx5_sq.post += qp_lane_count;
    mlx5_sq.sig += qp_lane_count;
    //mlx5_sq.sig += 1;
    // we are the last thread in the wavefront, so we have the last WQE posted
    mlx5_ring_doorbell(mlx5_sq.tail, wqe);
    // release SQ lock
    release_lock(&mlx5_sq.lock);
  }
}

// called with all active lanes using different QPs
__device__ void QueuePair::mlx5_post_wqe_rma_single(int pe, int32_t length, uintptr_t laddr,
                                                    uintptr_t raddr, uint8_t opcode, bool ring_db) {
  // get SQ lock
  acquire_lock(&mlx5_sq.lock);
  // poll until we have enough space for at least one WQE
  mlx5_poll_cq_until(1);

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_sq.tail;
  uint16_t sq_idx = wqe_idx % mlx5_sq.depth;

  // can we inline the data into the WQE?
  bool send_inline = gda_mlx5_wqe_rma::can_inline(opcode, length, inline_threshold);

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey, laddr, lkey, static_cast<uint32_t>(length), send_inline};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  // increment tail counter
  mlx5_sq.tail += 1;
  mlx5_sq.post += 1;
  mlx5_sq.sig += 1;

  if (ring_db) {
    // ring doorbell for this WQE (note: need to check this for correctness)
    mlx5_ring_doorbell(mlx5_sq.tail, wqe);
  }

  // release SQ lock
  release_lock(&mlx5_sq.lock);
}

// can be called with all active lanes using any number of different QPs, don't assume anything
__device__ uint64_t QueuePair::mlx5_post_wqe_amo(int pe, int32_t length, uintptr_t raddr, uint8_t opcode,
                                                 int64_t atomic_data, int64_t atomic_cmp, bool fetching) {
  uint64_t qp_lane_mask;
  uint8_t qp_lane_count;
  uint8_t qp_lane_id;

  qp_lane_mask  = get_same_qp_lane_mask();
  qp_lane_count = get_active_lane_count(qp_lane_mask);
  qp_lane_id    = get_active_lane_num(qp_lane_mask);

  /* since the leader needs to write the first 8 bytes of the LAST WQE to the doorbell register,
   * it's easier if the LAST thread is the leader; does this have any performance implications? */
  bool is_leader = (qp_lane_id == qp_lane_count - 1);

  if (is_leader) {
    // get SQ lock
    acquire_lock(&mlx5_sq.lock);
    // poll until we have enough WQEBB for all lanes using this QP
    mlx5_poll_cq_until(qp_lane_count);
  }

  uint64_t* atomic_laddr = nonfetching_atomic;
  uint32_t atomic_lkey = nonfetching_atomic_lkey;
  if (fetching) {
    uint32_t atomic_idx = (fetching_atomic_idx + qp_lane_id) % FETCHING_ATOMIC_CNT;
    atomic_laddr = &fetching_atomic[atomic_idx];
    atomic_lkey = fetching_atomic_lkey;
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = mlx5_wqe_idx(mlx5_sq, qp_lane_id);
  uint16_t sq_idx = wqe_idx % mlx5_sq.depth;

  // only generate CQEs for last WQE
  //uint8_t fm_ce_se = is_leader ? MLX5_WQE_CTRL_CQ_UPDATE : 0;
  uint8_t fm_ce_se = MLX5_WQE_CTRL_CQ_UPDATE;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, fm_ce_se,
                   raddr, rkey,
                   static_cast<uint64_t>(atomic_data), static_cast<uint64_t>(atomic_cmp),
                   reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  if (is_leader) {
    // increment tail and fetching atomic counters
    mlx5_sq.tail += qp_lane_count;
    mlx5_sq.post += qp_lane_count;
    mlx5_sq.sig += qp_lane_count;
    //mlx5_sq.sig += 1;
    if (fetching) {
      fetching_atomic_idx += qp_lane_count;
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

// called with all active lanes using different QPs
__device__ uint64_t QueuePair::mlx5_post_wqe_amo_single(int pe, int32_t length, uintptr_t raddr, uint8_t opcode,
                                                        int64_t atomic_data, int64_t atomic_cmp, bool fetching) {
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
  uint16_t wqe_idx = mlx5_sq.tail;
  uint16_t sq_idx = wqe_idx % mlx5_sq.depth;

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey,
                   static_cast<uint64_t>(atomic_data), static_cast<uint64_t>(atomic_cmp),
                   reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  // increment tail counter
  mlx5_sq.tail += 1;
  mlx5_sq.post += 1;
  mlx5_sq.sig += 1;
  if (fetching) {
    fetching_atomic_idx += 1;
  }
  // ring doorbell for this WQE (note: need to check this for correctness)
  mlx5_ring_doorbell(mlx5_sq.tail, wqe);
  // release SQ lock
  release_lock(&mlx5_sq.lock);

  if (fetching) {
    mlx5_quiet_single();
  }

  return fetching ? *atomic_laddr : 0;
}

}  // namespace rocshmem
