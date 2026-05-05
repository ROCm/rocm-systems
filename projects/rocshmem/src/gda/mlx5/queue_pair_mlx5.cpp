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

#define MLX5_LOCK_USE_S_SLEEP  0
#define MLX5_LOCK_USE_S_WAKEUP (0 && MLX5_LOCK_USE_S_SLEEP)

#if MLX5_LOCK_USE_S_SLEEP
// sleep for up to 64 * MLX5_LOCK_S_SLEEP_DELAY clock cycles
static constexpr int MLX5_LOCK_S_SLEEP_DELAY = 2;
#endif

#if MLX5_LOCK_USE_S_WAKEUP
__device__ static inline void amdgcn_s_wakeup() {
  /* why doesn't __builtin_amdgcn_s_wakeup() exist?
   * signals other wavefronts in the same workgroup to exit early from s_sleep */
  asm volatile("s_wakeup");
}
#endif

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

/* precondition: called with all active lanes using different QPs
 * wait until requested_idx number of WQEs have been completed */
__device__ void QueuePair::mlx5_poll_cq_until(uint64_t requested_idx) {
  uint64_t complete_idx = __hip_atomic_load(&mlx5_sq.complete_idx,
                                            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  /* complete_idx is an index to the next free WQE i.e. counts number of completed WQEs
   * requested_idx is the complete_idx you need to observe before returning */
  while (complete_idx < requested_idx) {
    struct mlx5_cqe64* cqe = mlx5_cq.buf;

    // lowered to s_waitcnt vmcnt(0) : order ld/rmw complete_idx -> ld wqe_counter
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    /* read wqe_counter and sig_op_own from CQE
     * 32-bit load: big-endian 16-bit field, then two 8-bit fields
     * wqe_counter is the sq_wqebb_counter; all our WQEs are exactly one WQEBB (64B) */
    uint32_t wqecnt_sig_op_own = __hip_atomic_load(reinterpret_cast<uint32_t*>(&cqe->wqe_counter),
                                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);

    /* GPU is little-endian, so op_own is loaded into the top half of wqecnt_sig_op_own;
     * opcode is the top 4 bits of op_own */
    uint8_t opcode = static_cast<uint8_t>(wqecnt_sig_op_own >> 28);
    // CQEs are initially invalid, retry until we see a valid CQE
    if (opcode == MLX5_CQE_INVALID) {
      LOGD_TRACE("CQ: invalid completion (%x)", opcode);
      continue;
    }

#if defined(BUILD_DEBUG_DEVICE)
    if (opcode != MLX5_CQE_REQ) {
      mlx5_print_cqe_error(cqe, opcode);
    }
#endif

    // GPU is little-endian, so wqe_counter is loaded into the low half of wqecnt_sig_op_own
    __be16 be_wqe_counter = static_cast<__be16>(wqecnt_sig_op_own);
    uint16_t wqe_counter = endian::from_be(be_wqe_counter);
    // wqe_counter is an index to the *last* completed WQE - need to add one to get *count* of completed WQEs
    uint16_t complete_idx16 = wqe_counter + 1;

    /**
     * complete_idx16_diff <= sq_depth
     * except when many more WQEs get committed & executed while inside poll_cq_until
     * but in that case complete_idx is necessarily updated since we need to call poll_cq_until
     * to ensure that the SQ slots are free and we catch it
     *
     * NOTE: does this need __ATOMIC_SEQ_CST to be correct?
     *   i.e. can this thread observe old complete_idx and updated cqe->wqe_counter
     *   while NIC observes updated complete_idx?
     */
    uint16_t complete_idx16_diff = complete_idx16 - static_cast<uint16_t>(complete_idx);
    uint64_t next_complete_idx   = complete_idx   + static_cast<uint64_t>(complete_idx16_diff);

    // lowered to s_waitcnt vmcnt(0) : order ld wqe_counter -> rmw complete_idx
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    /* accumulate the newly-complete indices into mlx5_sq.complete_idx,
     * if some other thread didn't get there first */
    if (__hip_atomic_compare_exchange_weak(&mlx5_sq.complete_idx, &complete_idx, next_complete_idx,
                                           __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT)) {
      complete_idx = next_complete_idx;
    }
  }
}

// precondition: called with all active lanes using different QPs
__device__ void QueuePair::mlx5_quiet() {
  // check how many WQEs have been posted so far
  uint64_t commit_idx = __hip_atomic_load(&mlx5_sq.commit_idx,
                                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  // poll until we have completed all these WQEs
  mlx5_poll_cq_until(commit_idx);
  // system-scope acquire fence, to ensure we see updated get or AMO data
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

/**
 * TODO: This function is redundant but kept because ionic has a different
 * quiet_single implementation. Remove once ionic's quiet is unified.
 */
__device__ void QueuePair::mlx5_quiet_single() {
  // check how many WQEs have been posted so far
  uint64_t commit_idx = __hip_atomic_load(&mlx5_sq.commit_idx,
                                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  // poll until we have completed all these WQEs
  mlx5_poll_cq_until(commit_idx);
  // system-scope acquire fence, to ensure we see updated get or AMO data
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

// can be called with all active lanes using any number of different QPs, don't assume anything
__device__ void QueuePair::mlx5_post_wqe_rma(int32_t length, uintptr_t laddr, uintptr_t raddr,
                                             uint8_t opcode, ActiveWFInfo &wf_info) {
  uint64_t reserve_idx_base = 0;
  uint64_t reserve_count = 0;
  if (wf_info.is_pe_group_last) {
    // reserve wf_info.num_pe_group_lanes slots in SQ
    reserve_idx_base = __hip_atomic_fetch_add(&mlx5_sq.reserve_idx, wf_info.num_pe_group_lanes,
                                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    /* wait until all indices are available in SQ
     * [reserve_idx_base ... reserve_idx_base + wf_info.num_pe_group_lanes - 1] % sq_depth */
    reserve_count = reserve_idx_base + wf_info.num_pe_group_lanes;
    // can skip checking CQ for first sq_depth WQEs
    uint64_t sq_depth = static_cast<uint64_t>(mlx5_sq.depth);
    if (reserve_count > sq_depth) {
      mlx5_poll_cq_until(reserve_count - sq_depth);
    }
  }
  // fetch reservation base index from PE group leader
  reserve_idx_base = __shfl(reserve_idx_base, wf_info.pe_group_last_phys_lane_id);
  // compute this thread's reserved index
  uint64_t reserve_idx = reserve_idx_base + wf_info.pe_group_logical_lane_id;

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = static_cast<uint16_t>(reserve_idx);
  uint16_t sq_idx = mlx5_sq_idx(mlx5_sq, wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = gda_mlx5_wqe_rma::can_inline(opcode, length, inline_threshold);

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey, laddr, lkey, static_cast<uint32_t>(length), send_inline};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    // wait until our turn to ring doorbell
    while (__hip_atomic_load(&mlx5_sq.commit_idx,
                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != reserve_idx_base) {
#if MLX5_LOCK_USE_S_SLEEP
      // sleep so we don't hammer the memory
      __builtin_amdgcn_s_sleep(MLX5_LOCK_S_SLEEP_DELAY);
#endif
      continue;
    }
    // we are the last thread in the wavefront, so we have the last WQE posted
    mlx5_ring_doorbell(reserve_count, wqe);
    // increment commit index and release any other waiting waves
    __hip_atomic_fetch_add(&mlx5_sq.commit_idx, wf_info.num_pe_group_lanes,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#if MLX5_LOCK_USE_S_WAKEUP
    // wake up any other sleeping waves (in the same workgroup)
    amdgcn_s_wakeup();
#endif
  }
}

// precondition: called with all active lanes using different QPs
__device__ void QueuePair::mlx5_post_wqe_rma_single(int32_t length, uintptr_t laddr, uintptr_t raddr,
                                                    uint8_t opcode, bool ring_db) {
  // reserve 1 slot in SQ
  uint64_t reserve_idx = __hip_atomic_fetch_add(&mlx5_sq.reserve_idx, 1,
                                                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  uint64_t reserve_count = reserve_idx + 1;
  /* wait until index reserve_idx % sq_depth is available in SQ
   * can skip checking CQ for first sq_depth WQEs */
  uint64_t sq_depth = static_cast<uint64_t>(mlx5_sq.depth);
  if (reserve_count > sq_depth) {
    mlx5_poll_cq_until(reserve_count - sq_depth);
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = static_cast<uint16_t>(reserve_idx);
  uint16_t sq_idx = mlx5_sq_idx(mlx5_sq, wqe_idx);

  // can we inline the data into the WQE?
  bool send_inline = gda_mlx5_wqe_rma::can_inline(opcode, length, inline_threshold);

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey, laddr, lkey, static_cast<uint32_t>(length), send_inline};

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  // wait until our turn to ring doorbell
  while (__hip_atomic_load(&mlx5_sq.commit_idx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != reserve_idx) {
#if MLX5_LOCK_USE_S_SLEEP
    // sleep so we don't hammer the memory
    __builtin_amdgcn_s_sleep(MLX5_LOCK_S_SLEEP_DELAY);
#endif
    continue;
  }
  // ring doorbell for this WQE
  if (ring_db) {
    mlx5_ring_doorbell(reserve_count, wqe);
  }
  // increment commit index and release any other waiting waves
  __hip_atomic_fetch_add(&mlx5_sq.commit_idx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#if MLX5_LOCK_USE_S_WAKEUP
  // wake up any other sleeping waves (in the same workgroup)
  amdgcn_s_wakeup();
#endif
}

/* can be called with all active lanes using any number of different QPs, don't assume anything
 * assumes that `fetching' is constant across all lanes using the same QP
 * TODO: make `fetching' a template parameter */
__device__ uint64_t QueuePair::mlx5_post_wqe_amo([[maybe_unused]] int32_t length,
                                                 uintptr_t raddr, uint8_t opcode,
                                                 int64_t atomic_data, int64_t atomic_cmp,
                                                 bool fetching, ActiveWFInfo &wf_info) {
  uint64_t reserve_idx_base = 0;
  uint64_t reserve_count = 0;
  if (wf_info.is_pe_group_last) {
    // reserve wf_info.num_pe_group_lanes slots in SQ
    reserve_idx_base = __hip_atomic_fetch_add(&mlx5_sq.reserve_idx, wf_info.num_pe_group_lanes,
                                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    /* wait until all indices are available in SQ
     * [reserve_idx_base ... reserve_idx_base + wf_info.num_pe_group_lanes - 1] % sq_depth */
    reserve_count = reserve_idx_base + wf_info.num_pe_group_lanes;
    // can skip checking CQ for first sq_depth WQEs
    uint64_t sq_depth = static_cast<uint64_t>(mlx5_sq.depth);
    if (reserve_count > sq_depth) {
      mlx5_poll_cq_until(reserve_count - sq_depth);
    }
  }
  // fetch reservation base index from PE group leader
  reserve_idx_base = __shfl(reserve_idx_base, wf_info.pe_group_last_phys_lane_id);
  // compute this thread's reserved index
  uint64_t reserve_idx = reserve_idx_base + wf_info.pe_group_logical_lane_id;

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = static_cast<uint16_t>(reserve_idx);
  uint16_t sq_idx = mlx5_sq_idx(mlx5_sq, wqe_idx);

  uint64_t* atomic_laddr = nonfetching_atomic;
  uint32_t  atomic_lkey  = nonfetching_atomic_lkey;
  if (fetching) {
    /* assumes the FETCHING_ATOMIC_CNT >= sq_depth; this is not generally true
     * TODO: fix this for the case where there are > 1024 concurrent AMOs */
    uint64_t atomic_idx = reserve_idx % FETCHING_ATOMIC_CNT;
    atomic_laddr = &fetching_atomic[atomic_idx];
    atomic_lkey  = fetching_atomic_lkey;
  }

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey,
                   static_cast<uint64_t>(atomic_data), static_cast<uint64_t>(atomic_cmp),
                   reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};
  uint64_t ret_val = 0;

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  if (wf_info.is_pe_group_last) {
    // wait until our turn to ring doorbell
    while (__hip_atomic_load(&mlx5_sq.commit_idx,
                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != reserve_idx_base) {
#if MLX5_LOCK_USE_S_SLEEP
      // sleep so we don't hammer the memory
      __builtin_amdgcn_s_sleep(MLX5_LOCK_S_SLEEP_DELAY);
#endif
      continue;
    }
    // we are the last thread in the wavefront, so we have the last WQE posted
    mlx5_ring_doorbell(reserve_count, wqe);
    // increment commit index and release any other waiting waves
    __hip_atomic_fetch_add(&mlx5_sq.commit_idx, wf_info.num_pe_group_lanes,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#if MLX5_LOCK_USE_S_WAKEUP
    // wake up any other sleeping waves (in the same workgroup)
    amdgcn_s_wakeup();
#endif
    // wait until leader's fetch completes; completion order ensures others are complete as well
    if (fetching) {
      mlx5_poll_cq_until(reserve_count);
    }
  }

  if (fetching) {
    // system-scope (cache-bypassing) load of AMO return data
    ret_val = __hip_atomic_load(atomic_laddr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  return ret_val;
}

// precondition: called with all active lanes using different QPs
__device__ uint64_t QueuePair::mlx5_post_wqe_amo_single([[maybe_unused]] int32_t length,
                                                        uintptr_t raddr, uint8_t opcode,
                                                        int64_t atomic_data, int64_t atomic_cmp,
                                                        bool fetching) {
  // reserve 1 slot in SQ
  uint64_t reserve_idx = __hip_atomic_fetch_add(&mlx5_sq.reserve_idx, 1,
                                                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  uint64_t reserve_count = reserve_idx + 1;
  /* wait until index reserve_idx % sq_depth is available in SQ
   * can skip checking CQ for first sq_depth WQEs */
  uint64_t sq_depth = static_cast<uint64_t>(mlx5_sq.depth);
  if (reserve_count > sq_depth) {
    mlx5_poll_cq_until(reserve_count - sq_depth);
  }

  // wqe_idx is the logical WQE id that wraps at 0xFFFF, sq_idx is the index into the actual SQ
  uint16_t wqe_idx = static_cast<uint16_t>(reserve_idx);
  uint16_t sq_idx = mlx5_sq_idx(mlx5_sq, wqe_idx);

  uint64_t* atomic_laddr = nonfetching_atomic;
  uint32_t atomic_lkey = nonfetching_atomic_lkey;
  if (fetching) {
    /* assumes the FETCHING_ATOMIC_CNT >= sq_depth; this is not generally true
     * TODO: fix this for the case where there are > 1024 concurrent AMOs */
    uint64_t atomic_idx = reserve_idx % FETCHING_ATOMIC_CNT;
    atomic_laddr = &fetching_atomic[atomic_idx];
    atomic_lkey  = fetching_atomic_lkey;
  }

  // construct the WQE on the stack
  gda_mlx5_wqe wqe{wqe_idx, opcode, qp_num, MLX5_WQE_CTRL_CQ_UPDATE,
                   raddr, rkey,
                   static_cast<uint64_t>(atomic_data), static_cast<uint64_t>(atomic_cmp),
                   reinterpret_cast<uintptr_t>(atomic_laddr), atomic_lkey};
  uint64_t ret_val = 0;

  // copy to SQ
  mlx5_sq.buf[sq_idx] = wqe;

  // wait until our turn to ring doorbell
  while (__hip_atomic_load(&mlx5_sq.commit_idx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != reserve_idx) {
#if MLX5_LOCK_USE_S_SLEEP
    // sleep so we don't hammer the memory
    __builtin_amdgcn_s_sleep(MLX5_LOCK_S_SLEEP_DELAY);
#endif
    continue;
  }
  // ring doorbell for this WQE
  mlx5_ring_doorbell(reserve_count, wqe);
  // increment commit index and release any other waiting waves
  __hip_atomic_fetch_add(&mlx5_sq.commit_idx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#if MLX5_LOCK_USE_S_WAKEUP
  // wake up any other sleeping waves (in the same workgroup)
  amdgcn_s_wakeup();
#endif
  // wait until fetch completes
  if (fetching) {
    mlx5_poll_cq_until(reserve_count);
    // system-scope (cache-bypassing) load of AMO return data
    ret_val = __hip_atomic_load(atomic_laddr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  return ret_val;
}

}  // namespace rocshmem
