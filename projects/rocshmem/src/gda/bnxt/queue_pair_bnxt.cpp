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

#include "gda/bnxt/queue_pair_bnxt.hpp"

namespace rocshmem {

__host__ QueuePairBNXT::QueuePairBNXT(uint32_t qpn, uintptr_t heap_laddr, uint32_t heap_lkey,
                                      uintptr_t heap_raddr, uint32_t heap_rkey, size_t heap_size,
                                      const QpSymmEntry *symm_entries, const int *symm_count,
                                      struct ibv_pd* pd,
                                      uint64_t* dbr, bnxt_device_sq&& sq, bnxt_device_cq&& cq)
  : QueuePairBase{qpn, heap_laddr, heap_lkey, heap_raddr, heap_rkey, heap_size,
                  symm_entries, symm_count, pd},
    dbr{dbr}, sq{std::move(sq)}, cq{std::move(cq)} {
}

__device__ static inline struct bnxt_re_msns* bnxt_re_pull_psn_buff(const bnxt_device_sq& sq) {
  return (struct bnxt_re_msns*)(((char *) sq.msntbl) + ((sq.msn) << sq.psn_sz_log2));
}

__device__ static inline uint64_t bnxt_re_update_msn_tbl(uint32_t st_idx, uint32_t npsn,
                                                         uint32_t start_psn) {
   return ((((uint64_t)(st_idx) << BNXT_RE_SQ_MSN_SEARCH_START_IDX_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_START_IDX_MASK) |
                       (((uint64_t)(npsn) << BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_MASK) |
                       (((start_psn) << BNXT_RE_SQ_MSN_SEARCH_START_PSN_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_START_PSN_MASK));
}

__device__ void QueuePairBNXT::fill_psns_for_msntbl(bnxt_device_sq& sq, uint32_t msg_len) {
   uint32_t npsn = 0, start_psn = 0, next_psn = 0;
   struct bnxt_re_msns msns;
   uint64_t *msns_ptr;
   uint32_t pkt_cnt = 0;
   /* Start slot index of the WQE */
   uint32_t st_idx = sq.tail; // * BNXT_RE_STATIC_WQE_SIZE_SLOTS; Do we need this?
   // Get the MSN table address
   msns_ptr = (uint64_t *)bnxt_re_pull_psn_buff(sq);
   // Start PSN is the last recorded PSN
   // Calculate the packet count based on the len of the WQE/MTU
   msns.start_idx_next_psn_start_psn = 0;
   start_psn = sq.psn;
   pkt_cnt = (msg_len / sq.mtu);

   if (msg_len % sq.mtu)
       pkt_cnt++;

   /* Increment the psn even for 0 len packets
    * e.g. for opcode rdma-write-with-imm-data
    * with length field = 0
    */
   if (msg_len == 0)
       pkt_cnt = 1;

   /* make it 24 bit */
   next_psn = sq.psn + pkt_cnt;
   npsn = next_psn;
   sq.psn = next_psn;
   msns.start_idx_next_psn_start_psn |= bnxt_re_update_msn_tbl(st_idx, npsn, start_psn);
   sq.msn++;
   sq.msn %= sq.msn_tbl_sz;

   memcpy(msns_ptr, &msns, sizeof(uint64_t));
}

__device__ void QueuePairBNXT::incr_tail(bnxt_device_sq& sq, uint8_t cnt) {
  sq.tail += cnt;
  if (sq.tail >= sq.depth) {
    sq.tail %= sq.depth;
    /* Rolled over, Toggle Tail bit in epoch flags */
    sq.flags ^= 1UL << BNXT_RE_FLAG_EPOCH_TAIL_SHIFT;
  }
}

__device__ void* QueuePairBNXT::get_hwqe(const bnxt_device_sq& sq, uint32_t idx) {
  idx += sq.tail;
  if (idx >= sq.depth)
    idx -= sq.depth;
  return (void *)((char*)sq.buf + (idx << 4));
}

__device__ void QueuePairBNXT::acquire_lock(uint32_t* lock) {
  uint32_t expected;

  do {
    expected = 0;
  } while (0 == __hip_atomic_compare_exchange_strong(lock, &expected, 1,
                                                     __ATOMIC_ACQUIRE,
                                                     __ATOMIC_ACQUIRE,
                                                     __HIP_MEMORY_SCOPE_SYSTEM));
}

__device__ void QueuePairBNXT::release_lock(uint32_t* lock) {
  __hip_atomic_store(lock, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ void QueuePairBNXT::ring_doorbell(uint32_t slot_idx) {
  struct bnxt_re_db_hdr hdr;
  uint32_t epoch;
  uint64_t key_lo;
  uint64_t key_hi;

  epoch = (sq.flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;

  key_lo = (slot_idx | epoch);

  key_hi = (qp_num & BNXT_RE_DB_QID_MASK)
         | (((uint64_t) BNXT_RE_QUE_TYPE_SQ & BNXT_RE_DB_TYP_MASK) << BNXT_RE_DB_TYP_SHIFT)
         | (0x1UL << BNXT_RE_DB_VALID_SHIFT);

  hdr.typ_qid_indx = (key_lo | (key_hi << 32));

  __threadfence_system();
  __hip_atomic_store(dbr, hdr.typ_qid_indx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

#if defined(BUILD_DEBUG_DEVICE)
__device__ __attribute__((noinline)) void QueuePairBNXT::print_cqe_error(uint8_t status) {
  switch (status) {
  case BNXT_RE_REQ_ST_BAD_RESP:
    LOGD_ERROR_ABORT("CQ error BAD_RESP (%x)", status);
    break;
  case BNXT_RE_REQ_ST_LOC_LEN:
    LOGD_ERROR_ABORT("CQ error LOC_LEN (%x)", status);
    break;
  case BNXT_RE_REQ_ST_LOC_QP_OP:
    LOGD_ERROR_ABORT("CQ error LOC_QP_OP (%x)", status);
    break;
  case BNXT_RE_REQ_ST_PROT:
    LOGD_ERROR_ABORT("CQ error PROT (%x)", status);
    break;
  case BNXT_RE_REQ_ST_MEM_OP:
    LOGD_ERROR_ABORT("CQ error MEM_OP (%x)", status);
    break;
  case BNXT_RE_REQ_ST_REM_INVAL:
    LOGD_ERROR_ABORT("CQ error REM_INVAL (%x)", status);
    break;
  case BNXT_RE_REQ_ST_REM_ACC:
    LOGD_ERROR_ABORT("CQ error REM_ACC (%x)", status);
    break;
  case BNXT_RE_REQ_ST_REM_OP:
    LOGD_ERROR_ABORT("CQ error REM_OP (%x)", status);
    break;
  case BNXT_RE_REQ_ST_RNR_NAK_XCED:
    LOGD_ERROR_ABORT("CQ error RNR_NAK_XCED (%x)", status);
    break;
  case BNXT_RE_REQ_ST_TRNSP_XCED:
    LOGD_ERROR_ABORT("CQ error TRNSP_XCED (%x)", status);
    break;
  case BNXT_RE_REQ_ST_WR_FLUSH:
    LOGD_ERROR_ABORT("CQ error WR_FLUSH (%x)", status);
    break;
  default:
    LOGD_ERROR_ABORT("CQ error unknown status (%x)", status);
    break;
  }
}
#endif  // BUILD_DEBUG_DEVICE

__device__ void QueuePairBNXT::poll_cq_until(uint32_t requested_available_slots) {
  struct bnxt_re_req_cqe *cqe;
  uint32_t sq_tail;
  uint32_t sq_head;
  uint32_t sq_depth;
  uint32_t consumed_slots;
  uint32_t available_slots;

  sq_depth = sq.depth;

  do {
    cqe = (struct bnxt_re_req_cqe *) cq.buf;

#ifdef BUILD_DEBUG_DEVICE
    uint32_t flg_val =
        __hip_atomic_load(static_cast<uint32_t*>(
            __builtin_assume_aligned((char*)cqe + sizeof(struct bnxt_re_req_cqe)
                                                + offsetof(struct bnxt_re_bcqe, flg_st_typ_ph),
                                     4)),
            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    uint8_t status = (flg_val >> BNXT_RE_BCQE_STATUS_SHIFT) & BNXT_RE_BCQE_STATUS_MASK;
    if (status != BNXT_RE_REQ_ST_OK) {
      print_cqe_error(status);
    }
#endif

    /* Update the SQ head
     * This param provides us the wqe_idx but we need to convert to the slot idx.
     * We assume a static slots size of GDA_BNXT_WQE_SLOT_COUNT thus can multiply by this value */
    sq_head = (((cqe->con_indx & 0xFFFF) * GDA_BNXT_WQE_SLOT_COUNT) % sq_depth);
    sq.head = sq_head;

    sq_tail = __hip_atomic_load(&sq.tail, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);

    consumed_slots  = (sq_tail - sq_head + sq_depth) % sq_depth;
    available_slots = sq_depth - consumed_slots;
  } while (available_slots < requested_available_slots);
}

__device__ __noinline__ void QueuePairBNXT::quiet_single() {
  poll_cq_until(sq.depth);
}

}  // namespace rocshmem
