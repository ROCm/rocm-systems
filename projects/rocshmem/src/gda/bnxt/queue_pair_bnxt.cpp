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

namespace rocshmem {

__device__ static inline struct bnxt_re_msns* bnxt_re_pull_psn_buff(struct bnxt_device_sq *sq) {
  return (struct bnxt_re_msns*)(((char *) sq->msntbl) + ((sq->msn) << sq->psn_sz_log2));
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

__device__ static inline void bnxt_re_fill_psns_for_msntbl(struct bnxt_device_sq *sq,
                                                           uint32_t msg_len) {
   uint32_t npsn = 0, start_psn = 0, next_psn = 0;
   struct bnxt_re_msns msns;
   uint64_t *msns_ptr;
   uint32_t pkt_cnt = 0;
   /* Start slot index of the WQE */
   uint32_t st_idx = sq->tail; // * BNXT_RE_STATIC_WQE_SIZE_SLOTS; Do we need this?
   // Get the MSN table address
   msns_ptr = (uint64_t *)bnxt_re_pull_psn_buff(sq);
   // Start PSN is the last recorded PSN
   // Calculate the packet count based on the len of the WQE/MTU
   msns.start_idx_next_psn_start_psn = 0;
   start_psn = sq->psn;
   pkt_cnt = (msg_len / sq->mtu);

   if (msg_len % sq->mtu)
       pkt_cnt++;

   /* Increment the psn even for 0 len packets
    * e.g. for opcode rdma-write-with-imm-data
    * with length field = 0
    */
   if (msg_len == 0)
       pkt_cnt = 1;

   /* make it 24 bit */
   next_psn = sq->psn + pkt_cnt;
   npsn = next_psn;
   sq->psn = next_psn;
   msns.start_idx_next_psn_start_psn |= bnxt_re_update_msn_tbl(st_idx, npsn, start_psn);
   sq->msn++;
   sq->msn %= sq->msn_tbl_sz;

   memcpy(msns_ptr, &msns, sizeof(uint64_t));
}

__device__ static inline void bnxt_re_incr_tail(struct bnxt_device_sq *sq, uint8_t cnt)
{
  sq->tail += cnt;
  if (sq->tail >= sq->depth) {
    sq->tail %= sq->depth;
    /* Rolled over, Toggle Tail bit in epoch flags */
    sq->flags ^= 1UL << BNXT_RE_FLAG_EPOCH_TAIL_SHIFT;
  }
}

__device__ static inline void* bnxt_re_get_hwqe(struct bnxt_device_sq *sq, uint32_t idx)
{
  idx += sq->tail;
  if (idx >= sq->depth)
    idx -= sq->depth;
  return (void *)((char*)sq->buf + (idx << 4));
}

__device__ static inline void lock(uint32_t *lock) {
  uint32_t expected;

  do {
    expected = 0;
  } while (0 == __hip_atomic_compare_exchange_strong(lock, &expected, 1,
                                                     __ATOMIC_ACQUIRE,
                                                     __ATOMIC_ACQUIRE,
                                                     __HIP_MEMORY_SCOPE_SYSTEM));
}

__device__ static inline void unlock(uint32_t *lock) {
  __hip_atomic_store(lock, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ void QueuePair::bnxt_ring_doorbell(uint32_t slot_idx) {
  struct bnxt_re_db_hdr hdr;
  uint32_t epoch;
  uint64_t key_lo;
  uint64_t key_hi;

  epoch = (bnxt_sq.flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;

  key_lo = (slot_idx | epoch);

  key_hi = (bnxt_sq.id & BNXT_RE_DB_QID_MASK)
         | (((uint64_t) BNXT_RE_QUE_TYPE_SQ & BNXT_RE_DB_TYP_MASK) << BNXT_RE_DB_TYP_SHIFT)
         | (0x1UL << BNXT_RE_DB_VALID_SHIFT);

  hdr.typ_qid_indx = (key_lo | (key_hi << 32));

  __threadfence_system();
  __hip_atomic_store(bnxt_dbr, hdr.typ_qid_indx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ void QueuePair::bnxt_check_cqe_error(struct bnxt_re_req_cqe *cqe) {
  struct bnxt_re_bcqe *hdr;
  uint32_t flg_val;
  uint8_t status;

  const char bnxt_re_wc_error_strings[12][14] = {
    "OK",
    "BAD_RESP",
    "LOC_LEN",
    "LOC_QP_OP",
    "PROT",
    "MEM_OP",
    "REM_INVAL",
    "REM_ACC",
    "REM_OP",
    "RNR_NAK_XCED",
    "TRNSP_XCED",
    "WR_FLUSH",
  };

  hdr = (struct bnxt_re_bcqe*) ((char*)cqe + sizeof(struct bnxt_re_req_cqe));

  flg_val = hdr->flg_st_typ_ph;

  __threadfence();

  // Is the CQE valid?
  status = (flg_val >> BNXT_RE_BCQE_STATUS_SHIFT)
         & BNXT_RE_BCQE_STATUS_MASK;

  if (status != BNXT_RE_REQ_ST_OK) {
    printf("CQ Error %s (%x)\n", bnxt_re_wc_error_strings[status], status);
    abort();
  }
}

__device__ void QueuePair::bnxt_poll_cq_until(uint32_t requested_available_slots) {
  struct bnxt_re_req_cqe *cqe;
  uint32_t sq_tail;
  uint32_t sq_head;
  uint32_t sq_depth;
  uint32_t consumed_slots;
  uint32_t available_slots;

  sq_depth = bnxt_sq.depth;

  do {
    cqe = (struct bnxt_re_req_cqe *) bnxt_cq.buf;

//#ifdef DEBUG
    bnxt_check_cqe_error(cqe);
//#endif

    /* Update the SQ head
     * This param provides us the wqe_idx but we need to convert to the slot idx.
     * We assume a static slots size of GDA_BNXT_WQE_SLOT_COUNT thus can multipy by this value */
    sq_head = (((cqe->con_indx & 0xFFFF) * GDA_BNXT_WQE_SLOT_COUNT) % sq_depth);
    bnxt_sq.head = sq_head;

    sq_tail = __hip_atomic_load(&bnxt_sq.tail, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);

    consumed_slots  = (sq_tail - sq_head + sq_depth) % sq_depth;
    available_slots = sq_depth - consumed_slots;
  } while (available_slots < requested_available_slots);
}

__device__ void QueuePair::bnxt_quiet() {
  uint64_t active_lane_mask;
  uint8_t active_lane_id;

  active_lane_mask  = get_same_qp_lane_mask();
  active_lane_id    = get_active_lane_num(active_lane_mask);

  if (0 == active_lane_id) {
    bnxt_poll_cq_until(bnxt_sq.depth);
  }
}

__device__ void QueuePair::bnxt_quiet_single() {
  bnxt_poll_cq_until(bnxt_sq.depth);
}

__device__ void QueuePair::bnxt_write_rma_wqe(uintptr_t raddr, uintptr_t laddr, int32_t length, uint8_t opcode) {
  struct bnxt_re_bsqe hdr;
  struct bnxt_re_rdma rdma;
  struct bnxt_re_sge sge;
  struct bnxt_re_bsqe *hdr_ptr;
  struct bnxt_re_rdma *rdma_ptr;
  struct bnxt_re_sge *sge_ptr;
  uint32_t wqe_size;
  uint32_t wqe_type;
  uint32_t hdr_flags;
  uint32_t inline_msg;

  inline_msg = length <= inline_threshold &&
               opcode == gda_op_rdma_write;

  bnxt_poll_cq_until(GDA_BNXT_WQE_SLOT_COUNT);

  hdr_ptr  = (struct bnxt_re_bsqe*) bnxt_re_get_hwqe(&bnxt_sq, 0);
  rdma_ptr = (struct bnxt_re_rdma*) bnxt_re_get_hwqe(&bnxt_sq, 1);
  sge_ptr  = (struct bnxt_re_sge*)  bnxt_re_get_hwqe(&bnxt_sq, 2);

  /* Populate Header Segment */
  wqe_type  = BNXT_RE_HDR_WT_MASK & opcode;
  wqe_size  = BNXT_RE_HDR_WS_MASK & GDA_BNXT_WQE_SLOT_COUNT;
  hdr_flags = ((uint32_t) BNXT_RE_HDR_FLAGS_MASK)
            & ((uint32_t) BNXT_RE_WR_FLAGS_SIGNALED);

  if (inline_msg) {
    hdr_flags |= ((uint32_t) BNXT_RE_WR_FLAGS_INLINE);
  }

  hdr.rsv_ws_fl_wt  = (wqe_size  << BNXT_RE_HDR_WS_SHIFT)
                    | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT)
                    | wqe_type;
  hdr.key_immd      = 0;
  hdr.lhdr.qkey_len = length;

  /* Populate RDMA Segment */
  rdma.rva  = raddr;
  rdma.rkey = rkey;

  if (!inline_msg) {
    /* Populate SG Segment */
    sge.pa     = laddr;
    sge.lkey   = lkey;
    sge.length = length;
  }

  /* Write WQE to SQ */
  memcpy(hdr_ptr,  &hdr,  sizeof(struct bnxt_re_bsqe));
  memcpy(rdma_ptr, &rdma, sizeof(struct bnxt_re_rdma));

  if (inline_msg) {
    memcpy(sge_ptr,  reinterpret_cast<const void*>(laddr),  length);
  } else {
    memcpy(sge_ptr,  &sge,  sizeof(struct bnxt_re_sge));
  }

  /* Populate MSN Table */
  bnxt_re_fill_psns_for_msntbl(&bnxt_sq, length);

  /* Update SQ Pointer */
  bnxt_re_incr_tail(&bnxt_sq, GDA_BNXT_WQE_SLOT_COUNT);
}

__device__ void QueuePair::bnxt_post_wqe_rma(int pe, int32_t length, uintptr_t laddr, uintptr_t raddr, uint8_t opcode) {
  uint64_t active_lane_mask;
  uint8_t active_lane_count;
  uint8_t active_lane_id;

  active_lane_mask  = get_same_qp_lane_mask();
  active_lane_count = get_active_lane_count(active_lane_mask);
  active_lane_id    = get_active_lane_num(active_lane_mask);

  if (0 == active_lane_id) {
    lock(&bnxt_sq.lock);
  }

  for (int i = 0; i < active_lane_count; i++) {
    if (i == active_lane_id) {
      /* Write WQE to SQ */
      bnxt_write_rma_wqe(raddr, laddr, length, opcode);

      /* Ring Doorbell */
      bnxt_ring_doorbell(bnxt_sq.tail);
    }
  }

  if (0 == active_lane_id) {
    unlock(&bnxt_sq.lock);
  }
}

__device__ void QueuePair::bnxt_post_wqe_rma_single(int32_t length, uintptr_t laddr,
                                                    uintptr_t raddr, uint8_t opcode,
                                                    bool ring_db) {

  lock(&bnxt_sq.lock);

  /* Write WQE to SQ */
  bnxt_write_rma_wqe(raddr, laddr, length, opcode);

  if (ring_db) {
    bnxt_ring_doorbell(bnxt_sq.tail);
  }

  unlock(&bnxt_sq.lock);
}

__device__ uint32_t QueuePair::bnxt_write_amo_wqe(uintptr_t raddr, uint8_t opcode,
                                                  int64_t atomic_data, int64_t atomic_cmp,
                                                  bool fetching) {
  struct bnxt_re_bsqe hdr;
  struct bnxt_re_atomic amo;
  struct bnxt_re_sge sge;
  struct bnxt_re_bsqe *hdr_ptr;
  struct bnxt_re_atomic *amo_ptr;
  struct bnxt_re_sge *sge_ptr;
  uint32_t wqe_size;
  uint32_t wqe_type;
  uint32_t hdr_flags;

  uint32_t atomic_idx = 0;
  uint32_t length = sizeof(uint64_t);

  bnxt_poll_cq_until(GDA_BNXT_WQE_SLOT_COUNT);

  hdr_ptr = (struct bnxt_re_bsqe*)   bnxt_re_get_hwqe(&bnxt_sq, 0);
  amo_ptr = (struct bnxt_re_atomic*) bnxt_re_get_hwqe(&bnxt_sq, 1);
  sge_ptr = (struct bnxt_re_sge*)    bnxt_re_get_hwqe(&bnxt_sq, 2);

  /* Populate Header Segment */
  wqe_size  = BNXT_RE_HDR_WS_MASK & GDA_BNXT_WQE_SLOT_COUNT;
  hdr_flags = ((uint32_t) BNXT_RE_HDR_FLAGS_MASK)
            & ((uint32_t) BNXT_RE_WR_FLAGS_SIGNALED);
  wqe_type  = BNXT_RE_HDR_WT_MASK & opcode;

  hdr.rsv_ws_fl_wt  = (wqe_size  << BNXT_RE_HDR_WS_SHIFT)
                    | (hdr_flags << BNXT_RE_HDR_FLAGS_SHIFT)
                    | wqe_type;
  hdr.key_immd = rkey;
  hdr.lhdr.rva = raddr;

  /* Populate AMO Segment */
  amo.swp_dt = atomic_data;
  amo.cmp_dt = atomic_cmp;

  /* Populate SG Segment - (Return address of atomic) */
  if (fetching) {
    atomic_idx = fetching_atomic_idx++ % FETCHING_ATOMIC_CNT;
    sge.pa     = (uint64_t) &fetching_atomic[atomic_idx];
    sge.lkey   = fetching_atomic_lkey;
  } else {
    sge.pa     = (uint64_t) nonfetching_atomic;
    sge.lkey   = nonfetching_atomic_lkey;
  }
  sge.length = length;

  /* Write WQE to SQ */
  memcpy(hdr_ptr, &hdr, sizeof(struct bnxt_re_bsqe));
  memcpy(amo_ptr, &amo, sizeof(struct bnxt_re_atomic));
  memcpy(sge_ptr, &sge, sizeof(struct bnxt_re_sge));

  /* Populate MSN Table */
  bnxt_re_fill_psns_for_msntbl(&bnxt_sq, length);

  /* Update SQ Pointer */
  bnxt_re_incr_tail(&bnxt_sq, GDA_BNXT_WQE_SLOT_COUNT);

  return atomic_idx;
}

__device__ uint64_t QueuePair::bnxt_post_wqe_amo(uintptr_t raddr, uint8_t opcode,
                                                 int64_t atomic_data, int64_t atomic_cmp,
                                                 bool fetching) {
  uint64_t active_lane_mask;
  uint8_t active_lane_count;
  uint8_t active_lane_id;
  uint32_t atomic_idx = 0;

  active_lane_mask  = get_same_qp_lane_mask();
  active_lane_count = get_active_lane_count(active_lane_mask);
  active_lane_id    = get_active_lane_num(active_lane_mask);

  if (0 == active_lane_id) {
    lock(&bnxt_sq.lock);
  }

  for (int i = 0; i < active_lane_count; i++) {
    if (i == active_lane_id) {
      atomic_idx = bnxt_write_amo_wqe(raddr, opcode, atomic_data, atomic_cmp, fetching);

      /* Ring Doorbell */
      bnxt_ring_doorbell(bnxt_sq.tail);
    }
  }

  if (0 == active_lane_id) {
    unlock(&bnxt_sq.lock);
  }

  if (fetching) {
    quiet();
    return fetching_atomic[atomic_idx];
  }

  return 0;
}

__device__ uint64_t QueuePair::bnxt_post_wqe_amo_single(uintptr_t raddr, uint8_t opcode,
                                                        int64_t atomic_data, int64_t atomic_cmp,
                                                        bool fetching) {
  uint32_t atomic_idx = 0;

  lock(&bnxt_sq.lock);

  /* Write WQE to SQ */
  atomic_idx = bnxt_write_amo_wqe(raddr, opcode, atomic_data, atomic_cmp, fetching);

  bnxt_ring_doorbell(bnxt_sq.tail);

  unlock(&bnxt_sq.lock);

  if (fetching) {
    quiet();
    return fetching_atomic[atomic_idx];
  }

  return 0;
}

}  // namespace rocshmem
