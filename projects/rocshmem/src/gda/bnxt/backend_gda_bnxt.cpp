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

#include <unistd.h> // getpagesize()
#include <new>

#include "bit.hpp"
#include "log.hpp"
#include "util.hpp"
#include "gda/backend_gda.hpp"
#include "gda/bnxt/provider_gda_bnxt.hpp"
#include "gda/bnxt/queue_pair_bnxt.hpp"
#include "gda/queue_pair_provider.hpp"

namespace rocshmem {

void GDABackend::bnxt_initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  struct bnxt_re_dv_obj dv_obj;
  struct bnxt_re_dv_cq dv_cq;
  struct bnxt_re_dv_qp dv_qp;
  struct ibv_qp *ib_qp;
  int err;

  int pe = conn_num % num_pes;
  int nic_idx = nic_idx_for_qp(conn_num);
  const NicDevice& nic = nic_for_qp(conn_num);

  ib_qp = qps[conn_num];

  /* Export SCQ */
  memset(&dv_obj, 0, sizeof(struct bnxt_re_dv_obj));
  dv_obj.cq.in  = bnxt_scqs[conn_num].cq;
  dv_obj.cq.out = &dv_cq;

  err = bnxt_re_dv.init_obj(&dv_obj, BNXT_RE_DV_OBJ_CQ);
  CHECK_ZERO(err, "bnxt_re_dv_init_obj(CQ)");

  /* Export QP */
  memset(&dv_obj, 0, sizeof(struct bnxt_re_dv_obj));
  dv_obj.qp.in  = ib_qp;
  dv_obj.qp.out = &dv_qp;

  err = bnxt_re_dv.init_obj(&dv_obj, BNXT_RE_DV_OBJ_QP);
  CHECK_ZERO(err, "bnxt_re_dv_init_obj(QP)");

  /* Export DB */
  void* gpu_dbr_ptr = nullptr;
  CHECK_HIP(hipHostRegister(bnxt_qps[conn_num].db_region_attr->dbr, getpagesize(), hipHostRegisterDefault));
  CHECK_HIP(hipHostGetDevicePointer(&gpu_dbr_ptr, bnxt_qps[conn_num].db_region_attr->dbr, 0));

  uint32_t  qpn        = ib_qp->qp_num;
  uintptr_t heap_laddr = reinterpret_cast<uintptr_t>(heap.get_local_heap_base());
  uintptr_t heap_raddr = reinterpret_cast<uintptr_t>(heap.get_heap_bases()[pe]);
  size_t    heap_size  = heap.get_size();
  uint32_t  lkey       = nic.heap_mr->lkey;
  uint32_t  rkey       = heap_rkey[flat_pe_nic_idx(pe, nic_idx)];

  host_qps.emplace_back(nic.pd_orig);
  const QueuePairHost &host_qp = host_qps.back();

  uint64_t*            fetching_atomic          = host_qp.fetching_atomic;
  uint32_t             fetching_atomic_lkey     = host_qp.fetching_atomic_mr->lkey;
  uint64_t*            nonfetching_atomic       = host_qp.nonfetching_atomic;
  uint32_t             nonfetching_atomic_lkey  = host_qp.nonfetching_atomic_mr->lkey;
  FreeList<uint64_t*>* fetching_atomic_freelist = host_qp.fetching_atomic_freelist;
  const BufferInfo*    local_buffers            = host_qp.buffer_info;
  size_t               num_user_buffers         = host_qp.num_user_buffers;

  const SymmBufferInfo *symm_buffers = get_symm_buffers_slice(pe, nic_idx);
  const int            *symm_count   = symm_count_;

  uint64_t* dbr = reinterpret_cast<uint64_t*>(gpu_dbr_ptr);

  void*    sq_buf      = bnxt_qps[conn_num].sq_buf;
  uint32_t sq_depth    = bnxt_qps[conn_num].mem_info.sq_slots;
  void*    msntbl      = bnxt_qps[conn_num].msntbl;
  uint32_t msn_tbl_sz  = bnxt_qps[conn_num].msn_tbl_sz;
  uint32_t psn_sz_log2 = bit_log2(bnxt_qps[conn_num].mem_info.sq_psn_sz);
  uint32_t mtu         = ibv_mtu_to_int(nic.portinfo.active_mtu);

  void*    cq_buf   = bnxt_scqs[conn_num].buf;
  uint32_t cq_depth = bnxt_scqs[conn_num].depth;

  if ((sq_depth % BNXT_RE_STATIC_WQE_BB) != 0) {
    LOG_WARN("SQ depth not divisible by BNXT_RE_STATIC_WQE_BB. "
             "There may be runtime errors.");
  }

  /* QueuePair is either QueuePairBNXT or QueuePairMux
   * both have a constructor that accepts rvalue reference QueuePairBNXT&&,
   * so just use that instead of trying to figure out which one we're using */
  new (gpu_qp) QueuePair{QueuePairBNXT{qpn, heap_laddr, lkey, heap_raddr, rkey, heap_size,
                                       fetching_atomic, fetching_atomic_lkey,
                                       nonfetching_atomic, nonfetching_atomic_lkey,
                                       fetching_atomic_freelist,
                                       local_buffers, num_user_buffers,
                                       symm_buffers, symm_count, dbr,
                                       bnxt_device_sq{sq_buf, sq_depth, msntbl, msn_tbl_sz,
                                                      psn_sz_log2, mtu},
                                       bnxt_device_cq{cq_buf, cq_depth}}};
}

void GDABackend::bnxt_create_cqs(int cqe) {
  struct bnxt_re_dv_cq_attr cq_attr;
  struct bnxt_re_dv_cq_init_attr cq_init_attr;
  struct bnxt_re_dv_umem_reg_attr umem_attr;

  /* Ignore value of cqe as we only need of length 1 to use CQE compression */
  cqe = 1;

  /* Create SCQs */
  for (size_t i = 0; i < qps.size(); i++) {
    auto *ctx = nic_for_qp(i).context;

    /* Allocate SCQ mem */
    memset(&cq_attr, 0, sizeof(struct bnxt_re_dv_cq_attr));
    bnxt_scqs[i].handle = bnxt_re_dv.cq_mem_alloc(ctx, cqe, &cq_attr);
    CHECK_NNULL(bnxt_scqs[i].handle, "bnxt_re_dv_cq_mem_alloc (SCQ)");

    /* We must force this to a value of 1 to use CQE Compression */
    cq_attr.ncqe = cqe;

    /* Allocate SCQ UMEM */
    bnxt_scqs[i].length = cq_attr.ncqe * cq_attr.cqe_size;
    bnxt_scqs[i].depth  = cq_attr.ncqe;
    qp_allocator_->allocate(reinterpret_cast<void**>(&bnxt_scqs[i].buf), bnxt_scqs[i].length);
    CHECK_HIP(hipMemset(bnxt_scqs[i].buf, 0, bnxt_scqs[i].length));

    /* Register SCQ UMEM — CQ/QP control buffers are host-accessible memory;
     * dmabuf registration is not supported by the bnxt_re driver for these
     * internal structures (only for the symmetric heap MR). */
    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = bnxt_scqs[i].buf;
    umem_attr.size         = bnxt_scqs[i].length;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;

    bnxt_scqs[i].umem_handle = bnxt_re_dv.umem_reg(ctx, &umem_attr);
    CHECK_NNULL(bnxt_scqs[i].umem_handle, "bnxt_re_dv_umem_reg(scq_buf)");

    /* Create SCQ */
    memset(&cq_init_attr, 0, sizeof(struct bnxt_re_dv_cq_init_attr));
    cq_init_attr.cq_handle   = (uint64_t) bnxt_scqs[i].handle;
    cq_init_attr.umem_handle = bnxt_scqs[i].umem_handle;
    cq_init_attr.ncqe        = cq_attr.ncqe;

    bnxt_scqs[i].cq = bnxt_re_dv.create_cq(ctx, &cq_init_attr);
    CHECK_NNULL(bnxt_scqs[i].cq, "bnxt_re_dv_create_cq (SCQ)");
  }

  /* Create RCQs */
  for (size_t i = 0; i < qps.size(); i++) {
    auto *ctx = nic_for_qp(i).context;

    /* Allocate RCQ mem */
    memset(&cq_attr, 0, sizeof(struct bnxt_re_dv_cq_attr));
    bnxt_rcqs[i].handle = bnxt_re_dv.cq_mem_alloc(ctx, cqe, &cq_attr);
    CHECK_NNULL(bnxt_rcqs[i].handle, "bnxt_re_dv_cq_mem_alloc (RCQ)");

    /* Allocate RCQ UMEM */
    bnxt_rcqs[i].length = cq_attr.ncqe * cq_attr.cqe_size;
    bnxt_rcqs[i].depth  = cq_attr.ncqe;
    qp_allocator_->allocate(reinterpret_cast<void**>(&bnxt_rcqs[i].buf), bnxt_rcqs[i].length);
    CHECK_HIP(hipMemset(bnxt_rcqs[i].buf, 0, bnxt_rcqs[i].length));

    /* Register RCQ UMEM — same reasoning as SCQ: plain path only */
    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = bnxt_rcqs[i].buf;
    umem_attr.size         = bnxt_rcqs[i].length;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;

    bnxt_rcqs[i].umem_handle = bnxt_re_dv.umem_reg(ctx, &umem_attr);
    CHECK_NNULL(bnxt_rcqs[i].umem_handle, "bnxt_re_dv_umem_reg(rcq_buf)");

    /* Create RCQ */
    memset(&cq_init_attr, 0, sizeof(struct bnxt_re_dv_cq_init_attr));
    cq_init_attr.cq_handle   = (uint64_t) bnxt_rcqs[i].handle;
    cq_init_attr.umem_handle = bnxt_rcqs[i].umem_handle;
    cq_init_attr.ncqe        = cq_attr.ncqe;

    bnxt_rcqs[i].cq = bnxt_re_dv.create_cq(ctx, &cq_init_attr);
    CHECK_NNULL(bnxt_rcqs[i].cq, "bnxt_re_dv_create_cq (RCQ)");
  }
}

void GDABackend::bnxt_create_qps(int sq_length) {
  struct ibv_qp_init_attr ib_qp_attr;
  struct bnxt_re_dv_umem_reg_attr umem_attr;
  void *sq_ptr;
  void *rq_ptr;
  void* sq_umem_handle;
  void* rq_umem_handle;
  uint64_t msntbl_len;
  uint64_t msntbl_offset;
  int err;

  for (size_t i = 0; i < qps.size(); i++) {
    NicDevice &nic = nic_for_qp(i);
    auto *ctx = nic.context;
    auto *pd  = nic.pd_orig;

    /* IB QP Init Attr */
    memset(&ib_qp_attr, 0, sizeof(struct ibv_qp_init_attr));
    ib_qp_attr.send_cq             = bnxt_scqs[i].cq;
    ib_qp_attr.recv_cq             = bnxt_rcqs[i].cq;
    ib_qp_attr.cap.max_send_wr     = sq_length;
    ib_qp_attr.cap.max_recv_wr     = 0;
    ib_qp_attr.cap.max_send_sge    = 1;
    ib_qp_attr.cap.max_recv_sge    = 0;
    ib_qp_attr.cap.max_inline_data = QueuePairTraits<QueuePairBNXT>::InlineThreshold;
    ib_qp_attr.qp_type             = IBV_QPT_RC;
    ib_qp_attr.sq_sig_all          = 0;

    /* Alloc qp_mem_info */
    memset(&bnxt_qps[i].mem_info, 0, sizeof(struct bnxt_re_dv_qp_mem_info));
    err = bnxt_re_dv.qp_mem_alloc(pd, &ib_qp_attr, &bnxt_qps[i].mem_info);
    CHECK_ZERO(err, "bnxt_re_dv_qp_mem_alloc");

    /* Alloc SQ */
    qp_allocator_->allocate(reinterpret_cast<void**>(&sq_ptr), bnxt_qps[i].mem_info.sq_len);
    CHECK_HIP(hipMemset(sq_ptr, 0,  bnxt_qps[i].mem_info.sq_len));
    bnxt_qps[i].mem_info.sq_va = (uint64_t) sq_ptr;
    bnxt_qps[i].sq_buf = sq_ptr;

    /* Obtain MSN Table Pointer */
    msntbl_len             = (bnxt_qps[i].mem_info.sq_psn_sz * bnxt_qps[i].mem_info.sq_npsn);
    msntbl_offset          = bnxt_qps[i].mem_info.sq_len - msntbl_len;
    bnxt_qps[i].msntbl     = (void*) ((char*) bnxt_qps[i].sq_buf + msntbl_offset);
    bnxt_qps[i].msn_tbl_sz = bnxt_qps[i].mem_info.sq_npsn;

    /* Register SQ UMEM — CQ/QP control buffers are finegrained CPU-accessible
     * memory; dmabuf registration is not supported by the bnxt_re driver for
     * these internal structures (only for the symmetric heap MR). */
    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = (void*) bnxt_qps[i].mem_info.sq_va;
    umem_attr.size         = bnxt_qps[i].mem_info.sq_len;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;

    sq_umem_handle = bnxt_re_dv.umem_reg(ctx, &umem_attr);
    CHECK_NNULL(sq_umem_handle, "bnxt_re_dv_umem_reg(sq)");

    /* Alloc RQ */
    qp_allocator_->allocate(reinterpret_cast<void**>(&rq_ptr), bnxt_qps[i].mem_info.rq_len);
    CHECK_HIP(hipMemset(rq_ptr, 0,  bnxt_qps[i].mem_info.rq_len));
    bnxt_qps[i].mem_info.rq_va = (uint64_t) rq_ptr;
    bnxt_qps[i].rq_buf = rq_ptr;

    /* Register RQ UMEM — same reasoning as SQ: plain path only */
    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = (void*) bnxt_qps[i].mem_info.rq_va;
    umem_attr.size         = bnxt_qps[i].mem_info.rq_len;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;

    rq_umem_handle = bnxt_re_dv.umem_reg(ctx, &umem_attr);
    CHECK_NNULL(rq_umem_handle, "bnxt_re_dv_umem_reg(rq)");

    /* Alloc DPI */
    bnxt_qps[i].db_region_attr = bnxt_re_dv.alloc_db_region(ctx);
    CHECK_NNULL(bnxt_qps[i].db_region_attr, "bnxt_re_dv_alloc_db_region");

    /* IB DV QP Init Attr */
    memset(&bnxt_qps[i].attr, 0, sizeof(struct bnxt_re_dv_qp_init_attr));
    bnxt_qps[i].attr.send_cq         = ib_qp_attr.send_cq;
    bnxt_qps[i].attr.recv_cq         = ib_qp_attr.recv_cq;
    bnxt_qps[i].attr.max_send_wr     = ib_qp_attr.cap.max_send_wr;
    bnxt_qps[i].attr.max_recv_wr     = ib_qp_attr.cap.max_recv_wr;
    bnxt_qps[i].attr.max_send_sge    = ib_qp_attr.cap.max_send_sge;
    bnxt_qps[i].attr.max_recv_sge    = ib_qp_attr.cap.max_recv_sge;
    bnxt_qps[i].attr.max_inline_data = ib_qp_attr.cap.max_inline_data;
    bnxt_qps[i].attr.qp_type         = ib_qp_attr.qp_type;

    bnxt_qps[i].attr.qp_handle      = bnxt_qps[i].mem_info.qp_handle;
    bnxt_qps[i].attr.dbr_handle     = bnxt_qps[i].db_region_attr;
    bnxt_qps[i].attr.sq_umem_handle = sq_umem_handle;
    bnxt_qps[i].attr.sq_len         = bnxt_qps[i].mem_info.sq_len;
    bnxt_qps[i].attr.sq_slots       = bnxt_qps[i].mem_info.sq_slots;
    bnxt_qps[i].attr.sq_wqe_sz      = bnxt_qps[i].mem_info.sq_wqe_sz;
    bnxt_qps[i].attr.sq_psn_sz      = bnxt_qps[i].mem_info.sq_psn_sz;
    bnxt_qps[i].attr.sq_npsn        = bnxt_qps[i].mem_info.sq_npsn;

    bnxt_qps[i].attr.rq_umem_handle = rq_umem_handle;
    bnxt_qps[i].attr.rq_len         = bnxt_qps[i].mem_info.rq_len;
    bnxt_qps[i].attr.rq_slots       = bnxt_qps[i].mem_info.rq_slots;
    bnxt_qps[i].attr.rq_wqe_sz      = bnxt_qps[i].mem_info.rq_wqe_sz;
    bnxt_qps[i].attr.comp_mask      = bnxt_qps[i].mem_info.comp_mask;

    /* Alloc QP */
    qps[i] = bnxt_re_dv.create_qp(pd, &bnxt_qps[i].attr);
    CHECK_NNULL(qps[i], "bnxt_re_dv_create_qp");
  }
}

void* GDABackend::bnxt_dv_dlopen() {
  void* dv_handle{nullptr};
  dv_handle = dlopen("libbnxt_re.so", RTLD_LAZY);
  if (!dv_handle) {
    // Try hard-coded PATH
    dv_handle = dlopen("/usr/local/lib/libbnxt_re.so", RTLD_LAZY);
    if (!dv_handle) {
      LOG_TRACE("Could not open libbnxt_re.so. Returning");
    }
  }
  return dv_handle;
}

int GDABackend::bnxt_dv_dl_init() {
  bnxtdv_handle_ = bnxt_dv_dlopen();
  if (!bnxtdv_handle_)
    return ROCSHMEM_ERROR;

  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, init_obj);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, create_qp);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, destroy_qp);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, modify_qp);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, qp_mem_alloc);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, create_cq);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, destroy_cq);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, cq_mem_alloc);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, umem_reg);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, umem_dereg);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, alloc_db_region);
  DLSYM_HELPER(bnxt_re_dv, bnxt_re_dv_, bnxtdv_handle_, free_db_region);

  return ROCSHMEM_SUCCESS;
}

}  // namespace rocshmem
