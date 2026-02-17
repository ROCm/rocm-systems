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

#include "gda/backend_gda.hpp"
#include "util.hpp"
#include <unistd.h> // getpagesize()

namespace rocshmem {

void GDABackend::bnxt_initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  struct bnxt_re_dv_obj dv_obj;
  struct bnxt_re_dv_cq dv_cq;
  struct bnxt_re_dv_qp dv_qp;
  struct ibv_qp *ib_qp;
  int err;

  ib_qp = qps[conn_num];

  /* Export SCQ */
  memset(&dv_obj, 0, sizeof(struct bnxt_re_dv_obj));
  dv_obj.cq.in  = bnxt_scqs[conn_num].cq;
  dv_obj.cq.out = &dv_cq;

  err = bnxt_re_dv.init_obj(&dv_obj, BNXT_RE_DV_OBJ_CQ);
  CHECK_ZERO(err, "bnxt_re_dv_init_obj(CQ)");

  memset(&gpu_qp->bnxt_cq, 0, sizeof(bnxt_device_cq));
  gpu_qp->bnxt_cq.buf   = bnxt_scqs[conn_num].buf;
  gpu_qp->bnxt_cq.depth = bnxt_scqs[conn_num].depth;
  gpu_qp->bnxt_cq.id    = dv_cq.cqn;

  /* Export QP */
  memset(&dv_obj, 0, sizeof(struct bnxt_re_dv_obj));
  dv_obj.qp.in  = ib_qp;
  dv_obj.qp.out = &dv_qp;

  err = bnxt_re_dv.init_obj(&dv_obj, BNXT_RE_DV_OBJ_QP);
  CHECK_ZERO(err, "bnxt_re_dv_init_obj(QP)");

  memset(&gpu_qp->bnxt_sq, 0, sizeof(bnxt_device_sq));
  gpu_qp->bnxt_sq.buf        = bnxt_qps[conn_num].sq_buf;
  gpu_qp->bnxt_sq.depth      = bnxt_qps[conn_num].mem_info.sq_slots;

  if ((gpu_qp->bnxt_sq.depth % BNXT_RE_STATIC_WQE_BB) != 0) {
    fprintf(stderr,
            "[WARNING] SQ depth not divisible by BNXT_RE_STATIC_WQE_BB. "
            "There may be runtime errors.\n");
  }

  gpu_qp->bnxt_sq.id          = ib_qp->qp_num;
  gpu_qp->bnxt_sq.msntbl      = bnxt_qps[conn_num].msntbl;
  gpu_qp->bnxt_sq.msn_tbl_sz  = bnxt_qps[conn_num].msn_tbl_sz;
  gpu_qp->bnxt_sq.psn_sz_log2 = std::log2(bnxt_qps[conn_num].mem_info.sq_psn_sz);
  gpu_qp->bnxt_sq.mtu         = ibv_mtu_to_int(portinfo.active_mtu);

  /* Export DB */
  CHECK_HIP(hipHostRegister(bnxt_qps[conn_num].db_region_attr->dbr, getpagesize(), hipHostRegisterDefault));
  CHECK_HIP(hipHostGetDevicePointer((void**) &gpu_qp->bnxt_dbr, bnxt_qps[conn_num].db_region_attr->dbr, 0));

  /* Export Memory Keys */
  gpu_qp->lkey = heap_mr->lkey;
  gpu_qp->rkey = heap_rkey[conn_num % num_pes];

  /* Export Inline Threshold */
  gpu_qp->inline_threshold = inline_threshold;
}

void GDABackend::bnxt_create_cqs(int cqe) {
  struct bnxt_re_dv_cq_attr cq_attr;
  struct bnxt_re_dv_cq_init_attr cq_init_attr;
  struct bnxt_re_dv_umem_reg_attr umem_attr;

  int dmabuf_enabled = ibv.is_dmabuf_supported();

  /* Ignore value of cqe as we only need of length 1 to use CQE compression */
  cqe = 1;

  /* Create SCQs */
  for (int i = 0; i < qps.size(); i++) {
    /* Allocate SCQ mem */
    memset(&cq_attr, 0, sizeof(struct bnxt_re_dv_cq_attr));
    bnxt_scqs[i].handle = bnxt_re_dv.cq_mem_alloc(context, cqe, &cq_attr);
    CHECK_NNULL(bnxt_scqs[i].handle, "bnxt_re_dv_cq_mem_alloc (SCQ)");

    /* We must force this to a value of 1 to use CQE Compression */
    cq_attr.ncqe = cqe;

    /* Allocate SCQ UMEM */
    bnxt_scqs[i].length = cq_attr.ncqe * cq_attr.cqe_size;
    bnxt_scqs[i].depth  = cq_attr.ncqe;
    CHECK_HIP(hipExtMallocWithFlags(&bnxt_scqs[i].buf, bnxt_scqs[i].length, hipDeviceMallocUncached));
    CHECK_HIP(hipMemset(bnxt_scqs[i].buf, 0, bnxt_scqs[i].length));

    if (dmabuf_enabled) {
      CHECK_HSA(hsa_amd_portable_export_dmabuf(bnxt_scqs[i].buf,
                                               bnxt_scqs[i].length,
                                               &bnxt_scqs[i].dmabuf_fd,
                                               &bnxt_scqs[i].dmabuf_offset));
    }

    /* Register SCQ UMEM */
    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = bnxt_scqs[i].buf;
    umem_attr.size         = bnxt_scqs[i].length;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;
    umem_attr.dmabuf_fd    = dmabuf_enabled ? bnxt_scqs[i].dmabuf_fd : 0;

    bnxt_scqs[i].umem_handle = bnxt_re_dv.umem_reg(context, &umem_attr);
    CHECK_NNULL(bnxt_scqs[i].umem_handle, "bnxt_re_dv_umem_reg(scq_buf)");

    /* Create SCQ */
    memset(&cq_init_attr, 0, sizeof(struct bnxt_re_dv_cq_init_attr));
    cq_init_attr.cq_handle   = (uint64_t) bnxt_scqs[i].handle;
    cq_init_attr.umem_handle = bnxt_scqs[i].umem_handle;
    cq_init_attr.ncqe        = cq_attr.ncqe;

    bnxt_scqs[i].cq = bnxt_re_dv.create_cq(context, &cq_init_attr);
    CHECK_NNULL(bnxt_scqs[i].cq, "bnxt_re_dv_create_cq (SCQ) ");
  }

  /* Create RCQs */
  for (int i = 0; i < qps.size(); i++) {
    /* Allocate RCQ mem */
    memset(&cq_attr, 0, sizeof(struct bnxt_re_dv_cq_attr));
    bnxt_rcqs[i].handle = bnxt_re_dv.cq_mem_alloc(context, cqe, &cq_attr);
    CHECK_NNULL(bnxt_rcqs[i].handle, "bnxt_re_dv_cq_mem_alloc (RCQ)");

    /* Allocate RCQ UMEM */
    bnxt_rcqs[i].length = cq_attr.ncqe * cq_attr.cqe_size;
    bnxt_rcqs[i].depth  = cq_attr.ncqe;
    CHECK_HIP(hipExtMallocWithFlags(&bnxt_rcqs[i].buf, bnxt_rcqs[i].length, hipDeviceMallocUncached));
    CHECK_HIP(hipMemset(bnxt_rcqs[i].buf, 0, bnxt_rcqs[i].length));

    if (dmabuf_enabled) {
      CHECK_HSA(hsa_amd_portable_export_dmabuf(bnxt_rcqs[i].buf,
                                               bnxt_rcqs[i].length,
                                               &bnxt_rcqs[i].dmabuf_fd,
                                               &bnxt_rcqs[i].dmabuf_offset));
    }

    /* Register RCQ UMEM */
    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = bnxt_rcqs[i].buf;
    umem_attr.size         = bnxt_rcqs[i].length;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;
    umem_attr.dmabuf_fd    = dmabuf_enabled ? bnxt_rcqs[i].dmabuf_fd : 0;

    bnxt_rcqs[i].umem_handle = bnxt_re_dv.umem_reg(context, &umem_attr);
    CHECK_NNULL(bnxt_rcqs[i].umem_handle, "bnxt_re_dv_umem_reg(rcq_buf)");

    /* Create RCQ */
    memset(&cq_init_attr, 0, sizeof(struct bnxt_re_dv_cq_init_attr));
    cq_init_attr.cq_handle   = (uint64_t) bnxt_rcqs[i].handle;
    cq_init_attr.umem_handle = bnxt_rcqs[i].umem_handle;
    cq_init_attr.ncqe        = cq_attr.ncqe;

    bnxt_rcqs[i].cq = bnxt_re_dv.create_cq(context, &cq_init_attr);
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

  int dmabuf_enabled = ibv.is_dmabuf_supported();

  for (int i = 0; i < qps.size(); i++) {
    /* IB QP Init Attr */
    memset(&ib_qp_attr, 0, sizeof(struct ibv_qp_init_attr));
    ib_qp_attr.send_cq             = bnxt_scqs[i].cq;
    ib_qp_attr.recv_cq             = bnxt_rcqs[i].cq;
    ib_qp_attr.cap.max_send_wr     = sq_length;
    ib_qp_attr.cap.max_recv_wr     = 0;
    ib_qp_attr.cap.max_send_sge    = 1;
    ib_qp_attr.cap.max_recv_sge    = 0;
    ib_qp_attr.cap.max_inline_data = inline_threshold;
    ib_qp_attr.qp_type             = IBV_QPT_RC;
    ib_qp_attr.sq_sig_all          = 0;

    /* Alloc qp_mem_info */
    memset(&bnxt_qps[i].mem_info, 0, sizeof(struct bnxt_re_dv_qp_mem_info));
    err = bnxt_re_dv.qp_mem_alloc(pd_orig, &ib_qp_attr, &bnxt_qps[i].mem_info);
    CHECK_ZERO(err, "bnxt_re_dv_qp_mem_alloc");

    /* Alloc SQ */
    CHECK_HIP(hipExtMallocWithFlags(&sq_ptr, bnxt_qps[i].mem_info.sq_len, hipDeviceMallocUncached));
    CHECK_HIP(hipMemset(sq_ptr, 0,  bnxt_qps[i].mem_info.sq_len));
    bnxt_qps[i].mem_info.sq_va = (uint64_t) sq_ptr;
    bnxt_qps[i].sq_buf = sq_ptr;

    if (dmabuf_enabled) {
      CHECK_HSA(hsa_amd_portable_export_dmabuf(sq_ptr,
                                               bnxt_qps[i].mem_info.sq_len,
                                               &bnxt_qps[i].sq_dmabuf_fd,
                                               &bnxt_qps[i].sq_dmabuf_offset));
    }

    /* Obtain MSN Table Pointer */
    msntbl_len             = (bnxt_qps[i].mem_info.sq_psn_sz * bnxt_qps[i].mem_info.sq_npsn);
    msntbl_offset          = bnxt_qps[i].mem_info.sq_len - msntbl_len;
    bnxt_qps[i].msntbl     = (void*) ((char*) bnxt_qps[i].sq_buf + msntbl_offset);
    bnxt_qps[i].msn_tbl_sz = bnxt_qps[i].mem_info.sq_npsn;

    /* Register SQ UMEM */
    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = (void*) bnxt_qps[i].mem_info.sq_va;
    umem_attr.size         = bnxt_qps[i].mem_info.sq_len;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;
    umem_attr.dmabuf_fd    = dmabuf_enabled ? bnxt_qps[i].sq_dmabuf_fd : 0;

    sq_umem_handle = bnxt_re_dv.umem_reg(context, &umem_attr);
    CHECK_NNULL(sq_umem_handle, "bnxt_re_dv_umem_reg(sq)");

    /* Alloc RQ */
    CHECK_HIP(hipExtMallocWithFlags(&rq_ptr, bnxt_qps[i].mem_info.rq_len, hipDeviceMallocUncached));
    CHECK_HIP(hipMemset(rq_ptr, 0,  bnxt_qps[i].mem_info.rq_len));
    bnxt_qps[i].mem_info.rq_va = (uint64_t) rq_ptr;
    bnxt_qps[i].rq_buf = rq_ptr;

    if (dmabuf_enabled) {
      CHECK_HSA(hsa_amd_portable_export_dmabuf(rq_ptr,
                                               bnxt_qps[i].mem_info.rq_len,
                                               &bnxt_qps[i].rq_dmabuf_fd,
                                               &bnxt_qps[i].rq_dmabuf_offset));
    }

    /* Register RQ UMEM */
    memset(&umem_attr, 0, sizeof(struct bnxt_re_dv_umem_reg_attr));
    umem_attr.addr         = (void*) bnxt_qps[i].mem_info.rq_va;
    umem_attr.size         = bnxt_qps[i].mem_info.rq_len;
    umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;
    umem_attr.dmabuf_fd    = dmabuf_enabled ? bnxt_qps[i].rq_dmabuf_fd : 0;

    rq_umem_handle = bnxt_re_dv.umem_reg(context, &umem_attr);
    CHECK_NNULL(rq_umem_handle, "bnxt_re_dv_umem_reg(rq)");

    /* Alloc DPI */
    bnxt_qps[i].db_region_attr = bnxt_re_dv.alloc_db_region(context);
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
    qps[i] = bnxt_re_dv.create_qp(pd_orig, &bnxt_qps[i].attr);
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
      DPRINTF("Could not open libbnxt_re.so. Returning\n");
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
