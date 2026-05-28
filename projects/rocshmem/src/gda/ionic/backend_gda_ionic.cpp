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

namespace rocshmem {

void GDABackend::ionic_initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  ionic_dv_ctx dvctx;
  ionic_dv.get_ctx(&dvctx, context);

  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));

  void* gpu_db_page = nullptr;
  rocm_memory_lock_to_fine_grain(dvctx.db_page, 0x1000, &gpu_db_page, hip_dev_id);

  uint64_t *db_page_u64 = reinterpret_cast<uint64_t*>(dvctx.db_page);
  uint64_t *gpu_db_page_u64 = reinterpret_cast<uint64_t*>(gpu_db_page);

  uint64_t *gpu_db_ptr = &gpu_db_page_u64[dvctx.db_ptr - db_page_u64];

  gpu_db_page = gpu_db_page;
  gpu_db_cq = &gpu_db_ptr[dvctx.cq_qtype];
  gpu_db_sq = &gpu_db_ptr[dvctx.sq_qtype];

  uint8_t udma_idx = ionic_dv.qp_get_udma_idx(qps[conn_num]);

  ionic_dv_cq dvcq;
  ionic_dv.get_cq(&dvcq, cqs[conn_num], udma_idx);

  gpu_qp->cq_dbreg = gpu_db_cq;
  gpu_qp->cq_dbval = dvcq.q.db_val;
  gpu_qp->cq_mask = dvcq.q.mask;

  gpu_qp->ionic_cq_buf = reinterpret_cast<ionic_v1_cqe*>(dvcq.q.ptr);

  ionic_dv_qp dvqp;
  ionic_dv.get_qp(&dvqp, qps[conn_num]);

  gpu_qp->sq_dbreg = gpu_db_sq;
  gpu_qp->sq_dbval = dvqp.sq.db_val;
  gpu_qp->sq_mask = dvqp.sq.mask;
  gpu_qp->ionic_sq_buf = reinterpret_cast<ionic_v1_wqe *>(dvqp.sq.ptr);

  strncpy(gpu_qp->dev_name,
          qps[conn_num]->context->device->name,
          sizeof(gpu_qp->dev_name));
  gpu_qp->dev_name[sizeof(gpu_qp->dev_name) - 1] = 0;

  gpu_qp->qp_num = qps[conn_num]->qp_num;
  gpu_qp->lkey = heap_mr->lkey;
  gpu_qp->rkey = heap_rkey[conn_num % num_pes];
  gpu_qp->inline_threshold = 32;
}

void GDABackend::ionic_setup_parent_domain(struct ibv_parent_domain_init_attr* pattr) {
  ionic_dv.pd_set_sqcmb(pd_parent, false, false, false);
  ionic_dv.pd_set_rqcmb(pd_parent, false, false, false);

  for (int uxdma_i = 0; uxdma_i < 2; ++uxdma_i) {
    pd_uxdma[uxdma_i] = ibv.alloc_parent_domain(context, pattr);
    CHECK_NNULL(pd_uxdma[uxdma_i], "ibv_alloc_parent_domain (uxdma)");

    ionic_dv.pd_set_sqcmb(pd_uxdma[uxdma_i], false, false, false);
    ionic_dv.pd_set_rqcmb(pd_uxdma[uxdma_i], false, false, false);
    ionic_dv.pd_set_udma_mask(pd_uxdma[uxdma_i], 1u << uxdma_i);
  }
}

void* GDABackend::ionic_dv_dlopen() {
  void* dv_handle{nullptr};
  dv_handle = dlopen("libionic.so", RTLD_LAZY);
  if (!dv_handle) {
    // Try hard-coded PATH
    dv_handle = dlopen("/usr/local/lib/libionic.so", RTLD_LAZY);
    if (!dv_handle) {
      DPRINTF("Could not open libionic.so. Returning\n");
    }
  }
  return dv_handle;
}

int GDABackend::ionic_dv_dl_init() {
  ionicdv_handle_ = ionic_dv_dlopen();
  if (!ionicdv_handle_)
    return ROCSHMEM_ERROR;

  DLSYM_HELPER(ionic_dv, ionic_dv_, ionicdv_handle_, get_ctx);
  DLSYM_HELPER(ionic_dv, ionic_dv_, ionicdv_handle_, qp_get_udma_idx);
  DLSYM_HELPER(ionic_dv, ionic_dv_, ionicdv_handle_, get_cq);
  DLSYM_HELPER(ionic_dv, ionic_dv_, ionicdv_handle_, get_qp);
  DLSYM_HELPER(ionic_dv, ionic_dv_, ionicdv_handle_, pd_set_sqcmb);
  DLSYM_HELPER(ionic_dv, ionic_dv_, ionicdv_handle_, pd_set_rqcmb);
  DLSYM_HELPER(ionic_dv, ionic_dv_, ionicdv_handle_, pd_set_udma_mask);

  return ROCSHMEM_SUCCESS;
}

}  // namespace rocshmem
