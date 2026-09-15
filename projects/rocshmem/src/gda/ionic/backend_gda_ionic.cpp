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

#include <new>

#include "gda/backend_gda.hpp"
#include "log.hpp"
#include "util.hpp"

namespace rocshmem {

void GDABackend::ionic_create_cqs(int ncqes) {
  struct ibv_cq_init_attr_ex cq_attr;
  struct ionic_cq_init_attr_ex ionic_cq_attr;

  memset(&cq_attr, 0, sizeof(cq_attr));
  cq_attr.cqe           = ncqes;
  cq_attr.cq_context    = nullptr;
  cq_attr.channel       = nullptr;
  cq_attr.comp_vector   = 0;
  cq_attr.flags         = 0;
  cq_attr.comp_mask     = IBV_CQ_INIT_ATTR_MASK_PD;

  memset(&ionic_cq_attr, 0, sizeof(ionic_cq_attr));
  if (ionic_dv.create_cq_ex) {
    ionic_cq_attr.comp_mask = IONIC_CQ_INIT_ATTR_MASK_FLAGS;
    ionic_cq_attr.flags = IONIC_CQ_INIT_ATTR_CCQE;
  }

  for (size_t i = 0; i < qps.size(); i++) {
    NicDevice &nic = nic_for_qp(i);
    struct ibv_cq_ex *cq_ex = nullptr;

    cq_attr.parent_domain = nic.pd_uxdma[i & 1];

    if (ionic_dv.create_cq_ex) {
      cq_ex = ionic_dv.create_cq_ex(nic.context, &cq_attr, &ionic_cq_attr);
      // If cq_ex is nullptr, fallback to ibv_create_cq_ex below.
    }

    if (!cq_ex) {
      cq_ex = ibv_create_cq_ex(nic.context, &cq_attr);
      CHECK_NNULL(cq_ex, "ibv_create_cq_ex");
    }

    cqs[i] = ibv.cq_ex_to_cq(cq_ex);
    CHECK_NNULL(cqs[i], "ibv_cq_ex_to_cq");
  }
}

void GDABackend::ionic_initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  ibv_qp* qp = qps[conn_num];
  int pe = conn_num % num_pes;
  int nic_idx = nic_idx_for_qp(conn_num);
  const NicDevice& nic = nic_for_qp(conn_num);

  ionic_dv_ctx dvctx;
  ionic_dv.get_ctx(&dvctx, nic.context);

  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));

  void* gpu_db_page = nullptr;
  rocm_memory_lock_to_fine_grain(dvctx.db_page, 0x1000, &gpu_db_page, hip_dev_id);

  uint64_t *db_page_u64 = reinterpret_cast<uint64_t*>(dvctx.db_page);
  uint64_t *gpu_db_page_u64 = reinterpret_cast<uint64_t*>(gpu_db_page);

  uint64_t *gpu_db_ptr = &gpu_db_page_u64[dvctx.db_ptr - db_page_u64];

  gpu_db_cq = &gpu_db_ptr[dvctx.cq_qtype];
  gpu_db_sq = &gpu_db_ptr[dvctx.sq_qtype];

  uint8_t udma_idx = ionic_dv.qp_get_udma_idx(qp);

  ionic_dv_cq dvcq;
  ionic_dv.get_cq(&dvcq, cqs[conn_num], udma_idx);

  ionic_dv_qp dvqp;
  ionic_dv.get_qp(&dvqp, qp);

  uint32_t  qpn        = qp->qp_num;
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

  ionic_v1_wqe* sq_buf   = reinterpret_cast<ionic_v1_wqe*>(dvqp.sq.ptr);
  uint64_t*     sq_dbreg = gpu_db_sq;
  uint64_t      sq_dbval = dvqp.sq.db_val;
  uint16_t      sq_mask  = dvqp.sq.mask;

  ionic_v1_cqe* cq_buf   = reinterpret_cast<ionic_v1_cqe*>(dvcq.q.ptr);
  uint64_t*     cq_dbreg = gpu_db_cq;
  uint64_t      cq_dbval = dvcq.q.db_val;
  uint16_t      cq_mask  = dvcq.q.mask;

  /* QueuePair is either QueuePairIONIC or QueuePairMux
   * both have a constructor that accepts rvalue reference QueuePairIONIC&&,
   * so just use that instead of trying to figure out which one we're using */
  new (gpu_qp) QueuePair{QueuePairIONIC{qpn, heap_laddr, lkey, heap_raddr, rkey, heap_size,
                                        fetching_atomic, fetching_atomic_lkey,
                                        nonfetching_atomic, nonfetching_atomic_lkey,
                                        fetching_atomic_freelist,
                                        local_buffers, num_user_buffers,
                                        symm_buffers, symm_count,
                                        ionic_device_sq{sq_buf, sq_dbreg, sq_dbval, sq_mask},
                                        ionic_device_cq{cq_buf, cq_dbreg, cq_dbval, cq_mask}}};
}

void GDABackend::ionic_setup_parent_domain(NicDevice &nic, struct ibv_parent_domain_init_attr* pattr) {
  ionic_dv.pd_set_sqcmb(nic.pd_parent, false, false, false);
  ionic_dv.pd_set_rqcmb(nic.pd_parent, false, false, false);

  for (int uxdma_i = 0; uxdma_i < 2; ++uxdma_i) {
    nic.pd_uxdma[uxdma_i] = ibv.alloc_parent_domain(nic.context, pattr);
    CHECK_NNULL(nic.pd_uxdma[uxdma_i], "ibv_alloc_parent_domain (uxdma)");

    ionic_dv.pd_set_sqcmb(nic.pd_uxdma[uxdma_i], false, false, false);
    ionic_dv.pd_set_rqcmb(nic.pd_uxdma[uxdma_i], false, false, false);
    ionic_dv.pd_set_udma_mask(nic.pd_uxdma[uxdma_i], 1u << uxdma_i);
  }
}

void* GDABackend::ionic_dv_dlopen() {
  void* dv_handle{nullptr};
  dv_handle = dlopen("libionic.so", RTLD_LAZY);
  if (!dv_handle) {
    // Try hard-coded PATH
    dv_handle = dlopen("/usr/local/lib/libionic.so", RTLD_LAZY);
    if (!dv_handle) {
      LOG_TRACE("Could not open libionic.so. Returning");
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
  DLSYM_OPT_HELPER(ionic_dv, ionic_dv_, ionicdv_handle_, create_cq_ex);

  return ROCSHMEM_SUCCESS;
}

}  // namespace rocshmem
