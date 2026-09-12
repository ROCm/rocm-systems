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

#include "log.hpp"
#include "util.hpp"
#include "gda/backend_gda.hpp"
#include "gda/mlx5/provider_gda_mlx5.hpp"
#include "gda/mlx5/queue_pair_mlx5.hpp"
#include "gda/queue_pair_provider.hpp"

namespace rocshmem {

void* GDABackend::mlx5_dv_dlopen() {
  void* dv_handle{nullptr};
  dv_handle = dlopen("libmlx5.so.1", RTLD_LAZY);
  if (!dv_handle) {
    LOG_TRACE("Could not open libmlx5.so.1. Returning");
  }
  return dv_handle;
}

int GDABackend::mlx5_dv_dl_init() {
  mlx5dv_handle_ = mlx5_dv_dlopen();
  if (!mlx5dv_handle_)
    return ROCSHMEM_ERROR;

  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, init_obj);
  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, open_device);
  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, devx_obj_create);
  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, devx_obj_modify);
  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, devx_obj_destroy);
  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, devx_umem_reg_ex);
  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, devx_umem_dereg);
  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, devx_alloc_uar);
  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, devx_free_uar);
  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, devx_query_eqn);
  return ROCSHMEM_SUCCESS;
}

void GDABackend::mlx5_create_qps(int sq_length) {
  for (size_t i = 0; i < mlx5_qps.size(); i++) {
    NicDevice &nic = nic_for_qp(i);
    int err = mlx5dv.create_qp(mlx5_qps[i], nic.context, nic.pd_orig, sq_length);
    CHECK_ZERO(err, "mlx5dv::create_qp");
  }
}

void GDABackend::mlx5_initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  mlx5_devx_qp& qp = mlx5_qps[conn_num];

  /*
   * struct mlx5_devx_qp {
   *   ibv_context*      ctx;
   *   mlx5dv_devx_obj*  devx_cq_obj;
   *   mlx5dv_devx_obj*  devx_qp_obj;
   *   mlx5dv_devx_uar*  uar;
   *   mlx5dv_devx_umem* umem;
   *   void*             cq;
   *   void*             sq;
   *   uint32_t*         cq_dbrec;
   *   uint32_t*         qp_dbrec;
   *   uint32_t          cqn;
   *   uint32_t          qpn;
   *   uint16_t          sq_depth;
   * };
   *
   * struct mlx5dv_devx_uar {
   *   void     *reg_addr;
   *   void     *base_addr;
   *   uint32_t page_id;
   *   off_t    mmap_off;
   *   uint64_t comp_mask;
   * };
   */

  int pe = conn_num % num_pes;
  int nic_idx = nic_idx_for_qp(conn_num);
  const NicDevice& nic = nic_for_qp(conn_num);

  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  void* gpu_db_ptr{nullptr};
  // not necessary to switch between BlueFlame buffer halves when using it as a doorbell only
  rocm_memory_lock_to_fine_grain(qp.uar->reg_addr, MLX5_DB_BLUEFLAME_BUFFER_SIZE,
                                 &gpu_db_ptr, hip_dev_id);

  uint32_t  qpn        = qp.qpn;
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

  gda_mlx5_wqe*      sq_buf   = reinterpret_cast<gda_mlx5_wqe*>(qp.sq);
  // qp.dbrec points to two __be32 values: RQ dbrec at MLX5_RCV_DBR and SQ dbrec at MLX5_SND_DBR
  __be32*            sq_dbrec = &qp.qp_dbrec[MLX5_SND_DBR];
  gda_mlx5_doorbell* sq_db    = reinterpret_cast<gda_mlx5_doorbell*>(gpu_db_ptr);
  uint16_t           sq_depth = static_cast<uint16_t>(qp.sq_depth);

  mlx5_cqe64* cq_buf   = reinterpret_cast<mlx5_cqe64*>(qp.cq);
  __be32*     cq_dbrec = qp.cq_dbrec;

  /* QueuePair is either QueuePairMLX5 or QueuePairMux
   * both have a constructor that accepts rvalue reference QueuePairMLX5&&,
   * so just use that instead of trying to figure out which one we're using */
  new (gpu_qp) QueuePair{QueuePairMLX5{qpn, heap_laddr, lkey, heap_raddr, rkey, heap_size,
                                       fetching_atomic, fetching_atomic_lkey,
                                       nonfetching_atomic, nonfetching_atomic_lkey,
                                       fetching_atomic_freelist,
                                       local_buffers, num_user_buffers,
                                       symm_buffers, symm_count,
                                       gda_mlx5_device_sq{sq_buf, sq_dbrec, sq_db, sq_depth},
                                       gda_mlx5_device_cq{cq_buf, cq_dbrec}}};
}

}  // namespace rocshmem
