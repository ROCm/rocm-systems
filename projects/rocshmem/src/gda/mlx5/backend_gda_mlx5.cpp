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

void* GDABackend::mlx5_dv_dlopen() {
  void* dv_handle{nullptr};
  dv_handle = dlopen("libmlx5.so", RTLD_LAZY);
  if (!dv_handle) {
    DPRINTF("Could not open libmlx5.so. Returning\n");
  }
  return dv_handle;
}

int GDABackend::mlx5_dv_dl_init() {
  mlx5dv_handle_ = mlx5_dv_dlopen();
  if (!mlx5dv_handle_)
    return ROCSHMEM_ERROR;

  DLSYM_HELPER(mlx5dv, mlx5dv_, mlx5dv_handle_, init_obj);
  return ROCSHMEM_SUCCESS;
}

void GDABackend::mlx5_initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  mlx5dv_cq cq_out;
  mlx5dv_obj mlx_obj;
  mlx_obj.cq.in = cqs[conn_num];
  mlx_obj.cq.out = &cq_out;
  mlx5dv.init_obj(&mlx_obj, MLX5DV_OBJ_CQ);
  dump_mlx5dv_cq(&cq_out, conn_num);

  /*
   * struct mlx5dv_cq {
   *   void                    *buf;
   *   __be32                  *dbrec;
   *   uint32_t                cqe_cnt;
   *   uint32_t                cqe_size;
   *   void                    *cq_uar;
   *   uint32_t                cqn;
   *   uint64_t                comp_mask;
   * };
   */

  gpu_qp->cq_buf = reinterpret_cast<mlx5_cqe64*>(cq_out.buf);
  gpu_qp->cq_cnt = cq_out.cqe_cnt;
  gpu_qp->cq_log_cnt = log2(cq_out.cqe_cnt);
  gpu_qp->cq_dbrec = cq_out.dbrec;

  mlx5dv_qp qp_out;
  mlx_obj.qp.in = qps[conn_num];
  mlx_obj.qp.out = &qp_out;
  mlx5dv.init_obj(&mlx_obj, MLX5DV_OBJ_QP);
  dump_mlx5dv_qp(&qp_out, conn_num);

  /*
   * struct mlx5dv_qp {
   *   __be32 *dbrec;
   *   struct {
   *     void *buf;
   *     uint32_t wqe_cnt;
   *     uint32_t stride;
   *   } sq;
   *   struct {
   *     void *buf;
   *     uint32_t wqe_cnt;
   *     uint32_t stride;
   *   } rq;
   *   struct {
   *     void *reg;
   *     uint32_t size;
   *   } bf;
   *   uint64_t comp_mask;
   *   off_t uar_mmap_offset;
   *   uint32_t tirn;
   *   uint32_t tisn;
   *   uint32_t rqn;
   *   uint32_t sqn;
   *   uint64_t tir_icm_addr;
   * };
   */

  gpu_qp->dbrec = &qp_out.dbrec[1]; // points to two pointers: 0 -> MLX5_REC_DBR, 1 -> MLX5_SND_DBR
  gpu_qp->sq_buf = reinterpret_cast<uint64_t*>(qp_out.sq.buf);
  gpu_qp->sq_wqe_cnt = qp_out.sq.wqe_cnt;
  gpu_qp->rkey = htobe32(heap_rkey[conn_num % num_pes]);
  gpu_qp->lkey = htobe32(heap_mr->lkey);
  gpu_qp->qp_num = qps[conn_num]->qp_num;
  gpu_qp->inline_threshold = inline_threshold;
  // The 2 in qp_out.bf.size * 2 below facilitates the switching between blue flame registers

  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  void* gpu_ptr{nullptr};
  rocm_memory_lock_to_fine_grain(qp_out.bf.reg, qp_out.bf.size * 2, &gpu_ptr, hip_dev_id);
  gpu_qp->db.ptr = reinterpret_cast<uint64_t*>(gpu_ptr);
}

}  // namespace rocshmem
