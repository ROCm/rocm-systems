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

#include "util.hpp"
#include "gda/backend_gda.hpp"
#include "gda/mlx5/mlx5dv_core.hpp"
#include "gda/mlx5/mlx5_ifc_core.hpp"

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
  // mlx5 provider can support up to 28B of inline data in a WQE
  inline_threshold = sizeof(gda_mlx5_wqe_inline_data::data);
  for (size_t i = 0; i < mlx5_qps.size(); i++) {
    int err = mlx5dv.create_qp(mlx5_qps[i], context, pd_orig, sq_length);
    CHECK_ZERO(err, "mlx5dv::create_qp");
  }
}

void GDABackend::mlx5_initialize_gpu_qp(QueuePair* gpu_qp, int conn_num) {
  mlx5_devx_qp& qp = mlx5_qps[conn_num];
  qp.dump(conn_num);

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

  gpu_qp->mlx5_cq = gda_mlx5_device_cq(reinterpret_cast<mlx5_cqe64*>(qp.cq), qp.cq_dbrec);

  int hip_dev_id{-1};
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  void* gpu_db_ptr{nullptr};
  // not necessary to switch between BlueFlame buffer halves when using it as a doorbell only
  rocm_memory_lock_to_fine_grain(qp.uar->reg_addr, MLX5_DB_BLUEFLAME_BUFFER_SIZE,
                                 &gpu_db_ptr, hip_dev_id);

  // qp.dbrec points to two __be32 values: RQ dbrec at MLX5_RCV_DBR and SQ dbrec at MLX5_SND_DBR
  gpu_qp->mlx5_sq = gda_mlx5_device_sq{reinterpret_cast<gda_mlx5_wqe*>(qp.sq),
                                       &qp.qp_dbrec[MLX5_SND_DBR],
                                       reinterpret_cast<gda_mlx5_doorbell*>(gpu_db_ptr),
                                       static_cast<uint16_t>(qp.sq_depth)};

  gpu_qp->rkey = htobe32(heap_rkey[conn_num % num_pes]);
  gpu_qp->lkey = htobe32(heap_mr->lkey);
  gpu_qp->qp_num = qp.qpn;
  gpu_qp->inline_threshold = inline_threshold;
}

}  // namespace rocshmem
