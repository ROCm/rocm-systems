/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "mlx5/mlx5dvwrap.h"
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>

#ifdef NCCL_BUILD_MLX5DV
#include <infiniband/mlx5dv.h>
#else
#include "mlx5/mlx5dvcore.h"
#endif
#include "mlx5/mlx5dvsymbols.h"

static std::once_flag initOnceFlag;
static ncclResult_t initResult;
struct ncclMlx5dvSymbols mlx5dvSymbols;

ncclResult_t wrap_mlx5dv_symbols(void) {
  std::call_once(initOnceFlag, []() { initResult = buildMlx5dvSymbols(&mlx5dvSymbols); });
  return initResult;
}

/* CHECK_NOT_NULL: helper macro to check for NULL symbol */
#define CHECK_NOT_NULL(container, internal_name) \
  if (container.internal_name == NULL) { \
    WARN("NET/MLX5: lib wrapper not initialized."); \
    return ncclInternalError; \
  }

/* Same as CHECK_NOT_NULL, and sets errno so a caller can fall back. */
#define CHECK_NOT_NULL_ERRNO(container, internal_name, err) \
  if (container.internal_name == NULL) { \
    WARN("NET/MLX5: lib wrapper not initialized."); \
    errno = (err); \
    return ncclInternalError; \
  }

#define MLX5DV_PTR_CHECK_ERRNO(container, internal_name, call, retval, error_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  retval = container.call; \
  if (retval == error_retval) { \
    WARN("NET/MLX5: Call to " name " failed with error %s", strerror(errno)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define MLX5DV_INT_CHECK_RET_ERRNO(container, internal_name, call, success_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  int ret = container.call; \
  if (ret != success_retval) { \
    WARN("NET/MLX5: Call to " name " failed with error %s", strerror(errno)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

bool wrap_mlx5dv_is_supported(struct ibv_device* device) {
  if (mlx5dvSymbols.mlx5dv_internal_is_supported == NULL) {
    return 0;
  }
  return mlx5dvSymbols.mlx5dv_internal_is_supported(device);
}

ncclResult_t wrap_mlx5dv_get_data_direct_sysfs_path(struct ibv_context* context, char* buf, size_t buf_len) {
  if (mlx5dvSymbols.mlx5dv_internal_get_data_direct_sysfs_path == NULL) {
    /* Stock rdma-core (e.g. 50.0) does not export MLX5_1.25. Treat as "not data-direct". */
    return ncclInvalidArgument;
  }
  int ret = mlx5dvSymbols.mlx5dv_internal_get_data_direct_sysfs_path(context, buf, buf_len);
  if (ret == 0) return ncclSuccess;
  /* ENODEV can happen if the devices is not data-direct but mlx5 is used. It's not an error*/
  if (ret == ENODEV) return ncclInvalidArgument;
  INFO(NCCL_NET, "NET/MLX5: Call to mlx5dv_internal_get_data_direct_sysfs_path failed with error %s errno %d",
       strerror(ret), ret);
  return ncclSystemError;
}

/* DMA-BUF support */
ncclResult_t wrap_mlx5dv_reg_dmabuf_mr(struct ibv_mr** ret, struct ibv_pd* pd, uint64_t offset, size_t length,
                                       uint64_t iova, int fd, int access, int mlx5_access) {
  CHECK_NOT_NULL_ERRNO(mlx5dvSymbols, mlx5dv_internal_reg_dmabuf_mr, EOPNOTSUPP);
  *ret = mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access, mlx5_access);
  if (*ret == NULL) {
    WARN("NET/MLX5: Call to mlx5dv_reg_dmabuf_mr failed with error %s", strerror(errno));
    return ncclSystemError;
  }
  return ncclSuccess;
}

struct ibv_mr* wrap_direct_mlx5dv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova,
                                                int fd, int access, int mlx5_access) {
  if (mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr == NULL) {
    // ENOSYS, not EOPNOTSUPP. The kernel DMA-BUF probe treats EOPNOTSUPP as
    // "ibv_reg_dmabuf_mr is unsupported" and then registers GPU memory with
    // ibv_reg_mr_iova2. A missing mlx5 symbol is not that.
    errno = ENOSYS;
    return NULL;
  }
  return mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access, mlx5_access);
}

/* One WARN per protection domain. MR cache misses would otherwise repeat it. */
static bool warnDmabufFallbackOnce(struct ibv_pd* pd) {
  constexpr int kMaxWarned = 32;
  static std::mutex mu;
  static struct ibv_pd* warned[kMaxWarned];
  static int nWarned = 0;
  std::lock_guard<std::mutex> lock(mu);
  for (int i = 0; i < nWarned; i++) {
    if (warned[i] == pd) return false;
  }
  if (nWarned < kMaxWarned) warned[nWarned++] = pd;
  return true;
}

ncclResult_t wrap_reg_dmabuf_mr(struct ibv_mr** ret, struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova,
                                int fd, int access, bool dataDirect) {
  if (dataDirect) {
    // Call the symbol directly. wrap_mlx5dv_reg_dmabuf_mr warns on every failure,
    // and MR cache misses would repeat that for a NIC that stays on the fallback.
    if (mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr == NULL) {
      errno = EOPNOTSUPP;
    } else {
      *ret = mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access,
                                                        MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
      if (*ret != NULL) return ncclSuccess;
    }
    int savedErrno = errno;
    if (warnDmabufFallbackOnce(pd)) {
      WARN("NET/IB: mlx5dv_reg_dmabuf_mr failed (%s); falling back to ibv_reg_dmabuf_mr", strerror(savedErrno));
    }
    errno = savedErrno;
  }
  return wrap_ibv_reg_dmabuf_mr(ret, pd, offset, length, iova, fd, access);
}

ncclResult_t wrap_mlx5dv_query_device(struct ibv_context* ctx_in, struct mlx5dv_context* attrs_out) {
  MLX5DV_INT_CHECK_RET_ERRNO(mlx5dvSymbols, mlx5dv_internal_query_device,
                             mlx5dv_internal_query_device(ctx_in, attrs_out), 0, "mlx5dv_query_device");
}

struct ibv_qp* wrap_mlx5dv_create_qp(struct ibv_context* context, struct ibv_qp_init_attr_ex* qp_attr,
                                     struct mlx5dv_qp_init_attr* mlx5_qp_attr) {
  if (mlx5dvSymbols.mlx5dv_internal_create_qp == NULL) {
    return 0;
  }
  return mlx5dvSymbols.mlx5dv_internal_create_qp(context, qp_attr, mlx5_qp_attr);
}
