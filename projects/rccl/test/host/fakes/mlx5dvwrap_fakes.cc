/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "mlx5dvwrap_fakes.h"

#include <cerrno>

#include "signature-drift.h"

static ncclResult_t DefaultMlx5dvSymbols() { return ncclSystemError; }
std::function<ncclResult_t()> g_mlx5dvSymbols = DefaultMlx5dvSymbols;

static bool DefaultMlx5dvIsSupported(struct ibv_device*) { return false; }
std::function<bool(struct ibv_device*)> g_mlx5dvIsSupported = DefaultMlx5dvIsSupported;

static ncclResult_t DefaultMlx5dvGetDataDirectSysfsPath(struct ibv_context*, char*, size_t) {
  return ncclInvalidArgument;
}
std::function<ncclResult_t(struct ibv_context*, char*, size_t)> g_mlx5dvGetDataDirectSysfsPath =
    DefaultMlx5dvGetDataDirectSysfsPath;

static ncclResult_t DefaultMlx5dvRegDmaBufMr(struct ibv_mr** mr, struct ibv_pd*, uint64_t, size_t, uint64_t, int,
                                             int, int) {
  if (mr) *mr = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int)>
    g_mlx5dvRegDmaBufMr = DefaultMlx5dvRegDmaBufMr;

static struct ibv_mr* DefaultMlx5dvDirectRegDmaBufMr(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int) {
  errno = EOPNOTSUPP;
  return nullptr;
}
std::function<struct ibv_mr*(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int)>
    g_mlx5dvDirectRegDmaBufMr = DefaultMlx5dvDirectRegDmaBufMr;

static ncclResult_t DefaultMlx5dvQueryDevice(struct ibv_context*, struct mlx5dv_context*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, struct mlx5dv_context*)> g_mlx5dvQueryDevice =
    DefaultMlx5dvQueryDevice;

static struct ibv_qp* DefaultMlx5dvCreateQp(struct ibv_context*, struct ibv_qp_init_attr_ex*,
                                             struct mlx5dv_qp_init_attr*) {
  return nullptr;
}
std::function<struct ibv_qp*(struct ibv_context*, struct ibv_qp_init_attr_ex*, struct mlx5dv_qp_init_attr*)>
    g_mlx5dvCreateQp = DefaultMlx5dvCreateQp;

ASSERT_HOOK_MATCHES_PROD(g_mlx5dvSymbols, wrap_mlx5dv_symbols);
ASSERT_HOOK_MATCHES_PROD(g_mlx5dvIsSupported, wrap_mlx5dv_is_supported);
ASSERT_HOOK_MATCHES_PROD(g_mlx5dvGetDataDirectSysfsPath, wrap_mlx5dv_get_data_direct_sysfs_path);
ASSERT_HOOK_MATCHES_PROD(g_mlx5dvRegDmaBufMr, wrap_mlx5dv_reg_dmabuf_mr);
ASSERT_HOOK_MATCHES_PROD(g_mlx5dvDirectRegDmaBufMr, wrap_direct_mlx5dv_reg_dmabuf_mr);
ASSERT_HOOK_MATCHES_PROD(g_mlx5dvQueryDevice, wrap_mlx5dv_query_device);
ASSERT_HOOK_MATCHES_PROD(g_mlx5dvCreateQp, wrap_mlx5dv_create_qp);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t wrap_mlx5dv_symbols() { return g_mlx5dvSymbols(); }
bool wrap_mlx5dv_is_supported(struct ibv_device* device) { return g_mlx5dvIsSupported(device); }
ncclResult_t wrap_mlx5dv_get_data_direct_sysfs_path(struct ibv_context* context, char* path, size_t size) {
  return g_mlx5dvGetDataDirectSysfsPath(context, path, size);
}
ncclResult_t wrap_mlx5dv_reg_dmabuf_mr(struct ibv_mr** mr, struct ibv_pd* pd, uint64_t offset, size_t length,
                                       uint64_t iova, int fd, int access, int mlx5Access) {
  return g_mlx5dvRegDmaBufMr(mr, pd, offset, length, iova, fd, access, mlx5Access);
}
struct ibv_mr* wrap_direct_mlx5dv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova,
                                                int fd, int access, int mlx5Access) {
  return g_mlx5dvDirectRegDmaBufMr(pd, offset, length, iova, fd, access, mlx5Access);
}
ncclResult_t wrap_mlx5dv_query_device(struct ibv_context* context, struct mlx5dv_context* attrs) {
  return g_mlx5dvQueryDevice(context, attrs);
}
struct ibv_qp* wrap_mlx5dv_create_qp(struct ibv_context* context, struct ibv_qp_init_attr_ex* attr,
                                     struct mlx5dv_qp_init_attr* dvAttr) {
  return g_mlx5dvCreateQp(context, attr, dvAttr);
}

void ResetMlx5dvwrapFakes() {
  g_mlx5dvSymbols = DefaultMlx5dvSymbols;
  g_mlx5dvIsSupported = DefaultMlx5dvIsSupported;
  g_mlx5dvGetDataDirectSysfsPath = DefaultMlx5dvGetDataDirectSysfsPath;
  g_mlx5dvRegDmaBufMr = DefaultMlx5dvRegDmaBufMr;
  g_mlx5dvDirectRegDmaBufMr = DefaultMlx5dvDirectRegDmaBufMr;
  g_mlx5dvQueryDevice = DefaultMlx5dvQueryDevice;
  g_mlx5dvCreateQp = DefaultMlx5dvCreateQp;
}
