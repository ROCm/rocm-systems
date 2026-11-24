/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/mlx5dvwrap.cc fakes. See mlx5dvwrap_fakes.h.

#include "mlx5dvwrap_fakes.h"

#include "mlx5/mlx5dvwrap.h"
#include "signature-drift.h"

static ncclResult_t DefaultMlx5dvSymbols() { return ncclSystemError; }
std::function<ncclResult_t()> g_mlx5dvSymbols = DefaultMlx5dvSymbols;

static bool DefaultMlx5dvIsSupported(struct ibv_device*) { return false; }
std::function<bool(struct ibv_device*)> g_mlx5dvIsSupported = DefaultMlx5dvIsSupported;

static ncclResult_t DefaultMlx5dvGetDataDirectSysfsPath(struct ibv_context*, char*, size_t) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, char*, size_t)> g_mlx5dvGetDataDirectSysfsPath =
    DefaultMlx5dvGetDataDirectSysfsPath;

static ncclResult_t DefaultMlx5dvRegDmabufMr(struct ibv_mr** ret, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int,
                                             int) {
  if (ret) *ret = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int)>
    g_mlx5dvRegDmabufMr = DefaultMlx5dvRegDmabufMr;

// dataDirect false forwards to the ibv seam so the IB-CAST registration tests still see that call.
// dataDirect true fails closed: those tests hook this seam, and an unhooked request must not look like ibv.
static ncclResult_t DefaultRegDmabufMr(struct ibv_mr** ret, struct ibv_pd* pd, uint64_t offset, size_t length,
                                       uint64_t iova, int fd, int access, bool dataDirect) {
  if (dataDirect) {
    if (ret) *ret = nullptr;
    return ncclSystemError;
  }
  return wrap_ibv_reg_dmabuf_mr(ret, pd, offset, length, iova, fd, access);
}
std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, bool)> g_regDmabufMr =
    DefaultRegDmabufMr;

static struct ibv_mr* DefaultMlx5dvDirectRegDmabufMr(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int) {
  return nullptr;
}
std::function<struct ibv_mr*(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int)> g_mlx5dvDirectRegDmabufMr =
    DefaultMlx5dvDirectRegDmabufMr;

static ncclResult_t DefaultMlx5dvQueryDevice(struct ibv_context*, struct mlx5dv_context*) { return ncclSystemError; }
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
ASSERT_HOOK_MATCHES_PROD(g_mlx5dvRegDmabufMr, wrap_mlx5dv_reg_dmabuf_mr);
ASSERT_HOOK_MATCHES_PROD(g_regDmabufMr, wrap_reg_dmabuf_mr);
ASSERT_HOOK_MATCHES_PROD(g_mlx5dvDirectRegDmabufMr, wrap_direct_mlx5dv_reg_dmabuf_mr);
ASSERT_HOOK_MATCHES_PROD(g_mlx5dvQueryDevice, wrap_mlx5dv_query_device);
ASSERT_HOOK_MATCHES_PROD(g_mlx5dvCreateQp, wrap_mlx5dv_create_qp);

ncclResult_t wrap_mlx5dv_symbols(void) { return g_mlx5dvSymbols(); }
bool wrap_mlx5dv_is_supported(struct ibv_device* device) { return g_mlx5dvIsSupported(device); }
ncclResult_t wrap_mlx5dv_get_data_direct_sysfs_path(struct ibv_context* context, char* buf, size_t buf_len) {
  return g_mlx5dvGetDataDirectSysfsPath(context, buf, buf_len);
}
ncclResult_t wrap_mlx5dv_reg_dmabuf_mr(struct ibv_mr** ret, struct ibv_pd* pd, uint64_t offset, size_t length,
                                       uint64_t iova, int fd, int access, int mlx5_access) {
  return g_mlx5dvRegDmabufMr(ret, pd, offset, length, iova, fd, access, mlx5_access);
}
ncclResult_t wrap_reg_dmabuf_mr(struct ibv_mr** ret, struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova,
                                int fd, int access, bool dataDirect) {
  return g_regDmabufMr(ret, pd, offset, length, iova, fd, access, dataDirect);
}
struct ibv_mr* wrap_direct_mlx5dv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova,
                                                int fd, int access, int mlx5_access) {
  return g_mlx5dvDirectRegDmabufMr(pd, offset, length, iova, fd, access, mlx5_access);
}
ncclResult_t wrap_mlx5dv_query_device(struct ibv_context* ctx_in, struct mlx5dv_context* attrs_out) {
  return g_mlx5dvQueryDevice(ctx_in, attrs_out);
}
struct ibv_qp* wrap_mlx5dv_create_qp(struct ibv_context* context, struct ibv_qp_init_attr_ex* qp_attr,
                                     struct mlx5dv_qp_init_attr* mlx5_qp_attr) {
  return g_mlx5dvCreateQp(context, qp_attr, mlx5_qp_attr);
}

void ResetMlx5dvwrapFakes() {
  g_mlx5dvSymbols = DefaultMlx5dvSymbols;
  g_mlx5dvIsSupported = DefaultMlx5dvIsSupported;
  g_mlx5dvGetDataDirectSysfsPath = DefaultMlx5dvGetDataDirectSysfsPath;
  g_mlx5dvRegDmabufMr = DefaultMlx5dvRegDmabufMr;
  g_regDmabufMr = DefaultRegDmabufMr;
  g_mlx5dvDirectRegDmabufMr = DefaultMlx5dvDirectRegDmabufMr;
  g_mlx5dvQueryDevice = DefaultMlx5dvQueryDevice;
  g_mlx5dvCreateQp = DefaultMlx5dvCreateQp;
}
