/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the mlx5 direct-verbs wrappers defined by
// src/misc/mlx5dvwrap.cc. Every default fails: acquisitions return NULL or an
// error and wrap_mlx5dv_is_supported reports false, so an un-hooked path never
// sees an mlx5 device.

#ifndef RCCL_TEST_HOST_MLX5DVWRAP_FAKES_H_
#define RCCL_TEST_HOST_MLX5DVWRAP_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "nccl.h"

struct ibv_context;
struct ibv_device;
struct ibv_pd;
struct ibv_mr;
struct ibv_qp;
struct ibv_qp_init_attr_ex;
struct mlx5dv_context;
struct mlx5dv_qp_init_attr;

extern std::function<ncclResult_t()> g_mlx5dvSymbols;
extern std::function<bool(struct ibv_device* /*device*/)> g_mlx5dvIsSupported;
extern std::function<ncclResult_t(struct ibv_context* /*context*/, char* /*buf*/, size_t /*buf_len*/)>
    g_mlx5dvGetDataDirectSysfsPath;
extern std::function<ncclResult_t(struct ibv_mr** /*ret*/, struct ibv_pd* /*pd*/, uint64_t /*offset*/,
                                  size_t /*length*/, uint64_t /*iova*/, int /*fd*/, int /*access*/,
                                  int /*mlx5_access*/)>
    g_mlx5dvRegDmabufMr;
extern std::function<ncclResult_t(struct ibv_mr** /*ret*/, struct ibv_pd* /*pd*/, uint64_t /*offset*/,
                                  size_t /*length*/, uint64_t /*iova*/, int /*fd*/, int /*access*/,
                                  bool /*dataDirect*/)>
    g_regDmabufMr;
extern std::function<struct ibv_mr*(struct ibv_pd* /*pd*/, uint64_t /*offset*/, size_t /*length*/,
                                    uint64_t /*iova*/, int /*fd*/, int /*access*/, int /*mlx5_access*/)>
    g_mlx5dvDirectRegDmabufMr;
extern std::function<ncclResult_t(struct ibv_context* /*ctx_in*/, struct mlx5dv_context* /*attrs_out*/)>
    g_mlx5dvQueryDevice;
extern std::function<struct ibv_qp*(struct ibv_context* /*context*/, struct ibv_qp_init_attr_ex* /*qp_attr*/,
                                    struct mlx5dv_qp_init_attr* /*mlx5_qp_attr*/)>
    g_mlx5dvCreateQp;

void ResetMlx5dvwrapFakes();

#endif  // RCCL_TEST_HOST_MLX5DVWRAP_FAKES_H_
