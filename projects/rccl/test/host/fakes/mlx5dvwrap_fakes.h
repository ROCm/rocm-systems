/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for src/misc/mlx5dvwrap.cc.

#ifndef RCCL_TEST_HOST_MLX5DVWRAP_FAKES_H_
#define RCCL_TEST_HOST_MLX5DVWRAP_FAKES_H_

#include <functional>

#include "mlx5/mlx5dvwrap.h"

extern std::function<ncclResult_t()> g_mlx5dvSymbols;
extern std::function<bool(struct ibv_device*)> g_mlx5dvIsSupported;
extern std::function<ncclResult_t(struct ibv_context*, char*, size_t)> g_mlx5dvGetDataDirectSysfsPath;
extern std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int)>
    g_mlx5dvRegDmaBufMr;
extern std::function<struct ibv_mr*(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int)>
    g_mlx5dvDirectRegDmaBufMr;
extern std::function<ncclResult_t(struct ibv_context*, struct mlx5dv_context*)> g_mlx5dvQueryDevice;
extern std::function<struct ibv_qp*(struct ibv_context*, struct ibv_qp_init_attr_ex*,
                                    struct mlx5dv_qp_init_attr*)>
    g_mlx5dvCreateQp;

void ResetMlx5dvwrapFakes();

#endif  // RCCL_TEST_HOST_MLX5DVWRAP_FAKES_H_
