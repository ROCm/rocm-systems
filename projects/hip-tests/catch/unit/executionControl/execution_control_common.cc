/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "execution_control_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

__global__ void kernel() {}

__global__ void kernel2() {}

__global__ void kernel_42(int* val) { *val = 42; }

__global__ void coop_kernel() {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  grid.sync();
}