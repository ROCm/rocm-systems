/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

extern "C" {
__global__ void NOPKernel() {}

__global__ void Kernel42(int* out) { *out = 42; }

// Interval is in millisecond
__global__ void Delay(uint32_t interval, const uint32_t ticks_per_ms) {
  while (interval--) {
    uint64_t start = clock();
    while (clock() - start < ticks_per_ms) {
    }
  }
}

__global__ void CoopKernel() {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  // TODO: remove syncthreads with grid sync when compiler issue is fixed
  // grid.sync();
  __syncthreads();
}
}