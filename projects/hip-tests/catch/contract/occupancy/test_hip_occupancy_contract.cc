/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
__global__ void OccupancyKernel(int* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && output != nullptr) {
    *output = 1;
  }
}
}

// @asserts: hipOccupancyMaxActiveBlocksPerMultiprocessor - reports a non-negative active-block count for a kernel
HIP_TEST_CASE(Contract_Occupancy_MaxActiveBlocksPerMultiprocessor_ReturnsNonNegativeValue) {
  int max_active_blocks = -1;

  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, OccupancyKernel, 1, 0));

  REQUIRE(max_active_blocks >= 0);
}

// @asserts: hipOccupancyMaxPotentialBlockSize - suggests a non-negative min grid size and a positive block size for a kernel
HIP_TEST_CASE(Contract_Occupancy_MaxPotentialBlockSize_ReturnsUsableValues) {
  int min_grid_size = 0;
  int block_size = 0;

  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, OccupancyKernel, 0, 0));

  REQUIRE(min_grid_size >= 0);
  REQUIRE(block_size > 0);
}

// @asserts: hipOccupancyAvailableDynamicSMemPerBlock - reports available dynamic shared memory not exceeding the device's per-block limit
HIP_TEST_CASE(Contract_Occupancy_AvailableDynamicSmem_IsWithinDeviceLimit) {
  int current_device = 0;
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDevice(&current_device));
  HIP_CHECK(hipGetDeviceProperties(&properties, current_device));

  int max_active_blocks = 0;
  size_t dynamic_smem_size = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, OccupancyKernel, 1, 0));
  REQUIRE(max_active_blocks > 0);
  HIP_CHECK(hipOccupancyAvailableDynamicSMemPerBlock(&dynamic_smem_size, OccupancyKernel,
                                                     max_active_blocks, 1));

  REQUIRE(dynamic_smem_size <= properties.sharedMemPerBlock);
}
