/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * Test Description
 * ------------------------
 *  - Basic test checks default behaviour
 * Test source
 * ------------------------
 *  - kernel/hipConfigureCall.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_ConfigureCall) {
  struct dim3 grid_dim{};
  struct dim3 block_dim{};
  size_t shared_memory_size = 1024;

  HIP_CHECK(hipConfigureCall(grid_dim, block_dim, shared_memory_size));
}

/**
 * Test Description
 * ------------------------
 *  - Basic test verifies parameters
 * Test source
 * ------------------------
 *  - kernel/hipConfigureCall.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_ConfigureCall_CheckParams) {
  struct dim3 grid_dim{16, 8, 1};
  struct dim3 test_grid_dim{};
  struct dim3 block_dim{16, 8, 1};
  struct dim3 test_block_dim{};
  size_t shmem_size = 1024;
  size_t test_shmem_size = 0;
  hipStream_t test_stream;

  HIP_CHECK(hipConfigureCall(grid_dim, block_dim, shmem_size));

  HIP_CHECK(
      __hipPopCallConfiguration(&test_grid_dim, &test_block_dim, &test_shmem_size, &test_stream));

  REQUIRE(test_grid_dim.x == grid_dim.x);
  REQUIRE(test_grid_dim.y == grid_dim.y);
  REQUIRE(test_grid_dim.z == grid_dim.z);

  REQUIRE(test_block_dim.x == block_dim.x);
  REQUIRE(test_block_dim.y == block_dim.y);
  REQUIRE(test_block_dim.z == block_dim.z);

  REQUIRE(test_shmem_size == shmem_size);
}
