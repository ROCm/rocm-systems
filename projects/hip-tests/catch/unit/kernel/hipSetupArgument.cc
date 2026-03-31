/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_array_common.hh>

__global__ void add_vectors(int* a, int* b, int* result, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    result[tid] = a[tid] + b[tid];
  }
}

HIP_TEST_CASE(Unit_hipSetupArgument_Simple) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(1, 1, 1);

  HIP_CHECK(hipConfigureCall(grid_dim, block_dim, 0, 0));

  int arg = 42;
  HIP_CHECK(hipSetupArgument(&arg, sizeof(int), 0));
}

/**
 * Test Description
 * ------------------------
 *  - Verifies that arguments sent to the kernel with hipSetupArgument are correct by executing
 *    kernel that calculates sum of two vectors, doing the same calculation on CPU and checking if
 * the results are the same, which proves that the arguments used in kernel are the proper ones
 *
 * Test source
 * ------------------------
 *  - unit/kernel/hipSetupArgument.cc
 */
HIP_TEST_CASE(Unit_hipSetupArgument_Execute_Kernel_And_Check_Result) {
  constexpr auto block_size = 32;
  auto vec_size = 256;
  size_t block_num = static_cast<size_t>(std::ceil(static_cast<float>(vec_size) / block_size));
  dim3 grid_dim(block_num, 1, 1);
  dim3 block_dim(block_size, 1, 1);

  HIP_CHECK(hipConfigureCall(grid_dim, block_dim, 0, 0));

  std::vector<int> vec_a, vec_b, vec_c;
  for (size_t i = 0; i < vec_size; i++) {
    vec_a.push_back(getRandom<int>());
    vec_b.push_back(getRandom<int>());
    vec_c.push_back(vec_a[i] + vec_b[i]);
  }

  size_t vec_size_in_bytes = sizeof(int) * vec_size;
  int *dev_a, *dev_b, *dev_c;
  HIP_CHECK(hipMalloc(&dev_a, vec_size_in_bytes));
  HIP_CHECK(hipMalloc(&dev_b, vec_size_in_bytes));
  HIP_CHECK(hipMalloc(&dev_c, vec_size_in_bytes));

  HIP_CHECK(hipMemcpy(dev_a, vec_a.data(), vec_size_in_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dev_b, vec_b.data(), vec_size_in_bytes, hipMemcpyHostToDevice));

  HIP_CHECK(hipSetupArgument(&dev_a, sizeof(int*), 0));
  HIP_CHECK(hipSetupArgument(&dev_b, sizeof(int*), sizeof(int*)));
  HIP_CHECK(hipSetupArgument(&dev_c, sizeof(int*), 2 * sizeof(int*)));
  HIP_CHECK(hipSetupArgument(&vec_size, sizeof(int), 3 * sizeof(int*)));

  HIP_CHECK(hipLaunchByPtr((const void*)add_vectors));

  std::vector<int> vec_res(vec_size);
  HIP_CHECK(hipMemcpy(vec_res.data(), dev_c, vec_size_in_bytes, hipMemcpyDeviceToHost));
  REQUIRE(vec_res == vec_c);

  HIP_CHECK(hipFree(dev_c));
  HIP_CHECK(hipFree(dev_b));
  HIP_CHECK(hipFree(dev_a));
}
