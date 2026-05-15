/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

__global__ void vectorAdd(float* Ad, float* Bd, size_t num_elements) {
  extern __shared__ float sBd[];
  int tx = threadIdx.x;
  for (size_t i = 0; i < num_elements / blockDim.x; i++) {
    sBd[tx + i * blockDim.x] = Ad[tx + i * blockDim.x] + 1.0f;
    Bd[tx + i * blockDim.x] = sBd[tx + i * blockDim.x];
  }

  const int remainder = static_cast<int>(num_elements % blockDim.x);
  if (tx < remainder) {
    const int idx = (num_elements - remainder) + tx;
    sBd[idx] = Ad[idx] + 1.0f;
    Bd[idx] = sBd[idx];
  }
}

/**
* @addtogroup hipLaunchKernelGGL hipLaunchKernelGGL
* @{
* @ingroup KernelTest
* `void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
   std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* Method to invocate kernel functions
*/

/**
 * Test Description
 * ------------------------
 *    - Assign max dynamic shared memory to kernel function and
 * verify the results.

 * Test source
 * ------------------------
 *    - catch/unit/kernel/hipDynamicShared2.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */
HIP_TEST_CASE(Unit_hipDynamicShared2) {
  hipFuncAttributes func_attributes{};
  HIP_CHECK(hipFuncGetAttributes(&func_attributes, reinterpret_cast<const void*>(&vectorAdd)));

  int max_shared_memory_per_block{};
  HIP_CHECK(hipDeviceGetAttribute(&max_shared_memory_per_block,
                                  hipDeviceAttributeMaxSharedMemoryPerBlock, 0));

  size_t threadPerBlock = 64;
  const size_t dynamic_shared_bytes = max_shared_memory_per_block - func_attributes.sharedSizeBytes;

  REQUIRE(dynamic_shared_bytes >= threadPerBlock * sizeof(float));
  const size_t num_elements = dynamic_shared_bytes / sizeof(float);

  std::vector<float> A(num_elements);
  std::vector<float> B(num_elements);
  for (int i = 0; i < num_elements; i++) {
    A[i] = 1.0f;
    B[i] = 1.0f;
  }

  float *Ad, *Bd;
  HIP_CHECK(hipMalloc(&Ad, dynamic_shared_bytes));
  HIP_CHECK(hipMalloc(&Bd, dynamic_shared_bytes));

  HIP_CHECK(hipMemcpy(Ad, A.data(), dynamic_shared_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B.data(), dynamic_shared_bytes, hipMemcpyHostToDevice));

  //  The sum of sharedSizeBytes (statically-allocated shared memory per block) and
  //  hipFuncAttributeMaxDynamicSharedMemorySize (dynamically-allocated shared memory per block)
  //  cannot exceed the size of hipDeviceAttributeMaxSharedMemoryPerBlock.
  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(&vectorAdd),
                                hipFuncAttributeMaxDynamicSharedMemorySize, dynamic_shared_bytes));

  hipLaunchKernelGGL(vectorAdd, dim3(1, 1, 1), dim3(threadPerBlock, 1, 1), dynamic_shared_bytes, 0,
                     Ad, Bd, num_elements);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(B.data(), Bd, dynamic_shared_bytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < num_elements; i++) {
    REQUIRE(B[i] > 1.0f);
    REQUIRE(B[i] < 3.0f);
  }
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
}

/**
 * End doxygen group KernelTest.
 * @}
 */
