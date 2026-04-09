/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#define THREADS_PER_BLOCK 512

HIP_TEST_CASE(Unit_hipGraphKernelNodeGetAttribute_Negative_Parameters) {
  constexpr int N = 1024;

  int *A_d, *B_d, *C_d;
  HIP_CHECK(hipMalloc(&A_d, sizeof(int) * N));
  HIP_CHECK(hipMalloc(&B_d, sizeof(int) * N));
  HIP_CHECK(hipMalloc(&C_d, sizeof(int) * N));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipKernelNodeParams node_params{};
  node_params.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  node_params.gridDim = dim3(N / THREADS_PER_BLOCK, 1, 1);
  node_params.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);

  size_t N_elem{N};
  void* kernel_params[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&N_elem)};
  node_params.kernelParams = reinterpret_cast<void**>(kernel_params);

  hipGraphNode_t graph_node;
  HIP_CHECK(hipGraphAddKernelNode(&graph_node, graph, nullptr, 0, &node_params));

  hipKernelNodeAttrValue node_attribute;

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphKernelNodeGetAttribute(
                        nullptr, hipKernelNodeAttributeAccessPolicyWindow, &node_attribute),
                    hipErrorInvalidValue);
  }

  SECTION("node is not a kernel node") {
    hipGraphNode_t empty_node;
    HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, nullptr, 0));
    HIP_CHECK_ERROR(hipGraphKernelNodeGetAttribute(
                        empty_node, hipKernelNodeAttributeAccessPolicyWindow, &node_attribute),
                    hipErrorInvalidValue);
  }

  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(hipGraphKernelNodeGetAttribute(graph_node, static_cast<hipKernelNodeAttrID>(-1),
                                                   &node_attribute),
                    hipErrorInvalidValue);
  }

#if HT_AMD  // segfaults on NVIDIA
  SECTION("value == nullptr") {
    HIP_CHECK_ERROR(hipGraphKernelNodeGetAttribute(
                        graph_node, hipKernelNodeAttributeAccessPolicyWindow, nullptr),
                    hipErrorInvalidValue);
  }
#endif

  HIP_CHECK(hipGraphDestroy(graph));

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}
