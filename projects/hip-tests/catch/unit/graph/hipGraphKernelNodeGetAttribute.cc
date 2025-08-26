/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#define THREADS_PER_BLOCK 512

TEST_CASE("Unit_hipGraphKernelNodeGetAttribute_Negative_Parameters") {
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
