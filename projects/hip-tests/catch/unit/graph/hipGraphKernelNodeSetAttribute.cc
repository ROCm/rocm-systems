/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

namespace {
constexpr std::array<hipAccessProperty, 3> kAccessProperties{
    hipAccessPropertyNormal, hipAccessPropertyStreaming, hipAccessPropertyPersisting};
}  // anonymous namespace

static bool CompareAccessPolicyWindow(const hipKernelNodeAttrValue& lhs,
                                      const hipKernelNodeAttrValue& rhs) {
  return lhs.accessPolicyWindow.base_ptr == rhs.accessPolicyWindow.base_ptr &&
         lhs.accessPolicyWindow.num_bytes == rhs.accessPolicyWindow.num_bytes &&
         lhs.accessPolicyWindow.hitRatio == rhs.accessPolicyWindow.hitRatio &&
         lhs.accessPolicyWindow.hitProp == rhs.accessPolicyWindow.hitProp &&
         lhs.accessPolicyWindow.missProp == rhs.accessPolicyWindow.missProp;
}

TEST_CASE("Unit_hipGraphKernelNodeSetAttribute_Positive_AccessPolicyWindow") {
  constexpr int N = 1024;

  const auto hit_prop = GENERATE(from_range(begin(kAccessProperties), end(kAccessProperties)));
  const auto miss_prop = GENERATE(from_range(begin(kAccessProperties), end(kAccessProperties) - 1));

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

  int max_window_size;
  HIP_CHECK(
      hipDeviceGetAttribute(&max_window_size, hipDeviceAttributeAccessPolicyMaxWindowSize, 0));

  hipKernelNodeAttrValue node_attribute_1;
  node_attribute_1.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(A_d);
  node_attribute_1.accessPolicyWindow.num_bytes =
      std::min<unsigned long>(static_cast<unsigned long>(max_window_size), sizeof(int) * N);
  node_attribute_1.accessPolicyWindow.hitRatio = 0.6;
  node_attribute_1.accessPolicyWindow.hitProp = hit_prop;
  node_attribute_1.accessPolicyWindow.missProp = miss_prop;

  HIP_CHECK(hipGraphKernelNodeSetAttribute(graph_node, hipKernelNodeAttributeAccessPolicyWindow,
                                           &node_attribute_1));

  hipKernelNodeAttrValue node_attribute_2;
  HIP_CHECK(hipGraphKernelNodeGetAttribute(graph_node, hipKernelNodeAttributeAccessPolicyWindow,
                                           &node_attribute_2));

  REQUIRE(CompareAccessPolicyWindow(node_attribute_1, node_attribute_2));

  HIP_CHECK(hipGraphDestroy(graph));

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}

TEST_CASE("Unit_hipGraphKernelNodeSetAttribute_Positive_Cooperative") {
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

  hipKernelNodeAttrValue node_attribute_1;
  node_attribute_1.cooperative = 2;

  HIP_CHECK(hipGraphKernelNodeSetAttribute(graph_node, hipKernelNodeAttributeCooperative,
                                           &node_attribute_1));

  hipKernelNodeAttrValue node_attribute_2;
  HIP_CHECK(hipGraphKernelNodeGetAttribute(graph_node, hipKernelNodeAttributeCooperative,
                                           &node_attribute_2));

  REQUIRE(node_attribute_1.cooperative == node_attribute_2.cooperative);

  HIP_CHECK(hipGraphDestroy(graph));

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}

TEST_CASE("Unit_hipGraphKernelNodeSetAttribute_Negative_Parameters") {
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

  int max_window_size;
  HIP_CHECK(
      hipDeviceGetAttribute(&max_window_size, hipDeviceAttributeAccessPolicyMaxWindowSize, 0));

  hipKernelNodeAttrValue node_attribute;
  node_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(A_d);
  node_attribute.accessPolicyWindow.num_bytes =
      std::min<unsigned long>(static_cast<unsigned long>(max_window_size), sizeof(int) * N);
  node_attribute.accessPolicyWindow.hitRatio = 0.6;
  node_attribute.accessPolicyWindow.hitProp = hipAccessPropertyPersisting;
  node_attribute.accessPolicyWindow.missProp = hipAccessPropertyStreaming;

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphKernelNodeSetAttribute(
                        nullptr, hipKernelNodeAttributeAccessPolicyWindow, &node_attribute),
                    hipErrorInvalidValue);
  }

  SECTION("node is not a kernel node") {
    hipGraphNode_t empty_node;
    HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, nullptr, 0));
    HIP_CHECK_ERROR(hipGraphKernelNodeSetAttribute(
                        empty_node, hipKernelNodeAttributeAccessPolicyWindow, &node_attribute),
                    hipErrorInvalidValue);
  }

  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(hipGraphKernelNodeSetAttribute(graph_node, static_cast<hipKernelNodeAttrID>(-1),
                                                   &node_attribute),
                    hipErrorInvalidValue);
  }

#if HT_AMD  // segfaults on NVIDIA
  SECTION("value == nullptr") {
    HIP_CHECK_ERROR(hipGraphKernelNodeSetAttribute(
                        graph_node, hipKernelNodeAttributeAccessPolicyWindow, nullptr),
                    hipErrorInvalidValue);
  }
#endif

  SECTION("accessPolicyWindow.num_bytes > accessPolicyMaxWindowSize") {
    node_attribute.accessPolicyWindow.num_bytes = max_window_size + 1;
    HIP_CHECK_ERROR(hipGraphKernelNodeSetAttribute(
                        graph_node, hipKernelNodeAttributeAccessPolicyWindow, &node_attribute),
                    hipErrorInvalidValue);
  }

  SECTION("accessPolicyWindow.hitRatio < 0") {
    node_attribute.accessPolicyWindow.hitRatio = -0.6;
    HIP_CHECK_ERROR(hipGraphKernelNodeSetAttribute(
                        graph_node, hipKernelNodeAttributeAccessPolicyWindow, &node_attribute),
                    hipErrorInvalidValue);
  }

  SECTION("accessPolicyWindow.hitRatio > 1.0") {
    node_attribute.accessPolicyWindow.hitRatio = 1.1;
    HIP_CHECK_ERROR(hipGraphKernelNodeSetAttribute(
                        graph_node, hipKernelNodeAttributeAccessPolicyWindow, &node_attribute),
                    hipErrorInvalidValue);
  }

  SECTION("accessPolicyWindow.missProp == hipAccessPropertyPersisting") {
    node_attribute.accessPolicyWindow.missProp = hipAccessPropertyPersisting;
    HIP_CHECK_ERROR(hipGraphKernelNodeSetAttribute(
                        graph_node, hipKernelNodeAttributeAccessPolicyWindow, &node_attribute),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}