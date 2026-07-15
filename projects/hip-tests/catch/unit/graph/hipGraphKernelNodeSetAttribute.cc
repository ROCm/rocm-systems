/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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

HIP_TEST_CASE(Unit_hipGraphKernelNodeSetAttribute_Positive_AccessPolicyWindow) {
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

HIP_TEST_CASE(Unit_hipGraphKernelNodeSetAttribute_Positive_Cooperative) {
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

HIP_TEST_CASE(Unit_hipGraphKernelNodeSetAttribute_Negative_Parameters) {
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

// Test kernel graph launch with kernel dimensions that are not cleanly divisible.
// Uses global size of 1000 and local size of 256
HIP_TEST_CASE(Unit_hipGraphKernelNodeSetAttribute_Positive_RemainderWithClusterDimension) {
  // Skip when cluster launch is not supported on this device.
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (!prop.clusterLaunch) {
    HIP_SKIP_TEST("Device does not support cluster launch; skipping T5 remainder+cluster test");
  }

  // Dimensions chosen so that remainder != 0 AND grid % cluster == 0.
  constexpr int kBlockDim = 256;      // local work size
  constexpr int kGlobalElems = 1000;  // conceptual global; kGlobalElems % kBlockDim = 232 ≠ 0
  constexpr int kGrid = kGlobalElems / kBlockDim;  // = 3 (integer division)
  constexpr int kCluster = 3;                      // divides kGrid: 3 % 3 == 0
  constexpr int kBufSize = kGrid * kBlockDim;      // = 768 (elements actually written)

  static_assert(kGlobalElems % kBlockDim != 0, "remainder must be non-zero for this test");
  static_assert(kGrid % kCluster == 0, "cluster must evenly divide grid");

  // Allocate and initialise host arrays.
  std::vector<int> A_h(kBufSize), B_h(kBufSize), C_h(kBufSize, 0);
  for (int i = 0; i < kBufSize; ++i) {
    A_h[i] = i;
    B_h[i] = i * 2;
  }

  // Allocate device arrays.
  int *A_d, *B_d, *C_d;
  HIP_CHECK(hipMalloc(&A_d, kBufSize * sizeof(int)));
  HIP_CHECK(hipMalloc(&B_d, kBufSize * sizeof(int)));
  HIP_CHECK(hipMalloc(&C_d, kBufSize * sizeof(int)));
  HIP_CHECK(hipMemcpy(A_d, A_h.data(), kBufSize * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h.data(), kBufSize * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(C_d, 0, kBufSize * sizeof(int)));

  // Build graph with a single kernel node.
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipKernelNodeParams node_params{};
  node_params.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  node_params.gridDim = dim3(kGrid, 1, 1);
  node_params.blockDim = dim3(kBlockDim, 1, 1);

  size_t N_elem = static_cast<size_t>(kBufSize);
  void* kernel_params[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&N_elem)};
  node_params.kernelParams = reinterpret_cast<void**>(kernel_params);

  hipGraphNode_t graph_node;
  HIP_CHECK(hipGraphAddKernelNode(&graph_node, graph, nullptr, 0, &node_params));

  // Set the cluster dimension attribute on the kernel node.
  hipKernelNodeAttrValue cluster_attr{};
  cluster_attr.clusterDim = {kCluster, 1, 1};
  HIP_CHECK(hipGraphKernelNodeSetAttribute(graph_node, hipLaunchAttributeClusterDimension,
                                           &cluster_attr));

  // Read back and verify the attribute was stored correctly.
  hipKernelNodeAttrValue readback_attr{};
  HIP_CHECK(hipGraphKernelNodeGetAttribute(graph_node, hipLaunchAttributeClusterDimension,
                                           &readback_attr));
  REQUIRE(readback_attr.clusterDim.x == static_cast<unsigned>(kCluster));
  REQUIRE(readback_attr.clusterDim.y == 1u);
  REQUIRE(readback_attr.clusterDim.z == 1u);

  // Instantiate, launch, and synchronise.
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipGraphExec_t graph_exec;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Copy results back and validate every element.
  HIP_CHECK(hipMemcpy(C_h.data(), C_d, kBufSize * sizeof(int), hipMemcpyDeviceToHost));
  for (int i = 0; i < kBufSize; ++i) {
    REQUIRE(C_h[i] == A_h[i] + B_h[i]);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
}
