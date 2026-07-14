/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr int kInitialValue = 7;

__global__ void WriteValueKernel(int* output, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = value;
  }
}

hipKernelNodeParams KernelNodeParams(void** args) {
  hipKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(WriteValueKernel);
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  params.sharedMemBytes = 0;
  params.kernelParams = args;
  params.extra = nullptr;
  return params;
}
}  // namespace

HIP_TEST_CASE(Contract_GraphNodeAttributes_Cooperative_RoundTrips) {
  hip::contract::ContractCleanup cleanup;
  int value = kInitialValue;
  int* device_value = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t kernel_node = nullptr;

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([&] { (void)hipFree(device_value); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  void* args[] = {&device_value, &value};
  auto params = KernelNodeParams(args);
  HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &params));

  // Request the cooperative attribute on the kernel node. Runtime paths that do
  // not support this attribute must be treated as a clean skip rather than a
  // contract failure.
  hipKernelNodeAttrValue set_value{};
  set_value.cooperative = 1;
  const hipError_t set_status = hipGraphKernelNodeSetAttribute(
      kernel_node, hipKernelNodeAttributeCooperative, &set_value);
  if (set_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Cooperative kernel node attribute is not supported by this runtime path.");
  }
  HIP_CHECK(set_status);

  // The attribute set on the node must be observable through a subsequent get.
  hipKernelNodeAttrValue get_value{};
  const hipError_t get_status = hipGraphKernelNodeGetAttribute(
      kernel_node, hipKernelNodeAttributeCooperative, &get_value);
  if (get_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Cooperative kernel node attribute is not supported by this runtime path.");
  }
  HIP_CHECK(get_status);

  REQUIRE(get_value.cooperative == 1);
}

HIP_TEST_CASE(Contract_GraphNodeAttributes_AccessPolicyWindow_RoundTrips) {
  int device_ordinal = 0;
  HIP_CHECK(hipGetDevice(&device_ordinal));

  // The access policy window is only meaningful on devices that report a
  // positive maximum window size. A reported size of zero indicates the device
  // does not support access policy windows, so the test skips cleanly.
  int max_window_size = 0;
  HIP_CHECK(hipDeviceGetAttribute(&max_window_size,
                                  hipDeviceAttributeAccessPolicyMaxWindowSize, device_ordinal));
  if (max_window_size == 0) {
    HIP_SKIP_TEST("Access policy windows are not supported on this device.");
  }

  hip::contract::ContractCleanup cleanup;
  constexpr size_t kBufferSize = 256;
  int value = kInitialValue;
  int* device_value = nullptr;
  void* device_buffer = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t kernel_node = nullptr;

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([&] { (void)hipFree(device_value); });
  HIP_CHECK(hipMalloc(&device_buffer, kBufferSize));
  cleanup.Add([&] { (void)hipFree(device_buffer); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  void* args[] = {&device_value, &value};
  auto params = KernelNodeParams(args);
  HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &params));

  // Build a portable access policy window over the tiny device buffer, clamping
  // the window size to the device-reported maximum.
  hipKernelNodeAttrValue set_value{};
  set_value.accessPolicyWindow.base_ptr = device_buffer;
  set_value.accessPolicyWindow.num_bytes =
      std::min(kBufferSize, static_cast<size_t>(max_window_size));
  set_value.accessPolicyWindow.hitRatio = 0.5f;
  set_value.accessPolicyWindow.hitProp = hipAccessPropertyPersisting;
  set_value.accessPolicyWindow.missProp = hipAccessPropertyStreaming;

  const hipError_t set_status = hipGraphKernelNodeSetAttribute(
      kernel_node, hipKernelNodeAttributeAccessPolicyWindow, &set_value);
  if (set_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Access policy window kernel node attribute is not supported by this runtime path.");
  }
  HIP_CHECK(set_status);

  // The window configured on the node must round-trip through a subsequent get.
  hipKernelNodeAttrValue get_value{};
  const hipError_t get_status = hipGraphKernelNodeGetAttribute(
      kernel_node, hipKernelNodeAttributeAccessPolicyWindow, &get_value);
  if (get_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Access policy window kernel node attribute is not supported by this runtime path.");
  }
  HIP_CHECK(get_status);

  REQUIRE(get_value.accessPolicyWindow.base_ptr == set_value.accessPolicyWindow.base_ptr);
  REQUIRE(get_value.accessPolicyWindow.num_bytes == set_value.accessPolicyWindow.num_bytes);
  REQUIRE(get_value.accessPolicyWindow.hitRatio == set_value.accessPolicyWindow.hitRatio);
  REQUIRE(get_value.accessPolicyWindow.hitProp == set_value.accessPolicyWindow.hitProp);
  REQUIRE(get_value.accessPolicyWindow.missProp == set_value.accessPolicyWindow.missProp);
}

HIP_TEST_CASE(Contract_GraphNodeAttributes_SetAttribute_RejectsInvalidInputs) {
  hip::contract::ContractCleanup cleanup;
  int value = kInitialValue;
  int* device_value = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t kernel_node = nullptr;
  hipGraphNode_t empty_node = nullptr;

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([&] { (void)hipFree(device_value); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  void* args[] = {&device_value, &value};
  auto params = KernelNodeParams(args);
  HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &params));
  HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, nullptr, 0));

  hipKernelNodeAttrValue attr_value{};
  attr_value.cooperative = 1;

  // A null node handle must be rejected with an invalid-value error.
  REQUIRE(hipGraphKernelNodeSetAttribute(nullptr, hipKernelNodeAttributeCooperative,
                                         &attr_value) == hipErrorInvalidValue);

  // Setting a kernel node attribute on a non-kernel (empty) node must be
  // rejected with an invalid-value error.
  REQUIRE(hipGraphKernelNodeSetAttribute(empty_node, hipKernelNodeAttributeCooperative,
                                         &attr_value) == hipErrorInvalidValue);

  // An attribute id outside the supported set must be rejected with an
  // invalid-value error. A large sentinel value is used instead of a named
  // upper-bound enumerator so this contract compiles portably on backends that
  // do not define such an enumerator.
  const auto invalid_attr = static_cast<hipKernelNodeAttrID>(0x7fffffff);
  REQUIRE(hipGraphKernelNodeSetAttribute(kernel_node, invalid_attr,
                                         &attr_value) == hipErrorInvalidValue);
}
