/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr int kInitialValue = 7;
constexpr int kUpdatedValue = 11;

__global__ void WriteValueKernel(int* output, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = value;
  }
}

int ReadDeviceInt(int* device_ptr) {
  int value = 0;
  HIP_CHECK(hipMemcpy(&value, device_ptr, sizeof(value), hipMemcpyDeviceToHost));
  return value;
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
}

// @asserts: hipGraphAddKernelNode - a kernel node instantiated and launched in a graph runs and writes its configured value
HIP_TEST_CASE(Contract_GraphKernel_AddKernelNode_WritesExpectedValue) {
  hip::contract::ContractCleanup cleanup;
  int value = kInitialValue;
  int* device_value = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t kernel_node = nullptr;

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  void* args[] = {&device_value, &value};
  auto params = KernelNodeParams(args);
  HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &params));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(ReadDeviceInt(device_value) == kInitialValue);
}

// @asserts: hipGraphKernelNodeGetParams - reading back a kernel node's params returns the func, dims, and shared-mem it was configured with
HIP_TEST_CASE(Contract_GraphKernel_KernelNodeGetParams_ReturnsConfiguredParams) {
  hip::contract::ContractCleanup cleanup;
  int value = kInitialValue;
  int* device_value = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t kernel_node = nullptr;

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  void* args[] = {&device_value, &value};
  auto params = KernelNodeParams(args);
  HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &params));

  hipKernelNodeParams readback{};
  HIP_CHECK(hipGraphKernelNodeGetParams(kernel_node, &readback));

  REQUIRE(readback.func == params.func);
  REQUIRE(readback.gridDim.x == params.gridDim.x);
  REQUIRE(readback.blockDim.x == params.blockDim.x);
  REQUIRE(readback.sharedMemBytes == params.sharedMemBytes);
}

// @asserts: hipGraphExecKernelNodeSetParams - updating an instantiated graph's kernel-node params changes the launched value without re-instantiation
HIP_TEST_CASE(Contract_GraphKernel_ExecKernelNodeSetParams_UpdatesLaunchValue) {
  hip::contract::ContractCleanup cleanup;
  int initial_value = kInitialValue;
  int updated_value = kUpdatedValue;
  int* device_value = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t kernel_node = nullptr;

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  void* initial_args[] = {&device_value, &initial_value};
  auto initial_params = KernelNodeParams(initial_args);
  HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &initial_params));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  void* updated_args[] = {&device_value, &updated_value};
  auto updated_params = KernelNodeParams(updated_args);
  HIP_CHECK(hipGraphExecKernelNodeSetParams(graph_exec, kernel_node, &updated_params));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(ReadDeviceInt(device_value) == kUpdatedValue);
}
