/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstring>

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

void HostNoop(void* /*user_data*/) {}
}  // namespace

HIP_TEST_CASE(Contract_GraphNodeSetters_KernelNodeSetParams_UpdatesLaunchValue) {
  hip::contract::ContractCleanup cleanup;
  int initial = kInitialValue;
  int updated = kUpdatedValue;
  int* device_value = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t kernel_node = nullptr;

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([&] { (void)hipFree(device_value); });
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  // Add the node with the initial value, then mutate its parameters on the graph
  // node (pre-instantiation) to the updated value. The instantiated graph must
  // reflect the mutated parameters, and a get-params round-trip must report the
  // swapped argument pointer array.
  void* initial_args[] = {&device_value, &initial};
  auto initial_params = KernelNodeParams(initial_args);
  HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &initial_params));

  void* updated_args[] = {&device_value, &updated};
  auto updated_params = KernelNodeParams(updated_args);
  HIP_CHECK(hipGraphKernelNodeSetParams(kernel_node, &updated_params));

  // The getter reports the configured launch geometry and function; the runtime
  // copies the argument array into internal storage, so the reported
  // kernelParams pointer is not required to alias the caller's array. The
  // observable effect of the swap is verified below via the launched write.
  hipKernelNodeParams read_back{};
  HIP_CHECK(hipGraphKernelNodeGetParams(kernel_node, &read_back));
  REQUIRE(read_back.func == reinterpret_cast<void*>(WriteValueKernel));
  REQUIRE(read_back.gridDim.x == 1);
  REQUIRE(read_back.blockDim.x == 1);

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(ReadDeviceInt(device_value) == kUpdatedValue);
}

HIP_TEST_CASE(Contract_GraphNodeSetters_MemcpyNodeSetParams1D_UpdatesCopySource) {
  hip::contract::ContractCleanup cleanup;
  constexpr size_t kBytes = sizeof(int);
  int source_a = kInitialValue;
  int source_b = kUpdatedValue;
  int* device_value = nullptr;
  int host_result = 0;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t memcpy_node = nullptr;

  HIP_CHECK(hipMalloc(&device_value, kBytes));
  cleanup.Add([&] { (void)hipFree(device_value); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  // Add a host-to-device copy from source_a, then rewrite the node to copy from
  // source_b before instantiation. The launched graph must observe the updated
  // source value on the device buffer.
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_node, graph, nullptr, 0, device_value, &source_a, kBytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpy_node, device_value, &source_b, kBytes,
                                          hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(&host_result, device_value, kBytes, hipMemcpyDeviceToHost));
  REQUIRE(host_result == kUpdatedValue);
}

HIP_TEST_CASE(Contract_GraphNodeSetters_HostNodeSetParams_RoundTripsFnAndUserData) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t host_node = nullptr;
  int first_user_data = 0;
  int second_user_data = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  hipHostNodeParams initial_params{};
  initial_params.fn = HostNoop;
  initial_params.userData = &first_user_data;
  HIP_CHECK(hipGraphAddHostNode(&host_node, graph, nullptr, 0, &initial_params));

  // Swap the user-data pointer on the host node; the getter must report the new
  // function/user-data pair.
  hipHostNodeParams updated_params{};
  updated_params.fn = HostNoop;
  updated_params.userData = &second_user_data;
  HIP_CHECK(hipGraphHostNodeSetParams(host_node, &updated_params));

  hipHostNodeParams read_back{};
  HIP_CHECK(hipGraphHostNodeGetParams(host_node, &read_back));
  REQUIRE(read_back.fn == HostNoop);
  REQUIRE(read_back.userData == &second_user_data);
}

HIP_TEST_CASE(Contract_GraphNodeSetters_EventNodesSetEvent_RoundTripSwappedEvents) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t record_node = nullptr;
  hipGraphNode_t wait_node = nullptr;
  hipEvent_t first = nullptr;
  hipEvent_t second = nullptr;

  HIP_CHECK(hipEventCreate(&first));
  cleanup.Add([&] { (void)hipEventDestroy(first); });
  HIP_CHECK(hipEventCreate(&second));
  cleanup.Add([&] { (void)hipEventDestroy(second); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  // Both the event-record and event-wait nodes must accept a swapped event and
  // report it back through their getters.
  HIP_CHECK(hipGraphAddEventRecordNode(&record_node, graph, nullptr, 0, first));
  HIP_CHECK(hipGraphEventRecordNodeSetEvent(record_node, second));
  hipEvent_t record_event = nullptr;
  HIP_CHECK(hipGraphEventRecordNodeGetEvent(record_node, &record_event));
  REQUIRE(record_event == second);

  HIP_CHECK(hipGraphAddEventWaitNode(&wait_node, graph, &record_node, 1, first));
  HIP_CHECK(hipGraphEventWaitNodeSetEvent(wait_node, second));
  hipEvent_t wait_event = nullptr;
  HIP_CHECK(hipGraphEventWaitNodeGetEvent(wait_node, &wait_event));
  REQUIRE(wait_event == second);
}

HIP_TEST_CASE(Contract_GraphNodeSetters_KernelNodeCopyAttributes_PropagatesToDestination) {
  hip::contract::ContractCleanup cleanup;
  int value = kInitialValue;
  int* device_value = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t source_node = nullptr;
  hipGraphNode_t dest_node = nullptr;

  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([&] { (void)hipFree(device_value); });
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  void* args[] = {&device_value, &value};
  auto params = KernelNodeParams(args);
  HIP_CHECK(hipGraphAddKernelNode(&source_node, graph, nullptr, 0, &params));
  HIP_CHECK(hipGraphAddKernelNode(&dest_node, graph, nullptr, 0, &params));

  // Set the cooperative attribute on the source node, then copy attributes to the
  // destination node.
  hipKernelNodeAttrValue source_attr{};
  source_attr.cooperative = 1;
  HIP_CHECK(hipGraphKernelNodeSetAttribute(source_node, hipKernelNodeAttributeCooperative,
                                           &source_attr));

  HIP_CHECK(hipGraphKernelNodeCopyAttributes(source_node, dest_node));

  hipKernelNodeAttrValue dest_attr{};
  HIP_CHECK(hipGraphKernelNodeGetAttribute(dest_node, hipKernelNodeAttributeCooperative,
                                           &dest_attr));
#if HT_AMD
  // On AMD the copy propagates the cooperative attribute to the destination node.
  REQUIRE(dest_attr.cooperative == source_attr.cooperative);
#else
  // On NVIDIA hipGraphKernelNodeCopyAttributes maps to
  // cudaGraphKernelNodeCopyAttributes, which copies the access-policy-window
  // attribute but does NOT propagate the cooperative flag (probe-confirmed:
  // set/get of cooperative on the source node round-trips, but after the copy the
  // destination reads 0). The copy still succeeds; only the cooperative field is
  // not carried. Assert the observed NVIDIA behavior so the divergence is
  // explicit; if the CUDA copy is reconciled to carry cooperative, this branch is
  // where the expectation changes back to matching the source value.
  REQUIRE(dest_attr.cooperative == 0);
#endif
}
