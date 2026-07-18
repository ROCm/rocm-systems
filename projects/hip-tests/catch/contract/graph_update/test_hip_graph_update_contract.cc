/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kElementCount = 128;

constexpr size_t kSmallElementCount = 64;

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

hipMemsetParams MakeMemsetParams(void* dst, size_t width, uint8_t value) {
  hipMemsetParams params{};
  params.dst = dst;
  params.value = value;
  params.pitch = width;
  params.elementSize = sizeof(uint8_t);
  params.width = width;
  params.height = 1;
  return params;
}

// Plain C callback: does not call HIP APIs, only mutates data through userData.
void IncrementCounter(void* userData) {
  auto* counter = static_cast<int32_t*>(userData);
  *counter += 1;
}
}

HIP_TEST_CASE(Contract_GraphUpdate_ExecMemcpyNodeSetParams1D_UpdatesCopySource) {
  hip::contract::ContractCleanup cleanup;
  const auto original = MakePattern(0x10);
  const auto updated = MakePattern(0x44);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t h2d_node = nullptr;
  hipGraphNode_t d2h_node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, original.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d_node, graph, nullptr, 0, device_ptr, original.data(),
                                    original.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h_node, graph, &h2d_node, 1, dst.data(), device_ptr,
                                    dst.size(), hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(graph_exec, h2d_node, device_ptr, updated.data(),
                                              updated.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == updated);
}

HIP_TEST_CASE(Contract_GraphUpdate_ExecMemsetNodeSetParams_UpdatesFillValue) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t original_pattern = 0x2a;
  constexpr uint8_t updated_pattern = 0x5c;
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipMemsetParams params{};
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t memset_node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  params.dst = device_ptr;
  params.value = original_pattern;
  params.pitch = dst.size();
  params.elementSize = sizeof(uint8_t);
  params.width = dst.size();
  params.height = 1;

  HIP_CHECK(hipGraphAddMemsetNode(&memset_node, graph, nullptr, 0, &params));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  params.value = updated_pattern;
  HIP_CHECK(hipGraphExecMemsetNodeSetParams(graph_exec, memset_node, &params));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == updated_pattern);
  }
}

HIP_TEST_CASE(Contract_GraphUpdate_ExecUpdate_AppliesChangedMemsetValue) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t original_pattern = 0x11;
  constexpr uint8_t updated_pattern = 0x22;
  std::array<uint8_t, kSmallElementCount> dst{};
  void* device_ptr = nullptr;
  hipGraph_t graph_a = nullptr;
  hipGraph_t graph_b = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t memset_a = nullptr;
  hipGraphNode_t memset_b = nullptr;
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipGraphCreate(&graph_a, 0));
  cleanup.Add([graph_a] { (void)hipGraphDestroy(graph_a); });
  hipMemsetParams params_a = MakeMemsetParams(device_ptr, dst.size(), original_pattern);
  HIP_CHECK(hipGraphAddMemsetNode(&memset_a, graph_a, nullptr, 0, &params_a));

  HIP_CHECK(hipGraphCreate(&graph_b, 0));
  cleanup.Add([graph_b] { (void)hipGraphDestroy(graph_b); });
  hipMemsetParams params_b = MakeMemsetParams(device_ptr, dst.size(), updated_pattern);
  HIP_CHECK(hipGraphAddMemsetNode(&memset_b, graph_b, nullptr, 0, &params_b));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph_a, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  HIP_CHECK(hipGraphExecUpdate(graph_exec, graph_b, &error_node, &update_result));
  REQUIRE(update_result == hipGraphExecUpdateSuccess);

  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == updated_pattern);
  }
}

HIP_TEST_CASE(Contract_GraphUpdate_ExecUpdate_ReportsTopologyChanged) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t pattern = 0x5a;
  void* device_ptr = nullptr;
  hipGraph_t single_node_graph = nullptr;
  hipGraph_t two_node_graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipGraphNode_t single_memset = nullptr;
  hipGraphNode_t first_memset = nullptr;
  hipGraphNode_t second_memset = nullptr;
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult update_result = hipGraphExecUpdateSuccess;

  HIP_CHECK(hipMalloc(&device_ptr, kSmallElementCount));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  HIP_CHECK(hipGraphCreate(&single_node_graph, 0));
  cleanup.Add([single_node_graph] { (void)hipGraphDestroy(single_node_graph); });
  hipMemsetParams single_params = MakeMemsetParams(device_ptr, kSmallElementCount, pattern);
  HIP_CHECK(hipGraphAddMemsetNode(&single_memset, single_node_graph, nullptr, 0, &single_params));

  HIP_CHECK(hipGraphCreate(&two_node_graph, 0));
  cleanup.Add([two_node_graph] { (void)hipGraphDestroy(two_node_graph); });
  hipMemsetParams two_params = MakeMemsetParams(device_ptr, kSmallElementCount, pattern);
  HIP_CHECK(hipGraphAddMemsetNode(&first_memset, two_node_graph, nullptr, 0, &two_params));
  HIP_CHECK(hipGraphAddMemsetNode(&second_memset, two_node_graph, &first_memset, 1, &two_params));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, single_node_graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  // The runtime may signal a topology mismatch either through a non-success
  // return code or through a non-success result enum. Accept both to remain
  // portable across implementations.
  const hipError_t status =
      hipGraphExecUpdate(graph_exec, two_node_graph, &error_node, &update_result);
  const bool reported_failure =
      (status != hipSuccess) || (update_result != hipGraphExecUpdateSuccess);
  REQUIRE(reported_failure);
}

HIP_TEST_CASE(Contract_GraphUpdate_ExecHostNodeSetParams_UpdatesCallbackUserData) {
  hip::contract::ContractCleanup cleanup;
  constexpr int32_t kInitialValue = 0;
  int32_t counter_a = kInitialValue;
  int32_t counter_b = kInitialValue;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t host_node = nullptr;

  hipHostNodeParams params{};
  params.fn = IncrementCounter;
  params.userData = &counter_a;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddHostNode(&host_node, graph, nullptr, 0, &params));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  params.userData = &counter_b;
  HIP_CHECK(hipGraphExecHostNodeSetParams(graph_exec, host_node, &params));

  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(counter_a == kInitialValue);
  REQUIRE(counter_b == kInitialValue + 1);
}

HIP_TEST_CASE(Contract_GraphUpdate_ExecChildGraphNodeSetParams_UpdatesEmbeddedMemset) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t original_pattern = 0x33;
  constexpr uint8_t updated_pattern = 0x44;
  std::array<uint8_t, kSmallElementCount> dst{};
  void* device_ptr = nullptr;
  hipGraph_t parent = nullptr;
  hipGraph_t child_original = nullptr;
  hipGraph_t child_updated = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t child_memset = nullptr;
  hipGraphNode_t updated_memset = nullptr;
  hipGraphNode_t child_node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipGraphCreate(&child_original, 0));
  cleanup.Add([child_original] { (void)hipGraphDestroy(child_original); });
  hipMemsetParams original_params = MakeMemsetParams(device_ptr, dst.size(), original_pattern);
  HIP_CHECK(hipGraphAddMemsetNode(&child_memset, child_original, nullptr, 0, &original_params));

  HIP_CHECK(hipGraphCreate(&child_updated, 0));
  cleanup.Add([child_updated] { (void)hipGraphDestroy(child_updated); });
  hipMemsetParams updated_params = MakeMemsetParams(device_ptr, dst.size(), updated_pattern);
  HIP_CHECK(hipGraphAddMemsetNode(&updated_memset, child_updated, nullptr, 0, &updated_params));

  HIP_CHECK(hipGraphCreate(&parent, 0));
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child_original));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, parent, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  HIP_CHECK(hipGraphExecChildGraphNodeSetParams(graph_exec, child_node, child_updated));

  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == updated_pattern);
  }
}

HIP_TEST_CASE(Contract_GraphUpdate_ExecEventRecordNodeSetEvent_UpdatesRecordedEvent) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t event_one = nullptr;
  hipEvent_t event_two = nullptr;
  hipGraphNode_t record_node = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipEventCreate(&event_one));
  cleanup.Add([event_one] { (void)hipEventDestroy(event_one); });
  HIP_CHECK(hipEventCreate(&event_two));
  cleanup.Add([event_two] { (void)hipEventDestroy(event_two); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEventRecordNode(&record_node, graph, nullptr, 0, event_one));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  HIP_CHECK(hipGraphExecEventRecordNodeSetEvent(graph_exec, record_node, event_two));

  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipEventQuery(event_two));
}

HIP_TEST_CASE(Contract_GraphUpdate_ExecEventWaitNodeSetEvent_AcceptsSwappedEvent) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t event_one = nullptr;
  hipEvent_t event_two = nullptr;
  hipGraphNode_t wait_node = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipEventCreate(&event_one));
  cleanup.Add([event_one] { (void)hipEventDestroy(event_one); });
  HIP_CHECK(hipEventCreate(&event_two));
  cleanup.Add([event_two] { (void)hipEventDestroy(event_two); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEventWaitNode(&wait_node, graph, nullptr, 0, event_one));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  // Ensure the swapped-in event is already recorded and complete so the wait
  // node has nothing outstanding to block on when the graph launches.
  HIP_CHECK(hipEventRecord(event_two, stream));
  HIP_CHECK(hipEventSynchronize(event_two));

  HIP_CHECK(hipGraphExecEventWaitNodeSetEvent(graph_exec, wait_node, event_two));

  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}

HIP_TEST_CASE(Contract_GraphUpdate_ExecMemcpyNodeSetParams1D_UpdatesCopyDestination) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x73);
  std::array<uint8_t, kElementCount> first_dst{};
  std::array<uint8_t, kElementCount> second_dst{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t h2d_node = nullptr;
  hipGraphNode_t d2h_node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d_node, graph, nullptr, 0, device_ptr, src.data(), src.size(),
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h_node, graph, &h2d_node, 1, first_dst.data(), device_ptr,
                                    first_dst.size(), hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(graph_exec, d2h_node, second_dst.data(), device_ptr,
                                              second_dst.size(), hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(first_dst != src);
  REQUIRE(second_dst == src);
}
