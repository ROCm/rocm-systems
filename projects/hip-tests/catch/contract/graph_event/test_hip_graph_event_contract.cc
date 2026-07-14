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

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}
}

HIP_TEST_CASE(Contract_GraphEvent_AddEventRecordNode_RecordsEvent) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t event = nullptr;
  hipGraphNode_t record_node = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipEventCreate(&event));
  cleanup.Add([&] { (void)hipEventDestroy(event); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEventRecordNode(&record_node, graph, nullptr, 0, event));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipEventQuery(event));
}

HIP_TEST_CASE(Contract_GraphEvent_AddEventWaitNode_WaitsForRecordedEvent) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t event = nullptr;
  hipGraphNode_t record_node = nullptr;
  hipGraphNode_t wait_node = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipEventCreate(&event));
  cleanup.Add([&] { (void)hipEventDestroy(event); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEventRecordNode(&record_node, graph, nullptr, 0, event));
  HIP_CHECK(hipGraphAddEventWaitNode(&wait_node, graph, &record_node, 1, event));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}

HIP_TEST_CASE(Contract_GraphEvent_RecordThenWait_OrdersMemcpy) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x7b);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t copy_ready = nullptr;
  hipGraphNode_t h2d_node = nullptr;
  hipGraphNode_t record_node = nullptr;
  hipGraphNode_t wait_node = nullptr;
  hipGraphNode_t d2h_node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipEventCreate(&copy_ready));
  cleanup.Add([&] { (void)hipEventDestroy(copy_ready); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d_node, graph, nullptr, 0, device_ptr, src.data(), src.size(),
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddEventRecordNode(&record_node, graph, &h2d_node, 1, copy_ready));
  HIP_CHECK(hipGraphAddEventWaitNode(&wait_node, graph, &record_node, 1, copy_ready));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h_node, graph, &wait_node, 1, dst.data(), device_ptr,
                                    dst.size(), hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}
