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

HIP_TEST_CASE(Contract_Graph_CreateDestroy_Succeeds) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  REQUIRE(graph != nullptr);
}

HIP_TEST_CASE(Contract_Graph_AddEmptyNode_Succeeds) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  REQUIRE(node != nullptr);
}

HIP_TEST_CASE(Contract_Graph_AddMemcpyNode1D_RoundTripsBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x51);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t h2d_node = nullptr;
  hipGraphNode_t d2h_node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d_node, graph, nullptr, 0, device_ptr, src.data(), src.size(),
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h_node, graph, &h2d_node, 1, dst.data(), device_ptr,
                                    dst.size(), hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_Graph_AddMemsetNode_FillsDeviceBuffer) {
  hip::contract::ContractCleanup cleanup;
  constexpr uint8_t pattern = 0x6d;
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipMemsetParams params{};
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t memset_node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, dst.size()));
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  params.dst = device_ptr;
  params.value = pattern;
  params.pitch = dst.size();
  params.elementSize = sizeof(uint8_t);
  params.width = dst.size();
  params.height = 1;

  HIP_CHECK(hipGraphAddMemsetNode(&memset_node, graph, nullptr, 0, &params));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(dst.data(), device_ptr, dst.size(), hipMemcpyDeviceToHost));

  for (const auto value : dst) {
    REQUIRE(value == pattern);
  }
}

HIP_TEST_CASE(Contract_Graph_InstantiateLaunchSynchronize_Succeeds) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}
