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

HIP_TEST_CASE(Contract_GraphClone_ClonedEmptyGraph_InstantiatesAndLaunches) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraph_t clone = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));
  HIP_CHECK(hipGraphClone(&clone, graph));
  cleanup.Add([&] { (void)hipGraphDestroy(clone); });
  HIP_CHECK(hipGraphInstantiate(&graph_exec, clone, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
}

HIP_TEST_CASE(Contract_GraphClone_ClonedMemcpyGraph_RoundTripsBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x45);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraph_t clone = nullptr;
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
  HIP_CHECK(hipGraphClone(&clone, graph));
  cleanup.Add([&] { (void)hipGraphDestroy(clone); });
  HIP_CHECK(hipGraphInstantiate(&graph_exec, clone, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_GraphClone_OriginalAndClone_AreDistinctHandles) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraph_t clone = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));
  HIP_CHECK(hipGraphClone(&clone, graph));
  cleanup.Add([&] { (void)hipGraphDestroy(clone); });

  REQUIRE(clone != nullptr);
  REQUIRE(clone != graph);
}
