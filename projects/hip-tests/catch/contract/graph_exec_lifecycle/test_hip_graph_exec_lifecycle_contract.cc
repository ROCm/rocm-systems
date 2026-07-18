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

void BuildMemcpyGraph(hipGraph_t* graph, void* device_ptr, const void* src, void* dst, size_t size) {
  hipGraphNode_t h2d_node = nullptr;
  hipGraphNode_t d2h_node = nullptr;

  HIP_CHECK(hipGraphCreate(graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d_node, *graph, nullptr, 0, device_ptr, src, size,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h_node, *graph, &h2d_node, 1, dst, device_ptr, size,
                                    hipMemcpyDeviceToHost));
}
}  // namespace

HIP_TEST_CASE(Contract_GraphExecLifecycle_InstantiateWithFlags_ZeroFlag_LaunchSucceeds) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x24);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  BuildMemcpyGraph(&graph, device_ptr, src.data(), dst.data(), dst.size());
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  HIP_CHECK(hipGraphInstantiateWithFlags(&graph_exec, graph, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

#if HIP_VERSION >= 60400000
HIP_TEST_CASE(Contract_GraphExecLifecycle_ExecGetFlags_ReflectsZero) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipGraphNode_t node = nullptr;
  unsigned long long flags = 1;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));
  HIP_CHECK(hipGraphInstantiateWithFlags(&graph_exec, graph, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphExecGetFlags(graph_exec, &flags));

  REQUIRE(flags == 0);
}

HIP_TEST_CASE(Contract_GraphExecLifecycle_ExecGetFlags_ReflectsAutoFreeOnLaunch) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipGraphNode_t node = nullptr;
  unsigned long long flags = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));
  HIP_CHECK(
      hipGraphInstantiateWithFlags(&graph_exec, graph, hipGraphInstantiateFlagAutoFreeOnLaunch));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphExecGetFlags(graph_exec, &flags));

  REQUIRE((flags & hipGraphInstantiateFlagAutoFreeOnLaunch) != 0);
}
#endif

HIP_TEST_CASE(Contract_GraphExecLifecycle_Upload_ThenLaunchSucceeds) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x51);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  BuildMemcpyGraph(&graph, device_ptr, src.data(), dst.data(), dst.size());
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  HIP_CHECK(hipGraphInstantiateWithFlags(&graph_exec, graph, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphUpload(graph_exec, stream));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_GraphExecLifecycle_InvalidArgs_AreRejected) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t node = nullptr;
  unsigned long long flags = 0;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  REQUIRE(hipGraphInstantiateWithFlags(nullptr, graph, 0) != hipSuccess);
  REQUIRE(hipGraphInstantiateWithFlags(&graph_exec, nullptr, 0) != hipSuccess);
  REQUIRE(hipGraphUpload(nullptr, stream) != hipSuccess);
#if HIP_VERSION >= 60400000
  REQUIRE(hipGraphExecGetFlags(nullptr, &flags) != hipSuccess);
#endif
}
