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
constexpr size_t kByteCount = 64;
constexpr uint8_t kSourcePattern = 0xAB;
constexpr uint8_t kBaselinePattern = 0x5C;

// Queries a node's enabled flag, translating an unsupported-capability code into
// a clean skip. Returns false (without touching the out flag) when the runtime
// path does not support the node enable/disable APIs; otherwise HIP_CHECKs any
// other error and returns true with the flag populated.
bool TryGetEnabled(hipGraphExec_t graph_exec, hipGraphNode_t node, unsigned int* enabled) {
  const hipError_t status = hipGraphNodeGetEnabled(graph_exec, node, enabled);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}

// Sets a node's enabled flag, translating an unsupported-capability code into a
// clean skip. Returns false when the runtime path does not support the node
// enable/disable APIs; otherwise HIP_CHECKs any other error and returns true.
bool TrySetEnabled(hipGraphExec_t graph_exec, hipGraphNode_t node, unsigned int enabled) {
  const hipError_t status = hipGraphNodeSetEnabled(graph_exec, node, enabled);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_GraphNodeEnabled_DefaultEnabled_ReportsOne) {
  hip::contract::ContractCleanup cleanup;
  std::array<uint8_t, kByteCount> host{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, host.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, device_ptr, host.data(), host.size(),
                                    hipMemcpyHostToDevice));

  // The enable/disable APIs operate on an instantiated executable graph, so the
  // graph must be instantiated before the node's enabled flag can be queried.
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  unsigned int enabled = 0;
  if (!TryGetEnabled(graph_exec, node, &enabled)) {
    HIP_SKIP_TEST("Graph node enable/disable APIs are not supported by this runtime path.");
  }

  // A freshly instantiated node must report as enabled by default.
  REQUIRE(enabled == 1);
}

HIP_TEST_CASE(Contract_GraphNodeEnabled_DisableThenQuery_ReportsZero) {
  hip::contract::ContractCleanup cleanup;
  std::array<uint8_t, kByteCount> host{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, host.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, device_ptr, host.data(), host.size(),
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  // Disabling the node must be reflected by a subsequent query reporting zero.
  if (!TrySetEnabled(graph_exec, node, 0)) {
    HIP_SKIP_TEST("Graph node enable/disable APIs are not supported by this runtime path.");
  }

  unsigned int enabled = 1;
  REQUIRE(TryGetEnabled(graph_exec, node, &enabled));
  REQUIRE(enabled == 0);

  // Re-enabling the node must round-trip back to a query reporting one.
  REQUIRE(TrySetEnabled(graph_exec, node, 1));
  enabled = 0;
  REQUIRE(TryGetEnabled(graph_exec, node, &enabled));
  REQUIRE(enabled == 1);
}

HIP_TEST_CASE(Contract_GraphNodeEnabled_DisabledNode_ActsAsEmpty) {
  std::array<uint8_t, kByteCount> source{};
  std::array<uint8_t, kByteCount> destination{};
  source.fill(kSourcePattern);
  destination.fill(0);

  hip::contract::ContractCleanup cleanup;
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t h2d_node = nullptr;
  hipGraphNode_t d2h_node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, source.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  // Pre-seed the device buffer with a baseline pattern distinct from the source
  // so that a disabled host-to-device copy leaves the baseline observable.
  HIP_CHECK(hipMemset(device_ptr, kBaselinePattern, source.size()));

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  // The host-to-device node writes the source pattern into the device buffer,
  // and the device-to-host node reads the device buffer back into the host
  // destination. Making the D2H node depend on the H2D node orders the copies so
  // that, when the H2D node is enabled, the destination sees the source pattern.
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d_node, graph, nullptr, 0, device_ptr, source.data(),
                                    source.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h_node, graph, &h2d_node, 1, destination.data(), device_ptr,
                                    destination.size(), hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  // Disabling the host-to-device node must make it behave as an empty node, so
  // it no longer overwrites the pre-seeded baseline in the device buffer.
  if (!TrySetEnabled(graph_exec, h2d_node, 0)) {
    HIP_SKIP_TEST("Graph node enable/disable APIs are not supported by this runtime path.");
  }

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // With the H2D node disabled, the still-enabled D2H node reads the untouched
  // baseline out of the device buffer. The destination must equal the baseline
  // pattern and must not equal the source pattern.
  for (const uint8_t byte : destination) {
    REQUIRE(byte == kBaselinePattern);
    REQUIRE(byte != kSourcePattern);
  }
}
