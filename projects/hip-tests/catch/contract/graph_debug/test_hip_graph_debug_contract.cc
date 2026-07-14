/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kByteCount = 64;

// Builds a unique temporary file path for the exported dot graph so concurrent
// test binaries do not collide on a shared filename.
std::string UniqueDotPath() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return std::string("hip_contract_graph_debug_") + std::to_string(device) + "_" +
         std::to_string(static_cast<long long>(reinterpret_cast<intptr_t>(&device))) + ".dot";
}

hipMemsetParams MakeByteMemsetParams(void* device_ptr, unsigned int value) {
  hipMemsetParams params{};
  params.dst = device_ptr;
  params.value = value;
  params.pitch = kByteCount;
  params.elementSize = sizeof(uint8_t);
  params.width = kByteCount;
  params.height = 1;
  return params;
}

size_t FileSize(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return 0;
  }
  return static_cast<size_t>(file.tellg());
}
}  // namespace

HIP_TEST_CASE(Contract_GraphDebug_DotPrint_WritesNonEmptyFile) {
  hip::contract::ContractCleanup cleanup;
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  hipMemsetParams memset_params = MakeByteMemsetParams(device_ptr, 0x5A);
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &memset_params));

  const std::string path = UniqueDotPath();
  std::remove(path.c_str());

  // Exporting a non-empty graph to a dot file must succeed (or report the
  // feature unsupported) and, on success, produce a non-empty file on disk.
  const hipError_t status = hipGraphDebugDotPrint(graph, path.c_str(), 0);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Graph dot export is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  REQUIRE(FileSize(path) > 0);

  std::remove(path.c_str());
}

HIP_TEST_CASE(Contract_GraphDebug_DotPrint_VerboseFlagIsAccepted) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  const std::string path = UniqueDotPath();
  std::remove(path.c_str());

  // The verbose flag augments the output but must not change the success
  // contract: a valid graph still exports to a non-empty file.
  const hipError_t status =
      hipGraphDebugDotPrint(graph, path.c_str(), hipGraphDebugDotFlagsVerbose);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Graph dot export is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  REQUIRE(FileSize(path) > 0);

  std::remove(path.c_str());
}
