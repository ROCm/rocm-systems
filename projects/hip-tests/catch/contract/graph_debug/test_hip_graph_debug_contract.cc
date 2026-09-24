/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <hip_test_filesystem.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kByteCount = 64;

// Random suffix for temp artifacts, the same way hip_test_process.hh names its
// capture files. Random rather than derived from the pid: a guessable path in a
// shared temp directory can be pre-seeded with a symlink, which the export then
// follows. The clock is mixed in so the name stays unique even where
// std::random_device is a weak source.
std::string RandomTag() {
  std::random_device dev;
  uint64_t value = (static_cast<uint64_t>(dev()) << 32) ^ dev();
  value ^= static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(value));
  return std::string(buf);
}

// Use the temp directory because installed test directories may be read-only.
std::string DotPath() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return (fs::temp_directory_path() /
          ("hip_contract_graph_debug_" + std::to_string(device) + "_" + RandomTag() + ".dot"))
      .string();
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

// @asserts: hipGraphDebugDotPrint - exporting a non-empty graph succeeds (or reports unsupported) and writes a non-empty dot file
HIP_TEST_CASE(Contract_GraphDebug_HipGraphDebugDotPrint_Default_WritesNonEmptyFile) {
  hip::contract::ContractCleanup cleanup;
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  hipMemsetParams memset_params = MakeByteMemsetParams(device_ptr, 0x5A);
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &memset_params));

  const std::string path = DotPath();

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

// @asserts: hipGraphDebugDotPrint - the verbose flag is accepted and still exports a valid graph to a non-empty dot file
HIP_TEST_CASE(Contract_GraphDebug_HipGraphDebugDotPrint_Default_VerboseFlagIsAccepted) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  const std::string path = DotPath();

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
