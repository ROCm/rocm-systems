/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr size_t kAllocBytes = 4096;

int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

// Builds allocation node parameters for a small pinned device-local allocation
// on the current device.
hipMemAllocNodeParams CurrentDeviceAllocParams() {
  hipMemAllocNodeParams params{};
  params.poolProps.allocType = hipMemAllocationTypePinned;
  params.poolProps.location.type = hipMemLocationTypeDevice;
  params.poolProps.location.id = CurrentDevice();
  params.bytesize = kAllocBytes;
  return params;
}

// Trims graph memory for the current device so that the graph memory state is
// restored to a known baseline before and after a test. Returns false when the
// runtime path does not support graph memory trimming (hipErrorNotSupported),
// which mirrors the unsupported-capability skip used for the alloc-node APIs;
// any other failure is an unexpected contract violation and aborts through
// HIP_CHECK.
bool TryTrimGraphMemory() {
  const hipError_t status = hipDeviceGraphMemTrim(CurrentDevice());
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}

// Attempts to add a memory allocation node. Returns false (with graph cleanup)
// if the runtime path does not support graph memory allocation nodes.
bool TryAddMemAllocNode(hipGraphNode_t* node, hipGraph_t graph,
                        hipMemAllocNodeParams* params) {
  const hipError_t status =
      hipGraphAddMemAllocNode(node, graph, nullptr, 0, params);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}

// Attempts to add a memory free node for dev_ptr that depends on alloc_node.
// Returns false if the runtime path does not support graph memory free nodes.
bool TryAddMemFreeNode(hipGraphNode_t* node, hipGraph_t graph,
                       const hipGraphNode_t* deps, size_t num_deps,
                       void* dev_ptr) {
  const hipError_t status =
      hipGraphAddMemFreeNode(node, graph, deps, num_deps, dev_ptr);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}

// Attempts to add a driver-style memory free node for dev_ptr. Returns false if
// the runtime path does not support driver graph memory free nodes.
bool TryAddDrvMemFreeNode(hipGraphNode_t* node, hipGraph_t graph,
                          const hipGraphNode_t* deps, size_t num_deps,
                          hipDeviceptr_t dev_ptr) {
  const hipError_t status =
      hipDrvGraphAddMemFreeNode(node, graph, deps, num_deps, dev_ptr);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_GraphMemNodes_AllocNode_ReturnsDevicePtr) {
  if (!TryTrimGraphMemory()) {
    HIP_SKIP_TEST("Graph memory trimming is not supported by this runtime path.");
  }

  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t alloc_node = nullptr;
  hipMemAllocNodeParams params = CurrentDeviceAllocParams();

  // The trailing graph-memory trim restores the baseline after the graph is
  // destroyed; registering it first means it runs last (after hipGraphDestroy)
  // when the guard unwinds.
  cleanup.Add([&] { (void)TryTrimGraphMemory(); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  if (!TryAddMemAllocNode(&alloc_node, graph, &params)) {
    HIP_SKIP_TEST("Graph memory allocation nodes are not supported by this runtime path.");
  }

  REQUIRE(params.dptr != nullptr);
}

HIP_TEST_CASE(Contract_GraphMemNodes_GetParams_RoundTripsBytesize) {
  if (!TryTrimGraphMemory()) {
    HIP_SKIP_TEST("Graph memory trimming is not supported by this runtime path.");
  }

  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t alloc_node = nullptr;
  hipMemAllocNodeParams params = CurrentDeviceAllocParams();

  cleanup.Add([&] { (void)TryTrimGraphMemory(); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  if (!TryAddMemAllocNode(&alloc_node, graph, &params)) {
    HIP_SKIP_TEST("Graph memory allocation nodes are not supported by this runtime path.");
  }

  hipMemAllocNodeParams retrieved{};
  HIP_CHECK(hipGraphMemAllocNodeGetParams(alloc_node, &retrieved));

  REQUIRE(retrieved.bytesize == kAllocBytes);
  REQUIRE(retrieved.poolProps.allocType == hipMemAllocationTypePinned);
  REQUIRE(retrieved.poolProps.location.type == hipMemLocationTypeDevice);
  REQUIRE(retrieved.poolProps.location.id == CurrentDevice());
}

HIP_TEST_CASE(Contract_GraphMemNodes_FreeNodeGetParams_RoundTripsPointer) {
  if (!TryTrimGraphMemory()) {
    HIP_SKIP_TEST("Graph memory trimming is not supported by this runtime path.");
  }

  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t alloc_node = nullptr;
  hipGraphNode_t free_node = nullptr;
  hipMemAllocNodeParams params = CurrentDeviceAllocParams();

  cleanup.Add([&] { (void)TryTrimGraphMemory(); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  if (!TryAddMemAllocNode(&alloc_node, graph, &params)) {
    HIP_SKIP_TEST("Graph memory allocation nodes are not supported by this runtime path.");
  }
  REQUIRE(params.dptr != nullptr);

  if (!TryAddMemFreeNode(&free_node, graph, &alloc_node, 1, params.dptr)) {
    HIP_SKIP_TEST("Graph memory free nodes are not supported by this runtime path.");
  }

  // The free node reports the device pointer it was created with. The getter
  // writes the pointer through its void* out-parameter.
  void* retrieved = nullptr;
  HIP_CHECK(hipGraphMemFreeNodeGetParams(free_node, &retrieved));
  REQUIRE(retrieved == params.dptr);
}

HIP_TEST_CASE(Contract_GraphMemNodes_GraphMemAttribute_TrimIsNonIncreasing) {
  const int device = CurrentDevice();

  uint64_t reserved_before = 0;
  const hipError_t query_status = hipDeviceGetGraphMemAttribute(
      device, hipGraphMemAttrReservedMemCurrent, &reserved_before);
  if (query_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Graph memory attribute queries are not supported by this runtime path.");
  }
  HIP_CHECK(query_status);

  // Trimming graph memory must not increase the amount of reserved graph
  // memory. Reserved memory is a non-negative size, so the trimmed value stays
  // within [0, reserved_before].
  HIP_CHECK(hipDeviceGraphMemTrim(device));

  uint64_t reserved_after = 0;
  HIP_CHECK(hipDeviceGetGraphMemAttribute(
      device, hipGraphMemAttrReservedMemCurrent, &reserved_after));

  REQUIRE(reserved_after <= reserved_before);
}

HIP_TEST_CASE(Contract_GraphMemNodes_SetGraphMemAttribute_ResetsHighWatermark) {
  const int device = CurrentDevice();

  uint64_t used_high = 0;
  const hipError_t query_status =
      hipDeviceGetGraphMemAttribute(device, hipGraphMemAttrUsedMemHigh, &used_high);
  if (query_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Graph memory attribute queries are not supported by this runtime path.");
  }
  HIP_CHECK(query_status);

  // The used-memory high watermark is resettable: writing zero must be accepted
  // (or reported unsupported), and a subsequent query must report the watermark
  // at zero. Only the high-watermark attributes are writable.
  uint64_t zero = 0;
  const hipError_t set_status =
      hipDeviceSetGraphMemAttribute(device, hipGraphMemAttrUsedMemHigh, &zero);
  if (set_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Setting graph memory attributes is not supported by this runtime path.");
  }
  HIP_CHECK(set_status);

  uint64_t used_high_after = 1;
  HIP_CHECK(
      hipDeviceGetGraphMemAttribute(device, hipGraphMemAttrUsedMemHigh, &used_high_after));
  REQUIRE(used_high_after == 0);
}

HIP_TEST_CASE(Contract_GraphMemNodes_SetGraphMemAttribute_CurrentAttribute_IsRejected) {
  const int device = CurrentDevice();

  // The current-usage attributes are read-only accounting values, not settable
  // knobs. Attempting to write one must be rejected rather than silently
  // accepted (or reported unsupported if the set path is unavailable).
  uint64_t zero = 0;
  const hipError_t status =
      hipDeviceSetGraphMemAttribute(device, hipGraphMemAttrUsedMemCurrent, &zero);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Setting graph memory attributes is not supported by this runtime path.");
  }
  REQUIRE(status != hipSuccess);
}

#if HT_AMD
HIP_TEST_CASE(Contract_GraphMemNodes_DrvFreeNode_AddsToGraph) {
  if (!TryTrimGraphMemory()) {
    HIP_SKIP_TEST("Graph memory trimming is not supported by this runtime path.");
  }

  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t alloc_node = nullptr;
  hipGraphNode_t free_node = nullptr;
  hipMemAllocNodeParams params = CurrentDeviceAllocParams();

  cleanup.Add([&] { (void)TryTrimGraphMemory(); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  if (!TryAddMemAllocNode(&alloc_node, graph, &params)) {
    HIP_SKIP_TEST("Graph memory allocation nodes are not supported by this runtime path.");
  }
  REQUIRE(params.dptr != nullptr);

  // The driver-style free node consumes the pointer produced by the alloc node
  // and must run after it, so the alloc node is its only dependency. The node is
  // only added and introspected structurally; launching a graph containing a
  // memory free node is intentionally not exercised (it aborts on some runtime
  // paths), matching the runtime free-node contract in this domain.
  if (!TryAddDrvMemFreeNode(&free_node, graph, &alloc_node, 1,
                            reinterpret_cast<hipDeviceptr_t>(params.dptr))) {
    HIP_SKIP_TEST("Driver graph memory free nodes are not supported by this runtime path.");
  }
  REQUIRE(free_node != nullptr);

  // The added node reports the memory-free node type.
  hipGraphNodeType type{};
  HIP_CHECK(hipGraphNodeGetType(free_node, &type));
  REQUIRE(type == hipGraphNodeTypeMemFree);
}
#endif  // HT_AMD
