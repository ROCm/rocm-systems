/*
 * Copyright Advanced Micro Devices, Inc.
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
}  // namespace

// @asserts: hipGraphAddChildGraphNode - launching a parent graph executes the embedded child graph's memcpy nodes end to end
HIP_TEST_CASE(Contract_GraphChild_HipGraphAddChildGraphNode_Default_ExecutesEmbeddedMemcpy) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x37);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t h2d_node = nullptr;
  hipGraphNode_t d2h_node = nullptr;
  hipGraphNode_t child_node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipGraphCreate(&child, 0));
  cleanup.Add([child] { (void)hipGraphDestroy(child); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d_node, child, nullptr, 0, device_ptr, src.data(), src.size(),
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h_node, child, &h2d_node, 1, dst.data(), device_ptr,
                                    dst.size(), hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphCreate(&parent, 0));
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, parent, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

// @asserts: hipGraphChildGraphNodeGetGraph - returns a non-null embedded graph handle whose node count matches the child graph
HIP_TEST_CASE(Contract_GraphChild_HipGraphChildGraphNodeGetGraph_Default_ReturnsHandle) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraph_t embedded = nullptr;
  hipGraphNode_t empty_node = nullptr;
  hipGraphNode_t child_node = nullptr;
  size_t node_count = 0;

  HIP_CHECK(hipGraphCreate(&child, 0));
  cleanup.Add([child] { (void)hipGraphDestroy(child); });
  HIP_CHECK(hipGraphAddEmptyNode(&empty_node, child, nullptr, 0));

  HIP_CHECK(hipGraphCreate(&parent, 0));
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child));

  HIP_CHECK(hipGraphChildGraphNodeGetGraph(child_node, &embedded));
  REQUIRE(embedded != nullptr);

  HIP_CHECK(hipGraphGetNodes(embedded, nullptr, &node_count));
  REQUIRE(node_count == 1);
}

// @asserts: hipGraphNodeGetType - a child-graph node reports node type hipGraphNodeTypeGraph
HIP_TEST_CASE(Contract_GraphChild_HipGraphNodeGetType_NodeType_ReportsGraph) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraphNode_t child_node = nullptr;
  hipGraphNodeType node_type{};

  HIP_CHECK(hipGraphCreate(&child, 0));
  cleanup.Add([child] { (void)hipGraphDestroy(child); });

  HIP_CHECK(hipGraphCreate(&parent, 0));
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child));

  HIP_CHECK(hipGraphNodeGetType(child_node, &node_type));
  REQUIRE(node_type == hipGraphNodeTypeGraph);
}

// Child graph node ownership was introduced in CUDA 12.9; hipnv aliases the enum only from
// that version, so the cases below cannot be compiled against older CUDA.
#if HT_AMD || (HT_NVIDIA && CUDA_VERSION >= 12090)

namespace {
// Adds `child` to `parent` with the requested ownership, returning the raw status so a case
// can assert on a rejection without aborting.
hipError_t AddChildWithOwnership(hipGraphNode_t* node_out, hipGraph_t parent, hipGraph_t child,
                                 hipGraphChildGraphNodeOwnership ownership,
                                 const hipGraphNode_t* deps = nullptr, size_t num_deps = 0) {
  hipGraphNodeParams params{};
  params.type = hipGraphNodeTypeGraph;
  params.graph.graph = child;
  params.graph.ownership = ownership;
  return hipGraphAddNode(node_out, parent, deps, num_deps, &params);
}
}  // namespace

// @asserts: hipGraphAddNode - a child graph added with ownership move is reported by hipGraphChildGraphNodeGetGraph as the caller's own graph handle
HIP_TEST_CASE(Contract_GraphChild_HipGraphAddNode_OwnershipMove_ReturnsCallerGraphAsEmbedded) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraphNode_t child_empty = nullptr;
  hipGraphNode_t moved_node = nullptr;
  hipGraph_t embedded = nullptr;

  HIP_CHECK(hipGraphCreate(&child, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&child_empty, child, nullptr, 0));
  HIP_CHECK(hipGraphCreate(&parent, 0));
  // Destroying the parent releases the moved child graph as well, so `child` must not be
  // added to the cleanup list.
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });

  HIP_CHECK(AddChildWithOwnership(&moved_node, parent, child, hipGraphChildGraphOwnershipMove));

  // Under move the parent takes the caller's graph object itself rather than a copy of it.
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(moved_node, &embedded));
  REQUIRE(embedded == child);
}

// @asserts: hipGraphAddNode - a child graph added with ownership clone leaves the caller's handle distinct from the node's embedded graph
HIP_TEST_CASE(Contract_GraphChild_HipGraphAddNode_OwnershipClone_ReturnsDistinctEmbeddedGraph) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraphNode_t child_empty = nullptr;
  hipGraphNode_t cloned_node = nullptr;
  hipGraph_t embedded = nullptr;

  HIP_CHECK(hipGraphCreate(&child, 0));
  cleanup.Add([child] { (void)hipGraphDestroy(child); });
  HIP_CHECK(hipGraphAddEmptyNode(&child_empty, child, nullptr, 0));
  HIP_CHECK(hipGraphCreate(&parent, 0));
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });

  HIP_CHECK(AddChildWithOwnership(&cloned_node, parent, child, hipGraphChildGraphOwnershipClone));

  // Clone copies into the node, so the caller keeps an independent graph.
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(cloned_node, &embedded));
  REQUIRE(embedded != nullptr);
  REQUIRE(embedded != child);
}

// @asserts: hipGraphDestroy - a graph moved into a child graph node is owned by its parent and can no longer be destroyed on its own
HIP_TEST_CASE(Contract_GraphChild_HipGraphDestroy_MovedChildGraph_ReturnsError) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraphNode_t child_empty = nullptr;
  hipGraphNode_t moved_node = nullptr;

  HIP_CHECK(hipGraphCreate(&child, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&child_empty, child, nullptr, 0));
  HIP_CHECK(hipGraphCreate(&parent, 0));
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });

  HIP_CHECK(AddChildWithOwnership(&moved_node, parent, child, hipGraphChildGraphOwnershipMove));

  REQUIRE(hipGraphDestroy(child) != hipSuccess);
  (void)hipGetLastError();
}

// @asserts: hipGraphAddNode - a move rejected for bad dependencies leaves the caller's graph intact and still destroyable
HIP_TEST_CASE(Contract_GraphChild_HipGraphAddNode_RejectedMove_LeavesCallerGraphDestroyable) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t owner = nullptr;
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraphNode_t foreign_dep = nullptr;
  hipGraphNode_t child_empty = nullptr;
  hipGraphNode_t moved_node = nullptr;
  size_t node_count = 0;

  // A dependency belonging to a different graph than the one being added to is invalid.
  HIP_CHECK(hipGraphCreate(&owner, 0));
  cleanup.Add([owner] { (void)hipGraphDestroy(owner); });
  HIP_CHECK(hipGraphAddEmptyNode(&foreign_dep, owner, nullptr, 0));

  HIP_CHECK(hipGraphCreate(&child, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&child_empty, child, nullptr, 0));
  HIP_CHECK(hipGraphCreate(&parent, 0));
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });

  REQUIRE(AddChildWithOwnership(&moved_node, parent, child, hipGraphChildGraphOwnershipMove,
                                &foreign_dep, 1) != hipSuccess);
  (void)hipGetLastError();

  // A call that reported failure must not have consumed the caller's graph.
  HIP_CHECK(hipGraphGetNodes(child, nullptr, &node_count));
  REQUIRE(node_count == 1);
  HIP_CHECK(hipGraphDestroy(child));
}

// @asserts: hipGraphAddNode - rejects an ownership move of a graph that is already embedded in another child graph node
HIP_TEST_CASE(Contract_GraphChild_HipGraphAddNode_EmbeddedSourceMove_ReturnsError) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t child = nullptr;
  hipGraph_t first_parent = nullptr;
  hipGraph_t second_parent = nullptr;
  hipGraphNode_t child_empty = nullptr;
  hipGraphNode_t cloned_node = nullptr;
  hipGraphNode_t moved_node = nullptr;
  hipGraph_t embedded = nullptr;

  HIP_CHECK(hipGraphCreate(&child, 0));
  cleanup.Add([child] { (void)hipGraphDestroy(child); });
  HIP_CHECK(hipGraphAddEmptyNode(&child_empty, child, nullptr, 0));

  HIP_CHECK(hipGraphCreate(&first_parent, 0));
  cleanup.Add([first_parent] { (void)hipGraphDestroy(first_parent); });
  HIP_CHECK(hipGraphAddChildGraphNode(&cloned_node, first_parent, nullptr, 0, child));
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(cloned_node, &embedded));
  REQUIRE(embedded != nullptr);

  HIP_CHECK(hipGraphCreate(&second_parent, 0));
  cleanup.Add([second_parent] { (void)hipGraphDestroy(second_parent); });

  // Moving an embedded graph would hand a second owner a node the first parent still frees.
  REQUIRE(AddChildWithOwnership(&moved_node, second_parent, embedded,
                                hipGraphChildGraphOwnershipMove) != hipSuccess);
  (void)hipGetLastError();
}

#endif  // HT_AMD || (HT_NVIDIA && CUDA_VERSION >= 12090)
