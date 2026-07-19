/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <array>
#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
bool ContainsNode(const hipGraphNode_t* nodes, size_t count, hipGraphNode_t expected) {
  return std::find(nodes, nodes + count, expected) != nodes + count;
}

struct TwoNodeGraph {
  hipGraph_t graph = nullptr;
  hipGraphNode_t root = nullptr;
  hipGraphNode_t dependent = nullptr;
};

TwoNodeGraph CreateTwoNodeGraph() {
  TwoNodeGraph result{};
  HIP_CHECK(hipGraphCreate(&result.graph, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&result.root, result.graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&result.dependent, result.graph, &result.root, 1));
  return result;
}
}

// @asserts: hipGraphGetNodes - reports the correct node count and returns every node added to the graph
HIP_TEST_CASE(Contract_GraphTopology_GetNodes_ReturnsAddedNodes) {
  hip::contract::ContractCleanup cleanup;
  auto graph = CreateTwoNodeGraph();
  cleanup.Add([p0 = graph.graph] { (void)hipGraphDestroy(p0); });
  size_t node_count = 0;

  HIP_CHECK(hipGraphGetNodes(graph.graph, nullptr, &node_count));
  REQUIRE(node_count == 2);

  std::array<hipGraphNode_t, 2> nodes{};
  HIP_CHECK(hipGraphGetNodes(graph.graph, nodes.data(), &node_count));

  REQUIRE(node_count == 2);
  REQUIRE(ContainsNode(nodes.data(), node_count, graph.root));
  REQUIRE(ContainsNode(nodes.data(), node_count, graph.dependent));
}

// @asserts: hipGraphGetRootNodes - returns only the dependency-free node(s) as roots
HIP_TEST_CASE(Contract_GraphTopology_GetRootNodes_ReturnsDependencyFreeNode) {
  hip::contract::ContractCleanup cleanup;
  auto graph = CreateTwoNodeGraph();
  cleanup.Add([p0 = graph.graph] { (void)hipGraphDestroy(p0); });
  size_t root_count = 0;

  HIP_CHECK(hipGraphGetRootNodes(graph.graph, nullptr, &root_count));
  REQUIRE(root_count == 1);

  std::array<hipGraphNode_t, 1> roots{};
  HIP_CHECK(hipGraphGetRootNodes(graph.graph, roots.data(), &root_count));

  REQUIRE(root_count == 1);
  REQUIRE(roots[0] == graph.root);
}

// @asserts: hipGraphGetEdges - returns each dependency as a from/to edge pair matching the configured direction
HIP_TEST_CASE(Contract_GraphTopology_GetEdges_ReturnsDependencyEdge) {
  hip::contract::ContractCleanup cleanup;
  auto graph = CreateTwoNodeGraph();
  cleanup.Add([p0 = graph.graph] { (void)hipGraphDestroy(p0); });
  size_t edge_count = 0;

  HIP_CHECK(hipGraphGetEdges(graph.graph, nullptr, nullptr, &edge_count));
  REQUIRE(edge_count == 1);

  std::array<hipGraphNode_t, 1> from{};
  std::array<hipGraphNode_t, 1> to{};
  HIP_CHECK(hipGraphGetEdges(graph.graph, from.data(), to.data(), &edge_count));

  REQUIRE(edge_count == 1);
  REQUIRE(from[0] == graph.root);
  REQUIRE(to[0] == graph.dependent);
}

// @asserts: hipGraphNodeGetDependencies - returns the upstream nodes a given node depends on
HIP_TEST_CASE(Contract_GraphTopology_NodeDependencies_ReturnsConfiguredDependency) {
  hip::contract::ContractCleanup cleanup;
  auto graph = CreateTwoNodeGraph();
  cleanup.Add([p0 = graph.graph] { (void)hipGraphDestroy(p0); });
  size_t dependency_count = 0;

  HIP_CHECK(hipGraphNodeGetDependencies(graph.dependent, nullptr, &dependency_count));
  REQUIRE(dependency_count == 1);

  std::array<hipGraphNode_t, 1> dependencies{};
  HIP_CHECK(hipGraphNodeGetDependencies(graph.dependent, dependencies.data(), &dependency_count));

  REQUIRE(dependency_count == 1);
  REQUIRE(dependencies[0] == graph.root);
}

// @asserts: hipGraphNodeGetDependentNodes - returns the downstream nodes that depend on a given node
HIP_TEST_CASE(Contract_GraphTopology_NodeDependents_ReturnsConfiguredDependent) {
  hip::contract::ContractCleanup cleanup;
  auto graph = CreateTwoNodeGraph();
  cleanup.Add([p0 = graph.graph] { (void)hipGraphDestroy(p0); });
  size_t dependent_count = 0;

  HIP_CHECK(hipGraphNodeGetDependentNodes(graph.root, nullptr, &dependent_count));
  REQUIRE(dependent_count == 1);

  std::array<hipGraphNode_t, 1> dependents{};
  HIP_CHECK(hipGraphNodeGetDependentNodes(graph.root, dependents.data(), &dependent_count));

  REQUIRE(dependent_count == 1);
  REQUIRE(dependents[0] == graph.dependent);
}
