/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

#if HT_AMD || (defined(CUDA_VERSION) && CUDA_VERSION >= 13010)

// @asserts: hipGraphNodeGetLocalId - returns hipErrorInvalidValue when either argument is null
HIP_TEST_CASE(Contract_GraphId_HipGraphNodeGetLocalId_NullArguments_ReturnInvalidValue) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  unsigned int node_id = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  REQUIRE(hipGraphNodeGetLocalId(nullptr, &node_id) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipGraphNodeGetLocalId(node, nullptr) == hipErrorInvalidValue);
  (void)hipGetLastError();
}

// @asserts: hipGraphNodeGetToolsId - returns hipErrorInvalidValue when either argument is null
HIP_TEST_CASE(Contract_GraphId_HipGraphNodeGetToolsId_NullArguments_ReturnInvalidValue) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  unsigned long long tools_id = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  REQUIRE(hipGraphNodeGetToolsId(nullptr, &tools_id) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipGraphNodeGetToolsId(node, nullptr) == hipErrorInvalidValue);
  (void)hipGetLastError();
}

// @asserts: hipGraphGetId - returns hipErrorInvalidValue when either argument is null
HIP_TEST_CASE(Contract_GraphId_HipGraphGetId_NullArguments_ReturnInvalidValue) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  unsigned int graph_id = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  REQUIRE(hipGraphGetId(nullptr, &graph_id) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipGraphGetId(graph, nullptr) == hipErrorInvalidValue);
  (void)hipGetLastError();
}

// @asserts: hipGraphExecGetId - returns hipErrorInvalidValue when either argument is null
HIP_TEST_CASE(Contract_GraphId_HipGraphExecGetId_NullArguments_ReturnInvalidValue) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  unsigned int graph_id = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));
  HIP_CHECK(hipGraphInstantiateWithFlags(&graph_exec, graph, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });

  REQUIRE(hipGraphExecGetId(nullptr, &graph_id) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipGraphExecGetId(graph_exec, nullptr) == hipErrorInvalidValue);
  (void)hipGetLastError();
}

// @asserts: hipGraphNodeGetLocalId - repeated calls return the same ID and different nodes have different IDs
HIP_TEST_CASE(
    Contract_GraphId_HipGraphNodeGetLocalId_SameNodeSameId_OtherNodeOtherId) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t first = nullptr;
  hipGraphNode_t second = nullptr;
  unsigned int first_id = 0;
  unsigned int first_id_repeat = 0;
  unsigned int second_id = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&first, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&second, graph, nullptr, 0));

  HIP_CHECK(hipGraphNodeGetLocalId(first, &first_id));
  HIP_CHECK(hipGraphNodeGetLocalId(first, &first_id_repeat));
  HIP_CHECK(hipGraphNodeGetLocalId(second, &second_id));

  REQUIRE(first_id == first_id_repeat);
  REQUIRE(first_id != second_id);
}

// @asserts: hipGraphNodeGetToolsId - repeated calls return the same ID and different nodes have different IDs
HIP_TEST_CASE(
    Contract_GraphId_HipGraphNodeGetToolsId_SameNodeSameId_OtherNodeOtherId) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t first = nullptr;
  hipGraphNode_t second = nullptr;
  unsigned long long first_id = 0;
  unsigned long long first_id_repeat = 0;
  unsigned long long second_id = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&first, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&second, graph, nullptr, 0));

  HIP_CHECK(hipGraphNodeGetToolsId(first, &first_id));
  HIP_CHECK(hipGraphNodeGetToolsId(first, &first_id_repeat));
  HIP_CHECK(hipGraphNodeGetToolsId(second, &second_id));

  REQUIRE(first_id == first_id_repeat);
  REQUIRE(first_id != second_id);
}

// @asserts: hipGraphGetId - repeated calls return the same ID and different graphs have different IDs
HIP_TEST_CASE(Contract_GraphId_HipGraphGetId_SameGraphSameId_OtherGraphOtherId) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t first = nullptr;
  hipGraph_t second = nullptr;
  unsigned int first_id = 0;
  unsigned int first_id_repeat = 0;
  unsigned int second_id = 0;

  HIP_CHECK(hipGraphCreate(&first, 0));
  cleanup.Add([first] { (void)hipGraphDestroy(first); });
  HIP_CHECK(hipGraphCreate(&second, 0));
  cleanup.Add([second] { (void)hipGraphDestroy(second); });

  HIP_CHECK(hipGraphGetId(first, &first_id));
  HIP_CHECK(hipGraphGetId(first, &first_id_repeat));
  HIP_CHECK(hipGraphGetId(second, &second_id));

  REQUIRE(first_id == first_id_repeat);
  REQUIRE(first_id != second_id);
}

// @asserts: hipGraphExecGetId - repeated calls return the same ID and different executable graphs have different IDs
HIP_TEST_CASE(
    Contract_GraphId_HipGraphExecGetId_SameExecSameId_OtherExecOtherId) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphExec_t first = nullptr;
  hipGraphExec_t second = nullptr;
  unsigned int first_id = 0;
  unsigned int first_id_repeat = 0;
  unsigned int second_id = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  HIP_CHECK(hipGraphInstantiateWithFlags(&first, graph, 0));
  cleanup.Add([first] { (void)hipGraphExecDestroy(first); });
  HIP_CHECK(hipGraphInstantiateWithFlags(&second, graph, 0));
  cleanup.Add([second] { (void)hipGraphExecDestroy(second); });

  HIP_CHECK(hipGraphExecGetId(first, &first_id));
  HIP_CHECK(hipGraphExecGetId(first, &first_id_repeat));
  HIP_CHECK(hipGraphExecGetId(second, &second_id));

  REQUIRE(first_id == first_id_repeat);
  REQUIRE(first_id != second_id);
}

#endif  // HT_AMD || CUDA_VERSION >= 13010
