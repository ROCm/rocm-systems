/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

// The graph symbol-copy nodes resolve a device global through the public symbol
// copy APIs, which look symbols up by their registered external name. The device
// global (and the kernel that keeps it live in the device image) must therefore
// have external linkage and live at file scope rather than in an anonymous
// namespace.
__device__ int g_contract_graph_symbol = 0;

__global__ void TouchGraphSymbolKernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    g_contract_graph_symbol = g_contract_graph_symbol + 1;
  }
}

namespace {
constexpr int kFirstValue = 7;
constexpr int kSecondValue = 11;

// Ensures the device global is emitted into the loaded image and reset to a
// known baseline before each contract runs.
void ResetSymbol() {
  // Launch the touch kernel once so the compiler keeps the symbol live, then
  // zero it through the runtime symbol-copy path.
  hipLaunchKernelGGL(TouchGraphSymbolKernel, dim3(1), dim3(1), 0, 0);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  const int zero = 0;
  HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_contract_graph_symbol), &zero, sizeof(zero)));
  HIP_CHECK(hipDeviceSynchronize());
}

int ReadSymbol() {
  int value = -1;
  HIP_CHECK(hipMemcpyFromSymbol(&value, HIP_SYMBOL(g_contract_graph_symbol), sizeof(value)));
  return value;
}
}  // namespace

HIP_TEST_CASE(Contract_GraphSymbolCopyNodes_AddToFromSymbol_RoundTripsValue) {
  ResetSymbol();

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t to_node = nullptr;
  hipGraphNode_t from_node = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // A to-symbol node writes a host value into the device global; a dependent
  // from-symbol node reads it back into a host destination. Launching the graph
  // must round-trip the value through the symbol.
  int source = kFirstValue;
  int result = -1;
  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&to_node, graph, nullptr, 0,
                                          &g_contract_graph_symbol, &source,
                                          sizeof(source), 0, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&from_node, graph, &to_node, 1, &result,
                                            &g_contract_graph_symbol, sizeof(result), 0,
                                            hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(result == kFirstValue);
  REQUIRE(ReadSymbol() == kFirstValue);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

HIP_TEST_CASE(Contract_GraphSymbolCopyNodes_SetParamsToFromSymbol_UpdatesGraphNodes) {
  ResetSymbol();

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t to_node = nullptr;
  hipGraphNode_t from_node = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Build the graph with a first source, then rewrite both nodes on the graph
  // (pre-instantiation) to use a second source/destination. The launched graph
  // must reflect the updated parameters.
  int first_source = kFirstValue;
  int first_result = -1;
  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&to_node, graph, nullptr, 0,
                                          &g_contract_graph_symbol, &first_source,
                                          sizeof(first_source), 0, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&from_node, graph, &to_node, 1, &first_result,
                                            &g_contract_graph_symbol,
                                            sizeof(first_result), 0, hipMemcpyDeviceToHost));

  int second_source = kSecondValue;
  int second_result = -1;
  HIP_CHECK(hipGraphMemcpyNodeSetParamsToSymbol(to_node, &g_contract_graph_symbol,
                                                &second_source, sizeof(second_source), 0,
                                                hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphMemcpyNodeSetParamsFromSymbol(from_node, &second_result,
                                                  &g_contract_graph_symbol,
                                                  sizeof(second_result), 0, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(second_result == kSecondValue);
  REQUIRE(ReadSymbol() == kSecondValue);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

HIP_TEST_CASE(Contract_GraphSymbolCopyNodes_ExecSetParamsToFromSymbol_UpdatesExecutable) {
  ResetSymbol();

  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t to_node = nullptr;
  hipGraphNode_t from_node = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int first_source = kFirstValue;
  int first_result = -1;
  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&to_node, graph, nullptr, 0,
                                          &g_contract_graph_symbol, &first_source,
                                          sizeof(first_source), 0, hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&from_node, graph, &to_node, 1, &first_result,
                                            &g_contract_graph_symbol,
                                            sizeof(first_result), 0, hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  // Rewrite both nodes on the executable graph before launch. The launched graph
  // must observe the updated source/destination.
  int second_source = kSecondValue;
  int second_result = -1;
  HIP_CHECK(hipGraphExecMemcpyNodeSetParamsToSymbol(graph_exec, to_node,
                                                    &g_contract_graph_symbol,
                                                    &second_source, sizeof(second_source), 0,
                                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphExecMemcpyNodeSetParamsFromSymbol(graph_exec, from_node, &second_result,
                                                      &g_contract_graph_symbol,
                                                      sizeof(second_result), 0,
                                                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(second_result == kSecondValue);
  REQUIRE(ReadSymbol() == kSecondValue);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}
