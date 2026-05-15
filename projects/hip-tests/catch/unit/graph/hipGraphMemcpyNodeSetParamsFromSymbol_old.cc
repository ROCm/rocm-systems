/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Testcase Scenarios of hipGraphMemcpyNodeSetParamsFromSymbol API:

Functional :
1) Allocate global symbol memory, add node to the graph.
 Set/Update the new values to the node. Make sure they are taking effect.
2) Allocate const symbol memory, add node to the graph.
 Set/Update the new values to the node. Make sure they are taking effect.

Negative :
1) Pass GraphNode as nullptr and check if api returns error.
2) Pass destination ptr as nullptr, api expected to return error code.
3) Pass source/symbol ptr as nullptr, api expected to return error code.
4) Pass count as zero, api expected to return error code.
5) Pass count more than allocated size for source and destination ptr, api should return error code.
6) Pass offset+count greater than allocated size, api expected to return error code.
7) Pass same symbol pointer as destination ptr and source ptr, api expected to return error code.
8) Pass both destination ptr and source ptr as 2 different symbol ptr, api expected to return error
code.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <limits>
#define SIZE 256

__device__ int globalIn[SIZE];
__device__ int globalOut[SIZE];
__device__ __constant__ int globalConst[SIZE];


/* Test verifies hipGraphMemcpyNodeSetParamsFromSymbol API Negative scenarios.
 */
HIP_TEST_CASE(Unit_hipGraphMemcpyNodeSetParamsFromSymbol_Negative) {
  constexpr size_t Nbytes = SIZE * sizeof(int);
  int *A_d{nullptr}, *B_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, nullptr, &A_h, &B_h, nullptr, SIZE, false);

  hipError_t ret;
  hipGraph_t graph;
  hipGraphNode_t memcpyToSymbolNode, memcpyFromSymbolNode, memcpyH2D_A;
  std::vector<hipGraphNode_t> dependencies;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Adding MemcpyNode
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  dependencies.push_back(memcpyH2D_A);

  // Adding MemcpyNodeToSymbol
  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&memcpyToSymbolNode, graph, dependencies.data(),
                                          dependencies.size(), HIP_SYMBOL(globalIn), A_d, Nbytes, 0,
                                          hipMemcpyDeviceToDevice));
  dependencies.clear();
  dependencies.push_back(memcpyToSymbolNode);

  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&memcpyFromSymbolNode, graph, dependencies.data(),
                                            dependencies.size(), B_h, HIP_SYMBOL(globalConst),
                                            Nbytes, 0, hipMemcpyDeviceToHost));
  SECTION("Pass GraphNode as nullptr") {
    ret = hipGraphMemcpyNodeSetParamsFromSymbol(nullptr, B_h, HIP_SYMBOL(globalConst), Nbytes, 0,
                                                hipMemcpyDeviceToHost);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass destination ptr as nullptr") {
    ret = hipGraphMemcpyNodeSetParamsFromSymbol(
        memcpyFromSymbolNode, nullptr, HIP_SYMBOL(globalConst), Nbytes, 0, hipMemcpyDeviceToHost);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass source/symbol ptr as nullptr") {
    ret = hipGraphMemcpyNodeSetParamsFromSymbol(memcpyFromSymbolNode, B_h, nullptr, Nbytes, 0,
                                                hipMemcpyDeviceToHost);
    REQUIRE(hipErrorInvalidSymbol == ret);
  }
  SECTION("Pass count as zero") {
    ret = hipGraphMemcpyNodeSetParamsFromSymbol(memcpyFromSymbolNode, B_h, HIP_SYMBOL(globalConst),
                                                0, 0, hipMemcpyDeviceToHost);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass count more than allocated size for source and dstn ptr") {
    ret = hipGraphMemcpyNodeSetParamsFromSymbol(memcpyFromSymbolNode, B_h, HIP_SYMBOL(globalConst),
                                                Nbytes + 10, 0, hipMemcpyDeviceToHost);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass offset non zero so that offset+count > allocated size") {
    ret = hipGraphMemcpyNodeSetParamsFromSymbol(memcpyFromSymbolNode, B_h, HIP_SYMBOL(globalConst),
                                                Nbytes, 10, hipMemcpyDeviceToHost);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass same symbol pointer as dstn ptr and source ptr") {
    ret = hipGraphMemcpyNodeSetParamsFromSymbol(memcpyFromSymbolNode, HIP_SYMBOL(globalConst),
                                                HIP_SYMBOL(globalConst), Nbytes, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass both dstn ptr and source ptr as 2 different symbol ptr") {
    ret = hipGraphMemcpyNodeSetParamsFromSymbol(memcpyFromSymbolNode, HIP_SYMBOL(globalOut),
                                                HIP_SYMBOL(globalIn), Nbytes, 0,
                                                hipMemcpyDeviceToDevice);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HipTest::freeArrays<int>(A_d, B_d, nullptr, A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphDestroy(graph));
}
