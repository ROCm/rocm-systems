/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <functional>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <memcpy1d_tests_common.hh>

#include "graph_tests_common.hh"

__device__ int graph_memcpy_1d_from_symbol[1];
__device__ int graph_memcpy_1d_to_symbol[1];

static inline hipMemcpyKind ReverseMemcpyDirection(const hipMemcpyKind direction) {
  switch (direction) {
    case hipMemcpyHostToDevice:
      return hipMemcpyDeviceToHost;
    case hipMemcpyDeviceToHost:
      return hipMemcpyHostToDevice;
    default:
      return direction;
  }
};

/**
 * @addtogroup hipGraphMemcpyNodeSetParams1D hipGraphMemcpyNodeSetParams1D
 * @{
 * @ingroup GraphTest
 * `hipGraphMemcpyNodeSetParams1D(hipGraphNode_t node, void *dst, const void *src, size_t count,
 * hipMemcpyKind kind)` - 	Sets a memcpy node's parameters to perform a 1-dimensional copy
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and the setting them to the correct values after which the graph is
 * executed and the results of the memcpy verified.
 * The test is run for all possible memcpy directions, with both the corresponding memcpy
 * kind and hipMemcpyDefault, as well as half page and full page allocation sizes.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphMemcpyNodeSetParams1D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphMemcpyNodeSetParams1D_Positive_Basic) {
  constexpr auto f = [](void* dst, void* src, size_t count, hipMemcpyKind direction) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, src, dst, count / 2,
                                      ReverseMemcpyDirection(direction)));
    HIP_CHECK(hipGraphMemcpyNodeSetParams1D(node, dst, src, count, direction));
    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

#if HT_NVIDIA
  MemcpyWithDirectionCommonTests<false>(f);
#else
  using namespace std::placeholders;

  SECTION("Device to host") {
    MemcpyDeviceToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDeviceToHost));
  }

  SECTION("Host to device") {
    MemcpyHostToDeviceShell<false>(std::bind(f, _1, _2, _3, hipMemcpyHostToDevice));
  }

  SECTION("Device to device") {
    SECTION("Peer access enabled") {
      MemcpyDeviceToDeviceShell<false, true>(std::bind(f, _1, _2, _3, hipMemcpyDeviceToDevice));
    }
    SECTION("Peer access disabled") {
      MemcpyDeviceToDeviceShell<false, false>(std::bind(f, _1, _2, _3, hipMemcpyDeviceToDevice));
    }
  }

  SECTION("Device to device with default kind") {
    SECTION("Peer access enabled") {
      MemcpyDeviceToDeviceShell<false, true>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
    }
    SECTION("Peer access disabled") {
      MemcpyDeviceToDeviceShell<false, false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
    }
  }

// Disabled on AMD due to defect - EXSWHTEC-209
#if 0
  SECTION("Host to host") {
    MemcpyHostToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyHostToHost));
  }

  SECTION("Host to host with default kind") {
    MemcpyHostToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
  }
#endif

// Disabled on AMD due to defect - EXSWHTEC-210
#if 0
  SECTION("Device to host with default kind") {
    MemcpyDeviceToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
  }

  SECTION("Host to device with default kind") {
    MemcpyHostToDeviceShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
  }
#endif

#endif
}

/**
 * Test Description
 * ------------------------
 *    - Verify API behaviour with invalid arguments:
 *        -# node is nullptr
 *        -# dst is nullptr
 *        -# src is nullptr
 *        -# kind is an invalid enum value
 *        -# count is zero
 *        -# count is larger than dst allocation size
 *        -# count is larger than src allocation size
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphMemcpyNodeSetParams1D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphMemcpyNodeSetParams1D_Negative_Parameters) {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int src[2] = {}, dst[2] = {};

  hipGraphNode_t node = nullptr;
  HIP_CHECK(
      hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, src, sizeof(dst), hipMemcpyDefault));


  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams1D(nullptr, dst, src, sizeof(dst), hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  MemcpyWithDirectionCommonNegativeTests(
      std::bind(hipGraphMemcpyNodeSetParams1D, node, _1, _2, _3, _4), dst, src, sizeof(dst),
      hipMemcpyDefault);

  SECTION("count == 0") {
    HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams1D(node, dst, src, 0, hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  SECTION("count larger than dst allocation size") {
    LinearAllocGuard<int> dev_dst(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(
        hipGraphMemcpyNodeSetParams1D(node, dev_dst.ptr(), src, sizeof(src), hipMemcpyDefault),
        hipErrorInvalidValue);
  }

  SECTION("count larger than src allocation size") {
    LinearAllocGuard<int> dev_src(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(
        hipGraphMemcpyNodeSetParams1D(node, dst, dev_src.ptr(), sizeof(dst), hipMemcpyDefault),
        hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Verify that the generic 1D setter retargets from-symbol and to-symbol nodes.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemcpyNodeSetParams1D.cc
 */
HIP_TEST_CASE(Unit_hipGraphMemcpyNodeSetParams1D_SymbolRetarget) {
  SECTION("from-symbol node") {
    LinearAllocGuard<int> src(LinearAllocs::hipMalloc, sizeof(int));
    int symbol_value = 61;
    int device_value = 62;
    int result = 0;
    HIP_CHECK(
        hipMemcpyToSymbol(HIP_SYMBOL(graph_memcpy_1d_from_symbol), &symbol_value, sizeof(int)));
    HIP_CHECK(hipMemcpy(src.ptr(), &device_value, sizeof(int), hipMemcpyHostToDevice));

    hipGraph_t graph = nullptr;
    hipGraphNode_t node = nullptr;
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&node, graph, nullptr, 0, &result,
                                              HIP_SYMBOL(graph_memcpy_1d_from_symbol), sizeof(int),
                                              0, hipMemcpyDeviceToHost));
    HIP_CHECK(hipGraphMemcpyNodeSetParams1D(node, &result, src.ptr(), sizeof(int),
                                            hipMemcpyDeviceToHost));
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    REQUIRE(result == device_value);

    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("to-symbol node") {
    LinearAllocGuard<int> src(LinearAllocs::hipMalloc, sizeof(int));
    LinearAllocGuard<int> dst(LinearAllocs::hipMalloc, sizeof(int));
    int value = 71;
    int zero = 0;
    HIP_CHECK(hipMemcpy(src.ptr(), &value, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dst.ptr(), &zero, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(graph_memcpy_1d_to_symbol), &zero, sizeof(int)));

    hipGraph_t graph = nullptr;
    hipGraphNode_t node = nullptr;
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&node, graph, nullptr, 0,
                                            HIP_SYMBOL(graph_memcpy_1d_to_symbol), src.ptr(),
                                            sizeof(int), 0, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipGraphMemcpyNodeSetParams1D(node, dst.ptr(), src.ptr(), sizeof(int),
                                            hipMemcpyDeviceToDevice));
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    int actual = 0;
    HIP_CHECK(hipMemcpy(&actual, dst.ptr(), sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(actual == value);
    int symbol_actual = -1;
    HIP_CHECK(hipMemcpyFromSymbol(&symbol_actual, HIP_SYMBOL(graph_memcpy_1d_to_symbol),
                                  sizeof(int)));
    REQUIRE(symbol_actual == zero);
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * End doxygen group GraphTest.
 * @}
 */
