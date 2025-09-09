/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <functional>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <resource_guards.hh>

#include "graph_memcpy_to_from_symbol_common.hh"

HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(char)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(int)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(float)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(double)

HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_ALTERNATE_GLOBALS(char)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_ALTERNATE_GLOBALS(int)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_ALTERNATE_GLOBALS(float)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_ALTERNATE_GLOBALS(double)

template <typename T>
void GraphExecMemcpyFromSymbolSetParamsShell(const void* symbol, const void* alt_symbol,
                                             size_t offset, const std::vector<T> expected) {
  const auto f = [alt_symbol, is_arr = expected.size() > 1](void* dst, const void* symbol,
                                                            size_t count, size_t offset,
                                                            hipMemcpyKind direction) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t node = nullptr;

    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(
        &node, graph, nullptr, 0, reinterpret_cast<T*>(dst) + is_arr, alt_symbol,
        count - is_arr * sizeof(T), offset + is_arr * sizeof(T), direction));

    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphExecMemcpyNodeSetParamsFromSymbol(graph_exec, node, dst, symbol, count,
                                                        offset, direction));

    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

  MemcpyFromSymbolShell(f, symbol, offset, std::move(expected));
}

/**
 * @addtogroup hipGraphExecMemcpyNodeSetParamsFromSymbol hipGraphExecMemcpyNodeSetParamsFromSymbol
 * @{
 * @ingroup GraphTest
 * `hipGraphExecMemcpyNodeSetParamsFromSymbol(hipGraphExec_t hGraphExec, hipGraphNode_t node, void
 * *dst, const void *symbol, size_t count, size_t offset, hipMemcpyKind kind)` -
 * Sets the parameters for a memcpy node in the given graphExec to copy from a symbol on the
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that data is correctly copied from a symbol after node parameters are set following
 * node addition. A graph is constructed to which a MemcpyFromSymbol node is added with valid but
 * incorrect parameters. After the graph is instantiated the parameters are updated to correct
 * values and the graph executed. Values in destination memory are compared against values known to
 * be in symbol memory.
 * The test is run for scalar, const scalar, array, and const array symbols of types char, int,
 * float and double. For array symbols, the test is repeated for zero and non-zero offset values.
 * Verification is performed for destination memory allocated on host and device.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecMemcpyNodeSetParamsFromSymbol.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Positive_Basic") {
  SECTION("char") {
    HIP_GRAPH_MEMCPY_NODE_SET_PARAMS_TO_FROM_SYMBOL_TEST(GraphExecMemcpyFromSymbolSetParamsShell, 1,
                                                         char);
  }

  SECTION("int") {
    HIP_GRAPH_MEMCPY_NODE_SET_PARAMS_TO_FROM_SYMBOL_TEST(GraphExecMemcpyFromSymbolSetParamsShell, 1,
                                                         int);
  }

  SECTION("float") {
    HIP_GRAPH_MEMCPY_NODE_SET_PARAMS_TO_FROM_SYMBOL_TEST(GraphExecMemcpyFromSymbolSetParamsShell, 1,
                                                         float);
  }

  SECTION("double") {
    HIP_GRAPH_MEMCPY_NODE_SET_PARAMS_TO_FROM_SYMBOL_TEST(GraphExecMemcpyFromSymbolSetParamsShell, 1,
                                                         double);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Verify API behavior with invalid arguments:
 *      -# gGraphExec is nullptr
 *      -# node is nullptr
 *      -# dst is nullptr
 *      -# symbol is nullptr
 *      -# count is zero
 *      -# count is larger than symbol size
 *      -# count + offset is larger than symbol size
 *      -# kind is illogical (hipMemcpyHostToDevice)
 *      -# kind is an invalid enum value
 *      -# Changing memcpy direction
 *      -# Changing dst to memory allocated on a different device than the original dst
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecMemcpyNodeSetParamsFromSymbol.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParamsFromSymbol_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  LinearAllocGuard<int> var(LinearAllocs::hipMalloc, sizeof(int));
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&node, graph, nullptr, 0, var.ptr(),
                                            SYMBOL(int_device_var), sizeof(*var.ptr()), 0,
                                            hipMemcpyDefault));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  SECTION("hGraphExec == nullptr") {
    HIP_CHECK_ERROR(
        hipGraphExecMemcpyNodeSetParamsFromSymbol(nullptr, node, var.ptr(), SYMBOL(int_device_var),
                                                  sizeof(*var.ptr()), 0, hipMemcpyDefault),
        hipErrorInvalidValue);
  }

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParamsFromSymbol(
                        graph_exec, nullptr, var.ptr(), SYMBOL(int_device_var), sizeof(*var.ptr()),
                        0, hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  MemcpyFromSymbolCommonNegative(
      std::bind(hipGraphExecMemcpyNodeSetParamsFromSymbol, graph_exec, node, _1, _2, _3, _4, _5),
      var.ptr(), SYMBOL(int_device_var), sizeof(*var.ptr()));

// Disabled on AMD due to defect
#if HT_NVIDIA
  SECTION("Changing memcpy direction") {
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParamsFromSymbol(
                        graph_exec, node, var.ptr(), SYMBOL(int_device_var), sizeof(*var.ptr()), 0,
                        hipMemcpyDeviceToHost),
                    hipErrorInvalidValue);
  }
#endif

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
