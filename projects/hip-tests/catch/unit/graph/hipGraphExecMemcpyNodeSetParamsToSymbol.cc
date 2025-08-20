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

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <functional>
#include <vector>
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
void GraphExecMemcpyToSymbolSetParamsShell(const void* symbol, const void* alt_symbol,
                                           size_t offset, const std::vector<T> set_values) {
  const auto f = [alt_symbol, is_arr = set_values.size() > 1](const void* symbol, void* src,
                                                              size_t count, size_t offset,
                                                              hipMemcpyKind direction) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(
        &node, graph, nullptr, 0, alt_symbol, reinterpret_cast<T*>(src) + is_arr,
        count - is_arr * sizeof(T), offset + is_arr * sizeof(T), direction));

    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphExecMemcpyNodeSetParamsToSymbol(graph_exec, node, symbol, src, count, offset,
                                                      direction));

    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

  MemcpyToSymbolShell(f, symbol, offset, std::move(set_values));
}

/**
 * @addtogroup hipGraphExecMemcpyNodeSetParamsToSymbol
 * hipGraphExecMemcpyNodeSetParamsToSymbol
 * @{
 * @ingroup GraphTest
 * `hipGraphExecMemcpyNodeSetParamsToSymbol(hipGraphExec_t hGraphExec,
 * hipGraphNode_t node, const void *symbol, void *src, size_t count, size_t
 * offset, hipMemcpyKind kind)` - Sets the parameters for a memcpy node in the
 * given graphExec to copy to a symbol on the device
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that data is correctly copied to a symbol after node parameters
 * are set following node addition. A graph is constructed to which a
 * MemcpyToSymbol node is added with valid but incorrect parameters. After the
 * graph is instantiated the parameters are updated to correct values and the
 * graph executed. After graph execution, a MemcpyFromSymbol is performed and
 * the copied values are compared against values known to have been copied to
 * symbol memory previously. The test is run for scalar, const scalar, array,
 * and const array symbols of types char, int, float and double. For array
 * symbols, the test is repeated for zero and non-zero offset values.
 * Verification is performed for destination memory allocated on host and
 * device. Test source
 * ------------------------
 *    - unit/graph/hipGraphExecMemcpyNodeSetParamsToSymbol.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Positive_Basic") {
  SECTION("char") {
    HIP_GRAPH_MEMCPY_NODE_SET_PARAMS_TO_FROM_SYMBOL_TEST(GraphExecMemcpyToSymbolSetParamsShell, 10,
                                                         char);
  }

  SECTION("int") {
    HIP_GRAPH_MEMCPY_NODE_SET_PARAMS_TO_FROM_SYMBOL_TEST(GraphExecMemcpyToSymbolSetParamsShell, 10,
                                                         int);
  }

  SECTION("float") {
    HIP_GRAPH_MEMCPY_NODE_SET_PARAMS_TO_FROM_SYMBOL_TEST(GraphExecMemcpyToSymbolSetParamsShell, 10,
                                                         float);
  }

  SECTION("double") {
    HIP_GRAPH_MEMCPY_NODE_SET_PARAMS_TO_FROM_SYMBOL_TEST(GraphExecMemcpyToSymbolSetParamsShell, 10,
                                                         double);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Verify API behavior with invalid arguments:
 *      -# gGraphExec is nullptr
 *      -# node is nullptr
 *      -# src is nullptr
 *      -# symbol is nullptr
 *      -# count is zero
 *      -# count is larger than symbol size
 *      -# count + offset is larger than symbol size
 *      -# kind is illogical (hipMemcpyDeviceToHost)
 *      -# kind is an invalid enum value
 *      -# Changing memcpy direction
 *      -# Changing src to memory allocated on a different device than the
 * original src Test source
 * ------------------------
 *    - unit/graph/hipGraphExecMemcpyNodeSetParamsToSymbol.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParamsToSymbol_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  LinearAllocGuard<int> var(LinearAllocs::hipMalloc, sizeof(int));
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNodeToSymbol(&node, graph, nullptr, 0, SYMBOL(int_device_var),
                                          var.ptr(), sizeof(*var.ptr()), 0, hipMemcpyDefault));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  SECTION("hGraphExec == nullptr") {
    HIP_CHECK_ERROR(
        hipGraphExecMemcpyNodeSetParamsToSymbol(nullptr, node, SYMBOL(int_device_var), var.ptr(),
                                                sizeof(*var.ptr()), 0, hipMemcpyDefault),
        hipErrorInvalidValue);
  }

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(
        hipGraphExecMemcpyNodeSetParamsToSymbol(graph_exec, nullptr, SYMBOL(int_device_var),
                                                var.ptr(), sizeof(*var.ptr()), 0, hipMemcpyDefault),
        hipErrorInvalidValue);
  }

  MemcpyToSymbolCommonNegative(
      std::bind(hipGraphExecMemcpyNodeSetParamsToSymbol, graph_exec, node, _1, _2, _3, _4, _5),
      SYMBOL(int_device_var), var.ptr(), sizeof(*var.ptr()));

  SECTION("Changing memcpy direction") {
    HIP_CHECK_ERROR(
        hipGraphExecMemcpyNodeSetParamsToSymbol(graph_exec, node, SYMBOL(int_device_var), var.ptr(),
                                                sizeof(*var.ptr()), 0, hipMemcpyHostToDevice),
        hipErrorInvalidValue);
  }

  SECTION("Changing src allocation device") {
    if (HipTest::getDeviceCount() < 2) {
      HipTest::HIP_SKIP_TEST("Test requires two connected GPUs");
    } else {
      HIP_CHECK(hipSetDevice(1));
      LinearAllocGuard<int> new_var(LinearAllocs::hipMalloc, sizeof(int));
      HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParamsToSymbol(
                          graph_exec, node, SYMBOL(int_device_var), new_var.ptr(),
                          sizeof(*new_var.ptr()), 0, hipMemcpyDefault),
                      hipErrorInvalidValue);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
