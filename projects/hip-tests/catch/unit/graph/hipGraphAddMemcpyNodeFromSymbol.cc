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

#include "graph_memcpy_to_from_symbol_common.hh"
#include "graph_tests_common.hh"

HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(char)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(int)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(float)
HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(double)

template <typename T>
void GraphMemcpyFromSymbolShell(void* symbol, size_t offset, const std::vector<T> expected) {
  const auto f = [](void* dst, const void* symbol, size_t count, size_t offset,
                    hipMemcpyKind direction) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphAddMemcpyNodeFromSymbol(&node, graph, nullptr, 0, dst, symbol, count, offset,
                                              direction));

    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

  MemcpyFromSymbolShell(f, symbol, offset, std::move(expected));
}

/**
 * @addtogroup hipGraphAddMemcpyNodeFromSymbol hipGraphAddMemcpyNodeFromSymbol
 * @{
 * @ingroup GraphTest
 * `hipGraphAddMemcpyNodeFromSymbol(hipGraphNode_t *pGraphNode, hipGraph_t graph, const
 * hipGraphNode_t *pDependencies, size_t numDependencies, void *dst, const void *symbol, size_t
 * count, size_t offset, hipMemcpyKind kind)` -
 * Creates a memcpy node to copy from a symbol on the device and adds it to a graph
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that data is correctly copied from a symbol. A graph is constructed to which a
 * MemcpyFromSymbol node is added. After graph execution, values in destination memory are compared
 * against values known to be in symbol memory.
 * The test is run for scalar, const scalar, array, and const array symbols of types char, int,
 * float and double. For array symbols, the test is repeated for zero and non-zero offset values.
 * Verification is performed for destination memory allocated on host and device.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddMemcpyNodeFromSymbol.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddMemcpyNodeFromSymbol_Positive_Basic") {
  SECTION("char") {
    HIP_GRAPH_ADD_MEMCPY_NODE_TO_FROM_SYMBOL_TEST(GraphMemcpyFromSymbolShell, 1, char);
  }

  SECTION("int") {
    HIP_GRAPH_ADD_MEMCPY_NODE_TO_FROM_SYMBOL_TEST(GraphMemcpyFromSymbolShell, 1, int);
  }

  SECTION("float") {
    HIP_GRAPH_ADD_MEMCPY_NODE_TO_FROM_SYMBOL_TEST(GraphMemcpyFromSymbolShell, 1, float);
  }

  SECTION("double") {
    HIP_GRAPH_ADD_MEMCPY_NODE_TO_FROM_SYMBOL_TEST(GraphMemcpyFromSymbolShell, 1, double);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Verify API behavior with invalid arguments:
 *      -# pGraphNodes is nullptr
 *      -# graph is nullptr
 *      -# pDependencies is nullptr when numDependencies is non-zero
 *      -# A node in pDependencies belongs to a different graph
 *      -# numDependencies in invalid
 *      -# A node appears twice in pDependencies
 *      -# dst is nullptr
 *      -# symbol is nullptr
 *      -# count is zero
 *      -# count is larger than symbol size
 *      -# count + offset is larger than symbol size
 *      -# kind is illogical (hipMemcpyHostToDevice)
 *      -# kind is an invalid enum value
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddMemcpyNodeFromSymbol.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphAddMemcpyNodeFromSymbol_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int var = 0;
  hipGraphNode_t node = nullptr;

  GraphAddNodeCommonNegativeTests(
      std::bind(hipGraphAddMemcpyNodeFromSymbol, _1, _2, _3, _4, &var, SYMBOL(int_device_var),
                sizeof(var), 0, hipMemcpyDefault),
      graph);

  MemcpyFromSymbolCommonNegative(
      std::bind(hipGraphAddMemcpyNodeFromSymbol, &node, graph, nullptr, 0, _1, _2, _3, _4, _5),
      &var, SYMBOL(int_device_var), sizeof(var));

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
