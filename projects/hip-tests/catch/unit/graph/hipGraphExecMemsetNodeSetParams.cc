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

#include <hip_test_common.hh>

#include "graph_memset_node_test_common.hh"
#include "graph_tests_common.hh"

/**
 * @addtogroup hipGraphExecMemsetNodeSetParams hipGraphExecMemsetNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphExecMemsetNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node, const
 * hipMemsetParams *pNodeParams)` -
 * Sets the parameters for a memset node in the given graphExec
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and then setting them to the correct values in the executable graph.
 * The executable graph is run and the results of the memset is verified.
 * hipGraphMemsetNodeGetParams is used to verify that node parameters in the graph were not updated,
 * which also constitutes a test for said API.
 * The test is repeated for all valid element sizes(1,
 * 2, 4), and several allocations of different width(height is always 1 because only 1D memset nodes
 * can be updated), both on host and device
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecMemsetNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE("Unit_hipGraphExecMemsetNodeSetParams_Positive_Basic", "", uint8_t, uint16_t,
                   uint32_t) {
  CHECK_IMAGE_SUPPORT

  const size_t width = GENERATE(1, 64, kPageSize / sizeof(TestType) + 1);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t node = nullptr;
  LinearAllocGuard<TestType> initial_alloc(LinearAllocs::hipMalloc, 2 * sizeof(TestType));

  hipMemsetParams initial_params = {};
  initial_params.dst = initial_alloc.ptr();
  initial_params.elementSize = sizeof(TestType);
  initial_params.width = 2;
  initial_params.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &initial_params));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  LinearAllocGuard2D<TestType> alloc(width, 1);
  constexpr TestType set_value = 42;
  hipMemsetParams params = {};
  params.dst = alloc.ptr();
  params.elementSize = sizeof(TestType);
  params.width = width;
  params.height = 1;
  params.value = set_value;
  HIP_CHECK(hipGraphExecMemsetNodeSetParams(graph_exec, node, &params));

  hipMemsetParams retrieved_params = {};
  HIP_CHECK(hipGraphMemsetNodeGetParams(node, &retrieved_params));
  REQUIRE(initial_params.dst == retrieved_params.dst);
  REQUIRE(initial_params.elementSize == retrieved_params.elementSize);
  REQUIRE(initial_params.width == retrieved_params.width);
  REQUIRE(initial_params.height == retrieved_params.height);
  REQUIRE(initial_params.pitch == retrieved_params.pitch);
  REQUIRE(initial_params.value == retrieved_params.value);

  HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));

  LinearAllocGuard<TestType> buffer(LinearAllocs::hipHostMalloc, width * sizeof(TestType));
  HIP_CHECK(hipMemcpy2D(buffer.ptr(), width * sizeof(TestType), alloc.ptr(), alloc.pitch(),
                        width * sizeof(TestType), 1, hipMemcpyDeviceToHost));
  ArrayFindIfNot(buffer.ptr(), set_value, width);
}

/**
 * Test Description
 * ------------------------
 *    - Verify API behaviour with invalid arguments:
 *        -# pGraphExec is nullptr
 *        -# node is nullptr
 *        -# pNodeParams is nullptr
 *        -# pNodeParams::dst is nullptr
 *        -# pNodeParams::elementSize is different from 1, 2, and 4
 *        -# pNodeParams::width is zero
 *        -# pNodeParams::width is larger than the allocated memory region
 *        -# pNodeParams::height is zero
 *        -# pNodeParams::pitch is less than width when height is more than 1
 *        -# pNodeParams::pitch * pMemsetParams::height is larger than the allocated memory region
 *        -# pNodeParams::dst holds a pointer to memory allocated on a device different from the one
 * the original dst was allocated on
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecMemsetNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecMemsetNodeSetParams_Negative_Parameters") {
  // FIXME: this test tests 1D/2D/3D stuff in one single go, need to decouple it so that it can run
  // on devices with no image support
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  LinearAllocGuard<int> alloc(LinearAllocs::hipMalloc, 4 * sizeof(int));
  hipMemsetParams params = {};
  params.dst = alloc.ptr();
  params.elementSize = sizeof(*alloc.ptr());
  params.width = 1;
  params.height = 1;
  params.value = 42;

  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params))

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  SECTION("pGraphExec == nullptr") {
    HIP_CHECK_ERROR(hipGraphExecMemsetNodeSetParams(nullptr, node, &params), hipErrorInvalidValue);
  }

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphExecMemsetNodeSetParams(graph_exec, nullptr, &params),
                    hipErrorInvalidValue);
  }

  MemsetCommonNegative(std::bind(hipGraphExecMemsetNodeSetParams, graph_exec, node, _1), params);

  SECTION("Changing dst allocation device") {
    if (HipTest::getDeviceCount() < 2) {
      HipTest::HIP_SKIP_TEST("Test requires two connected GPUs");
    } else {
      HIP_CHECK(hipSetDevice(1));
      LinearAllocGuard<int> new_alloc(LinearAllocs::hipMalloc, 4 * sizeof(int));
      params.dst = new_alloc.ptr();
      HIP_CHECK_ERROR(hipGraphExecMemsetNodeSetParams(graph_exec, node, &params),
                      hipErrorInvalidValue);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify that a 2D node cannot be updated
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecMemsetNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecMemsetNodeSetParams_Negative_Updating_Non1D_Node") {
  CHECK_IMAGE_SUPPORT

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  LinearAllocGuard2D<int> alloc(2, 2);
  hipMemsetParams params = {};
  params.dst = alloc.ptr();
  params.elementSize = sizeof(*alloc.ptr());
  params.width = 1;
  params.height = 2;
  params.pitch = alloc.pitch();
  params.value = 42;

  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params))

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  params.width = 2;
  HIP_CHECK_ERROR(hipGraphExecMemsetNodeSetParams(graph_exec, node, &params), hipErrorInvalidValue);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
