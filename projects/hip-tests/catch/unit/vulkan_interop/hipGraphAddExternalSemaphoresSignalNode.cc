/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef _WIN64
#define NOMINMAX
#endif /* _WIN64 */

#include <functional>

#include "vulkan_test.hh"
#include "signal_semaphore_common.hh"
#include "graph_tests_common.hh"

/**
 * @addtogroup hipGraphAddExternalSemaphoresSignalNode hipGraphAddExternalSemaphoresSignalNode
 * @{
 * @ingroup GraphTest
 * `hipGraphAddExternalSemaphoresSignalNode(hipGraphNode_t* pGraphNode, hipGraph_t graph, const
 * hipGraphNode_t* pDependencies, size_t numDependencies, const
 * hipExternalSemaphoreSignalNodeParams* nodeParams);` - Creates a external semaphor signal node and
 * adds it to a graph.
 */

/**
 * Test Description
 * ------------------------
 *  - Creates two host visible Vulkan buffers.
 *  - Adds a buffer copy command which will copy from one buffer to another.
 *  - Creates an external Vulkan binary semaphore.
 *  - Creates a Vulkan fence and signals semaphore asynchronously.
 *  - Waits for the operation to finish successfully.
 * Test source
 * ------------------------
 *  - unit/vulkan_interop/hipGraphAddExternalSemaphoresSignalNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddExternalSemaphoresSignalNode_Positive_Basic") {
  SignalExternalSemaphoreCommon(GraphExtSemaphoreSignalWrapper<>);
}

// Timeline semaphores unsupported on AMD
#if HT_NVIDIA

/**
 * Test Description
 * ------------------------
 *  - Creates an external Vulkan timeline semaphore.
 *  - Imports the semaphore and signals.
 *  - Waits for the operation to finish successfully.
 * Test source
 * ------------------------
 *  - unit/vulkan_interop/hipGraphAddExternalSemaphoresSignalNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddExternalSemaphoresSignalNode_Vulkan_Positive_Timeline_Semaphore") {
  SignalExternalTimelineSemaphoreCommon(GraphExtSemaphoreSignalWrapper<>);
}

/**
 * Test Description
 * ------------------------
 *  - Creates two host visible Vulkan buffers.
 *  - Adds a buffer copy command which will copy from one buffer to another.
 *  - Creates multiple external Vulkan binary semaphores.
 *  - Createas a Vulkan fence and signals semaphores.
 *  - Waits for the operations to finish successfully.
 * Test source
 * ------------------------
 *  - unit/vulkan_interop/hipGraphAddExternalSemaphoresSignalNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddExternalSemaphoresSignalNode_Vulkan_Positive_Multiple_Semaphores") {
  SignalExternalMultipleSemaphoresCommon(GraphExtSemaphoreSignalWrapper<>);
}
#endif


/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddExternalSemaphoresSignalNode behavior with invalid arguments:
 *    -# Nullptr graph
 *    -# Nullptr graph node
 *    -# Invalid numDependencies for null list of dependencies
 *    -# Node in dependency is from different graph
 *    -# Invalid numNodes
 *    -# Duplicate node in dependencies
 * Test source
 * ------------------------
 *  - /unit/vulkan_interop/hipGraphAddExternalSemaphoresSignalNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddExternalSemaphoresSignalNode_Vulkan_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  VulkanTest vkt(enable_validation);
  hipExternalSemaphoreSignalParams signal_params = {};
  signal_params.params.fence.value = 1;
  auto hip_ext_semaphore = ImportBinarySemaphore(vkt);

  hipExternalSemaphoreSignalNodeParams node_params = {};
  node_params.extSemArray = &hip_ext_semaphore;
  node_params.paramsArray = &signal_params;
  node_params.numExtSems = 1;

  GraphAddNodeCommonNegativeTests(
      std::bind(hipGraphAddExternalSemaphoresSignalNode, _1, _2, _3, _4, &node_params), graph);

  HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
