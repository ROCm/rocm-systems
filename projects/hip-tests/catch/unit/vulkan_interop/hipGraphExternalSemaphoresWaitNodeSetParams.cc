/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "vulkan_test.hh"
#include "wait_semaphore_common.hh"

/**
 * @addtogroup hipGraphExternalSemaphoresWaitNodeSetParams
 * hipGraphExternalSemaphoresWaitNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphExternalSemaphoresWaitNodeSetParams(hipGraphNode_t hNode, const
 * hipExternalSemaphoreWaitNodeParams* nodeParams)` - Updates node parameters in the external
 * semaphore wait node.
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and the setting them to the correct values. The graph is run and it is
 * verified that the graph node waits for the external binary semaphore and operation finishes
 * successfully.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipGraphExternalSemaphoresWaitNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE(Unit_hipGraphExternalSemaphoresWaitNodeSetParams_Positive_Basic) {
  WaitExternalSemaphoreCommon(GraphExtSemaphoreWaitWrapper<true>);
}

// Timeline semaphores unsupported on AMD
#if HT_NVIDIA

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and the setting them to the correct values. The graph is run and it is
 * verified that the graph node waits for the external timeline semaphore and operation finishes
 * successfully.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipGraphExternalSemaphoresWaitNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE(Unit_hipGraphExternalSemaphoresWaitNodeSetParams_Vulkan_Positive_Timeline_Semaphore) {
  WaitExternalTimelineSemaphoreCommon(GraphExtSemaphoreWaitWrapper<true>);
}
#endif

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and the setting them to the correct values. The graph is run and it is
 * verified that the graph node waits for the external binary semaphores and operation finishes
 * successfully.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipGraphExternalSemaphoresWaitNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE(Unit_hipGraphExternalSemaphoresWaitNodeSetParams_Vulkan_Positive_Multiple_Semaphores) {
  WaitExternalMultipleSemaphoresCommon(GraphExtSemaphoreWaitWrapper<true>);
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphExternalSemaphoresWaitNodeSetParams behavior with invalid
 * arguments:
 *    -# Nullptr graph node
 *    -# Nullptr params
 * Test source
 * ------------------------
 *  - /unit/vulkan_interop/hipGraphExternalSemaphoresWaitNodeSetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE(Unit_hipGraphExternalSemaphoresWaitNodeSetParams_Vulkan_Negative_Parameters) {
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  VulkanTest vkt(enable_validation);
  hipExternalSemaphoreWaitParams wait_params = {};
  wait_params.params.fence.value = 1;
  auto hip_ext_semaphore = ImportBinarySemaphore(vkt);

  hipExternalSemaphoreWaitNodeParams node_params = {};
  node_params.extSemArray = &hip_ext_semaphore;
  node_params.paramsArray = &wait_params;
  node_params.numExtSems = 1;

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresWaitNodeSetParams(nullptr, &node_params),
                    hipErrorInvalidValue);
  }

  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddExternalSemaphoresWaitNode(&node, graph, nullptr, 0, &node_params));

  SECTION("params == nullptr") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresWaitNodeSetParams(node, nullptr),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
