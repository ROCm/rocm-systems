/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "vulkan_test.hh"
#include "wait_semaphore_common.hh"

/**
 * @addtogroup hipGraphExternalSemaphoresWaitNodeGetParams
 * hipGraphExternalSemaphoresWaitNodeGetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphExternalSemaphoresWaitNodeGetParams(hipGraphNode_t hNode,
 * hipExternalSemaphoreWaitNodeParams* params_out)` - Returns external semaphore wait node params.
 * ________________________
 * Test cases from other APIs:
 *  - @ref Unit_hipGraphExternalSemaphoresWaitNodeSetParams_Positive_Basic
 *  - @ref Unit_hipGraphExternalSemaphoresWaitNodeSetParams_Vulkan_Positive_Timeline_Semaphore
 *  - @ref Unit_hipGraphExternalSemaphoresWaitNodeSetParams_Vulkan_Positive_Multiple_Semaphores
 */


/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphExternalSemaphoresWaitNodeGetParams behavior with invalid
 * arguments:
 *    -# Nullptr graph node
 *    -# Nullptr params
 *    -# Node is destroyed
 * Test source
 * ------------------------
 *  - /unit/vulkan_interop/hipGraphExternalSemaphoresWaitNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipGraphExternalSemaphoresWaitNodeGetParams_Negative_Parameters) {
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
  hipExternalSemaphoreWaitNodeParams retrieved_params;

  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddExternalSemaphoresWaitNode(&node, graph, nullptr, 0, &node_params));

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresWaitNodeGetParams(nullptr, &retrieved_params),
                    hipErrorInvalidValue);
  }

  SECTION("params_out == nullptr") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresWaitNodeGetParams(node, nullptr),
                    hipErrorInvalidValue);
  }

// Disabled on AMD due to defect - EXSWHTEC-208
#if HT_NVIDIA
  SECTION("Node is destroyed") {
    hipGraph_t graph_temp = nullptr;
    HIP_CHECK(hipGraphCreate(&graph_temp, 0));
    hipGraphNode_t node_temp = nullptr;
    HIP_CHECK(
        hipGraphAddExternalSemaphoresWaitNode(&node_temp, graph_temp, nullptr, 0, &node_params));
    HIP_CHECK(hipGraphDestroy(graph_temp));
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresWaitNodeGetParams(node_temp, &retrieved_params),
                    hipErrorInvalidValue);
  }
#endif
}

/**
 * End doxygen group GraphTest.
 * @}
 */
