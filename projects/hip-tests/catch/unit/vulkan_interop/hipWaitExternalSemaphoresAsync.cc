/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "vulkan_test.hh"
#include "wait_semaphore_common.hh"

TEST_CASE(Unit_hipWaitExternalSemaphoresAsync_Vulkan_Positive_Binary_Semaphore) {
  WaitExternalSemaphoreCommon(hipWaitExternalSemaphoresAsync);
}

// Timeline semaphores unsupported on AMD
#if HT_NVIDIA
TEST_CASE(Unit_hipWaitExternalSemaphoresAsync_Vulkan_Positive_Timeline_Semaphore) {
  WaitExternalTimelineSemaphoreCommon(hipWaitExternalSemaphoresAsync);
}
#endif

TEST_CASE(Unit_hipWaitExternalSemaphoresAsync_Vulkan_Positive_Multiple_Semaphores) {
  WaitExternalMultipleSemaphoresCommon(hipWaitExternalSemaphoresAsync);
}

TEST_CASE(Unit_hipWaitExternalSemaphoresAsync_Vulkan_Negative_Parameters) {
  VulkanTest vkt(enable_validation);
  hipExternalSemaphoreWaitParams wait_params = {};
  wait_params.params.fence.value = 1;

  SECTION("extSemArray == nullptr") {
    HIP_CHECK_ERROR(hipWaitExternalSemaphoresAsync(nullptr, &wait_params, 1, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("paramsArray == nullptr") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    HIP_CHECK_ERROR(hipWaitExternalSemaphoresAsync(&hip_ext_semaphore, nullptr, 1, nullptr),
                    hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }

  SECTION("Wait params flag != 0") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    wait_params.flags = 1;
    HIP_CHECK_ERROR(hipWaitExternalSemaphoresAsync(&hip_ext_semaphore, &wait_params, 1, nullptr),
                    hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }

  SECTION("Invalid stream") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipWaitExternalSemaphoresAsync(&hip_ext_semaphore, &wait_params, 1, stream),
                    hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }
}