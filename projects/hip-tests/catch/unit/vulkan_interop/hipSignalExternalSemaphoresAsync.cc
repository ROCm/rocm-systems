/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "vulkan_test.hh"
#include "signal_semaphore_common.hh"

TEST_CASE(Unit_hipSignalExternalSemaphoresAsync_Vulkan_Positive_Binary_Semaphore) {
  SignalExternalSemaphoreCommon(hipSignalExternalSemaphoresAsync);
}

// Timeline semaphores unsupported on AMD
#if HT_NVIDIA
TEST_CASE(Unit_hipSignalExternalSemaphoresAsync_Vulkan_Positive_Timeline_Semaphore) {
  SignalExternalTimelineSemaphoreCommon(hipSignalExternalSemaphoresAsync);
}

TEST_CASE(Unit_hipSignalExternalSemaphoresAsync_Vulkan_Positive_Multiple_Semaphores) {
  SignalExternalMultipleSemaphoresCommon(hipSignalExternalSemaphoresAsync);
}
#endif

TEST_CASE(Unit_hipSignalExternalSemaphoresAsync_Vulkan_Negative_Parameters) {
  VulkanTest vkt(enable_validation);
  hipExternalSemaphoreSignalParams signal_params = {};
  signal_params.params.fence.value = 1;

  SECTION("extSemArray == nullptr") {
    HIP_CHECK_ERROR(hipSignalExternalSemaphoresAsync(nullptr, &signal_params, 1, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("paramsArray == nullptr") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    HIP_CHECK_ERROR(hipSignalExternalSemaphoresAsync(&hip_ext_semaphore, nullptr, 1, nullptr),
                    hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }

  SECTION("Wait params flags  != 0") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    signal_params.flags = 1;
    HIP_CHECK_ERROR(
        hipSignalExternalSemaphoresAsync(&hip_ext_semaphore, &signal_params, 1, nullptr),
        hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }

  SECTION("Invalid stream") {
    const auto hip_ext_semaphore = ImportBinarySemaphore(vkt);
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipSignalExternalSemaphoresAsync(&hip_ext_semaphore, &signal_params, 1, stream),
                    hipErrorInvalidValue);
    HIP_CHECK(hipDestroyExternalSemaphore(hip_ext_semaphore));
  }
}