/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "vulkan_test.hh"

constexpr bool enable_validation = false;

HIP_TEST_CASE(Unit_hipDestroyExternalSemaphore_Vulkan_Negative_Parameters) {
  SECTION("extSem == nullptr") {
    HIP_CHECK_ERROR(hipDestroyExternalSemaphore(nullptr), hipErrorInvalidValue);
  }

// Segfaults in Nvidia and Amd
#if 0
  SECTION("Double free") {
    VulkanTest vkt(enable_validation);
    const auto ext_semaphore = ImportBinarySemaphore(vkt);
    HIP_CHECK(hipDestroyExternalSemaphore(ext_semaphore));
    HIP_CHECK_ERROR(hipDestroyExternalSemaphore(ext_semaphore), hipErrorInvalidValue);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Test hipDestroyExternalSemaphore while stream is capturing.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipDestroyExternalSemaphore.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipDestroyExternalSemaphore_Vulkan_Capture) {
  VulkanTest vkt(enable_validation);
  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  auto handle_desc = vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_BINARY);
  hipExternalSemaphore_t ext_semaphore;

  hipError_t memcpy_err = hipSuccess;
  HIP_CHECK_ERROR(hipImportExternalSemaphore(&ext_semaphore, &handle_desc), memcpy_err);

  BEGIN_CAPTURE_SYNC(memcpy_err, true);
  HIP_CHECK_ERROR(hipDestroyExternalSemaphore(ext_semaphore), memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);
}
