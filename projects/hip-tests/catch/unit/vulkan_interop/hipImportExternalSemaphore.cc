/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "vulkan_test.hh"

constexpr bool enable_validation = false;

TEST_CASE(Unit_hipImportExternalSemaphore_Vulkan_Negative_Parameters) {
  VulkanTest vkt(enable_validation);
  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  auto handle_desc = vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_BINARY);
  hipExternalSemaphore_t ext_semaphore;

  SECTION("extSem_out == nullptr") {
    HIP_CHECK_ERROR(hipImportExternalSemaphore(nullptr, &handle_desc), hipErrorInvalidValue);
  }

  SECTION("semHandleDesc == nullptr") {
    HIP_CHECK_ERROR(hipImportExternalSemaphore(&ext_semaphore, nullptr), hipErrorInvalidValue);
  }
  /*
   * CUDA doesn't specify the case
  SECTION("semHandleDesc.flags != 0") {
    handle_desc.flags = 1;
    HIP_CHECK_ERROR(hipImportExternalSemaphore(&ext_semaphore, &handle_desc), hipErrorInvalidValue);
  }
  */

  SECTION("Invalid semHandleDesc.type") {
    handle_desc.type = static_cast<hipExternalSemaphoreHandleType>(-1);
    HIP_CHECK_ERROR(hipImportExternalSemaphore(&ext_semaphore, &handle_desc), hipErrorInvalidValue);
  }

#ifdef _WIN32
  SECTION("semHandleDesc.handle == NULL") {
    handle_desc.handle.win32.handle = NULL;
    HIP_CHECK_ERROR(hipImportExternalSemaphore(&ext_semaphore, &handle_desc), hipErrorInvalidValue);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Test hipImportExternalSemaphore while stream is capturing.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipImportExternalSemaphore.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE(Unit_hipImportExternalSemaphore_Vulkan_Capture) {
  VulkanTest vkt(enable_validation);
  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  auto handle_desc = vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_BINARY);
  hipExternalSemaphore_t ext_semaphore;

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, true);
  HIP_CHECK_ERROR(hipImportExternalSemaphore(&ext_semaphore, &handle_desc), memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);
}