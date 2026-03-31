/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "vulkan_test.hh"

#if HT_AMD && 0
constexpr bool enable_validation = false;
#endif

HIP_TEST_CASE(Unit_hipDestroyExternalMemory_Vulkan_Negative_Parameters) {
  SECTION("extMem == nullptr") {
    HIP_CHECK_ERROR(hipDestroyExternalMemory(nullptr), hipErrorInvalidValue);
  }

// Segfaults in CUDA
// Disabled on AMD due to defect - EXSWHTEC-187
#if HT_AMD && 0
  SECTION("Double free") {
    VulkanTest vkt(enable_validation);
    const auto storage = vkt.CreateMappedStorage<int>(1, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    auto desc = vkt.BuildMemoryDescriptor(storage.memory, sizeof(*storage.host_ptr));
    hipExternalMemory_t ext_memory;
    HIP_CHECK(hipImportExternalMemory(&ext_memory, &desc));

    HIP_CHECK(hipDestroyExternalMemory(ext_memory));
    HIP_CHECK_ERROR(hipDestroyExternalMemory(ext_memory), hipErrorInvalidValue);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Test hipDestroyExternalMemory while stream is capturing.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipDestroyExternalMemory.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipDestroyExternalMemory_Vulkan_Capture) {
// Segfaults in CUDA
// Disabled on AMD due to defect - EXSWHTEC-187
#if HT_AMD && 0
  VulkanTest vkt(enable_validation);
  const auto storage = vkt.CreateMappedStorage<int>(1, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  auto desc = vkt.BuildMemoryDescriptor(storage.memory, sizeof(*storage.host_ptr));
  hipExternalMemory_t ext_memory;
  HIP_CHECK(hipImportExternalMemory(&ext_memory, &desc));

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, true);
  HIP_CHECK_ERROR(hipDestroyExternalMemory(ext_memory), memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);
#endif
}
