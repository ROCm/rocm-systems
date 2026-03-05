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

#include "vulkan_test.hh"

constexpr bool enable_validation = false;

TEST_CASE("Unit_hipImportExternalMemory_Vulkan_Negative_Parameters") {
  VulkanTest vkt(enable_validation);
#if HT_NVIDIA
  const auto storage = vkt.CreateMappedStorage<int>(1, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  auto desc = vkt.BuildMemoryDescriptor(storage.memory, sizeof(*storage.host_ptr));
  hipExternalMemory_t ext_memory;
#endif

// Disabled due to defect - EXSWHTEC-182
#if HT_NVIDIA
  SECTION("extMem_out == nullptr") {
    HIP_CHECK_ERROR(hipImportExternalMemory(nullptr, &desc), hipErrorInvalidValue);
  }
#endif

// Disabled due to defect - EXSWHTEC-183
#if HT_NVIDIA
  SECTION("memHandleDesc == nullptr") {
    HIP_CHECK_ERROR(hipImportExternalMemory(&ext_memory, nullptr), hipErrorInvalidValue);
  }
#endif

// Disabled due to defect - EXSWHTEC-185
#if HT_NVIDIA
  SECTION("memHandleDesc.size == 0") {
    desc.size = 0;
    HIP_CHECK_ERROR(hipImportExternalMemory(&ext_memory, &desc), hipErrorInvalidValue);
  }
#endif

// Disabled due to defect - EXSWHTEC-186
#if HT_NVIDIA
  SECTION("Invalid memHandleDesc.flags") {
    desc.flags = 2;
    HIP_CHECK_ERROR(hipImportExternalMemory(&ext_memory, &desc), hipErrorInvalidValue);
  }
#endif

// Disabled due to defect - EXSWHTEC-184
#if HT_NVIDIA
  SECTION("Invalid memHandleDesc.type") {
    desc.type = static_cast<hipExternalMemoryHandleType>(-1);
    HIP_CHECK_ERROR(hipImportExternalMemory(&ext_memory, &desc), hipErrorInvalidValue);
  }
#endif

#ifdef _WIN32
  SECTION("memHandleDesc.handle == NULL") {
    hipExternalMemory_t ext_memory;
    hipExternalMemoryHandleDesc desc;
    desc.handle.win32.handle = NULL;
    HIP_CHECK_ERROR(hipImportExternalMemory(&ext_memory, &desc), hipErrorInvalidValue);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Test hipImportExternalMemory while stream is capturing.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipImportExternalMemory.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipImportExternalMemory_Vulkan_Capture") {
  VulkanTest vkt(enable_validation);
  using type = uint8_t;
  constexpr uint32_t count = 2;

  const auto vk_storage =
      vkt.CreateMappedStorage<type>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  if (vk_storage.memory == nullptr) {
    return;
  }

  const auto hip_ext_mem_desc = vkt.BuildMemoryDescriptor(vk_storage.memory, vk_storage.size);
  hipExternalMemory_t hip_ext_memory;

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, true);
  HIP_CHECK_ERROR(hipImportExternalMemory(&hip_ext_memory, &hip_ext_mem_desc),
                  memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);
}