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

template <typename T> bool WriteAndValidateData(hipArray_t& array, size_t array_size) {
  T* write_data = new T[array_size];
  for (size_t i = 0; i < array_size; i++) {
    write_data[i] = rand() % 10;
  }
  HIP_CHECK(
      hipMemcpyToArray(array, 0, 0, write_data, array_size * sizeof(T), hipMemcpyHostToDevice));

  T* read_data = new T[array_size];
  HIP_CHECK(
      hipMemcpyFromArray(read_data, array, 0, 0, array_size * sizeof(T), hipMemcpyDeviceToHost));

  bool is_valid = true;
  for (size_t i = 0; i < array_size && is_valid; i++) {
    if (write_data[i] != read_data[i]) {
      is_valid = false;
    }
  }

  free(read_data);
  free(write_data);

  return is_valid;
}

TEST_CASE("Unit_hipExternalMemoryGetMappedMipmappedArray_Vulkan_Positive_Read_Write") {
  VulkanTest vkt(enable_validation);
  using type = uint8_t;
  constexpr uint32_t count = 16384;

  const auto vk_storage =
      vkt.CreateMappedStorage<type>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  if (vk_storage.memory == nullptr) {
    return;
  }

  const auto ext_mem_desc = vkt.BuildMemoryDescriptor(vk_storage.memory, vk_storage.size);
  hipExternalMemory_t ext_memory;
  HIP_CHECK(hipImportExternalMemory(&ext_memory, &ext_mem_desc));

  hipExternalMemoryMipmappedArrayDesc mipmapped_arr_desc = {};
  mipmapped_arr_desc.extent = {};
  mipmapped_arr_desc.extent.width = GENERATE(32, 128, 256);
  mipmapped_arr_desc.extent.height = GENERATE(1, 4, 16);
  mipmapped_arr_desc.extent.depth = GENERATE(0, 4);
  mipmapped_arr_desc.flags = hipArrayDefault;
  mipmapped_arr_desc.formatDesc = hipCreateChannelDesc<type>();
  mipmapped_arr_desc.numLevels = GENERATE(1, 2, 4);
  mipmapped_arr_desc.offset = 0;

  hipMipmappedArray_t mipmapped_arr = nullptr;
  HIP_CHECK(
      hipExternalMemoryGetMappedMipmappedArray(&mipmapped_arr, ext_memory, &mipmapped_arr_desc));

  hipArray_t level_arr = nullptr;
  HIP_CHECK(hipGetMipmappedArrayLevel(&level_arr, mipmapped_arr, 1));

  size_t level_arr_size = mipmapped_arr_desc.extent.width * mipmapped_arr_desc.extent.height *
                          mipmapped_arr_desc.extent.depth;

  REQUIRE(WriteAndValidateData<type>(level_arr, level_arr_size) == true);

  HIP_CHECK(hipFreeArray(level_arr));
  HIP_CHECK(hipFreeMipmappedArray(mipmapped_arr));
  HIP_CHECK(hipDestroyExternalMemory(ext_memory));
}

TEST_CASE("Unit_hipExternalMemoryGetMappedMipmappedArray_Vulkan_Array_Layered") {
  VulkanTest vkt(enable_validation);
  using type = uint8_t;
  constexpr uint32_t count = 16384;

  const auto vk_storage =
      vkt.CreateMappedStorage<type>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  if (vk_storage.memory == nullptr) {
    return;
  }

  const auto ext_mem_desc = vkt.BuildMemoryDescriptor(vk_storage.memory, vk_storage.size);
  hipExternalMemory_t ext_memory;
  HIP_CHECK(hipImportExternalMemory(&ext_memory, &ext_mem_desc));

  hipExternalMemoryMipmappedArrayDesc mipmapped_arr_desc = {};
  mipmapped_arr_desc.extent = {};
  mipmapped_arr_desc.extent.width = GENERATE(32, 128, 256);
  mipmapped_arr_desc.extent.height = GENERATE(0, 4);
  mipmapped_arr_desc.extent.depth = 16;
  mipmapped_arr_desc.flags = hipArrayLayered;
  mipmapped_arr_desc.formatDesc = hipCreateChannelDesc<type>();
  mipmapped_arr_desc.numLevels = GENERATE(1, 2, 4);
  mipmapped_arr_desc.offset = 0;

  hipMipmappedArray_t mipmapped_arr = nullptr;
  HIP_CHECK(
      hipExternalMemoryGetMappedMipmappedArray(&mipmapped_arr, ext_memory, &mipmapped_arr_desc));

  HIP_CHECK(hipFreeMipmappedArray(mipmapped_arr));
  HIP_CHECK(hipDestroyExternalMemory(ext_memory));
}

TEST_CASE("Unit_hipExternalMemoryGetMappedMipmappedArray_Vulkan_Array_Cubemap") {
  VulkanTest vkt(enable_validation);
  using type = uint8_t;
  //  cubemap HIP array is allocated if all three extents are non-zero and the hipArrayCubemap
  //  flag is set. Width must be equal to height, and depth must be six
  constexpr uint32_t cube_size = 32;
  constexpr uint32_t depth = 6;
  constexpr uint32_t ext_mem_size = cube_size * cube_size * depth;

  const auto vk_storage =
      vkt.CreateMappedStorage<type>(ext_mem_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  if (vk_storage.memory == nullptr) {
    return;
  }

  const auto ext_mem_desc = vkt.BuildMemoryDescriptor(vk_storage.memory, vk_storage.size);
  hipExternalMemory_t ext_memory;
  HIP_CHECK(hipImportExternalMemory(&ext_memory, &ext_mem_desc));

  hipExternalMemoryMipmappedArrayDesc mipmapped_arr_desc = {};
  mipmapped_arr_desc.extent = {};
  mipmapped_arr_desc.extent.width = cube_size;
  mipmapped_arr_desc.extent.height = cube_size;
  mipmapped_arr_desc.extent.depth = depth;
  mipmapped_arr_desc.flags = hipArrayCubemap;
  mipmapped_arr_desc.formatDesc = hipCreateChannelDesc<type>();
  mipmapped_arr_desc.numLevels = GENERATE(1, 2, 4);
  mipmapped_arr_desc.offset = 0;

  hipMipmappedArray_t mipmapped_arr = nullptr;
  HIP_CHECK(
      hipExternalMemoryGetMappedMipmappedArray(&mipmapped_arr, ext_memory, &mipmapped_arr_desc));

  HIP_CHECK(hipFreeMipmappedArray(mipmapped_arr));
  HIP_CHECK(hipDestroyExternalMemory(ext_memory));
}

TEST_CASE("Unit_hipExternalMemoryGetMappedMipmappedArray_Vulkan_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

  VulkanTest vkt(enable_validation);
  using type = uint8_t;
  constexpr uint32_t count = 256;

  const auto vk_storage =
      vkt.CreateMappedStorage<type>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  if (vk_storage.memory == nullptr) {
    return;
  }

  const auto ext_mem_desc = vkt.BuildMemoryDescriptor(vk_storage.memory, vk_storage.size);
  hipExternalMemory_t ext_memory;
  HIP_CHECK(hipImportExternalMemory(&ext_memory, &ext_mem_desc));

  hipExternalMemoryMipmappedArrayDesc mipmapped_arr_desc = {};
  mipmapped_arr_desc.extent = {.width = count, .height = 0, .depth = 0};
  mipmapped_arr_desc.flags = hipArrayDefault;
  mipmapped_arr_desc.formatDesc = hipCreateChannelDesc<type>();
  mipmapped_arr_desc.numLevels = 2;
  mipmapped_arr_desc.offset = 0;

  hipMipmappedArray_t mipmapped_arr = nullptr;

  SECTION("Nullptr_Array") {
    HIP_CHECK_ERROR(
        hipExternalMemoryGetMappedMipmappedArray(nullptr, ext_memory, &mipmapped_arr_desc),
        hipErrorInvalidValue);
  }

  SECTION("Nullptr_ExternalMemory") {
    HIP_CHECK_ERROR(
        hipExternalMemoryGetMappedMipmappedArray(&mipmapped_arr, nullptr, &mipmapped_arr_desc),
        hipErrorInvalidValue);
  }

  SECTION("Nullptr_ArrayDescription") {
    HIP_CHECK_ERROR(hipExternalMemoryGetMappedMipmappedArray(&mipmapped_arr, ext_memory, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid_Offset") {
    mipmapped_arr_desc.offset = 1 * sizeof(type);
    HIP_CHECK_ERROR(
        hipExternalMemoryGetMappedMipmappedArray(&mipmapped_arr, ext_memory, &mipmapped_arr_desc),
        hipErrorInvalidValue);
  }

  HIP_CHECK(hipDestroyExternalMemory(ext_memory));
}

/**
 * Test Description
 * ------------------------
 *    - Test hipExternalMemoryGetMappedMipmappedArray while stream is capturing.
 * Test source
 * ------------------------
 *    - unit/vulkan_interop/hipExternalMemoryGetMappedMipmappedArray.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipExternalMemoryGetMappedMipmappedArray_Vulkan_Capture") {
  VulkanTest vkt(enable_validation);
  using type = uint8_t;
  //  cubemap HIP array is allocated if all three extents are non-zero and the hipArrayCubemap
  //  flag is set. Width must be equal to height, and depth must be six
  constexpr uint32_t cube_size = 32;
  constexpr uint32_t depth = 6;
  constexpr uint32_t ext_mem_size = cube_size * cube_size * depth;

  const auto vk_storage =
      vkt.CreateMappedStorage<type>(ext_mem_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  if (vk_storage.memory == nullptr) {
    return;
  }

  const auto ext_mem_desc = vkt.BuildMemoryDescriptor(vk_storage.memory, vk_storage.size);
  hipExternalMemory_t ext_memory;
  HIP_CHECK(hipImportExternalMemory(&ext_memory, &ext_mem_desc));

  hipExternalMemoryMipmappedArrayDesc mipmapped_arr_desc = {};
  mipmapped_arr_desc.extent = {};
  mipmapped_arr_desc.extent.width = cube_size;
  mipmapped_arr_desc.extent.height = cube_size;
  mipmapped_arr_desc.extent.depth = depth;
  mipmapped_arr_desc.flags = hipArrayCubemap;
  mipmapped_arr_desc.formatDesc = hipCreateChannelDesc<type>();
  mipmapped_arr_desc.numLevels = GENERATE(1, 2, 4);
  mipmapped_arr_desc.offset = 0;
  hipMipmappedArray_t mipmapped_arr = nullptr;

  hipError_t memcpy_err = hipSuccess;
  BEGIN_CAPTURE_SYNC(memcpy_err, true);
  HIP_CHECK_ERROR(
      hipExternalMemoryGetMappedMipmappedArray(&mipmapped_arr, ext_memory, &mipmapped_arr_desc),
                                               memcpy_err);
  END_CAPTURE_SYNC(memcpy_err);

  HIP_CHECK(hipFreeMipmappedArray(mipmapped_arr));
  HIP_CHECK(hipDestroyExternalMemory(ext_memory));
}