/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hip_test_common.hh>

#include "vulkan_test.hh"

constexpr bool enable_validation = false;

static constexpr uint32_t kWidth = 64;
static constexpr uint32_t kHeight = 64;
static constexpr uint32_t kNumPixels = kWidth * kHeight;

// ── Device kernels ────────────────────────────────────────────────────────────

__global__ static void kernel_fill_surface(hipSurfaceObject_t surf, uint32_t width,
                                           uint32_t height, uchar4 color) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  surf2Dwrite(color, surf, x * sizeof(uchar4), y);
}

__global__ static void kernel_read_surface(hipSurfaceObject_t surf, uint32_t width,
                                           uint32_t height, uchar4* out) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  surf2Dread(&out[y * width + x], surf, x * sizeof(uchar4), y);
}

__global__ static void kernel_write_unique(hipSurfaceObject_t surf, uint32_t width,
                                           uint32_t height) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  uchar4 px{static_cast<uint8_t>(x & 0xFFu), static_cast<uint8_t>(y & 0xFFu),
             static_cast<uint8_t>((x + y) & 0xFFu), 0xFFu};
  surf2Dwrite(px, surf, x * sizeof(uchar4), y);
}

__global__ static void kernel_invert_surface(hipSurfaceObject_t surf, uint32_t width,
                                             uint32_t height) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  uchar4 px;
  surf2Dread(&px, surf, x * sizeof(uchar4), y);
  px.x = 255u - px.x;
  px.y = 255u - px.y;
  px.z = 255u - px.z;
  surf2Dwrite(px, surf, x * sizeof(uchar4), y);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static constexpr dim3 kBlock{16, 16, 1};

static dim3 grid2d(uint32_t w, uint32_t h) {
  return dim3((w + 15u) / 16u, (h + 15u) / 16u, 1u);
}

static hipSurfaceObject_t make_surface(hipArray_t array) {
  hipResourceDesc res_desc{};
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = array;
  hipSurfaceObject_t surf = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surf, &res_desc));
  return surf;
}

struct VkImageMem {
  hipExternalMemory_t ext_mem = nullptr;
  hipMipmappedArray_t mip_array = nullptr;

  void destroy() {
    if (mip_array) {
      HIP_CHECK(hipFreeMipmappedArray(mip_array));
      mip_array = nullptr;
    }
    if (ext_mem) {
      HIP_CHECK(hipDestroyExternalMemory(ext_mem));
      ext_mem = nullptr;
    }
  }
};

static VkImageMem ImportVulkanImage(VulkanTest& vkt, uint32_t width, uint32_t height,
                                    uint32_t num_levels = 1) {
  const auto vk_buf = vkt.CreateMappedStorage<uchar4>(width * height,
                                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  if (vk_buf.memory == nullptr) return {};

  VkImageMem result;
  const auto mem_desc = vkt.BuildMemoryDescriptor(vk_buf.memory, vk_buf.size);
  HIP_CHECK(hipImportExternalMemory(&result.ext_mem, &mem_desc));

  hipExternalMemoryMipmappedArrayDesc arr_desc{};
  arr_desc.extent = {width, height, 0};
  arr_desc.formatDesc = hipCreateChannelDesc<uchar4>();
  arr_desc.flags = hipArrayDefault;
  arr_desc.numLevels = num_levels;
  arr_desc.offset = 0;
  HIP_CHECK(hipExternalMemoryGetMappedMipmappedArray(&result.mip_array, result.ext_mem, &arr_desc));
  return result;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// Write a solid color via surf2Dwrite, read back via hipMemcpyFromArray and verify.
HIP_TEST_CASE(Unit_hipVulkanImageDataVerification_Positive_HIPWrite_HIPRead_SolidColor) {
  CHECK_IMAGE_SUPPORT
  VulkanTest vkt(enable_validation);
  auto img = ImportVulkanImage(vkt, kWidth, kHeight);
  REQUIRE(img.ext_mem != nullptr);

  hipArray_t array = nullptr;
  HIP_CHECK(hipGetMipmappedArrayLevel(&array, img.mip_array, 0));

  const uchar4 color{128, 64, 32, 255};
  auto surf = make_surface(array);
  hipLaunchKernelGGL(kernel_fill_surface, grid2d(kWidth, kHeight), kBlock, 0, nullptr, surf,
                     kWidth, kHeight, color);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uchar4> host(kNumPixels);
  HIP_CHECK(hipMemcpyFromArray(host.data(), array, 0, 0, kNumPixels * sizeof(uchar4),
                               hipMemcpyDeviceToHost));
  for (uint32_t i = 0; i < kNumPixels; ++i) {
    REQUIRE(host[i].x == color.x);
    REQUIRE(host[i].y == color.y);
    REQUIRE(host[i].z == color.z);
    REQUIRE(host[i].w == color.w);
  }

  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipFreeArray(array));
  img.destroy();
}

// Write per-pixel unique values via surf2Dwrite, read back and verify each pixel.
HIP_TEST_CASE(Unit_hipVulkanImageDataVerification_Positive_HIPWrite_HIPRead_UniquePerPixel) {
  CHECK_IMAGE_SUPPORT
  VulkanTest vkt(enable_validation);
  auto img = ImportVulkanImage(vkt, kWidth, kHeight);
  REQUIRE(img.ext_mem != nullptr);

  hipArray_t array = nullptr;
  HIP_CHECK(hipGetMipmappedArrayLevel(&array, img.mip_array, 0));

  auto surf = make_surface(array);
  hipLaunchKernelGGL(kernel_write_unique, grid2d(kWidth, kHeight), kBlock, 0, nullptr, surf,
                     kWidth, kHeight);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uchar4> host(kNumPixels);
  HIP_CHECK(hipMemcpyFromArray(host.data(), array, 0, 0, kNumPixels * sizeof(uchar4),
                               hipMemcpyDeviceToHost));
  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      const auto& px = host[y * kWidth + x];
      REQUIRE(px.x == static_cast<uint8_t>(x & 0xFFu));
      REQUIRE(px.y == static_cast<uint8_t>(y & 0xFFu));
      REQUIRE(px.z == static_cast<uint8_t>((x + y) & 0xFFu));
      REQUIRE(px.w == 0xFFu);
    }
  }

  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipFreeArray(array));
  img.destroy();
}

// Write via hipMemcpyToArray, read back via surf2Dread kernel and verify.
HIP_TEST_CASE(Unit_hipVulkanImageDataVerification_Positive_MemcpyWrite_SurfRead) {
  CHECK_IMAGE_SUPPORT
  VulkanTest vkt(enable_validation);
  auto img = ImportVulkanImage(vkt, kWidth, kHeight);
  REQUIRE(img.ext_mem != nullptr);

  hipArray_t array = nullptr;
  HIP_CHECK(hipGetMipmappedArrayLevel(&array, img.mip_array, 0));

  const uchar4 fill{200, 100, 50, 255};
  std::vector<uchar4> host_in(kNumPixels, fill);
  HIP_CHECK(hipMemcpyToArray(array, 0, 0, host_in.data(), kNumPixels * sizeof(uchar4),
                             hipMemcpyHostToDevice));

  uchar4* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, kNumPixels * sizeof(uchar4)));
  auto surf = make_surface(array);
  hipLaunchKernelGGL(kernel_read_surface, grid2d(kWidth, kHeight), kBlock, 0, nullptr, surf,
                     kWidth, kHeight, d_out);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uchar4> host_out(kNumPixels);
  HIP_CHECK(hipMemcpy(host_out.data(), d_out, kNumPixels * sizeof(uchar4), hipMemcpyDeviceToHost));
  for (uint32_t i = 0; i < kNumPixels; ++i) {
    REQUIRE(host_out[i].x == fill.x);
    REQUIRE(host_out[i].y == fill.y);
    REQUIRE(host_out[i].z == fill.z);
    REQUIRE(host_out[i].w == fill.w);
  }

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipFreeArray(array));
  img.destroy();
}

// Write solid color, invert RGB channels via kernel, read back and verify.
HIP_TEST_CASE(Unit_hipVulkanImageDataVerification_Positive_RoundTrip_Invert) {
  CHECK_IMAGE_SUPPORT
  VulkanTest vkt(enable_validation);
  auto img = ImportVulkanImage(vkt, kWidth, kHeight);
  REQUIRE(img.ext_mem != nullptr);

  hipArray_t array = nullptr;
  HIP_CHECK(hipGetMipmappedArrayLevel(&array, img.mip_array, 0));

  const uchar4 color{100, 150, 200, 255};
  std::vector<uchar4> host_in(kNumPixels, color);
  HIP_CHECK(hipMemcpyToArray(array, 0, 0, host_in.data(), kNumPixels * sizeof(uchar4),
                             hipMemcpyHostToDevice));

  auto surf = make_surface(array);
  hipLaunchKernelGGL(kernel_invert_surface, grid2d(kWidth, kHeight), kBlock, 0, nullptr, surf,
                     kWidth, kHeight);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uchar4> host_out(kNumPixels);
  HIP_CHECK(hipMemcpyFromArray(host_out.data(), array, 0, 0, kNumPixels * sizeof(uchar4),
                               hipMemcpyDeviceToHost));
  for (uint32_t i = 0; i < kNumPixels; ++i) {
    REQUIRE(host_out[i].x == static_cast<uint8_t>(255u - color.x));
    REQUIRE(host_out[i].y == static_cast<uint8_t>(255u - color.y));
    REQUIRE(host_out[i].z == static_cast<uint8_t>(255u - color.z));
    REQUIRE(host_out[i].w == color.w);
  }

  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipFreeArray(array));
  img.destroy();
}

// Fill each mip level with a distinct solid color and verify per-level.
HIP_TEST_CASE(Unit_hipVulkanImageDataVerification_Positive_Mipmap_MultiLevel) {
  CHECK_IMAGE_SUPPORT
  VulkanTest vkt(enable_validation);

  constexpr uint32_t kNumLevels = 3;
  // Allocate enough Vulkan memory to cover all mip levels (64*64 + 32*32 + 16*16) * sizeof(uchar4)
  constexpr uint32_t kTotalCount =
      kWidth * kHeight + (kWidth / 2) * (kHeight / 2) + (kWidth / 4) * (kHeight / 4);
  const auto vk_buf =
      vkt.CreateMappedStorage<uchar4>(kTotalCount, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  REQUIRE(vk_buf.memory != nullptr);

  hipExternalMemory_t ext_mem = nullptr;
  const auto mem_desc = vkt.BuildMemoryDescriptor(vk_buf.memory, vk_buf.size);
  HIP_CHECK(hipImportExternalMemory(&ext_mem, &mem_desc));

  hipExternalMemoryMipmappedArrayDesc arr_desc{};
  arr_desc.extent = {kWidth, kHeight, 0};
  arr_desc.formatDesc = hipCreateChannelDesc<uchar4>();
  arr_desc.flags = hipArrayDefault;
  arr_desc.numLevels = kNumLevels;
  arr_desc.offset = 0;

  hipMipmappedArray_t mip_array = nullptr;
  HIP_CHECK(hipExternalMemoryGetMappedMipmappedArray(&mip_array, ext_mem, &arr_desc));

  const uchar4 level_colors[kNumLevels] = {{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}};

  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t lw = kWidth >> lvl;
    const uint32_t lh = kHeight >> lvl;
    const uint32_t n = lw * lh;

    hipArray_t array = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&array, mip_array, lvl));

    auto surf = make_surface(array);
    hipLaunchKernelGGL(kernel_fill_surface, grid2d(lw, lh), kBlock, 0, nullptr, surf, lw, lh,
                       level_colors[lvl]);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uchar4> host(n);
    HIP_CHECK(
        hipMemcpyFromArray(host.data(), array, 0, 0, n * sizeof(uchar4), hipMemcpyDeviceToHost));
    for (uint32_t i = 0; i < n; ++i) {
      REQUIRE(host[i].x == level_colors[lvl].x);
      REQUIRE(host[i].y == level_colors[lvl].y);
      REQUIRE(host[i].z == level_colors[lvl].z);
      REQUIRE(host[i].w == level_colors[lvl].w);
    }

    HIP_CHECK(hipDestroySurfaceObject(surf));
    HIP_CHECK(hipFreeArray(array));
  }

  HIP_CHECK(hipFreeMipmappedArray(mip_array));
  HIP_CHECK(hipDestroyExternalMemory(ext_mem));
}
