/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Cross-side Vulkan-image ↔ HIP mipmapped-array data verification tests.
 *
 * Architecture:
 *   - Create a real VkImage (DEVICE_LOCAL, externally exportable, N mip levels).
 *   - One VkImageView per mip level.
 *   - Export the image memory handle; import into HIP via hipImportExternalMemory.
 *   - Staging VkBuffer (HOST_VISIBLE) per mip level for CPU access.
 *   - Tests write on one side, read and verify on the other, for every mip level.
 *
 * VulkanWrite→HIPRead per level:
 *   CPU fills staging → CopyBufferToImage(mip i) → hipGetMipmappedArrayLevel(i)
 *                     → hipMemcpyFromArray → verify
 *
 * HIPWrite→VulkanRead per level:
 *   hipMemcpyToArray(mip i) → hipDeviceSynchronize → CopyImageToBuffer(mip i) → verify staging
 */

#include <hip_test_common.hh>
#include "vulkan_test.hh"

#include <vector>
#include <cstring>

constexpr bool enable_validation = false;

static constexpr uint32_t kBaseWidth  = 64;
static constexpr uint32_t kBaseHeight = 64;
static constexpr uint32_t kNumLevels  = 3;   // 64×64, 32×32, 16×16
static constexpr VkFormat kVkFormat   = VK_FORMAT_R8G8B8A8_UINT;

__host__ __device__ static uint32_t pack_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  return (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
}

// Per-pixel expected value for a given (x, y, mip-level) — used by both host and device code.
// Distinct primes per channel ensure no two pixels at different (x,y,lvl) produce the same word.
__host__ __device__ static uint32_t PixelValue(uint32_t x, uint32_t y, uint32_t lvl) {
  return pack_rgba(static_cast<uint8_t>((x * 3 + lvl * 17) & 0xFF),
                   static_cast<uint8_t>((y * 5 + lvl * 13) & 0xFF),
                   static_cast<uint8_t>((x + y + lvl *  7) & 0xFF),
                   0xFF);
}

// ── Surface kernels (uint32_t packed RGBA, matching VK_FORMAT_R8G8B8A8_UINT) ─────────────────────

/* Write per-pixel coord-encoded values to a surface (mip level passed to reproduce PixelValue). */
__global__ void kernel_surf_write_coords(hipSurfaceObject_t surf, uint32_t lvl,
                                          uint32_t width, uint32_t height) {
#if !__HIP_NO_IMAGE_SUPPORT
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  surf2Dwrite(PixelValue(x, y, lvl), surf,
              static_cast<int>(x * sizeof(uint32_t)), static_cast<int>(y));
#endif
}

/* Read every texel of a surface into a flat device buffer. */
__global__ void kernel_surf_read(hipSurfaceObject_t surf, uint32_t* out,
                                 uint32_t width, uint32_t height) {
#if !__HIP_NO_IMAGE_SUPPORT
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  uint32_t px;
  surf2Dread(&px, surf, static_cast<int>(x * sizeof(uint32_t)), static_cast<int>(y));
  out[y * width + x] = px;
#endif
}

// ── Surface helper ────────────────────────────────────────────────────────────────────────────────

static hipSurfaceObject_t MakeSurface(hipArray_t level_array) {
  hipResourceDesc res = {};
  res.resType         = hipResourceTypeArray;
  res.res.array.array = level_array;
  hipSurfaceObject_t surf = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surf, &res));
  return surf;
}

static dim3 MipGrid(uint32_t w, uint32_t h) {
  constexpr uint32_t kBlock = 16;
  return dim3((w + kBlock - 1) / kBlock, (h + kBlock - 1) / kBlock);
}

// ── Per-mip-level metadata ────────────────────────────────────────────────────────────────────

static uint32_t MipWidth(uint32_t level)  { return std::max(1u, kBaseWidth  >> level); }
static uint32_t MipHeight(uint32_t level) { return std::max(1u, kBaseHeight >> level); }
static uint32_t MipPixels(uint32_t level) { return MipWidth(level) * MipHeight(level); }

// ── VulkanMipmapImage: owns VkImage + per-level views + staging buffers ───────────────────────

struct VulkanMipmapImage {
  VkDevice            device       = VK_NULL_HANDLE;
  VkImage             image        = VK_NULL_HANDLE;
  VkDeviceMemory      image_memory = VK_NULL_HANDLE;
  VkDeviceSize        image_size   = 0;
  std::vector<VkImageView> level_views;          // one per mip level

  // Per-level staging buffers (HOST_VISIBLE, one per mip level)
  struct StagingLevel {
    VkBuffer        buf  = VK_NULL_HANDLE;
    VkDeviceMemory  mem  = VK_NULL_HANDLE;
    uint32_t*       host = nullptr;
  };
  std::vector<StagingLevel> staging;

  hipExternalMemory_t    ext_mem   = nullptr;
  hipMipmappedArray_t    mip_array = nullptr;

  bool valid() const { return image != VK_NULL_HANDLE && ext_mem != nullptr; }

  void destroy(VulkanTest& vkt) {
    if (mip_array) { HIP_CHECK(hipFreeMipmappedArray(mip_array)); mip_array = nullptr; }
    if (ext_mem)   { HIP_CHECK(hipDestroyExternalMemory(ext_mem)); ext_mem = nullptr; }

    for (auto& s : staging) {
      if (s.host)  vkUnmapMemory(device, s.mem);
      if (s.buf)   vkDestroyBuffer(device, s.buf, nullptr);
      if (s.mem)   vkFreeMemory(device, s.mem, nullptr);
    }
    staging.clear();

    for (auto v : level_views) {
      if (v != VK_NULL_HANDLE) vkDestroyImageView(device, v, nullptr);
    }
    level_views.clear();

    if (image)        { vkDestroyImage(device, image, nullptr); image = VK_NULL_HANDLE; }
    if (image_memory) { vkFreeMemory(device, image_memory, nullptr); image_memory = VK_NULL_HANDLE; }
  }
};

// ── Factory ───────────────────────────────────────────────────────────────────────────────────

static VulkanMipmapImage CreateVulkanMipmapImage(VulkanTest& vkt) {
  VulkanMipmapImage result;
  result.device = vkt.GetDevice();

  // Create VkImage with kNumLevels mip levels, DEVICE_LOCAL + externally exportable.
  vkt.CreateImage(kBaseWidth, kBaseHeight, kNumLevels, kVkFormat,
                  result.image, result.image_memory, result.image_size, /*external=*/true);

  // Create one VkImageView per mip level.
  result.level_views.resize(kNumLevels);
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    result.level_views[lvl] = vkt.CreateImageView(result.image, kVkFormat, lvl, 1);
    REQUIRE(result.level_views[lvl] != VK_NULL_HANDLE);
  }

  // Create per-level HOST_VISIBLE staging buffers.
  result.staging.resize(kNumLevels);
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const VkDeviceSize bytes = MipPixels(lvl) * sizeof(uint32_t);
    vkt.CreateBuffer(bytes,
                     static_cast<VkBufferUsageFlagBits>(VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT),
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     result.staging[lvl].buf, result.staging[lvl].mem, /*external=*/false);
    VK_CHECK_RESULT(vkMapMemory(vkt.GetDevice(), result.staging[lvl].mem, 0, bytes, 0,
                                 reinterpret_cast<void**>(&result.staging[lvl].host)));
  }

  // Import the exported Vulkan device memory via the standard HIP API.
  auto mem_desc = vkt.BuildMemoryDescriptor(result.image_memory,
                                             static_cast<uint32_t>(result.image_size));
  HIP_CHECK(hipImportExternalMemory(&result.ext_mem, &mem_desc));

  // Map as a mipmapped array using the standard HIP API (works for both Vulkan and D3D12).
  hipExternalMemoryMipmappedArrayDesc arr_desc = {};
  arr_desc.extent      = {kBaseWidth, kBaseHeight, 0};
  arr_desc.formatDesc  = hipCreateChannelDesc<unsigned int>();
  arr_desc.flags       = hipArraySurfaceLoadStore;
  arr_desc.numLevels   = kNumLevels;
  arr_desc.offset      = 0;
  HIP_CHECK(hipExternalMemoryGetMappedMipmappedArray(
      &result.mip_array, result.ext_mem, &arr_desc));

  return result;
}

// ── Tests ─────────────────────────────────────────────────────────────────────────────────────

// Vulkan writes all mip levels first (per-pixel coord pattern), then HIP reads and verifies
// the whole image via hipMemcpyFromArray.
HIP_TEST_CASE(Unit_hipVulkanImageDataVerification_Positive_VulkanWrite_HIPRead) {
  CHECK_IMAGE_SUPPORT
  VulkanTest vkt(enable_validation);
  auto img = CreateVulkanMipmapImage(vkt);
  REQUIRE(img.valid());

  // Write phase: Vulkan fills every pixel of every level via staging.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t W = MipWidth(lvl), H = MipHeight(lvl);
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
        img.staging[lvl].host[y * W + x] = PixelValue(x, y, lvl);
    vkt.CopyBufferToImage(img.staging[lvl].buf, img.image, lvl, W, H);
  }

  // Verify phase: HIP reads every pixel of every level and compares.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t W = MipWidth(lvl), H = MipHeight(lvl), N = MipPixels(lvl);
    hipArray_t hip_level = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&hip_level, img.mip_array, lvl));

    std::vector<uint32_t> hip_pixels(N, 0);
    HIP_CHECK(hipMemcpyFromArray(hip_pixels.data(), hip_level, 0, 0,
                                  N * sizeof(uint32_t), hipMemcpyDeviceToHost));
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
        REQUIRE(hip_pixels[y * W + x] == PixelValue(x, y, lvl));
  }

  img.destroy(vkt);
}

// HIP writes all mip levels first (per-pixel coord pattern via hipMemcpyToArray), then Vulkan
// reads and verifies the whole image via staging buffers.
HIP_TEST_CASE(Unit_hipVulkanImageDataVerification_Positive_HIPWrite_VulkanRead) {
  CHECK_IMAGE_SUPPORT
  VulkanTest vkt(enable_validation);
  auto img = CreateVulkanMipmapImage(vkt);
  REQUIRE(img.valid());

  // Write phase: HIP copies per-pixel data into every level.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t W = MipWidth(lvl), H = MipHeight(lvl), N = MipPixels(lvl);
    hipArray_t hip_level = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&hip_level, img.mip_array, lvl));

    std::vector<uint32_t> host_in(N);
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
        host_in[y * W + x] = PixelValue(x, y, lvl);
    HIP_CHECK(hipMemcpyToArray(hip_level, 0, 0, host_in.data(),
                                N * sizeof(uint32_t), hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Verify phase: Vulkan reads every pixel of every level via staging.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t W = MipWidth(lvl), H = MipHeight(lvl);
    vkt.CopyImageToBuffer(img.image, lvl, W, H, img.staging[lvl].buf);
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
        REQUIRE(img.staging[lvl].host[y * W + x] == PixelValue(x, y, lvl));
  }

  img.destroy(vkt);
}

// Round-trip: Vulkan writes per-pixel data into all levels, HIP reads the whole image and
// inverts all RGB channels across all levels, writes back, then Vulkan reads the whole image
// and verifies every pixel of every level was inverted correctly.
HIP_TEST_CASE(Unit_hipVulkanImageDataVerification_Positive_RoundTrip) {
  CHECK_IMAGE_SUPPORT
  VulkanTest vkt(enable_validation);
  auto img = CreateVulkanMipmapImage(vkt);
  REQUIRE(img.valid());

  // Write phase: Vulkan fills all levels with per-pixel coord patterns.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t W = MipWidth(lvl), H = MipHeight(lvl);
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
        img.staging[lvl].host[y * W + x] = PixelValue(x, y, lvl);
    vkt.CopyBufferToImage(img.staging[lvl].buf, img.image, lvl, W, H);
  }

  // HIP modify phase: read all levels, invert RGB of every pixel, write all levels back.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t N = MipPixels(lvl);
    hipArray_t hip_level = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&hip_level, img.mip_array, lvl));

    std::vector<uint32_t> tmp(N);
    HIP_CHECK(hipMemcpyFromArray(tmp.data(), hip_level, 0, 0,
                                  N * sizeof(uint32_t), hipMemcpyDeviceToHost));
    for (auto& px : tmp) {
      const uint8_t r = 255u - (px        & 0xFFu);
      const uint8_t g = 255u - ((px >>  8) & 0xFFu);
      const uint8_t b = 255u - ((px >> 16) & 0xFFu);
      const uint8_t a =         (px >> 24) & 0xFFu;
      px = pack_rgba(r, g, b, a);
    }
    HIP_CHECK(hipMemcpyToArray(hip_level, 0, 0, tmp.data(),
                                N * sizeof(uint32_t), hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Verify phase: Vulkan reads the whole image and checks every pixel is RGB-inverted.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t W = MipWidth(lvl), H = MipHeight(lvl);
    vkt.CopyImageToBuffer(img.image, lvl, W, H, img.staging[lvl].buf);
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x) {
        const uint32_t orig = PixelValue(x, y, lvl);
        const uint32_t expected = pack_rgba(
            static_cast<uint8_t>(255u - (orig        & 0xFFu)),
            static_cast<uint8_t>(255u - ((orig >>  8) & 0xFFu)),
            static_cast<uint8_t>(255u - ((orig >> 16) & 0xFFu)),
            static_cast<uint8_t>(          (orig >> 24) & 0xFFu));
        REQUIRE(img.staging[lvl].host[y * W + x] == expected);
      }
  }

  img.destroy(vkt);
}

// HIP surface kernels write per-pixel coord data into all mip levels first, then Vulkan reads
// the whole image and verifies every pixel.
HIP_TEST_CASE(Unit_hipVulkanImageDataVerification_Positive_SurfaceWrite_VulkanRead) {
  CHECK_IMAGE_SUPPORT
  VulkanTest vkt(enable_validation);
  auto img = CreateVulkanMipmapImage(vkt);
  REQUIRE(img.valid());

  constexpr dim3 kBlock(16, 16);

  // Write phase: GPU surface-writes per-pixel coord values into every level.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t W = MipWidth(lvl), H = MipHeight(lvl);
    hipArray_t hip_level = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&hip_level, img.mip_array, lvl));
    hipSurfaceObject_t surf = MakeSurface(hip_level);
    kernel_surf_write_coords<<<MipGrid(W, H), kBlock>>>(surf, lvl, W, H);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDestroySurfaceObject(surf));
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Verify phase: Vulkan reads the whole image and checks every pixel.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t W = MipWidth(lvl), H = MipHeight(lvl);
    vkt.CopyImageToBuffer(img.image, lvl, W, H, img.staging[lvl].buf);
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
        REQUIRE(img.staging[lvl].host[y * W + x] == PixelValue(x, y, lvl));
  }

  img.destroy(vkt);
}

// Vulkan writes per-pixel coord data into all mip levels first, then HIP surface kernels read
// the whole image and verify every pixel.
HIP_TEST_CASE(Unit_hipVulkanImageDataVerification_Positive_VulkanWrite_SurfaceRead) {
  CHECK_IMAGE_SUPPORT
  VulkanTest vkt(enable_validation);
  auto img = CreateVulkanMipmapImage(vkt);
  REQUIRE(img.valid());

  constexpr dim3 kBlock(16, 16);

  // Write phase: Vulkan fills all levels with per-pixel coord patterns via staging.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t W = MipWidth(lvl), H = MipHeight(lvl);
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
        img.staging[lvl].host[y * W + x] = PixelValue(x, y, lvl);
    vkt.CopyBufferToImage(img.staging[lvl].buf, img.image, lvl, W, H);
  }

  // Verify phase: GPU surface-reads every pixel of every level and checks against PixelValue.
  for (uint32_t lvl = 0; lvl < kNumLevels; ++lvl) {
    const uint32_t W = MipWidth(lvl), H = MipHeight(lvl), N = MipPixels(lvl);
    hipArray_t hip_level = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&hip_level, img.mip_array, lvl));
    hipSurfaceObject_t surf = MakeSurface(hip_level);

    uint32_t* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    kernel_surf_read<<<MipGrid(W, H), kBlock>>>(surf, d_out, W, H);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipDestroySurfaceObject(surf));

    std::vector<uint32_t> h_out(N);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_out));

    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x)
        REQUIRE(h_out[y * W + x] == PixelValue(x, y, lvl));
  }

  img.destroy(vkt);
}
