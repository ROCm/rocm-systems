/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#if (!HT_NVIDIA) || (CUDA_VERSION < CUDA_12000)
/**
 * @addtogroup hipBindTextureToMipmappedArray hipBindTextureToMipmappedArray
 * @{
 * @ingroup TextureTest
 * `hipBindTextureToMipmappedArray(const textureReference* tex,
 * hipMipmappedArray_const_t mipmappedArray, const hipChannelFormatDesc* desc)` -
 * Binds a mipmapped array to a texture.
 */
texture<float, 2, hipReadModeElementType> texRef;

#if !__HIP_NO_IMAGE_SUPPORT
struct MipmappedTextureGuard {
  hipMipmappedArray_t mip_array = nullptr;
  float* output = nullptr;
  bool bound = false;

  MipmappedTextureGuard() = default;
  MipmappedTextureGuard(const MipmappedTextureGuard&) = delete;
  MipmappedTextureGuard& operator=(const MipmappedTextureGuard&) = delete;

  ~MipmappedTextureGuard() {
    if (output != nullptr) {
      static_cast<void>(hipFree(output));
    }
    if (bound) {
      static_cast<void>(hipUnbindTexture(texRef));
    }
    if (mip_array != nullptr) {
      static_cast<void>(hipFreeMipmappedArray(mip_array));
    }
  }
};

__global__ void tex2DKernel(float* outputData, int width, int height, float level) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float u = x / (float)width;
  float v = y / (float)height;
  outputData[y * width + x] = tex2DLod<float>(texRef, u, v, level);
}

static void runMipMapTest(unsigned int width, unsigned int height, unsigned int mipmap_level) {
  INFO("Width: " << width << "Height: " << height << "mip: " << mipmap_level);

  // Create new width & height to be tested
  unsigned int orig_width = width;
  unsigned int orig_height = height;
  unsigned int i, j;
  width /= pow(2, mipmap_level);
  height /= pow(2, mipmap_level);
  unsigned int size = width * height * sizeof(float);

  std::vector<float> hData(width * height);
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      hData[i * width + j] = i * width + j;
    }
  }

  // Allocate memory for Mipmapped array and set data to mipmap_level
  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();

  MipmappedTextureGuard resources;
  hipError_t status = hipMallocMipmappedArray(&resources.mip_array, &channelDesc,
                                              make_hipExtent(orig_width, orig_height, 0),
                                              2 * mipmap_level, hipArrayDefault);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kMipmappedArraysUnsupported);
  }
  HIP_CHECK(status);

  // The level is borrowed from mip_array and released with its parent.
  hipArray_t level_array = nullptr;
  HIP_CHECK(hipGetMipmappedArrayLevel(&level_array, resources.mip_array, mipmap_level));
  HIP_CHECK(hipMemcpy2DToArray(level_array, 0, 0, hData.data(), width * sizeof(float),
                               width * sizeof(float), height, hipMemcpyHostToDevice));

  // Set texture parameters
  texRef.addressMode[0] = hipAddressModeWrap;
  texRef.addressMode[1] = hipAddressModeWrap;
  texRef.filterMode = hipFilterModePoint;
  texRef.mipmapFilterMode = hipFilterModePoint;
  texRef.minMipmapLevelClamp = 0;
  texRef.maxMipmapLevelClamp = mipmap_level;
  texRef.normalized = 1;

  // Bind the array to the texture
  resources.bound = true;
  HIP_CHECK(hipBindTextureToMipmappedArray(&texRef, resources.mip_array, &channelDesc));

  // Allocate device memory for result
  HIP_CHECK(hipMalloc(&resources.output, size));
  REQUIRE(resources.output != nullptr);

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  hipLaunchKernelGGL(tex2DKernel, dim3(dimGrid), dim3(dimBlock), 0, 0, resources.output, width,
                     height, mipmap_level);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Allocate memory on host and copy result from device to host
  std::vector<float> hOutputData(width * height);
  HIP_CHECK(hipMemcpy(hOutputData.data(), resources.output, size, hipMemcpyDeviceToHost));

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      if (hData[i * width + j] != hOutputData[i * width + j]) {
        INFO("Difference found at [ " << i << j << " ]: " << hData[i * width + j]
                                      << hOutputData[i * width + j]);
        REQUIRE(false);
      }
    }
  }
  HIP_CHECK(hipUnbindTexture(texRef));
  resources.bound = false;
  HIP_CHECK(hipFree(resources.output));
  resources.output = nullptr;
  HIP_CHECK(hipFreeMipmappedArray(resources.mip_array));
  resources.mip_array = nullptr;
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Maps texture to the mipmapped array for different mipmapped array
 *    sizes and number of levels.
 * Test source
 * ------------------------
 *  - unit/texture/hipBindTextureToMipmappedArray.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipTextureMipmapRef2D_Positive_Check) {
  CHECK_IMAGE_SUPPORT

  // Height Width Vector
  std::vector<unsigned int> hw_vec = {2048, 1024, 512, 256, 64};
  std::vector<unsigned int> mip_vec = {8, 4, 2, 1};
#if !__HIP_NO_IMAGE_SUPPORT
  for (auto& hw : hw_vec) {
    for (auto& mip : mip_vec) {
      if ((hw / static_cast<int>(pow(2, (mip * 2)))) > 0) {
        runMipMapTest(hw, hw, mip);
      }
    }
  }
#else
  HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
#endif
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When texture reference is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When mipmapped array handle is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When channel descriptor is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/texture/hipBindTextureToMipmappedArray.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipTextureMipmapRef2D_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

#if !__HIP_NO_IMAGE_SUPPORT
  unsigned int width = 64;
  unsigned int height = 64;
  unsigned int mipmap_level = 1;

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();

  MipmappedTextureGuard resources;
  hipError_t status =
      hipMallocMipmappedArray(&resources.mip_array, &channelDesc, make_hipExtent(width, height, 0),
                              mipmap_level, hipArrayDefault);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kMipmappedArraysUnsupported);
  }
  HIP_CHECK(status);

  texRef.addressMode[0] = hipAddressModeWrap;
  texRef.addressMode[1] = hipAddressModeWrap;
  texRef.filterMode = hipFilterModePoint;
  texRef.mipmapFilterMode = hipFilterModePoint;
  texRef.minMipmapLevelClamp = 0;
  texRef.maxMipmapLevelClamp = 0;
  texRef.normalized = 1;
  // Establish valid binding before varying one argument in each section.
  resources.bound = true;
  HIP_CHECK(hipBindTextureToMipmappedArray(&texRef, resources.mip_array, &channelDesc));
  hipError_t ret;

  SECTION("textureReference is nullptr") {
    ret = hipBindTextureToMipmappedArray(nullptr, resources.mip_array, &channelDesc);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("MipmappedArray is nullptr") {
    hipError_t ret = hipBindTextureToMipmappedArray(&texRef, nullptr, &channelDesc);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("Channel descriptor is nullptr") {
    ret = hipBindTextureToMipmappedArray(&texRef, resources.mip_array, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  HIP_CHECK(hipUnbindTexture(texRef));
  resources.bound = false;
  HIP_CHECK(hipFreeMipmappedArray(resources.mip_array));
  resources.mip_array = nullptr;
#else
  HIP_SKIP_TEST(HipTest::SkipReason::kTextureImageUnsupported);
#endif
}
#endif

/**
 * End doxygen group TextureTest.
 * @}
 */
