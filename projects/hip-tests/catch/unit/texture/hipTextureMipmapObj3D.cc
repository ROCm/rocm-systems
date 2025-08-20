/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <hip_array_common.hh>
#include <hip_test_checkers.hh>
#include <hip_texture_helper.hh>
#include <algorithm>

#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"

// #define TEST_TEXTURE
static constexpr bool printLog = false;  // Print log for debugging

// Populate mipmap next level array
template <typename T, hipTextureReadMode readMode>
static __global__ void populateMipmapNextLevelArray(hipSurfaceObject_t surfOut,
                                                    hipTextureObject_t texIn, unsigned int width,
                                                    unsigned int height, unsigned int depth,
                                                    T* data = nullptr) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  float px = 1.0 / float(width);
  float py = 1.0 / float(height);
  float pz = 1.0 / float(depth);

  if (x < width && y < height && z < depth) {
    if constexpr (readMode == hipReadModeElementType) {
      T t = tex3D<T>(texIn, x * px, y * py, z * pz);
#ifdef TEST_TEXTURE
      printf("(%d/%u, %d/%u):(%f, %f)\n", x, width, y, height, t.x, t.y);
#endif
      surf3Dwrite<T>(t, surfOut, x * sizeof(T), y, z);
      if (data) data[z * width * height + y * width + x] = t;  // record it for later verification
    }
    if constexpr (readMode == hipReadModeNormalizedFloat) {
      float4 t = tex3D<float4>(texIn, x * px, y * py, z * pz);
      T tc = getTypeFromNormalizedFloat<T, float4>(t);
      surf3Dwrite<T>(tc, surfOut, x * sizeof(T), y, z);
      if (data) data[z * width * height + y * width + x] = tc;
#ifdef TEST_TEXTURE
      printf(
          "populateMipmapNextLevelArray(%d/%u, %d/%u, %d/%u->%d): t.x=%f, t.y=%f, t.z=%f, t.w=%f, "
          "tc=%d\n",
          x, width, y, height, z, depth, z * width * height + y * width + x, t.x, t.y, t.z, t.w,
          (int)tc);
#endif
    }
    // Users have freedom to use other methods to init level array
    // for example, use averge of surrounding pixes
  }
#endif
}

template <typename T>
static __global__ void getMipmap(hipTextureObject_t texMipmap, unsigned int width,
                                 unsigned int height, unsigned int depth, float offsetX,
                                 float offsetY, float offsetZ, float lod, T* data = nullptr) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  float px = 1.0 / float(width);
  float py = 1.0 / float(height);
  float pz = 1.0 / float(depth);

  if (x < width && y < height && z < depth) {
    T t = tex3DLod<T>(texMipmap, (x + offsetX) * px, (y + offsetY) * py, (z + offsetZ) * pz, lod);
    if (data) data[z * width * height + y * width + x] = t;
  }
#endif
}

template <typename T, hipTextureReadMode readMode = hipReadModeElementType,
          hipTextureFilterMode filterMode = hipFilterModePoint,
          hipTextureAddressMode addressMode = hipAddressModeClamp>
static void populateMipmaps(hipMipmappedArray_t mipmapArray, hipExtent size,
                            std::vector<mipmapLevelArray<T>>& mipmapData) {
  size_t width = size.width;
  size_t height = size.height;
  size_t depth = size.depth;
  unsigned int level = 0;

  while (width != 1 || height != 1 || depth != 1) {
    hipArray_t levelArray = nullptr, nextLevelArray = nullptr;
    HIP_CHECK(hipGetMipmappedArrayLevel(&levelArray, mipmapArray, level));
    HIP_CHECK(hipGetMipmappedArrayLevel(&nextLevelArray, mipmapArray, level + 1));

    hipExtent levelArraySize{0, 0, 0};
    HIP_CHECK(hipArrayGetInfo(nullptr, &levelArraySize, nullptr, levelArray));
    if (levelArraySize.width != width || levelArraySize.height != height ||
        levelArraySize.depth != depth) {
      fprintf(stderr, "Level %u: size (%zu, %zu, %zu) != Expected size (%zu, %zu, %zu)\n", level,
              levelArraySize.width, levelArraySize.height, levelArraySize.depth, width, height,
              depth);
      REQUIRE(false);
    }

    width = width >> 1 ? width >> 1 : 1;
    height = height >> 1 ? height >> 1 : 1;
    depth = depth >> 1 ? depth >> 1 : 1;

    hipExtent nextLevelArraySize{0, 0, 0};
    HIP_CHECK(hipArrayGetInfo(nullptr, &nextLevelArraySize, nullptr, nextLevelArray));
    if (nextLevelArraySize.width != width || nextLevelArraySize.height != height ||
        nextLevelArraySize.depth != depth) {
      fprintf(stderr, "Next level %u: size (%zu, %zu, %zu) != Expected size (%zu, %zu, %zu)\n",
              level + 1, nextLevelArraySize.width, nextLevelArraySize.height,
              nextLevelArraySize.depth, width, height, depth);
      REQUIRE(false);
    }

    hipTextureObject_t texIn;
    hipResourceDesc texRes;
    memset(&texRes, 0, sizeof(hipResourceDesc));
    texRes.resType = hipResourceTypeArray;
    texRes.res.array.array = levelArray;

    hipTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(hipTextureDesc));
    texDescr.normalizedCoords = 1;  // To populate next level array smoothly
    texDescr.filterMode = filterMode;
    texDescr.addressMode[0] = addressMode;
    texDescr.addressMode[1] = addressMode;
    texDescr.addressMode[2] = addressMode;
    texDescr.readMode = readMode;
    HIP_CHECK(hipCreateTextureObject(&texIn, &texRes, &texDescr, NULL));

    hipSurfaceObject_t surfOut;
    hipResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(hipResourceDesc));
    surfRes.resType = hipResourceTypeArray;
    surfRes.res.array.array = nextLevelArray;

    HIP_CHECK(hipCreateSurfaceObject(&surfOut, &surfRes));
    size_t size = width * height * depth * sizeof(T);
    mipmapLevelArray<T> data{nullptr, {width, height, depth}};
    HIP_CHECK(hipHostMalloc((void**)&data.data, size));
    memset(data.data, 0, size);

    dim3 blockSize(16, 16, 4);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y,
                  (depth + blockSize.z - 1) / blockSize.z);

    populateMipmapNextLevelArray<T, readMode>
        <<<gridSize, blockSize>>>(surfOut, texIn, width, height, depth, data.data);

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipDestroySurfaceObject(surfOut));
    HIP_CHECK(hipDestroyTextureObject(texIn));
    HIP_CHECK(hipFreeArray(levelArray));
    HIP_CHECK(hipFreeArray(nextLevelArray));
    mipmapData.push_back(data);  // For later verification
    level++;
  }
}

template <typename T, hipTextureFilterMode filterMode = hipFilterModePoint,
          hipTextureAddressMode addressMode = hipAddressModeClamp>
static void verifyMipmapLevel(hipTextureObject_t texMipmap, T* data, size_t width, size_t height,
                              size_t depth, float level, float offsetX, float offsetY,
                              float offsetZ) {
  T* hOutput = nullptr;
  size_t size = width * height * depth * sizeof(T);
  HIP_CHECK(hipHostMalloc((void**)&hOutput, size));
  memset(hOutput, 0, size);

  dim3 blockSize(16, 16, 4);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y,
                (depth + blockSize.z - 1) / blockSize.z);

  getMipmap<T><<<gridSize, blockSize>>>(texMipmap, width, height, depth, offsetX, offsetY, offsetZ,
                                        level, hOutput);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());

  for (unsigned int k = 0; k < depth; k++) {
    for (unsigned int j = 0; j < height; j++) {
      for (unsigned int i = 0; i < width; i++) {
        unsigned int index = k * width * height + j * width + i;
        T cpuExpected = getExpectedValue<T, addressMode, filterMode, false>(
            width, height, depth, i + offsetX, j + offsetY, k + offsetZ, data);
        T gpuOutput = hOutput[index];
        bool match = hipTextureSamplingVerify<T, filterMode, false>(gpuOutput, cpuExpected);
        if (!match) {
          WARN("Mismatch at (level " << level << ": " << i << ", " << j << ", " << k << "->"
                                     << (i + offsetX) << ", " << (j + offsetY) << ", "
                                     << (k + offsetZ) << ") GPU output : " << getString(gpuOutput)
                                     << " CPU expected: " << getString(cpuExpected) << ", data["
                                     << index << "]:" << getString(data[index]) << "\n");
          REQUIRE(false);
        } else if (printLog) {
          WARN("Matching at (level " << level << ": " << i << ", " << j << ", " << k << "->"
                                     << (i + offsetX) << ", " << (j + offsetY) << ", "
                                     << (k + offsetZ) << ") GPU output : " << getString(gpuOutput)
                                     << " CPU expected: " << getString(cpuExpected) << ", data["
                                     << index << "]:" << getString(data[index]) << "\n");
        }
      }
    }
  }
  HIP_CHECK(hipHostFree(hOutput));
}

template <typename T, hipTextureReadMode readMode = hipReadModeElementType,
          hipTextureFilterMode filterMode = hipFilterModePoint,
          hipTextureAddressMode addressMode = hipAddressModeClamp>
static void testMipmapTextureObj(size_t width, size_t height, size_t depth, float offsetX = 0.,
                                 float offsetY = 0., float offsetZ = 0.) {
  std::vector<mipmapLevelArray<T>> mipmapData;
  size_t size = width * height * depth * sizeof(T);
  mipmapLevelArray<T> data{nullptr, {width, height, depth}};
  HIP_CHECK(hipHostMalloc((void**)&data.data, size));
  memset(data.data, 0, size);
  for (int k = 0; k < depth; k++) {
    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; i++) {
        if (isFloat<T>() && filterMode == hipFilterModeLinear) {
          /*
           * For linear sampling of images, the GPU does not use IEEE floating point types, it
           * uses lower-precision sampling optimized formats. Also, those formats often change
           * between GPU generations. So counting on IEEE precision and accuracy when doing linear
           * sampling is mistaken. To workaround this issue, we can initialize float pixels on a
           * retively smoothy surface.
           */
          data.data[k * width * height + j * width + i] =
              T(float(i) * (float(i) - width + 1) * float(j) * (float(j) - height + 1) * float(k) *
                (float(k) - depth + 1));
        } else {
          initVal(data.data[k * width * height + j * width + i]);  // Randomize initial values
        }
      }
    }
  }
  mipmapData.push_back(data);  // record level 0 data for later verification

  // Get the max mipmap levels in terms of image size
  const unsigned int maxLevels = 1 + std::log2(std::max(width, std::max(height, depth)));

  // create mipmap array
  hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
  hipMipmappedArray_t mipmapArray = nullptr;
  hipExtent extent{width, height, depth};
  HIP_CHECK(hipMallocMipmappedArray(&mipmapArray, &desc, extent, maxLevels));

  // Initialize level 0
  hipArray_t levelArray;
  HIP_CHECK(hipGetMipmappedArrayLevel(&levelArray, mipmapArray, 0));
  hipMemcpy3DParms copyParams{};
  copyParams.srcPtr = make_hipPitchedPtr(data.data, width * sizeof(T), width, height);
  copyParams.dstArray = levelArray;
  copyParams.extent.width = width;
  copyParams.extent.height = height;
  copyParams.extent.depth = depth;
  copyParams.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipMemcpy3D(&copyParams));

  // Populate other mipmap levels based on level 0
  populateMipmaps<T, readMode, filterMode, addressMode>(mipmapArray, extent, mipmapData);

  if (maxLevels != mipmapData.size()) {
    fprintf(stderr, "maxLevels %u != mipmapData.size() %zu\n", maxLevels, mipmapData.size());
    REQUIRE(false);
  }
  hipResourceDesc resDescr;
  memset(&resDescr, 0, sizeof(hipResourceDesc));
  resDescr.resType = hipResourceTypeMipmappedArray;  // For mipmap texture
  resDescr.res.mipmap.mipmap = mipmapArray;

  hipTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(hipTextureDesc));

  texDescr.normalizedCoords = 1;  // normalizedCoords must be 1 for mipmap array
  texDescr.filterMode = filterMode;
  texDescr.mipmapFilterMode = filterMode;
  texDescr.addressMode[0] = addressMode;
  texDescr.addressMode[1] = addressMode;
  texDescr.addressMode[2] = addressMode;
  // Ignored in AMD Hw sampler SRD, but cuda need it
  texDescr.maxMipmapLevelClamp = float(maxLevels - 1);
  texDescr.readMode = readMode;

  hipTextureObject_t texMipmap = nullptr;
  HIP_CHECK(hipCreateTextureObject(&texMipmap, &resDescr, &texDescr, NULL));

  for (unsigned int level = 0; level < mipmapData.size(); level++) {
    mipmapLevelArray<T>& data = mipmapData.at(level);

    if constexpr (hipReadModeNormalizedFloat == readMode) {
      typedef decltype(getNormalizedFloatType(*data.data)) TargetType;
      std::vector<TargetType> fData;
      fData.resize(data.e.width * data.e.height * data.e.depth);
      for (unsigned int k = 0; k < data.e.depth; k++) {
        for (unsigned int j = 0; j < data.e.height; j++) {
          for (unsigned int i = 0; i < data.e.width; i++) {
            unsigned int index = k * data.e.width * data.e.height + j * data.e.width + i;
            fData[index] = getNormalizedFloatType<T>(data.data[index]);
#ifdef TEST_TEXTURE
            fprintf(stderr, "level=%u, (%u/%zu, %u/%zu, %u/%zu-->%u): data.data=%s, fData=%s\n",
                    level, i, data.e.width, j, data.e.height, k, data.e.depth, index,
                    getString(data.data[index]).c_str(), getString(fData[index]).c_str());
#endif
          }
        }
      }
      verifyMipmapLevel<TargetType, filterMode, addressMode>(texMipmap, fData.data(), data.e.width,
                                                             data.e.height, data.e.depth, level,
                                                             offsetX, offsetY, offsetZ);

    } else {  // hipReadModeElementType == readMode
      verifyMipmapLevel<T, filterMode, addressMode>(texMipmap, data.data, data.e.width,
                                                    data.e.height, data.e.depth, level, offsetX,
                                                    offsetY, offsetZ);
    }
    HIP_CHECK(hipHostFree(data.data));
    memset(&data, 0, sizeof(data));
  }

  HIP_CHECK(hipDestroyTextureObject(texMipmap));
  HIP_CHECK(hipFreeMipmappedArray(mipmapArray));
}

/**
 * Test Description
 * ------------------------
 * - The suite will test following functions with hipReadModeElementType and hipFilterModePoint
     in 3D
       creating MipMap array,
       getting level array,
       creating/initilizing texture and surface objects on level array,
       creating texture object on the mipmap array,
       verifing the texture object
 * Test source
 * ------------------------
 * - catch\unit\texture\hipTextureMipmapObj3D.cc
 * Test requirements
 * ------------------------
 *  - Host specific (WINDOWS)
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType", "", char, uchar,
                   short, ushort, int, uint, float, char1, uchar1, short1, ushort1, int1, uint1,
                   float1, char2, uchar2, short2, ushort2, int2, uint2, float2, char4, uchar4,
                   short4, ushort4, int4, uint4, float4) {
  CHECK_IMAGE_SUPPORT

#if __HIP_NO_IMAGE_SUPPORT
  HipTest::HIP_SKIP_TEST("__HIP_NO_IMAGE_SUPPORT is set");
  return;
#endif

  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType, hipFilterModePoint, "
      "hipAddressModeClamp 23, 21, 47") {
    testMipmapTextureObj<TestType, hipReadModeElementType, hipFilterModePoint, hipAddressModeClamp>(
        23, 21, 47, 0.4, -0.9, 0.77);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType, hipFilterModePoint, "
      "hipAddressModeClamp 67, 131, 99") {
    testMipmapTextureObj<TestType, hipReadModeElementType, hipFilterModePoint, hipAddressModeClamp>(
        67, 131, 99, -0.3, -0.67, 0.49);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType, hipFilterModePoint, "
      "hipAddressModeBorder 131, 263, 31") {
    testMipmapTextureObj<TestType, hipReadModeElementType, hipFilterModePoint,
                         hipAddressModeBorder>(131, 263, 31, 0.15, -0.34, 0.85);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType, hipFilterModePoint, "
      "hipAddressModeBorder 263, 67, 17") {
    testMipmapTextureObj<TestType, hipReadModeElementType, hipFilterModePoint,
                         hipAddressModeBorder>(263, 67, 17, 0.13, 0.96, -0.57);
  }
}

/**
 * Test Description
 * ------------------------
 * - The suite will test following functions with hipReadModeNormalizedFloat on integer types in 3D
       creating MipMap array,
       getting level array,
       creating/initilizing texture and surface objects on level array,
       creating texture object on the mipmap array,
       verifing the texture object
 * Test source
 * ------------------------
 * - catch\unit\texture\hipTextureMipmapObj3D.cc
 * Test requirements
 * ------------------------
 *  - Host specific (WINDOWS)
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_hipTextureMipmapObj3D_Check - hipReadModeNormalizedFloat", "", char, uchar,
                   short, ushort, char1, uchar1, short1, ushort1, char2, uchar2, short2, ushort2,
                   char4, uchar4, short4, ushort4) {
  CHECK_IMAGE_SUPPORT

#if __HIP_NO_IMAGE_SUPPORT
  HipTest::HIP_SKIP_TEST("__HIP_NO_IMAGE_SUPPORT is set");
  return;
#endif

  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeNormalizedFloat, hipFilterModePoint, "
      "hipAddressModeClamp 23, 21, 67") {
    testMipmapTextureObj<TestType, hipReadModeNormalizedFloat, hipFilterModePoint,
                         hipAddressModeClamp>(23, 21, 67, 0.4, -0.9, 0.37);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeNormalizedFloat, hipFilterModePoint, "
      "hipAddressModeClamp 131, 263, 11") {
    testMipmapTextureObj<TestType, hipReadModeNormalizedFloat, hipFilterModePoint,
                         hipAddressModeClamp>(131, 263, 11, 0.15, -0.34, 0.83);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeNormalizedFloat, hipFilterModeLinear, "
      "hipAddressModeClamp 67, 131, 53") {
    testMipmapTextureObj<TestType, hipReadModeNormalizedFloat, hipFilterModeLinear,
                         hipAddressModeClamp>(23, 17, 53, -0.3, -0.67, 0.78);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeNormalizedFloat, hipFilterModeLinear, "
      "hipAddressModeClamp 263, 67, 37") {
    testMipmapTextureObj<TestType, hipReadModeNormalizedFloat, hipFilterModeLinear,
                         hipAddressModeClamp>(263, 67, 37, 0.13, 0.96, -0.96);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeNormalizedFloat, hipFilterModePoint, "
      "hipAddressModeBorder 131, 263, 11") {
    testMipmapTextureObj<TestType, hipReadModeNormalizedFloat, hipFilterModePoint,
                         hipAddressModeBorder>(131, 263, 11, 0.15, -0.34, -0.11);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeNormalizedFloat, hipFilterModePoint, "
      "hipAddressModeBorder 23, 21, 201") {
    testMipmapTextureObj<TestType, hipReadModeNormalizedFloat, hipFilterModePoint,
                         hipAddressModeBorder>(23, 21, 201, 0.4, -0.9, 0.54);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeNormalizedFloat, hipFilterModeLinear, "
      "hipAddressModeBorder 263, 67, 51") {
    testMipmapTextureObj<TestType, hipReadModeNormalizedFloat, hipFilterModeLinear,
                         hipAddressModeBorder>(263, 67, 51, 0.13, 0.96, 0.77);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeNormalizedFloat, hipFilterModeLinear, "
      "hipAddressModeBorder 67, 131, 87") {
    testMipmapTextureObj<TestType, hipReadModeNormalizedFloat, hipFilterModeLinear,
                         hipAddressModeBorder>(67, 131, 87, -0.3, -0.67, -0.29);
  }
}

/**
 * Test Description
 * ------------------------
 * - The suite will test following functions with hipReadModeElementType and hipFilterModeLinear
     on float types in 3D,
       creating MipMap array,
       getting level array,
       creating/initilizing texture and surface objects on level array,
       creating texture object on the mipmap array,
       verifing the texture object
 * Test source
 * ------------------------
 * - catch\unit\texture\hipTextureMipmapObj3D.cc
 * Test requirements
 * ------------------------
 *  - Host specific (WINDOWS)
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType float only", "",
                   float, float1, float2, float4) {
  CHECK_IMAGE_SUPPORT

#if __HIP_NO_IMAGE_SUPPORT
  HipTest::HIP_SKIP_TEST("__HIP_NO_IMAGE_SUPPORT is set");
  return;
#endif

  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType, hipFilterModeLinear, "
      "hipAddressModeClamp 23, 17, 301, 0., 0., 0.") {
    testMipmapTextureObj<TestType, hipReadModeElementType, hipFilterModeLinear,
                         hipAddressModeClamp>(23, 17, 301, 0.19, 0.46, -0.9);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType, hipFilterModeLinear, "
      "hipAddressModeClamp 23, 17, 243") {
    testMipmapTextureObj<TestType, hipReadModeElementType, hipFilterModeLinear,
                         hipAddressModeClamp>(23, 17, 243, -0.3, -0.67, 0.65);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType, hipFilterModeLinear, "
      "hipAddressModeClamp 263, 67, 39") {
    testMipmapTextureObj<TestType, hipReadModeElementType, hipFilterModeLinear,
                         hipAddressModeClamp>(263, 67, 39, 0.13, 0.96, 0.66);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType, hipFilterModeLinear, "
      "hipAddressModeBorder 263, 67, 117") {
    testMipmapTextureObj<TestType, hipReadModeElementType, hipFilterModeLinear,
                         hipAddressModeBorder>(263, 67, 117, 0.13, 0.96, -0.93);
  }
  SECTION(
      "Unit_hipTextureMipmapObj3D_Check - hipReadModeElementType, hipFilterModeLinear, "
      "hipAddressModeBorder 67, 131, 67") {
    testMipmapTextureObj<TestType, hipReadModeElementType, hipFilterModeLinear,
                         hipAddressModeBorder>(67, 131, 67, -0.3, -0.67, 0.88);
  }
}
