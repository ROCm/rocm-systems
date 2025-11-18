/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @addtogroup hipCreateTextureObject hipCreateTextureObject
 * @{
 * @ingroup TextureTest
 */

#define UNALIGN_OFFSET 1
#define N 512

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid linear resource:
 *    -# When device pointer is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When size in bytes is 0
 *      - Expected output: do not return `hipSuccess`
 *    -# When size in bytes is `size_t` maximum
 *      - Expected output: do not return `hipSuccess`
 *    -# When a valid resource view descriptor is provided
 *      - Platform specific (AMD)
 *      - Expected output: do not return `hipSuccess`
 *    -# When device pointer is not aligned appropriately
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/texture/hipCreateTextureObject_Linear.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipCreateTextureObject_LinearResource") {
  CHECK_IMAGE_SUPPORT

  float* texBuf;
  hipError_t ret;
  constexpr int xsize = 32;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipResourceViewDesc resViewDesc;
  hipTextureObject_t texObj;
  hipDeviceProp_t devProp;

  // Initialization
  HIP_CHECK(hipMalloc(&texBuf, N * sizeof(float)));
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  memset(&resDesc, 0, sizeof(resDesc));
  memset(&texDesc, 0, sizeof(texDesc));
  resDesc.resType = hipResourceTypeLinear;

  // Sections
  SECTION("hipResourceTypeLinear and devPtr(nullptr)") {
    // Populate resource descriptor
    resDesc.res.linear.devPtr = nullptr;
    resDesc.res.linear.desc = hipCreateChannelDesc(xsize, 0, 0, 0, hipChannelFormatKindFloat);
    resDesc.res.linear.sizeInBytes = N * sizeof(float);

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;
    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypeLinear and sizeInBytes(0)") {
    // Populate resource descriptor
    resDesc.res.linear.devPtr = texBuf;
    resDesc.res.linear.desc = hipCreateChannelDesc(xsize, 0, 0, 0, hipChannelFormatKindFloat);
    resDesc.res.linear.sizeInBytes = 0;

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;
    HIP_CHECK(hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    HIP_CHECK(hipDestroyTextureObject(texObj));
  }

  SECTION("hipResourceTypeLinear and sizeInBytes(max(size_t))") {
    // Populate resource descriptor
    resDesc.res.linear.devPtr = texBuf;
    resDesc.res.linear.desc = hipCreateChannelDesc(xsize, 0, 0, 0, hipChannelFormatKindFloat);
    resDesc.res.linear.sizeInBytes = std::numeric_limits<std::size_t>::max();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;
    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypeLinear and valid resource view descriptor") {
#if HT_AMD
    // Populate resource descriptor
    resDesc.res.linear.devPtr = texBuf;
    resDesc.res.linear.desc = hipCreateChannelDesc(xsize, 0, 0, 0, hipChannelFormatKindFloat);
    resDesc.res.linear.sizeInBytes = N * sizeof(float);

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    // Populate resourceview descriptor
    memset(&resViewDesc, 0, sizeof(resViewDesc));
    resViewDesc.format = hipResViewFormatFloat1;
    resViewDesc.width = N * sizeof(float);
    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, &resViewDesc);
    REQUIRE(ret != hipSuccess);
#else
    // API expected to return error according to cuda documentation.
    WARN("Resource view descriptor test skipped on nvidia");
#endif
  }

  SECTION("hipResourceTypeLinear and devicePtr un-aligned") {
    if (devProp.textureAlignment > UNALIGN_OFFSET) {
      // Populate resource descriptor
      resDesc.res.linear.devPtr = reinterpret_cast<char*>(texBuf) + UNALIGN_OFFSET;
      resDesc.res.linear.desc = hipCreateChannelDesc(xsize, 0, 0, 0, hipChannelFormatKindFloat);
      resDesc.res.linear.sizeInBytes = N * sizeof(float);

      // Populate texture descriptor
      texDesc.readMode = hipReadModeElementType;
      ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
      REQUIRE(ret != hipSuccess);
    }
  }

  // De-Initialization
  HIP_CHECK(hipFree(texBuf));
}

/**
 * End doxygen group TextureTest.
 * @}
 */
