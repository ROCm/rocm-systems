/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @addtogroup hipGreenCtx hipGreenCtx
 * @{
 * @ingroup GreenContextTest
 * `hipGreenCtx*` APIs - basic sanity tests
 */

#include <hip_test_common.hh>

static hipError_t GetSmResourceDesc(hipDevResourceDesc_t* desc) {
  hipDevice_t device;
  hipDevResource resource{};
  hipError_t ret = hipDeviceGet(&device, 0);
  if (ret != hipSuccess) {
    return ret;
  }
  ret = hipDeviceGetDevResource(device, &resource, hipDevResourceTypeSm);
  if (ret != hipSuccess) {
    return ret;
  }
  return hipDevResourceGenerateDesc(desc, &resource, 1);
}

/**
 * Test Description
 * ------------------------
 *  - Creates and destroys a green context using SM resources
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipGreenCtxCreateDestroy_Sanity") {
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipGreenCtx_t green_ctx = nullptr;
  ret = hipGreenCtxCreate(&green_ctx, desc, 0, 0);
  REQUIRE(ret == hipSuccess);
  REQUIRE(green_ctx != nullptr);
  HIP_CHECK(hipGreenCtxDestroy(green_ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Creates a green context and its stream using SM resources
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipGreenCtxStreamCreate_Sanity") {
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipGreenCtx_t green_ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipGreenCtxStreamCreate(&stream, green_ctx, 0, 0));
  REQUIRE(stream != nullptr);

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGreenCtxDestroy(green_ctx));
}

/**
 * Test Description
 * ------------------------
 *  - Retrieves the green context from a stream
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
TEST_CASE("Unit_hipGreenCtxStreamGetGreenCtx_Basic") {
  hipDevResourceDesc_t desc{};
  hipError_t ret = GetSmResourceDesc(&desc);
  REQUIRE(ret == hipSuccess);

  hipGreenCtx_t green_ctx = nullptr;
  HIP_CHECK(hipGreenCtxCreate(&green_ctx, desc, 0, 0));
  REQUIRE(green_ctx != nullptr);

  hipStream_t stream = nullptr;
  HIP_CHECK(hipGreenCtxStreamCreate(&stream, green_ctx, 0, 0));
  REQUIRE(stream != nullptr);

  hipGreenCtx_t out_green_ctx = nullptr;
  HIP_CHECK(hipStreamGetGreenCtx(stream, &out_green_ctx));
  REQUIRE(out_green_ctx == green_ctx);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGreenCtxDestroy(green_ctx));
}

/**
 * End doxygen group hipGreenCtx.
 * @}
 */
